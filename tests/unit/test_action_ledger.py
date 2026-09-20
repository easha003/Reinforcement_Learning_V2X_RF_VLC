"""Phase 3 identity-preserving packet and frame resource accounting."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.config.models import CostConfig
from hybrid_v2x_rl.core.policy_actions import ActionResourceMap, PolicyAction
from hybrid_v2x_rl.mean_field.action_ledger import (
    ActionAggregationError,
    FrameActionLedger,
    PairActionReservation,
    PairResourceAccounting,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

TRACE_ID = "synthetic-d10-train-000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resource_map(
    *,
    rf_activation: float = 1.0,
    vlc_activation: float = 1.0,
) -> ActionResourceMap:
    config = load_headline_config(PROJECT_ROOT)
    costs = CostConfig(
        rf_activation=rf_activation,
        vlc_activation=vlc_activation,
    )
    return ActionResourceMap.from_config(config.environment, costs)


def _ledger(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
    *,
    resource_map: ActionResourceMap | None = None,
) -> FrameActionLedger:
    return FrameActionLedger.from_frame(
        frame,
        actions,
        resource_map=_resource_map() if resource_map is None else resource_map,
    )


def _vehicle(vehicle_id: str, *, time_s: float = 0.2) -> VehicleTraceRecord:
    index = int(vehicle_id.split("-")[-1])
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=time_s,
        vehicle_id=vehicle_id,
        x_m=10.0 * index,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=5.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


def _frame(
    pair_endpoints: tuple[tuple[str, str, str], ...],
    *,
    index: int = 2,
    time_s: float = 0.2,
) -> PopulationFrame:
    vehicle_ids = sorted(
        {endpoint for _, tx_id, rx_id in pair_endpoints for endpoint in (tx_id, rx_id)}
    )
    vehicles = tuple(_vehicle(vehicle_id, time_s=time_s) for vehicle_id in vehicle_ids)
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=index,
            transmitter=by_id[tx_id],
            receiver=by_id[rx_id],
            lifecycle=PairLifecycle(born=index == 0),
        )
        for pair_id, tx_id, rx_id in sorted(pair_endpoints)
    )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="train",
            density=10.0,
            replicate=0,
        ),
        index=index,
        time_s=time_s,
        vehicles=vehicles,
        pairs=pairs,
    )


def test_mixed_population_aggregates_committed_attempts_in_frame_order() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
            ("pair-c", "veh-5", "veh-6"),
            ("pair-d", "veh-7", "veh-8"),
        )
    )

    ledger = _ledger(
        frame,
        {
            "pair-d": PolicyAction.DUP_2,
            "pair-b": PolicyAction.RF_1,
            "pair-c": PolicyAction.RF_4,
            "pair-a": PolicyAction.VLC,
        },
    )

    assert ledger.pair_ids == frame.active_pair_ids
    assert ledger.reserved_rf_attempts_by_pair == (
        ("pair-a", 0),
        ("pair-b", 1),
        ("pair-c", 4),
        ("pair-d", 2),
    )
    assert ledger.total_reserved_rf_attempts == 7
    assert ledger.active_pairs == 4
    assert ledger.total_vlc_activations == 2
    assert ledger.rf_using_pairs == 3
    assert ledger.vlc_using_pairs == 2
    assert ledger.duplicated_pairs == 1
    assert ledger.activation_costs_by_pair == (
        ("pair-a", 1.0),
        ("pair-b", 1.0),
        ("pair-c", 4.0),
        ("pair-d", 3.0),
    )
    assert ledger.total_activation_cost == 9.0
    assert ledger.total_reward == -9.0


def test_all_nine_actions_use_the_authoritative_attempt_counts() -> None:
    endpoints = tuple(
        (f"pair-{index}", f"veh-{2 * index + 1}", f"veh-{2 * index + 2}")
        for index in range(9)
    )
    frame = _frame(endpoints)

    ledger = _ledger(
        frame,
        {f"pair-{index}": action for index, action in enumerate(PolicyAction)},
    )

    assert tuple(
        attempts for _, attempts in ledger.reserved_rf_attempts_by_pair
    ) == (0, 1, 2, 3, 4, 1, 2, 3, 4)
    assert tuple(record.uses_rf for record in ledger.pair_accounting) == (
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    assert tuple(record.uses_vlc for record in ledger.pair_accounting) == (
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    )
    assert tuple(record.duplicates for record in ledger.pair_accounting) == (
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    )
    assert tuple(record.activation_cost for record in ledger.pair_accounting) == (
        1.0,
        1.0,
        2.0,
        3.0,
        4.0,
        2.0,
        3.0,
        4.0,
        5.0,
    )
    assert tuple(record.reward for record in ledger.pair_accounting) == (
        -1.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
    )
    assert ledger.total_reserved_rf_attempts == 20
    assert ledger.total_vlc_activations == 5
    assert ledger.rf_using_pairs == 8
    assert ledger.vlc_using_pairs == 5
    assert ledger.duplicated_pairs == 4
    assert ledger.total_activation_cost == 25.0
    assert ledger.total_reward == -25.0


def test_vlc_only_population_contributes_zero_rf_demand() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
        )
    )

    ledger = _ledger(
        frame,
        {"pair-a": PolicyAction.VLC, "pair-b": PolicyAction.VLC},
    )

    assert ledger.reserved_rf_attempts_by_pair == (("pair-a", 0), ("pair-b", 0))
    assert ledger.total_reserved_rf_attempts == 0
    assert ledger.rf_using_pairs == 0
    assert ledger.vlc_using_pairs == 2
    assert ledger.duplicated_pairs == 0
    assert ledger.total_vlc_activations == 2
    assert ledger.total_activation_cost == 2.0


def test_shared_physical_endpoint_does_not_merge_pair_reservations() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-2", "veh-3"),
        )
    )

    ledger = _ledger(
        frame,
        {"pair-a": PolicyAction.RF_4, "pair-b": PolicyAction.DUP_3},
    )

    assert frame.shared_endpoint_ids == ("veh-2",)
    assert ledger.reserved_rf_attempts_by_pair == (("pair-a", 4), ("pair-b", 3))
    assert ledger.total_reserved_rf_attempts == 7


def test_empty_population_has_valid_zero_demand() -> None:
    frame = _frame(())

    ledger = _ledger(frame, {})

    assert ledger.pair_ids == ()
    assert ledger.active_pairs == 0
    assert ledger.total_reserved_rf_attempts == 0
    assert ledger.total_vlc_activations == 0
    assert ledger.rf_using_pairs == 0
    assert ledger.vlc_using_pairs == 0
    assert ledger.duplicated_pairs == 0
    assert ledger.activation_costs_by_pair == ()
    assert ledger.rewards_by_pair == ()
    assert ledger.total_activation_cost == 0.0
    assert ledger.total_reward == 0.0


@pytest.mark.parametrize(
    "actions",
    [
        {"pair-a": PolicyAction.RF_1},
        {
            "pair-a": PolicyAction.RF_1,
            "pair-b": PolicyAction.RF_2,
            "pair-extra": PolicyAction.RF_3,
        },
    ],
)
def test_joint_action_must_cover_active_population_exactly(
    actions: dict[str, PolicyAction],
) -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
        )
    )

    with pytest.raises(ActionAggregationError, match="cover"):
        _ledger(frame, actions)


@pytest.mark.parametrize("invalid", [1, "RF-1", True])
def test_aggregation_rejects_actions_that_bypassed_mask_resolution(
    invalid: object,
) -> None:
    frame = _frame((("pair-a", "veh-1", "veh-2"),))

    with pytest.raises(ActionAggregationError, match="mask-validated"):
        _ledger(
            frame,
            {"pair-a": invalid},  # type: ignore[dict-item]
        )


def test_ledger_retains_reproducible_frame_identity() -> None:
    frame = _frame((("pair-a", "veh-1", "veh-2"),), index=7, time_s=0.7)

    ledger = _ledger(frame, {"pair-a": PolicyAction.DUP_4})

    assert ledger.trace_id == TRACE_ID
    assert ledger.frame_index == 7
    assert ledger.time_s == pytest.approx(0.7)


def test_direct_records_reject_invalid_identity_and_unresolved_action() -> None:
    with pytest.raises(ActionAggregationError, match="pair_id"):
        PairActionReservation(pair_id="", action=PolicyAction.RF_1)
    with pytest.raises(ActionAggregationError, match="mask-validated"):
        PairActionReservation(pair_id="pair-a", action=1)  # type: ignore[arg-type]


def test_direct_ledger_rejects_noncanonical_or_duplicate_pair_rows() -> None:
    pair_a = PairActionReservation("pair-a", PolicyAction.RF_1)
    pair_b = PairActionReservation("pair-b", PolicyAction.RF_2)
    resources = _resource_map()

    with pytest.raises(ActionAggregationError, match="canonical"):
        FrameActionLedger(TRACE_ID, 0, 0.0, resources, (pair_b, pair_a))
    with pytest.raises(ActionAggregationError, match="duplicate"):
        FrameActionLedger(TRACE_ID, 0, 0.0, resources, (pair_a, pair_a))


def test_nonunit_configured_prices_flow_into_every_packet_and_frame_total() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
            ("pair-c", "veh-5", "veh-6"),
        )
    )
    resources = _resource_map(rf_activation=0.3, vlc_activation=2.0)

    ledger = _ledger(
        frame,
        {
            "pair-a": PolicyAction.VLC,
            "pair-b": PolicyAction.RF_4,
            "pair-c": PolicyAction.DUP_4,
        },
        resource_map=resources,
    )

    assert tuple(pair_id for pair_id, _ in ledger.activation_costs_by_pair) == (
        "pair-a",
        "pair-b",
        "pair-c",
    )
    assert tuple(cost for _, cost in ledger.activation_costs_by_pair) == pytest.approx(
        (2.0, 1.2, 3.2)
    )
    assert tuple(reward for _, reward in ledger.rewards_by_pair) == pytest.approx(
        (-2.0, -1.2, -3.2)
    )
    assert ledger.total_activation_cost == pytest.approx(6.4)
    assert ledger.total_reward == pytest.approx(-6.4)


def test_population_totals_conserve_all_per_packet_accounting() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
            ("pair-c", "veh-5", "veh-6"),
        )
    )
    ledger = _ledger(
        frame,
        {
            "pair-a": PolicyAction.VLC,
            "pair-b": PolicyAction.RF_3,
            "pair-c": PolicyAction.DUP_2,
        },
    )
    records = ledger.pair_accounting

    assert ledger.total_reserved_rf_attempts == sum(
        record.reserved_rf_attempts for record in records
    )
    assert ledger.total_vlc_activations == sum(
        record.vlc_activations for record in records
    )
    assert ledger.rf_using_pairs == sum(record.uses_rf for record in records)
    assert ledger.vlc_using_pairs == sum(record.uses_vlc for record in records)
    assert ledger.duplicated_pairs == sum(record.duplicates for record in records)
    assert ledger.total_activation_cost == sum(
        record.activation_cost for record in records
    )
    assert ledger.total_reward == sum(record.reward for record in records)


def test_materialized_packet_accounting_rejects_resource_or_reward_drift() -> None:
    valid = PairActionReservation("pair-a", PolicyAction.DUP_2).account(
        _resource_map()
    )

    assert valid.action_index == 6
    assert valid.action_name == "DUP-2"
    assert valid.reserved_rf_attempts == 2
    assert valid.vlc_activations == 1
    assert valid.uses_rf and valid.uses_vlc and valid.duplicates
    assert valid.activation_cost == 3.0
    assert valid.reward == -3.0

    with pytest.raises(ActionAggregationError, match="authoritative"):
        PairResourceAccounting(
            pair_id="pair-a",
            action=PolicyAction.DUP_2,
            reserved_rf_attempts=1,
            vlc_activations=1,
            uses_rf=True,
            uses_vlc=True,
            duplicates=True,
            activation_cost=3.0,
            reward=-3.0,
        )
    with pytest.raises(ActionAggregationError, match="negative activation"):
        PairResourceAccounting(
            pair_id="pair-a",
            action=PolicyAction.DUP_2,
            reserved_rf_attempts=2,
            vlc_activations=1,
            uses_rf=True,
            uses_vlc=True,
            duplicates=True,
            activation_cost=3.0,
            reward=-2.0,
        )
