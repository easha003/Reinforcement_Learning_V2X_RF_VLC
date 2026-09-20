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
    PairActionLifecycle,
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
    lifecycles: dict[str, PairLifecycle] | None = None,
    episode_steps: dict[str, int] | None = None,
) -> PopulationFrame:
    lifecycle_by_pair = {} if lifecycles is None else lifecycles
    step_by_pair = {} if episode_steps is None else episode_steps
    vehicle_ids = sorted(
        {endpoint for _, tx_id, rx_id in pair_endpoints for endpoint in (tx_id, rx_id)}
    )
    vehicles = tuple(_vehicle(vehicle_id, time_s=time_s) for vehicle_id in vehicle_ids)
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=step_by_pair.get(pair_id, index),
            transmitter=by_id[tx_id],
            receiver=by_id[rx_id],
            lifecycle=lifecycle_by_pair.get(
                pair_id,
                PairLifecycle(born=index == 0),
            ),
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
    assert ledger.rf_reservation_releases_by_pair == (
        ("pair-a", True),
        ("pair-b", False),
        ("pair-c", False),
        ("pair-d", False),
    )
    assert ledger.released_rf_pair_ids == ("pair-a",)
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
    assert tuple(
        record.rf_reservation_released for record in ledger.pair_accounting
    ) == (
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
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
    assert ledger.rf_reservation_releases_by_pair == (
        ("pair-a", True),
        ("pair-b", True),
    )
    assert ledger.released_rf_pair_ids == ("pair-a", "pair-b")


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
    assert ledger.rf_reservation_releases_by_pair == ()
    assert ledger.released_rf_pair_ids == ()
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


def test_current_vlc_action_releases_a_prior_frame_rf_reservation() -> None:
    endpoints = (("pair-a", "veh-1", "veh-2"),)
    previous_frame = _frame(endpoints, index=2, time_s=0.2)
    current_frame = _frame(endpoints, index=3, time_s=0.3)

    previous = _ledger(previous_frame, {"pair-a": PolicyAction.DUP_4})
    current = _ledger(current_frame, {"pair-a": PolicyAction.VLC})

    assert previous.total_reserved_rf_attempts == 4
    assert previous.released_rf_pair_ids == ()
    assert not previous.pair_accounting[0].rf_reservation_released
    assert current.total_reserved_rf_attempts == 0
    assert current.released_rf_pair_ids == ("pair-a",)
    assert current.pair_accounting[0].rf_reservation_released


def test_rf_and_dup_actions_commit_reservations_without_early_release() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
        )
    )

    ledger = _ledger(
        frame,
        {"pair-a": PolicyAction.RF_1, "pair-b": PolicyAction.DUP_4},
    )

    assert ledger.reserved_rf_attempts_by_pair == (("pair-a", 1), ("pair-b", 4))
    assert ledger.rf_reservation_releases_by_pair == (
        ("pair-a", False),
        ("pair-b", False),
    )
    assert ledger.released_rf_pair_ids == ()


def test_lifecycle_rows_include_births_and_final_packets_before_state_release() -> None:
    endpoints = (
        ("pair-born", "veh-1", "veh-2"),
        ("pair-continuing", "veh-3", "veh-4"),
        ("pair-internal", "veh-5", "veh-6"),
        ("pair-natural", "veh-7", "veh-8"),
        ("pair-trace-end", "veh-9", "veh-10"),
    )
    frame = _frame(
        endpoints,
        index=10,
        time_s=1.0,
        lifecycles={
            "pair-born": PairLifecycle(born=True),
            "pair-continuing": PairLifecycle(born=False),
            "pair-internal": PairLifecycle(
                born=False,
                truncated=True,
                bootstrap_valid=True,
                end_reason="max_duration",
            ),
            "pair-natural": PairLifecycle(
                born=False,
                terminated=True,
                end_reason="route_diverged",
            ),
            "pair-trace-end": PairLifecycle(
                born=False,
                truncated=True,
                end_reason="trace_end",
            ),
        },
        episode_steps={
            "pair-born": 0,
            "pair-continuing": 4,
            "pair-internal": 10,
            "pair-natural": 7,
            "pair-trace-end": 3,
        },
    )

    ledger = _ledger(
        frame,
        {pair_id: PolicyAction.RF_1 for pair_id in frame.active_pair_ids},
    )

    assert ledger.active_pairs == 5
    assert ledger.total_reserved_rf_attempts == 5
    assert ledger.born_pair_ids == ("pair-born",)
    assert ledger.continuing_pair_ids == (
        "pair-continuing",
        "pair-internal",
        "pair-natural",
        "pair-trace-end",
    )
    assert ledger.terminated_pair_ids == ("pair-natural",)
    assert ledger.truncated_pair_ids == ("pair-internal", "pair-trace-end")
    assert ledger.bootstrap_valid_pair_ids == ("pair-internal",)
    assert ledger.release_after_frame_pair_ids == (
        "pair-internal",
        "pair-natural",
        "pair-trace-end",
    )
    by_id = {record.pair_id: record for record in ledger.pair_accounting}
    assert by_id["pair-born"].lifecycle.episode_step == 0
    assert by_id["pair-born"].lifecycle.born
    assert by_id["pair-natural"].lifecycle.end_reason == "route_diverged"
    assert by_id["pair-natural"].activation_cost == 1.0
    assert by_id["pair-internal"].lifecycle.bootstrap_valid
    assert by_id["pair-trace-end"].lifecycle.release_after_frame


def test_terminated_pair_becomes_inactive_after_its_accounted_final_frame() -> None:
    endpoints = (("pair-a", "veh-1", "veh-2"),)
    final_frame = _frame(
        endpoints,
        index=4,
        time_s=0.4,
        lifecycles={
            "pair-a": PairLifecycle(
                born=False,
                terminated=True,
                end_reason="route_diverged",
            )
        },
        episode_steps={"pair-a": 4},
    )
    inactive_frame = _frame((), index=5, time_s=0.5)

    final_ledger = _ledger(final_frame, {"pair-a": PolicyAction.DUP_2})
    inactive_ledger = _ledger(inactive_frame, {})

    assert final_ledger.pair_ids == ("pair-a",)
    assert final_ledger.total_reserved_rf_attempts == 2
    assert final_ledger.release_after_frame_pair_ids == ("pair-a",)
    assert inactive_ledger.pair_ids == ()
    assert inactive_ledger.total_reserved_rf_attempts == 0
    with pytest.raises(ActionAggregationError, match="cover"):
        _ledger(inactive_frame, {"pair-a": PolicyAction.DUP_2})


def test_invalid_action_lifecycle_combinations_fail_closed() -> None:
    with pytest.raises(ActionAggregationError, match="step zero"):
        PairActionLifecycle(1, True, False, False, False, None)
    with pytest.raises(ActionAggregationError, match="both"):
        PairActionLifecycle(1, False, True, True, False, "trace_end")
    with pytest.raises(ActionAggregationError, match="appear together"):
        PairActionLifecycle(1, False, True, False, False, None)
    with pytest.raises(ActionAggregationError, match="truncated"):
        PairActionLifecycle(1, False, False, False, True, None)


def test_direct_records_reject_invalid_identity_and_unresolved_action() -> None:
    lifecycle = PairActionLifecycle(1, False, False, False, False, None)
    with pytest.raises(ActionAggregationError, match="pair_id"):
        PairActionReservation(
            pair_id="",
            action=PolicyAction.RF_1,
            lifecycle=lifecycle,
        )
    with pytest.raises(ActionAggregationError, match="mask-validated"):
        PairActionReservation(  # type: ignore[arg-type]
            pair_id="pair-a",
            action=1,
            lifecycle=lifecycle,
        )


def test_direct_ledger_rejects_noncanonical_or_duplicate_pair_rows() -> None:
    lifecycle = PairActionLifecycle(1, False, False, False, False, None)
    pair_a = PairActionReservation("pair-a", PolicyAction.RF_1, lifecycle)
    pair_b = PairActionReservation("pair-b", PolicyAction.RF_2, lifecycle)
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
    lifecycle = PairActionLifecycle(1, False, False, False, False, None)
    valid = PairActionReservation(
        "pair-a",
        PolicyAction.DUP_2,
        lifecycle,
    ).account(_resource_map())

    assert valid.action_index == 6
    assert valid.action_name == "DUP-2"
    assert valid.reserved_rf_attempts == 2
    assert valid.vlc_activations == 1
    assert valid.uses_rf and valid.uses_vlc and valid.duplicates
    assert not valid.rf_reservation_released
    assert valid.activation_cost == 3.0
    assert valid.reward == -3.0

    with pytest.raises(ActionAggregationError, match="authoritative"):
        PairResourceAccounting(
            pair_id="pair-a",
            action=PolicyAction.DUP_2,
            lifecycle=lifecycle,
            reserved_rf_attempts=1,
            vlc_activations=1,
            uses_rf=True,
            uses_vlc=True,
            duplicates=True,
            rf_reservation_released=False,
            activation_cost=3.0,
            reward=-3.0,
        )
    with pytest.raises(ActionAggregationError, match="negative activation"):
        PairResourceAccounting(
            pair_id="pair-a",
            action=PolicyAction.DUP_2,
            lifecycle=lifecycle,
            reserved_rf_attempts=2,
            vlc_activations=1,
            uses_rf=True,
            uses_vlc=True,
            duplicates=True,
            rf_reservation_released=False,
            activation_cost=3.0,
            reward=-2.0,
        )

    with pytest.raises(ActionAggregationError, match="authoritative"):
        PairResourceAccounting(
            pair_id="pair-a",
            action=PolicyAction.VLC,
            lifecycle=lifecycle,
            reserved_rf_attempts=0,
            vlc_activations=1,
            uses_rf=False,
            uses_vlc=True,
            duplicates=False,
            rf_reservation_released=False,
            activation_cost=1.0,
            reward=-1.0,
        )
