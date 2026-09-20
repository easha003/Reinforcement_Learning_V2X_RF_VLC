"""Phase 3 identity-preserving frame RF-demand aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.action_ledger import (
    ActionAggregationError,
    FrameActionLedger,
    PairActionReservation,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

TRACE_ID = "synthetic-d10-train-000"


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

    ledger = FrameActionLedger.from_frame(
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


def test_all_nine_actions_use_the_authoritative_attempt_counts() -> None:
    endpoints = tuple(
        (f"pair-{index}", f"veh-{2 * index + 1}", f"veh-{2 * index + 2}")
        for index in range(9)
    )
    frame = _frame(endpoints)

    ledger = FrameActionLedger.from_frame(
        frame,
        {f"pair-{index}": action for index, action in enumerate(PolicyAction)},
    )

    assert tuple(
        attempts for _, attempts in ledger.reserved_rf_attempts_by_pair
    ) == (0, 1, 2, 3, 4, 1, 2, 3, 4)
    assert ledger.total_reserved_rf_attempts == 20


def test_vlc_only_population_contributes_zero_rf_demand() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-3", "veh-4"),
        )
    )

    ledger = FrameActionLedger.from_frame(
        frame,
        {"pair-a": PolicyAction.VLC, "pair-b": PolicyAction.VLC},
    )

    assert ledger.reserved_rf_attempts_by_pair == (("pair-a", 0), ("pair-b", 0))
    assert ledger.total_reserved_rf_attempts == 0


def test_shared_physical_endpoint_does_not_merge_pair_reservations() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-2", "veh-3"),
        )
    )

    ledger = FrameActionLedger.from_frame(
        frame,
        {"pair-a": PolicyAction.RF_4, "pair-b": PolicyAction.DUP_3},
    )

    assert frame.shared_endpoint_ids == ("veh-2",)
    assert ledger.reserved_rf_attempts_by_pair == (("pair-a", 4), ("pair-b", 3))
    assert ledger.total_reserved_rf_attempts == 7


def test_empty_population_has_valid_zero_demand() -> None:
    frame = _frame(())

    ledger = FrameActionLedger.from_frame(frame, {})

    assert ledger.pair_ids == ()
    assert ledger.active_pairs == 0
    assert ledger.total_reserved_rf_attempts == 0


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
        FrameActionLedger.from_frame(frame, actions)


@pytest.mark.parametrize("invalid", [1, "RF-1", True])
def test_aggregation_rejects_actions_that_bypassed_mask_resolution(
    invalid: object,
) -> None:
    frame = _frame((("pair-a", "veh-1", "veh-2"),))

    with pytest.raises(ActionAggregationError, match="mask-validated"):
        FrameActionLedger.from_frame(
            frame,
            {"pair-a": invalid},  # type: ignore[dict-item]
        )


def test_ledger_retains_reproducible_frame_identity() -> None:
    frame = _frame((("pair-a", "veh-1", "veh-2"),), index=7, time_s=0.7)

    ledger = FrameActionLedger.from_frame(frame, {"pair-a": PolicyAction.DUP_4})

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

    with pytest.raises(ActionAggregationError, match="canonical"):
        FrameActionLedger(TRACE_ID, 0, 0.0, (pair_b, pair_a))
    with pytest.raises(ActionAggregationError, match="duplicate"):
        FrameActionLedger(TRACE_ID, 0, 0.0, (pair_a, pair_a))
