"""Tests for mobility metrics and explicit Gate-1 decisions."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from hybrid_v2x_rl.mobility.tagged_pairs import TaggedPairSegment
from hybrid_v2x_rl.mobility.validation import (
    MobilityFrame,
    MobilityValidationCriteria,
    MobilityValidationReport,
    MobilityValidator,
    _separation_bin_edges,
)


def _pair(
    *,
    initial_distance_m: float = 30.0,
    end_s: float = 2.0,
    blocker: bool = False,
) -> TaggedPairSegment:
    return TaggedPairSegment(
        trace_id="trace-001",
        pair_id=f"pair-{initial_distance_m}-{end_s}-{blocker}",
        tx_id="tx",
        rx_id="rx",
        start_s=0.0,
        end_s=end_s,
        initial_distance_m=initial_distance_m,
        route_id="route-main",
        eligibility_reason="trace_end",
        has_intervening_vehicle=blocker,
    )


def _frame(
    time_s: float,
    *,
    active: int = 2,
    lane_length_km: float = 0.1,
    state: str = "rG",
    departed: int = 0,
    insertion_failures: int = 0,
    teleports: tuple[str, ...] = (),
) -> MobilityFrame:
    speeds = tuple(0.0 if index == 0 else 8.0 for index in range(active))
    return MobilityFrame(
        trace_id="trace-001",
        time_s=time_s,
        total_lane_length_km=lane_length_km,
        vehicle_speeds_mps=speeds,
        queued_vehicle_count=1 if active else 0,
        queue_length_m=7.0 if active else 0.0,
        departed_vehicle_count=departed,
        insertion_failures=insertion_failures,
        teleport_reasons=teleports,
        signal_states=(("tls-1", state),),
    )


def _criteria(**changes: object) -> MobilityValidationCriteria:
    baseline = MobilityValidationCriteria(
        target_density_veh_per_lane_km=20.0,
        speed_limit_mps=11.18,
    )
    return replace(baseline, **changes)


def _check(report_name: str, report: MobilityValidationReport) -> bool:
    return next(check.passed for check in report.gate1_checks if check.name == report_name)


def test_complete_report_computes_metrics_and_passes_gate_one() -> None:
    frames = (
        _frame(0.0, state="rG"),
        _frame(1.0, state="Gr", departed=1),
        _frame(2.0, state="rG", departed=1),
    )

    report = MobilityValidator(_criteria()).validate(
        frames,
        pair_segments=(_pair(blocker=True),),
        manifest_archived=True,
    )

    assert report.gate1_passed is True
    assert report.realized_density_mean == 20.0
    assert report.realized_density_std == 0.0
    assert report.mean_speed_mps == 4.0
    assert report.speed_std_mps == 4.0
    assert report.stopped_fraction == 0.5
    assert report.mean_queue_proxy_vehicles == 1.0
    assert report.mean_queue_length_m == 7.0
    assert report.throughput_vehicle_count == 2
    assert report.throughput_vehicles_per_hour == 3600.0
    assert report.insertion_failures == 0
    assert report.teleport_count == 0
    assert report.signal_count == 1
    assert report.signals_with_state_variation == 1
    assert report.signal_transition_count == 2
    assert report.pair_statistics.count == 1
    assert report.pair_statistics.blocker_fraction == 1.0
    # Bins follow the configured separation window rather than fixed edges, so
    # the label is derived the same way the report derives it.
    edges = _separation_bin_edges(10.0, 60.0)
    middle = f"{edges[1]:g}-{edges[2]:g}"
    assert dict(report.pair_statistics.distance_bin_counts)[middle] == 1
    assert report.to_record()["gate1_passed"] is True


def test_density_mean_and_population_standard_deviation_use_realized_counts() -> None:
    frames = (
        _frame(0.0, active=19, lane_length_km=1.0, state="rG"),
        _frame(1.0, active=20, lane_length_km=1.0, state="Gr", departed=1),
        _frame(2.0, active=21, lane_length_km=1.0, state="rG", departed=1),
    )

    report = MobilityValidator(_criteria()).validate(
        frames,
        pair_segments=(_pair(),),
        manifest_archived=True,
    )

    assert report.realized_density_mean == 20.0
    assert report.realized_density_std == pytest.approx(math.sqrt(2.0 / 3.0))
    assert report.realized_density_min == 19.0
    assert report.realized_density_max == 21.0
    assert _check("density_within_tolerance", report) is True


def test_density_tolerance_is_inclusive_at_five_percent() -> None:
    on_boundary = (
        _frame(0.0, active=21, lane_length_km=1.0, state="rG"),
        _frame(1.0, active=21, lane_length_km=1.0, state="Gr", departed=1),
    )
    outside = (
        _frame(0.0, active=21, lane_length_km=0.99, state="rG"),
        _frame(1.0, active=21, lane_length_km=0.99, state="Gr", departed=1),
    )
    validator = MobilityValidator(_criteria())

    boundary_report = validator.validate(
        on_boundary,
        pair_segments=(_pair(end_s=1.0),),
        manifest_archived=True,
    )
    outside_report = validator.validate(
        outside,
        pair_segments=(_pair(end_s=1.0),),
        manifest_archived=True,
    )

    assert boundary_report.density_relative_error == pytest.approx(0.05)
    assert _check("density_within_tolerance", boundary_report) is True
    assert _check("density_within_tolerance", outside_report) is False
    assert outside_report.gate1_passed is False


def test_unexplained_teleport_fails_but_explicitly_allowed_reason_passes() -> None:
    frames = (
        _frame(0.0, state="rG"),
        _frame(1.0, state="Gr", departed=1, teleports=("documented-collision",)),
    )
    pair = _pair(end_s=1.0)

    unexplained = MobilityValidator(_criteria()).validate(
        frames,
        pair_segments=(pair,),
        manifest_archived=True,
    )
    allowed = MobilityValidator(
        _criteria(allowed_teleport_reasons=frozenset({"documented-collision"}))
    ).validate(
        frames,
        pair_segments=(pair,),
        manifest_archived=True,
    )

    assert unexplained.teleport_count == 1
    assert unexplained.unexplained_teleport_count == 1
    assert dict(unexplained.teleport_reason_counts) == {"documented-collision": 1}
    assert _check("no_unexplained_teleportation", unexplained) is False
    assert allowed.unexplained_teleport_count == 0
    assert _check("no_unexplained_teleportation", allowed) is True
    assert allowed.gate1_passed is True


def test_insertion_signal_pair_and_manifest_failures_are_independently_visible() -> None:
    frames = (
        _frame(0.0, state="rG", insertion_failures=1),
        _frame(1.0, state="rG", departed=0),
    )
    invalid_distance_pair = _pair(initial_distance_m=65.0, end_s=1.0)

    report = MobilityValidator(_criteria()).validate(
        frames,
        pair_segments=(invalid_distance_pair,),
    )

    assert _check("stable_insertion_and_routing", report) is False
    assert _check("signal_state_variation", report) is False
    assert _check("tagged_pair_separation_logic", report) is False
    assert _check("manifest_archived", report) is False
    assert report.gate1_passed is False


def test_queues_are_derived_from_stopped_speeds_when_not_supplied() -> None:
    frames = (
        MobilityFrame(
            trace_id="trace-001",
            time_s=0.0,
            total_lane_length_km=0.1,
            vehicle_speeds_mps=(0.0, 8.0),
            signal_states=(("tls-1", "rG"),),
        ),
        MobilityFrame(
            trace_id="trace-001",
            time_s=1.0,
            total_lane_length_km=0.1,
            vehicle_speeds_mps=(0.05, 8.0),
            departed_vehicle_count=1,
            signal_states=(("tls-1", "Gr"),),
        ),
    )

    report = MobilityValidator(_criteria()).validate(
        frames,
        pair_segments=(_pair(end_s=1.0),),
        manifest_archived=True,
    )

    assert report.mean_queue_proxy_vehicles == 1.0
    assert report.mean_queue_length_m == 7.5


def test_mapping_input_and_exact_frame_deduplication() -> None:
    first: dict[str, object] = {
        "trace_id": "trace-001",
        "time_s": 0.0,
        "total_lane_length_km": 0.1,
        "vehicle_speeds_mps": [0.0, 8.0],
        "queued_vehicle_count": 1,
        "queue_length_m": 7.0,
        "departed_vehicle_count": 0,
        "insertion_failures": 0,
        "teleport_reasons": [],
        "signal_states": {"tls-1": "rG"},
    }
    second = {
        **first,
        "time_s": 1.0,
        "departed_vehicle_count": 1,
        "signal_states": {"tls-1": "Gr"},
    }

    report = MobilityValidator(_criteria()).validate(
        (first, first, second),
        pair_segments=(_pair(end_s=1.0).to_record(),),
        manifest_archived=True,
    )

    assert report.gate1_passed is True
    assert report.throughput_vehicle_count == 1


def test_conflicting_frame_and_multiple_trace_inputs_are_rejected() -> None:
    first = _frame(0.0)
    conflict = replace(first, vehicle_speeds_mps=(5.0, 5.0))
    other_trace = replace(_frame(1.0), trace_id="trace-002")
    validator = MobilityValidator(_criteria())

    with pytest.raises(ValueError, match="conflicting mobility frame"):
        validator.validate((first, conflict))
    with pytest.raises(ValueError, match="multiple trace IDs"):
        validator.validate((first, other_trace))
