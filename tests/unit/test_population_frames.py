"""Phase 2 chronological population-frame and lifecycle invariants."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.mean_field.frames import (
    FrameReplayError,
    FrameTraceSource,
    PopulationFrame,
    PopulationFrameReader,
    TraceCatalog,
)
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

TRACE_ID = "synthetic-d10-train-000"
CONFIG_HASH = "a" * 64


def _vehicle(trace_id: str, time_s: float, index: int) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=trace_id,
        time_s=time_s,
        vehicle_id=f"veh-{index}",
        x_m=10.0 * index + time_s,
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


def _pair(
    pair_id: str,
    tx_id: str,
    rx_id: str,
    start_s: float,
    end_s: float,
    reason: str,
) -> dict[str, object]:
    return {
        "trace_id": TRACE_ID,
        "pair_id": pair_id,
        "tx_id": tx_id,
        "rx_id": rx_id,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "initial_distance_m": 10.0,
        "route_id": "route-0",
        "eligibility_reason": reason,
        "has_intervening_vehicle": False,
    }


@pytest.fixture()
def trace_path(tmp_path: Path) -> Path:
    times = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
    vehicles = [
        _vehicle(TRACE_ID, time_s, vehicle)
        for time_s in times
        for vehicle in range(1, 5)
    ]
    pairs = [
        _pair("pair-a", "veh-1", "veh-2", 0.0, 0.2, "route_diverged"),
        # The birth lies between decision frames and first appears at 0.1 s.
        _pair("pair-b", "veh-2", "veh-3", 0.05, 0.3, "trace_end"),
        _pair("pair-c", "veh-3", "veh-4", 0.0, 0.1, "max_duration"),
        # Positive source duration, but no decision grid point lies inside it.
        _pair("pair-short", "veh-1", "veh-4", 0.01, 0.05, "vehicle_missing"),
        # A zero-duration source record must never become a policy transition.
        _pair("pair-zero", "veh-1", "veh-3", 0.1, 0.1, "vehicle_missing"),
    ]
    artifact = MobilityTraceWriter(
        ArtifactStore(tmp_path / "artifacts"),
        rows_per_part=7,
    ).write(
        trace_id=TRACE_ID,
        vehicles=vehicles,
        signals=(),
        pairs=pairs,
        network_definition="{}",
        route_definition="{}",
        resolved_config_yaml="service:\n  generation_period_s: 0.1\n",
        config_hash=CONFIG_HASH,
        code_version="test",
        random_seeds={"mobility": 7},
    )
    return artifact.path


def _reader(trace_path: Path) -> PopulationFrameReader:
    return PopulationFrameReader(
        FrameTraceSource.discover(trace_path, expected_split="train"),
        generation_period_s=0.1,
        expected_config_hash=CONFIG_HASH,
    )


def _signature(frames: list[PopulationFrame]) -> list[tuple[object, ...]]:
    return [
        (
            frame.index,
            frame.time_s,
            frame.active_pair_ids,
            tuple(
                (
                    pair.pair_id,
                    pair.episode_step,
                    pair.lifecycle.born,
                    pair.lifecycle.terminated,
                    pair.lifecycle.truncated,
                    pair.lifecycle.bootstrap_valid,
                    pair.lifecycle.end_reason,
                )
                for pair in frame.pairs
            ),
        )
        for frame in frames
    ]


def test_catalog_preserves_configured_split_membership_and_order(tmp_path: Path) -> None:
    splits = TraceSplitConfig(
        train=("synthetic-d10-train-002", "synthetic-d10-train-000"),
        validation=("synthetic-d20-validation-000",),
        test=("synthetic-d30-test-001",),
    )
    catalog = TraceCatalog.from_splits(tmp_path, splits)

    assert tuple(source.trace_id for source in catalog.for_split("train")) == splits.train
    assert catalog.source("synthetic-d20-validation-000").split == "validation"
    with pytest.raises(FrameReplayError, match="not a unique configured source"):
        catalog.source("synthetic-d10-test-999")


def test_an_encoded_split_cannot_be_relabelled(tmp_path: Path) -> None:
    path = tmp_path / "synthetic-d10-test-000"

    with pytest.raises(FrameReplayError, match="does not match configured membership"):
        FrameTraceSource.discover(path, expected_split="train")


def test_frames_are_chronological_simultaneous_and_stably_ordered(trace_path: Path) -> None:
    frames = list(_reader(trace_path).iter_frames())

    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    assert [frame.time_s for frame in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert [frame.active_pair_ids for frame in frames] == [
        ("pair-a", "pair-c"),
        ("pair-a", "pair-b", "pair-c"),
        ("pair-a", "pair-b"),
        ("pair-b",),
    ]
    assert all(tuple(v.vehicle_id for v in frame.vehicles) == (
        "veh-1", "veh-2", "veh-3", "veh-4"
    ) for frame in frames)
    assert "pair-short" not in {pair for frame in frames for pair in frame.active_pair_ids}
    assert "pair-zero" not in {pair for frame in frames for pair in frame.active_pair_ids}


def test_lifecycle_flags_distinguish_birth_termination_and_both_truncations(
    trace_path: Path,
) -> None:
    frames = list(_reader(trace_path).iter_frames())
    by_frame = [{pair.pair_id: pair for pair in frame.pairs} for frame in frames]

    assert by_frame[0]["pair-a"].lifecycle.born
    assert by_frame[1]["pair-b"].lifecycle.born
    internal = by_frame[1]["pair-c"].lifecycle
    assert internal.continuing
    assert internal.truncated and not internal.terminated
    assert internal.bootstrap_valid
    assert internal.end_reason == "max_duration"

    natural = by_frame[2]["pair-a"].lifecycle
    assert natural.terminated and not natural.truncated
    assert not natural.bootstrap_valid
    assert natural.end_reason == "route_diverged"

    trace_end = by_frame[3]["pair-b"].lifecycle
    assert trace_end.truncated and not trace_end.terminated
    assert not trace_end.bootstrap_valid
    assert trace_end.end_reason == "trace_end"


def test_replay_is_deterministic_and_reconciles_source_counts(trace_path: Path) -> None:
    reader = _reader(trace_path)
    first = list(reader.iter_frames())
    second = list(reader.iter_frames())

    assert _signature(first) == _signature(second)
    report = reader.validate()
    assert report.source_vehicle_rows == 28
    assert report.source_pair_rows == 5
    assert report.positive_duration_pair_rows == 4
    assert report.zero_duration_pair_rows == 1
    assert report.no_decision_pair_rows == 1
    assert report.decision_pair_episodes == 3
    assert report.frames == 4
    assert report.nonempty_frames == 4
    assert report.pair_instances == 8
    assert report.births == 3
    assert report.continuing_instances == 5
    assert report.natural_terminations == 1
    assert report.internal_truncations == 1
    assert report.trace_end_truncations == 1


def test_endpoint_overlap_is_reported_without_changing_the_population(trace_path: Path) -> None:
    reader = _reader(trace_path)
    frames = list(reader.iter_frames())
    report = reader.validate()

    assert frames[1].shared_endpoint_ids == ("veh-2", "veh-3")
    assert frames[1].pairs_with_shared_endpoint == 3
    assert report.frames_with_endpoint_overlap == 2
    assert report.pair_instances_with_endpoint_overlap == 5
    assert report.overlapping_endpoint_assignments == 6
    assert report.max_endpoint_multiplicity == 2


def test_artifact_configuration_mismatch_is_rejected(trace_path: Path) -> None:
    source = FrameTraceSource.discover(trace_path)

    with pytest.raises(FrameReplayError, match="different configuration"):
        PopulationFrameReader(
            source,
            generation_period_s=0.1,
            expected_config_hash="b" * 64,
        )


@pytest.mark.parametrize("period", [0.0, -0.1, float("nan"), True])
def test_invalid_decision_period_is_rejected(trace_path: Path, period: object) -> None:
    source = FrameTraceSource.discover(trace_path)

    with pytest.raises(ValueError, match="finite and positive"):
        PopulationFrameReader(source, generation_period_s=period)  # type: ignore[arg-type]
