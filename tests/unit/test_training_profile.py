"""Phase 8 trace-training performance profile integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.trace_training import TRACE_TRAINING_STAGE_NAMES, TraceTrainingError
from hybrid_v2x_rl.agents.training_profile import (
    TRACE_TRAINING_PROFILE_SCHEMA,
    run_trace_training_profile,
)
from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d10-train-000"


def _vehicle(time_s: float, index: int) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=time_s,
        vehicle_id=f"veh-{index}",
        x_m=9.0 * index + 3.0 * time_s,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=3.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


@pytest.fixture(scope="module")
def config():
    return load_headline_config(PROJECT_ROOT)


@pytest.fixture()
def source(tmp_path: Path, config) -> FrameTraceSource:
    times = tuple(0.05 * index for index in range(11))
    vehicles = [
        _vehicle(time_s, vehicle_index) for time_s in times for vehicle_index in range(1, 3)
    ]
    pair = {
        "trace_id": TRACE_ID,
        "pair_id": "pair-profile",
        "tx_id": "veh-1",
        "rx_id": "veh-2",
        "start_s": 0.0,
        "end_s": 0.4,
        "duration_s": 0.4,
        "initial_distance_m": 9.0,
        "route_id": "route-0",
        "eligibility_reason": "route_diverged",
        "has_intervening_vehicle": False,
    }
    artifact = MobilityTraceWriter(
        ArtifactStore(tmp_path / "artifacts"),
        rows_per_part=8,
    ).write(
        trace_id=TRACE_ID,
        vehicles=vehicles,
        signals=(),
        pairs=(pair,),
        network_definition="{}",
        route_definition="{}",
        resolved_config_yaml="test: training-profile\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 31},
    )
    return FrameTraceSource.discover(artifact.path, expected_split="train")


def test_trace_training_profile_publishes_complete_measurements(
    config,
    source: FrameTraceSource,
    tmp_path: Path,
) -> None:
    result = run_trace_training_profile(
        config,
        source,
        output_root=tmp_path / "profile",
        policy_seed=1001,
        environment_seed=71,
        max_frames=3,
    )

    assert result.report["schema"] == TRACE_TRAINING_PROFILE_SCHEMA
    assert result.training_result.metrics.environment_transitions == 3
    assert json.loads(result.report_path.read_text(encoding="utf-8")) == dict(result.report)

    measurement = result.report["measurement"]
    assert isinstance(measurement, dict)
    assert measurement["total_seconds"] > 0.0
    assert measurement["dominant_stage"] in TRACE_TRAINING_STAGE_NAMES
    stage_rows = measurement["stages"]
    assert isinstance(stage_rows, list)
    assert {row["name"] for row in stage_rows} == set(TRACE_TRAINING_STAGE_NAMES)
    assert all(row["seconds"] > 0.0 for row in stage_rows)

    throughput = result.report["throughput"]
    assert isinstance(throughput, dict)
    assert all(value > 0.0 for value in throughput.values())
    memory = result.report["memory"]
    assert isinstance(memory, dict)
    assert memory["process_peak_rss_after_mib"] >= memory["process_peak_rss_before_mib"]
    estimates = result.report["wall_clock_estimates"]
    assert isinstance(estimates, dict)
    assert estimates["core_compute_hours_per_seed"] > 0.0
    assert estimates["core_compute_hours_all_configured_seeds_serial"] == pytest.approx(
        estimates["core_compute_hours_per_seed"] * len(config.training.policy_seeds)
    )


def test_trace_training_profile_requires_representative_frame_count(
    config,
    source: FrameTraceSource,
    tmp_path: Path,
) -> None:
    with pytest.raises(TraceTrainingError, match="at least three frames"):
        run_trace_training_profile(
            config,
            source,
            output_root=tmp_path / "too-short",
            policy_seed=1001,
            environment_seed=71,
            max_frames=2,
        )
