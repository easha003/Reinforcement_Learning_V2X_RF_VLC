"""Trace-backed Phase 8 PPO smoke-training integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.checkpointing import restore_training_checkpoint
from hybrid_v2x_rl.agents.trace_training import (
    TRACE_SMOKE_REPORT_SCHEMA,
    TraceTrainingError,
    run_trace_smoke_training,
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
        "pair_id": "pair-smoke",
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
        resolved_config_yaml="test: trace-smoke-training\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 29},
    )
    return FrameTraceSource.discover(artifact.path, expected_split="train")


def test_trace_smoke_run_publishes_metrics_and_complete_checkpoint(
    config,
    source: FrameTraceSource,
    tmp_path: Path,
) -> None:
    result = run_trace_smoke_training(
        config,
        source,
        output_root=tmp_path / "run",
        policy_seed=1001,
        environment_seed=71,
        max_frames=3,
    )

    assert result.report["schema"] == TRACE_SMOKE_REPORT_SCHEMA
    assert result.report["scope"] == (
        "integration smoke run; not convergence or feasibility evidence"
    )
    assert result.metrics.environment_transitions == 3
    assert result.metrics.rollout_transitions == 2
    assert result.metrics.learning_rows == 1
    assert result.metrics.ppo.minibatch_updates == config.training.update_epochs
    assert result.metrics.ppo.optimizer_rows == config.training.update_epochs
    represented = [row for row in result.metrics.densities if row.sample_count]
    assert len(represented) == 1
    assert represented[0].density_veh_per_lane_km == 10.0
    assert represented[0].sample_count == 2
    assert sum(result.report["action_counts"].values()) == 2
    assert "episode_clusters" not in result.report["environment_report"]

    metric_rows = result.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(metric_rows) == 1
    assert json.loads(metric_rows[0]) == result.metrics.as_dict()
    restored = restore_training_checkpoint(
        result.checkpoint.path,
        config=config,
        expected_sha256=result.checkpoint.sha256,
        restore_global_rng=False,
    )
    assert restored.counters == result.checkpoint.counters
    assert tuple(restored.numpy_generators) == ("training_streams",)
    assert tuple(restored.torch_generators) == (
        "policy_actions",
        "ppo_minibatches",
    )


def test_trace_smoke_run_is_reproducible_and_output_is_immutable(
    config,
    source: FrameTraceSource,
    tmp_path: Path,
) -> None:
    first = run_trace_smoke_training(
        config,
        source,
        output_root=tmp_path / "first",
        policy_seed=1001,
        environment_seed=71,
        max_frames=3,
    )
    second = run_trace_smoke_training(
        config,
        source,
        output_root=tmp_path / "second",
        policy_seed=1001,
        environment_seed=71,
        max_frames=3,
    )

    assert second.metrics == first.metrics
    assert second.report["action_counts"] == first.report["action_counts"]
    assert second.report["environment_report"] == first.report["environment_report"]
    with pytest.raises(TraceTrainingError, match="must be empty"):
        run_trace_smoke_training(
            config,
            source,
            output_root=tmp_path / "first",
            policy_seed=1001,
            environment_seed=71,
            max_frames=3,
        )
