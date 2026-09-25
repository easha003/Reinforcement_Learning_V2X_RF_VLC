"""Density-balanced Phase 8 training-iteration integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.checkpointing import restore_training_checkpoint
from hybrid_v2x_rl.agents.joint_training import (
    JOINT_DENSITY_REPORT_SCHEMA,
    JointDensityTrainingError,
    run_joint_density_training_iteration,
)
from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def config():
    return load_headline_config(PROJECT_ROOT)


def _vehicle(trace_id: str, time_s: float, index: int) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=trace_id,
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


def _source(
    root: Path,
    config,
    *,
    density: int,
) -> FrameTraceSource:
    trace_id = f"synthetic-d{density}-train-000"
    times = tuple(0.05 * index for index in range(11))
    vehicles = [
        _vehicle(trace_id, time_s, vehicle_index)
        for time_s in times
        for vehicle_index in range(1, 3)
    ]
    pair = {
        "trace_id": trace_id,
        "pair_id": f"pair-d{density}",
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
        ArtifactStore(root / f"artifacts-d{density}"),
        rows_per_part=8,
    ).write(
        trace_id=trace_id,
        vehicles=vehicles,
        signals=(),
        pairs=(pair,),
        network_definition="{}",
        route_definition="{}",
        resolved_config_yaml=f"test: joint-density-{density}\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": density},
    )
    return FrameTraceSource.discover(artifact.path, expected_split="train")


@pytest.fixture()
def sources(tmp_path: Path, config) -> tuple[FrameTraceSource, ...]:
    return tuple(
        _source(tmp_path, config, density=density)
        for density in (10, 20, 30)
    )


def test_joint_iteration_updates_one_shared_policy_across_every_density(
    config,
    sources: tuple[FrameTraceSource, ...],
    tmp_path: Path,
) -> None:
    result = run_joint_density_training_iteration(
        config,
        sources,
        output_root=tmp_path / "joint-run",
        policy_seed=1001,
        rollout_packets=7,
    )

    assert result.report["schema"] == JOINT_DENSITY_REPORT_SCHEMA
    assert result.report["scope"] == (
        "one density-balanced training iteration; not convergence evidence"
    )
    assert result.report["configured_densities_veh_per_lane_km"] == [
        10.0,
        20.0,
        30.0,
    ]
    assert result.report["rollout_target_packets"] == 7
    assert result.report["rollout_transitions"] == 12
    assert result.report["rollout_overshoot_packets"] == 5
    assert result.report["balanced_rounds"] == 2
    assert result.report["frame_scheduling"] == "adaptive_3_to_20"
    assert result.report["max_frames_per_trace"] is None
    assert result.report["density_rollout_transitions"] == {
        "10": 4,
        "20": 4,
        "30": 4,
    }
    assert len(result.report["segments"]) == 6
    assert {row["requested_max_frames"] for row in result.report["segments"]} == {3}
    assert result.metrics.environment_transitions == 18
    assert result.metrics.rollout_transitions == 12
    assert result.metrics.learning_rows == 6
    assert tuple(row.sample_count for row in result.metrics.densities) == (4, 4, 4)
    assert sum(result.report["action_counts"].values()) == 12

    rows = result.metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == result.metrics.as_dict()
    restored = restore_training_checkpoint(
        result.checkpoint.path,
        config=config,
        expected_sha256=result.checkpoint.sha256,
        restore_global_rng=False,
    )
    assert restored.counters == result.checkpoint.counters
    assert restored.counters.completed_iterations == 1
    assert restored.counters.environment_transitions == 18
    assert restored.counters.learning_transitions == 6
    assert tuple(restored.numpy_generators) == ("training_streams",)
    assert tuple(restored.torch_generators) == (
        "policy_actions",
        "ppo_minibatches",
    )


def test_joint_iteration_rejects_missing_density_before_publishing(
    config,
    sources: tuple[FrameTraceSource, ...],
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing-density"

    with pytest.raises(JointDensityTrainingError, match="every and only"):
        run_joint_density_training_iteration(
            config,
            sources[:-1],
            output_root=output,
            policy_seed=1001,
            rollout_packets=4,
            max_frames_per_trace=3,
        )

    assert not output.exists()


def test_joint_iteration_rejects_nonempty_output(
    config,
    sources: tuple[FrameTraceSource, ...],
    tmp_path: Path,
) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "existing.txt").write_text("preserve\n", encoding="utf-8")

    with pytest.raises(JointDensityTrainingError, match="must be empty"):
        run_joint_density_training_iteration(
            config,
            sources,
            output_root=output,
            policy_seed=1001,
            rollout_packets=4,
            max_frames_per_trace=3,
        )

    assert (output / "existing.txt").read_text(encoding="utf-8") == "preserve\n"
