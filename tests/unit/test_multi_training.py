"""Resumable joint-density Phase 8 training integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from hybrid_v2x_rl.agents.checkpointing import restore_training_checkpoint
from hybrid_v2x_rl.agents.constraint_diagnostics import (
    CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA,
)
from hybrid_v2x_rl.agents.joint_training import JointDensityTrainingError
from hybrid_v2x_rl.agents.multi_training import (
    MULTI_TRAINING_ITERATION_SCHEMA,
    curriculum_position,
    run_joint_density_training,
)
from hybrid_v2x_rl.agents.trace_windows import TRACE_WINDOW_SCHEDULE_SCHEMA
from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _source(root: Path, config, *, density: int) -> FrameTraceSource:
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
        resolved_config_yaml=f"test: resumable-training-{density}\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": density},
    )
    return FrameTraceSource.discover(artifact.path, expected_split="train")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_restored_learners_equal(left, right) -> None:
    for left_model, right_model in (
        (left.updater.actor, right.updater.actor),
        (left.updater.reward_critic, right.updater.reward_critic),
        (left.updater.cost_critic, right.updater.cost_critic),
    ):
        for name, tensor in left_model.state_dict().items():
            torch.testing.assert_close(tensor, right_model.state_dict()[name], rtol=0, atol=0)
    assert left.dual_ascent.snapshot() == right.dual_ascent.snapshot()
    assert left.normalizer.state_dict() == right.normalizer.state_dict()
    assert (
        left.numpy_generators["training_streams"].bit_generator.state
        == right.numpy_generators["training_streams"].bit_generator.state
    )
    for name in ("policy_actions", "ppo_minibatches"):
        assert torch.equal(
            left.torch_generators[name].get_state(),
            right.torch_generators[name].get_state(),
        )


def test_resume_matches_uninterrupted_training_and_advances_curriculum(
    tmp_path: Path,
) -> None:
    base = load_headline_config(PROJECT_ROOT)
    training = base.training.model_copy(update={"total_transitions_per_seed": 150})
    config = base.model_copy(update={"training": training})
    sources = tuple(_source(tmp_path, config, density=value) for value in (10, 20, 30))

    uninterrupted = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "uninterrupted",
        rollout_packets=7,
        max_frames_per_trace=3,
        max_iterations=2,
    )
    first = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "resumed",
        rollout_packets=7,
        max_frames_per_trace=3,
        max_iterations=1,
    )
    immutable = {
        path: _sha256(path) for path in (*first.checkpoint_paths, *first.iteration_report_paths)
    }
    resumed = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "resumed",
        resume_checkpoint=first.latest_checkpoint.path,
        expected_checkpoint_sha256=first.latest_checkpoint.sha256,
        rollout_packets=7,
        max_frames_per_trace=3,
        max_iterations=1,
    )

    assert resumed.iterations_run == 1
    assert resumed.latest_checkpoint.counters.completed_iterations == 2
    assert resumed.latest_checkpoint.counters.environment_transitions == 33
    assert all(_sha256(path) == digest for path, digest in immutable.items())
    assert uninterrupted.metrics_path.read_text(encoding="utf-8") == (
        resumed.metrics_path.read_text(encoding="utf-8")
    )

    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((tmp_path / "resumed" / "iterations").iterdir())
    ]
    assert [report["schema"] for report in reports] == [
        MULTI_TRAINING_ITERATION_SCHEMA,
        MULTI_TRAINING_ITERATION_SCHEMA,
    ]
    assert [report["curriculum"]["index"] for report in reports] == [0, 1]
    assert [report["curriculum"]["miss_budget"] for report in reports] == [
        0.01,
        0.001,
    ]
    assert reports[0]["curriculum"]["boundary_crossed"] is True
    assert reports[0]["curriculum"]["boundary_overshoot_transitions"] == 0
    assert [report["trace_window_schedule"] for report in reports] == [
        TRACE_WINDOW_SCHEDULE_SCHEMA,
        TRACE_WINDOW_SCHEDULE_SCHEMA,
    ]
    assert [report["constraint_pressure"]["schema"] for report in reports] == [
        CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA,
        CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA,
    ]
    assert [report["constraint_pressure"]["learning_rows"] for report in reports] == [
        report["learning_rows"] for report in reports
    ]
    assert [
        [segment["trace_window"]["start_frame_index"] for segment in report["segments"]]
        for report in reports
    ] == [
        [0, 0, 0, 3, 3, 3],
        [1, 1, 1, 0, 0, 0],
    ]

    uninterrupted_state = restore_training_checkpoint(
        uninterrupted.latest_checkpoint.path,
        config=config,
        restore_global_rng=False,
    )
    resumed_state = restore_training_checkpoint(
        resumed.latest_checkpoint.path,
        config=config,
        restore_global_rng=False,
    )
    assert uninterrupted_state.counters == resumed_state.counters
    _assert_restored_learners_equal(uninterrupted_state, resumed_state)


def test_constraint_diagnostics_do_not_change_checkpoint_or_metrics(
    tmp_path: Path,
) -> None:
    base = load_headline_config(PROJECT_ROOT)
    training = base.training.model_copy(update={"total_transitions_per_seed": 150})
    config = base.model_copy(update={"training": training})
    sources = tuple(_source(tmp_path, config, density=value) for value in (10, 20, 30))

    enabled = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "diagnostics-enabled",
        rollout_packets=7,
        max_frames_per_trace=3,
        max_iterations=1,
    )
    disabled = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "diagnostics-disabled",
        rollout_packets=7,
        max_frames_per_trace=3,
        max_iterations=1,
        _record_constraint_diagnostics=False,
    )

    assert enabled.latest_checkpoint.sha256 == disabled.latest_checkpoint.sha256
    assert enabled.metrics_path.read_bytes() == disabled.metrics_path.read_bytes()
    enabled_report = json.loads(enabled.iteration_report_paths[0].read_text())
    disabled_report = json.loads(disabled.iteration_report_paths[0].read_text())
    pressure = enabled_report["constraint_pressure"]
    assert pressure["schema"] == CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA
    assert pressure["learning_rows"] == enabled_report["learning_rows"]
    assert disabled_report["constraint_pressure"] is None


def test_budget_never_starts_an_overrunning_balanced_round(tmp_path: Path) -> None:
    base = load_headline_config(PROJECT_ROOT)
    only_stage = base.training.curriculum[0].model_copy(update={"fraction": 1.0})
    training = base.training.model_copy(
        update={
            "total_transitions_per_seed": 20,
            "curriculum": (only_stage,),
        }
    )
    config = base.model_copy(update={"training": training})
    sources = tuple(_source(tmp_path, config, density=value) for value in (10, 20, 30))

    result = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "budgeted",
        rollout_packets=7,
        max_frames_per_trace=3,
    )

    assert result.stop_reason == "insufficient_budget_for_balanced_round"
    assert result.latest_checkpoint.counters.environment_transitions == 15
    assert result.report["configured_transition_budget"] == 20
    assert result.report["unused_transition_budget"] == 5
    assert result.iterations_run == 1
    assert len(result.metrics_path.read_text(encoding="utf-8").splitlines()) == 1


def test_resume_rejects_iteration_history_from_before_trace_window_contract(
    tmp_path: Path,
) -> None:
    base = load_headline_config(PROJECT_ROOT)
    training = base.training.model_copy(update={"total_transitions_per_seed": 180})
    config = base.model_copy(update={"training": training})
    sources = tuple(_source(tmp_path, config, density=value) for value in (10, 20, 30))
    result = run_joint_density_training(
        config,
        sources,
        output_root=tmp_path / "old-history",
        rollout_packets=7,
        max_frames_per_trace=3,
        max_iterations=1,
    )
    report_path = result.iteration_report_paths[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema"] = "hybrid-rf-vlc-rl.joint-density-training-iteration.v1"
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JointDensityTrainingError, match="predates or violates"):
        run_joint_density_training(
            config,
            sources,
            output_root=result.output_root,
            resume_checkpoint=result.latest_checkpoint.path,
            expected_checkpoint_sha256=result.latest_checkpoint.sha256,
            rollout_packets=7,
            max_frames_per_trace=3,
            max_iterations=1,
        )


def test_curriculum_positions_use_cumulative_declared_fractions() -> None:
    base = load_headline_config(PROJECT_ROOT)
    training = base.training.model_copy(update={"total_transitions_per_seed": 100})
    config = base.model_copy(update={"training": training})

    assert curriculum_position(config, 0).index == 0
    assert curriculum_position(config, 9).miss_budget == pytest.approx(0.01)
    assert curriculum_position(config, 10).index == 1
    assert curriculum_position(config, 29).miss_budget == pytest.approx(0.001)
    assert curriculum_position(config, 30).index == 2
    assert curriculum_position(config, 99).miss_budget == pytest.approx(0.0001)
