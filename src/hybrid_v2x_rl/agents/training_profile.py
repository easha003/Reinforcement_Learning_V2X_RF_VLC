"""Measured Phase 8 performance profile for trace-backed PPO training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Final

import torch

from hybrid_v2x_rl import __version__
from hybrid_v2x_rl.agents.trace_training import (
    TRACE_TRAINING_STAGE_NAMES,
    TraceSmokeTrainingResult,
    TraceTrainingError,
    TraceTrainingTimingObserver,
    run_trace_smoke_training,
)
from hybrid_v2x_rl.agents.training_metrics import TrainingIterationMetrics
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource

TRACE_TRAINING_PROFILE_SCHEMA: Final = "hybrid-rf-vlc-rl.training-profile.v1"


@dataclass(frozen=True, slots=True)
class TraceTrainingProfileResult:
    """One immutable machine-readable profile and its underlying smoke run."""

    report_path: Path
    training_result: TraceSmokeTrainingResult
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.report_path.is_file() or self.report_path.is_symlink():
            raise TraceTrainingError(
                "published profile report must be a regular file",
                artifact_path=self.report_path,
            )
        if not isinstance(self.training_result, TraceSmokeTrainingResult):
            raise TraceTrainingError("profile result requires a smoke-training result")
        if not isinstance(self.report, Mapping):
            raise TraceTrainingError("profile report must be a mapping")


class _StageRecorder(TraceTrainingTimingObserver):
    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}

    def record_stage(self, *, name: str, elapsed_seconds: float) -> None:
        if name not in TRACE_TRAINING_STAGE_NAMES:
            raise TraceTrainingError(
                "training profile received an unknown stage",
                context={"stage": name},
            )
        if name in self._seconds:
            raise TraceTrainingError(
                "training profile received a duplicate stage",
                context={"stage": name},
            )
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
            raise TraceTrainingError(
                "training profile stage duration must be finite and positive",
                context={"stage": name, "elapsed_seconds": elapsed_seconds},
            )
        self._seconds[name] = elapsed_seconds

    def complete(self) -> dict[str, float]:
        missing = tuple(name for name in TRACE_TRAINING_STAGE_NAMES if name not in self._seconds)
        if missing:
            raise TraceTrainingError(
                "training profile is missing stage timings",
                context={"missing": missing},
            )
        return {name: self._seconds[name] for name in TRACE_TRAINING_STAGE_NAMES}


def run_trace_training_profile(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    output_root: str | Path,
    policy_seed: int = 1001,
    environment_seed: int | None = None,
    max_frames: int = 20,
) -> TraceTrainingProfileResult:
    """Measure one representative CPU training iteration and publish estimates.

    The underlying rollout, update, checkpoint, and metrics are exactly the
    smoke-training path. Timing callbacks observe stage boundaries only and do
    not alter random streams or learner state.
    """

    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 3:
        raise TraceTrainingError("a performance profile requires at least three frames")
    recorder = _StageRecorder()
    peak_before_mib = _maximum_resident_set_mib()
    profile_started = perf_counter()
    training_result = run_trace_smoke_training(
        config,
        source,
        output_root=output_root,
        policy_seed=policy_seed,
        environment_seed=environment_seed,
        max_frames=max_frames,
        timing_observer=recorder,
    )
    total_seconds = perf_counter() - profile_started
    peak_after_mib = _maximum_resident_set_mib()
    stages = recorder.complete()
    payload = _profile_payload(
        config=config,
        source=source,
        max_frames=max_frames,
        training_result=training_result,
        stages=stages,
        total_seconds=total_seconds,
        peak_before_mib=peak_before_mib,
        peak_after_mib=peak_after_mib,
    )
    report_path = training_result.report_path.parent / "profile.json"
    if report_path.exists():
        raise TraceTrainingError(
            "profile report path unexpectedly exists",
            artifact_path=report_path,
        )
    report_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TraceTrainingProfileResult(
        report_path=report_path,
        training_result=training_result,
        report=MappingProxyType(payload),
    )


def _profile_payload(
    *,
    config: ProjectConfig,
    source: FrameTraceSource,
    max_frames: int,
    training_result: TraceSmokeTrainingResult,
    stages: dict[str, float],
    total_seconds: float,
    peak_before_mib: float,
    peak_after_mib: float,
) -> dict[str, object]:
    if not math.isfinite(total_seconds) or total_seconds <= 0.0:
        raise TraceTrainingError("training profile total duration must be finite and positive")
    measured_stage_seconds = math.fsum(stages.values())
    stage_ranking = sorted(stages, key=stages.__getitem__, reverse=True)
    stage_rows = [
        {
            "name": name,
            "seconds": stages[name],
            "fraction_of_total": stages[name] / total_seconds,
        }
        for name in stage_ranking
    ]

    metrics = training_result.metrics
    environment_tps = metrics.environment_transitions / stages["rollout"]
    rollout_tps = metrics.rollout_transitions / stages["rollout"]
    preparation_tps = metrics.rollout_transitions / stages["rollout_preparation"]
    optimizer_row_tps = metrics.ppo.optimizer_rows / stages["ppo_optimization"]
    end_to_end_tps = metrics.environment_transitions / total_seconds
    estimates = _wall_clock_estimates(
        config=config,
        metrics=metrics,
        environment_tps=environment_tps,
        preparation_tps=preparation_tps,
        optimizer_row_tps=optimizer_row_tps,
        end_to_end_tps=end_to_end_tps,
    )
    training_report_sha256 = hashlib.sha256(training_result.report_path.read_bytes()).hexdigest()
    return {
        "schema": TRACE_TRAINING_PROFILE_SCHEMA,
        "scope": "single-process CPU profile; not a training or feasibility result",
        "package_version": __version__,
        "config_hash": config_hash(config),
        "trace_id": source.trace_id,
        "split": source.split,
        "density_veh_per_lane_km": source.density,
        "frames_processed": max_frames,
        "measurement": {
            "total_seconds": total_seconds,
            "measured_stage_seconds": measured_stage_seconds,
            "unattributed_seconds": max(0.0, total_seconds - measured_stage_seconds),
            "dominant_stage": stage_ranking[0],
            "stages": stage_rows,
        },
        "throughput": {
            "environment_transitions_per_second": environment_tps,
            "rollout_transitions_per_second": rollout_tps,
            "rollout_preparation_rows_per_second": preparation_tps,
            "optimizer_row_exposures_per_second": optimizer_row_tps,
            "end_to_end_environment_transitions_per_second": end_to_end_tps,
        },
        "memory": {
            "process_peak_rss_before_mib": peak_before_mib,
            "process_peak_rss_after_mib": peak_after_mib,
            "observed_peak_growth_mib": max(0.0, peak_after_mib - peak_before_mib),
            "measurement": "getrusage(RUSAGE_SELF).ru_maxrss high-water mark",
        },
        "hardware": _hardware_payload(),
        "workload": {
            "environment_transitions": metrics.environment_transitions,
            "rollout_transitions": metrics.rollout_transitions,
            "learning_rows": metrics.learning_rows,
            "optimizer_updates": metrics.ppo.minibatch_updates,
            "optimizer_row_exposures": metrics.ppo.optimizer_rows,
            "configured_rollout_packets": config.training.rollout_packets,
            "configured_minibatch_size": config.training.minibatch_size,
            "configured_update_epochs": config.training.update_epochs,
        },
        "wall_clock_estimates": estimates,
        "training_artifacts": {
            "report_path": str(training_result.report_path),
            "report_sha256": training_report_sha256,
            "checkpoint_path": str(training_result.checkpoint.path),
            "checkpoint_sha256": training_result.checkpoint.sha256,
        },
        "limitations": [
            "The profile covers one density-10 trace and one process on the recorded host.",
            "Peak RSS is a process high-water mark; growth can understate allocations after an earlier peak.",
            "Core estimates extrapolate measured stage throughput and exclude validation, evaluation, and queue time.",
            "The smoke-scale end-to-end estimate repeats setup and checkpoint overhead more often than a full trainer would.",
            "Longer multi-density runs can expose cache, I/O, lifecycle, and thermal behavior absent here.",
        ],
    }


def _wall_clock_estimates(
    *,
    config: ProjectConfig,
    metrics: TrainingIterationMetrics,
    environment_tps: float,
    preparation_tps: float,
    optimizer_row_tps: float,
    end_to_end_tps: float,
) -> dict[str, object]:
    environment_transitions = metrics.environment_transitions
    rollout_transitions = metrics.rollout_transitions
    learning_rows = metrics.learning_rows
    target = config.training.total_transitions_per_seed
    rollout_fraction = rollout_transitions / environment_transitions
    learning_fraction = learning_rows / environment_transitions
    optimizer_exposures = target * learning_fraction * config.training.update_epochs
    environment_seconds = target / environment_tps
    preparation_seconds = target * rollout_fraction / preparation_tps
    optimizer_seconds = optimizer_exposures / optimizer_row_tps
    core_seconds = environment_seconds + preparation_seconds + optimizer_seconds
    naive_seconds = target / end_to_end_tps
    seed_count = len(config.training.policy_seeds)
    update_count = math.ceil(target / config.training.rollout_packets)
    per_update_core_seconds = core_seconds / update_count
    return {
        "target_environment_transitions_per_seed": target,
        "configured_policy_seed_count": seed_count,
        "estimated_update_count_per_seed": update_count,
        "measured_rollout_fraction": rollout_fraction,
        "measured_learning_fraction": learning_fraction,
        "estimated_optimizer_row_exposures_per_seed": optimizer_exposures,
        "environment_hours_per_seed": environment_seconds / 3600.0,
        "rollout_preparation_hours_per_seed": preparation_seconds / 3600.0,
        "ppo_optimization_hours_per_seed": optimizer_seconds / 3600.0,
        "core_compute_hours_per_seed": core_seconds / 3600.0,
        "core_compute_hours_all_configured_seeds_serial": (core_seconds * seed_count / 3600.0),
        "estimated_core_seconds_per_configured_update": per_update_core_seconds,
        "naive_smoke_scale_hours_per_seed": naive_seconds / 3600.0,
    }


def _maximum_resident_set_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = float(1024**2 if sys.platform == "darwin" else 1024)
    value = raw / divisor
    if not math.isfinite(value) or value < 0.0:
        raise TraceTrainingError("process peak RSS is invalid")
    return value


def _hardware_payload() -> dict[str, object]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_mib": _physical_memory_mib(),
    }


def _physical_memory_mib() -> float | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size / float(1024**2)


__all__ = [
    "TRACE_TRAINING_PROFILE_SCHEMA",
    "TraceTrainingProfileResult",
    "run_trace_training_profile",
]
