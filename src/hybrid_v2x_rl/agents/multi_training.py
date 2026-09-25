"""Resumable, budget-bounded joint-density primal-dual PPO training."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import torch

from hybrid_v2x_rl.agents.checkpointing import (
    TrainingCheckpointSummary,
    TrainingCounters,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.joint_training import (
    JOINT_DENSITY_POLICY_NAME,
    JointDensityTrainingError,
    _adaptive_frame_limit,
    _sources_by_density,
    _validate_request,
    merge_prepared_rollouts,
)
from hybrid_v2x_rl.agents.ppo import PPOUpdater
from hybrid_v2x_rl.agents.trace_training import (
    PreparedRollout,
    TracePPOCollector,
    optimize_ppo,
    prepare_rollout,
)
from hybrid_v2x_rl.agents.trace_windows import (
    TRACE_WINDOW_SCHEDULE_SCHEMA,
    TraceWindowSelection,
    select_trace_window,
)
from hybrid_v2x_rl.agents.training_metrics import (
    TrainingIterationMetrics,
    TrainingMetricsJSONL,
    build_training_iteration_metrics,
)
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.critic_observations import CentralizedCriticBuilder
from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout_with_state
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PopulationFrameReader,
)
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizer

MULTI_TRAINING_ITERATION_SCHEMA: Final = "hybrid-rf-vlc-rl.joint-density-training-iteration.v2"
MULTI_TRAINING_SESSION_SCHEMA: Final = "hybrid-rf-vlc-rl.joint-density-training-session.v1"
_ACTION_STREAM_XOR: Final = 0xA17C_D315_5EED_1001
_MINIBATCH_STREAM_XOR: Final = 0xB47C_D315_5EED_1002
_NUMPY_STREAM_XOR: Final = 0xC57C_D315_5EED_1003
_NUMPY_GENERATOR_NAME: Final = "training_streams"
_ACTION_GENERATOR_NAME: Final = "policy_actions"
_MINIBATCH_GENERATOR_NAME: Final = "ppo_minibatches"


@dataclass(frozen=True, slots=True)
class CurriculumPosition:
    """The curriculum stage owning one cumulative transition position."""

    index: int
    start_transition: int
    end_transition: int
    fraction: float
    miss_budget: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "index": self.index,
            "start_transition": self.start_transition,
            "end_transition": self.end_transition,
            "fraction": self.fraction,
            "miss_budget": self.miss_budget,
        }


@dataclass(frozen=True, slots=True)
class MultiIterationTrainingResult:
    """Artifacts published by one fresh or resumed training invocation."""

    output_root: Path
    metrics_path: Path
    session_report_path: Path
    latest_checkpoint: TrainingCheckpointSummary
    checkpoint_paths: tuple[Path, ...]
    iteration_report_paths: tuple[Path, ...]
    iterations_run: int
    stop_reason: str
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        for path in (
            self.metrics_path,
            self.session_report_path,
            self.latest_checkpoint.path,
            *self.checkpoint_paths,
            *self.iteration_report_paths,
        ):
            if not path.is_file() or path.is_symlink():
                raise JointDensityTrainingError(
                    "multi-iteration training published an invalid artifact",
                    artifact_path=path,
                )
        if self.iterations_run != len(self.checkpoint_paths) or self.iterations_run != len(
            self.iteration_report_paths
        ):
            raise JointDensityTrainingError(
                "multi-iteration result artifact counts do not match iterations_run"
            )


@dataclass(slots=True)
class _Runtime:
    updater: PPOUpdater
    dual_ascent: PerDensityDualAscent
    normalizer: ObservationNormalizer
    counters: TrainingCounters
    numpy_generator: np.random.Generator
    action_generator: torch.Generator
    minibatch_generator: torch.Generator


@dataclass(frozen=True, slots=True)
class _IterationOutcome:
    metrics: TrainingIterationMetrics
    prepared: PreparedRollout
    counters: TrainingCounters
    normalizer: ObservationNormalizer
    balanced_rounds: int
    environment_transitions: int
    episodes_completed: int
    segment_reports: tuple[dict[str, object], ...]
    curriculum: CurriculumPosition
    rollout_target: int
    budget_limited: bool


def curriculum_position(
    config: ProjectConfig,
    environment_transitions: int,
) -> CurriculumPosition:
    """Resolve a zero-based transition position against cumulative stage fractions."""

    if not isinstance(config, ProjectConfig):
        raise JointDensityTrainingError("curriculum selection requires a ProjectConfig")
    if (
        not isinstance(environment_transitions, int)
        or isinstance(environment_transitions, bool)
        or environment_transitions < 0
    ):
        raise JointDensityTrainingError(
            "curriculum environment transitions must be a nonnegative integer"
        )
    total = config.training.total_transitions_per_seed
    if environment_transitions >= total:
        raise JointDensityTrainingError(
            "curriculum position must be below the configured transition budget"
        )

    start = 0
    cumulative_fraction = 0.0
    stages = config.training.curriculum
    for index, stage in enumerate(stages):
        cumulative_fraction += stage.fraction
        end = (
            total
            if index == len(stages) - 1
            else min(total, int(round(total * cumulative_fraction)))
        )
        if environment_transitions < end:
            return CurriculumPosition(
                index=index,
                start_transition=start,
                end_transition=end,
                fraction=stage.fraction,
                miss_budget=stage.miss_budget,
            )
        start = end
    raise JointDensityTrainingError("curriculum fractions do not cover the seed budget")


def run_joint_density_training(
    config: ProjectConfig,
    sources: tuple[FrameTraceSource, ...],
    *,
    output_root: str | Path,
    policy_seed: int = 1001,
    resume_checkpoint: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    rollout_packets: int | None = None,
    max_frames_per_trace: int | None = None,
    max_iterations: int | None = None,
) -> MultiIterationTrainingResult:
    """Train until the configured budget, or a declared invocation limit, is reached.

    A balanced density round is the smallest collection unit. Before each round,
    the exact number of acted pair transitions is computed from immutable pair
    schedules. A round that would cross the configured per-seed budget is never
    started. Consequently, the run can finish with a small unused tail when the
    remaining budget cannot hold one complete density-balanced round.
    """

    base_rollout_target = _validate_request(
        config=config,
        sources=sources,
        policy_seed=policy_seed,
        rollout_packets=rollout_packets,
        max_frames_per_trace=max_frames_per_trace,
    )
    if max_iterations is not None and (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise JointDensityTrainingError("max_iterations must be positive or None")
    grouped = _sources_by_density(config, sources)
    destination = _prepare_run_root(
        output_root,
        resume=resume_checkpoint is not None,
    )
    metrics_path = destination / "metrics.jsonl"

    if resume_checkpoint is None:
        runtime = _fresh_runtime(config, policy_seed=policy_seed)
        start_iteration = 0
    else:
        runtime = _resume_runtime(
            config,
            destination=destination,
            checkpoint_path=resume_checkpoint,
            expected_sha256=expected_checkpoint_sha256,
            policy_seed=policy_seed,
            metrics_path=metrics_path,
        )
        start_iteration = runtime.counters.completed_iterations

    checkpoints: list[TrainingCheckpointSummary] = []
    iteration_paths: list[Path] = []
    total_budget = config.training.total_transitions_per_seed
    stop_reason = "configured_transition_budget_reached"
    blocked_by_round = False

    while runtime.counters.environment_transitions < total_budget:
        if max_iterations is not None and len(checkpoints) >= max_iterations:
            stop_reason = "invocation_iteration_limit_reached"
            break
        stage = curriculum_position(config, runtime.counters.environment_transitions)
        remaining_budget = total_budget - runtime.counters.environment_transitions
        remaining_stage = stage.end_transition - runtime.counters.environment_transitions
        iteration_target = min(
            base_rollout_target,
            remaining_budget,
            max(1, remaining_stage),
        )
        outcome = _execute_iteration(
            config=config,
            grouped=grouped,
            policy_seed=policy_seed,
            runtime=runtime,
            rollout_target=iteration_target,
            max_frames_per_trace=max_frames_per_trace,
            curriculum=stage,
        )
        if outcome is None:
            stop_reason = "insufficient_budget_for_balanced_round"
            blocked_by_round = True
            break

        completion = outcome.counters.completed_iterations
        checkpoint_path = destination / "checkpoints" / f"checkpoint-iteration-{completion:06d}.pt"
        checkpoint = save_training_checkpoint(
            checkpoint_path,
            config=config,
            policy_seed=policy_seed,
            updater=runtime.updater,
            dual_ascent=runtime.dual_ascent,
            normalizer=outcome.normalizer,
            counters=outcome.counters,
            numpy_generators={_NUMPY_GENERATOR_NAME: runtime.numpy_generator},
            torch_generators={
                _ACTION_GENERATOR_NAME: runtime.action_generator,
                _MINIBATCH_GENERATOR_NAME: runtime.minibatch_generator,
            },
        )
        iteration_payload = _iteration_payload(
            config=config,
            policy_seed=policy_seed,
            outcome=outcome,
            checkpoint=checkpoint,
            max_frames_per_trace=max_frames_per_trace,
        )
        iteration_path = destination / "iterations" / f"iteration-{completion:06d}.json"
        _write_new_json(iteration_path, iteration_payload)
        TrainingMetricsJSONL(metrics_path).append(outcome.metrics)

        runtime.normalizer = outcome.normalizer
        runtime.counters = outcome.counters
        checkpoints.append(checkpoint)
        iteration_paths.append(iteration_path)

    if not checkpoints and resume_checkpoint is None:
        reason = "the configured budget cannot hold one density-balanced round"
        raise JointDensityTrainingError(reason)

    if checkpoints:
        latest = checkpoints[-1]
    else:
        if resume_checkpoint is None:  # pragma: no cover - fresh no-update fails above.
            raise JointDensityTrainingError("fresh training produced no checkpoint")
        latest = _checkpoint_summary(
            Path(resume_checkpoint).expanduser().resolve(strict=True),
            config=config,
            policy_seed=policy_seed,
            counters=runtime.counters,
            sha256=expected_checkpoint_sha256,
        )
    if runtime.counters.environment_transitions == total_budget:
        stop_reason = "configured_transition_budget_reached"
    elif blocked_by_round:
        stop_reason = "insufficient_budget_for_balanced_round"

    session_payload: dict[str, object] = {
        "schema": MULTI_TRAINING_SESSION_SCHEMA,
        "scope": "one fresh or resumed immutable joint-density training invocation",
        "config_hash": config_hash(config),
        "policy_seed": policy_seed,
        "resumed": resume_checkpoint is not None,
        "start_completed_iterations": start_iteration,
        "end_completed_iterations": runtime.counters.completed_iterations,
        "iterations_run": len(checkpoints),
        "stop_reason": stop_reason,
        "configured_transition_budget": total_budget,
        "unused_transition_budget": (total_budget - runtime.counters.environment_transitions),
        "final_counters": runtime.counters.as_dict(),
        "metrics_path": str(metrics_path),
        "latest_checkpoint": {
            "path": str(latest.path),
            "sha256": latest.sha256,
            "size_bytes": latest.size_bytes,
        },
        "published_checkpoints": [str(item.path) for item in checkpoints],
        "published_iteration_reports": [str(path) for path in iteration_paths],
    }
    session_path = (
        destination
        / "sessions"
        / (f"session-{start_iteration:06d}-{runtime.counters.completed_iterations:06d}.json")
    )
    _write_new_json(session_path, session_payload)
    return MultiIterationTrainingResult(
        output_root=destination,
        metrics_path=metrics_path,
        session_report_path=session_path,
        latest_checkpoint=latest,
        checkpoint_paths=tuple(item.path for item in checkpoints),
        iteration_report_paths=tuple(iteration_paths),
        iterations_run=len(checkpoints),
        stop_reason=stop_reason,
        report=MappingProxyType(session_payload),
    )


def _fresh_runtime(config: ProjectConfig, *, policy_seed: int) -> _Runtime:
    random.seed(policy_seed)
    np.random.seed(policy_seed % (2**32))
    torch.manual_seed(policy_seed)
    numpy_generator = np.random.Generator(np.random.PCG64(policy_seed ^ _NUMPY_STREAM_XOR))
    action_seed = int(numpy_generator.integers(0, 2**63)) ^ _ACTION_STREAM_XOR
    minibatch_seed = int(numpy_generator.integers(0, 2**63)) ^ _MINIBATCH_STREAM_XOR
    critic_builder = CentralizedCriticBuilder.from_config(config)
    return _Runtime(
        updater=PPOUpdater.from_config(
            actor_observation_width=critic_builder.schema.actor_width,
            critic_observation_width=critic_builder.schema.critic_width,
            training=config.training,
        ),
        dual_ascent=PerDensityDualAscent.from_config(config.training),
        normalizer=ObservationNormalizer.from_config(config),
        counters=TrainingCounters(
            completed_iterations=0,
            environment_transitions=0,
            learning_transitions=0,
            episodes_completed=0,
            optimizer_steps=0,
        ),
        numpy_generator=numpy_generator,
        action_generator=torch.Generator().manual_seed(action_seed),
        minibatch_generator=torch.Generator().manual_seed(minibatch_seed),
    )


def _resume_runtime(
    config: ProjectConfig,
    *,
    destination: Path,
    checkpoint_path: str | Path,
    expected_sha256: str | None,
    policy_seed: int,
    metrics_path: Path,
) -> _Runtime:
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    expected_parent = (destination / "checkpoints").resolve(strict=True)
    if checkpoint.parent != expected_parent:
        raise JointDensityTrainingError(
            "resume checkpoint must belong to the output root's checkpoints directory",
            artifact_path=checkpoint,
        )
    restored = restore_training_checkpoint(
        checkpoint,
        config=config,
        expected_sha256=expected_sha256,
        restore_global_rng=True,
    )
    if restored.policy_seed != policy_seed:
        raise JointDensityTrainingError(
            "resume checkpoint policy seed does not match the requested seed"
        )
    expected_name = f"checkpoint-iteration-{restored.counters.completed_iterations:06d}.pt"
    if checkpoint.name != expected_name:
        raise JointDensityTrainingError(
            "resume checkpoint filename does not match its completed-iteration counter",
            artifact_path=checkpoint,
        )
    _validate_existing_history(
        destination,
        metrics_path=metrics_path,
        counters=restored.counters,
        policy_seed=policy_seed,
        expected_config_hash=config_hash(config),
        resume_checkpoint=checkpoint,
    )
    try:
        numpy_generator = restored.numpy_generators[_NUMPY_GENERATOR_NAME]
        action_generator = restored.torch_generators[_ACTION_GENERATOR_NAME]
        minibatch_generator = restored.torch_generators[_MINIBATCH_GENERATOR_NAME]
    except KeyError as exc:
        raise JointDensityTrainingError(
            "resume checkpoint is missing a required named random stream"
        ) from exc
    if set(restored.numpy_generators) != {_NUMPY_GENERATOR_NAME} or set(
        restored.torch_generators
    ) != {_ACTION_GENERATOR_NAME, _MINIBATCH_GENERATOR_NAME}:
        raise JointDensityTrainingError(
            "resume checkpoint named random streams do not match the trainer contract"
        )
    return _Runtime(
        updater=restored.updater,
        dual_ascent=restored.dual_ascent,
        normalizer=restored.normalizer,
        counters=restored.counters,
        numpy_generator=numpy_generator,
        action_generator=action_generator,
        minibatch_generator=minibatch_generator,
    )


def _execute_iteration(
    *,
    config: ProjectConfig,
    grouped: dict[float, tuple[FrameTraceSource, ...]],
    policy_seed: int,
    runtime: _Runtime,
    rollout_target: int,
    max_frames_per_trace: int | None,
    curriculum: CurriculumPosition,
) -> _IterationOutcome | None:
    normalization_state = runtime.normalizer.snapshot()
    parts: list[PreparedRollout] = []
    segment_reports: list[dict[str, object]] = []
    environment_transitions = 0
    episodes_completed = 0
    accumulated_packets = 0
    balanced_rounds = 0
    observed_packets_per_joint_frame: float | None = None
    budget_limited = False
    reader_cache: dict[Path, PopulationFrameReader] = {}
    transition_cache: dict[tuple[Path, int, int], int] = {}

    while accumulated_packets < rollout_target:
        frame_limit = (
            _adaptive_frame_limit(
                target=rollout_target,
                accumulated=accumulated_packets,
                observed_packets_per_joint_frame=observed_packets_per_joint_frame,
            )
            if max_frames_per_trace is None
            else max_frames_per_trace
        )
        selected: list[tuple[float, FrameTraceSource, TraceWindowSelection]] = []
        for density, candidates in sorted(grouped.items()):
            source = candidates[
                (runtime.counters.completed_iterations + balanced_rounds) % len(candidates)
            ]
            reader = _population_reader(config, source, cache=reader_cache)
            window = select_trace_window(
                available_frames=reader.decision_frame_count,
                requested_frames=frame_limit,
                completed_iterations=runtime.counters.completed_iterations,
                balanced_round=balanced_rounds,
            )
            selected.append((density, source, window))
        projected_round = sum(
            _segment_environment_transitions(
                config,
                source,
                window=window,
                reader_cache=reader_cache,
                cache=transition_cache,
            )
            for _, source, window in selected
        )
        projected_total = (
            runtime.counters.environment_transitions + environment_transitions + projected_round
        )
        if projected_total > config.training.total_transitions_per_seed:
            if not parts:
                return None
            budget_limited = True
            break

        packets_before_round = accumulated_packets
        environment_before_round = environment_transitions
        for density, source, window in selected:
            environment_seed = int(runtime.numpy_generator.integers(0, 2**63))
            collector = TracePPOCollector(
                config=config,
                updater=runtime.updater,
                action_generator=runtime.action_generator,
                policy_name=JOINT_DENSITY_POLICY_NAME,
            )
            rollout = run_policy_rollout_with_state(
                config,
                source,
                policy=collector,
                environment_seed=environment_seed,
                policy_seed=policy_seed,
                start_frame_index=window.start_frame_index,
                max_frames=frame_limit,
                normalization_state=normalization_state,
                frame_observer=collector,
            )
            frames = collector.frames
            if len(frames) != window.window_frames or len(frames) < 2:
                raise JointDensityTrainingError(
                    "a joint-density trace window did not produce its declared frames",
                    context={
                        "trace_id": source.trace_id,
                        "frames": len(frames),
                        "expected": window.window_frames,
                    },
                )
            if (
                rollout.report.first_frame_index != window.start_frame_index
                or rollout.report.last_frame_index != window.end_frame_index
            ):
                raise JointDensityTrainingError(
                    "rollout report does not match the scheduled trace window",
                    context={"trace_id": source.trace_id},
                )
            prepared = prepare_rollout(
                frames=frames,
                config=config,
                updater=runtime.updater,
                dual_ascent=runtime.dual_ascent,
                density_veh_per_lane_km=density,
            )
            parts.append(prepared)
            normalization_state = rollout.normalization_state
            accumulated_packets += prepared.rollout_transitions
            environment_transitions += rollout.report.transitions
            completed = (
                rollout.report.natural_terminations
                + rollout.report.internal_truncations
                + rollout.report.trace_end_truncations
            )
            episodes_completed += completed
            segment_reports.append(
                {
                    "round": balanced_rounds,
                    "trace_id": source.trace_id,
                    "density_veh_per_lane_km": density,
                    "environment_seed": environment_seed,
                    "requested_max_frames": frame_limit,
                    "frames": rollout.report.frames,
                    "trace_window": window.as_dict(),
                    "source_exhausted": rollout.report.source_exhausted,
                    "environment_transitions": rollout.report.transitions,
                    "rollout_transitions": prepared.rollout_transitions,
                    "learning_rows": prepared.batch.batch_size,
                    "episodes_completed": completed,
                    "internal_truncations": rollout.report.internal_truncations,
                    "normalization_total_training_rows": (
                        rollout.report.normalization_total_training_rows
                    ),
                    "matched_tape_fingerprint": rollout.report.matched_tape_fingerprint,
                    "fingerprint": rollout.report.fingerprint,
                }
            )
        if environment_transitions - environment_before_round != projected_round:
            raise JointDensityTrainingError(
                "projected and collected environment transition accounting diverged"
            )
        packets_in_round = accumulated_packets - packets_before_round
        if max_frames_per_trace is None:
            observed_packets_per_joint_frame = packets_in_round / (frame_limit - 1)
        balanced_rounds += 1

    prepared = merge_prepared_rollouts(tuple(parts))
    updates = optimize_ppo(
        updater=runtime.updater,
        batch=prepared.batch,
        update_epochs=config.training.update_epochs,
        minibatch_size=config.training.minibatch_size,
        generator=runtime.minibatch_generator,
    )
    dual_report = runtime.dual_ascent.update(
        densities_veh_per_lane_km=prepared.all_densities,
        costs=prepared.all_training_costs,
        miss_budget=curriculum.miss_budget,
    )
    counters = TrainingCounters(
        completed_iterations=runtime.counters.completed_iterations + 1,
        environment_transitions=(
            runtime.counters.environment_transitions + environment_transitions
        ),
        learning_transitions=(runtime.counters.learning_transitions + prepared.batch.batch_size),
        episodes_completed=runtime.counters.episodes_completed + episodes_completed,
        optimizer_steps=runtime.counters.optimizer_steps + len(updates),
    )
    metrics = build_training_iteration_metrics(
        config_hash=config_hash(config),
        policy_seed=policy_seed,
        iteration=runtime.counters.completed_iterations,
        environment_transitions=counters.environment_transitions,
        rollout_transitions=prepared.rollout_transitions,
        ppo_updates=updates,
        reward_predictions=prepared.reward_predictions,
        reward_targets=prepared.batch.reward_value_targets,
        cost_predictions=prepared.cost_predictions,
        cost_targets=prepared.batch.cost_value_targets,
        dual_report=dual_report,
        dual_snapshot=runtime.dual_ascent.snapshot(),
    )
    return _IterationOutcome(
        metrics=metrics,
        prepared=prepared,
        counters=counters,
        normalizer=ObservationNormalizer.from_state_dict(
            config,
            normalization_state.as_dict(),
        ),
        balanced_rounds=balanced_rounds,
        environment_transitions=environment_transitions,
        episodes_completed=episodes_completed,
        segment_reports=tuple(segment_reports),
        curriculum=curriculum,
        rollout_target=rollout_target,
        budget_limited=budget_limited,
    )


def _segment_environment_transitions(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    window: TraceWindowSelection,
    reader_cache: dict[Path, PopulationFrameReader],
    cache: dict[tuple[Path, int, int], int],
) -> int:
    source_path = source.path.resolve(strict=True)
    key = (source_path, window.start_frame_index, window.end_frame_index)
    cached = cache.get(key)
    if cached is not None:
        return cached
    reader = _population_reader(config, source, cache=reader_cache)
    transitions = sum(
        max(
            0,
            min(episode.last_frame, window.end_frame_index)
            - max(episode.first_frame, window.start_frame_index)
            + 1,
        )
        for episode in reader.episode_schedule
        if episode.first_frame <= window.end_frame_index
        and episode.last_frame >= window.start_frame_index
    )
    if transitions <= 0:
        raise JointDensityTrainingError(
            "a selected training trace window has no acted pair transitions",
            context={
                "trace_id": source.trace_id,
                "start_frame_index": window.start_frame_index,
                "end_frame_index": window.end_frame_index,
            },
        )
    cache[key] = transitions
    return transitions


def _population_reader(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    cache: dict[Path, PopulationFrameReader],
) -> PopulationFrameReader:
    source_path = source.path.resolve(strict=True)
    reader = cache.get(source_path)
    if reader is None:
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
        )
        cache[source_path] = reader
    return reader


def _iteration_payload(
    *,
    config: ProjectConfig,
    policy_seed: int,
    outcome: _IterationOutcome,
    checkpoint: TrainingCheckpointSummary,
    max_frames_per_trace: int | None,
) -> dict[str, object]:
    density_counts = Counter(float(value) for value in outcome.prepared.all_densities.tolist())
    configured_densities = tuple(sorted(density_counts))
    stage_end = outcome.curriculum.end_transition
    stage_overshoot = max(0, outcome.counters.environment_transitions - stage_end)
    return {
        "schema": MULTI_TRAINING_ITERATION_SCHEMA,
        "scope": "one immutable update in a resumable joint-density training run",
        "config_hash": config_hash(config),
        "policy_seed": policy_seed,
        "iteration": outcome.metrics.iteration,
        "device": "cpu",
        "trace_window_schedule": TRACE_WINDOW_SCHEDULE_SCHEMA,
        "curriculum": {
            **outcome.curriculum.as_dict(),
            "boundary_crossed": stage_overshoot > 0
            or outcome.counters.environment_transitions == stage_end,
            "boundary_overshoot_transitions": stage_overshoot,
        },
        "rollout_target_packets": outcome.rollout_target,
        "rollout_transitions": outcome.prepared.rollout_transitions,
        "rollout_overshoot_packets": max(
            0,
            outcome.prepared.rollout_transitions - outcome.rollout_target,
        ),
        "rollout_shortfall_packets": max(
            0,
            outcome.rollout_target - outcome.prepared.rollout_transitions,
        ),
        "learning_rows": outcome.prepared.batch.batch_size,
        "iteration_environment_transitions": outcome.environment_transitions,
        "cumulative_counters": outcome.counters.as_dict(),
        "balanced_rounds": outcome.balanced_rounds,
        "budget_limited": outcome.budget_limited,
        "max_frames_per_trace": max_frames_per_trace,
        "frame_scheduling": ("adaptive_3_to_20" if max_frames_per_trace is None else "fixed"),
        "density_rollout_transitions": {
            f"{density:g}": density_counts[density] for density in configured_densities
        },
        "action_counts": {
            action.label: outcome.prepared.action_counts[int(action)] for action in PolicyAction
        },
        "segments": list(outcome.segment_reports),
        "training_iteration": outcome.metrics.as_dict(),
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "size_bytes": checkpoint.size_bytes,
            "counters": checkpoint.counters.as_dict(),
        },
    }


def _prepare_run_root(path: str | Path, *, resume: bool) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    if destination.is_symlink():
        raise JointDensityTrainingError(
            "multi-iteration output root cannot be a symlink",
            artifact_path=destination,
        )
    if resume:
        if not destination.is_dir():
            raise JointDensityTrainingError(
                "resume output root must be an existing directory",
                artifact_path=destination,
            )
        for child in ("checkpoints", "iterations", "sessions"):
            candidate = destination / child
            if not candidate.is_dir() or candidate.is_symlink():
                raise JointDensityTrainingError(
                    "resume output root is missing a regular artifact directory",
                    artifact_path=candidate,
                )
        return destination
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise JointDensityTrainingError(
                "fresh multi-iteration output root must be new or empty",
                artifact_path=destination,
            )
    else:
        destination.mkdir(parents=True)
    for child in ("checkpoints", "iterations", "sessions"):
        (destination / child).mkdir()
    return destination


def _validate_existing_history(
    destination: Path,
    *,
    metrics_path: Path,
    counters: TrainingCounters,
    policy_seed: int,
    expected_config_hash: str,
    resume_checkpoint: Path,
) -> None:
    if counters.completed_iterations <= 0:
        raise JointDensityTrainingError("resume checkpoint has no completed iteration")
    if not metrics_path.is_file() or metrics_path.is_symlink():
        raise JointDensityTrainingError(
            "resume metrics history must be a regular file",
            artifact_path=metrics_path,
        )
    try:
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise JointDensityTrainingError(
            "resume metrics history is unreadable",
            artifact_path=metrics_path,
        ) from exc
    if len(rows) != counters.completed_iterations:
        raise JointDensityTrainingError("resume metric count does not match completed iterations")
    for index, row in enumerate(rows):
        row_counters = row.get("counters") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("iteration") != index
            or row.get("config_hash") != expected_config_hash
            or row.get("policy_seed") != policy_seed
            or not isinstance(row_counters, dict)
        ):
            raise JointDensityTrainingError("resume metrics history is not contiguous for this run")
    if rows[-1]["counters"].get("environment_transitions") != (counters.environment_transitions):
        raise JointDensityTrainingError(
            "resume metrics and checkpoint transition counters disagree"
        )

    expected_checkpoints = {
        f"checkpoint-iteration-{index:06d}.pt"
        for index in range(1, counters.completed_iterations + 1)
    }
    actual_checkpoints = {path.name for path in (destination / "checkpoints").iterdir()}
    expected_reports = {
        f"iteration-{index:06d}.json" for index in range(1, counters.completed_iterations + 1)
    }
    actual_reports = {path.name for path in (destination / "iterations").iterdir()}
    if actual_checkpoints != expected_checkpoints or actual_reports != expected_reports:
        raise JointDensityTrainingError(
            "resume artifact history is incomplete or contains a future branch"
        )
    if resume_checkpoint.name not in actual_checkpoints:
        raise JointDensityTrainingError("resume checkpoint is absent from artifact history")
    for completion in range(1, counters.completed_iterations + 1):
        report_path = destination / "iterations" / f"iteration-{completion:06d}.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise JointDensityTrainingError(
                "resume iteration history is unreadable",
                artifact_path=report_path,
            ) from exc
        segments = report.get("segments") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("schema") != MULTI_TRAINING_ITERATION_SCHEMA
            or report.get("iteration") != completion - 1
            or report.get("trace_window_schedule") != TRACE_WINDOW_SCHEDULE_SCHEMA
            or not isinstance(segments, list)
            or not segments
            or any(
                not isinstance(segment, dict)
                or not isinstance(segment.get("trace_window"), dict)
                or segment["trace_window"].get("schema") != TRACE_WINDOW_SCHEDULE_SCHEMA
                for segment in segments
            )
        ):
            raise JointDensityTrainingError(
                "resume iteration history predates or violates the trace-window contract",
                artifact_path=report_path,
            )


def _checkpoint_summary(
    path: Path,
    *,
    config: ProjectConfig,
    policy_seed: int,
    counters: TrainingCounters,
    sha256: str | None,
) -> TrainingCheckpointSummary:
    digest = sha256
    if digest is None:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    return TrainingCheckpointSummary(
        path=path,
        config_hash=config_hash(config),
        policy_seed=policy_seed,
        counters=counters,
        size_bytes=path.stat().st_size,
        sha256=digest,
    )


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise JointDensityTrainingError(
            "immutable training artifact already exists",
            artifact_path=path,
        )
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "MULTI_TRAINING_ITERATION_SCHEMA",
    "MULTI_TRAINING_SESSION_SCHEMA",
    "CurriculumPosition",
    "MultiIterationTrainingResult",
    "curriculum_position",
    "run_joint_density_training",
]
