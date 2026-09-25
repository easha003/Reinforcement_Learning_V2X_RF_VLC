"""One density-balanced primal-dual PPO iteration over training traces.

This is the first production-shaped Phase 8 trainer boundary.  It owns one
shared actor, reward critic, cost critic, normalization stream, and set of
per-density dual variables while collecting complete trace segments from all
configured training densities.  Collection stops only after a full balanced
round, so every update represents every density even when the packet target is
crossed partway through the round.

The driver intentionally performs one iteration.  Multi-iteration resume,
curriculum advancement, validation checkpoint selection, and five-seed
orchestration remain separate tasks; the immutable checkpoint produced here
contains the complete state needed by those extensions.
"""

from __future__ import annotations

import json
import math
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
    save_training_checkpoint,
)
from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.ppo import PPOBatch, PPOUpdater
from hybrid_v2x_rl.agents.trace_training import (
    PreparedRollout,
    TracePPOCollector,
    TraceTrainingError,
    optimize_ppo,
    prepare_rollout,
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
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationState,
    ObservationNormalizer,
)

JOINT_DENSITY_REPORT_SCHEMA: Final = "hybrid-rf-vlc-rl.joint-density-training.v1"
JOINT_DENSITY_POLICY_NAME: Final = "primal-dual-ppo-joint-density"
_ACTION_STREAM_XOR: Final = 0xA17C_D315_5EED_1001
_MINIBATCH_STREAM_XOR: Final = 0xB47C_D315_5EED_1002
_NUMPY_STREAM_XOR: Final = 0xC57C_D315_5EED_1003
_AUTO_INITIAL_FRAMES: Final = 3
_AUTO_MAX_FRAMES: Final = 20


class JointDensityTrainingError(TraceTrainingError):
    """A density-balanced training iteration violates its declared contract."""


@dataclass(frozen=True, slots=True)
class JointDensityTrainingResult:
    """Published artifacts from one validated joint-density update."""

    report_path: Path
    metrics_path: Path
    checkpoint: TrainingCheckpointSummary
    metrics: TrainingIterationMetrics
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        for name, path in (("report", self.report_path), ("metrics", self.metrics_path)):
            if not path.is_file() or path.is_symlink():
                raise JointDensityTrainingError(
                    f"published joint-density {name} must be a regular file",
                    artifact_path=path,
                )
        if not isinstance(self.checkpoint, TrainingCheckpointSummary):
            raise JointDensityTrainingError(
                "joint-density result requires a checkpoint summary"
            )
        if not isinstance(self.metrics, TrainingIterationMetrics):
            raise JointDensityTrainingError(
                "joint-density result requires training metrics"
            )
        if not isinstance(self.report, Mapping):
            raise JointDensityTrainingError("joint-density report must be a mapping")


def run_joint_density_training_iteration(
    config: ProjectConfig,
    sources: tuple[FrameTraceSource, ...],
    *,
    output_root: str | Path,
    policy_seed: int = 1001,
    rollout_packets: int | None = None,
    max_frames_per_trace: int | None = None,
) -> JointDensityTrainingResult:
    """Collect balanced trace rounds and perform one shared PPO/dual update.

    ``rollout_packets`` counts optimized packet transitions.  Because a shared
    policy update must retain every configured density, collection finishes the
    current density-balanced round after reaching the target.  The immutable
    report records the resulting nonnegative overshoot.

    ``max_frames_per_trace`` is a fixed development/test control.  ``None``
    enables adaptive bounds: a three-frame calibration round estimates aggregate
    packets per three-density frame, then later rounds choose between 2 and 20
    frames toward the remaining target.  Every segment's final acted frame is
    retained only as the next-value source, exactly as in the smoke runner.
    """

    target = _validate_request(
        config=config,
        sources=sources,
        policy_seed=policy_seed,
        rollout_packets=rollout_packets,
        max_frames_per_trace=max_frames_per_trace,
    )
    grouped = _sources_by_density(config, sources)
    destination = _prepare_output_root(output_root)

    random.seed(policy_seed)
    np.random.seed(policy_seed % (2**32))
    torch.manual_seed(policy_seed)
    numpy_generator = np.random.Generator(
        np.random.PCG64(policy_seed ^ _NUMPY_STREAM_XOR)
    )
    action_seed = int(numpy_generator.integers(0, 2**63)) ^ _ACTION_STREAM_XOR
    minibatch_seed = int(numpy_generator.integers(0, 2**63)) ^ _MINIBATCH_STREAM_XOR
    action_generator = torch.Generator().manual_seed(action_seed)
    minibatch_generator = torch.Generator().manual_seed(minibatch_seed)

    critic_builder = CentralizedCriticBuilder.from_config(config)
    updater = PPOUpdater.from_config(
        actor_observation_width=critic_builder.schema.actor_width,
        critic_observation_width=critic_builder.schema.critic_width,
        training=config.training,
    )
    dual_ascent = PerDensityDualAscent.from_config(config.training)
    normalization_state: ObservationNormalizationState | None = None
    parts: list[PreparedRollout] = []
    segment_reports: list[dict[str, object]] = []
    environment_transitions = 0
    episodes_completed = 0
    accumulated_packets = 0
    balanced_rounds = 0
    observed_packets_per_joint_frame: float | None = None

    while accumulated_packets < target:
        if max_frames_per_trace is None:
            frame_limit = _adaptive_frame_limit(
                target=target,
                accumulated=accumulated_packets,
                observed_packets_per_joint_frame=observed_packets_per_joint_frame,
            )
        else:
            frame_limit = max_frames_per_trace
        packets_before_round = accumulated_packets
        for density in sorted(grouped):
            candidates = grouped[density]
            source = candidates[balanced_rounds % len(candidates)]
            environment_seed = int(numpy_generator.integers(0, 2**63))
            collector = TracePPOCollector(
                config=config,
                updater=updater,
                action_generator=action_generator,
                policy_name=JOINT_DENSITY_POLICY_NAME,
            )
            rollout = run_policy_rollout_with_state(
                config,
                source,
                policy=collector,
                environment_seed=environment_seed,
                policy_seed=policy_seed,
                max_frames=frame_limit,
                normalization_state=normalization_state,
                frame_observer=collector,
            )
            frames = collector.frames
            if len(frames) < 2:
                raise JointDensityTrainingError(
                    "a joint-density trace segment needs at least two acted frames",
                    context={"trace_id": source.trace_id, "frames": len(frames)},
                )
            prepared = prepare_rollout(
                frames=frames,
                config=config,
                updater=updater,
                dual_ascent=dual_ascent,
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
        packets_in_round = accumulated_packets - packets_before_round
        if max_frames_per_trace is None:
            observed_packets_per_joint_frame = packets_in_round / (frame_limit - 1)
        balanced_rounds += 1

    prepared = merge_prepared_rollouts(tuple(parts))
    if prepared.rollout_transitions != accumulated_packets:
        raise JointDensityTrainingError(
            "merged rollout transitions do not match accumulated segments"
        )
    if environment_transitions > config.training.total_transitions_per_seed:
        raise JointDensityTrainingError(
            "one joint-density iteration exceeds the configured seed budget",
            context={
                "environment_transitions": environment_transitions,
                "configured_budget": config.training.total_transitions_per_seed,
            },
        )

    updates = optimize_ppo(
        updater=updater,
        batch=prepared.batch,
        update_epochs=config.training.update_epochs,
        minibatch_size=config.training.minibatch_size,
        generator=minibatch_generator,
    )
    miss_budget = config.training.curriculum[0].miss_budget
    dual_report = dual_ascent.update(
        densities_veh_per_lane_km=prepared.all_densities,
        costs=prepared.all_training_costs,
        miss_budget=miss_budget,
    )
    metrics = build_training_iteration_metrics(
        config_hash=config_hash(config),
        policy_seed=policy_seed,
        iteration=0,
        environment_transitions=environment_transitions,
        rollout_transitions=prepared.rollout_transitions,
        ppo_updates=updates,
        reward_predictions=prepared.reward_predictions,
        reward_targets=prepared.batch.reward_value_targets,
        cost_predictions=prepared.cost_predictions,
        cost_targets=prepared.batch.cost_value_targets,
        dual_report=dual_report,
        dual_snapshot=dual_ascent.snapshot(),
    )

    metrics_path = destination / "metrics.jsonl"
    TrainingMetricsJSONL(metrics_path).append(metrics)
    counters = TrainingCounters(
        completed_iterations=1,
        environment_transitions=environment_transitions,
        learning_transitions=prepared.batch.batch_size,
        episodes_completed=episodes_completed,
        optimizer_steps=len(updates),
    )
    if normalization_state is None:  # pragma: no cover - at least one segment is required.
        raise JointDensityTrainingError("joint-density normalization state is absent")
    normalizer = ObservationNormalizer.from_state_dict(
        config,
        normalization_state.as_dict(),
    )
    checkpoint = save_training_checkpoint(
        destination / "checkpoint-iteration-000001.pt",
        config=config,
        policy_seed=policy_seed,
        updater=updater,
        dual_ascent=dual_ascent,
        normalizer=normalizer,
        counters=counters,
        numpy_generators={"training_streams": numpy_generator},
        torch_generators={
            "policy_actions": action_generator,
            "ppo_minibatches": minibatch_generator,
        },
    )
    report_payload = _report_payload(
        config=config,
        policy_seed=policy_seed,
        target=target,
        balanced_rounds=balanced_rounds,
        max_frames_per_trace=max_frames_per_trace,
        prepared=prepared,
        environment_transitions=environment_transitions,
        segment_reports=tuple(segment_reports),
        metrics=metrics,
        checkpoint=checkpoint,
    )
    report_path = destination / "report.json"
    report_path.write_text(
        json.dumps(report_payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return JointDensityTrainingResult(
        report_path=report_path,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        metrics=metrics,
        report=MappingProxyType(report_payload),
    )


def merge_prepared_rollouts(parts: tuple[PreparedRollout, ...]) -> PreparedRollout:
    """Concatenate independently estimated trace segments into one PPO update."""

    if not isinstance(parts, tuple) or not parts:
        raise JointDensityTrainingError(
            "joint-density accumulation requires at least one prepared segment"
        )
    if any(not isinstance(part, PreparedRollout) for part in parts):
        raise JointDensityTrainingError(
            "joint-density accumulation received an invalid prepared segment"
        )
    batch = PPOBatch(
        actor_observations=torch.cat(
            [part.batch.actor_observations for part in parts]
        ),
        critic_observations=torch.cat(
            [part.batch.critic_observations for part in parts]
        ),
        action_masks=torch.cat([part.batch.action_masks for part in parts]),
        actions=torch.cat([part.batch.actions for part in parts]),
        old_log_probabilities=torch.cat(
            [part.batch.old_log_probabilities for part in parts]
        ),
        reward_advantages=torch.cat(
            [part.batch.reward_advantages for part in parts]
        ),
        cost_advantages=torch.cat([part.batch.cost_advantages for part in parts]),
        reward_value_targets=torch.cat(
            [part.batch.reward_value_targets for part in parts]
        ),
        cost_value_targets=torch.cat(
            [part.batch.cost_value_targets for part in parts]
        ),
        cost_penalty_weights=torch.cat(
            [part.batch.cost_penalty_weights for part in parts]
        ),
    )
    action_counts = tuple(
        sum(part.action_counts[index] for part in parts)
        for index in range(len(PolicyAction))
    )
    rollout_transitions = sum(part.rollout_transitions for part in parts)
    if rollout_transitions != sum(action_counts):
        raise JointDensityTrainingError(
            "merged action counts do not partition rollout transitions"
        )
    return PreparedRollout(
        batch=batch,
        reward_predictions=torch.cat(
            [part.reward_predictions for part in parts]
        ),
        cost_predictions=torch.cat([part.cost_predictions for part in parts]),
        all_rewards=torch.cat([part.all_rewards for part in parts]),
        all_training_costs=torch.cat(
            [part.all_training_costs for part in parts]
        ),
        all_sampled_miss_costs=torch.cat(
            [part.all_sampled_miss_costs for part in parts]
        ),
        all_densities=torch.cat([part.all_densities for part in parts]),
        action_counts=action_counts,
        rollout_transitions=rollout_transitions,
    )


def _validate_request(
    *,
    config: ProjectConfig,
    sources: tuple[FrameTraceSource, ...],
    policy_seed: int,
    rollout_packets: int | None,
    max_frames_per_trace: int | None,
) -> int:
    if not isinstance(config, ProjectConfig):
        raise JointDensityTrainingError(
            "joint-density training requires a ProjectConfig"
        )
    if not isinstance(sources, tuple) or not sources:
        raise JointDensityTrainingError(
            "joint-density training requires configured training sources"
        )
    if any(not isinstance(source, FrameTraceSource) for source in sources):
        raise JointDensityTrainingError("joint-density sources are invalid")
    if len({source.trace_id for source in sources}) != len(sources):
        raise JointDensityTrainingError("joint-density source IDs must be unique")
    configured_ids = set(config.environment.splits.train)
    unexpected = tuple(
        source.trace_id
        for source in sources
        if source.split != "train" or source.trace_id not in configured_ids
    )
    if unexpected:
        raise JointDensityTrainingError(
            "joint-density sources must belong to the configured training split",
            context={"trace_ids": unexpected},
        )
    if policy_seed not in config.training.policy_seeds:
        raise JointDensityTrainingError(
            "joint-density policy seed is not declared by configuration"
        )
    target = config.training.rollout_packets if rollout_packets is None else rollout_packets
    if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
        raise JointDensityTrainingError("rollout_packets must be a positive integer")
    if target > config.training.total_transitions_per_seed:
        raise JointDensityTrainingError(
            "rollout_packets cannot exceed the configured seed budget"
        )
    if max_frames_per_trace is not None and (
        not isinstance(max_frames_per_trace, int)
        or isinstance(max_frames_per_trace, bool)
        or max_frames_per_trace < 2
    ):
        raise JointDensityTrainingError(
            "max_frames_per_trace must be an integer of at least two or None"
        )
    return target


def _sources_by_density(
    config: ProjectConfig,
    sources: tuple[FrameTraceSource, ...],
) -> dict[float, tuple[FrameTraceSource, ...]]:
    configured = tuple(
        sorted(
            multiplier.density_veh_per_lane_km
            for multiplier in config.training.density_multipliers
        )
    )
    groups = {
        density: tuple(
            sorted(
                (source for source in sources if source.density == density),
                key=lambda source: (source.replicate, source.trace_id),
            )
        )
        for density in configured
    }
    missing = tuple(density for density, candidates in groups.items() if not candidates)
    supplied_densities = tuple(sorted({source.density for source in sources}))
    unexpected = tuple(
        density for density in supplied_densities if density not in set(configured)
    )
    if missing or unexpected:
        raise JointDensityTrainingError(
            "joint-density sources must represent every and only configured density",
            context={"missing_densities": missing, "unexpected_densities": unexpected},
        )
    return groups


def _adaptive_frame_limit(
    *,
    target: int,
    accumulated: int,
    observed_packets_per_joint_frame: float | None,
) -> int:
    """Choose a bounded next round from observed aggregate population size."""

    if observed_packets_per_joint_frame is None:
        return _AUTO_INITIAL_FRAMES
    if (
        not math.isfinite(observed_packets_per_joint_frame)
        or observed_packets_per_joint_frame <= 0.0
    ):
        raise JointDensityTrainingError(
            "adaptive joint-density packet rate must be finite and positive"
        )
    remaining = max(1, target - accumulated)
    optimized_frames = math.ceil(remaining / observed_packets_per_joint_frame)
    return min(
        _AUTO_MAX_FRAMES,
        max(_AUTO_INITIAL_FRAMES, optimized_frames + 1),
    )


def _prepare_output_root(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    if destination.is_symlink():
        raise JointDensityTrainingError(
            "joint-density output root cannot be a symlink",
            artifact_path=destination,
        )
    if destination.exists():
        if not destination.is_dir():
            raise JointDensityTrainingError(
                "joint-density output root must be a directory",
                artifact_path=destination,
            )
        if any(destination.iterdir()):
            raise JointDensityTrainingError(
                "joint-density output root must be empty",
                artifact_path=destination,
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _report_payload(
    *,
    config: ProjectConfig,
    policy_seed: int,
    target: int,
    balanced_rounds: int,
    max_frames_per_trace: int | None,
    prepared: PreparedRollout,
    environment_transitions: int,
    segment_reports: tuple[dict[str, object], ...],
    metrics: TrainingIterationMetrics,
    checkpoint: TrainingCheckpointSummary,
) -> dict[str, object]:
    density_counts = Counter(float(value) for value in prepared.all_densities.tolist())
    configured_densities = tuple(
        sorted(
            multiplier.density_veh_per_lane_km
            for multiplier in config.training.density_multipliers
        )
    )
    if tuple(sorted(density_counts)) != configured_densities:
        raise JointDensityTrainingError(
            "joint-density accumulated transitions do not cover every density"
        )
    return {
        "schema": JOINT_DENSITY_REPORT_SCHEMA,
        "scope": "one density-balanced training iteration; not convergence evidence",
        "config_hash": config_hash(config),
        "policy_seed": policy_seed,
        "iteration": 0,
        "device": "cpu",
        "configured_densities_veh_per_lane_km": list(configured_densities),
        "rollout_target_packets": target,
        "rollout_transitions": prepared.rollout_transitions,
        "rollout_overshoot_packets": prepared.rollout_transitions - target,
        "learning_rows": prepared.batch.batch_size,
        "environment_transitions": environment_transitions,
        "balanced_rounds": balanced_rounds,
        "max_frames_per_trace": max_frames_per_trace,
        "frame_scheduling": (
            "adaptive_3_to_20" if max_frames_per_trace is None else "fixed"
        ),
        "density_rollout_transitions": {
            f"{density:g}": density_counts[density]
            for density in configured_densities
        },
        "action_counts": {
            action.label: prepared.action_counts[int(action)] for action in PolicyAction
        },
        "segments": list(segment_reports),
        "training_iteration": metrics.as_dict(),
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "size_bytes": checkpoint.size_bytes,
            "counters": checkpoint.counters.as_dict(),
        },
    }


__all__ = [
    "JOINT_DENSITY_POLICY_NAME",
    "JOINT_DENSITY_REPORT_SCHEMA",
    "JointDensityTrainingError",
    "JointDensityTrainingResult",
    "merge_prepared_rollouts",
    "run_joint_density_training_iteration",
]
