"""One-iteration PPO smoke training on the authoritative trace rollout path.

The smoke boundary is intentionally smaller than the future full trainer.  It
collects several consecutive frames from one declared training trace, reserves
the final collected frame as the value-bootstrap source, performs one complete
primal-dual PPO iteration, and publishes metrics plus a complete checkpoint.
Internal time-limit truncations use separately materialized next-physical
observations for one-step critic bootstrapping without continuing GAE across
the episode reset.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Final, Protocol

import numpy as np
import torch

from hybrid_v2x_rl.agents.advantages import (
    reward_cost_generalized_advantage_estimate,
)
from hybrid_v2x_rl.agents.checkpointing import (
    TrainingCheckpointSummary,
    TrainingCounters,
    save_training_checkpoint,
)
from hybrid_v2x_rl.agents.cost_signal import reliability_costs_from_config
from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.lifecycle_bootstrap import (
    PairedCriticValues,
    assemble_lifecycle_bootstrap,
)
from hybrid_v2x_rl.agents.ppo import PPOBatch, PPOUpdateMetrics, PPOUpdater
from hybrid_v2x_rl.agents.training_metrics import (
    TrainingIterationMetrics,
    TrainingMetricsJSONL,
    build_training_iteration_metrics,
)
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.critic_observations import CentralizedCriticBuilder
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    PopulationRolloutObserver,
    run_policy_rollout_with_state,
)
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation, FrameStepOutput
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizer
from hybrid_v2x_rl.mean_field.packet_outcomes import FramePacketOutcomes
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.return_boundaries import FrameReturnBoundary

TRACE_SMOKE_REPORT_SCHEMA: Final = "hybrid-rf-vlc-rl.trace-smoke-training.v1"
TRACE_SMOKE_POLICY_NAME: Final = "primal-dual-ppo-smoke"
TRACE_TRAINING_STAGE_NAMES: Final = (
    "setup",
    "rollout",
    "rollout_preparation",
    "ppo_optimization",
    "metrics_and_dual",
    "artifact_publication",
)
_ACTION_STREAM_XOR: Final = 0xA17C_10A5_5EED_0001
_MINIBATCH_STREAM_XOR: Final = 0xB47C_10A5_5EED_0002
_NUMPY_STREAM_XOR: Final = 0xC57C_10A5_5EED_0003
_ENVIRONMENT_REPORT_FIELDS: Final = (
    "available_frames",
    "frames",
    "source_exhausted",
    "transitions",
    "usable_transitions",
    "fallback_transitions",
    "births",
    "natural_terminations",
    "internal_truncations",
    "trace_end_truncations",
    "misses",
    "reward_sum",
    "conditional_risk_sum",
    "reserved_rf_attempts",
    "vlc_activations",
    "max_population",
    "normalization_training_rows",
    "normalization_total_training_rows",
    "normalization_frozen",
    "matched_tape_fingerprint",
    "fingerprint",
)


class TraceTrainingError(HybridV2XError):
    """A trace-backed smoke-training request or rollout is invalid."""


class TraceTrainingTimingObserver(Protocol):
    """Receive non-overlapping wall-clock measurements from one training run."""

    def record_stage(self, *, name: str, elapsed_seconds: float) -> None: ...


@dataclass(frozen=True, slots=True)
class TraceSmokeTrainingResult:
    """Published artifacts and validated in-memory result for one smoke run."""

    report_path: Path
    metrics_path: Path
    checkpoint: TrainingCheckpointSummary
    metrics: TrainingIterationMetrics
    report: Mapping[str, object]

    def __post_init__(self) -> None:
        for name, path in (
            ("report", self.report_path),
            ("metrics", self.metrics_path),
        ):
            if not path.is_file() or path.is_symlink():
                raise TraceTrainingError(
                    f"published {name} artifact must be a regular file",
                    artifact_path=path,
                )
        if not isinstance(self.checkpoint, TrainingCheckpointSummary):
            raise TraceTrainingError("smoke result requires a checkpoint summary")
        if not isinstance(self.metrics, TrainingIterationMetrics):
            raise TraceTrainingError("smoke result requires training metrics")
        if not isinstance(self.report, Mapping):
            raise TraceTrainingError("smoke result report must be a mapping")


@dataclass(frozen=True, slots=True)
class _PolicyFrameSample:
    observation: FrameObservation
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    reward_values: torch.Tensor
    cost_values: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ObservedFrameSample:
    policy: _PolicyFrameSample
    outcomes: FramePacketOutcomes
    boundary: FrameReturnBoundary
    final_observation: Mapping[str, FrameObservation]


@dataclass(frozen=True, slots=True)
class _LearningRow:
    order: int
    frame_index: int
    pair_id: str
    actor_observation: torch.Tensor
    critic_observation: torch.Tensor
    action_mask: torch.Tensor
    action: torch.Tensor
    old_log_probability: torch.Tensor
    reward: torch.Tensor
    cost: torch.Tensor
    reward_value: torch.Tensor
    cost_value: torch.Tensor
    reward_next_value: torch.Tensor
    cost_next_value: torch.Tensor
    value_bootstrap: torch.Tensor
    gae_continuation: torch.Tensor
    density: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PreparedRollout:
    batch: PPOBatch
    reward_predictions: torch.Tensor
    cost_predictions: torch.Tensor
    all_rewards: torch.Tensor
    all_training_costs: torch.Tensor
    all_sampled_miss_costs: torch.Tensor
    all_densities: torch.Tensor
    action_counts: tuple[int, ...]
    rollout_transitions: int


class _TracePPOCollector(PopulationRolloutObserver):
    """Sample one shared policy and retain completed, pair-aligned frames."""

    name = TRACE_SMOKE_POLICY_NAME
    requires_oracle_truth = False

    def __init__(
        self,
        *,
        config: ProjectConfig,
        updater: PPOUpdater,
        action_generator: torch.Generator,
    ) -> None:
        self._config = config
        self._updater = updater
        self._action_generator = action_generator
        self._critic_builder = CentralizedCriticBuilder.from_config(config)
        self._pending: dict[int, _PolicyFrameSample] = {}
        self._frames: list[_ObservedFrameSample] = []

    @property
    def frames(self) -> tuple[_ObservedFrameSample, ...]:
        if self._pending:
            raise TraceTrainingError("policy samples and completed rollout frames do not reconcile")
        return tuple(self._frames)

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyAction | None, ...]:
        if channel_truth is not None:
            raise TraceTrainingError("the trainable policy cannot receive oracle truth")
        frame_index = decision.frame.index
        if frame_index in self._pending:
            raise TraceTrainingError("a policy frame was sampled more than once")

        observation = decision.observation
        actor = torch.tensor(
            observation.actor_observations,
            dtype=torch.float32,
        )
        critic_frame = self._critic_builder.build(decision.frame, observation)
        critic = torch.tensor(
            critic_frame.critic_observations,
            dtype=torch.float32,
        )
        masks = torch.tensor(observation.action_masks, dtype=torch.bool)
        usable = torch.tensor(decision.actor_frame.usable_mask, dtype=torch.bool)
        population = observation.population_size
        fallback = PolicyAction.DUP_4
        actions = torch.full((population,), int(fallback), dtype=torch.long)
        log_probabilities = torch.zeros(population, dtype=torch.float32)

        proposals: list[PolicyAction | None] = [None] * population
        with torch.no_grad():
            usable_indices = torch.nonzero(usable, as_tuple=False).flatten()
            if usable_indices.numel():
                selected = self._updater.actor.select(
                    actor[usable_indices],
                    masks[usable_indices],
                    generator=self._action_generator,
                )
                actions[usable_indices] = selected.actions
                log_probabilities[usable_indices] = selected.log_probabilities
                for row, action_index in zip(
                    usable_indices.tolist(),
                    selected.actions.tolist(),
                    strict=True,
                ):
                    proposals[int(row)] = PolicyAction(int(action_index))
            reward_values = self._updater.reward_critic(critic)
            cost_values = self._updater.cost_critic(critic)

        self._pending[frame_index] = _PolicyFrameSample(
            observation=observation,
            actor_observations=actor.detach(),
            critic_observations=critic.detach(),
            action_masks=masks.detach(),
            actions=actions.detach(),
            old_log_probabilities=log_probabilities.detach(),
            reward_values=reward_values.detach(),
            cost_values=cost_values.detach(),
        )
        return tuple(proposals)

    def observe_frame(
        self,
        *,
        decision: PopulationPolicyFrame,
        actions: tuple[PolicyAction, ...],
        outcomes: FramePacketOutcomes,
        boundary: FrameReturnBoundary,
        final_observation: Mapping[str, FrameObservation],
    ) -> None:
        frame_index = decision.frame.index
        try:
            sample = self._pending.pop(frame_index)
        except KeyError as error:
            raise TraceTrainingError(
                "a completed frame has no matching policy sample",
                context={"frame_index": frame_index},
            ) from error
        actual_actions = torch.tensor(tuple(int(action) for action in actions), dtype=torch.long)
        if not torch.equal(actual_actions, sample.actions):
            raise TraceTrainingError("rollout actions differ from sampled/fallback actions")
        if (
            outcomes.pair_ids != sample.observation.pair_ids
            or boundary.pair_ids != sample.observation.pair_ids
        ):
            raise TraceTrainingError("completed rollout rows do not align with policy rows")
        expected_final_ids = boundary.bootstrap_pair_ids
        if tuple(final_observation) != expected_final_ids:
            raise TraceTrainingError(
                "completed final observations do not cover bootstrap pairs exactly",
                context={
                    "actual": tuple(final_observation),
                    "expected": expected_final_ids,
                },
            )
        for pair_id, observation in final_observation.items():
            if (
                not isinstance(observation, FrameObservation)
                or observation.pair_ids != (pair_id,)
                or observation.trace_id != sample.observation.trace_id
                or observation.frame_index != sample.observation.frame_index + 1
                or observation.time_s <= sample.observation.time_s
            ):
                raise TraceTrainingError(
                    "a final observation must be one old-ID row at the next physical instant",
                    context={"pair_id": pair_id},
                )
        self._frames.append(
            _ObservedFrameSample(
                policy=sample,
                outcomes=outcomes,
                boundary=boundary,
                final_observation=MappingProxyType(dict(final_observation)),
            )
        )


def run_trace_smoke_training(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    output_root: str | Path,
    policy_seed: int = 1001,
    environment_seed: int | None = None,
    max_frames: int = 5,
    timing_observer: TraceTrainingTimingObserver | None = None,
) -> TraceSmokeTrainingResult:
    """Run one real-trace primal-dual PPO iteration and publish its artifacts.

    At least two frames are required.  The first ``max_frames - 1`` frames are
    optimized; the final frame is acted and retained only as the authoritative
    next-value source.  This is a smoke/integration run, not a convergence or
    feasibility experiment.
    """

    _validate_smoke_request(
        config=config,
        source=source,
        policy_seed=policy_seed,
        max_frames=max_frames,
    )
    if timing_observer is not None and not callable(getattr(timing_observer, "record_stage", None)):
        raise TraceTrainingError("timing_observer must provide a record_stage method")

    stage_started = perf_counter()
    destination = _prepare_output_root(output_root)
    random.seed(policy_seed)
    np.random.seed(policy_seed % (2**32))
    torch.manual_seed(policy_seed)
    numpy_generator = np.random.Generator(np.random.PCG64(policy_seed ^ _NUMPY_STREAM_XOR))
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
    collector = _TracePPOCollector(
        config=config,
        updater=updater,
        action_generator=action_generator,
    )
    _record_stage(timing_observer, "setup", stage_started)

    stage_started = perf_counter()
    rollout_result = run_policy_rollout_with_state(
        config,
        source,
        policy=collector,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        max_frames=max_frames,
        frame_observer=collector,
    )
    _record_stage(timing_observer, "rollout", stage_started)
    frames = collector.frames
    if len(frames) != max_frames:
        raise TraceTrainingError(
            "trace ended before the requested smoke frame count",
            context={"actual": len(frames), "requested": max_frames},
        )

    miss_budget = config.training.curriculum[0].miss_budget
    stage_started = perf_counter()
    prepared = _prepare_rollout(
        frames=frames,
        config=config,
        updater=updater,
        dual_ascent=dual_ascent,
        density_veh_per_lane_km=source.density,
    )
    _record_stage(timing_observer, "rollout_preparation", stage_started)

    stage_started = perf_counter()
    update_metrics = _optimize(
        updater=updater,
        batch=prepared.batch,
        update_epochs=config.training.update_epochs,
        minibatch_size=config.training.minibatch_size,
        generator=minibatch_generator,
    )
    _record_stage(timing_observer, "ppo_optimization", stage_started)

    stage_started = perf_counter()
    dual_report = dual_ascent.update(
        densities_veh_per_lane_km=prepared.all_densities,
        costs=prepared.all_training_costs,
        miss_budget=miss_budget,
    )
    metrics = build_training_iteration_metrics(
        config_hash=config_hash(config),
        policy_seed=policy_seed,
        iteration=0,
        environment_transitions=rollout_result.report.transitions,
        rollout_transitions=prepared.rollout_transitions,
        ppo_updates=update_metrics,
        reward_predictions=prepared.reward_predictions,
        reward_targets=prepared.batch.reward_value_targets,
        cost_predictions=prepared.cost_predictions,
        cost_targets=prepared.batch.cost_value_targets,
        dual_report=dual_report,
        dual_snapshot=dual_ascent.snapshot(),
    )
    _record_stage(timing_observer, "metrics_and_dual", stage_started)

    stage_started = perf_counter()
    metrics_path = destination / "metrics.jsonl"
    TrainingMetricsJSONL(metrics_path).append(metrics)
    counters = TrainingCounters(
        completed_iterations=1,
        environment_transitions=rollout_result.report.transitions,
        learning_transitions=prepared.batch.batch_size,
        episodes_completed=(
            rollout_result.report.natural_terminations
            + rollout_result.report.internal_truncations
            + rollout_result.report.trace_end_truncations
        ),
        optimizer_steps=len(update_metrics),
    )
    normalizer = ObservationNormalizer.from_state_dict(
        config,
        rollout_result.normalization_state.as_dict(),
    )
    checkpoint = save_training_checkpoint(
        destination / "checkpoint.pt",
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
        source=source,
        policy_seed=policy_seed,
        environment_seed=(
            config.training.root_seed if environment_seed is None else environment_seed
        ),
        max_frames=max_frames,
        rollout_result=rollout_result.report.as_dict(),
        prepared=prepared,
        metrics=metrics,
        checkpoint=checkpoint,
    )
    report_path = destination / "report.json"
    report_path.write_text(
        json.dumps(report_payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _record_stage(timing_observer, "artifact_publication", stage_started)
    return TraceSmokeTrainingResult(
        report_path=report_path,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        metrics=metrics,
        report=MappingProxyType(report_payload),
    )


def _record_stage(
    observer: TraceTrainingTimingObserver | None,
    name: str,
    started: float,
) -> None:
    if observer is None:
        return
    observer.record_stage(name=name, elapsed_seconds=perf_counter() - started)


def _validate_smoke_request(
    *,
    config: ProjectConfig,
    source: FrameTraceSource,
    policy_seed: int,
    max_frames: int,
) -> None:
    if not isinstance(config, ProjectConfig):
        raise TraceTrainingError("trace smoke training requires a ProjectConfig")
    if not isinstance(source, FrameTraceSource):
        raise TraceTrainingError("trace smoke training requires a FrameTraceSource")
    if source.split != "train" or source.trace_id not in config.environment.splits.train:
        raise TraceTrainingError("smoke training requires a configured training trace")
    configured_densities = tuple(
        row.density_veh_per_lane_km for row in config.training.density_multipliers
    )
    if source.density not in configured_densities:
        raise TraceTrainingError(
            "smoke trace density has no configured dual multiplier",
            context={"density": source.density},
        )
    if policy_seed not in config.training.policy_seeds:
        raise TraceTrainingError("smoke policy seed is not declared by the training configuration")
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 2:
        raise TraceTrainingError("max_frames must be an integer of at least two")


def _prepare_output_root(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve(strict=False)
    if destination.is_symlink():
        raise TraceTrainingError(
            "smoke output root cannot be a symlink",
            artifact_path=destination,
        )
    if destination.exists():
        if not destination.is_dir():
            raise TraceTrainingError(
                "smoke output root must be a directory",
                artifact_path=destination,
            )
        if any(destination.iterdir()):
            raise TraceTrainingError(
                "smoke output root must be empty to preserve immutable evidence",
                artifact_path=destination,
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _final_observation_values(
    *,
    current: _ObservedFrameSample,
    following: _ObservedFrameSample,
    updater: PPOUpdater,
) -> PairedCriticValues | None:
    """Evaluate old-ID final rows with the next population's global context."""

    pair_ids = current.boundary.bootstrap_pair_ids
    if not pair_ids:
        if current.final_observation:
            raise TraceTrainingError(
                "a frame without bootstrap-valid truncations has final observations"
            )
        return None
    if tuple(current.final_observation) != pair_ids:
        raise TraceTrainingError(
            "final observations do not use canonical bootstrap-pair order"
        )
    next_observation = following.policy.observation
    if following.policy.critic_observations.shape[0] == 0:
        raise TraceTrainingError(
            "internal truncation bootstrapping requires a non-empty next population"
        )

    final_rows: list[torch.Tensor] = []
    for pair_id in pair_ids:
        observation = current.final_observation[pair_id]
        if (
            observation.trace_id != next_observation.trace_id
            or observation.frame_index != next_observation.frame_index
            or not math.isclose(
                observation.time_s,
                next_observation.time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or observation.pair_ids != (pair_id,)
        ):
            raise TraceTrainingError(
                "final and ordinary next observations do not share one physical frame",
                context={"pair_id": pair_id},
            )
        final_rows.append(
            torch.tensor(observation.actor_observations[0], dtype=torch.float32)
        )

    actor = torch.stack(final_rows)
    actor_width = actor.shape[1]
    critic_width = following.policy.critic_observations.shape[1]
    if critic_width <= actor_width:
        raise TraceTrainingError(
            "next critic observations are missing their population-global suffix"
        )
    global_suffix = following.policy.critic_observations[0, actor_width:]
    final_critic = torch.cat(
        (actor, global_suffix.unsqueeze(0).expand(len(pair_ids), -1)),
        dim=1,
    )
    with torch.no_grad():
        reward = updater.reward_critic(final_critic).detach()
        cost = updater.cost_critic(final_critic).detach()
    return PairedCriticValues(pair_ids=pair_ids, reward=reward, cost=cost)


def _prepare_rollout(
    *,
    frames: tuple[_ObservedFrameSample, ...],
    config: ProjectConfig,
    updater: PPOUpdater,
    dual_ascent: PerDensityDualAscent,
    density_veh_per_lane_km: float,
) -> _PreparedRollout:
    optimized = frames[:-1]
    if not optimized:
        raise TraceTrainingError("smoke rollout contains no optimization frames")
    rows: list[_LearningRow] = []
    all_rewards: list[torch.Tensor] = []
    all_costs: list[torch.Tensor] = []
    all_sampled: list[torch.Tensor] = []
    all_densities: list[torch.Tensor] = []
    action_counts: Counter[int] = Counter()
    order = 0

    for current, following in zip(optimized, frames[1:], strict=True):
        boundary_info = dict(
            current.boundary.as_step_info(
                final_observation=current.final_observation,
            )
        )
        outcome_info = dict(current.outcomes.as_step_info())
        step = FrameStepOutput(
            next_observation=following.policy.observation,
            transition_pair_ids=current.policy.observation.pair_ids,
            rewards=current.outcomes.rewards,
            terminated=current.boundary.terminated,
            truncated=current.boundary.truncated,
            bootstrap_valid=current.boundary.bootstrap_valid,
            learn_mask=current.boundary.learn_mask,
            info={**outcome_info, **boundary_info},
        )
        final_values = _final_observation_values(
            current=current,
            following=following,
            updater=updater,
        )
        lifecycle = assemble_lifecycle_bootstrap(
            step_output=step,
            current_values=PairedCriticValues(
                pair_ids=current.policy.observation.pair_ids,
                reward=current.policy.reward_values,
                cost=current.policy.cost_values,
            ),
            next_population_values=PairedCriticValues(
                pair_ids=following.policy.observation.pair_ids,
                reward=following.policy.reward_values,
                cost=following.policy.cost_values,
            ),
            final_observation_values=final_values,
        )
        costs = reliability_costs_from_config(
            sampled_miss_costs=torch.tensor(
                current.outcomes.sampled_miss_costs,
                dtype=torch.float32,
            ),
            conditional_miss_probabilities=torch.tensor(
                current.outcomes.conditional_miss_probabilities,
                dtype=torch.float32,
            ),
            training=config.training,
        )
        rewards = torch.tensor(current.outcomes.rewards, dtype=torch.float32)
        density = torch.full(
            (current.outcomes.population_size,),
            density_veh_per_lane_km,
            dtype=torch.float32,
        )
        all_rewards.append(rewards)
        all_costs.append(costs.training_costs)
        all_sampled.append(costs.sampled_miss_costs)
        all_densities.append(density)
        action_counts.update(int(value) for value in current.policy.actions.tolist())

        for row, pair_id in enumerate(current.policy.observation.pair_ids):
            if not bool(lifecycle.learn_mask[row].item()):
                continue
            rows.append(
                _LearningRow(
                    order=order,
                    frame_index=current.policy.observation.frame_index,
                    pair_id=pair_id,
                    actor_observation=current.policy.actor_observations[row],
                    critic_observation=current.policy.critic_observations[row],
                    action_mask=current.policy.action_masks[row],
                    action=current.policy.actions[row],
                    old_log_probability=current.policy.old_log_probabilities[row],
                    reward=rewards[row],
                    cost=costs.training_costs[row],
                    reward_value=lifecycle.reward_values[row],
                    cost_value=lifecycle.cost_values[row],
                    reward_next_value=lifecycle.reward_next_values[row],
                    cost_next_value=lifecycle.cost_next_values[row],
                    value_bootstrap=lifecycle.value_bootstrap_mask[row],
                    gae_continuation=lifecycle.gae_continuation_mask[row],
                    density=density[row],
                )
            )
            order += 1

    if not rows:
        raise TraceTrainingError("smoke rollout produced no learning-eligible rows")
    advantage_by_order = _advantages_by_row(rows, config=config)
    ordered = sorted(rows, key=lambda row: row.order)
    actor_observations = torch.stack([row.actor_observation for row in ordered])
    critic_observations = torch.stack([row.critic_observation for row in ordered])
    action_masks = torch.stack([row.action_mask for row in ordered])
    actions = torch.stack([row.action for row in ordered])
    old_log_probabilities = torch.stack([row.old_log_probability for row in ordered])
    reward_advantages = torch.stack([advantage_by_order[row.order][0] for row in ordered])
    cost_advantages = torch.stack([advantage_by_order[row.order][1] for row in ordered])
    reward_targets = torch.stack([advantage_by_order[row.order][2] for row in ordered])
    cost_targets = torch.stack([advantage_by_order[row.order][3] for row in ordered])
    learning_densities = torch.stack([row.density for row in ordered])
    batch = PPOBatch(
        actor_observations=actor_observations,
        critic_observations=critic_observations,
        action_masks=action_masks,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        reward_advantages=reward_advantages,
        cost_advantages=cost_advantages,
        reward_value_targets=reward_targets,
        cost_value_targets=cost_targets,
        cost_penalty_weights=dual_ascent.penalty_weights(learning_densities),
    )
    reward_predictions = torch.stack([row.reward_value for row in ordered])
    cost_predictions = torch.stack([row.cost_value for row in ordered])
    all_reward_tensor = torch.cat(all_rewards)
    all_cost_tensor = torch.cat(all_costs)
    all_sampled_tensor = torch.cat(all_sampled)
    all_density_tensor = torch.cat(all_densities)
    rollout_transitions = int(all_reward_tensor.numel())
    if rollout_transitions != sum(action_counts.values()):
        raise TraceTrainingError("rollout action counts do not partition transitions")
    return _PreparedRollout(
        batch=batch,
        reward_predictions=reward_predictions,
        cost_predictions=cost_predictions,
        all_rewards=all_reward_tensor,
        all_training_costs=all_cost_tensor,
        all_sampled_miss_costs=all_sampled_tensor,
        all_densities=all_density_tensor,
        action_counts=tuple(action_counts[index] for index in range(len(PolicyAction))),
        rollout_transitions=rollout_transitions,
    )


def _advantages_by_row(
    rows: list[_LearningRow],
    *,
    config: ProjectConfig,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    by_pair: dict[str, list[_LearningRow]] = defaultdict(list)
    for row in rows:
        by_pair[row.pair_id].append(row)
    result: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for pair_rows in by_pair.values():
        pair_rows.sort(key=lambda row: row.frame_index)
        continuation_values: list[bool] = []
        for index, row in enumerate(pair_rows):
            has_next_learning_row = (
                index + 1 < len(pair_rows)
                and pair_rows[index + 1].frame_index == row.frame_index + 1
            )
            continuation_values.append(bool(row.gae_continuation.item()) and has_next_learning_row)
        estimates = reward_cost_generalized_advantage_estimate(
            rewards=torch.stack([row.reward for row in pair_rows]),
            costs=torch.stack([row.cost for row in pair_rows]),
            reward_values=torch.stack([row.reward_value for row in pair_rows]),
            reward_next_values=torch.stack([row.reward_next_value for row in pair_rows]),
            cost_values=torch.stack([row.cost_value for row in pair_rows]),
            cost_next_values=torch.stack([row.cost_next_value for row in pair_rows]),
            value_bootstrap_mask=torch.stack([row.value_bootstrap for row in pair_rows]),
            gae_continuation_mask=torch.tensor(
                continuation_values,
                dtype=torch.bool,
            ),
            active_mask=torch.ones(len(pair_rows), dtype=torch.bool),
            gamma=config.training.gamma,
            gae_lambda=config.training.gae_lambda,
            time_dimension=0,
        )
        for index, row in enumerate(pair_rows):
            result[row.order] = (
                estimates.reward.advantages[index],
                estimates.cost.advantages[index],
                estimates.reward.value_targets[index],
                estimates.cost.value_targets[index],
            )
    return result


def _optimize(
    *,
    updater: PPOUpdater,
    batch: PPOBatch,
    update_epochs: int,
    minibatch_size: int,
    generator: torch.Generator,
) -> tuple[PPOUpdateMetrics, ...]:
    updates: list[PPOUpdateMetrics] = []
    for _ in range(update_epochs):
        permutation = torch.randperm(batch.batch_size, generator=generator)
        for start in range(0, batch.batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            updates.append(updater.update(_index_batch(batch, indices)))
    if not updates:
        raise TraceTrainingError("smoke optimization produced no PPO updates")
    return tuple(updates)


def _index_batch(batch: PPOBatch, indices: torch.Tensor) -> PPOBatch:
    return PPOBatch(
        actor_observations=batch.actor_observations[indices],
        critic_observations=batch.critic_observations[indices],
        action_masks=batch.action_masks[indices],
        actions=batch.actions[indices],
        old_log_probabilities=batch.old_log_probabilities[indices],
        reward_advantages=batch.reward_advantages[indices],
        cost_advantages=batch.cost_advantages[indices],
        reward_value_targets=batch.reward_value_targets[indices],
        cost_value_targets=batch.cost_value_targets[indices],
        cost_penalty_weights=batch.cost_penalty_weights[indices],
    )


def _report_payload(
    *,
    config: ProjectConfig,
    source: FrameTraceSource,
    policy_seed: int,
    environment_seed: int,
    max_frames: int,
    rollout_result: Mapping[str, object],
    prepared: _PreparedRollout,
    metrics: TrainingIterationMetrics,
    checkpoint: TrainingCheckpointSummary,
) -> dict[str, object]:
    reward_mean = float(prepared.all_rewards.to(torch.float64).mean().item())
    conditional_mean = float(prepared.all_training_costs.to(torch.float64).mean().item())
    sampled_mean = float(prepared.all_sampled_miss_costs.to(torch.float64).mean().item())
    for name, value in (
        ("reward_mean", reward_mean),
        ("conditional_miss_mean", conditional_mean),
        ("sampled_miss_rate", sampled_mean),
    ):
        if not math.isfinite(value):
            raise TraceTrainingError(f"{name} is non-finite")
    return {
        "schema": TRACE_SMOKE_REPORT_SCHEMA,
        "scope": "integration smoke run; not convergence or feasibility evidence",
        "config_hash": config_hash(config),
        "trace_id": source.trace_id,
        "split": source.split,
        "density_veh_per_lane_km": source.density,
        "policy_seed": policy_seed,
        "environment_seed": environment_seed,
        "device": "cpu",
        "frames_processed": max_frames,
        "optimization_frames": max_frames - 1,
        "bootstrap_frame_index": max_frames - 1,
        "rollout_transitions": prepared.rollout_transitions,
        "learning_rows": prepared.batch.batch_size,
        "action_counts": {
            action.label: prepared.action_counts[int(action)] for action in PolicyAction
        },
        "reward_mean": reward_mean,
        "conditional_miss_mean": conditional_mean,
        "sampled_miss_rate": sampled_mean,
        "training_iteration": metrics.as_dict(),
        "environment_report": {
            field: rollout_result[field] for field in _ENVIRONMENT_REPORT_FIELDS
        },
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "size_bytes": checkpoint.size_bytes,
            "counters": checkpoint.counters.as_dict(),
        },
    }


__all__ = [
    "TRACE_SMOKE_POLICY_NAME",
    "TRACE_SMOKE_REPORT_SCHEMA",
    "TRACE_TRAINING_STAGE_NAMES",
    "TraceSmokeTrainingResult",
    "TraceTrainingError",
    "TraceTrainingTimingObserver",
    "run_trace_smoke_training",
]
