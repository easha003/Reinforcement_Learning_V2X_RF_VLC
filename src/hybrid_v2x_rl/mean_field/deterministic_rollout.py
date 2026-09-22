"""Replay-check harness for the complete Phase 5 population data path.

This is deliberately not a training environment.  It composes the real trace
reader, causal observation boundary, action masks, joint-action accounting,
shared RF pool, matched packet tapes, physical channels, packet outcomes, and
pair lifecycle masks so long runs can fail fast before PPO is introduced.

Unusable causal rows never reach a policy.  The train-only normalizer produces
the finite rectangular actor API, holds statistics frozen across the complete
joint decision, verifies fallback actions, and only then updates once from the
valid raw rows.  The return boundary independently marks unavailable rows as
non-learning transitions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

import numpy as np

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.core.randomness import make_generator
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.env.episodes import VehiclePose
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource, PopulationFrameReader
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizer
from hybrid_v2x_rl.mean_field.packet_outcomes import assemble_frame_outcomes
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PopulationPolicy,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.return_boundaries import FrameReturnBoundary
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel
from hybrid_v2x_rl.mean_field.seeding import EnvironmentSeedState

POLICY_RANDOM: Final = "random"
POLICY_CYCLE: Final = "cycle"
POLICY_NAMES: Final = (POLICY_RANDOM, POLICY_CYCLE)
_POLICY_STREAM: Final = "hybrid-rf-vlc-rl.phase5-validation-policy.v1"


class DeterministicRolloutError(HybridV2XError):
    """A validation rollout request or composed invariant is invalid."""


class _Digest(Protocol):
    def update(self, data: bytes) -> object: ...


@dataclass(frozen=True, slots=True)
class BootstrapObservationReference:
    """Identity of a required internal-truncation bootstrap observation.

    The Phase 5 harness validates exact coverage without inventing a numeric
    actor tensor.  A future trainable environment must replace this reference
    with the separately materialized final observation before value inference.
    """

    trace_id: str
    pair_id: str
    next_frame_index: int


@dataclass(frozen=True, slots=True)
class DeterministicRolloutReport:
    """Compact counters and a content fingerprint for one validation run."""

    trace_id: str
    policy: str
    environment_seed: int
    policy_seed: int
    requested_max_frames: int | None
    available_frames: int
    frames: int
    source_exhausted: bool
    nonempty_frames: int
    transitions: int
    usable_transitions: int
    fallback_transitions: int
    births: int
    natural_terminations: int
    internal_truncations: int
    trace_end_truncations: int
    misses: int
    reward_sum: float
    conditional_risk_sum: float
    reserved_rf_attempts: int
    vlc_activations: int
    max_population: int
    max_pool_utilization: float
    normalization_training_rows: int
    fingerprint: str

    def __post_init__(self) -> None:
        integer_fields = (
            "environment_seed",
            "policy_seed",
            "available_frames",
            "frames",
            "nonempty_frames",
            "transitions",
            "usable_transitions",
            "fallback_transitions",
            "births",
            "natural_terminations",
            "internal_truncations",
            "trace_end_truncations",
            "misses",
            "reserved_rf_attempts",
            "vlc_activations",
            "max_population",
            "normalization_training_rows",
        )
        if any(
            not isinstance(getattr(self, name), int)
            or isinstance(getattr(self, name), bool)
            or getattr(self, name) < 0
            for name in integer_fields
        ):
            raise DeterministicRolloutError("rollout report integer counters must be non-negative")
        if self.requested_max_frames is not None and (
            not isinstance(self.requested_max_frames, int)
            or isinstance(self.requested_max_frames, bool)
            or self.requested_max_frames <= 0
        ):
            raise DeterministicRolloutError("requested_max_frames must be positive or None")
        if type(self.source_exhausted) is not bool:
            raise DeterministicRolloutError("source_exhausted must be boolean")
        if not math.isfinite(self.reward_sum) or self.reward_sum > 0.0:
            raise DeterministicRolloutError("reward_sum must be finite and non-positive")
        if not math.isfinite(self.conditional_risk_sum) or self.conditional_risk_sum < 0:
            raise DeterministicRolloutError("conditional_risk_sum must be finite and non-negative")
        if not math.isfinite(self.max_pool_utilization) or self.max_pool_utilization < 0.0:
            raise DeterministicRolloutError("max_pool_utilization must be finite and non-negative")
        if len(self.fingerprint) != 64:
            raise DeterministicRolloutError("fingerprint must be a SHA-256 digest")
        if self.usable_transitions + self.fallback_transitions != self.transitions:
            raise DeterministicRolloutError(
                "usable and fallback counts must partition all transitions"
            )
        if self.normalization_training_rows != self.usable_transitions:
            raise DeterministicRolloutError(
                "normalization must update from every and only usable transition"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def canonical_policy_name(policy: str) -> str:
    """Normalize a harness policy name or one fixed contract action."""

    if not isinstance(policy, str) or not policy.strip():
        raise DeterministicRolloutError("policy must be a non-empty string")
    normalized = policy.strip().lower().replace("_", "-")
    if normalized in POLICY_NAMES:
        return normalized
    candidate = normalized.upper()
    try:
        return action_resources(candidate).name
    except HybridV2XError as error:
        fixed = tuple(action.label for action in PolicyAction)
        raise DeterministicRolloutError(
            "unknown validation policy",
            context={"policy": policy, "known": POLICY_NAMES + fixed},
        ) from error


def _select_proposal(
    *,
    policy: str,
    policy_seed: int,
    trace_id: str,
    pair_id: str,
    episode_step: int,
    allowed_actions: tuple[PolicyAction, ...],
) -> PolicyAction:
    if not allowed_actions:
        raise DeterministicRolloutError("validation policy has no legal action")
    if policy == POLICY_RANDOM:
        generator = make_generator(
            policy_seed,
            _POLICY_STREAM,
            trace_id=trace_id,
            episode_id=pair_id,
            packet_index=episode_step,
        )
        return allowed_actions[int(generator.integers(0, len(allowed_actions)))]
    if policy == POLICY_CYCLE:
        return allowed_actions[episode_step % len(allowed_actions)]
    fixed = action_resources(policy).action
    if fixed not in allowed_actions:
        raise DeterministicRolloutError(
            "fixed validation policy selects a hardware-masked action",
            context={"policy": policy},
        )
    return fixed


@dataclass(frozen=True, slots=True)
class _DeterministicPolicy:
    """Adapt the Phase 5 random/cycle/fixed selectors to the shared engine."""

    canonical_name: str
    policy_seed: int

    @property
    def name(self) -> str:
        return self.canonical_name

    @property
    def requires_oracle_truth(self) -> bool:
        return False

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyAction | None, ...]:
        if channel_truth is not None:
            raise DeterministicRolloutError(
                "a Phase 5 validation policy cannot receive oracle truth"
            )
        return tuple(
            (
                _select_proposal(
                    policy=self.canonical_name,
                    policy_seed=self.policy_seed,
                    trace_id=decision.frame.trace_id,
                    pair_id=pair.pair_id,
                    episode_step=pair.episode_step,
                    allowed_actions=decision.action_space.mask.allowed_actions,
                )
                if actor_row.usable
                else None
            )
            for pair, actor_row in zip(
                decision.frame.pairs,
                decision.actor_frame.rows,
                strict=True,
            )
        )


def _fingerprint_update(digest: _Digest, payload: object) -> None:
    def numpy_scalar(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"fingerprint payload contains unsupported {type(value).__name__}")

    encoded = json.dumps(
        payload,
        allow_nan=False,
        default=numpy_scalar,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(encoded)
    digest.update(b"\n")


def run_policy_rollout(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    policy: PopulationPolicy,
    environment_seed: int | None = None,
    policy_seed: int = 0,
    max_frames: int | None = None,
) -> DeterministicRolloutReport:
    """Run any population policy through the one authoritative environment path.

    ``max_frames`` is a diagnostic processing cutoff.  It does not manufacture
    truncation flags for still-active pairs; only trace lifecycle metadata is
    counted as a training boundary.
    """

    if not isinstance(config, ProjectConfig):
        raise DeterministicRolloutError("rollout requires a resolved ProjectConfig")
    if not isinstance(source, FrameTraceSource):
        raise DeterministicRolloutError("rollout requires a FrameTraceSource")
    if not isinstance(policy, PopulationPolicy):
        raise DeterministicRolloutError("rollout requires a PopulationPolicy")
    if not isinstance(policy.name, str) or not policy.name.strip():
        raise DeterministicRolloutError("population policy name must be non-empty")
    if type(policy.requires_oracle_truth) is not bool:
        raise DeterministicRolloutError(
            "population policy oracle declaration must be boolean"
        )
    if max_frames is not None and (
        not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0
    ):
        raise DeterministicRolloutError("max_frames must be positive or None")
    canonical_policy = policy.name

    seed_state = EnvironmentSeedState.from_config(
        config,
        reset_seed=environment_seed,
    )
    randomness = seed_state.for_trace(source.trace_id)
    actor_assembler = randomness.actor_assembler(config)
    physical = build_rollout(
        config,
        buildings=(),
        root_seed=seed_state.active_root_seed,
        band=SensitivityBand.NOMINAL,
    )
    pool_model = RFPoolModel(
        parameters=physical.lifecycle.rf.collision,
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )
    reader = PopulationFrameReader(
        source,
        generation_period_s=config.service.generation_period_s,
        expected_config_hash=config_hash(config),
    )
    action_space = MaskedActionSpace.from_config(
        config.environment,
        config.rf,
        config.vlc,
    )
    normalizer = ObservationNormalizer.from_config(config)
    resource_map = ActionResourceMap.from_config(config.environment, config.cost)

    digest = hashlib.sha256()
    _fingerprint_update(
        digest,
        {
            "schema": _POLICY_STREAM,
            "trace_id": source.trace_id,
            "policy": canonical_policy,
            "environment_seed": seed_state.active_root_seed,
            "policy_seed": policy_seed,
            "max_frames": max_frames,
        },
    )

    frames = nonempty_frames = transitions = usable_transitions = 0
    fallback_transitions = births = natural_terminations = 0
    internal_truncations = trace_end_truncations = misses = 0
    reserved_rf_attempts = vlc_activations = max_population = 0
    reward_terms: list[float] = []
    conditional_risk_terms: list[float] = []
    max_pool_utilization = 0.0

    for frame in reader.iter_frames(max_frames=max_frames):
        actor_frame = actor_assembler.begin_frame(frame)
        normalized_frame = normalizer.begin_frame(frame, actor_frame)
        observation = normalized_frame.observation
        tapes = randomness.packet_tapes(frame)
        # Physical truth is action-independent and may be materialized before
        # the joint decision.  It crosses the policy boundary only for the
        # explicitly non-deployable oracle; deployable policies receive None.
        channel_evaluations = {
            pair.pair_id: physical.evaluate_channels(
                trace_id=frame.trace_id,
                pair_id=pair.pair_id,
                density=source.density,
                time_s=frame.time_s,
                transmitter=cast(VehiclePose, pair.transmitter),
                receiver=cast(VehiclePose, pair.receiver),
                neighbours=cast(tuple[VehiclePose, ...], frame.vehicles),
                index_of_frame=frame.spatial_index,
                vlc_randomness=tapes[pair.pair_id].vlc,
            )
            for pair in frame.pairs
        }
        decision = PopulationPolicyFrame(
            frame=frame,
            actor_frame=actor_frame,
            observation=observation,
            action_space=action_space,
            resource_map=resource_map,
            pool_model=pool_model,
            miss_budget=config.service.miss_budget,
        )
        visible_truth: OracleChannelTruth | None = None
        if policy.requires_oracle_truth:
            visible_truth = MappingProxyType(channel_evaluations)
        proposed = policy.select_actions(decision, channel_truth=visible_truth)
        if not isinstance(proposed, tuple) or len(proposed) != len(frame.pairs):
            raise DeterministicRolloutError(
                "policy proposals must be a pair-aligned tuple",
                context={"actual": len(proposed), "expected": len(frame.pairs)},
            )
        proposals: list[PolicyAction] = []
        actions_by_pair: dict[str, PolicyAction] = {}
        for pair, actor_row, proposal in zip(
            frame.pairs,
            actor_frame.rows,
            proposed,
            strict=True,
        ):
            selected = action_space.select(
                proposal,
                observation_usable=actor_row.usable,
            )
            proposals.append(selected)
            actions_by_pair[pair.pair_id] = selected

        action_array = np.asarray(proposals, dtype=np.int64)
        normalizer.complete_frame(normalized_frame, action_array)
        ledger = FrameActionLedger.from_frame(
            frame,
            actions_by_pair,
            resource_map=resource_map,
        )
        demand = RFPoolDemand.from_ledger(ledger)
        pool_response = pool_model.evaluate(demand)
        rf_risks = {
            row.pair_id: pool_model.combine_attempt_risk(
                pool_response,
                pair_id=row.pair_id,
                propagation=channel_evaluations[row.pair_id].rf_propagation,
            )
            for row in ledger.pair_accounting
            if row.uses_rf
        }
        vlc_results = {
            row.pair_id: channel_evaluations[row.pair_id].vlc_result
            for row in ledger.pair_accounting
            if row.uses_vlc
        }
        outcomes = assemble_frame_outcomes(
            ledger,
            pool_response,
            tapes_by_pair=tapes,
            rf_risks_by_pair=rf_risks,
            vlc_results_by_pair=vlc_results,
        )
        boundary = FrameReturnBoundary.from_frame(ledger, actor_frame)
        final_references = {
            pair_id: BootstrapObservationReference(
                trace_id=frame.trace_id,
                pair_id=pair_id,
                next_frame_index=frame.index + 1,
            )
            for pair_id in boundary.bootstrap_pair_ids
        }
        boundary.as_step_info(final_observation=final_references)

        for outcome in outcomes.pair_outcomes:
            spec = action_resources(outcome.action)
            completion_s = max(
                len(outcome.rf_attempts) * config.rf.timing.airtime_s,
                config.vlc.timing.airtime_s if spec.uses_vlc else 0.0,
            )
            if completion_s > config.service.deadline_s + 1e-12:
                raise DeterministicRolloutError(
                    "selected packet completion exceeds the service deadline",
                    context={
                        "pair_id": outcome.pair_id,
                        "completion_s": completion_s,
                        "deadline_s": config.service.deadline_s,
                    },
                )
            actor_assembler.record_feedback(
                outcome.pair_id,
                action=outcome.action,
                at_s=frame.time_s + completion_s,
                delivered=outcome.delivered,
                measurements=randomness.feedback_measurements(
                    outcome,
                    tapes[outcome.pair_id],
                ),
            )
        actor_assembler.close_frame(pool_response)
        for pair_id in boundary.final_pair_ids:
            physical.release(pair_id)

        frames += 1
        population = len(frame.pairs)
        nonempty_frames += int(population > 0)
        transitions += population
        usable = sum(actor_frame.usable_mask)
        usable_transitions += usable
        fallback_transitions += population - usable
        births += len(ledger.born_pair_ids)
        natural_terminations += int(np.count_nonzero(boundary.terminated))
        internal_truncations += len(boundary.bootstrap_pair_ids)
        trace_end_truncations += sum(reason == "trace_end" for reason in boundary.end_reasons)
        misses += int(np.sum(outcomes.sampled_miss_costs, dtype=np.float64))
        reward_terms.extend(float(value) for value in outcomes.rewards)
        conditional_risk_terms.extend(
            float(value) for value in outcomes.conditional_miss_probabilities
        )
        reserved_rf_attempts += ledger.total_reserved_rf_attempts
        vlc_activations += ledger.total_vlc_activations
        max_population = max(max_population, population)
        max_pool_utilization = max(
            max_pool_utilization,
            pool_response.pool_utilization,
        )

        _fingerprint_update(
            digest,
            {
                "frame_index": frame.index,
                "time_s": frame.time_s,
                "pair_ids": frame.active_pair_ids,
                "actor_rows": [
                    (observation.actor_observations[index].tolist() if actor_row.usable else None)
                    for index, actor_row in enumerate(actor_frame.rows)
                ],
                "normalization_count_before": list(normalized_frame.statistics_count_before),
                "actions": [action.label for action in proposals],
                "pool": {
                    "offered_rf_attempts": demand.offered_rf_attempts,
                    "pool_utilization": pool_response.pool_utilization,
                    "channel_busy_ratio": pool_response.channel_busy_ratio,
                    "collision_probability": (pool_response.per_attempt_collision_probability),
                },
                "outcomes": [row.as_dict() for row in outcomes.pair_outcomes],
                "lifecycle": {
                    "terminated": boundary.terminated.tolist(),
                    "truncated": boundary.truncated.tolist(),
                    "bootstrap_valid": boundary.bootstrap_valid.tolist(),
                    "learn_mask": boundary.learn_mask.tolist(),
                    "end_reasons": boundary.end_reasons,
                },
            },
        )

    normalization_state = dict(normalizer.state_dict())
    _fingerprint_update(digest, {"normalization_state": normalization_state})
    source_exhausted = frames == reader.decision_frame_count
    return DeterministicRolloutReport(
        trace_id=source.trace_id,
        policy=canonical_policy,
        environment_seed=seed_state.active_root_seed,
        policy_seed=policy_seed,
        requested_max_frames=max_frames,
        available_frames=reader.decision_frame_count,
        frames=frames,
        source_exhausted=source_exhausted,
        nonempty_frames=nonempty_frames,
        transitions=transitions,
        usable_transitions=usable_transitions,
        fallback_transitions=fallback_transitions,
        births=births,
        natural_terminations=natural_terminations,
        internal_truncations=internal_truncations,
        trace_end_truncations=trace_end_truncations,
        misses=misses,
        reward_sum=math.fsum(reward_terms),
        conditional_risk_sum=math.fsum(conditional_risk_terms),
        reserved_rf_attempts=reserved_rf_attempts,
        vlc_activations=vlc_activations,
        max_population=max_population,
        max_pool_utilization=max_pool_utilization,
        normalization_training_rows=normalizer.training_rows,
        fingerprint=digest.hexdigest(),
    )


def run_deterministic_rollout(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    policy: str,
    environment_seed: int | None = None,
    policy_seed: int = 0,
    max_frames: int | None = None,
) -> DeterministicRolloutReport:
    """Run a Phase 5 random, cycle, or fixed-action validation policy."""

    canonical = canonical_policy_name(policy)
    return run_policy_rollout(
        config,
        source,
        policy=_DeterministicPolicy(canonical, policy_seed),
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        max_frames=max_frames,
    )


def trace_source(project_root: str | Path, trace_id: str) -> FrameTraceSource:
    """Resolve one canonical trace beneath a project for scripts and notebooks."""

    root = Path(project_root).expanduser().resolve()
    return FrameTraceSource.discover(root / "artifacts" / "traces" / trace_id)


__all__ = [
    "POLICY_CYCLE",
    "POLICY_NAMES",
    "POLICY_RANDOM",
    "BootstrapObservationReference",
    "DeterministicRolloutError",
    "DeterministicRolloutReport",
    "canonical_policy_name",
    "run_deterministic_rollout",
    "run_policy_rollout",
    "trace_source",
]
