"""Replay-check harness for the complete Phase 5 population data path.

This is deliberately not a training environment.  It composes the real trace
reader, causal observation boundary, action masks, joint-action accounting,
pair-local RF physics, matched packet tapes, physical channels, packet outcomes, and
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
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, cast

import numpy as np

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.channels.rf.diversity import RFReceiveDiversity
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
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
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource, PopulationFrameReader
from hybrid_v2x_rl.mean_field.local_rf_pipeline import LocalRFPhysicsModel
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationState,
    ObservationNormalizer,
)
from hybrid_v2x_rl.mean_field.packet_outcomes import (
    FramePacketOutcomes,
    assemble_frame_outcomes,
)
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PopulationPolicy,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.random_tape import MATCHED_TAPE_SCHEMA
from hybrid_v2x_rl.mean_field.return_boundaries import FrameReturnBoundary
from hybrid_v2x_rl.mean_field.seeding import EnvironmentSeedState

POLICY_RANDOM: Final = "random"
POLICY_CYCLE: Final = "cycle"
POLICY_NAMES: Final = (POLICY_RANDOM, POLICY_CYCLE)
_POLICY_STREAM: Final = "hybrid-rf-vlc-rl.phase5-validation-policy.v1"


class DeterministicRolloutError(HybridV2XError):
    """A validation rollout request or composed invariant is invalid."""


class _Digest(Protocol):
    def update(self, data: bytes) -> object: ...


class PopulationRolloutObserver(Protocol):
    """Read-only hook over completed frames from the authoritative rollout path."""

    def observe_frame(
        self,
        *,
        decision: PopulationPolicyFrame,
        actions: tuple[PolicyAction, ...],
        outcomes: FramePacketOutcomes,
        boundary: FrameReturnBoundary,
        final_observation: Mapping[str, FrameObservation],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class EpisodeClusterTally:
    """Compact policy metrics for one trajectory/pair episode cluster."""

    pair_id: str
    packets: int
    misses: int
    conditional_risk_sum: float
    reward_sum: float
    action_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise DeterministicRolloutError("episode cluster pair_id must be non-empty")
        if (
            not isinstance(self.packets, int)
            or isinstance(self.packets, bool)
            or self.packets <= 0
            or not isinstance(self.misses, int)
            or isinstance(self.misses, bool)
            or not 0 <= self.misses <= self.packets
        ):
            raise DeterministicRolloutError("episode cluster counts are invalid")
        if (
            not math.isfinite(self.conditional_risk_sum)
            or not 0.0 <= self.conditional_risk_sum <= self.packets
        ):
            raise DeterministicRolloutError(
                "episode conditional-risk sum must lie within its packet count"
            )
        if not math.isfinite(self.reward_sum) or self.reward_sum >= 0.0:
            raise DeterministicRolloutError(
                "episode reward sum must be finite and negative"
            )
        if (
            not isinstance(self.action_counts, tuple)
            or len(self.action_counts) != len(PolicyAction)
            or any(
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for count in self.action_counts
            )
            or sum(self.action_counts) != self.packets
        ):
            raise DeterministicRolloutError(
                "episode action counts must partition its packets"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeterministicRolloutReport:
    """Compact counters and a content fingerprint for one validation run."""

    trace_id: str
    split: str
    policy: str
    environment_seed: int
    policy_seed: int
    requested_start_frame_index: int
    requested_max_frames: int | None
    available_frames: int
    first_frame_index: int
    last_frame_index: int
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
    action_counts: tuple[int, ...]
    max_population: int
    pool_utilization_sum: float
    max_pool_utilization: float
    normalization_training_rows: int
    normalization_total_training_rows: int
    normalization_updates_enabled: bool
    normalization_frozen: bool
    matched_tape_fingerprint: str
    fingerprint: str
    episode_clusters: tuple[EpisodeClusterTally, ...]

    def __post_init__(self) -> None:
        integer_fields = (
            "environment_seed",
            "policy_seed",
            "requested_start_frame_index",
            "available_frames",
            "first_frame_index",
            "last_frame_index",
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
            "normalization_total_training_rows",
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
        if self.available_frames <= 0:
            raise DeterministicRolloutError("available_frames must be positive")
        if not 0 <= self.requested_start_frame_index < self.available_frames:
            raise DeterministicRolloutError(
                "requested_start_frame_index must identify an available frame"
            )
        if self.frames <= 0:
            raise DeterministicRolloutError("a rollout report must contain a frame")
        if self.first_frame_index != self.requested_start_frame_index:
            raise DeterministicRolloutError(
                "first_frame_index must equal the requested window start"
            )
        if self.last_frame_index != self.first_frame_index + self.frames - 1:
            raise DeterministicRolloutError(
                "last_frame_index does not match the processed frame count"
            )
        if self.last_frame_index >= self.available_frames:
            raise DeterministicRolloutError("rollout window exceeds available frames")
        if type(self.source_exhausted) is not bool:
            raise DeterministicRolloutError("source_exhausted must be boolean")
        if self.source_exhausted != (self.last_frame_index == self.available_frames - 1):
            raise DeterministicRolloutError(
                "source_exhausted does not match the final physical frame"
            )
        if self.split not in ("train", "validation", "test"):
            raise DeterministicRolloutError("rollout split must be train, validation, or test")
        if type(self.normalization_updates_enabled) is not bool:
            raise DeterministicRolloutError(
                "normalization_updates_enabled must be boolean"
            )
        if type(self.normalization_frozen) is not bool:
            raise DeterministicRolloutError("normalization_frozen must be boolean")
        if not math.isfinite(self.reward_sum) or self.reward_sum > 0.0:
            raise DeterministicRolloutError("reward_sum must be finite and non-positive")
        if not math.isfinite(self.conditional_risk_sum) or self.conditional_risk_sum < 0:
            raise DeterministicRolloutError("conditional_risk_sum must be finite and non-negative")
        for name in ("pool_utilization_sum", "max_pool_utilization"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise DeterministicRolloutError(f"{name} must be finite and non-negative")
        for name in ("matched_tape_fingerprint", "fingerprint"):
            if len(getattr(self, name)) != 64:
                raise DeterministicRolloutError(f"{name} must be a SHA-256 digest")
        if self.usable_transitions + self.fallback_transitions != self.transitions:
            raise DeterministicRolloutError(
                "usable and fallback counts must partition all transitions"
            )
        if self.normalization_total_training_rows < self.normalization_training_rows:
            raise DeterministicRolloutError(
                "total normalization rows cannot be smaller than this rollout's updates"
            )
        expected_updates = self.usable_transitions if self.normalization_updates_enabled else 0
        if self.normalization_training_rows != expected_updates:
            raise DeterministicRolloutError(
                "normalization must update from every and only eligible usable transition"
            )
        if self.normalization_updates_enabled and self.split != "train":
            raise DeterministicRolloutError(
                "only a training rollout may update normalization"
            )
        if (
            not isinstance(self.action_counts, tuple)
            or len(self.action_counts) != len(PolicyAction)
            or any(
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for count in self.action_counts
            )
            or sum(self.action_counts) != self.transitions
        ):
            raise DeterministicRolloutError(
                "rollout action counts must partition all transitions"
            )
        if not isinstance(self.episode_clusters, tuple) or any(
            not isinstance(cluster, EpisodeClusterTally)
            for cluster in self.episode_clusters
        ):
            raise DeterministicRolloutError(
                "rollout episode clusters must be immutable validated tallies"
            )
        pair_ids = tuple(cluster.pair_id for cluster in self.episode_clusters)
        if pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(set(pair_ids)):
            raise DeterministicRolloutError(
                "rollout episode clusters must use unique canonical pair IDs"
            )
        cluster_action_counts = tuple(
            sum(cluster.action_counts[index] for cluster in self.episode_clusters)
            for index in range(len(PolicyAction))
        )
        if (
            sum(cluster.packets for cluster in self.episode_clusters) != self.transitions
            or sum(cluster.misses for cluster in self.episode_clusters) != self.misses
            or cluster_action_counts != self.action_counts
            or not math.isclose(
                math.fsum(
                    cluster.conditional_risk_sum for cluster in self.episode_clusters
                ),
                self.conditional_risk_sum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                math.fsum(cluster.reward_sum for cluster in self.episode_clusters),
                self.reward_sum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise DeterministicRolloutError(
                "episode cluster tallies do not reconcile with rollout totals"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyRolloutResult:
    """One rollout report plus the checkpointable normalizer state it produced."""

    report: DeterministicRolloutReport
    normalization_state: ObservationNormalizationState

    def __post_init__(self) -> None:
        if not isinstance(self.report, DeterministicRolloutReport):
            raise DeterministicRolloutError("policy rollout result requires a report")
        if not isinstance(self.normalization_state, ObservationNormalizationState):
            raise DeterministicRolloutError(
                "policy rollout result requires normalization state"
            )
        if (
            self.report.normalization_total_training_rows
            != self.normalization_state.count[
                self.normalization_state.standardized.index(True)
            ]
        ):
            raise DeterministicRolloutError(
                "rollout report and normalization checkpoint counts differ"
            )
        if self.report.normalization_frozen != self.normalization_state.frozen:
            raise DeterministicRolloutError(
                "rollout report and normalization checkpoint freeze state differ"
            )


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


def run_policy_rollout_with_state(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    policy: PopulationPolicy,
    environment_seed: int | None = None,
    policy_seed: int = 0,
    start_frame_index: int = 0,
    max_frames: int | None = None,
    normalization_state: ObservationNormalizationState | Mapping[str, object] | None = None,
    freeze_normalization_at_end: bool = False,
    frame_observer: PopulationRolloutObserver | None = None,
    sensitivity_band: SensitivityBand = SensitivityBand.NOMINAL,
    collision_subchannels: int | None = None,
    receive_diversity: RFReceiveDiversity | None = None,
    oracle_controls_unusable_rows: bool = False,
) -> PolicyRolloutResult:
    """Run any population policy through the one authoritative environment path.

    ``start_frame_index`` begins a fresh sampled episode at that physical trace
    instant. ``max_frames`` is a diagnostic processing cutoff. Neither option
    manufactures truncation flags for still-active pairs; only trace lifecycle
    metadata is counted as a training boundary.
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
    if (
        not isinstance(start_frame_index, int)
        or isinstance(start_frame_index, bool)
        or start_frame_index < 0
    ):
        raise DeterministicRolloutError("start_frame_index must be a nonnegative integer")
    if max_frames is not None and (
        not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0
    ):
        raise DeterministicRolloutError("max_frames must be positive or None")
    if type(freeze_normalization_at_end) is not bool:
        raise DeterministicRolloutError("freeze_normalization_at_end must be boolean")
    if not isinstance(sensitivity_band, SensitivityBand):
        raise DeterministicRolloutError(
            "sensitivity_band must be a declared SensitivityBand"
        )
    if collision_subchannels is not None and (
        not isinstance(collision_subchannels, int)
        or isinstance(collision_subchannels, bool)
        or collision_subchannels < 1
    ):
        raise DeterministicRolloutError(
            "collision_subchannels must be a positive integer or None"
        )
    if receive_diversity is not None and not isinstance(
        receive_diversity, RFReceiveDiversity
    ):
        raise DeterministicRolloutError(
            "receive_diversity must be an RFReceiveDiversity profile or None"
        )
    if type(oracle_controls_unusable_rows) is not bool:
        raise DeterministicRolloutError(
            "oracle_controls_unusable_rows must be boolean"
        )
    if oracle_controls_unusable_rows and not policy.requires_oracle_truth:
        raise DeterministicRolloutError(
            "only a declared non-deployable oracle may control unusable rows"
        )
    if frame_observer is not None and not callable(
        getattr(frame_observer, "observe_frame", None)
    ):
        raise DeterministicRolloutError(
            "frame_observer must provide an observe_frame method"
        )
    canonical_policy = policy.name

    seed_state = EnvironmentSeedState.from_config(
        config,
        reset_seed=environment_seed,
    )
    randomness = seed_state.for_trace(source.trace_id)
    actor_assembler = randomness.actor_assembler(
        config,
        start_frame_index=start_frame_index,
    )
    local_rf_model = LocalRFPhysicsModel.from_config(
        config,
        sensitivity_band=sensitivity_band,
        collision_subchannels=collision_subchannels,
    )
    physical = build_rollout(
        config,
        buildings=local_rf_model.buildings,
        root_seed=seed_state.active_root_seed,
        band=sensitivity_band,
        collision_subchannels=collision_subchannels,
        receive_diversity=receive_diversity,
    )
    reader = PopulationFrameReader(
        source,
        generation_period_s=config.service.generation_period_s,
        expected_config_hash=config_hash(config),
        expected_config_scope_hashes={
            "mobility_trace": scope_hash(config, "mobility_trace")
        },
    )
    action_space = MaskedActionSpace.from_config(
        config.environment,
        config.rf,
        config.vlc,
    )
    if normalization_state is None:
        normalizer = ObservationNormalizer.from_config(config)
    else:
        payload = (
            normalization_state.as_dict()
            if isinstance(normalization_state, ObservationNormalizationState)
            else normalization_state
        )
        normalizer = ObservationNormalizer.from_state_dict(config, payload)
    normalization_rows_before = normalizer.training_rows
    normalization_updates_enabled = not normalizer.frozen
    resource_map = ActionResourceMap.from_config(config.environment, config.cost)

    digest = hashlib.sha256()
    tape_digest = hashlib.sha256()
    policy_identity: dict[str, object] = {
        "schema": _POLICY_STREAM,
        "trace_id": source.trace_id,
        "policy": canonical_policy,
        "environment_seed": seed_state.active_root_seed,
        "policy_seed": policy_seed,
        "start_frame_index": start_frame_index,
        "max_frames": max_frames,
        "sensitivity_band": sensitivity_band.value,
        "collision_subchannels": collision_subchannels,
        "oracle_controls_unusable_rows": oracle_controls_unusable_rows,
    }
    if receive_diversity is not None:
        policy_identity["receive_diversity"] = receive_diversity.as_dict()
    _fingerprint_update(digest, policy_identity)
    _fingerprint_update(
        tape_digest,
        {
            "schema": MATCHED_TAPE_SCHEMA,
            "trace_id": source.trace_id,
            "environment_seed": seed_state.active_root_seed,
            "start_frame_index": start_frame_index,
            "max_frames": max_frames,
        },
    )

    frames = nonempty_frames = transitions = usable_transitions = 0
    fallback_transitions = births = natural_terminations = 0
    internal_truncations = trace_end_truncations = misses = 0
    reserved_rf_attempts = vlc_activations = max_population = 0
    reward_terms: list[float] = []
    conditional_risk_terms: list[float] = []
    pool_utilization_terms: list[float] = []
    action_counts = [0] * len(PolicyAction)
    cluster_packets: dict[str, int] = {}
    cluster_misses: dict[str, int] = {}
    cluster_risks: dict[str, list[float]] = {}
    cluster_rewards: dict[str, list[float]] = {}
    cluster_actions: dict[str, list[int]] = {}
    max_pool_utilization = 0.0

    reader_frame_limit = None if max_frames is None else max_frames + 1
    frame_iterator = iter(
        reader.iter_frames(
            start_frame_index=start_frame_index,
            max_frames=reader_frame_limit,
        )
    )
    try:
        frame = next(frame_iterator)
    except StopIteration:
        frame = None
    while frame is not None and (max_frames is None or frames < max_frames):
        try:
            next_frame = next(frame_iterator)
        except StopIteration:
            next_frame = None
        actor_frame = actor_assembler.begin_frame(frame)
        local_rf_context = local_rf_model.context_for(frame)
        normalized_frame = normalizer.begin_frame(frame, actor_frame)
        observation = normalized_frame.observation
        tapes = randomness.packet_tapes(frame)
        _fingerprint_update(
            tape_digest,
            {
                "frame_index": frame.index,
                "tapes": [
                    asdict(tapes[pair_id]) for pair_id in frame.active_pair_ids
                ],
            },
        )
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
            local_rf_model=local_rf_model,
            local_rf_context=local_rf_context,
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
                observation_usable=(
                    actor_row.usable or oracle_controls_unusable_rows
                ),
            )
            proposals.append(selected)
            actions_by_pair[pair.pair_id] = selected

        action_array = np.asarray(proposals, dtype=np.int64)
        normalizer.complete_frame(
            normalized_frame,
            action_array,
            oracle_controls_unusable_rows=oracle_controls_unusable_rows,
        )
        ledger = FrameActionLedger.from_frame(
            frame,
            actions_by_pair,
            resource_map=resource_map,
        )
        local_rf_physics = local_rf_model.evaluate(
            local_rf_context,
            ledger,
            propagation_by_pair={
                pair_id: evaluation.rf_propagation
                for pair_id, evaluation in channel_evaluations.items()
            },
        )
        vlc_results = {
            row.pair_id: channel_evaluations[row.pair_id].vlc_result
            for row in ledger.pair_accounting
            if row.uses_vlc
        }
        outcomes = assemble_frame_outcomes(
            ledger,
            local_rf_physics,
            tapes_by_pair=tapes,
            vlc_results_by_pair=vlc_results,
        )
        boundary = FrameReturnBoundary.from_frame(ledger, actor_frame)

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
            pair_id = outcome.pair_id
            cluster_packets[pair_id] = cluster_packets.get(pair_id, 0) + 1
            cluster_misses[pair_id] = (
                cluster_misses.get(pair_id, 0) + outcome.sampled_miss_cost
            )
            cluster_risks.setdefault(pair_id, []).append(
                outcome.conditional_miss_probability
            )
            cluster_rewards.setdefault(pair_id, []).append(outcome.reward)
            per_action = cluster_actions.setdefault(
                pair_id,
                [0] * len(PolicyAction),
            )
            per_action[int(outcome.action)] += 1
            action_counts[int(outcome.action)] += 1
        final_actor_frame = actor_assembler.close_frame(
            local_rf_physics,
            next_frame=next_frame,
        )
        final_observation: dict[str, FrameObservation] = {}
        if final_actor_frame is not None:
            if next_frame is None:  # pragma: no cover - assembler rejects this first.
                raise DeterministicRolloutError(
                    "final actor observations require a next physical frame"
                )
            if final_actor_frame.pair_ids != boundary.bootstrap_pair_ids:
                raise DeterministicRolloutError(
                    "final actor observations do not cover bootstrap pairs exactly",
                    context={
                        "actual": final_actor_frame.pair_ids,
                        "expected": boundary.bootstrap_pair_ids,
                    },
                )
            if not all(final_actor_frame.usable_mask):
                raise DeterministicRolloutError(
                    "a bootstrap-valid final observation must be causally usable",
                    context={"pair_ids": final_actor_frame.unusable_pair_ids},
                )
            normalized_final = normalizer.transform_final_observations(
                next_frame,
                final_actor_frame,
            )
            final_batch = normalized_final.observation
            for row, pair_id in enumerate(final_batch.pair_ids):
                final_observation[pair_id] = FrameObservation(
                    trace_id=final_batch.trace_id,
                    frame_index=final_batch.frame_index,
                    time_s=final_batch.time_s,
                    pair_ids=(pair_id,),
                    actor_observations=final_batch.actor_observations[row : row + 1],
                    action_masks=final_batch.action_masks[row : row + 1],
                )
        boundary.as_step_info(final_observation=final_observation)
        frozen_final_observation = MappingProxyType(final_observation)
        if frame_observer is not None:
            frame_observer.observe_frame(
                decision=decision,
                actions=tuple(proposals),
                outcomes=outcomes,
                boundary=boundary,
                final_observation=frozen_final_observation,
            )
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
        internal_truncations += sum(
            reason == "max_duration" for reason in boundary.end_reasons
        )
        trace_end_truncations += sum(reason == "trace_end" for reason in boundary.end_reasons)
        misses += int(np.sum(outcomes.sampled_miss_costs, dtype=np.float64))
        reward_terms.extend(outcome.reward for outcome in outcomes.pair_outcomes)
        conditional_risk_terms.extend(
            outcome.conditional_miss_probability
            for outcome in outcomes.pair_outcomes
        )
        reserved_rf_attempts += ledger.total_reserved_rf_attempts
        vlc_activations += ledger.total_vlc_activations
        max_population = max(max_population, population)
        max_pool_utilization = max(
            max_pool_utilization,
            local_rf_physics.max_local_pool_utilization,
        )
        pool_utilization_terms.append(
            local_rf_physics.mean_local_pool_utilization
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
                "local_rf": {
                    "total_reserved_rf_attempts": (
                        ledger.total_reserved_rf_attempts
                    ),
                    "mean_local_pool_utilization": (
                        local_rf_physics.mean_local_pool_utilization
                    ),
                    "max_local_pool_utilization": (
                        local_rf_physics.max_local_pool_utilization
                    ),
                    "pair_responses": [
                        response.as_dict()
                        for response in local_rf_physics.responses.responses
                    ],
                },
                "outcomes": [row.as_dict() for row in outcomes.pair_outcomes],
                "lifecycle": {
                    "terminated": boundary.terminated.tolist(),
                    "truncated": boundary.truncated.tolist(),
                    "bootstrap_valid": boundary.bootstrap_valid.tolist(),
                    "learn_mask": boundary.learn_mask.tolist(),
                    "end_reasons": boundary.end_reasons,
                    "final_observation": {
                        pair_id: value.actor_observations[0].tolist()
                        for pair_id, value in final_observation.items()
                    },
                },
            },
        )
        frame = next_frame

    if freeze_normalization_at_end and not normalizer.frozen:
        final_normalization_state = normalizer.freeze()
    else:
        final_normalization_state = normalizer.snapshot()
    final_normalization_payload = final_normalization_state.as_dict()
    _fingerprint_update(
        digest,
        {"normalization_state": final_normalization_payload},
    )
    first_frame_index = start_frame_index
    last_frame_index = first_frame_index + frames - 1
    source_exhausted = last_frame_index == reader.last_frame_index
    episode_clusters = tuple(
        EpisodeClusterTally(
            pair_id=pair_id,
            packets=cluster_packets[pair_id],
            misses=cluster_misses[pair_id],
            conditional_risk_sum=math.fsum(cluster_risks[pair_id]),
            reward_sum=math.fsum(cluster_rewards[pair_id]),
            action_counts=tuple(cluster_actions[pair_id]),
        )
        for pair_id in sorted(cluster_packets)
    )
    report = DeterministicRolloutReport(
        trace_id=source.trace_id,
        split=source.split,
        policy=canonical_policy,
        environment_seed=seed_state.active_root_seed,
        policy_seed=policy_seed,
        requested_start_frame_index=start_frame_index,
        requested_max_frames=max_frames,
        available_frames=reader.decision_frame_count,
        first_frame_index=first_frame_index,
        last_frame_index=last_frame_index,
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
        action_counts=tuple(action_counts),
        max_population=max_population,
        pool_utilization_sum=math.fsum(pool_utilization_terms),
        max_pool_utilization=max_pool_utilization,
        normalization_training_rows=(
            normalizer.training_rows - normalization_rows_before
        ),
        normalization_total_training_rows=normalizer.training_rows,
        normalization_updates_enabled=normalization_updates_enabled,
        normalization_frozen=final_normalization_state.frozen,
        matched_tape_fingerprint=tape_digest.hexdigest(),
        fingerprint=digest.hexdigest(),
        episode_clusters=episode_clusters,
    )
    return PolicyRolloutResult(
        report=report,
        normalization_state=final_normalization_state,
    )


def run_policy_rollout(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    policy: PopulationPolicy,
    environment_seed: int | None = None,
    policy_seed: int = 0,
    start_frame_index: int = 0,
    max_frames: int | None = None,
) -> DeterministicRolloutReport:
    """Backward-compatible report-only population rollout entry point."""

    return run_policy_rollout_with_state(
        config,
        source,
        policy=policy,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        start_frame_index=start_frame_index,
        max_frames=max_frames,
    ).report


def run_deterministic_rollout(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    policy: str,
    environment_seed: int | None = None,
    policy_seed: int = 0,
    start_frame_index: int = 0,
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
        start_frame_index=start_frame_index,
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
    "DeterministicRolloutError",
    "DeterministicRolloutReport",
    "EpisodeClusterTally",
    "PopulationRolloutObserver",
    "PolicyRolloutResult",
    "canonical_policy_name",
    "run_deterministic_rollout",
    "run_policy_rollout",
    "run_policy_rollout_with_state",
    "trace_source",
]
