"""Frozen validation evaluation of the exact population-joint risk oracle."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch

from hybrid_v2x_rl.agents.checkpointing import restore_training_checkpoint
from hybrid_v2x_rl.agents.regime_evaluation import (
    EvaluationWindow,
    load_frozen_regime_audit,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER, PolicyAction
from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout_with_state
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource, TraceCatalog
from hybrid_v2x_rl.mean_field.joint_risk_oracle import (
    JOINT_RISK_ORACLE_METHOD,
    JointRiskOracleProblem,
    JointRiskOracleSolution,
    solve_joint_risk_floor,
)
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizationState
from hybrid_v2x_rl.mean_field.packet_outcomes import FramePacketOutcomes
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PolicyProposal,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.return_boundaries import FrameReturnBoundary
from hybrid_v2x_rl.mean_field.state_regime_audit import STATE_REGIME_AUDIT_SCHEMA

JOINT_ORACLE_EVALUATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.population-joint-risk-oracle-evaluation.v1"
)
_RISK_RECONCILIATION_TOLERANCE: Final = 2e-6


class JointOracleEvaluationError(HybridV2XError):
    """A joint-oracle evaluation input, invariant, or artifact is invalid."""


@dataclass(slots=True)
class _JointOracleTally:
    frames: int = 0
    transitions: int = 0
    usable_transitions: int = 0
    forced_fallback_transitions: int = 0
    conditional_risk_sum: float = 0.0
    usable_conditional_risk_sum: float = 0.0
    forced_fallback_conditional_risk_sum: float = 0.0
    sampled_misses: int = 0
    activation_cost_sum: float = 0.0
    rf_attempts: int = 0
    candidate_loads_total: int = 0
    candidate_loads_evaluated: int = 0
    candidate_loads_pruned: int = 0
    minimum_selected_load: int | None = None
    maximum_selected_load: int | None = None
    maximum_population: int = 0
    action_counts: list[int] = field(
        default_factory=lambda: [0] * len(PolicyAction)
    )

    def observe(
        self,
        solution: JointRiskOracleSolution,
        outcomes: FramePacketOutcomes,
    ) -> None:
        if outcomes.pair_ids != solution.pair_ids:
            raise JointOracleEvaluationError("joint solution and outcomes are misaligned")
        actual_actions = tuple(outcome.action for outcome in outcomes.pair_outcomes)
        if actual_actions != solution.actions:
            raise JointOracleEvaluationError("rollout actions differ from the joint solution")
        actual_risk = math.fsum(
            float(value) for value in outcomes.conditional_miss_probabilities.tolist()
        )
        if not math.isclose(
            actual_risk,
            solution.total_conditional_miss_risk,
            rel_tol=0.0,
            abs_tol=_RISK_RECONCILIATION_TOLERANCE * len(solution.pair_ids),
        ):
            raise JointOracleEvaluationError(
                "realized conditional risk differs from the exact joint solution"
            )
        actual_cost = -math.fsum(float(value) for value in outcomes.rewards.tolist())
        if not math.isclose(
            actual_cost,
            solution.total_activation_cost,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise JointOracleEvaluationError(
                "realized activation cost differs from the exact joint solution"
            )
        self.frames += 1
        self.transitions += len(solution.pair_ids)
        self.usable_transitions += solution.usable_pairs
        self.forced_fallback_transitions += solution.forced_fallback_pairs
        self.conditional_risk_sum += solution.total_conditional_miss_risk
        self.usable_conditional_risk_sum += solution.usable_conditional_miss_risk
        self.forced_fallback_conditional_risk_sum += (
            solution.forced_fallback_conditional_miss_risk
        )
        self.sampled_misses += int(outcomes.sampled_miss_costs.sum())
        self.activation_cost_sum += solution.total_activation_cost
        self.rf_attempts += solution.total_rf_attempts
        self.candidate_loads_total += solution.candidate_loads_total
        self.candidate_loads_evaluated += solution.candidate_loads_evaluated
        self.candidate_loads_pruned += solution.candidate_loads_pruned
        self.minimum_selected_load = (
            solution.total_rf_attempts
            if self.minimum_selected_load is None
            else min(self.minimum_selected_load, solution.total_rf_attempts)
        )
        self.maximum_selected_load = (
            solution.total_rf_attempts
            if self.maximum_selected_load is None
            else max(self.maximum_selected_load, solution.total_rf_attempts)
        )
        self.maximum_population = max(self.maximum_population, len(solution.pair_ids))
        for index, count in enumerate(solution.action_counts):
            self.action_counts[index] += count

    def merge(self, other: _JointOracleTally) -> None:
        if not isinstance(other, _JointOracleTally):
            raise JointOracleEvaluationError("joint tally merge requires matching types")
        self.frames += other.frames
        self.transitions += other.transitions
        self.usable_transitions += other.usable_transitions
        self.forced_fallback_transitions += other.forced_fallback_transitions
        self.conditional_risk_sum += other.conditional_risk_sum
        self.usable_conditional_risk_sum += other.usable_conditional_risk_sum
        self.forced_fallback_conditional_risk_sum += (
            other.forced_fallback_conditional_risk_sum
        )
        self.sampled_misses += other.sampled_misses
        self.activation_cost_sum += other.activation_cost_sum
        self.rf_attempts += other.rf_attempts
        self.candidate_loads_total += other.candidate_loads_total
        self.candidate_loads_evaluated += other.candidate_loads_evaluated
        self.candidate_loads_pruned += other.candidate_loads_pruned
        if other.minimum_selected_load is not None:
            self.minimum_selected_load = (
                other.minimum_selected_load
                if self.minimum_selected_load is None
                else min(self.minimum_selected_load, other.minimum_selected_load)
            )
        if other.maximum_selected_load is not None:
            self.maximum_selected_load = (
                other.maximum_selected_load
                if self.maximum_selected_load is None
                else max(self.maximum_selected_load, other.maximum_selected_load)
            )
        self.maximum_population = max(self.maximum_population, other.maximum_population)
        for index, count in enumerate(other.action_counts):
            self.action_counts[index] += count

    def as_dict(self, *, density: float | None, miss_budget: float) -> dict[str, object]:
        if self.frames < 1 or self.transitions < 1:
            raise JointOracleEvaluationError("joint oracle tally is empty")
        if self.usable_transitions + self.forced_fallback_transitions != self.transitions:
            raise JointOracleEvaluationError("joint oracle transition counts do not reconcile")
        if (
            self.candidate_loads_evaluated + self.candidate_loads_pruned
            != self.candidate_loads_total
        ):
            raise JointOracleEvaluationError("joint oracle candidate counts do not reconcile")
        mean_risk = self.conditional_risk_sum / self.transitions
        return {
            "density_vehicles_per_lane_km": density,
            "frames": self.frames,
            "transitions": self.transitions,
            "usable_transitions": self.usable_transitions,
            "forced_fallback_transitions": self.forced_fallback_transitions,
            "mean_joint_oracle_conditional_miss_risk": mean_risk,
            "joint_oracle_risk_budget_multiple": mean_risk / miss_budget,
            "joint_oracle_mean_meets_budget": mean_risk <= miss_budget,
            "mean_usable_conditional_miss_risk": (
                self.usable_conditional_risk_sum / self.usable_transitions
                if self.usable_transitions
                else None
            ),
            "mean_forced_fallback_conditional_miss_risk": (
                self.forced_fallback_conditional_risk_sum
                / self.forced_fallback_transitions
                if self.forced_fallback_transitions
                else None
            ),
            "forced_fallback_risk_fraction": (
                self.forced_fallback_conditional_risk_sum / self.conditional_risk_sum
                if self.conditional_risk_sum > 0.0
                else 0.0
            ),
            "sampled_miss_rate_diagnostic_only": self.sampled_misses / self.transitions,
            "mean_activation_cost": self.activation_cost_sum / self.transitions,
            "mean_rf_attempts_per_pair": self.rf_attempts / self.transitions,
            "selected_frame_load": {
                "mean": self.rf_attempts / self.frames,
                "minimum": self.minimum_selected_load,
                "maximum": self.maximum_selected_load,
            },
            "maximum_population": self.maximum_population,
            "candidate_loads": {
                "total": self.candidate_loads_total,
                "evaluated": self.candidate_loads_evaluated,
                "pruned_by_monotone_lower_bound": self.candidate_loads_pruned,
                "evaluated_fraction": (
                    self.candidate_loads_evaluated / self.candidate_loads_total
                ),
            },
            "action_counts": {
                name: self.action_counts[index]
                for index, name in enumerate(POLICY_ACTION_ORDER)
            },
            "action_fractions": {
                name: self.action_counts[index] / self.transitions
                for index, name in enumerate(POLICY_ACTION_ORDER)
            },
        }


@dataclass(slots=True)
class _JointOraclePolicy:
    tallies: dict[float, _JointOracleTally] = field(default_factory=dict)
    name: str = "exact-population-joint-risk-oracle"
    requires_oracle_truth: bool = True
    _pending: dict[int, JointRiskOracleSolution] = field(default_factory=dict, init=False)
    _empty_pending: set[int] = field(default_factory=set, init=False)

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise JointOracleEvaluationError("joint oracle requires isolated channel truth")
        if decision.frame.source.split != "validation":
            raise JointOracleEvaluationError("joint oracle evaluation accepts validation only")
        if (
            decision.frame.index in self._pending
            or decision.frame.index in self._empty_pending
        ):
            raise JointOracleEvaluationError("joint oracle frame is already pending")
        if decision.population_size == 0:
            self._empty_pending.add(decision.frame.index)
            return ()
        solution = solve_joint_risk_floor(
            JointRiskOracleProblem.from_decision(decision, channel_truth)
        )
        self._pending[decision.frame.index] = solution
        return tuple(
            action if actor_row.usable else None
            for action, actor_row in zip(
                solution.actions,
                decision.actor_frame.rows,
                strict=True,
            )
        )

    def observe_frame(
        self,
        *,
        decision: PopulationPolicyFrame,
        actions: tuple[PolicyAction, ...],
        outcomes: FramePacketOutcomes,
        boundary: FrameReturnBoundary,
        final_observation: Mapping[str, FrameObservation],
    ) -> None:
        del boundary, final_observation
        if decision.frame.index in self._empty_pending:
            self._empty_pending.remove(decision.frame.index)
            if actions or outcomes.pair_ids:
                raise JointOracleEvaluationError(
                    "empty joint-oracle frame produced actions or outcomes"
                )
            return
        try:
            solution = self._pending.pop(decision.frame.index)
        except KeyError as error:
            raise JointOracleEvaluationError(
                "completed joint-oracle frame has no pending solution"
            ) from error
        if actions != solution.actions:
            raise JointOracleEvaluationError("completed actions differ from joint solution")
        self.tallies.setdefault(
            decision.frame.source.density,
            _JointOracleTally(),
        ).observe(solution, outcomes)

    def assert_complete(self) -> None:
        if self._pending or self._empty_pending:
            raise JointOracleEvaluationError("joint oracle evaluation ended with pending frames")


@dataclass(frozen=True, slots=True)
class JointOracleEvaluationReport:
    config_hash: str
    policy_environment_scope_hash: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_policy_seed: int
    checkpoint_completed_iterations: int
    normalization_training_rows: int
    miss_budget: float
    audit_path: Path
    audit_sha256: str
    windows: tuple[EvaluationWindow, ...]
    densities: tuple[dict[str, object], ...]
    campaign: dict[str, object]
    generated_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        all_pass = all(
            row["joint_oracle_mean_meets_budget"] is True for row in self.densities
        )
        return {
            "schema": JOINT_ORACLE_EVALUATION_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "scope": "exact population-joint conditional-risk floor on frozen validation windows",
            "test_split_opened": False,
            "non_deployable_oracle_truth": True,
            "optimization_method": JOINT_RISK_ORACLE_METHOD,
            "exactness_boundary": (
                "exact for the current per-frame nine-action model because RF contention "
                "depends only on aggregate offered attempts and fixed-load action risk has "
                "diminishing marginal reductions"
            ),
            "policy_environment_scope_hash": self.policy_environment_scope_hash,
            "config_hash": self.config_hash,
            "reliability_miss_budget": self.miss_budget,
            "checkpoint_normalization_source": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
                "policy_seed": self.checkpoint_policy_seed,
                "completed_iterations": self.checkpoint_completed_iterations,
                "normalization_training_rows": self.normalization_training_rows,
                "actor_used": False,
            },
            "window_source": {
                "path": str(self.audit_path),
                "sha256": self.audit_sha256,
                "schema": STATE_REGIME_AUDIT_SCHEMA,
            },
            "sampled_windows": [window.as_dict() for window in self.windows],
            "densities": list(self.densities),
            "campaign": self.campaign,
            "decision": {
                "all_densities_meet_joint_oracle_floor": all_pass,
                "coordination_aware_recovery_authorized": all_pass,
                "standard_independent_ppo_recovery_authorized": False,
                "next_action": (
                    "design a coordination-aware learner and joint-oracle pretraining labels"
                    if all_pass
                    else "revise the action/resource or physical system before further PPO training"
                ),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target


def _numpy_state_equal(left: object, right: object) -> bool:
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return False
    return bool(
        len(left) == len(right) == 5
        and left[0] == right[0]
        and isinstance(left[1], np.ndarray)
        and isinstance(right[1], np.ndarray)
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def build_joint_oracle_evaluation(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    state_regime_audit_path: str | Path,
) -> JointOracleEvaluationReport:
    """Evaluate the exact joint risk floor on frozen validation windows only."""

    if not isinstance(config, ProjectConfig):
        raise JointOracleEvaluationError("joint oracle evaluation requires ProjectConfig")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    audit = Path(state_regime_audit_path).expanduser().resolve()
    digest = config_hash(config)
    environment_digest = scope_hash(config, "policy_environment")
    _, windows, environment_seed, audit_sha256 = load_frozen_regime_audit(
        audit,
        expected_policy_environment_scope_hash=environment_digest,
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    restored = restore_training_checkpoint(
        checkpoint,
        config=config,
        restore_global_rng=False,
    )
    normalization_training_rows = restored.normalizer.training_rows
    frozen_normalization: ObservationNormalizationState = restored.normalizer.freeze()
    policy = _JointOraclePolicy()
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation_sources = {
        source.trace_id: source for source in catalog.for_split("validation")
    }
    for window in windows:
        try:
            source: FrameTraceSource = validation_sources[window.trace_id]
        except KeyError as error:
            raise JointOracleEvaluationError(
                "audit validation window is absent from the configured catalog"
            ) from error
        if source.density != window.density:
            raise JointOracleEvaluationError("audit window density differs from trace catalog")
        result = run_policy_rollout_with_state(
            config,
            source,
            policy=policy,
            environment_seed=environment_seed,
            policy_seed=restored.policy_seed,
            start_frame_index=window.start_frame_index,
            max_frames=window.frames,
            normalization_state=frozen_normalization,
            frame_observer=policy,
        )
        if result.normalization_state != frozen_normalization:
            raise JointOracleEvaluationError("frozen evaluation normalization state changed")
    policy.assert_complete()
    if (
        random.getstate() != python_rng
        or not _numpy_state_equal(np.random.get_state(), numpy_rng)
        or not torch.equal(torch.random.get_rng_state(), torch_rng)
    ):
        raise JointOracleEvaluationError("joint oracle evaluation mutated global RNG state")
    expected_densities = tuple(sorted(set(window.density for window in windows)))
    if set(policy.tallies) != set(expected_densities):
        raise JointOracleEvaluationError("joint oracle density coverage is incomplete")
    density_rows = tuple(
        policy.tallies[density].as_dict(
            density=density,
            miss_budget=config.service.miss_budget,
        )
        for density in expected_densities
    )
    campaign_tally = _JointOracleTally()
    for density in expected_densities:
        campaign_tally.merge(policy.tallies[density])
    campaign = campaign_tally.as_dict(
        density=None,
        miss_budget=config.service.miss_budget,
    )
    return JointOracleEvaluationReport(
        config_hash=digest,
        policy_environment_scope_hash=environment_digest,
        checkpoint_path=checkpoint,
        checkpoint_sha256=restored.sha256,
        checkpoint_policy_seed=restored.policy_seed,
        checkpoint_completed_iterations=restored.counters.completed_iterations,
        normalization_training_rows=normalization_training_rows,
        miss_budget=config.service.miss_budget,
        audit_path=audit,
        audit_sha256=audit_sha256,
        windows=windows,
        densities=density_rows,
        campaign=campaign,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "JOINT_ORACLE_EVALUATION_SCHEMA",
    "JointOracleEvaluationError",
    "JointOracleEvaluationReport",
    "build_joint_oracle_evaluation",
]
