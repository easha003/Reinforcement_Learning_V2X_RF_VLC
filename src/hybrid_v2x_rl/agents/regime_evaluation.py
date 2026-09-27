"""Bounded, truth-isolated regime evaluation for a frozen PPO checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch

from hybrid_v2x_rl.agents.checkpointing import restore_training_checkpoint
from hybrid_v2x_rl.agents.masked_categorical import (
    ACTION_COUNT,
    MaskedCategorical,
    SharedCategoricalActor,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout_with_state
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource, TraceCatalog
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizationState
from hybrid_v2x_rl.mean_field.packet_outcomes import FramePacketOutcomes
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PolicyProposal,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.return_boundaries import FrameReturnBoundary
from hybrid_v2x_rl.mean_field.state_regime_audit import (
    LOAD_PROFILES,
    REGIME_NAMES,
    STATE_REGIME_AUDIT_SCHEMA,
    CounterfactualActionSet,
    RegimeName,
    RegimeThresholds,
    assess_counterfactual_actions,
    assess_counterfactual_actions_at_load,
    classify_regimes,
)

PPO_REGIME_EVALUATION_SCHEMA: Final = "hybrid-rf-vlc-rl.ppo-regime-evaluation.v1"
POLICY_INDUCED_LOAD: Final = "policy_induced_load"
_RISK_TOLERANCE: Final = 2e-6


class PPORegimeEvaluationError(HybridV2XError):
    """A bounded PPO regime-evaluation input or invariant is invalid."""


def _profile_tallies() -> dict[str, _ProfileTally]:
    return {
        POLICY_INDUCED_LOAD: _ProfileTally(),
        **{str(name): _ProfileTally() for name in LOAD_PROFILES},
    }


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """One validation window inherited exactly from the causal coverage audit."""

    trace_id: str
    density: float
    start_frame_index: int
    frames: int

    def __post_init__(self) -> None:
        if not self.trace_id or not math.isfinite(self.density) or self.density <= 0.0:
            raise PPORegimeEvaluationError("evaluation window identity is invalid")
        if self.start_frame_index < 0 or self.frames < 1:
            raise PPORegimeEvaluationError("evaluation window bounds are invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "split": "validation",
            "density_vehicles_per_lane_km": self.density,
            "start_frame_index": self.start_frame_index,
            "frames": self.frames,
        }


@dataclass(slots=True)
class _ProfileTally:
    rows: int = 0
    feasible_rows: int = 0
    feasible_probability_mass_sum: float = 0.0
    expected_miss_risk_sum: float = 0.0
    selected_miss_risk_sum: float = 0.0
    expected_resource_cost_sum: float = 0.0
    feasible_conditioned_resource_regret_sum: float = 0.0
    feasible_conditioned_resource_regret_rows: int = 0
    selected_feasible_resource_regret_sum: float = 0.0
    selected_feasible_resource_regret_rows: int = 0
    oracle_actions: Counter[str] = field(default_factory=Counter)
    feasible_actions: Counter[str] = field(default_factory=Counter)

    def observe(
        self,
        *,
        probabilities: tuple[float, ...],
        selected_action: PolicyAction,
        assessment: CounterfactualActionSet,
    ) -> None:
        by_action = {row.action: row for row in assessment.actions}
        if set(by_action) != set(PolicyAction):
            # Hardware masks are permitted, but masked actions must have exactly
            # zero actor mass and are deliberately absent from the assessment.
            positive = {
                PolicyAction(index)
                for index, probability in enumerate(probabilities)
                if probability > 0.0
            }
            if not positive.issubset(by_action):
                raise PPORegimeEvaluationError(
                    "policy assigned mass to an action absent from its counterfactual set"
                )
        selected = by_action.get(selected_action)
        if selected is None:
            raise PPORegimeEvaluationError("deterministic PPO action is hardware-masked")
        self.rows += 1
        self.feasible_rows += int(assessment.any_feasible)
        self.oracle_actions[assessment.selected_action.label] += 1
        feasible = tuple(row for row in assessment.actions if row.feasible)
        for row in feasible:
            self.feasible_actions[row.action.label] += 1
        feasible_mass = math.fsum(probabilities[int(row.action)] for row in feasible)
        expected_risk = math.fsum(
            probabilities[int(row.action)] * row.conditional_miss_probability
            for row in assessment.actions
        )
        expected_cost = math.fsum(
            probabilities[int(row.action)] * row.activation_cost
            for row in assessment.actions
        )
        self.feasible_probability_mass_sum += feasible_mass
        self.expected_miss_risk_sum += expected_risk
        self.selected_miss_risk_sum += selected.conditional_miss_probability
        self.expected_resource_cost_sum += expected_cost
        if feasible:
            minimum_cost = min(row.activation_cost for row in feasible)
            if feasible_mass > 0.0:
                conditioned_cost = math.fsum(
                    probabilities[int(row.action)] * row.activation_cost for row in feasible
                ) / feasible_mass
                regret = max(0.0, conditioned_cost - minimum_cost)
                self.feasible_conditioned_resource_regret_sum += regret
                self.feasible_conditioned_resource_regret_rows += 1
            if selected.feasible:
                self.selected_feasible_resource_regret_sum += max(
                    0.0,
                    selected.activation_cost - minimum_cost,
                )
                self.selected_feasible_resource_regret_rows += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "any_feasible_fraction": _mean(float(self.feasible_rows), self.rows),
            "mean_feasible_action_probability_mass": _mean(
                self.feasible_probability_mass_sum,
                self.rows,
            ),
            "mean_infeasible_action_probability_mass": (
                _mean(self.rows - self.feasible_probability_mass_sum, self.rows)
            ),
            "mean_policy_expected_conditional_miss_risk": _mean(
                self.expected_miss_risk_sum,
                self.rows,
            ),
            "mean_deterministic_selected_conditional_miss_risk": _mean(
                self.selected_miss_risk_sum,
                self.rows,
            ),
            "mean_policy_expected_resource_cost": _mean(
                self.expected_resource_cost_sum,
                self.rows,
            ),
            "mean_feasible_conditioned_resource_regret": _mean(
                self.feasible_conditioned_resource_regret_sum,
                self.feasible_conditioned_resource_regret_rows,
            ),
            "feasible_conditioned_resource_regret_rows": (
                self.feasible_conditioned_resource_regret_rows
            ),
            "mean_deterministic_selected_resource_regret_when_feasible": _mean(
                self.selected_feasible_resource_regret_sum,
                self.selected_feasible_resource_regret_rows,
            ),
            "deterministic_selected_feasible_fraction": _mean(
                float(self.selected_feasible_resource_regret_rows),
                self.rows,
            ),
            "oracle_action_counts": {
                action: int(self.oracle_actions[action]) for action in POLICY_ACTION_ORDER
            },
            "feasible_action_rows": {
                action: int(self.feasible_actions[action]) for action in POLICY_ACTION_ORDER
            },
        }


@dataclass(slots=True)
class _RegimeTally:
    rows: int = 0
    actual_rows: int = 0
    traces: set[str] = field(default_factory=set)
    clusters: set[str] = field(default_factory=set)
    probability_sums: list[float] = field(
        default_factory=lambda: [0.0] * ACTION_COUNT
    )
    selected_actions: Counter[str] = field(default_factory=Counter)
    actual_conditional_risk_sum: float = 0.0
    actual_sampled_miss_sum: int = 0
    actual_resource_cost_sum: float = 0.0
    actual_selected_feasible: int = 0
    profiles: dict[str, _ProfileTally] = field(default_factory=_profile_tallies)

    def observe_policy(
        self,
        *,
        trace_id: str,
        pair_id: str,
        probabilities: tuple[float, ...],
        selected_action: PolicyAction,
        assessments: Mapping[str, CounterfactualActionSet],
    ) -> None:
        self.rows += 1
        self.traces.add(trace_id)
        self.clusters.add(f"{trace_id}/{pair_id}")
        for index, probability in enumerate(probabilities):
            self.probability_sums[index] += probability
        self.selected_actions[selected_action.label] += 1
        if set(assessments) != set(self.profiles):
            raise PPORegimeEvaluationError("counterfactual profile set is incomplete")
        for name, assessment in assessments.items():
            self.profiles[name].observe(
                probabilities=probabilities,
                selected_action=selected_action,
                assessment=assessment,
            )

    def observe_actual(
        self,
        *,
        conditional_risk: float,
        sampled_miss: int,
        resource_cost: float,
        feasible: bool,
    ) -> None:
        self.actual_rows += 1
        self.actual_conditional_risk_sum += conditional_risk
        self.actual_sampled_miss_sum += sampled_miss
        self.actual_resource_cost_sum += resource_cost
        self.actual_selected_feasible += int(feasible)

    def as_dict(self, *, regime: RegimeName, density: float | None) -> dict[str, object]:
        if self.actual_rows != self.rows:
            raise PPORegimeEvaluationError(
                "policy and completed-outcome regime rows do not reconcile"
            )
        probabilities = tuple(_mean(value, self.rows) for value in self.probability_sums)
        return {
            "split": "validation",
            "density_vehicles_per_lane_km": density,
            "regime": regime,
            "rows": self.rows,
            "pair_episode_clusters": len(self.clusters),
            "trace_count": len(self.traces),
            "trace_ids": sorted(self.traces),
            "mean_action_probabilities": {
                action.label: probabilities[int(action)] for action in PolicyAction
            },
            "deterministic_selected_action_counts": {
                action: int(self.selected_actions[action]) for action in POLICY_ACTION_ORDER
            },
            "actual_policy_load": {
                "mean_selected_conditional_miss_risk": _mean(
                    self.actual_conditional_risk_sum,
                    self.actual_rows,
                ),
                "sampled_miss_rate": _mean(
                    float(self.actual_sampled_miss_sum),
                    self.actual_rows,
                ),
                "mean_selected_resource_cost": _mean(
                    self.actual_resource_cost_sum,
                    self.actual_rows,
                ),
                "selected_feasible_fraction": _mean(
                    float(self.actual_selected_feasible),
                    self.actual_rows,
                ),
            },
            "counterfactuals": {
                name: self.profiles[name].as_dict()
                for name in (POLICY_INDUCED_LOAD, *LOAD_PROFILES)
            },
        }


@dataclass(frozen=True, slots=True)
class _PendingRow:
    labels: tuple[RegimeName, ...]
    density: float
    selected_action: PolicyAction
    selected_risk: float
    selected_resource_cost: float
    miss_budget: float


@dataclass(slots=True)
class RegimeEvaluationAccumulator:
    """Accumulate overlapping regime metrics at density and campaign levels."""

    _cells: dict[tuple[float, RegimeName], _RegimeTally] = field(
        default_factory=dict,
        init=False,
    )
    _campaign: dict[RegimeName, _RegimeTally] = field(default_factory=dict, init=False)
    _usable: Counter[float] = field(default_factory=Counter, init=False)
    _unusable: Counter[float] = field(default_factory=Counter, init=False)
    _unclassified: Counter[float] = field(default_factory=Counter, init=False)

    def observe_unusable(self, density: float) -> None:
        self._unusable[density] += 1

    def observe_policy(
        self,
        *,
        trace_id: str,
        pair_id: str,
        density: float,
        labels: tuple[RegimeName, ...],
        probabilities: tuple[float, ...],
        selected_action: PolicyAction,
        assessments: Mapping[str, CounterfactualActionSet],
    ) -> None:
        if len(probabilities) != ACTION_COUNT or not math.isclose(
            math.fsum(probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise PPORegimeEvaluationError("policy probabilities must sum to one")
        self._usable[density] += 1
        if not labels:
            self._unclassified[density] += 1
            return
        for label in labels:
            for tally in (
                self._cells.setdefault((density, label), _RegimeTally()),
                self._campaign.setdefault(label, _RegimeTally()),
            ):
                tally.observe_policy(
                    trace_id=trace_id,
                    pair_id=pair_id,
                    probabilities=probabilities,
                    selected_action=selected_action,
                    assessments=assessments,
                )

    def observe_actual(
        self,
        pending: _PendingRow,
        *,
        conditional_risk: float,
        sampled_miss: int,
    ) -> None:
        if not math.isclose(
            conditional_risk,
            pending.selected_risk,
            rel_tol=0.0,
            abs_tol=_RISK_TOLERANCE,
        ):
            raise PPORegimeEvaluationError(
                "actual selected risk differs from the policy-load counterfactual"
            )
        for label in pending.labels:
            for tally in (self._cells[(pending.density, label)], self._campaign[label]):
                tally.observe_actual(
                    conditional_risk=conditional_risk,
                    sampled_miss=sampled_miss,
                    resource_cost=pending.selected_resource_cost,
                    feasible=conditional_risk <= pending.miss_budget,
                )

    def density_rows(self, densities: Sequence[float]) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for density in sorted(set(densities)):
            for regime in REGIME_NAMES:
                payload = self._cells.get((density, regime), _RegimeTally()).as_dict(
                    regime=regime,
                    density=density,
                )
                payload["density_usable_rows"] = self._usable[density]
                payload["density_unusable_rows"] = self._unusable[density]
                payload["density_unclassified_rows"] = self._unclassified[density]
                rows.append(payload)
        return tuple(rows)

    def campaign_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            self._campaign.get(regime, _RegimeTally()).as_dict(
                regime=regime,
                density=None,
            )
            for regime in REGIME_NAMES
        )


@dataclass(slots=True)
class _FrozenPPORegimePolicy:
    actor: SharedCategoricalActor
    thresholds: RegimeThresholds
    accumulator: RegimeEvaluationAccumulator
    name: str = "frozen-ppo-regime-evaluation"
    requires_oracle_truth: bool = True
    _pending: dict[int, dict[str, _PendingRow]] = field(default_factory=dict, init=False)

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise PPORegimeEvaluationError(
                "regime scoring requires truth isolated from actor inputs"
            )
        if decision.frame.source.split != "validation":
            raise PPORegimeEvaluationError("PPO regime evaluation accepts validation only")
        if decision.frame.index in self._pending:
            raise PPORegimeEvaluationError("a policy frame is already pending")
        observations = torch.tensor(
            decision.observation.actor_observations,
            dtype=torch.float32,
        )
        masks = torch.tensor(decision.observation.action_masks, dtype=torch.bool)
        usable_indices = [
            index for index, row in enumerate(decision.actor_frame.rows) if row.usable
        ]
        selected_by_row: dict[int, PolicyAction] = {}
        probabilities_by_row: dict[int, tuple[float, ...]] = {}
        if usable_indices:
            indices = torch.tensor(usable_indices, dtype=torch.long)
            with torch.inference_mode():
                distribution = MaskedCategorical(
                    self.actor(observations[indices]),
                    masks[indices],
                )
                selected = distribution.select(deterministic=True)
                probability_tensor = distribution.probabilities
            for row, action_index, probability_row in zip(
                usable_indices,
                selected.actions.tolist(),
                probability_tensor.tolist(),
                strict=True,
            ):
                selected_by_row[row] = PolicyAction(int(action_index))
                probabilities_by_row[row] = tuple(float(value) for value in probability_row)

        proposals: list[PolicyProposal] = []
        effective_actions: list[PolicyAction] = []
        for index, actor_row in enumerate(decision.actor_frame.rows):
            proposal = selected_by_row.get(index)
            proposals.append(proposal)
            effective_actions.append(
                decision.action_space.select(
                    proposal,
                    observation_usable=actor_row.usable,
                )
            )
        total_rf_attempts = sum(
            action_resources(action).reserved_rf_attempts for action in effective_actions
        )

        pending: dict[str, _PendingRow] = {}
        for index in usable_indices:
            pair = decision.frame.pairs[index]
            actor_row = decision.actor_frame.rows[index]
            if actor_row.values is None:
                raise PPORegimeEvaluationError("usable actor row has no raw values")
            selected_action = selected_by_row[index]
            row_probabilities = probabilities_by_row[index]
            truth = channel_truth[pair.pair_id]
            rf_failure = truth.rf_propagation.decoding_failure_probability
            vlc_failure = truth.vlc_result.total_failure_probability
            own_attempts = action_resources(selected_action).reserved_rf_attempts
            induced = assess_counterfactual_actions_at_load(
                decision,
                rf_decoding_failure_probability=rf_failure,
                vlc_failure_probability=vlc_failure,
                other_pair_rf_attempts=total_rf_attempts - own_attempts,
            )
            assessments: dict[str, CounterfactualActionSet] = {
                POLICY_INDUCED_LOAD: induced,
                **{
                    name: assess_counterfactual_actions(
                        decision,
                        rf_decoding_failure_probability=rf_failure,
                        vlc_failure_probability=vlc_failure,
                        load_profile=name,
                    )
                    for name in LOAD_PROFILES
                },
            }
            labels = classify_regimes(
                actor_row.values,
                decision.columns,
                self.thresholds,
            )
            selected_row = next(
                row for row in induced.actions if row.action is selected_action
            )
            self.accumulator.observe_policy(
                trace_id=decision.frame.trace_id,
                pair_id=pair.pair_id,
                density=decision.frame.source.density,
                labels=labels,
                probabilities=row_probabilities,
                selected_action=selected_action,
                assessments=assessments,
            )
            if labels:
                pending[pair.pair_id] = _PendingRow(
                    labels=labels,
                    density=decision.frame.source.density,
                    selected_action=selected_action,
                    selected_risk=selected_row.conditional_miss_probability,
                    selected_resource_cost=selected_row.activation_cost,
                    miss_budget=decision.miss_budget,
                )
        for actor_row in decision.actor_frame.rows:
            if not actor_row.usable:
                self.accumulator.observe_unusable(decision.frame.source.density)
        self._pending[decision.frame.index] = pending
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
        del boundary, final_observation
        try:
            pending = self._pending.pop(decision.frame.index)
        except KeyError as error:
            raise PPORegimeEvaluationError("a completed frame has no policy diagnostics") from error
        if outcomes.pair_ids != decision.frame.active_pair_ids or len(actions) != len(
            outcomes.pair_ids
        ):
            raise PPORegimeEvaluationError("completed outcomes do not align with policy rows")
        for pair_id, action, conditional_risk, sampled_miss in zip(
            outcomes.pair_ids,
            actions,
            outcomes.conditional_miss_probabilities.tolist(),
            outcomes.sampled_miss_costs.tolist(),
            strict=True,
        ):
            row = pending.get(pair_id)
            if row is None:
                continue
            if action is not row.selected_action:
                raise PPORegimeEvaluationError("completed action differs from PPO argmax")
            self.accumulator.observe_actual(
                row,
                conditional_risk=float(conditional_risk),
                sampled_miss=int(sampled_miss),
            )

    def assert_complete(self) -> None:
        if self._pending:
            raise PPORegimeEvaluationError("evaluation ended with pending policy frames")


@dataclass(frozen=True, slots=True)
class PPORegimeEvaluationReport:
    config_hash: str
    policy_environment_scope_hash: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_policy_seed: int
    checkpoint_completed_iterations: int
    checkpoint_environment_transitions: int
    normalization_training_rows: int
    miss_budget: float
    audit_path: Path
    audit_sha256: str
    thresholds: RegimeThresholds
    windows: tuple[EvaluationWindow, ...]
    density_rows: tuple[dict[str, object], ...]
    campaign_rows: tuple[dict[str, object], ...]
    generated_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PPO_REGIME_EVALUATION_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "scope": "bounded deterministic validation evaluation of a frozen PPO checkpoint",
            "config_hash": self.config_hash,
            "policy_environment_scope_hash": self.policy_environment_scope_hash,
            "test_split_opened": False,
            "action_selection": "deterministic masked argmax",
            "reliability_miss_budget": self.miss_budget,
            "normalization": {
                "fit_split": "train",
                "training_rows": self.normalization_training_rows,
                "frozen_for_evaluation": True,
            },
            "policy_truth_boundary": (
                "actor probabilities and actions use normalized causal observations only; "
                "simulator truth is consumed afterward by the non-deployable evaluator"
            ),
            "coverage_claim": {
                "level": "campaign",
                "density_conditioned_support_reported": True,
                "every_regime_at_every_density_claimed": False,
            },
            "checkpoint": {
                "path": str(self.checkpoint_path),
                "sha256": self.checkpoint_sha256,
                "policy_seed": self.checkpoint_policy_seed,
                "completed_iterations": self.checkpoint_completed_iterations,
                "environment_transitions": self.checkpoint_environment_transitions,
            },
            "threshold_source": {
                "path": str(self.audit_path),
                "sha256": self.audit_sha256,
                "schema": STATE_REGIME_AUDIT_SCHEMA,
                "fit_split": "train",
            },
            "thresholds": self.thresholds.as_dict(),
            "sampled_windows": [window.as_dict() for window in self.windows],
            "campaign_regimes": list(self.campaign_rows),
            "density_conditioned_regimes": list(self.density_rows),
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


def _load_audit(
    path: Path,
    *,
    expected_policy_environment_scope_hash: str,
) -> tuple[RegimeThresholds, tuple[EvaluationWindow, ...], int, str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise PPORegimeEvaluationError(
            "cannot load the frozen state-regime audit",
            artifact_path=path,
        ) from error
    if not isinstance(payload, Mapping):
        raise PPORegimeEvaluationError("state-regime audit must contain a mapping")
    if payload.get("schema") != STATE_REGIME_AUDIT_SCHEMA:
        raise PPORegimeEvaluationError("state-regime audit schema is not frozen v3")
    audit_scope_hash = payload.get("policy_environment_scope_hash")
    if audit_scope_hash != expected_policy_environment_scope_hash:
        raise PPORegimeEvaluationError(
            "state-regime audit policy-environment scope differs from evaluation"
        )
    if payload.get("test_split_opened") is not False:
        raise PPORegimeEvaluationError("state-regime audit does not prove test isolation")
    claim = payload.get("coverage_claim")
    if not isinstance(claim, Mapping) or (
        claim.get("level") != "campaign"
        or claim.get("density_conditioned_support_reported") is not True
        or claim.get("every_regime_at_every_density_claimed") is not False
    ):
        raise PPORegimeEvaluationError("state-regime coverage claim is not frozen")
    thresholds_raw = payload.get("thresholds")
    if not isinstance(thresholds_raw, Mapping):
        raise PPORegimeEvaluationError("state-regime audit has no threshold mapping")
    thresholds = RegimeThresholds.from_dict(thresholds_raw)
    environment_seed = payload.get("environment_seed")
    if not isinstance(environment_seed, int) or isinstance(environment_seed, bool):
        raise PPORegimeEvaluationError("state-regime audit environment seed is invalid")
    windows_raw = payload.get("sampled_windows")
    if not isinstance(windows_raw, list):
        raise PPORegimeEvaluationError("state-regime audit sampled windows are invalid")
    windows: list[EvaluationWindow] = []
    for item in windows_raw:
        if not isinstance(item, Mapping) or item.get("split") != "validation":
            continue
        try:
            windows.append(
                EvaluationWindow(
                    trace_id=cast(str, item["trace_id"]),
                    density=float(item["density_vehicles_per_lane_km"]),
                    start_frame_index=int(item["start_frame_index"]),
                    frames=int(item["frames"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PPORegimeEvaluationError(
                "state-regime validation window is invalid"
            ) from error
    if not windows:
        raise PPORegimeEvaluationError("state-regime audit has no validation windows")
    if len({(row.trace_id, row.start_frame_index, row.frames) for row in windows}) != len(
        windows
    ):
        raise PPORegimeEvaluationError("state-regime validation windows are duplicated")
    return thresholds, tuple(windows), environment_seed, hashlib.sha256(raw).hexdigest()


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


def build_ppo_regime_evaluation(
    config: ProjectConfig,
    *,
    checkpoint_path: str | Path,
    state_regime_audit_path: str | Path,
) -> PPORegimeEvaluationReport:
    """Evaluate a frozen PPO actor on the audit's bounded validation windows."""

    if not isinstance(config, ProjectConfig):
        raise PPORegimeEvaluationError("PPO regime evaluation requires ProjectConfig")
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    audit = Path(state_regime_audit_path).expanduser().resolve(strict=True)
    digest = config_hash(config)
    policy_environment_digest = scope_hash(config, "policy_environment")
    thresholds, windows, environment_seed, audit_sha256 = _load_audit(
        audit,
        expected_policy_environment_scope_hash=policy_environment_digest,
    )
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()
    restored = restore_training_checkpoint(
        checkpoint,
        config=config,
        restore_global_rng=False,
    )
    actor_state = {
        name: value.detach().clone() for name, value in restored.updater.actor.state_dict().items()
    }
    normalization_training_rows = restored.normalizer.training_rows
    frozen_normalization: ObservationNormalizationState = restored.normalizer.freeze()
    accumulator = RegimeEvaluationAccumulator()
    policy = _FrozenPPORegimePolicy(
        actor=restored.updater.actor,
        thresholds=thresholds,
        accumulator=accumulator,
    )
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation_sources = {source.trace_id: source for source in catalog.for_split("validation")}
    for window in windows:
        try:
            source: FrameTraceSource = validation_sources[window.trace_id]
        except KeyError as error:
            raise PPORegimeEvaluationError(
                "audit validation window is absent from the configured catalog"
            ) from error
        if source.density != window.density:
            raise PPORegimeEvaluationError("audit window density differs from trace catalog")
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
            raise PPORegimeEvaluationError("frozen evaluation normalization state changed")
    policy.assert_complete()
    for name, value in restored.updater.actor.state_dict().items():
        if not torch.equal(value, actor_state[name]):
            raise PPORegimeEvaluationError("regime evaluation mutated actor parameters")
    if (
        random.getstate() != python_rng
        or not _numpy_state_equal(np.random.get_state(), numpy_rng)
        or not torch.equal(torch.random.get_rng_state(), torch_rng)
    ):
        raise PPORegimeEvaluationError("regime evaluation mutated global RNG state")
    densities = tuple(window.density for window in windows)
    return PPORegimeEvaluationReport(
        config_hash=digest,
        policy_environment_scope_hash=policy_environment_digest,
        checkpoint_path=checkpoint,
        checkpoint_sha256=restored.sha256,
        checkpoint_policy_seed=restored.policy_seed,
        checkpoint_completed_iterations=restored.counters.completed_iterations,
        checkpoint_environment_transitions=restored.counters.environment_transitions,
        normalization_training_rows=normalization_training_rows,
        miss_budget=config.service.miss_budget,
        audit_path=audit,
        audit_sha256=audit_sha256,
        thresholds=thresholds,
        windows=windows,
        density_rows=accumulator.density_rows(densities),
        campaign_rows=accumulator.campaign_rows(),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "POLICY_INDUCED_LOAD",
    "PPO_REGIME_EVALUATION_SCHEMA",
    "EvaluationWindow",
    "PPORegimeEvaluationError",
    "PPORegimeEvaluationReport",
    "RegimeEvaluationAccumulator",
    "build_ppo_regime_evaluation",
]
