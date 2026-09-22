"""Phase 6 deployable baselines and the explicitly non-deployable oracle.

The policies in this module only choose actions.  They do not evaluate packets,
sample outcomes, or maintain a second resource model.  Their proposals are
validated by the common action-mask boundary and then pass through the same
``FrameActionLedger``, RF pool, matched random tapes, physical channels,
feedback, and lifecycle code used by random policies and, later, PPO.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PolicyProposal,
    PopulationPolicy,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolResponse

if TYPE_CHECKING:
    from hybrid_v2x_rl.config.models import ProjectConfig
    from hybrid_v2x_rl.mean_field.deterministic_rollout import DeterministicRolloutReport
    from hybrid_v2x_rl.mean_field.frames import FrameTraceSource

BASELINE_ALWAYS_VLC: Final = "always-vlc"
BASELINE_DUPLICATE_ALL: Final = "duplicate-all"
BASELINE_GEOMETRY_THRESHOLD: Final = "geometry-threshold"
BASELINE_CONTEXTUAL: Final = "contextual-no-history"
BASELINE_SUPERVISED: Final = "supervised-risk-allocation"
BASELINE_ORACLE: Final = "truth-risk-oracle"
OPTICAL_ESTIMATOR_SCHEMA: Final = "hybrid-rf-vlc-rl.supervised-optical-risk.v1"
BASELINE_ALWAYS_RF: Final = tuple(f"always-rf-{attempts}" for attempts in range(1, 5))
BASELINE_NAMES: Final = (
    *BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_DUPLICATE_ALL,
    BASELINE_GEOMETRY_THRESHOLD,
    BASELINE_CONTEXTUAL,
    BASELINE_SUPERVISED,
    BASELINE_ORACLE,
)

# This comparator may use current geometry, a current local CBR measurement,
# and a causal forecast, but none of the action-dependent link or outcome
# history that turns the full problem into an MDP.
CONTEXTUAL_FEATURES: Final = (
    "rf_channel_busy_ratio",
    "neighbor_count",
    "pair_distance",
    "pair_bearing",
    "relative_speed",
    "heading_difference",
    "optical_fov_margin",
    "distance_to_junction",
    "path_spans_junction",
    "predicted_blockage_probability",
    "predictor_confidence",
    "track_age",
)

OPTICAL_ESTIMATOR_FEATURES: Final = (
    "neighbor_count",
    "pair_distance",
    "pair_bearing",
    "relative_speed",
    "heading_difference",
    "optical_fov_margin",
    "distance_to_junction",
    "path_spans_junction",
    "predicted_blockage_probability",
    "predictor_confidence",
    "track_age",
)


class BaselinePolicyError(HybridV2XError):
    """A baseline definition, fit, or joint decision is invalid."""


def _column_map(columns: Sequence[str], required: Sequence[str]) -> dict[str, int]:
    indices = {name: index for index, name in enumerate(columns)}
    missing = tuple(name for name in required if name not in indices)
    if missing:
        raise BaselinePolicyError(
            f"actor schema is missing baseline features: {', '.join(missing)}"
        )
    return indices


def _raw_matrix(decision: PopulationPolicyFrame) -> NDArray[np.float64]:
    """Return raw causal rows with zero placeholders only for unusable pairs."""

    values = np.zeros((decision.population_size, len(decision.columns)), dtype=np.float64)
    for index, row in enumerate(decision.actor_frame.rows):
        if row.values is not None:
            values[index] = row.values
    return values


def _largest_allowed(
    decision: PopulationPolicyFrame,
    *,
    uses_rf: bool,
    uses_vlc: bool,
) -> PolicyAction:
    candidates = tuple(
        action
        for action in decision.action_space.mask.allowed_actions
        if action_resources(action).uses_rf is uses_rf
        and action_resources(action).uses_vlc is uses_vlc
    )
    if not candidates:
        raise BaselinePolicyError(
            "hardware mask has no action for the requested baseline",
        )
    return max(candidates, key=lambda action: action_resources(action).reserved_rf_attempts)


@dataclass(frozen=True, slots=True)
class FixedActionBaseline:
    """Always propose one contract action for every usable actor row."""

    action: PolicyAction
    baseline_name: str

    @property
    def name(self) -> str:
        return self.baseline_name

    @property
    def requires_oracle_truth(self) -> bool:
        return False

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is not None:
            raise BaselinePolicyError("a deployable fixed baseline cannot receive oracle truth")
        decision.action_space.mask.require_allowed(self.action)
        return tuple(
            self.action if row.usable else None for row in decision.actor_frame.rows
        )


@dataclass(frozen=True, slots=True)
class GeometryThresholdBaseline:
    """Use VLC only inside a conservative observable range/alignment region."""

    reach_m: float = 9.5
    minimum_fov_margin_rad: float = 0.90

    def __post_init__(self) -> None:
        if not math.isfinite(self.reach_m) or self.reach_m <= 0.0:
            raise BaselinePolicyError("geometry threshold reach must be positive")
        if not math.isfinite(self.minimum_fov_margin_rad):
            raise BaselinePolicyError("geometry threshold FOV margin must be finite")

    @property
    def name(self) -> str:
        return BASELINE_GEOMETRY_THRESHOLD

    @property
    def requires_oracle_truth(self) -> bool:
        return False

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is not None:
            raise BaselinePolicyError("the geometry baseline cannot receive oracle truth")
        columns = _column_map(
            decision.columns,
            ("pair_distance", "optical_fov_margin"),
        )
        vlc = _largest_allowed(decision, uses_rf=False, uses_vlc=True)
        rf = _largest_allowed(decision, uses_rf=True, uses_vlc=False)
        proposals: list[PolicyProposal] = []
        for row in decision.actor_frame.rows:
            if row.values is None:
                proposals.append(None)
                continue
            accepted = (
                row.values[columns["pair_distance"]] < self.reach_m
                and row.values[columns["optical_fov_margin"]]
                > self.minimum_fov_margin_rad
            )
            proposals.append(vlc if accepted else rf)
        return tuple(proposals)


def _action_risk(action: PolicyAction, rf_attempt_risk: float, vlc_risk: float) -> float:
    spec = action_resources(action)
    risk = 1.0
    if spec.uses_rf:
        risk *= rf_attempt_risk**spec.reserved_rf_attempts
    if spec.uses_vlc:
        risk *= vlc_risk
    return float(min(1.0, max(0.0, risk)))


def _cheapest_feasible_action(
    decision: PopulationPolicyFrame,
    *,
    rf_attempt_risk: float,
    vlc_risk: float,
) -> PolicyAction:
    alternatives = tuple(
        (
            action,
            decision.resource_map.activation_cost(action),
            _action_risk(action, rf_attempt_risk, vlc_risk),
        )
        for action in decision.action_space.mask.allowed_actions
    )
    feasible = tuple(row for row in alternatives if row[2] <= decision.miss_budget)
    if feasible:
        return min(feasible, key=lambda row: (row[1], row[2], int(row[0])))[0]
    return min(alternatives, key=lambda row: (row[2], row[1], int(row[0])))[0]


@dataclass(frozen=True, slots=True)
class ContextualNoHistoryBaseline:
    """Stateless risk selection from current causal context only.

    The optical estimate is the current blockage forecast, promoted to certain
    failure outside the observable FOV or across a junction.  The RF estimate
    is the current measured CBR with the configured per-attempt half-duplex
    floor.  No link quality, action, outcome, or history column is read.
    """

    @property
    def name(self) -> str:
        return BASELINE_CONTEXTUAL

    @property
    def requires_oracle_truth(self) -> bool:
        return False

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is not None:
            raise BaselinePolicyError("the contextual baseline cannot receive oracle truth")
        columns = _column_map(decision.columns, CONTEXTUAL_FEATURES)
        parameters = decision.pool_model.attempt_parameters
        half_duplex_floor = parameters.airtime_s / parameters.generation_period_s
        proposals: list[PolicyProposal] = []
        for row in decision.actor_frame.rows:
            if row.values is None:
                proposals.append(None)
                continue
            raw = row.values
            cbr = float(np.clip(raw[columns["rf_channel_busy_ratio"]], 0.0, 1.0))
            rf_risk = 1.0 - (1.0 - cbr) * (1.0 - half_duplex_floor)
            forecast = float(
                np.clip(raw[columns["predicted_blockage_probability"]], 0.0, 1.0)
            )
            confidence = float(np.clip(raw[columns["predictor_confidence"]], 0.0, 1.0))
            # Shrink an uncertain forecast toward an uninformative 0.5 rather
            # than treating low confidence as evidence for a clear path.
            vlc_risk = confidence * forecast + (1.0 - confidence) * 0.5
            if (
                raw[columns["optical_fov_margin"]] <= 0.0
                or raw[columns["path_spans_junction"]] >= 0.5
            ):
                vlc_risk = 1.0
            proposals.append(
                _cheapest_feasible_action(
                    decision,
                    rf_attempt_risk=rf_risk,
                    vlc_risk=vlc_risk,
                )
            )
        return tuple(proposals)


@dataclass(frozen=True, slots=True)
class SupervisedOpticalRiskEstimator:
    """Ridge fit of causal current-context features to optical miss risk."""

    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    ridge: float
    training_rows: int

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if self.feature_names != OPTICAL_ESTIMATOR_FEATURES:
            raise BaselinePolicyError("optical estimator features do not match Phase 6")
        if len(self.mean) != width or len(self.scale) != width:
            raise BaselinePolicyError("optical estimator statistics have the wrong width")
        if len(self.weights) != width + 1:
            raise BaselinePolicyError("optical estimator weights must include an intercept")
        if any(not math.isfinite(value) for value in (*self.mean, *self.scale, *self.weights)):
            raise BaselinePolicyError("optical estimator parameters must be finite")
        if any(value <= 0.0 for value in self.scale):
            raise BaselinePolicyError("optical estimator scales must be positive")
        if not math.isfinite(self.ridge) or self.ridge <= 0.0:
            raise BaselinePolicyError("optical estimator ridge must be positive")
        if self.training_rows < width + 1:
            raise BaselinePolicyError("optical estimator has too few training rows")

    @classmethod
    def fit(
        cls,
        observations: NDArray[np.floating],
        optical_risks: NDArray[np.floating],
        *,
        columns: Sequence[str],
        ridge: float = 1e-3,
    ) -> SupervisedOpticalRiskEstimator:
        """Fit on training observations and simulator conditional VLC risks."""

        matrix = np.asarray(observations, dtype=np.float64)
        targets = np.asarray(optical_risks, dtype=np.float64)
        if matrix.ndim != 2 or targets.shape != (matrix.shape[0],):
            raise BaselinePolicyError("fit observations and risks must be row-aligned")
        if matrix.shape[0] < len(OPTICAL_ESTIMATOR_FEATURES) + 1:
            raise BaselinePolicyError("too few rows to fit the optical-risk estimator")
        if not bool(np.all(np.isfinite(matrix))) or not bool(np.all(np.isfinite(targets))):
            raise BaselinePolicyError("fit data must contain only finite values")
        if bool(np.any((targets < 0.0) | (targets > 1.0))):
            raise BaselinePolicyError("optical risks must lie in [0, 1]")
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise BaselinePolicyError("ridge must be finite and positive")

        indices = _column_map(columns, OPTICAL_ESTIMATOR_FEATURES)
        features = matrix[:, [indices[name] for name in OPTICAL_ESTIMATOR_FEATURES]]
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1.0
        design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
        clipped = np.clip(targets, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ logits)
        return cls(
            feature_names=OPTICAL_ESTIMATOR_FEATURES,
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            weights=tuple(float(value) for value in weights),
            ridge=ridge,
            training_rows=matrix.shape[0],
        )

    def predict(
        self,
        observations: NDArray[np.floating],
        *,
        columns: Sequence[str],
    ) -> NDArray[np.float64]:
        """Predict optical miss probabilities without any oracle-side field."""

        matrix = np.asarray(observations, dtype=np.float64)
        if matrix.ndim != 2 or not bool(np.all(np.isfinite(matrix))):
            raise BaselinePolicyError("prediction observations must be a finite matrix")
        indices = _column_map(columns, self.feature_names)
        features = matrix[:, [indices[name] for name in self.feature_names]]
        standardized = (features - np.asarray(self.mean)) / np.asarray(self.scale)
        design = np.column_stack((standardized, np.ones(len(features))))
        logits = np.clip(design @ np.asarray(self.weights), -40.0, 40.0)
        return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)

    def as_dict(self) -> dict[str, object]:
        """Return a versioned JSON-safe training artifact."""

        return {
            "schema": OPTICAL_ESTIMATOR_SCHEMA,
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "ridge": self.ridge,
            "training_rows": self.training_rows,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> SupervisedOpticalRiskEstimator:
        """Restore an estimator only from its exact artifact schema."""

        expected = {
            "schema",
            "feature_names",
            "mean",
            "scale",
            "weights",
            "ridge",
            "training_rows",
        }
        if set(payload) != expected:
            raise BaselinePolicyError("optical estimator artifact fields are invalid")
        if payload["schema"] != OPTICAL_ESTIMATOR_SCHEMA:
            raise BaselinePolicyError("optical estimator artifact schema is unsupported")

        def numeric_tuple(name: str) -> tuple[float, ...]:
            values = payload[name]
            if not isinstance(values, list | tuple):
                raise BaselinePolicyError(f"optical estimator {name} must be an array")
            try:
                return tuple(float(value) for value in values)
            except (TypeError, ValueError) as error:
                raise BaselinePolicyError(
                    f"optical estimator {name} must be numeric"
                ) from error

        names = payload["feature_names"]
        ridge = payload["ridge"]
        training_rows = payload["training_rows"]
        if not isinstance(names, list | tuple) or any(
            not isinstance(name, str) for name in names
        ):
            raise BaselinePolicyError("optical estimator feature_names must be strings")
        if not isinstance(ridge, int | float) or isinstance(ridge, bool):
            raise BaselinePolicyError("optical estimator ridge must be numeric")
        if not isinstance(training_rows, int) or isinstance(training_rows, bool):
            raise BaselinePolicyError("optical estimator training_rows must be an integer")
        return cls(
            feature_names=tuple(names),
            mean=numeric_tuple("mean"),
            scale=numeric_tuple("scale"),
            weights=numeric_tuple("weights"),
            ridge=float(ridge),
            training_rows=training_rows,
        )


def _risk_matrix(
    decision: PopulationPolicyFrame,
    response: RFPoolResponse,
    *,
    vlc_risk: NDArray[np.float64],
    rf_decoding_risk: NDArray[np.float64],
) -> NDArray[np.float64]:
    population = decision.population_size
    allowed = decision.action_space.mask.allowed_actions
    access = decision.pool_model.access_failure_probability(response)
    per_attempt = 1.0 - (1.0 - access) * (1.0 - rf_decoding_risk)
    result = np.empty((population, len(allowed)), dtype=np.float64)
    for action_index, action in enumerate(allowed):
        spec = action_resources(action)
        risk = np.ones(population, dtype=np.float64)
        if spec.uses_rf:
            risk *= per_attempt**spec.reserved_rf_attempts
        if spec.uses_vlc:
            risk *= vlc_risk
        result[:, action_index] = np.clip(risk, 0.0, 1.0)
    return result


def _greedy_budget_allocation(
    decision: PopulationPolicyFrame,
    risks: NDArray[np.float64],
) -> tuple[PolicyAction, ...]:
    """Deterministically reduce mean risk by best marginal risk per cost."""

    allowed = decision.action_space.mask.allowed_actions
    costs = np.asarray(
        [decision.resource_map.activation_cost(action) for action in allowed],
        dtype=np.float64,
    )
    actions: list[PolicyAction] = []
    chosen_indices: list[int] = []
    usable: list[bool] = []
    for row_index, actor_row in enumerate(decision.actor_frame.rows):
        if not actor_row.usable:
            action = decision.action_space.fallback_action
            actions.append(action)
            chosen_indices.append(allowed.index(action))
            usable.append(False)
            continue
        minimum = min(
            range(len(allowed)),
            key=lambda index: (costs[index], risks[row_index, index], int(allowed[index])),
        )
        actions.append(allowed[minimum])
        chosen_indices.append(minimum)
        usable.append(True)

    total_risk = float(
        sum(risks[row, selected] for row, selected in enumerate(chosen_indices))
    )
    target = decision.miss_budget * max(1, decision.population_size)
    while total_risk > target:
        best: tuple[float, float, float, int, int] | None = None
        best_row = -1
        best_candidate = -1
        for row_index, selected in enumerate(chosen_indices):
            if not usable[row_index]:
                continue
            old_risk = risks[row_index, selected]
            old_cost = costs[selected]
            for candidate in range(len(allowed)):
                reduction = float(old_risk - risks[row_index, candidate])
                added_cost = float(costs[candidate] - old_cost)
                if reduction <= 1e-15 or added_cost < -1e-12:
                    continue
                efficiency = math.inf if added_cost <= 1e-12 else reduction / added_cost
                rank = (
                    efficiency,
                    reduction,
                    -added_cost,
                    -row_index,
                    -int(allowed[candidate]),
                )
                if best is None or rank > best:
                    best = rank
                    best_row = row_index
                    best_candidate = candidate
        if best is None:
            break
        if best_row < 0 or best_candidate < 0:  # pragma: no cover - defensive
            raise BaselinePolicyError("analytical allocation lost its best candidate")
        previous = chosen_indices[best_row]
        total_risk -= float(risks[best_row, previous] - risks[best_row, best_candidate])
        chosen_indices[best_row] = best_candidate
        actions[best_row] = allowed[best_candidate]
    return tuple(actions)


def _analytical_fixed_point(
    decision: PopulationPolicyFrame,
    *,
    vlc_risk: NDArray[np.float64],
    rf_decoding_risk: NDArray[np.float64],
    max_iterations: int = 32,
) -> tuple[PolicyProposal, ...]:
    population = decision.population_size
    if vlc_risk.shape != (population,) or rf_decoding_risk.shape != (population,):
        raise BaselinePolicyError("analytical risks must align with the population")
    if not bool(
        np.all(np.isfinite(vlc_risk))
        and np.all(np.isfinite(rf_decoding_risk))
        and np.all((vlc_risk >= 0.0) & (vlc_risk <= 1.0))
        and np.all((rf_decoding_risk >= 0.0) & (rf_decoding_risk <= 1.0))
    ):
        raise BaselinePolicyError("analytical risks must be finite probabilities")
    if population == 0:
        return ()

    vlc_seed = _largest_allowed(decision, uses_rf=False, uses_vlc=True)
    rf_seed = _largest_allowed(decision, uses_rf=True, uses_vlc=False)

    def seed_actions(action: PolicyAction) -> tuple[PolicyAction, ...]:
        return tuple(
            action if row.usable else decision.action_space.fallback_action
            for row in decision.actor_frame.rows
        )

    seeds = (
        seed_actions(decision.action_space.fallback_action),
        seed_actions(vlc_seed),
        seed_actions(rf_seed),
    )
    candidates: list[tuple[bool, float, float, tuple[PolicyAction, ...]]] = []
    for seed in seeds:
        actions = seed
        seen: set[tuple[PolicyAction, ...]] = set()
        for _ in range(max_iterations):
            if actions in seen:
                break
            seen.add(actions)
            ledger = FrameActionLedger.from_frame(
                decision.frame,
                dict(zip(decision.frame.active_pair_ids, actions, strict=True)),
                resource_map=decision.resource_map,
            )
            response = decision.pool_model.evaluate(
                decision.pool_model.demand_from_ledger(ledger)
            )
            risks = _risk_matrix(
                decision,
                response,
                vlc_risk=vlc_risk,
                rf_decoding_risk=rf_decoding_risk,
            )
            updated = _greedy_budget_allocation(decision, risks)
            if updated == actions:
                break
            actions = updated

        ledger = FrameActionLedger.from_frame(
            decision.frame,
            dict(zip(decision.frame.active_pair_ids, actions, strict=True)),
            resource_map=decision.resource_map,
        )
        response = decision.pool_model.evaluate(
            decision.pool_model.demand_from_ledger(ledger)
        )
        risks = _risk_matrix(
            decision,
            response,
            vlc_risk=vlc_risk,
            rf_decoding_risk=rf_decoding_risk,
        )
        allowed = decision.action_space.mask.allowed_actions
        indices = tuple(allowed.index(action) for action in actions)
        mean_risk = float(
            sum(risks[row, index] for row, index in enumerate(indices)) / population
        )
        mean_cost = float(
            sum(decision.resource_map.activation_cost(action) for action in actions)
            / population
        )
        candidates.append((mean_risk <= decision.miss_budget, mean_cost, mean_risk, actions))

    feasible = tuple(candidate for candidate in candidates if candidate[0])
    chosen = min(
        feasible or tuple(candidates),
        key=lambda candidate: (
            candidate[1] if candidate[0] else candidate[2],
            candidate[2] if candidate[0] else candidate[1],
            tuple(int(action) for action in candidate[3]),
        ),
    )
    return tuple(
        action if row.usable else None
        for action, row in zip(chosen[3], decision.actor_frame.rows, strict=True)
    )


@dataclass(frozen=True, slots=True)
class SupervisedRiskAllocationBaseline:
    """Training-fitted optical risk plus the action-coupled analytical pool."""

    estimator: SupervisedOpticalRiskEstimator

    @property
    def name(self) -> str:
        return BASELINE_SUPERVISED

    @property
    def requires_oracle_truth(self) -> bool:
        return False

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is not None:
            raise BaselinePolicyError("the supervised baseline cannot receive oracle truth")
        optical = self.estimator.predict(_raw_matrix(decision), columns=decision.columns)
        # The inherited analytical comparator treats RF decoding as optimistic
        # and models only action-coupled access risk.  Keeping that assumption
        # explicit makes this a strong deployable comparator without leaking
        # the current simulator SINR.
        rf_decoding = np.zeros(decision.population_size, dtype=np.float64)
        return _analytical_fixed_point(
            decision,
            vlc_risk=optical,
            rf_decoding_risk=rf_decoding,
        )


@dataclass(frozen=True, slots=True)
class TruthRiskOracleBaseline:
    """Non-deployable fixed point using current exact RF and VLC risks."""

    @property
    def name(self) -> str:
        return BASELINE_ORACLE

    @property
    def requires_oracle_truth(self) -> bool:
        return True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise BaselinePolicyError("the truth-risk oracle requires channel truth")
        expected = decision.frame.active_pair_ids
        if set(channel_truth) != set(expected):
            raise BaselinePolicyError("oracle truth must cover the current population exactly")
        optical = np.asarray(
            [channel_truth[pair_id].vlc_result.total_failure_probability for pair_id in expected],
            dtype=np.float64,
        )
        rf_decoding = np.asarray(
            [
                channel_truth[pair_id].rf_propagation.decoding_failure_probability
                for pair_id in expected
            ],
            dtype=np.float64,
        )
        return _analytical_fixed_point(
            decision,
            vlc_risk=optical,
            rf_decoding_risk=rf_decoding,
        )


def baseline_policy(
    name: str,
    *,
    supervised_estimator: SupervisedOpticalRiskEstimator | None = None,
) -> PopulationPolicy:
    """Construct one canonical Phase 6 baseline by artifact-facing name."""

    if not isinstance(name, str) or not name.strip():
        raise BaselinePolicyError("baseline name must be a non-empty string")
    normalized = name.strip().lower().replace("_", "-")
    if normalized in BASELINE_ALWAYS_RF:
        attempts = int(normalized.rsplit("-", 1)[1])
        return FixedActionBaseline(PolicyAction(attempts), normalized)
    if normalized == BASELINE_ALWAYS_VLC:
        return FixedActionBaseline(PolicyAction.VLC, normalized)
    if normalized == BASELINE_DUPLICATE_ALL:
        return FixedActionBaseline(PolicyAction.DUP_4, normalized)
    if normalized == BASELINE_GEOMETRY_THRESHOLD:
        return GeometryThresholdBaseline()
    if normalized == BASELINE_CONTEXTUAL:
        return ContextualNoHistoryBaseline()
    if normalized == BASELINE_SUPERVISED:
        if supervised_estimator is None:
            raise BaselinePolicyError(
                "supervised-risk-allocation requires a training-fitted estimator"
            )
        return SupervisedRiskAllocationBaseline(supervised_estimator)
    if normalized == BASELINE_ORACLE:
        return TruthRiskOracleBaseline()
    raise BaselinePolicyError(
        f"unknown baseline {name!r}; expected one of {', '.join(BASELINE_NAMES)}"
    )


def run_baseline_rollout(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    policy: PopulationPolicy,
    environment_seed: int | None = None,
    policy_seed: int = 0,
    max_frames: int | None = None,
) -> DeterministicRolloutReport:
    """Run a Phase 6 policy through the shared population rollout engine."""

    from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout

    return run_policy_rollout(
        config,
        source,
        policy=policy,
        environment_seed=environment_seed,
        policy_seed=policy_seed,
        max_frames=max_frames,
    )


__all__ = [
    "BASELINE_ALWAYS_RF",
    "BASELINE_ALWAYS_VLC",
    "BASELINE_CONTEXTUAL",
    "BASELINE_DUPLICATE_ALL",
    "BASELINE_GEOMETRY_THRESHOLD",
    "BASELINE_NAMES",
    "BASELINE_ORACLE",
    "BASELINE_SUPERVISED",
    "CONTEXTUAL_FEATURES",
    "OPTICAL_ESTIMATOR_FEATURES",
    "OPTICAL_ESTIMATOR_SCHEMA",
    "BaselinePolicyError",
    "ContextualNoHistoryBaseline",
    "FixedActionBaseline",
    "GeometryThresholdBaseline",
    "SupervisedOpticalRiskEstimator",
    "SupervisedRiskAllocationBaseline",
    "TruthRiskOracleBaseline",
    "baseline_policy",
    "run_baseline_rollout",
]
