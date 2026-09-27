"""Exact population-joint conditional-risk floor under the shared RF pool."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolModel

JOINT_RISK_ORACLE_METHOD: Final = (
    "exhaustive aggregate-load enumeration with exact separable concave allocation "
    "and monotone lower-bound pruning"
)
_NUMERICAL_TOLERANCE: Final = 1e-12


class JointRiskOracleError(HybridV2XError):
    """The joint-risk problem or its exactness certificate is invalid."""


@dataclass(frozen=True, slots=True)
class JointRiskOracleProblem:
    """One frame's action-independent truth and resource boundary."""

    pair_ids: tuple[str, ...]
    usable_mask: tuple[bool, ...]
    allowed_actions: tuple[PolicyAction, ...]
    fallback_action: PolicyAction
    resource_map: ActionResourceMap
    pool_model: RFPoolModel
    rf_decoding_failure_probabilities: NDArray[np.float64]
    vlc_failure_probabilities: NDArray[np.float64]

    def __post_init__(self) -> None:
        population = len(self.pair_ids)
        if population < 1 or len(set(self.pair_ids)) != population:
            raise JointRiskOracleError("joint oracle pair IDs must be nonempty and unique")
        if len(self.usable_mask) != population or any(
            type(value) is not bool for value in self.usable_mask
        ):
            raise JointRiskOracleError("usable mask must align with the population")
        if (
            not self.allowed_actions
            or len(set(self.allowed_actions)) != len(self.allowed_actions)
            or any(type(action) is not PolicyAction for action in self.allowed_actions)
        ):
            raise JointRiskOracleError("allowed actions must be unique PolicyAction values")
        if self.fallback_action not in self.allowed_actions:
            raise JointRiskOracleError("fallback action must be hardware-allowed")
        if not isinstance(self.resource_map, ActionResourceMap):
            raise JointRiskOracleError("joint oracle requires an ActionResourceMap")
        if not isinstance(self.pool_model, RFPoolModel):
            raise JointRiskOracleError("joint oracle requires an RFPoolModel")
        for name, values in (
            ("RF decoding risks", self.rf_decoding_failure_probabilities),
            ("VLC risks", self.vlc_failure_probabilities),
        ):
            if (
                not isinstance(values, np.ndarray)
                or values.shape != (population,)
                or values.dtype != np.float64
                or not bool(np.all(np.isfinite(values)))
                or not bool(np.all((values >= 0.0) & (values <= 1.0)))
            ):
                raise JointRiskOracleError(
                    f"{name} must be an aligned finite float64 probability vector"
                )

        attempt_levels = {
            action_resources(action).reserved_rf_attempts
            for action in self.allowed_actions
        }
        maximum = max(attempt_levels)
        if attempt_levels != set(range(maximum + 1)):
            raise JointRiskOracleError(
                "exact joint oracle requires a contiguous RF-attempt ladder from zero"
            )

    @classmethod
    def from_decision(
        cls,
        decision: PopulationPolicyFrame,
        channel_truth: OracleChannelTruth,
    ) -> JointRiskOracleProblem:
        """Bind one authoritative policy frame to isolated simulator truth."""

        if not isinstance(decision, PopulationPolicyFrame):
            raise JointRiskOracleError("joint oracle requires a PopulationPolicyFrame")
        pair_ids = decision.frame.active_pair_ids
        if set(channel_truth) != set(pair_ids):
            raise JointRiskOracleError("channel truth must cover the population exactly")
        return cls(
            pair_ids=pair_ids,
            usable_mask=decision.actor_frame.usable_mask,
            allowed_actions=decision.action_space.mask.allowed_actions,
            fallback_action=decision.action_space.fallback_action,
            resource_map=decision.resource_map,
            pool_model=decision.pool_model,
            rf_decoding_failure_probabilities=np.asarray(
                [
                    channel_truth[pair_id].rf_propagation.decoding_failure_probability
                    for pair_id in pair_ids
                ],
                dtype=np.float64,
            ),
            vlc_failure_probabilities=np.asarray(
                [channel_truth[pair_id].vlc_result.total_failure_probability for pair_id in pair_ids],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True, slots=True)
class JointRiskOracleSolution:
    """Exact joint assignment and proof-oriented load-search diagnostics."""

    pair_ids: tuple[str, ...]
    actions: tuple[PolicyAction, ...]
    usable_pairs: int
    forced_fallback_pairs: int
    total_rf_attempts: int
    total_conditional_miss_risk: float
    usable_conditional_miss_risk: float
    forced_fallback_conditional_miss_risk: float
    total_activation_cost: float
    candidate_loads_total: int
    candidate_loads_evaluated: int
    candidate_loads_pruned: int
    minimum_load: int
    maximum_load: int
    action_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        population = len(self.pair_ids)
        if population < 1 or len(self.actions) != population:
            raise JointRiskOracleError("joint solution must cover the population exactly")
        if self.usable_pairs + self.forced_fallback_pairs != population:
            raise JointRiskOracleError("joint solution usability counts do not reconcile")
        expected_attempts = sum(
            action_resources(action).reserved_rf_attempts for action in self.actions
        )
        if self.total_rf_attempts != expected_attempts:
            raise JointRiskOracleError("joint solution RF attempts do not reconcile")
        if not math.isclose(
            self.total_conditional_miss_risk,
            self.usable_conditional_miss_risk
            + self.forced_fallback_conditional_miss_risk,
            rel_tol=0.0,
            abs_tol=_NUMERICAL_TOLERANCE,
        ):
            raise JointRiskOracleError("joint solution risk partition does not reconcile")
        for name in (
            "total_conditional_miss_risk",
            "usable_conditional_miss_risk",
            "forced_fallback_conditional_miss_risk",
            "total_activation_cost",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise JointRiskOracleError(f"{name} must be finite and nonnegative")
        if not (
            self.candidate_loads_total
            == self.candidate_loads_evaluated + self.candidate_loads_pruned
            and self.candidate_loads_evaluated >= 1
            and self.minimum_load <= self.total_rf_attempts <= self.maximum_load
            and self.candidate_loads_total == self.maximum_load - self.minimum_load + 1
        ):
            raise JointRiskOracleError("joint load-search counts do not reconcile")
        if (
            len(self.action_counts) != len(PolicyAction)
            or sum(self.action_counts) != population
        ):
            raise JointRiskOracleError("joint action counts do not partition the population")

    @property
    def mean_conditional_miss_risk(self) -> float:
        return self.total_conditional_miss_risk / len(self.pair_ids)

    @property
    def mean_activation_cost(self) -> float:
        return self.total_activation_cost / len(self.pair_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "population": len(self.pair_ids),
            "usable_pairs": self.usable_pairs,
            "forced_fallback_pairs": self.forced_fallback_pairs,
            "total_rf_attempts": self.total_rf_attempts,
            "mean_rf_attempts_per_pair": self.total_rf_attempts / len(self.pair_ids),
            "total_conditional_miss_risk": self.total_conditional_miss_risk,
            "mean_conditional_miss_risk": self.mean_conditional_miss_risk,
            "usable_conditional_miss_risk": self.usable_conditional_miss_risk,
            "forced_fallback_conditional_miss_risk": (
                self.forced_fallback_conditional_miss_risk
            ),
            "total_activation_cost": self.total_activation_cost,
            "mean_activation_cost": self.mean_activation_cost,
            "candidate_loads_total": self.candidate_loads_total,
            "candidate_loads_evaluated": self.candidate_loads_evaluated,
            "candidate_loads_pruned": self.candidate_loads_pruned,
            "minimum_load": self.minimum_load,
            "maximum_load": self.maximum_load,
            "action_counts": {
                name: self.action_counts[index]
                for index, name in enumerate(POLICY_ACTION_ORDER)
            },
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    actions: tuple[PolicyAction, ...]
    total_rf_attempts: int
    total_risk: float
    usable_risk: float
    forced_risk: float
    total_cost: float


def _attempt_failure_probabilities(
    problem: JointRiskOracleProblem,
    total_rf_attempts: int,
) -> NDArray[np.float64]:
    access = problem.pool_model.counterfactual_attempt_failure_probability(
        active_pairs=len(problem.pair_ids),
        offered_rf_attempts=total_rf_attempts,
        decoding_failure_probability=0.0,
    )
    return 1.0 - (1.0 - access) * (
        1.0 - problem.rf_decoding_failure_probabilities
    )


def _action_risk(
    action: PolicyAction,
    *,
    attempt_failure: float,
    vlc_failure: float,
) -> float:
    spec = action_resources(action)
    risk = 1.0
    if spec.uses_rf:
        risk *= attempt_failure**spec.reserved_rf_attempts
    if spec.uses_vlc:
        risk *= vlc_failure
    return float(min(1.0, max(0.0, risk)))


def _usable_ladder(
    problem: JointRiskOracleProblem,
    *,
    attempt_failure: NDArray[np.float64],
    usable_indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    maximum = max(
        action_resources(action).reserved_rf_attempts
        for action in problem.allowed_actions
    )
    rows = len(usable_indices)
    risks = np.empty((rows, maximum + 1), dtype=np.float64)
    costs = np.empty((rows, maximum + 1), dtype=np.float64)
    actions = np.empty((rows, maximum + 1), dtype=np.int64)
    usable_attempt_failure = attempt_failure[usable_indices]
    usable_vlc_failure = problem.vlc_failure_probabilities[usable_indices]
    action_specs = tuple(
        (
            action,
            action_resources(action),
            problem.resource_map.activation_cost(action),
        )
        for action in problem.allowed_actions
    )
    for attempts in range(maximum + 1):
        candidates = sorted(
            (
                (action, spec, cost)
                for action, spec, cost in action_specs
                if spec.reserved_rf_attempts == attempts
            ),
            key=lambda candidate: (candidate[2], int(candidate[0])),
        )
        if not candidates:
            raise JointRiskOracleError("an RF-attempt ladder level has no action")
        candidate_risks = np.stack(
            [
                np.power(usable_attempt_failure, attempts)
                * (usable_vlc_failure if spec.uses_vlc else 1.0)
                for _, spec, _ in candidates
            ],
            axis=0,
        )
        winners = np.argmin(candidate_risks, axis=0)
        risks[:, attempts] = np.take_along_axis(
            candidate_risks,
            winners[np.newaxis, :],
            axis=0,
        )[0]
        candidate_costs = np.asarray(
            [candidate[2] for candidate in candidates], dtype=np.float64
        )
        candidate_actions = np.asarray(
            [int(candidate[0]) for candidate in candidates], dtype=np.int64
        )
        costs[:, attempts] = candidate_costs[winners]
        actions[:, attempts] = candidate_actions[winners]
    return risks, costs, actions


def _allocation_from_marginals(
    risks: NDArray[np.float64],
    costs: NDArray[np.float64],
    *,
    attempts: int,
    pair_ids: tuple[str, ...],
) -> NDArray[np.int64]:
    rows, levels = risks.shape
    maximum = levels - 1
    if not 0 <= attempts <= rows * maximum:
        raise JointRiskOracleError("requested usable RF attempts are out of range")
    gains = risks[:, :-1] - risks[:, 1:]
    if bool(np.any(gains < -_NUMERICAL_TOLERANCE)):
        raise JointRiskOracleError("more RF attempts increased a fixed-load action risk")
    gains = np.maximum(gains, 0.0)
    if maximum > 1 and bool(
        np.any(gains[:, :-1] + _NUMERICAL_TOLERANCE < gains[:, 1:])
    ):
        raise JointRiskOracleError(
            "joint action ladder lacks diminishing marginal risk reduction"
        )
    incremental_costs = costs[:, 1:] - costs[:, :-1]
    chosen = np.zeros(rows, dtype=np.int64)
    if attempts == 0:
        return chosen

    # With strictly diminishing gains, sorting every marginal once is equivalent
    # to the priority-queue allocation: every selected increment necessarily has
    # all earlier increments from the same row ahead of it.  NumPy performs this
    # ordering in compiled code.  Exact within-row gain ties need the sequential
    # prefix-aware fallback below.
    has_within_row_tie = maximum > 1 and bool(
        np.any(gains[:, :-1] == gains[:, 1:])
    )
    if not has_within_row_tie:
        row_indices = np.repeat(np.arange(rows, dtype=np.int64), maximum)
        levels = np.tile(np.arange(1, maximum + 1, dtype=np.int64), rows)
        lexical_rows = np.empty(rows, dtype=np.int64)
        lexical_rows[np.argsort(np.asarray(pair_ids), kind="stable")] = np.arange(
            rows, dtype=np.int64
        )
        ordering = np.lexsort(
            (
                levels,
                lexical_rows[row_indices],
                incremental_costs.reshape(-1),
                -gains.reshape(-1),
            )
        )
        chosen = np.bincount(
            row_indices[ordering[:attempts]], minlength=rows
        ).astype(np.int64)
        if bool(np.any(chosen > maximum)):  # pragma: no cover - defensive.
            raise JointRiskOracleError("sorted marginal allocation exceeded its ladder")
        return chosen

    heap: list[tuple[float, float, str, int, int]] = []
    for row in range(rows):
        heapq.heappush(
            heap,
            (
                -float(gains[row, 0]),
                float(incremental_costs[row, 0]),
                pair_ids[row],
                1,
                row,
            ),
        )
    for _ in range(attempts):
        if not heap:  # pragma: no cover - bounded attempts prove this cannot occur.
            raise JointRiskOracleError("marginal allocation exhausted its action ladder")
        _, _, pair_id, level, row = heapq.heappop(heap)
        if level != chosen[row] + 1 or pair_id != pair_ids[row]:
            raise JointRiskOracleError("marginal allocation lost its prefix invariant")
        chosen[row] = level
        if level < maximum:
            next_level = level + 1
            heapq.heappush(
                heap,
                (
                    -float(gains[row, level]),
                    float(incremental_costs[row, level]),
                    pair_ids[row],
                    next_level,
                    row,
                ),
            )
    return chosen


def solve_joint_risk_floor(problem: JointRiskOracleProblem) -> JointRiskOracleSolution:
    """Return the exact minimum-risk population action assignment for one frame."""

    if not isinstance(problem, JointRiskOracleProblem):
        raise JointRiskOracleError("joint risk solver requires JointRiskOracleProblem")
    population = len(problem.pair_ids)
    usable_indices = np.flatnonzero(problem.usable_mask).astype(np.int64)
    forced_indices = np.flatnonzero(np.logical_not(problem.usable_mask)).astype(np.int64)
    maximum_per_usable = max(
        action_resources(action).reserved_rf_attempts
        for action in problem.allowed_actions
    )
    fallback_attempts = action_resources(
        problem.fallback_action
    ).reserved_rf_attempts
    minimum_load = len(forced_indices) * fallback_attempts
    maximum_load = minimum_load + len(usable_indices) * maximum_per_usable
    total_candidates = maximum_load - minimum_load + 1
    best: _Candidate | None = None
    evaluated = 0
    pruned = 0
    previous_lower_bound: float | None = None
    usable_pair_ids = tuple(problem.pair_ids[index] for index in usable_indices.tolist())

    for total_attempts in range(minimum_load, maximum_load + 1):
        usable_attempts = total_attempts - minimum_load
        if total_attempts == 0:
            attempt_failure = np.zeros(population, dtype=np.float64)
        else:
            attempt_failure = _attempt_failure_probabilities(problem, total_attempts)

        forced_risks = np.asarray(
            [
                _action_risk(
                    problem.fallback_action,
                    attempt_failure=float(attempt_failure[index]),
                    vlc_failure=float(problem.vlc_failure_probabilities[index]),
                )
                for index in forced_indices.tolist()
            ],
            dtype=np.float64,
        )
        forced_risk = float(math.fsum(forced_risks.tolist()))
        forced_cost = len(forced_indices) * problem.resource_map.activation_cost(
            problem.fallback_action
        )

        if len(usable_indices):
            risks, costs, ladder_actions = _usable_ladder(
                problem,
                attempt_failure=attempt_failure,
                usable_indices=usable_indices,
            )
            lower_bound = forced_risk + float(math.fsum(np.min(risks, axis=1).tolist()))
        else:
            risks = np.empty((0, maximum_per_usable + 1), dtype=np.float64)
            costs = np.empty_like(risks)
            ladder_actions = np.empty(risks.shape, dtype=np.int64)
            lower_bound = forced_risk

        if total_attempts > 0:
            if (
                previous_lower_bound is not None
                and lower_bound + _NUMERICAL_TOLERANCE < previous_lower_bound
            ):
                raise JointRiskOracleError(
                    "aggregate-load lower bound is not monotone; pruning is unsafe"
                )
            previous_lower_bound = lower_bound
        if best is not None and lower_bound > best.total_risk + _NUMERICAL_TOLERANCE:
            pruned = maximum_load - total_attempts + 1
            break

        chosen_levels = (
            _allocation_from_marginals(
                risks,
                costs,
                attempts=usable_attempts,
                pair_ids=usable_pair_ids,
            )
            if len(usable_indices)
            else np.empty(0, dtype=np.int64)
        )
        usable_risk = float(
            math.fsum(
                risks[row, level]
                for row, level in enumerate(chosen_levels.tolist())
            )
        )
        usable_cost = float(
            math.fsum(
                costs[row, level]
                for row, level in enumerate(chosen_levels.tolist())
            )
        )
        actions_by_row: list[PolicyAction | None] = [None] * population
        for index in forced_indices.tolist():
            actions_by_row[index] = problem.fallback_action
        for local_row, (index, level) in enumerate(
            zip(usable_indices.tolist(), chosen_levels.tolist(), strict=True)
        ):
            actions_by_row[index] = PolicyAction(int(ladder_actions[local_row, level]))
        if any(action is None for action in actions_by_row):  # pragma: no cover - defensive.
            raise JointRiskOracleError("joint assignment left a population row uncovered")
        actions = tuple(action for action in actions_by_row if action is not None)
        candidate = _Candidate(
            actions=actions,
            total_rf_attempts=total_attempts,
            total_risk=usable_risk + forced_risk,
            usable_risk=usable_risk,
            forced_risk=forced_risk,
            total_cost=usable_cost + forced_cost,
        )
        evaluated += 1
        if best is None or (
            candidate.total_risk < best.total_risk - _NUMERICAL_TOLERANCE
            or (
                math.isclose(
                    candidate.total_risk,
                    best.total_risk,
                    rel_tol=0.0,
                    abs_tol=_NUMERICAL_TOLERANCE,
                )
                and (
                    candidate.total_cost,
                    candidate.total_rf_attempts,
                    tuple(int(action) for action in candidate.actions),
                )
                < (
                    best.total_cost,
                    best.total_rf_attempts,
                    tuple(int(action) for action in best.actions),
                )
            )
        ):
            best = candidate

    if best is None:  # pragma: no cover - the finite load range is nonempty.
        raise JointRiskOracleError("joint load enumeration produced no candidate")
    if evaluated + pruned != total_candidates:
        raise JointRiskOracleError("joint load enumeration did not cover its search range")
    counts = tuple(best.actions.count(action) for action in PolicyAction)
    return JointRiskOracleSolution(
        pair_ids=problem.pair_ids,
        actions=best.actions,
        usable_pairs=len(usable_indices),
        forced_fallback_pairs=len(forced_indices),
        total_rf_attempts=best.total_rf_attempts,
        total_conditional_miss_risk=best.total_risk,
        usable_conditional_miss_risk=best.usable_risk,
        forced_fallback_conditional_miss_risk=best.forced_risk,
        total_activation_cost=best.total_cost,
        candidate_loads_total=total_candidates,
        candidate_loads_evaluated=evaluated,
        candidate_loads_pruned=pruned,
        minimum_load=minimum_load,
        maximum_load=maximum_load,
        action_counts=counts,
    )


__all__ = [
    "JOINT_RISK_ORACLE_METHOD",
    "JointRiskOracleError",
    "JointRiskOracleProblem",
    "JointRiskOracleSolution",
    "solve_joint_risk_floor",
]
