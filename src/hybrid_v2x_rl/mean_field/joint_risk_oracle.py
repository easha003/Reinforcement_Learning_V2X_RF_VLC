"""Certificate-aware population-joint search under pair-local RF physics.

The former oracle reduced every frame to one global offered-load scalar. That
separability is invalid once collision and half-duplex exposure differ by
pair. This module evaluates every candidate through the authoritative local RF
pipeline. It exhaustively proves optimality when the complete joint action
space is below a declared cap; larger frames return a realizable deterministic
multi-start solution together with a valid zero-contention lower bound. A
non-exact failed candidate is therefore inconclusive, never an infeasibility
claim.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.channels.rf.model import RFPropagationResult
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.mean_field.local_rf_pipeline import (
    FrameLocalRFContext,
    FrameLocalRFPhysics,
    LocalRFPhysicsModel,
)
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PopulationPolicyFrame,
)

JOINT_RISK_ORACLE_METHOD: Final = (
    "pair-local exhaustive joint-action enumeration below the declared cap; "
    "otherwise deterministic multi-start simultaneous best-response search "
    "with a certified zero-contention lower bound"
)
MAX_EXACT_JOINT_ASSIGNMENTS: Final = 100_000
MAX_SEARCH_ITERATIONS: Final = 16
_NUMERICAL_TOLERANCE: Final = 1e-12


class JointRiskOracleError(HybridV2XError):
    """The pair-local joint search or its certificate is invalid."""


def _valid_probability_vector(values: object, population: int) -> bool:
    return bool(
        isinstance(values, np.ndarray)
        and values.shape == (population,)
        and values.dtype == np.float64
        and np.all(np.isfinite(values))
        and np.all((values >= 0.0) & (values <= 1.0))
    )


@dataclass(frozen=True, slots=True)
class JointRiskOracleProblem:
    """One frame's action space, local RF model, and isolated channel truth."""

    frame: PopulationFrame
    context: FrameLocalRFContext
    usable_mask: tuple[bool, ...]
    allowed_actions: tuple[PolicyAction, ...]
    fallback_action: PolicyAction
    resource_map: ActionResourceMap
    local_rf_model: LocalRFPhysicsModel
    rf_propagation_by_pair: Mapping[str, RFPropagationResult]
    vlc_failure_probabilities: NDArray[np.float64]
    exact_assignment_cap: int = MAX_EXACT_JOINT_ASSIGNMENTS
    max_search_iterations: int = MAX_SEARCH_ITERATIONS

    def __post_init__(self) -> None:
        population = len(self.pair_ids)
        if population < 1:
            raise JointRiskOracleError("joint oracle requires a nonempty population")
        if self.context.frame != self.frame:
            raise JointRiskOracleError("joint oracle context and frame differ")
        if len(self.usable_mask) != population or any(
            type(value) is not bool for value in self.usable_mask
        ):
            raise JointRiskOracleError("usable mask must align with the population")
        if (
            not self.allowed_actions
            or len(set(self.allowed_actions)) != len(self.allowed_actions)
            or any(type(action) is not PolicyAction for action in self.allowed_actions)
        ):
            raise JointRiskOracleError(
                "allowed actions must be unique PolicyAction values"
            )
        if self.fallback_action not in self.allowed_actions:
            raise JointRiskOracleError("fallback action must be hardware-allowed")
        if not isinstance(self.resource_map, ActionResourceMap):
            raise JointRiskOracleError("joint oracle requires an ActionResourceMap")
        if not isinstance(self.local_rf_model, LocalRFPhysicsModel):
            raise JointRiskOracleError("joint oracle requires LocalRFPhysicsModel")
        if set(self.rf_propagation_by_pair) != set(self.pair_ids) or any(
            not isinstance(self.rf_propagation_by_pair[pair_id], RFPropagationResult)
            for pair_id in self.pair_ids
        ):
            raise JointRiskOracleError(
                "RF propagation truth must cover the population exactly"
            )
        if not _valid_probability_vector(
            self.vlc_failure_probabilities,
            population,
        ):
            raise JointRiskOracleError(
                "VLC risks must be an aligned finite float64 probability vector"
            )
        if (
            not isinstance(self.exact_assignment_cap, int)
            or isinstance(self.exact_assignment_cap, bool)
            or self.exact_assignment_cap < 1
        ):
            raise JointRiskOracleError("exact assignment cap must be positive")
        if (
            not isinstance(self.max_search_iterations, int)
            or isinstance(self.max_search_iterations, bool)
            or self.max_search_iterations < 1
        ):
            raise JointRiskOracleError("search iteration limit must be positive")

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return self.frame.active_pair_ids

    @property
    def assignment_count(self) -> int:
        result = 1
        choices = len(self.allowed_actions)
        for usable in self.usable_mask:
            result *= choices if usable else 1
        return result

    @property
    def search_space_log10(self) -> float:
        return math.log10(self.assignment_count)

    @classmethod
    def from_decision(
        cls,
        decision: PopulationPolicyFrame,
        channel_truth: OracleChannelTruth,
        *,
        usable_mask: tuple[bool, ...] | None = None,
        exact_assignment_cap: int = MAX_EXACT_JOINT_ASSIGNMENTS,
        max_search_iterations: int = MAX_SEARCH_ITERATIONS,
    ) -> JointRiskOracleProblem:
        if not isinstance(decision, PopulationPolicyFrame):
            raise JointRiskOracleError(
                "joint oracle requires a PopulationPolicyFrame"
            )
        pair_ids = decision.frame.active_pair_ids
        if set(channel_truth) != set(pair_ids):
            raise JointRiskOracleError(
                "channel truth must cover the population exactly"
            )
        effective_usable_mask = (
            decision.actor_frame.usable_mask
            if usable_mask is None
            else usable_mask
        )
        return cls(
            frame=decision.frame,
            context=decision.local_rf_context,
            usable_mask=effective_usable_mask,
            allowed_actions=decision.action_space.mask.allowed_actions,
            fallback_action=decision.action_space.fallback_action,
            resource_map=decision.resource_map,
            local_rf_model=decision.local_rf_model,
            rf_propagation_by_pair={
                pair_id: channel_truth[pair_id].rf_propagation
                for pair_id in pair_ids
            },
            vlc_failure_probabilities=np.asarray(
                [
                    channel_truth[pair_id].vlc_result.total_failure_probability
                    for pair_id in pair_ids
                ],
                dtype=np.float64,
            ),
            exact_assignment_cap=exact_assignment_cap,
            max_search_iterations=max_search_iterations,
        )


@dataclass(frozen=True, slots=True)
class JointRiskOracleSolution:
    """Best realizable assignment plus an explicit optimality certificate."""

    pair_ids: tuple[str, ...]
    actions: tuple[PolicyAction, ...]
    usable_pairs: int
    forced_fallback_pairs: int
    total_rf_attempts: int
    total_conditional_miss_risk: float
    usable_conditional_miss_risk: float
    forced_fallback_conditional_miss_risk: float
    total_activation_cost: float
    optimality_proven: bool
    certified_lower_bound: float
    absolute_optimality_gap: float
    assignment_space_size: int
    assignments_evaluated: int
    search_starts: int
    search_iterations: int
    search_space_log10: float
    action_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        population = len(self.pair_ids)
        if population < 1 or len(self.actions) != population:
            raise JointRiskOracleError(
                "joint solution must cover the population exactly"
            )
        if self.usable_pairs + self.forced_fallback_pairs != population:
            raise JointRiskOracleError(
                "joint solution usability counts do not reconcile"
            )
        if self.total_rf_attempts != sum(
            action_resources(action).reserved_rf_attempts
            for action in self.actions
        ):
            raise JointRiskOracleError(
                "joint solution RF attempts do not reconcile"
            )
        if not math.isclose(
            self.total_conditional_miss_risk,
            self.usable_conditional_miss_risk
            + self.forced_fallback_conditional_miss_risk,
            rel_tol=0.0,
            abs_tol=_NUMERICAL_TOLERANCE,
        ):
            raise JointRiskOracleError(
                "joint solution risk partition does not reconcile"
            )
        numeric = (
            self.total_conditional_miss_risk,
            self.usable_conditional_miss_risk,
            self.forced_fallback_conditional_miss_risk,
            self.total_activation_cost,
            self.certified_lower_bound,
            self.absolute_optimality_gap,
            self.search_space_log10,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise JointRiskOracleError(
                "joint solution metrics must be finite and nonnegative"
            )
        expected_gap = (
            self.total_conditional_miss_risk - self.certified_lower_bound
        )
        if (
            self.certified_lower_bound
            > self.total_conditional_miss_risk + _NUMERICAL_TOLERANCE
            or not math.isclose(
                self.absolute_optimality_gap,
                max(0.0, expected_gap),
                rel_tol=0.0,
                abs_tol=_NUMERICAL_TOLERANCE,
            )
        ):
            raise JointRiskOracleError(
                "joint solution lower bound or optimality gap is invalid"
            )
        if self.optimality_proven and self.absolute_optimality_gap > _NUMERICAL_TOLERANCE:
            raise JointRiskOracleError(
                "an exact solution must have zero certified optimality gap"
            )
        if (
            self.assignment_space_size < 1
            or not 1 <= self.assignments_evaluated <= self.assignment_space_size
            or self.search_starts < 1
            or self.search_iterations < 0
        ):
            raise JointRiskOracleError("joint search accounting is invalid")
        if self.optimality_proven and (
            self.assignments_evaluated != self.assignment_space_size
        ):
            raise JointRiskOracleError(
                "exhaustive exactness requires every assignment to be evaluated"
            )
        if (
            len(self.action_counts) != len(PolicyAction)
            or sum(self.action_counts) != population
        ):
            raise JointRiskOracleError(
                "joint action counts do not partition the population"
            )

    @property
    def mean_conditional_miss_risk(self) -> float:
        return self.total_conditional_miss_risk / len(self.pair_ids)

    @property
    def mean_certified_lower_bound(self) -> float:
        return self.certified_lower_bound / len(self.pair_ids)

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
            "optimality_proven": self.optimality_proven,
            "certified_lower_bound": self.certified_lower_bound,
            "mean_certified_lower_bound": self.mean_certified_lower_bound,
            "absolute_optimality_gap": self.absolute_optimality_gap,
            "assignment_space_size": self.assignment_space_size,
            "assignments_evaluated": self.assignments_evaluated,
            "search_starts": self.search_starts,
            "search_iterations": self.search_iterations,
            "search_space_log10": self.search_space_log10,
            "action_counts": {
                name: self.action_counts[index]
                for index, name in enumerate(POLICY_ACTION_ORDER)
            },
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    actions: tuple[PolicyAction, ...]
    total_risk: float
    usable_risk: float
    forced_risk: float
    total_cost: float
    total_rf_attempts: int
    physics: FrameLocalRFPhysics

    @property
    def rank(self) -> tuple[float, float, int, tuple[int, ...]]:
        return (
            self.total_risk,
            self.total_cost,
            self.total_rf_attempts,
            tuple(int(action) for action in self.actions),
        )


def _evaluate_actions(
    problem: JointRiskOracleProblem,
    actions: tuple[PolicyAction, ...],
) -> _Candidate:
    if len(actions) != len(problem.pair_ids):
        raise JointRiskOracleError("candidate actions do not cover the population")
    if any(
        action not in problem.allowed_actions
        or (not usable and action is not problem.fallback_action)
        for action, usable in zip(actions, problem.usable_mask, strict=True)
    ):
        raise JointRiskOracleError(
            "candidate contains a masked or non-fallback action"
        )
    ledger = FrameActionLedger.from_frame(
        problem.frame,
        dict(zip(problem.pair_ids, actions, strict=True)),
        resource_map=problem.resource_map,
    )
    physics = problem.local_rf_model.evaluate(
        problem.context,
        ledger,
        propagation_by_pair=problem.rf_propagation_by_pair,
    )
    risks: list[float] = []
    for index, action in enumerate(actions):
        spec = action_resources(action)
        risk = 1.0
        if spec.uses_rf:
            risk *= (
                physics.attempt_risks.risk_for(
                    problem.pair_ids[index]
                ).total_failure_probability
                ** spec.reserved_rf_attempts
            )
        if spec.uses_vlc:
            risk *= float(problem.vlc_failure_probabilities[index])
        risks.append(float(min(1.0, max(0.0, risk))))
    usable_risk = math.fsum(
        risk for risk, usable in zip(risks, problem.usable_mask, strict=True) if usable
    )
    forced_risk = math.fsum(
        risk
        for risk, usable in zip(risks, problem.usable_mask, strict=True)
        if not usable
    )
    return _Candidate(
        actions=actions,
        total_risk=usable_risk + forced_risk,
        usable_risk=usable_risk,
        forced_risk=forced_risk,
        total_cost=math.fsum(
            problem.resource_map.activation_cost(action) for action in actions
        ),
        total_rf_attempts=sum(
            action_resources(action).reserved_rf_attempts for action in actions
        ),
        physics=physics,
    )


def _certified_zero_contention_lower_bound(
    problem: JointRiskOracleProblem,
) -> float:
    """Optimistic bound with collision and receiver activity both removed."""

    terms: list[float] = []
    for index, usable in enumerate(problem.usable_mask):
        actions = (
            problem.allowed_actions if usable else (problem.fallback_action,)
        )
        decoding = problem.rf_propagation_by_pair[
            problem.pair_ids[index]
        ].decoding_failure_probability
        vlc = float(problem.vlc_failure_probabilities[index])
        candidate_risks: list[float] = []
        for action in actions:
            spec = action_resources(action)
            risk = 1.0
            if spec.uses_rf:
                risk *= decoding**spec.reserved_rf_attempts
            if spec.uses_vlc:
                risk *= vlc
            candidate_risks.append(risk)
        terms.append(min(candidate_risks))
    return math.fsum(terms)


def _best_responses(
    problem: JointRiskOracleProblem,
    candidate: _Candidate,
) -> tuple[PolicyAction, ...]:
    """Return simultaneous own-risk best responses to one realized local state."""

    proposals: list[PolicyAction] = []
    for index, (pair_id, usable) in enumerate(
        zip(problem.pair_ids, problem.usable_mask, strict=True)
    ):
        if not usable:
            proposals.append(problem.fallback_action)
            continue
        response = candidate.physics.responses.response_for(pair_id)
        exposure = candidate.physics.endpoint_schedule.exposure_for(pair_id)
        decoding = problem.rf_propagation_by_pair[
            pair_id
        ].decoding_failure_probability
        access = 1.0 - (
            1.0 - response.per_attempt_collision_probability
        ) * (1.0 - exposure.half_duplex_probability)
        attempt_risk = 1.0 - (1.0 - access) * (1.0 - decoding)
        vlc = float(problem.vlc_failure_probabilities[index])

        def rank(
            action: PolicyAction,
            attempt_failure: float = attempt_risk,
            optical_failure: float = vlc,
        ) -> tuple[float, float, int, int]:
            spec = action_resources(action)
            risk = 1.0
            if spec.uses_rf:
                risk *= attempt_failure**spec.reserved_rf_attempts
            if spec.uses_vlc:
                risk *= optical_failure
            return (
                risk,
                problem.resource_map.activation_cost(action),
                spec.reserved_rf_attempts,
                int(action),
            )

        proposals.append(min(problem.allowed_actions, key=rank))
    return tuple(proposals)


def _make_solution(
    problem: JointRiskOracleProblem,
    candidate: _Candidate,
    *,
    exact: bool,
    lower_bound: float,
    assignments_evaluated: int,
    search_starts: int,
    search_iterations: int,
) -> JointRiskOracleSolution:
    certified = candidate.total_risk if exact else lower_bound
    return JointRiskOracleSolution(
        pair_ids=problem.pair_ids,
        actions=candidate.actions,
        usable_pairs=sum(problem.usable_mask),
        forced_fallback_pairs=sum(not value for value in problem.usable_mask),
        total_rf_attempts=candidate.total_rf_attempts,
        total_conditional_miss_risk=candidate.total_risk,
        usable_conditional_miss_risk=candidate.usable_risk,
        forced_fallback_conditional_miss_risk=candidate.forced_risk,
        total_activation_cost=candidate.total_cost,
        optimality_proven=exact,
        certified_lower_bound=certified,
        absolute_optimality_gap=max(0.0, candidate.total_risk - certified),
        assignment_space_size=problem.assignment_count,
        assignments_evaluated=assignments_evaluated,
        search_starts=search_starts,
        search_iterations=search_iterations,
        search_space_log10=problem.search_space_log10,
        action_counts=tuple(candidate.actions.count(action) for action in PolicyAction),
    )


def solve_pair_local_joint_risk(
    problem: JointRiskOracleProblem,
) -> JointRiskOracleSolution:
    """Return an exact small-frame optimum or certified large-frame search."""

    if not isinstance(problem, JointRiskOracleProblem):
        raise JointRiskOracleError(
            "joint risk solver requires JointRiskOracleProblem"
        )
    choices = tuple(
        problem.allowed_actions if usable else (problem.fallback_action,)
        for usable in problem.usable_mask
    )
    lower_bound = _certified_zero_contention_lower_bound(problem)
    if problem.assignment_count <= problem.exact_assignment_cap:
        exact_best: _Candidate | None = None
        evaluated = 0
        for actions in itertools.product(*choices):
            candidate = _evaluate_actions(problem, actions)
            evaluated += 1
            if exact_best is None or candidate.rank < exact_best.rank:
                exact_best = candidate
        if exact_best is None:  # pragma: no cover - every pair has at least one action.
            raise JointRiskOracleError("exhaustive search produced no candidate")
        return _make_solution(
            problem,
            exact_best,
            exact=True,
            lower_bound=exact_best.total_risk,
            assignments_evaluated=evaluated,
            search_starts=1,
            search_iterations=0,
        )

    seeds = tuple(
        dict.fromkeys(
            tuple(
                action if usable else problem.fallback_action
                for usable in problem.usable_mask
            )
            for action in (
                problem.fallback_action,
                *problem.allowed_actions,
            )
        )
    )
    cache: dict[tuple[PolicyAction, ...], _Candidate] = {}

    def evaluate(actions: tuple[PolicyAction, ...]) -> _Candidate:
        cached = cache.get(actions)
        if cached is None:
            cached = _evaluate_actions(problem, actions)
            cache[actions] = cached
        return cached

    search_best: _Candidate | None = None
    iterations = 0
    for seed in seeds:
        actions = seed
        seen: set[tuple[PolicyAction, ...]] = set()
        for _ in range(problem.max_search_iterations):
            if actions in seen:
                break
            seen.add(actions)
            candidate = evaluate(actions)
            iterations += 1
            if search_best is None or candidate.rank < search_best.rank:
                search_best = candidate
            updated = _best_responses(problem, candidate)
            if updated == actions:
                break
            actions = updated
        candidate = evaluate(actions)
        if search_best is None or candidate.rank < search_best.rank:
            search_best = candidate
    if search_best is None:  # pragma: no cover - seeds are nonempty.
        raise JointRiskOracleError("pair-local search produced no candidate")
    return _make_solution(
        problem,
        search_best,
        exact=False,
        lower_bound=lower_bound,
        assignments_evaluated=len(cache),
        search_starts=len(seeds),
        search_iterations=iterations,
    )


__all__ = [
    "JOINT_RISK_ORACLE_METHOD",
    "MAX_EXACT_JOINT_ASSIGNMENTS",
    "MAX_SEARCH_ITERATIONS",
    "JointRiskOracleError",
    "JointRiskOracleProblem",
    "JointRiskOracleSolution",
    "solve_pair_local_joint_risk",
]
