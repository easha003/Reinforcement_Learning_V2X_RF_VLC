"""Exact population-joint reliability-floor solver."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.joint_risk_oracle import (
    JointRiskOracleError,
    JointRiskOracleProblem,
    solve_joint_risk_floor,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=1.0,
    vlc_activation_cost=1.0,
)


def _model() -> RFPoolModel:
    config = load_headline_config(PROJECT_ROOT)
    return RFPoolModel(
        parameters=build_rf_channel(config, band=SensitivityBand.NOMINAL).collision,
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )


def _problem(*, usable: tuple[bool, ...] = (True, True, True)) -> JointRiskOracleProblem:
    return JointRiskOracleProblem(
        pair_ids=tuple(f"pair-{index}" for index in range(len(usable))),
        usable_mask=usable,
        allowed_actions=tuple(PolicyAction),
        fallback_action=PolicyAction.DUP_4,
        resource_map=RESOURCE_MAP,
        pool_model=_model(),
        rf_decoding_failure_probabilities=np.asarray(
            (0.02, 0.15, 0.40)[: len(usable)],
            dtype=np.float64,
        ),
        vlc_failure_probabilities=np.asarray(
            (0.001, 0.35, 1.0)[: len(usable)],
            dtype=np.float64,
        ),
    )


def _risk(problem: JointRiskOracleProblem, actions: tuple[PolicyAction, ...]) -> float:
    total_attempts = sum(
        action_resources(action).reserved_rf_attempts for action in actions
    )
    if total_attempts:
        access = problem.pool_model.counterfactual_attempt_failure_probability(
            active_pairs=len(actions),
            offered_rf_attempts=total_attempts,
            decoding_failure_probability=0.0,
        )
        attempt_failures = 1.0 - (1.0 - access) * (
            1.0 - problem.rf_decoding_failure_probabilities
        )
    else:
        attempt_failures = np.zeros(len(actions), dtype=np.float64)
    terms: list[float] = []
    for row, action in enumerate(actions):
        spec = action_resources(action)
        value = 1.0
        if spec.uses_rf:
            value *= float(attempt_failures[row]) ** spec.reserved_rf_attempts
        if spec.uses_vlc:
            value *= float(problem.vlc_failure_probabilities[row])
        terms.append(value)
    return math.fsum(terms)


def _brute_force(
    problem: JointRiskOracleProblem,
) -> tuple[float, float, int, tuple[PolicyAction, ...]]:
    choices = tuple(
        problem.allowed_actions if usable else (problem.fallback_action,)
        for usable in problem.usable_mask
    )
    best: tuple[float, float, int, tuple[PolicyAction, ...]] | None = None
    for actions in itertools.product(*choices):
        total_risk = _risk(problem, actions)
        total_cost = math.fsum(
            problem.resource_map.activation_cost(action) for action in actions
        )
        attempts = sum(
            action_resources(action).reserved_rf_attempts for action in actions
        )
        candidate = (total_risk, total_cost, attempts, actions)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best


@pytest.mark.parametrize("usable", [(True, True, True), (True, False, True)])
def test_exact_solver_matches_exhaustive_joint_action_search(
    usable: tuple[bool, ...],
) -> None:
    problem = _problem(usable=usable)

    solution = solve_joint_risk_floor(problem)
    brute = _brute_force(problem)

    assert solution.total_conditional_miss_risk == pytest.approx(brute[0], abs=1e-12)
    assert solution.total_activation_cost == pytest.approx(brute[1])
    assert solution.total_rf_attempts == brute[2]
    assert solution.actions == brute[3]
    assert solution.candidate_loads_evaluated + solution.candidate_loads_pruned == (
        solution.candidate_loads_total
    )
    assert solution.action_counts == tuple(
        solution.actions.count(action) for action in PolicyAction
    )


def test_all_unusable_population_is_forced_to_fallback() -> None:
    problem = _problem(usable=(False, False, False))

    solution = solve_joint_risk_floor(problem)

    assert solution.actions == (PolicyAction.DUP_4,) * 3
    assert solution.total_rf_attempts == 12
    assert solution.candidate_loads_total == 1
    assert solution.candidate_loads_evaluated == 1
    assert solution.candidate_loads_pruned == 0
    assert solution.usable_conditional_miss_risk == 0.0
    assert solution.forced_fallback_conditional_miss_risk == pytest.approx(
        solution.total_conditional_miss_risk
    )


@pytest.mark.parametrize("seed", range(10))
def test_sorted_marginal_allocator_matches_random_exhaustive_search(seed: int) -> None:
    rng = np.random.default_rng(seed)
    problem = replace(
        _problem(),
        rf_decoding_failure_probabilities=rng.uniform(0.0, 1.0, 3).astype(
            np.float64
        ),
        vlc_failure_probabilities=rng.uniform(0.0, 1.0, 3).astype(np.float64),
    )

    solution = solve_joint_risk_floor(problem)
    brute = _brute_force(problem)

    assert solution.total_conditional_miss_risk == pytest.approx(brute[0], abs=1e-12)
    assert solution.total_activation_cost == pytest.approx(brute[1])
    assert solution.total_rf_attempts == brute[2]
    assert solution.actions == brute[3]


def test_problem_rejects_a_noncontiguous_action_ladder() -> None:
    base = _problem()

    with pytest.raises(JointRiskOracleError, match="contiguous"):
        JointRiskOracleProblem(
            pair_ids=base.pair_ids,
            usable_mask=base.usable_mask,
            allowed_actions=(PolicyAction.VLC, PolicyAction.RF_2),
            fallback_action=PolicyAction.VLC,
            resource_map=base.resource_map,
            pool_model=base.pool_model,
            rf_decoding_failure_probabilities=base.rf_decoding_failure_probabilities,
            vlc_failure_probabilities=base.vlc_failure_probabilities,
        )
