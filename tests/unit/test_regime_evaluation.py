"""Regime-conditioned PPO feasibility, risk, and resource diagnostics."""

from __future__ import annotations

import pytest

from hybrid_v2x_rl.agents.regime_evaluation import (
    POLICY_INDUCED_LOAD,
    PPORegimeEvaluationError,
    RegimeEvaluationAccumulator,
    _PendingRow,
)
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.state_regime_audit import (
    LOAD_PROFILES,
    CounterfactualAction,
    CounterfactualActionSet,
)


def _assessment(name: str) -> CounterfactualActionSet:
    actions = tuple(
        CounterfactualAction(
            action=action,
            conditional_miss_probability=(0.001 if action is PolicyAction.VLC else 0.1),
            activation_cost=float(int(action) + 1),
            feasible=action is PolicyAction.VLC,
        )
        for action in PolicyAction
    )
    return CounterfactualActionSet(
        load_profile=name,
        other_pair_rf_attempts=7,
        actions=actions,
        selected_action=PolicyAction.VLC,
        any_feasible=True,
    )


def test_regime_accumulator_separates_feasibility_risk_and_resource_regret() -> None:
    accumulator = RegimeEvaluationAccumulator()
    probabilities = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assessments = {
        POLICY_INDUCED_LOAD: _assessment(POLICY_INDUCED_LOAD),
        **{name: _assessment(name) for name in LOAD_PROFILES},
    }
    accumulator.observe_policy(
        trace_id="validation-trace",
        pair_id="pair-1",
        density=20.0,
        labels=("easy_state",),
        probabilities=probabilities,
        selected_action=PolicyAction.VLC,
        assessments=assessments,
    )
    accumulator.observe_actual(
        _PendingRow(
            labels=("easy_state",),
            density=20.0,
            selected_action=PolicyAction.VLC,
            selected_risk=0.001,
            selected_resource_cost=1.0,
            miss_budget=0.01,
        ),
        conditional_risk=0.001,
        sampled_miss=0,
    )

    campaign = accumulator.campaign_rows()[0]
    actual = campaign["actual_policy_load"]
    profiles = campaign["counterfactuals"]
    assert isinstance(actual, dict)
    assert isinstance(profiles, dict)
    induced = profiles[POLICY_INDUCED_LOAD]
    assert campaign["rows"] == 1
    assert campaign["mean_action_probabilities"] == {
        "VLC": 0.5,
        "RF-1": 0.5,
        "RF-2": 0.0,
        "RF-3": 0.0,
        "RF-4": 0.0,
        "DUP-1": 0.0,
        "DUP-2": 0.0,
        "DUP-3": 0.0,
        "DUP-4": 0.0,
    }
    assert actual["mean_selected_conditional_miss_risk"] == pytest.approx(0.001)
    assert actual["selected_feasible_fraction"] == pytest.approx(1.0)
    assert induced["mean_feasible_action_probability_mass"] == pytest.approx(0.5)
    assert induced["mean_infeasible_action_probability_mass"] == pytest.approx(0.5)
    assert induced["mean_policy_expected_conditional_miss_risk"] == pytest.approx(
        0.0505
    )
    assert induced["mean_feasible_conditioned_resource_regret"] == pytest.approx(0.0)


def test_regime_accumulator_rejects_actual_risk_that_does_not_match_joint_load() -> None:
    accumulator = RegimeEvaluationAccumulator()
    assessments = {
        POLICY_INDUCED_LOAD: _assessment(POLICY_INDUCED_LOAD),
        **{name: _assessment(name) for name in LOAD_PROFILES},
    }
    accumulator.observe_policy(
        trace_id="validation-trace",
        pair_id="pair-1",
        density=20.0,
        labels=("easy_state",),
        probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        selected_action=PolicyAction.VLC,
        assessments=assessments,
    )

    with pytest.raises(PPORegimeEvaluationError, match="policy-load counterfactual"):
        accumulator.observe_actual(
            _PendingRow(
                labels=("easy_state",),
                density=20.0,
                selected_action=PolicyAction.VLC,
                selected_risk=0.001,
                selected_resource_cost=1.0,
                miss_budget=0.01,
            ),
            conditional_risk=0.5,
            sampled_miss=0,
        )
