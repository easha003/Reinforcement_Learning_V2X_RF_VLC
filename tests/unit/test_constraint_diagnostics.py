"""Read-only constrained-PPO pressure diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from hybrid_v2x_rl.agents.constraint_diagnostics import (
    CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA,
    ConstraintDiagnosticsError,
    build_constraint_pressure_diagnostics,
)
from hybrid_v2x_rl.agents.masked_categorical import ACTION_COUNT, SharedCategoricalActor
from hybrid_v2x_rl.agents.ppo import PPOBatch


def _actor() -> SharedCategoricalActor:
    actor = SharedCategoricalActor(observation_width=2, hidden_units=(3,))
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    return actor


def _batch(actor: SharedCategoricalActor) -> PPOBatch:
    observations = torch.tensor(
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.5, -0.5]],
        dtype=torch.float32,
    )
    masks = torch.ones((4, ACTION_COUNT), dtype=torch.bool)
    masks[1, 2:] = False
    actions = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    with torch.no_grad():
        old = actor.evaluate_actions(observations, masks, actions).log_probabilities
    return PPOBatch(
        actor_observations=observations,
        critic_observations=torch.zeros((4, 3), dtype=torch.float32),
        action_masks=masks,
        actions=actions,
        old_log_probabilities=old,
        reward_advantages=torch.tensor([1.0, -1.0, 2.0, -2.0]),
        cost_advantages=torch.tensor([0.5, 0.25, -0.5, -0.25]),
        reward_value_targets=torch.zeros(4),
        cost_value_targets=torch.zeros(4),
        cost_penalty_weights=torch.tensor([0.2, 0.2, 2.0, 2.0]),
    )


def test_constraint_pressure_reports_advantage_scale_and_policy_mix() -> None:
    actor = _actor()
    diagnostics = build_constraint_pressure_diagnostics(
        actor=actor,
        batch=_batch(actor),
        learning_densities=torch.tensor([10.0, 10.0, 20.0, 20.0]),
    )
    payload = diagnostics.as_dict()

    assert payload["schema"] == CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA
    assert diagnostics.learning_rows == 4
    assert [row.density_veh_per_lane_km for row in diagnostics.densities] == [
        10.0,
        20.0,
    ]
    low, high = diagnostics.densities
    assert low.dual_multiplier_before_update == pytest.approx(0.2)
    assert low.reward_advantage.mean == pytest.approx(0.0)
    assert low.reward_advantage.mean_absolute == pytest.approx(1.0)
    assert low.cost_advantage.mean == pytest.approx(0.375)
    assert low.dual_weighted_cost_advantage.mean_absolute == pytest.approx(0.075)
    assert low.mean_absolute_constraint_to_reward_ratio == pytest.approx(0.075)
    assert high.mean_absolute_constraint_to_reward_ratio == pytest.approx(0.375)
    assert low.positive_combined_advantage_fraction == pytest.approx(0.5)
    assert high.positive_combined_advantage_fraction == pytest.approx(0.5)
    assert low.entropy.mean == pytest.approx(
        0.5 * (math.log(ACTION_COUNT) + math.log(2.0))
    )
    assert low.mean_action_probabilities == pytest.approx(
        (11.0 / 36.0, 11.0 / 36.0, *(1.0 / 18.0,) * 7)
    )
    assert low.action_availability_fractions == (1.0, 1.0, *(0.5,) * 7)
    assert low.selected_action_counts[:4] == (1, 1, 0, 0)
    assert high.selected_action_counts[:4] == (0, 0, 1, 1)


def test_constraint_pressure_does_not_mutate_model_or_global_rng() -> None:
    actor = _actor()
    actor.train()
    batch = _batch(actor)
    before_model = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    before_rng = torch.random.get_rng_state().clone()

    build_constraint_pressure_diagnostics(
        actor=actor,
        batch=batch,
        learning_densities=torch.tensor([10.0, 10.0, 20.0, 20.0]),
    )

    assert actor.training
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    for name, value in actor.state_dict().items():
        assert torch.equal(value, before_model[name])


def test_constraint_pressure_rejects_an_actor_that_differs_from_rollout_policy() -> None:
    actor = _actor()
    batch = _batch(actor)
    with torch.no_grad():
        final_layer = actor.network[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        final_layer.bias[0] = 2.0

    with pytest.raises(ConstraintDiagnosticsError, match="rollout policy"):
        build_constraint_pressure_diagnostics(
            actor=actor,
            batch=batch,
            learning_densities=torch.tensor([10.0, 10.0, 20.0, 20.0]),
        )
