"""Phase 7 clipped PPO actor and independent critic updates."""

from __future__ import annotations

import math
from dataclasses import astuple, replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hybrid_v2x_rl.agents.masked_categorical import (
    ACTION_COUNT,
    CategoricalActionBatch,
    SharedCategoricalActor,
)
from hybrid_v2x_rl.agents.ppo import (
    PPOBatch,
    PPOError,
    PPOUpdater,
    ScalarCritic,
    clipped_policy_surrogate,
    mean_squared_value_loss,
    ppo_loss_terms,
)
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.config.models import NetworkArchitectureConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _batch(
    *,
    batch_size: int = 8,
    actor_width: int = 3,
    critic_width: int = 5,
) -> PPOBatch:
    actor_observations = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * actor_width,
    ).reshape(batch_size, actor_width)
    critic_observations = torch.linspace(
        1.0,
        -1.0,
        steps=batch_size * critic_width,
    ).reshape(batch_size, critic_width)
    return PPOBatch(
        actor_observations=actor_observations,
        critic_observations=critic_observations,
        action_masks=torch.ones((batch_size, ACTION_COUNT), dtype=torch.bool),
        actions=torch.arange(batch_size, dtype=torch.long) % ACTION_COUNT,
        old_log_probabilities=torch.full((batch_size,), -2.0),
        reward_advantages=torch.linspace(0.25, 2.0, steps=batch_size),
        cost_advantages=torch.linspace(0.1, 0.8, steps=batch_size),
        reward_value_targets=torch.linspace(-2.0, -0.25, steps=batch_size),
        cost_value_targets=torch.linspace(0.05, 0.4, steps=batch_size),
        cost_penalty_weights=torch.zeros(batch_size),
    )


def test_clipped_surrogate_handles_both_advantage_signs() -> None:
    ratios = torch.tensor([1.5, 0.5, 0.5, 1.5], requires_grad=True)
    old_log_probabilities = torch.full((4,), -2.0)
    new_log_probabilities = old_log_probabilities + ratios.log()
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])

    surrogate = clipped_policy_surrogate(
        new_log_probabilities=new_log_probabilities,
        old_log_probabilities=old_log_probabilities,
        advantages=advantages,
        clip_ratio=0.2,
    )
    surrogate.loss.backward()

    torch.testing.assert_close(
        surrogate.ratios.detach(),
        torch.tensor([1.5, 0.5, 0.5, 1.5]),
    )
    torch.testing.assert_close(
        surrogate.clipped_ratios.detach(),
        torch.tensor([1.2, 0.8, 0.8, 1.2]),
    )
    assert surrogate.loss.item() == pytest.approx(0.15)
    assert surrogate.clip_fraction.item() == pytest.approx(1.0)
    assert ratios.grad is not None
    assert ratios.grad[:2].count_nonzero().item() == 0
    assert ratios.grad[2].item() < 0.0
    assert ratios.grad[3].item() > 0.0


def test_probability_ratio_one_reduces_to_negative_mean_advantage() -> None:
    advantages = torch.tensor([2.0, -1.0, 0.5])

    surrogate = clipped_policy_surrogate(
        new_log_probabilities=torch.tensor([-0.5, -1.0, -2.0]),
        old_log_probabilities=torch.tensor([-0.5, -1.0, -2.0]),
        advantages=advantages,
        clip_ratio=0.2,
    )

    assert surrogate.ratios == pytest.approx(torch.ones(3))
    assert surrogate.loss.item() == pytest.approx(-advantages.mean().item())
    assert surrogate.approximate_kl.item() == pytest.approx(0.0)
    assert surrogate.clip_fraction.item() == pytest.approx(0.0)


def test_cost_penalty_is_subtracted_before_policy_clipping() -> None:
    batch = replace(
        _batch(batch_size=2),
        old_log_probabilities=torch.zeros(2),
        reward_advantages=torch.tensor([2.0, 2.0]),
        cost_advantages=torch.tensor([1.0, 0.5]),
        cost_penalty_weights=torch.tensor([3.0, 2.0]),
    )
    evaluation = CategoricalActionBatch(
        actions=batch.actions,
        log_probabilities=torch.zeros(2, requires_grad=True),
        entropy=torch.tensor([0.5, 0.5], requires_grad=True),
    )

    losses = ppo_loss_terms(
        batch=batch,
        action_evaluation=evaluation,
        reward_predictions=torch.zeros(2, requires_grad=True),
        cost_predictions=torch.zeros(2, requires_grad=True),
        clip_ratio=0.2,
        entropy_coefficient=0.1,
    )

    assert batch.combined_advantages == pytest.approx(torch.tensor([-1.0, 1.0]))
    assert losses.policy_loss.item() == pytest.approx(0.0)
    assert losses.actor_loss.item() == pytest.approx(-0.05)


def test_reward_and_cost_critic_losses_use_only_their_own_targets() -> None:
    reward_predictions = torch.tensor([1.0, 3.0], requires_grad=True)
    cost_predictions = torch.tensor([4.0, 7.0], requires_grad=True)
    reward_loss = mean_squared_value_loss(
        predictions=reward_predictions,
        targets=torch.tensor([2.0, 1.0]),
    )
    cost_loss = mean_squared_value_loss(
        predictions=cost_predictions,
        targets=torch.tensor([4.0, 3.0]),
    )

    reward_loss.backward()

    assert reward_loss.item() == pytest.approx(2.5)
    assert cost_loss.item() == pytest.approx(8.0)
    assert reward_predictions.grad is not None
    assert cost_predictions.grad is None


def test_scalar_critics_use_configured_independent_two_by_64_tanh_networks() -> None:
    architecture = NetworkArchitectureConfig()
    reward = ScalarCritic.from_config(
        observation_width=78,
        architecture=architecture,
        role="reward",
    )
    cost = ScalarCritic.from_config(
        observation_width=78,
        architecture=architecture,
        role="cost",
    )

    for critic in (reward, cost):
        modules = list(critic.network)
        assert [layer.in_features for layer in modules if isinstance(layer, nn.Linear)] == [
            78,
            64,
            64,
        ]
        assert [layer.out_features for layer in modules if isinstance(layer, nn.Linear)] == [
            64,
            64,
            1,
        ]
        assert sum(isinstance(layer, nn.Tanh) for layer in modules) == 2
        assert critic(torch.zeros((4, 78))).shape == (4,)
    assert all(
        reward_parameter is not cost_parameter
        for reward_parameter, cost_parameter in zip(
            reward.parameters(),
            cost.parameters(),
            strict=True,
        )
    )


def test_one_update_changes_actor_and_both_critics() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(71)
        actor = SharedCategoricalActor(observation_width=3, hidden_units=(8, 8))
        reward_critic = ScalarCritic(observation_width=5, hidden_units=(8, 8))
        cost_critic = ScalarCritic(observation_width=5, hidden_units=(8, 8))
        batch = _batch()
        with torch.no_grad():
            old = actor.evaluate_actions(
                batch.actor_observations,
                batch.action_masks,
                batch.actions,
            ).log_probabilities
        batch = replace(batch, old_log_probabilities=old)
        updater = PPOUpdater(
            actor=actor,
            reward_critic=reward_critic,
            cost_critic=cost_critic,
            learning_rate=1e-2,
            clip_ratio=0.2,
            entropy_coefficient=0.01,
        )
        before = {
            "actor": tuple(parameter.detach().clone() for parameter in actor.parameters()),
            "reward": tuple(parameter.detach().clone() for parameter in reward_critic.parameters()),
            "cost": tuple(parameter.detach().clone() for parameter in cost_critic.parameters()),
        }

        metrics = updater.update(batch)

    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before["actor"], actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before["reward"], reward_critic.parameters(), strict=True)
    )
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before["cost"], cost_critic.parameters(), strict=True)
    )
    assert metrics.approximate_kl == pytest.approx(0.0, abs=1e-6)
    assert metrics.clip_fraction == pytest.approx(0.0)
    assert metrics.minibatch_size == batch.batch_size
    assert all(math.isfinite(value) for value in astuple(metrics))


def test_updater_from_config_binds_headline_hyperparameters_and_widths() -> None:
    config = load_headline_config(PROJECT_ROOT)

    with torch.random.fork_rng():
        updater = PPOUpdater.from_config(
            actor_observation_width=37,
            critic_observation_width=78,
            training=config.training,
        )

    assert updater.actor.observation_width == 37
    assert updater.reward_critic.observation_width == 78
    assert updater.cost_critic.observation_width == 78
    assert updater.learning_rate == pytest.approx(config.training.learning_rate)
    assert updater.clip_ratio == pytest.approx(config.training.clip_ratio)
    assert updater.entropy_coefficient == pytest.approx(config.training.entropy_coefficient)


def test_updater_rejects_actor_or_critic_width_drift() -> None:
    updater = PPOUpdater(
        actor=SharedCategoricalActor(observation_width=4),
        reward_critic=ScalarCritic(observation_width=5),
        cost_critic=ScalarCritic(observation_width=5),
        learning_rate=3e-4,
        clip_ratio=0.2,
        entropy_coefficient=0.01,
    )

    with pytest.raises(PPOError, match="actor observation width"):
        updater.compute_losses(_batch(actor_width=3, critic_width=5))


def test_scalar_critic_rejects_an_unknown_runtime_role() -> None:
    with pytest.raises(PPOError, match="role"):
        ScalarCritic.from_config(
            observation_width=78,
            architecture=NetworkArchitectureConfig(),
            role="constraint",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("action_masks", torch.zeros((8, ACTION_COUNT), dtype=torch.bool), "allow"),
        ("actions", torch.full((8,), ACTION_COUNT, dtype=torch.long), "outside"),
        ("old_log_probabilities", torch.ones(8), "positive"),
        ("cost_penalty_weights", torch.full((8,), -1.0), "nonnegative"),
        ("reward_advantages", torch.zeros(7), "one value"),
        ("cost_value_targets", torch.full((8,), float("nan")), "non-finite"),
    ],
)
def test_ppo_batch_rejects_invalid_rollout_rows(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(PPOError, match=message):
        replace(_batch(), **{field: value})


def test_ppo_batch_rejects_a_selected_masked_action() -> None:
    batch = _batch()
    masks = batch.action_masks.clone()
    masks[0, batch.actions[0]] = False

    with pytest.raises(PPOError, match="masked"):
        replace(batch, action_masks=masks)


@pytest.mark.parametrize(
    ("clip_ratio", "message"),
    [
        (0.0, "positive"),
        (1.0, r"\(0, 1\)"),
        (float("nan"), "positive"),
        (True, "positive"),
    ],
)
def test_clipped_surrogate_rejects_invalid_ratios(
    clip_ratio: float,
    message: str,
) -> None:
    with pytest.raises(PPOError, match=message):
        clipped_policy_surrogate(
            new_log_probabilities=torch.zeros(2),
            old_log_probabilities=torch.zeros(2),
            advantages=torch.ones(2),
            clip_ratio=clip_ratio,
        )


def test_value_loss_rejects_gradient_carrying_targets() -> None:
    with pytest.raises(PPOError, match="detached"):
        mean_squared_value_loss(
            predictions=torch.zeros(2, requires_grad=True),
            targets=torch.ones(2, requires_grad=True),
        )


def test_clipped_surrogate_rejects_positive_categorical_log_probability() -> None:
    with pytest.raises(PPOError, match="cannot be positive"):
        clipped_policy_surrogate(
            new_log_probabilities=torch.tensor([0.1]),
            old_log_probabilities=torch.tensor([-1.0]),
            advantages=torch.ones(1),
            clip_ratio=0.2,
        )


def test_empty_ppo_minibatches_are_rejected() -> None:
    with pytest.raises(PPOError, match="empty"):
        _batch(batch_size=0)
