"""End-to-end convergence on a tiny deterministic constrained problem."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from hybrid_v2x_rl.agents.advantages import reward_cost_generalized_advantage_estimate
from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.masked_categorical import (
    ACTION_COUNT,
    MaskedCategorical,
    SharedCategoricalActor,
)
from hybrid_v2x_rl.agents.ppo import PPOBatch, PPOUpdater, ScalarCritic
from hybrid_v2x_rl.config.models import DensityMultiplierConfig

RISKY_ACTION = 0
SAFE_ACTION = 1
DENSITY_VEH_PER_LANE_KM = 10.0
MISS_BUDGET = 0.1
INITIAL_DUAL = 0.25
BATCH_SIZE = 128
ITERATIONS = 50
UPDATE_EPOCHS = 4


@dataclass(frozen=True, slots=True)
class TinyCMDP:
    """One-state CMDP whose reward-only and feasible optima disagree.

    The risky action earns reward 1 but always violates reliability.  The safe
    action earns reward 0.8 and has zero miss cost.  Thus reward-only learning
    selects risky, while the best deterministic feasible policy selects safe.
    The remaining seven hardware actions are masked.
    """

    def observations(self, rows: int) -> torch.Tensor:
        return torch.ones((rows, 1), dtype=torch.float32)

    def action_masks(self, rows: int) -> torch.Tensor:
        masks = torch.zeros((rows, ACTION_COUNT), dtype=torch.bool)
        masks[:, RISKY_ACTION] = True
        masks[:, SAFE_ACTION] = True
        return masks

    def outcomes(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim != 1 or actions.dtype != torch.long:
            raise ValueError("tiny CMDP actions must be a torch.long vector")
        if bool(((actions != RISKY_ACTION) & (actions != SAFE_ACTION)).any().item()):
            raise ValueError("tiny CMDP received a masked action")
        rewards = torch.where(actions == RISKY_ACTION, 1.0, 0.8)
        miss_costs = (actions == RISKY_ACTION).to(torch.float32)
        return rewards, miss_costs


@dataclass(frozen=True, slots=True)
class TinyTrainingResult:
    initial_safe_probability: float
    risky_probability: float
    safe_probability: float
    invalid_probability: float
    deterministic_action: int
    final_reward_value: float
    final_cost_value: float
    recent_miss_rate: float
    dual_trajectory: tuple[float, ...]


def _learner(seed: int) -> tuple[PPOUpdater, PerDensityDualAscent]:
    torch.manual_seed(seed)
    actor = SharedCategoricalActor(observation_width=1, hidden_units=(8,))
    reward_critic = ScalarCritic(observation_width=1, hidden_units=(8,))
    cost_critic = ScalarCritic(observation_width=1, hidden_units=(8,))
    updater = PPOUpdater(
        actor=actor,
        reward_critic=reward_critic,
        cost_critic=cost_critic,
        learning_rate=0.003,
        clip_ratio=0.2,
        entropy_coefficient=0.005,
    )
    dual = PerDensityDualAscent(
        (
            DensityMultiplierConfig(
                density_veh_per_lane_km=DENSITY_VEH_PER_LANE_KM,
                initial_value=INITIAL_DUAL,
                learning_rate=0.05,
                maximum=2.0,
            ),
        )
    )
    return updater, dual


def _probabilities(
    environment: TinyCMDP,
    updater: PPOUpdater,
) -> torch.Tensor:
    observations = environment.observations(1)
    masks = environment.action_masks(1)
    with torch.no_grad():
        return MaskedCategorical(updater.actor(observations), masks).probabilities[0]


def _train_tiny(*, seed: int, constrained: bool) -> TinyTrainingResult:
    environment = TinyCMDP()
    updater, dual = _learner(seed)
    policy_generator = torch.Generator().manual_seed(seed + 10_000)
    observations = environment.observations(BATCH_SIZE)
    action_masks = environment.action_masks(BATCH_SIZE)
    densities = torch.full((BATCH_SIZE,), DENSITY_VEH_PER_LANE_KM)
    active = torch.ones((1, BATCH_SIZE), dtype=torch.bool)
    no_bootstrap = torch.zeros_like(active)
    initial_safe_probability = float(_probabilities(environment, updater)[SAFE_ACTION].item())
    miss_history: list[float] = []
    dual_trajectory = [dual.multiplier_for_density(DENSITY_VEH_PER_LANE_KM)]

    for _ in range(ITERATIONS):
        with torch.no_grad():
            selected = updater.actor.select(
                observations,
                action_masks,
                generator=policy_generator,
            )
            reward_values = updater.reward_critic(observations)
            cost_values = updater.cost_critic(observations)
        rewards, miss_costs = environment.outcomes(selected.actions)
        estimates = reward_cost_generalized_advantage_estimate(
            rewards=rewards.unsqueeze(0),
            costs=miss_costs.unsqueeze(0),
            reward_values=reward_values.unsqueeze(0),
            reward_next_values=torch.zeros_like(reward_values).unsqueeze(0),
            cost_values=cost_values.unsqueeze(0),
            cost_next_values=torch.zeros_like(cost_values).unsqueeze(0),
            value_bootstrap_mask=no_bootstrap,
            gae_continuation_mask=no_bootstrap,
            active_mask=active,
            gamma=0.99,
            gae_lambda=0.95,
            time_dimension=0,
        )
        penalty_weights = (
            dual.penalty_weights(densities)
            if constrained
            else torch.zeros(BATCH_SIZE, dtype=torch.float32)
        )
        batch = PPOBatch(
            actor_observations=observations,
            critic_observations=observations,
            action_masks=action_masks,
            actions=selected.actions,
            old_log_probabilities=selected.log_probabilities,
            reward_advantages=estimates.reward.advantages[0],
            cost_advantages=estimates.cost.advantages[0],
            reward_value_targets=estimates.reward.value_targets[0],
            cost_value_targets=estimates.cost.value_targets[0],
            cost_penalty_weights=penalty_weights,
        )
        for _ in range(UPDATE_EPOCHS):
            updater.update(batch)
        if constrained:
            dual.update(
                densities_veh_per_lane_km=densities,
                costs=miss_costs,
                miss_budget=MISS_BUDGET,
            )
        miss_history.append(float(miss_costs.mean().item()))
        dual_trajectory.append(dual.multiplier_for_density(DENSITY_VEH_PER_LANE_KM))

    probabilities = _probabilities(environment, updater)
    probe_observation = environment.observations(1)
    probe_mask = environment.action_masks(1)
    with torch.no_grad():
        deterministic_action = int(
            updater.actor.select(
                probe_observation,
                probe_mask,
                deterministic=True,
            ).actions.item()
        )
        reward_value = float(updater.reward_critic(probe_observation).item())
        cost_value = float(updater.cost_critic(probe_observation).item())
    return TinyTrainingResult(
        initial_safe_probability=initial_safe_probability,
        risky_probability=float(probabilities[RISKY_ACTION].item()),
        safe_probability=float(probabilities[SAFE_ACTION].item()),
        invalid_probability=float(probabilities[2:].sum().item()),
        deterministic_action=deterministic_action,
        final_reward_value=reward_value,
        final_cost_value=cost_value,
        recent_miss_rate=sum(miss_history[-10:]) / 10.0,
        dual_trajectory=tuple(dual_trajectory),
    )


def test_primal_dual_ppo_converges_to_known_feasible_tiny_policy() -> None:
    constrained = _train_tiny(seed=1001, constrained=True)
    reward_only = _train_tiny(seed=1001, constrained=False)

    assert constrained.safe_probability > 0.97
    assert constrained.safe_probability > constrained.initial_safe_probability + 0.25
    assert constrained.deterministic_action == SAFE_ACTION
    assert constrained.invalid_probability == 0.0
    assert constrained.recent_miss_rate < 0.05 < MISS_BUDGET
    assert max(constrained.dual_trajectory) > INITIAL_DUAL
    assert constrained.dual_trajectory[-1] > 0.1
    assert constrained.final_reward_value == pytest.approx(0.8, abs=0.05)
    assert constrained.final_cost_value == pytest.approx(0.0, abs=0.08)

    # The identical initialization without the reliability penalty learns the
    # higher-reward but infeasible action, proving that the constraint changes
    # the solution rather than merely accelerating ordinary reward learning.
    assert reward_only.risky_probability > 0.97
    assert reward_only.deterministic_action == RISKY_ACTION
    assert reward_only.recent_miss_rate > 0.95
    assert reward_only.invalid_probability == 0.0
