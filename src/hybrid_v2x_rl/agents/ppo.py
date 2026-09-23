"""Clipped PPO actor and independent reward/cost critic updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

import torch
import torch.nn as nn

from hybrid_v2x_rl.agents.masked_categorical import (
    ACTION_COUNT,
    CategoricalActionBatch,
    SharedCategoricalActor,
)
from hybrid_v2x_rl.config.models import NetworkArchitectureConfig, TrainingConfig
from hybrid_v2x_rl.core.errors import HybridV2XError


class PPOError(HybridV2XError):
    """A PPO minibatch, loss, network, or optimizer update is invalid."""


@dataclass(frozen=True, slots=True)
class PPOBatch:
    """One flattened minibatch containing only active, learning-usable rows."""

    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    action_masks: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    reward_advantages: torch.Tensor
    cost_advantages: torch.Tensor
    reward_value_targets: torch.Tensor
    cost_value_targets: torch.Tensor
    cost_penalty_weights: torch.Tensor

    def __post_init__(self) -> None:
        actor = self.actor_observations
        critic = self.critic_observations
        if not isinstance(actor, torch.Tensor) or actor.ndim != 2 or actor.shape[1] < 1:
            raise PPOError("actor observations must have shape (batch, actor_width)")
        batch_size = actor.shape[0]
        if batch_size == 0:
            raise PPOError("a PPO minibatch cannot be empty")
        if not isinstance(critic, torch.Tensor) or critic.ndim != 2 or critic.shape[1] < 1:
            raise PPOError("critic observations must have shape (batch, critic_width)")
        if critic.shape[0] != batch_size:
            raise PPOError("actor and critic observations must share the batch dimension")

        float_tensors = (
            ("actor_observations", actor),
            ("critic_observations", critic),
            ("old_log_probabilities", self.old_log_probabilities),
            ("reward_advantages", self.reward_advantages),
            ("cost_advantages", self.cost_advantages),
            ("reward_value_targets", self.reward_value_targets),
            ("cost_value_targets", self.cost_value_targets),
            ("cost_penalty_weights", self.cost_penalty_weights),
        )
        for name, values in float_tensors:
            if not isinstance(values, torch.Tensor) or not values.is_floating_point():
                raise PPOError(f"{name} must be a floating-point torch.Tensor")
            expected_shape = actor.shape if name == "actor_observations" else None
            if name == "critic_observations":
                expected_shape = critic.shape
            if expected_shape is None and values.shape != (batch_size,):
                raise PPOError(
                    f"{name} must have one value per minibatch row",
                    context={"actual": tuple(values.shape), "expected": (batch_size,)},
                )
            if values.dtype != actor.dtype:
                raise PPOError(f"{name} must use the actor observation dtype")
            if values.device != actor.device:
                raise PPOError(f"{name} must use the actor observation device")
            if values.requires_grad:
                raise PPOError(f"{name} must be detached rollout data")
            if not bool(torch.isfinite(values).all().item()):
                raise PPOError(f"{name} contains a non-finite value")

        if not isinstance(self.action_masks, torch.Tensor):
            raise PPOError("action masks must be a torch.Tensor")
        if self.action_masks.shape != (batch_size, ACTION_COUNT):
            raise PPOError(
                "action masks must have one canonical mask per minibatch row",
                context={
                    "actual": tuple(self.action_masks.shape),
                    "expected": (batch_size, ACTION_COUNT),
                },
            )
        if self.action_masks.dtype != torch.bool:
            raise PPOError("action masks must use torch.bool")
        if self.action_masks.device != actor.device:
            raise PPOError("action masks must use the actor observation device")
        if not bool(self.action_masks.any(dim=1).all().item()):
            raise PPOError("every PPO row must allow at least one action")

        if not isinstance(self.actions, torch.Tensor):
            raise PPOError("actions must be a torch.Tensor")
        if self.actions.shape != (batch_size,) or self.actions.dtype != torch.long:
            raise PPOError("actions must be a torch.long vector aligned with the minibatch")
        if self.actions.device != actor.device:
            raise PPOError("actions must use the actor observation device")
        in_range = (self.actions >= 0) & (self.actions < ACTION_COUNT)
        if not bool(in_range.all().item()):
            raise PPOError("actions contain an index outside the policy action space")
        selected_allowed = self.action_masks.gather(1, self.actions.unsqueeze(1)).squeeze(1)
        if not bool(selected_allowed.all().item()):
            raise PPOError("actions contain a masked selection")

        if bool((self.old_log_probabilities > 1e-6).any().item()):
            raise PPOError("old log probabilities cannot be positive")
        if bool((self.cost_penalty_weights < 0.0).any().item()):
            raise PPOError("cost penalty weights must be nonnegative")

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def combined_advantages(self) -> torch.Tensor:
        """Reward advantage minus the externally supplied reliability penalty."""

        return self.reward_advantages - self.cost_penalty_weights * self.cost_advantages


class ScalarCritic(nn.Module):
    """Feed-forward scalar value function for centralized critic observations."""

    def __init__(
        self,
        *,
        observation_width: int,
        hidden_units: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        _validate_width("observation_width", observation_width)
        if not hidden_units:
            raise PPOError("critic hidden_units cannot be empty")
        for width in hidden_units:
            _validate_width("critic hidden width", width)

        self.observation_width = observation_width
        self.hidden_units = hidden_units
        layers: list[nn.Module] = []
        input_width = observation_width
        for output_width in hidden_units:
            layers.extend((nn.Linear(input_width, output_width), nn.Tanh()))
            input_width = output_width
        layers.append(nn.Linear(input_width, 1))
        self.network = nn.Sequential(*layers)

    @classmethod
    def from_config(
        cls,
        *,
        observation_width: int,
        architecture: NetworkArchitectureConfig,
        role: Literal["reward", "cost"],
    ) -> ScalarCritic:
        if role not in ("reward", "cost"):
            raise PPOError("critic role must be reward or cost")
        if architecture.activation != "tanh":
            raise PPOError("the Phase 7 critics require tanh activation")
        if architecture.recurrent:
            raise PPOError("the initial Phase 7 critics must be feed-forward")
        hidden_units = (
            architecture.reward_critic_hidden_units
            if role == "reward"
            else architecture.cost_critic_hidden_units
        )
        return cls(observation_width=observation_width, hidden_units=hidden_units)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor):
            raise PPOError("critic observations must be a torch.Tensor")
        if observations.ndim != 2 or observations.shape[1] != self.observation_width:
            raise PPOError(
                "critic observations have the wrong shape",
                context={
                    "actual": tuple(observations.shape),
                    "expected": ("batch", self.observation_width),
                },
            )
        if not observations.is_floating_point():
            raise PPOError("critic observations must be floating point")
        if not bool(torch.isfinite(observations).all().item()):
            raise PPOError("critic observations contain a non-finite value")
        predictions = cast(torch.Tensor, self.network(observations))
        return predictions.squeeze(-1)


@dataclass(frozen=True, slots=True)
class ClippedPolicySurrogate:
    """Differentiable PPO surrogate plus pre-update diagnostics."""

    loss: torch.Tensor
    ratios: torch.Tensor
    clipped_ratios: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass(frozen=True, slots=True)
class PPOLossTerms:
    """Differentiable losses for one actor and two independent critics."""

    actor_loss: torch.Tensor
    policy_loss: torch.Tensor
    reward_value_loss: torch.Tensor
    cost_value_loss: torch.Tensor
    entropy: torch.Tensor
    approximate_kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("actor_loss", self.actor_loss),
            ("policy_loss", self.policy_loss),
            ("reward_value_loss", self.reward_value_loss),
            ("cost_value_loss", self.cost_value_loss),
            ("entropy", self.entropy),
            ("approximate_kl", self.approximate_kl),
            ("clip_fraction", self.clip_fraction),
            ("ratio_mean", self.ratio_mean),
        ):
            if not isinstance(value, torch.Tensor) or value.ndim != 0:
                raise PPOError(f"{name} must be a scalar tensor")
            if not bool(torch.isfinite(value).item()):
                raise PPOError(f"{name} is non-finite")


@dataclass(frozen=True, slots=True)
class PPOUpdateMetrics:
    """Detached scalar result from one optimizer step."""

    actor_loss: float
    policy_loss: float
    reward_value_loss: float
    cost_value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    ratio_mean: float


def clipped_policy_surrogate(
    *,
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> ClippedPolicySurrogate:
    """Return the sign-correct clipped PPO surrogate for maximization."""

    ratio_limit = _validate_open_unit("clip_ratio", clip_ratio)
    _validate_loss_vector("new_log_probabilities", new_log_probabilities)
    _validate_loss_vector(
        "old_log_probabilities",
        old_log_probabilities,
        reference=new_log_probabilities,
    )
    _validate_loss_vector("advantages", advantages, reference=new_log_probabilities)
    if old_log_probabilities.requires_grad:
        raise PPOError("old log probabilities must be detached")
    if advantages.requires_grad:
        raise PPOError("advantages must be detached")
    if bool((new_log_probabilities > 1e-6).any().item()) or bool(
        (old_log_probabilities > 1e-6).any().item()
    ):
        raise PPOError("categorical log probabilities cannot be positive")

    log_ratios = new_log_probabilities - old_log_probabilities
    ratios = torch.exp(log_ratios)
    if not bool(torch.isfinite(ratios).all().item()):
        raise PPOError("policy probability ratios are non-finite")
    clipped_ratios = torch.clamp(ratios, 1.0 - ratio_limit, 1.0 + ratio_limit)
    unclipped_objective = ratios * advantages
    clipped_objective = clipped_ratios * advantages
    loss = -torch.minimum(unclipped_objective, clipped_objective).mean()
    approximate_kl = ((ratios - 1.0) - log_ratios).mean()
    clip_fraction = (torch.abs(ratios - 1.0) > ratio_limit).to(ratios.dtype).mean()
    return ClippedPolicySurrogate(
        loss=loss,
        ratios=ratios,
        clipped_ratios=clipped_ratios,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )


def mean_squared_value_loss(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean squared critic loss with detached GAE value targets."""

    _validate_loss_vector("predictions", predictions)
    _validate_loss_vector("targets", targets, reference=predictions)
    if targets.requires_grad:
        raise PPOError("critic value targets must be detached")
    loss = torch.mean(torch.square(predictions - targets))
    if not bool(torch.isfinite(loss).item()):
        raise PPOError("critic value loss is non-finite")
    return loss


def ppo_loss_terms(
    *,
    batch: PPOBatch,
    action_evaluation: CategoricalActionBatch,
    reward_predictions: torch.Tensor,
    cost_predictions: torch.Tensor,
    clip_ratio: float,
    entropy_coefficient: float,
) -> PPOLossTerms:
    """Assemble actor, reward-critic, and cost-critic minibatch losses."""

    if not isinstance(batch, PPOBatch):
        raise PPOError("PPO loss construction requires a PPOBatch")
    if not isinstance(action_evaluation, CategoricalActionBatch):
        raise PPOError("PPO loss construction requires evaluated categorical actions")
    if not torch.equal(action_evaluation.actions, batch.actions):
        raise PPOError("evaluated actions do not match the PPO minibatch")
    _validate_loss_vector(
        "new_log_probabilities",
        action_evaluation.log_probabilities,
        reference=batch.old_log_probabilities,
    )
    _validate_loss_vector(
        "entropy",
        action_evaluation.entropy,
        reference=batch.old_log_probabilities,
    )
    if bool((action_evaluation.entropy < -1e-6).any().item()):
        raise PPOError("categorical entropy cannot be negative")

    entropy_weight = _validate_nonnegative("entropy_coefficient", entropy_coefficient)
    policy = clipped_policy_surrogate(
        new_log_probabilities=action_evaluation.log_probabilities,
        old_log_probabilities=batch.old_log_probabilities,
        advantages=batch.combined_advantages,
        clip_ratio=clip_ratio,
    )
    entropy = action_evaluation.entropy.mean()
    actor_loss = policy.loss - entropy_weight * entropy
    reward_value_loss = mean_squared_value_loss(
        predictions=reward_predictions,
        targets=batch.reward_value_targets,
    )
    cost_value_loss = mean_squared_value_loss(
        predictions=cost_predictions,
        targets=batch.cost_value_targets,
    )
    return PPOLossTerms(
        actor_loss=actor_loss,
        policy_loss=policy.loss,
        reward_value_loss=reward_value_loss,
        cost_value_loss=cost_value_loss,
        entropy=entropy,
        approximate_kl=policy.approximate_kl,
        clip_fraction=policy.clip_fraction,
        ratio_mean=policy.ratios.mean(),
    )


class PPOUpdater:
    """Own three Adam optimizers and apply one validated PPO minibatch step."""

    def __init__(
        self,
        *,
        actor: SharedCategoricalActor,
        reward_critic: ScalarCritic,
        cost_critic: ScalarCritic,
        learning_rate: float,
        clip_ratio: float,
        entropy_coefficient: float,
    ) -> None:
        if not isinstance(actor, SharedCategoricalActor):
            raise PPOError("PPO updater requires a SharedCategoricalActor")
        if not isinstance(reward_critic, ScalarCritic) or not isinstance(cost_critic, ScalarCritic):
            raise PPOError("PPO updater requires separate ScalarCritic networks")
        if reward_critic is cost_critic:
            raise PPOError("reward and cost critics must have separate parameters")
        self.actor = actor
        self.reward_critic = reward_critic
        self.cost_critic = cost_critic
        self.learning_rate = _validate_positive("learning_rate", learning_rate)
        self.clip_ratio = _validate_open_unit("clip_ratio", clip_ratio)
        self.entropy_coefficient = _validate_nonnegative(
            "entropy_coefficient",
            entropy_coefficient,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )
        self.reward_critic_optimizer = torch.optim.Adam(
            self.reward_critic.parameters(),
            lr=self.learning_rate,
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(),
            lr=self.learning_rate,
        )

    @classmethod
    def from_config(
        cls,
        *,
        actor_observation_width: int,
        critic_observation_width: int,
        training: TrainingConfig,
    ) -> PPOUpdater:
        if not isinstance(training, TrainingConfig):
            raise PPOError("PPO updater requires a validated TrainingConfig")
        actor = SharedCategoricalActor.from_config(
            observation_width=actor_observation_width,
            architecture=training.architecture,
        )
        reward_critic = ScalarCritic.from_config(
            observation_width=critic_observation_width,
            architecture=training.architecture,
            role="reward",
        )
        cost_critic = ScalarCritic.from_config(
            observation_width=critic_observation_width,
            architecture=training.architecture,
            role="cost",
        )
        return cls(
            actor=actor,
            reward_critic=reward_critic,
            cost_critic=cost_critic,
            learning_rate=training.learning_rate,
            clip_ratio=training.clip_ratio,
            entropy_coefficient=training.entropy_coefficient,
        )

    def compute_losses(self, batch: PPOBatch) -> PPOLossTerms:
        """Evaluate all three networks without mutating parameters."""

        self._validate_network_batch_contract(batch)
        action_evaluation = self.actor.evaluate_actions(
            batch.actor_observations,
            batch.action_masks,
            batch.actions,
        )
        reward_predictions = self.reward_critic(batch.critic_observations)
        cost_predictions = self.cost_critic(batch.critic_observations)
        return ppo_loss_terms(
            batch=batch,
            action_evaluation=action_evaluation,
            reward_predictions=reward_predictions,
            cost_predictions=cost_predictions,
            clip_ratio=self.clip_ratio,
            entropy_coefficient=self.entropy_coefficient,
        )

    def update(self, batch: PPOBatch) -> PPOUpdateMetrics:
        """Apply one actor, reward-critic, and cost-critic optimizer step."""

        self.actor.train()
        self.reward_critic.train()
        self.cost_critic.train()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.reward_critic_optimizer.zero_grad(set_to_none=True)
        self.cost_critic_optimizer.zero_grad(set_to_none=True)

        losses = self.compute_losses(batch)
        losses.actor_loss.backward()  # type: ignore[no-untyped-call]
        losses.reward_value_loss.backward()  # type: ignore[no-untyped-call]
        losses.cost_value_loss.backward()  # type: ignore[no-untyped-call]
        _require_finite_gradients("actor", self.actor)
        _require_finite_gradients("reward critic", self.reward_critic)
        _require_finite_gradients("cost critic", self.cost_critic)

        self.actor_optimizer.step()
        self.reward_critic_optimizer.step()
        self.cost_critic_optimizer.step()
        _require_finite_parameters("actor", self.actor)
        _require_finite_parameters("reward critic", self.reward_critic)
        _require_finite_parameters("cost critic", self.cost_critic)
        return PPOUpdateMetrics(
            actor_loss=float(losses.actor_loss.detach().item()),
            policy_loss=float(losses.policy_loss.detach().item()),
            reward_value_loss=float(losses.reward_value_loss.detach().item()),
            cost_value_loss=float(losses.cost_value_loss.detach().item()),
            entropy=float(losses.entropy.detach().item()),
            approximate_kl=float(losses.approximate_kl.detach().item()),
            clip_fraction=float(losses.clip_fraction.detach().item()),
            ratio_mean=float(losses.ratio_mean.detach().item()),
        )

    def _validate_network_batch_contract(self, batch: PPOBatch) -> None:
        if not isinstance(batch, PPOBatch):
            raise PPOError("PPO updater requires a PPOBatch")
        if batch.actor_observations.shape[1] != self.actor.observation_width:
            raise PPOError("PPO actor observation width does not match its network")
        if batch.critic_observations.shape[1] != self.reward_critic.observation_width:
            raise PPOError("PPO critic observation width does not match its network")
        if self.reward_critic.observation_width != self.cost_critic.observation_width:
            raise PPOError("reward and cost critics must use the same observation width")

        reference = next(self.actor.parameters())
        if batch.actor_observations.device != reference.device:
            raise PPOError("PPO minibatch and networks must use the same device")
        if batch.actor_observations.dtype != reference.dtype:
            raise PPOError("PPO minibatch and networks must use the same dtype")
        for name, network in (
            ("reward critic", self.reward_critic),
            ("cost critic", self.cost_critic),
        ):
            parameter = next(network.parameters())
            if parameter.device != reference.device or parameter.dtype != reference.dtype:
                raise PPOError(f"{name} must share the actor device and dtype")


def _validate_loss_vector(
    name: str,
    values: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> None:
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise PPOError(f"{name} must be a floating-point torch.Tensor")
    if values.ndim != 1 or values.shape[0] == 0:
        raise PPOError(f"{name} must be a nonempty rank-one tensor")
    if reference is not None:
        if values.shape != reference.shape:
            raise PPOError(f"{name} must match the reference shape")
        if values.dtype != reference.dtype:
            raise PPOError(f"{name} must match the reference dtype")
        if values.device != reference.device:
            raise PPOError(f"{name} must match the reference device")
    if not bool(torch.isfinite(values).all().item()):
        raise PPOError(f"{name} contains a non-finite value")


def _validate_width(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PPOError(f"{name} must be a positive integer")


def _validate_positive(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise PPOError(f"{name} must be finite and positive", context={"actual": value})
    return float(value)


def _validate_nonnegative(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise PPOError(f"{name} must be finite and nonnegative", context={"actual": value})
    return float(value)


def _validate_open_unit(name: str, value: object) -> float:
    validated = _validate_positive(name, value)
    if validated >= 1.0:
        raise PPOError(f"{name} must lie in (0, 1)", context={"actual": value})
    return validated


def _require_finite_gradients(name: str, network: nn.Module) -> None:
    found = False
    for parameter in network.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise PPOError(f"{name} has a missing gradient")
        found = True
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise PPOError(f"{name} has a non-finite gradient")
    if not found:
        raise PPOError(f"{name} has no trainable parameters")


def _require_finite_parameters(name: str, network: nn.Module) -> None:
    for parameter in network.parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise PPOError(f"{name} has a non-finite parameter after update")


__all__ = [
    "ClippedPolicySurrogate",
    "PPOBatch",
    "PPOError",
    "PPOLossTerms",
    "PPOUpdateMetrics",
    "PPOUpdater",
    "ScalarCritic",
    "clipped_policy_surrogate",
    "mean_squared_value_loss",
    "ppo_loss_terms",
]
