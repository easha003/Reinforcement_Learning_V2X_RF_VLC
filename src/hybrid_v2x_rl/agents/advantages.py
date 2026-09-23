"""Reward and reliability-cost generalized advantage estimation.

The estimator consumes explicit masks instead of an ambiguous ``done`` flag.
``value_bootstrap_mask`` controls the one-step temporal-difference target,
whereas ``gae_continuation_mask`` controls whether residuals recurse across
the next time index.  Lifecycle interpretation and final-observation value
construction belong to the rollout boundary, not to this numerical kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

from hybrid_v2x_rl.core.errors import HybridV2XError


class AdvantageEstimationError(HybridV2XError):
    """A rollout tensor or estimator parameter violates the GAE contract."""


@dataclass(frozen=True, slots=True)
class AdvantageEstimate:
    """Detached advantages and critic targets for one scalar signal."""

    advantages: torch.Tensor
    value_targets: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.advantages, torch.Tensor) or not isinstance(
            self.value_targets, torch.Tensor
        ):
            raise AdvantageEstimationError("advantage outputs must be torch tensors")
        if self.advantages.shape != self.value_targets.shape:
            raise AdvantageEstimationError(
                "advantages and value targets must have matching shapes",
                context={
                    "advantages": tuple(self.advantages.shape),
                    "value_targets": tuple(self.value_targets.shape),
                },
            )
        if self.advantages.dtype != self.value_targets.dtype:
            raise AdvantageEstimationError("advantages and value targets must have matching dtypes")
        if self.advantages.device != self.value_targets.device:
            raise AdvantageEstimationError("advantages and value targets must use the same device")
        if not self.advantages.is_floating_point():
            raise AdvantageEstimationError("advantage outputs must be floating point")
        if self.advantages.requires_grad or self.value_targets.requires_grad:
            raise AdvantageEstimationError("advantage outputs must be detached")
        for name, values in (
            ("advantages", self.advantages),
            ("value_targets", self.value_targets),
        ):
            if not bool(torch.isfinite(values).all().item()):
                raise AdvantageEstimationError(f"{name} contains a non-finite value")


@dataclass(frozen=True, slots=True)
class RewardCostAdvantageBatch:
    """Independent reward and reliability-cost estimates for one rollout."""

    reward: AdvantageEstimate
    cost: AdvantageEstimate

    def __post_init__(self) -> None:
        if self.reward.advantages.shape != self.cost.advantages.shape:
            raise AdvantageEstimationError(
                "reward and cost estimates must have matching rollout shapes",
                context={
                    "reward": tuple(self.reward.advantages.shape),
                    "cost": tuple(self.cost.advantages.shape),
                },
            )
        if self.reward.advantages.device != self.cost.advantages.device:
            raise AdvantageEstimationError("reward and cost estimates must use the same device")
        if self.reward.advantages.dtype != self.cost.advantages.dtype:
            raise AdvantageEstimationError("reward and cost estimates must use the same dtype")


def generalized_advantage_estimate(
    *,
    signals: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    value_bootstrap_mask: torch.Tensor,
    gae_continuation_mask: torch.Tensor,
    active_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    time_dimension: int,
) -> AdvantageEstimate:
    """Compute finite-horizon GAE along one explicit time dimension.

    For every active row ``t`` the temporal-difference residual is

    ``delta_t = signal_t + gamma * bootstrap_t * V_next_t - V_t``.

    The reverse recursion is

    ``A_t = delta_t + gamma * gae_lambda * continuation_t * A_(t+1)``.

    ``next_values`` is explicit because a valid time-limit truncation may need
    a value evaluated from its final physical observation rather than the
    reset observation at the next rollout index.  Padded rows are zeroed and
    cannot participate in either the bootstrap or the recursion.
    """

    discount = _validate_hyperparameter("gamma", gamma)
    trace_parameter = _validate_hyperparameter("gae_lambda", gae_lambda)
    _validate_float_rollout("signals", signals)
    _validate_time_dimension(time_dimension, rank=signals.ndim)
    for name, tensor in (
        ("values", values),
        ("next_values", next_values),
    ):
        _validate_float_rollout(
            name,
            tensor,
            expected_shape=signals.shape,
            expected_dtype=signals.dtype,
            expected_device=signals.device,
        )
    for name, mask in (
        ("value_bootstrap_mask", value_bootstrap_mask),
        ("gae_continuation_mask", gae_continuation_mask),
        ("active_mask", active_mask),
    ):
        _validate_mask(
            name,
            mask,
            expected_shape=signals.shape,
            expected_device=signals.device,
        )

    _validate_mask_relationships(
        value_bootstrap_mask=value_bootstrap_mask,
        gae_continuation_mask=gae_continuation_mask,
        active_mask=active_mask,
        time_dimension=time_dimension,
    )

    time_dimension = time_dimension % signals.ndim
    with torch.no_grad():
        signal_time = torch.movedim(signals.detach(), time_dimension, 0)
        value_time = torch.movedim(values.detach(), time_dimension, 0)
        next_value_time = torch.movedim(next_values.detach(), time_dimension, 0)
        bootstrap_time = torch.movedim(value_bootstrap_mask, time_dimension, 0)
        continuation_time = torch.movedim(gae_continuation_mask, time_dimension, 0)
        active_time = torch.movedim(active_mask, time_dimension, 0)

        advantages_time = torch.zeros_like(signal_time)
        next_advantage = torch.zeros_like(signal_time[0]) if signal_time.shape[0] else None
        trace_discount = discount * trace_parameter

        for time_index in range(signal_time.shape[0] - 1, -1, -1):
            assert next_advantage is not None
            bootstrap = bootstrap_time[time_index].to(dtype=signals.dtype)
            continuation = continuation_time[time_index].to(dtype=signals.dtype)
            delta = (
                signal_time[time_index]
                + discount * bootstrap * next_value_time[time_index]
                - value_time[time_index]
            )
            estimate = delta + trace_discount * continuation * next_advantage
            next_advantage = torch.where(
                active_time[time_index],
                estimate,
                torch.zeros_like(estimate),
            )
            advantages_time[time_index] = next_advantage

        advantages = torch.movedim(advantages_time, 0, time_dimension)
        value_targets = torch.where(
            active_mask,
            advantages + values.detach(),
            torch.zeros_like(advantages),
        )

    return AdvantageEstimate(
        advantages=advantages,
        value_targets=value_targets,
    )


def reward_cost_generalized_advantage_estimate(
    *,
    rewards: torch.Tensor,
    costs: torch.Tensor,
    reward_values: torch.Tensor,
    reward_next_values: torch.Tensor,
    cost_values: torch.Tensor,
    cost_next_values: torch.Tensor,
    value_bootstrap_mask: torch.Tensor,
    gae_continuation_mask: torch.Tensor,
    active_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    time_dimension: int,
) -> RewardCostAdvantageBatch:
    """Estimate reward and cost streams with independent critic predictions."""

    reward = generalized_advantage_estimate(
        signals=rewards,
        values=reward_values,
        next_values=reward_next_values,
        value_bootstrap_mask=value_bootstrap_mask,
        gae_continuation_mask=gae_continuation_mask,
        active_mask=active_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
        time_dimension=time_dimension,
    )
    cost = generalized_advantage_estimate(
        signals=costs,
        values=cost_values,
        next_values=cost_next_values,
        value_bootstrap_mask=value_bootstrap_mask,
        gae_continuation_mask=gae_continuation_mask,
        active_mask=active_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
        time_dimension=time_dimension,
    )
    return RewardCostAdvantageBatch(reward=reward, cost=cost)


def _validate_hyperparameter(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 < float(value) <= 1.0
    ):
        raise AdvantageEstimationError(
            f"{name} must be finite and lie in (0, 1]",
            context={"actual": value},
        )
    return float(value)


def _validate_float_rollout(
    name: str,
    values: torch.Tensor,
    *,
    expected_shape: torch.Size | None = None,
    expected_dtype: torch.dtype | None = None,
    expected_device: torch.device | None = None,
) -> None:
    if not isinstance(values, torch.Tensor):
        raise AdvantageEstimationError(f"{name} must be a torch.Tensor")
    if values.ndim == 0:
        raise AdvantageEstimationError(f"{name} must include an explicit time dimension")
    if not values.is_floating_point():
        raise AdvantageEstimationError(f"{name} must be floating point")
    if expected_shape is not None and values.shape != expected_shape:
        raise AdvantageEstimationError(
            f"{name} must match the signal shape",
            context={"actual": tuple(values.shape), "expected": tuple(expected_shape)},
        )
    if expected_dtype is not None and values.dtype != expected_dtype:
        raise AdvantageEstimationError(
            f"{name} must match the signal dtype",
            context={"actual": str(values.dtype), "expected": str(expected_dtype)},
        )
    if expected_device is not None and values.device != expected_device:
        raise AdvantageEstimationError(f"{name} must use the signal device")
    if not bool(torch.isfinite(values).all().item()):
        raise AdvantageEstimationError(f"{name} contains a non-finite value")


def _validate_mask(
    name: str,
    mask: torch.Tensor,
    *,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> None:
    if not isinstance(mask, torch.Tensor):
        raise AdvantageEstimationError(f"{name} must be a torch.Tensor")
    if mask.dtype != torch.bool:
        raise AdvantageEstimationError(f"{name} must use torch.bool")
    if mask.shape != expected_shape:
        raise AdvantageEstimationError(
            f"{name} must match the signal shape",
            context={"actual": tuple(mask.shape), "expected": tuple(expected_shape)},
        )
    if mask.device != expected_device:
        raise AdvantageEstimationError(f"{name} must use the signal device")


def _validate_time_dimension(time_dimension: int, *, rank: int) -> None:
    if not isinstance(time_dimension, int) or isinstance(time_dimension, bool):
        raise AdvantageEstimationError("time_dimension must be an integer")
    if not -rank <= time_dimension < rank:
        raise AdvantageEstimationError(
            "time_dimension lies outside the rollout rank",
            context={"actual": time_dimension, "rank": rank},
        )


def _validate_mask_relationships(
    *,
    value_bootstrap_mask: torch.Tensor,
    gae_continuation_mask: torch.Tensor,
    active_mask: torch.Tensor,
    time_dimension: int,
) -> None:
    for name, mask in (
        ("value_bootstrap_mask", value_bootstrap_mask),
        ("gae_continuation_mask", gae_continuation_mask),
    ):
        invalid = mask & ~active_mask
        if bool(invalid.any().item()):
            raise AdvantageEstimationError(f"{name} cannot enable a padded row")

    if bool((gae_continuation_mask & ~value_bootstrap_mask).any().item()):
        raise AdvantageEstimationError("GAE continuation requires a valid one-step value bootstrap")

    normalized_dimension = time_dimension % active_mask.ndim
    active_time = torch.movedim(active_mask, normalized_dimension, 0)
    continuation_time = torch.movedim(
        gae_continuation_mask,
        normalized_dimension,
        0,
    )
    if active_time.shape[0] > 1:
        crosses_padding = continuation_time[:-1] & ~active_time[1:]
        if bool(crosses_padding.any().item()):
            raise AdvantageEstimationError(
                "GAE continuation cannot enter an inactive next-time row"
            )


__all__ = [
    "AdvantageEstimate",
    "AdvantageEstimationError",
    "RewardCostAdvantageBatch",
    "generalized_advantage_estimate",
    "reward_cost_generalized_advantage_estimate",
]
