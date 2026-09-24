"""Reliability-cost selection at the rollout-to-training boundary.

The environment emits both a sampled binary miss and the selected action's
conditional miss probability.  Phase 7 deliberately optimizes with the latter
while retaining the former as the only final-feasibility outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import torch

from hybrid_v2x_rl.config.models import TrainingConfig
from hybrid_v2x_rl.core.errors import HybridV2XError

ReliabilityCostSignal: TypeAlias = Literal["conditional_miss_probability"]
CONDITIONAL_MISS_PROBABILITY: Final[ReliabilityCostSignal] = "conditional_miss_probability"


class ReliabilityCostSignalError(HybridV2XError):
    """A reliability signal selection or rollout tensor is invalid."""


@dataclass(frozen=True, slots=True)
class ReliabilityCostBatch:
    """Detached training costs plus retained sampled evaluation outcomes.

    ``training_costs`` is the sole input to cost GAE, the cost critic, and
    training-time dual estimates.  ``sampled_miss_costs`` remains available
    for realized-miss reporting and must never be replaced by the smoother
    training target during final feasibility analysis.
    """

    signal_name: ReliabilityCostSignal
    training_costs: torch.Tensor
    sampled_miss_costs: torch.Tensor
    conditional_miss_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        if self.signal_name != CONDITIONAL_MISS_PROBABILITY:
            raise ReliabilityCostSignalError(
                "Phase 7 training requires conditional_miss_probability"
            )
        _validate_probability_tensor(
            "conditional_miss_probabilities",
            self.conditional_miss_probabilities,
        )
        _validate_probability_tensor(
            "sampled_miss_costs",
            self.sampled_miss_costs,
            reference=self.conditional_miss_probabilities,
            binary=True,
        )
        _validate_probability_tensor(
            "training_costs",
            self.training_costs,
            reference=self.conditional_miss_probabilities,
        )
        if not torch.equal(self.training_costs, self.conditional_miss_probabilities):
            raise ReliabilityCostSignalError(
                "training costs must equal conditional miss probabilities"
            )


def select_reliability_costs(
    *,
    sampled_miss_costs: torch.Tensor,
    conditional_miss_probabilities: torch.Tensor,
    cost_signal: object,
) -> ReliabilityCostBatch:
    """Select the configured dense training signal and retain sampled misses.

    Copies prevent later mutation of environment/rollout inputs from changing
    an already selected training batch.  No sampled or staged mode is accepted:
    such a change would alter the declared estimator and needs a new versioned
    experimental configuration rather than an implicit runtime branch.
    """

    if cost_signal != CONDITIONAL_MISS_PROBABILITY:
        raise ReliabilityCostSignalError(
            "unsupported reliability cost signal",
            context={
                "actual": cost_signal,
                "required": CONDITIONAL_MISS_PROBABILITY,
            },
        )
    _validate_probability_tensor(
        "conditional_miss_probabilities",
        conditional_miss_probabilities,
    )
    _validate_probability_tensor(
        "sampled_miss_costs",
        sampled_miss_costs,
        reference=conditional_miss_probabilities,
        binary=True,
    )
    return ReliabilityCostBatch(
        signal_name=cost_signal,
        training_costs=conditional_miss_probabilities.detach().clone(),
        sampled_miss_costs=sampled_miss_costs.detach().clone(),
        conditional_miss_probabilities=(conditional_miss_probabilities.detach().clone()),
    )


def reliability_costs_from_config(
    *,
    sampled_miss_costs: torch.Tensor,
    conditional_miss_probabilities: torch.Tensor,
    training: TrainingConfig,
) -> ReliabilityCostBatch:
    """Select reliability costs from a validated training configuration."""

    if not isinstance(training, TrainingConfig):
        raise ReliabilityCostSignalError(
            "reliability cost selection requires a validated TrainingConfig"
        )
    return select_reliability_costs(
        sampled_miss_costs=sampled_miss_costs,
        conditional_miss_probabilities=conditional_miss_probabilities,
        cost_signal=training.cost_signal,
    )


def _validate_probability_tensor(
    name: str,
    values: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
    binary: bool = False,
) -> None:
    if not isinstance(values, torch.Tensor):
        raise ReliabilityCostSignalError(f"{name} must be a torch.Tensor")
    if values.ndim == 0:
        raise ReliabilityCostSignalError(f"{name} must include an explicit rollout dimension")
    if not values.is_floating_point():
        raise ReliabilityCostSignalError(f"{name} must be floating point")
    if values.requires_grad:
        raise ReliabilityCostSignalError(f"{name} must be detached rollout data")
    if not bool(torch.isfinite(values).all().item()):
        raise ReliabilityCostSignalError(f"{name} contains a non-finite value")
    if bool(((values < 0.0) | (values > 1.0)).any().item()):
        raise ReliabilityCostSignalError(f"{name} must lie in [0, 1]")
    if binary and not bool(((values == 0.0) | (values == 1.0)).all().item()):
        raise ReliabilityCostSignalError(f"{name} must be binary")
    if reference is not None:
        if values.shape != reference.shape:
            raise ReliabilityCostSignalError(
                f"{name} must match conditional miss probability shape",
                context={
                    "actual": tuple(values.shape),
                    "expected": tuple(reference.shape),
                },
            )
        if values.dtype != reference.dtype:
            raise ReliabilityCostSignalError(
                f"{name} must match conditional miss probability dtype"
            )
        if values.device != reference.device:
            raise ReliabilityCostSignalError(
                f"{name} must use the conditional miss probability device"
            )


__all__ = [
    "CONDITIONAL_MISS_PROBABILITY",
    "ReliabilityCostBatch",
    "ReliabilityCostSignal",
    "ReliabilityCostSignalError",
    "reliability_costs_from_config",
    "select_reliability_costs",
]
