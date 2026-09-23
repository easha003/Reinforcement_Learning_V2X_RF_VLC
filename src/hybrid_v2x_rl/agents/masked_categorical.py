"""Masked categorical sampling for the shared nine-action policy.

The mask in this module is a feasibility mask, not a channel oracle.  It may
encode only the hardware/profile availability already represented by
``mean_field.action_masks.ActionMask``.  Transient link quality remains an
observation for the actor to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from hybrid_v2x_rl.config.models import NetworkArchitectureConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.action_masks import ActionMask

ACTION_COUNT = len(PolicyAction)


class MaskedCategoricalError(HybridV2XError):
    """Policy logits, masks, or selected actions violate their contract."""


@dataclass(frozen=True, slots=True)
class CategoricalActionBatch:
    """Actions and policy statistics for one population decision batch."""

    actions: torch.Tensor
    log_probabilities: torch.Tensor
    entropy: torch.Tensor

    def __post_init__(self) -> None:
        if self.actions.ndim != 1:
            raise MaskedCategoricalError("actions must be a rank-one tensor")
        if self.actions.dtype != torch.long:
            raise MaskedCategoricalError("actions must use torch.long indices")
        expected_shape = self.actions.shape
        for name, values in (
            ("log_probabilities", self.log_probabilities),
            ("entropy", self.entropy),
        ):
            if values.shape != expected_shape:
                raise MaskedCategoricalError(
                    f"{name} must have one value per action",
                    context={"actual": tuple(values.shape), "expected": tuple(expected_shape)},
                )
            if not values.is_floating_point():
                raise MaskedCategoricalError(f"{name} must be floating point")
            if not bool(torch.isfinite(values).all().item()):
                raise MaskedCategoricalError(f"{name} contains a non-finite value")


class MaskedCategorical:
    """A categorical distribution whose impossible actions have zero mass.

    A one-dimensional mask is broadcast across the batch.  A two-dimensional
    mask can instead express one feasibility profile per active pair.  Every
    row must leave at least one action available.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        action_mask: ActionMask | torch.Tensor,
    ) -> None:
        _validate_logits(logits)
        self._logits = logits
        self._mask = _coerce_mask(action_mask, logits=logits)
        self._masked_logits = logits.masked_fill(~self._mask, -torch.inf)
        self._distribution = (
            None
            if logits.shape[0] == 0
            else torch.distributions.Categorical(logits=self._masked_logits)
        )

    @property
    def mask(self) -> torch.Tensor:
        """The batch-shaped boolean feasibility mask."""

        return self._mask

    @property
    def probabilities(self) -> torch.Tensor:
        """Normalized action probabilities, exactly zero on masked entries."""

        if self._distribution is None:
            return self._logits.new_empty((0, ACTION_COUNT))
        return self._distribution.probs

    def select(
        self,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> CategoricalActionBatch:
        """Select actions stochastically or by deterministic masked argmax.

        Deterministic ties use the lowest persistent action index, matching
        ``torch.argmax``.  Stochastic sampling accepts an explicit generator
        so policy randomness is isolated from PyTorch's global RNG.
        """

        if type(deterministic) is not bool:
            raise MaskedCategoricalError("deterministic must be boolean")
        if self._distribution is None:
            return _empty_action_batch(self._logits)

        if deterministic:
            actions = torch.argmax(self._masked_logits, dim=-1)
        else:
            actions = torch.multinomial(
                self._distribution.probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
        return self.evaluate_actions(actions)

    def evaluate_actions(self, actions: torch.Tensor) -> CategoricalActionBatch:
        """Evaluate already selected actions under the same masked policy."""

        _validate_actions(actions, logits=self._logits, mask=self._mask)
        if self._distribution is None:
            return _empty_action_batch(self._logits)
        log_probabilities = self._distribution.log_prob(  # type: ignore[no-untyped-call]
            actions
        )
        entropy = self._distribution.entropy()  # type: ignore[no-untyped-call]
        return CategoricalActionBatch(
            actions=actions,
            log_probabilities=log_probabilities,
            entropy=entropy,
        )


class SharedCategoricalActor(nn.Module):
    """Feed-forward shared actor producing logits in canonical action order."""

    def __init__(
        self,
        *,
        observation_width: int,
        hidden_units: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        if (
            not isinstance(observation_width, int)
            or isinstance(observation_width, bool)
            or observation_width <= 0
        ):
            raise MaskedCategoricalError("observation_width must be a positive integer")
        if not hidden_units or any(
            not isinstance(width, int) or isinstance(width, bool) or width <= 0
            for width in hidden_units
        ):
            raise MaskedCategoricalError("hidden_units must contain positive integers")

        self.observation_width = observation_width
        self.hidden_units = hidden_units
        layers: list[nn.Module] = []
        input_width = observation_width
        for output_width in hidden_units:
            layers.extend((nn.Linear(input_width, output_width), nn.Tanh()))
            input_width = output_width
        layers.append(nn.Linear(input_width, ACTION_COUNT))
        self.network = nn.Sequential(*layers)

    @classmethod
    def from_config(
        cls,
        *,
        observation_width: int,
        architecture: NetworkArchitectureConfig,
    ) -> SharedCategoricalActor:
        """Construct the initial actor architecture from validated config."""

        if architecture.activation != "tanh":
            raise MaskedCategoricalError("the Phase 7 actor requires tanh activation")
        if architecture.recurrent:
            raise MaskedCategoricalError("the initial Phase 7 actor must be feed-forward")
        return cls(
            observation_width=observation_width,
            hidden_units=architecture.actor_hidden_units,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one vector of nine unnormalized logits per active pair."""

        if not isinstance(observations, torch.Tensor):
            raise MaskedCategoricalError("observations must be a torch.Tensor")
        if observations.ndim != 2 or observations.shape[1] != self.observation_width:
            raise MaskedCategoricalError(
                "observations have the wrong shape",
                context={
                    "actual": tuple(observations.shape),
                    "expected": ("batch", self.observation_width),
                },
            )
        if not observations.is_floating_point():
            raise MaskedCategoricalError("observations must be floating point")
        if not bool(torch.isfinite(observations).all().item()):
            raise MaskedCategoricalError("observations contain a non-finite value")
        return cast(torch.Tensor, self.network(observations))

    def select(
        self,
        observations: torch.Tensor,
        action_mask: ActionMask | torch.Tensor,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> CategoricalActionBatch:
        """Produce logits and select a valid action for every observation."""

        return MaskedCategorical(self(observations), action_mask).select(
            deterministic=deterministic,
            generator=generator,
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        action_mask: ActionMask | torch.Tensor,
        actions: torch.Tensor,
    ) -> CategoricalActionBatch:
        """Re-evaluate rollout actions for a future PPO minibatch update."""

        return MaskedCategorical(self(observations), action_mask).evaluate_actions(actions)


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise MaskedCategoricalError("logits must be a torch.Tensor")
    if logits.ndim != 2 or logits.shape[1] != ACTION_COUNT:
        raise MaskedCategoricalError(
            "logits must have one column per policy action",
            context={"actual": tuple(logits.shape), "expected": ("batch", ACTION_COUNT)},
        )
    if not logits.is_floating_point():
        raise MaskedCategoricalError("logits must be floating point")
    if not bool(torch.isfinite(logits).all().item()):
        raise MaskedCategoricalError("logits contain a non-finite value")


def _coerce_mask(
    action_mask: ActionMask | torch.Tensor,
    *,
    logits: torch.Tensor,
) -> torch.Tensor:
    if isinstance(action_mask, ActionMask):
        mask = torch.tensor(action_mask.values, dtype=torch.bool, device=logits.device)
    elif isinstance(action_mask, torch.Tensor):
        mask = action_mask
        if mask.dtype != torch.bool:
            raise MaskedCategoricalError("action mask must use torch.bool")
        if mask.device != logits.device:
            raise MaskedCategoricalError("action mask and logits must use the same device")
    else:
        raise MaskedCategoricalError("action mask must be ActionMask or torch.Tensor")

    if mask.ndim == 1:
        if mask.shape[0] != ACTION_COUNT:
            raise MaskedCategoricalError(
                "action mask has the wrong width",
                context={"actual": mask.shape[0], "expected": ACTION_COUNT},
            )
        mask = mask.unsqueeze(0).expand(logits.shape[0], -1)
    elif mask.ndim == 2:
        if mask.shape != logits.shape:
            raise MaskedCategoricalError(
                "batched action mask must match logits",
                context={"actual": tuple(mask.shape), "expected": tuple(logits.shape)},
            )
    else:
        raise MaskedCategoricalError("action mask must be rank one or rank two")

    invalid_rows = torch.nonzero(~mask.any(dim=-1), as_tuple=False).flatten()
    if invalid_rows.numel() > 0:
        raise MaskedCategoricalError(
            "every population row must allow at least one action",
            context={"rows": invalid_rows.detach().cpu().tolist()},
        )
    return mask


def _validate_actions(
    actions: torch.Tensor,
    *,
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if not isinstance(actions, torch.Tensor):
        raise MaskedCategoricalError("actions must be a torch.Tensor")
    if actions.ndim != 1 or actions.shape[0] != logits.shape[0]:
        raise MaskedCategoricalError(
            "actions must have one index per policy row",
            context={"actual": tuple(actions.shape), "expected": (logits.shape[0],)},
        )
    if actions.dtype != torch.long:
        raise MaskedCategoricalError("actions must use torch.long indices")
    if actions.device != logits.device:
        raise MaskedCategoricalError("actions and logits must use the same device")
    if actions.numel() == 0:
        return
    out_of_range = (actions < 0) | (actions >= ACTION_COUNT)
    if bool(out_of_range.any().item()):
        rows = torch.nonzero(out_of_range, as_tuple=False).flatten()
        raise MaskedCategoricalError(
            "action index lies outside the policy action space",
            context={"rows": rows.detach().cpu().tolist()},
        )
    allowed = mask.gather(1, actions.unsqueeze(1)).squeeze(1)
    if not bool(allowed.all().item()):
        rows = torch.nonzero(~allowed, as_tuple=False).flatten()
        raise MaskedCategoricalError(
            "selected action is masked",
            context={"rows": rows.detach().cpu().tolist()},
        )


def _empty_action_batch(logits: torch.Tensor) -> CategoricalActionBatch:
    return CategoricalActionBatch(
        actions=torch.empty(0, dtype=torch.long, device=logits.device),
        log_probabilities=logits.new_empty((0,)),
        entropy=logits.new_empty((0,)),
    )


__all__ = [
    "ACTION_COUNT",
    "CategoricalActionBatch",
    "MaskedCategorical",
    "MaskedCategoricalError",
    "SharedCategoricalActor",
]
