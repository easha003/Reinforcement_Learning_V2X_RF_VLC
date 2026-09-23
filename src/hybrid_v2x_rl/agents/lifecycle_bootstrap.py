"""Stable-identity value bootstraps for pair lifecycle boundaries.

This module is the adapter between the NumPy frame-environment contract and
the Torch GAE kernel.  It never infers lifecycle from one combined ``done``
flag and never assumes that an array position identifies the same pair in the
next population.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.environment_api import FrameStepOutput


class LifecycleBootstrapError(HybridV2XError):
    """A critic-value frame cannot be reconciled with pair lifecycle state."""


@dataclass(frozen=True, slots=True)
class PairedCriticValues:
    """Reward and cost critic predictions keyed by stable pair identity."""

    pair_ids: tuple[str, ...]
    reward: torch.Tensor
    cost: torch.Tensor

    def __post_init__(self) -> None:
        _validate_pair_ids(self.pair_ids, name="critic value pair_ids")
        for name, values in (("reward", self.reward), ("cost", self.cost)):
            if not isinstance(values, torch.Tensor):
                raise LifecycleBootstrapError(f"{name} critic values must be a torch.Tensor")
            if values.ndim != 1 or values.shape[0] != len(self.pair_ids):
                raise LifecycleBootstrapError(
                    f"{name} critic values must have one entry per pair ID",
                    context={
                        "actual": tuple(values.shape),
                        "expected": (len(self.pair_ids),),
                    },
                )
            if not values.is_floating_point():
                raise LifecycleBootstrapError(f"{name} critic values must be floating point")
            if not bool(torch.isfinite(values).all().item()):
                raise LifecycleBootstrapError(f"{name} critic values must be finite")
        if self.reward.dtype != self.cost.dtype:
            raise LifecycleBootstrapError("reward and cost critic values must use the same dtype")
        if self.reward.device != self.cost.device:
            raise LifecycleBootstrapError("reward and cost critic values must use the same device")


@dataclass(frozen=True, slots=True)
class LifecycleBootstrapBatch:
    """Current values, selected next values, and lifecycle masks for one frame."""

    transition_pair_ids: tuple[str, ...]
    reward_values: torch.Tensor
    cost_values: torch.Tensor
    reward_next_values: torch.Tensor
    cost_next_values: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_valid: torch.Tensor
    learn_mask: torch.Tensor
    value_bootstrap_mask: torch.Tensor
    gae_continuation_mask: torch.Tensor

    def __post_init__(self) -> None:
        _validate_pair_ids(self.transition_pair_ids, name="transition_pair_ids")
        population = len(self.transition_pair_ids)
        float_tensors = (
            ("reward_values", self.reward_values),
            ("cost_values", self.cost_values),
            ("reward_next_values", self.reward_next_values),
            ("cost_next_values", self.cost_next_values),
        )
        reference_dtype: torch.dtype | None = None
        reference_device: torch.device | None = None
        for name, values in float_tensors:
            if not isinstance(values, torch.Tensor):
                raise LifecycleBootstrapError(f"{name} must be a torch.Tensor")
            if values.shape != (population,):
                raise LifecycleBootstrapError(
                    f"{name} must align with transition pair IDs",
                    context={"actual": tuple(values.shape), "expected": (population,)},
                )
            if not values.is_floating_point():
                raise LifecycleBootstrapError(f"{name} must be floating point")
            if values.requires_grad:
                raise LifecycleBootstrapError(f"{name} must be detached")
            if not bool(torch.isfinite(values).all().item()):
                raise LifecycleBootstrapError(f"{name} must be finite")
            if reference_dtype is None:
                reference_dtype = values.dtype
                reference_device = values.device
            elif values.dtype != reference_dtype or values.device != reference_device:
                raise LifecycleBootstrapError(
                    "all lifecycle value tensors must share dtype and device"
                )

        bool_tensors = (
            ("terminated", self.terminated),
            ("truncated", self.truncated),
            ("bootstrap_valid", self.bootstrap_valid),
            ("learn_mask", self.learn_mask),
            ("value_bootstrap_mask", self.value_bootstrap_mask),
            ("gae_continuation_mask", self.gae_continuation_mask),
        )
        for name, mask in bool_tensors:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise LifecycleBootstrapError(f"{name} must be a torch.bool tensor")
            if mask.shape != (population,):
                raise LifecycleBootstrapError(
                    f"{name} must align with transition pair IDs",
                    context={"actual": tuple(mask.shape), "expected": (population,)},
                )
            if reference_device is not None and mask.device != reference_device:
                raise LifecycleBootstrapError("lifecycle masks and values must use the same device")

        if bool((self.terminated & self.truncated).any().item()):
            raise LifecycleBootstrapError("a transition cannot terminate and truncate")
        if bool((self.bootstrap_valid & ~self.truncated).any().item()):
            raise LifecycleBootstrapError("only a truncation may use a final bootstrap")
        expected_bootstrap = (~self.terminated & ~self.truncated) | self.bootstrap_valid
        expected_continuation = ~(self.terminated | self.truncated)
        if not torch.equal(self.value_bootstrap_mask, expected_bootstrap):
            raise LifecycleBootstrapError("value bootstrap mask does not match lifecycle semantics")
        if not torch.equal(self.gae_continuation_mask, expected_continuation):
            raise LifecycleBootstrapError(
                "GAE continuation mask does not stop at every lifecycle boundary"
            )

        zero_bootstrap = ~self.value_bootstrap_mask
        if bool(self.reward_next_values[zero_bootstrap].count_nonzero().item()) or bool(
            self.cost_next_values[zero_bootstrap].count_nonzero().item()
        ):
            raise LifecycleBootstrapError(
                "terminal and non-bootstrap truncation values must be exact zero"
            )


def assemble_lifecycle_bootstrap(
    *,
    step_output: FrameStepOutput,
    current_values: PairedCriticValues,
    next_population_values: PairedCriticValues,
    final_observation_values: PairedCriticValues | None = None,
) -> LifecycleBootstrapBatch:
    """Select each transition's next critic value from the correct source.

    Normal continuations look up the same stable pair ID in the ordinary next
    population.  Bootstrap-valid truncations instead use values evaluated from
    ``info['final_observation']`` before reset.  Every other final transition
    receives zero, even if a newly reset population happens to occupy the same
    array position.
    """

    if not isinstance(step_output, FrameStepOutput):
        raise LifecycleBootstrapError("bootstrap assembly requires a FrameStepOutput")
    for name, values in (
        ("current_values", current_values),
        ("next_population_values", next_population_values),
    ):
        if not isinstance(values, PairedCriticValues):
            raise LifecycleBootstrapError(f"{name} must be PairedCriticValues")

    if current_values.pair_ids != step_output.transition_pair_ids:
        raise LifecycleBootstrapError(
            "current critic values do not align with transition pair IDs",
            context={
                "values": current_values.pair_ids,
                "transitions": step_output.transition_pair_ids,
            },
        )
    if next_population_values.pair_ids != step_output.next_observation.pair_ids:
        raise LifecycleBootstrapError(
            "next critic values do not align with the next observation",
            context={
                "values": next_population_values.pair_ids,
                "observation": step_output.next_observation.pair_ids,
            },
        )
    _require_compatible_values(
        current_values,
        next_population_values,
        name="next population",
    )

    expected_final_ids = tuple(
        pair_id
        for pair_id, valid in zip(
            step_output.transition_pair_ids,
            step_output.bootstrap_valid,
            strict=True,
        )
        if bool(valid)
    )
    supplied_final_ids = (
        () if final_observation_values is None else final_observation_values.pair_ids
    )
    if supplied_final_ids != expected_final_ids:
        raise LifecycleBootstrapError(
            "final critic values must cover bootstrap-valid truncations exactly",
            context={
                "actual": supplied_final_ids,
                "expected": expected_final_ids,
            },
        )
    if final_observation_values is not None:
        _require_compatible_values(
            current_values,
            final_observation_values,
            name="final observation",
        )

    info_final = step_output.info["final_observation"]
    if not isinstance(info_final, Mapping):
        raise LifecycleBootstrapError("step final observations must be a pair-ID mapping")
    if tuple(sorted(info_final)) != expected_final_ids:
        raise LifecycleBootstrapError(
            "step final observations and final critic values do not share exact IDs"
        )

    transition_ids = step_output.transition_pair_ids
    final_ids = tuple(
        pair_id
        for pair_id, ended in zip(
            transition_ids,
            step_output.terminated | step_output.truncated,
            strict=True,
        )
        if bool(ended)
    )
    next_id_set = set(next_population_values.pair_ids)
    continuing_ids = tuple(
        pair_id
        for pair_id, terminated, truncated in zip(
            transition_ids,
            step_output.terminated,
            step_output.truncated,
            strict=True,
        )
        if not bool(terminated) and not bool(truncated)
    )
    missing_continuing = tuple(pair_id for pair_id in continuing_ids if pair_id not in next_id_set)
    repeated_final = tuple(pair_id for pair_id in final_ids if pair_id in next_id_set)
    if missing_continuing or repeated_final:
        raise LifecycleBootstrapError(
            "next population violates stable pair lifecycle",
            context={
                "missing_continuing_pair_ids": missing_continuing,
                "repeated_final_pair_ids": repeated_final,
            },
        )

    next_indices = {pair_id: index for index, pair_id in enumerate(next_population_values.pair_ids)}
    final_indices = {pair_id: index for index, pair_id in enumerate(expected_final_ids)}
    with torch.no_grad():
        reward_next = torch.zeros_like(current_values.reward)
        cost_next = torch.zeros_like(current_values.cost)
        for row, pair_id in enumerate(transition_ids):
            if pair_id in next_indices:
                source_row = next_indices[pair_id]
                reward_next[row] = next_population_values.reward[source_row]
                cost_next[row] = next_population_values.cost[source_row]
            elif pair_id in final_indices:
                assert final_observation_values is not None
                source_row = final_indices[pair_id]
                reward_next[row] = final_observation_values.reward[source_row]
                cost_next[row] = final_observation_values.cost[source_row]

        device = current_values.reward.device
        terminated = _bool_tensor(step_output.terminated, device=device)
        truncated = _bool_tensor(step_output.truncated, device=device)
        bootstrap_valid = _bool_tensor(step_output.bootstrap_valid, device=device)
        learn_mask = _bool_tensor(step_output.learn_mask, device=device)
        value_bootstrap_mask = (~terminated & ~truncated) | bootstrap_valid
        gae_continuation_mask = ~(terminated | truncated)

        return LifecycleBootstrapBatch(
            transition_pair_ids=transition_ids,
            reward_values=current_values.reward.detach().clone(),
            cost_values=current_values.cost.detach().clone(),
            reward_next_values=reward_next.detach(),
            cost_next_values=cost_next.detach(),
            terminated=terminated,
            truncated=truncated,
            bootstrap_valid=bootstrap_valid,
            learn_mask=learn_mask,
            value_bootstrap_mask=value_bootstrap_mask,
            gae_continuation_mask=gae_continuation_mask,
        )


def _validate_pair_ids(pair_ids: tuple[str, ...], *, name: str) -> None:
    if not isinstance(pair_ids, tuple):
        raise LifecycleBootstrapError(f"{name} must be an immutable tuple")
    if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in pair_ids):
        raise LifecycleBootstrapError(f"{name} entries must be non-empty strings")
    if pair_ids != tuple(sorted(pair_ids)):
        raise LifecycleBootstrapError(f"{name} must use canonical stable-ID order")
    if len(pair_ids) != len(set(pair_ids)):
        raise LifecycleBootstrapError(f"{name} cannot contain duplicates")


def _require_compatible_values(
    reference: PairedCriticValues,
    candidate: PairedCriticValues,
    *,
    name: str,
) -> None:
    if candidate.reward.dtype != reference.reward.dtype:
        raise LifecycleBootstrapError(f"{name} critic values must use the current dtype")
    if candidate.reward.device != reference.reward.device:
        raise LifecycleBootstrapError(f"{name} critic values must use the current device")


def _bool_tensor(values: NDArray[np.bool_], *, device: torch.device) -> torch.Tensor:
    return torch.tensor(values.tolist(), dtype=torch.bool, device=device)


__all__ = [
    "LifecycleBootstrapBatch",
    "LifecycleBootstrapError",
    "PairedCriticValues",
    "assemble_lifecycle_bootstrap",
]
