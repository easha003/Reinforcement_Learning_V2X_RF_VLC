"""Typed boundary for the variable-population mean-field environment.

The environment keeps Gymnasium's reset/step vocabulary, reset seeding
arguments, five-part step result, and separate termination/truncation
semantics.  It intentionally does not inherit from :class:`gymnasium.Env`:
Gymnasium's base protocol requires scalar rewards and scalar episode flags,
whereas one population step returns aligned vectors for every pair active in
the decision frame and the next frame may contain a different population.

This module defines that deviation without implementing Phase 5 dynamics.  In
particular, step outputs align to ``transition_pair_ids`` (the actors that just
acted), while ``next_observation.pair_ids`` identify the possibly different
next population.  Keeping both identities explicit prevents an array position
from being mistaken for a persistent agent identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER
from hybrid_v2x_rl.mean_field.congestion_feedback import ActorObservationSchema
from hybrid_v2x_rl.observation.builder import ObservationBuilder

FRAME_API_VERSION = "1.0.0"
CONTRACT_ACTOR_WIDTH = 37
CONTRACT_ACTION_COUNT = len(POLICY_ACTION_ORDER)

ActorObservationArray: TypeAlias = NDArray[np.float32]
ActionMaskArray: TypeAlias = NDArray[np.bool_]
ActionArray: TypeAlias = NDArray[np.int64]
RewardArray: TypeAlias = NDArray[np.float32]
DoneArray: TypeAlias = NDArray[np.bool_]
FrameInfo: TypeAlias = Mapping[str, object]


class FrameAPIError(HybridV2XError):
    """A frame observation or API result violates contract 1.0.0."""


def _validate_pair_ids(pair_ids: tuple[str, ...], *, name: str) -> None:
    if not isinstance(pair_ids, tuple):
        raise FrameAPIError(f"{name} must be an immutable tuple")
    if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in pair_ids):
        raise FrameAPIError(f"{name} entries must be non-empty strings")
    if pair_ids != tuple(sorted(pair_ids)):
        raise FrameAPIError(f"{name} must use canonical stable-ID order")
    if len(pair_ids) != len(set(pair_ids)):
        raise FrameAPIError(f"{name} cannot contain duplicate stable IDs")


def _freeze_array(
    value: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    frozen = value.copy(order="C")
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True)
class FrameAPISchema:
    """Versioned per-row shapes derived from the resolved configuration."""

    contract_version: str
    actor_width: int
    action_count: int

    def __post_init__(self) -> None:
        if self.contract_version != FRAME_API_VERSION:
            raise FrameAPIError(
                "frame API version does not match the frozen environment contract",
                context={
                    "actual": self.contract_version,
                    "expected": FRAME_API_VERSION,
                },
            )
        if self.actor_width != CONTRACT_ACTOR_WIDTH:
            raise FrameAPIError(
                "actor width does not match contract 1.0.0",
                context={
                    "actual": self.actor_width,
                    "expected": CONTRACT_ACTOR_WIDTH,
                },
            )
        if self.action_count != CONTRACT_ACTION_COUNT:
            raise FrameAPIError(
                "action count does not match contract 1.0.0",
                context={
                    "actual": self.action_count,
                    "expected": CONTRACT_ACTION_COUNT,
                },
            )

    @classmethod
    def from_config(cls, config: ProjectConfig) -> FrameAPISchema:
        """Derive and assert the versioned widths instead of trusting literals."""

        if not isinstance(config, ProjectConfig):
            raise FrameAPIError("frame API schema requires a resolved ProjectConfig")
        local_schema = ObservationBuilder.from_config(config.observation).schema
        actor_schema = ActorObservationSchema(local=local_schema)
        return cls(
            contract_version=config.environment.contract_version,
            actor_width=actor_schema.width,
            action_count=len(config.environment.actions),
        )

    def validate_observation(self, observation: FrameObservation) -> None:
        """Assert that a self-consistent frame also matches this contract."""

        if not isinstance(observation, FrameObservation):
            raise FrameAPIError("frame API requires a FrameObservation")
        if observation.actor_observations.shape[1] != self.actor_width:
            raise FrameAPIError(
                "actor observation width does not match the API schema",
                context={
                    "actual": observation.actor_observations.shape[1],
                    "expected": self.actor_width,
                },
            )
        if observation.action_masks.shape[1] != self.action_count:
            raise FrameAPIError(
                "action-mask width does not match the API schema",
                context={
                    "actual": observation.action_masks.shape[1],
                    "expected": self.action_count,
                },
            )


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """One simultaneous decentralized observation for all active pairs."""

    trace_id: str
    frame_index: int
    time_s: float
    pair_ids: tuple[str, ...]
    actor_observations: ActorObservationArray
    action_masks: ActionMaskArray

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise FrameAPIError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise FrameAPIError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise FrameAPIError("time_s must be finite and non-negative")
        _validate_pair_ids(self.pair_ids, name="pair_ids")

        actor = self.actor_observations
        masks = self.action_masks
        if not isinstance(actor, np.ndarray) or actor.dtype != np.dtype(np.float32):
            raise FrameAPIError("actor observations must be a float32 ndarray")
        if actor.ndim != 2 or actor.shape[1] < 1:
            raise FrameAPIError("actor observations must have shape (N_t, actor_width)")
        if not bool(np.all(np.isfinite(actor))):
            raise FrameAPIError("actor observations must contain only finite values")
        if not isinstance(masks, np.ndarray) or masks.dtype != np.dtype(np.bool_):
            raise FrameAPIError("action masks must be a bool ndarray")
        if masks.ndim != 2 or masks.shape[1] < 1:
            raise FrameAPIError("action masks must have shape (N_t, action_count)")

        population = len(self.pair_ids)
        if actor.shape[0] != population or masks.shape[0] != population:
            raise FrameAPIError(
                "pair IDs, actor observations, and action masks must share N_t",
                context={
                    "pair_ids": population,
                    "actor_rows": actor.shape[0],
                    "mask_rows": masks.shape[0],
                },
            )
        if population > 0 and not bool(np.all(np.any(masks, axis=1))):
            raise FrameAPIError("every active pair must have at least one legal action")

        object.__setattr__(
            self,
            "actor_observations",
            cast(ActorObservationArray, _freeze_array(actor)),
        )
        object.__setattr__(
            self,
            "action_masks",
            cast(ActionMaskArray, _freeze_array(masks)),
        )

    @property
    def population_size(self) -> int:
        return len(self.pair_ids)


@dataclass(frozen=True, slots=True)
class FrameStepOutput:
    """Validated vector step result before conversion to the five-part tuple.

    Reward and lifecycle arrays align with ``transition_pair_ids`` from the
    frame that selected ``actions``.  The next observation has its own IDs and
    may therefore have a different row count.
    """

    next_observation: FrameObservation
    transition_pair_ids: tuple[str, ...]
    rewards: RewardArray
    terminated: DoneArray
    truncated: DoneArray
    bootstrap_valid: DoneArray
    learn_mask: DoneArray
    info: FrameInfo = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.next_observation, FrameObservation):
            raise FrameAPIError("step output requires a next FrameObservation")
        _validate_pair_ids(self.transition_pair_ids, name="transition_pair_ids")

        population = len(self.transition_pair_ids)
        for name, value, dtype in (
            ("rewards", self.rewards, np.dtype(np.float32)),
            ("terminated", self.terminated, np.dtype(np.bool_)),
            ("truncated", self.truncated, np.dtype(np.bool_)),
            ("bootstrap_valid", self.bootstrap_valid, np.dtype(np.bool_)),
            ("learn_mask", self.learn_mask, np.dtype(np.bool_)),
        ):
            if not isinstance(value, np.ndarray) or value.dtype != dtype:
                raise FrameAPIError(f"{name} must be a {dtype.name} ndarray")
            if value.shape != (population,):
                raise FrameAPIError(
                    f"{name} must align one-to-one with transition_pair_ids",
                    context={"actual": value.shape, "expected": (population,)},
                )
        if not bool(np.all(np.isfinite(self.rewards))):
            raise FrameAPIError("rewards must contain only finite values")
        if bool(np.any(self.terminated & self.truncated)):
            raise FrameAPIError("a transition cannot be terminated and truncated")
        if bool(np.any(self.bootstrap_valid & ~self.truncated)):
            raise FrameAPIError("only a truncated transition may bootstrap from final state")
        if not isinstance(self.info, Mapping):
            raise FrameAPIError("step info must be a mapping")

        payload = dict(self.info)
        supplied_ids = payload.get("transition_pair_ids")
        if supplied_ids is not None and supplied_ids != self.transition_pair_ids:
            raise FrameAPIError("info transition_pair_ids do not match step outputs")
        payload["transition_pair_ids"] = self.transition_pair_ids

        expected_final_ids = tuple(
            pair_id
            for pair_id, is_valid in zip(
                self.transition_pair_ids, self.bootstrap_valid, strict=True
            )
            if bool(is_valid)
        )
        final_observation = payload.get("final_observation", {})
        if not isinstance(final_observation, Mapping):
            raise FrameAPIError("info final_observation must be a pair-ID mapping")
        supplied_final = dict(final_observation)
        invalid_final_ids = tuple(
            repr(pair_id)
            for pair_id in supplied_final
            if not isinstance(pair_id, str) or not pair_id.strip()
        )
        if invalid_final_ids:
            raise FrameAPIError(
                "info final_observation keys must be non-empty pair IDs",
                context={"invalid_pair_ids": invalid_final_ids},
            )
        missing_final = tuple(
            pair_id for pair_id in expected_final_ids if pair_id not in supplied_final
        )
        unexpected_final = tuple(sorted(set(supplied_final) - set(expected_final_ids)))
        if missing_final or unexpected_final:
            raise FrameAPIError(
                "info final_observation must cover bootstrap-valid pairs exactly",
                context={
                    "missing_pair_ids": missing_final,
                    "unexpected_pair_ids": unexpected_final,
                },
            )
        if any(value is None for value in supplied_final.values()):
            raise FrameAPIError("a final bootstrap observation cannot be None")

        expected_value_bootstrap = (~self.terminated & ~self.truncated) | self.bootstrap_valid
        expected_gae_continuation = ~(self.terminated | self.truncated)
        for name, expected in (
            ("bootstrap_valid", self.bootstrap_valid),
            ("learn_mask", self.learn_mask),
            ("value_bootstrap_mask", expected_value_bootstrap),
            ("gae_continuation_mask", expected_gae_continuation),
        ):
            supplied = payload.get(name)
            if supplied is None:
                continue
            if (
                not isinstance(supplied, np.ndarray)
                or supplied.dtype != np.dtype(np.bool_)
                or supplied.shape != (population,)
                or not np.array_equal(supplied, expected)
            ):
                raise FrameAPIError(f"info {name} does not match step outputs")

        object.__setattr__(
            self,
            "rewards",
            cast(RewardArray, _freeze_array(self.rewards)),
        )
        object.__setattr__(
            self,
            "terminated",
            cast(DoneArray, _freeze_array(self.terminated)),
        )
        object.__setattr__(
            self,
            "truncated",
            cast(DoneArray, _freeze_array(self.truncated)),
        )
        object.__setattr__(
            self,
            "bootstrap_valid",
            cast(DoneArray, _freeze_array(self.bootstrap_valid)),
        )
        object.__setattr__(
            self,
            "learn_mask",
            cast(DoneArray, _freeze_array(self.learn_mask)),
        )
        value_bootstrap_mask = (~self.terminated & ~self.truncated) | self.bootstrap_valid
        value_bootstrap_mask.setflags(write=False)
        gae_continuation_mask = ~(self.terminated | self.truncated)
        gae_continuation_mask.setflags(write=False)
        payload["bootstrap_valid"] = self.bootstrap_valid
        payload["learn_mask"] = self.learn_mask
        payload["value_bootstrap_mask"] = value_bootstrap_mask
        payload["gae_continuation_mask"] = gae_continuation_mask
        payload["final_observation"] = MappingProxyType(supplied_final)
        object.__setattr__(self, "info", MappingProxyType(payload))

    def as_tuple(self) -> StepReturn:
        """Return the Gymnasium-shaped, deliberately vector-valued result."""

        return (
            self.next_observation,
            self.rewards,
            self.terminated,
            self.truncated,
            self.info,
        )


ResetReturn: TypeAlias = tuple[FrameObservation, FrameInfo]
StepReturn: TypeAlias = tuple[
    FrameObservation,
    RewardArray,
    DoneArray,
    DoneArray,
    FrameInfo,
]


@runtime_checkable
class MultiAgentFrameEnv(Protocol):
    """Structural protocol for the Phase 5 variable-population environment."""

    @property
    def api_schema(self) -> FrameAPISchema: ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> ResetReturn: ...

    def step(self, actions: ActionArray) -> StepReturn: ...

    def close(self) -> None: ...


__all__ = [
    "CONTRACT_ACTION_COUNT",
    "CONTRACT_ACTOR_WIDTH",
    "FRAME_API_VERSION",
    "ActionArray",
    "ActionMaskArray",
    "ActorObservationArray",
    "DoneArray",
    "FrameAPIError",
    "FrameAPISchema",
    "FrameInfo",
    "FrameObservation",
    "FrameStepOutput",
    "MultiAgentFrameEnv",
    "ResetReturn",
    "RewardArray",
    "StepReturn",
]
