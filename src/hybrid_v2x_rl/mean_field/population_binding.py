"""Bind variable population frames to stable-ID actor rows and masks.

Phase 2 is the authority for which pair episodes are active and for their
canonical order.  Phase 3 is the authority for hardware-feasible actions.
This module joins those completed boundaries without treating an array row as
an identity: actor rows arrive keyed by stable pair ID, are reordered to match
``PopulationFrame.active_pair_ids``, and receive the frozen profile mask.

The binding is stateful only to enforce chronological frames and population
identity changes.  A pair ID that leaves cannot reappear in the same sampled
episode; the lifecycle contract requires a new episode identity after a gap.
Reset clears that retirement history because a sampled episode is a new state
boundary even when it begins later in the same source trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.action_masks import ActionMask, MaskedActionSpace
from hybrid_v2x_rl.mean_field.environment_api import (
    FrameAPISchema,
    FrameObservation,
)
from hybrid_v2x_rl.mean_field.frames import PopulationFrame


class PopulationBindingError(HybridV2XError):
    """A variable population cannot be aligned to the frame API contract."""


def _require_canonical_ids(ids: tuple[str, ...], *, name: str) -> None:
    if not isinstance(ids, tuple):
        raise PopulationBindingError(f"{name} must be an immutable tuple")
    if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in ids):
        raise PopulationBindingError(f"{name} entries must be non-empty strings")
    if ids != tuple(sorted(ids)):
        raise PopulationBindingError(f"{name} must use canonical stable-ID order")
    if len(ids) != len(set(ids)):
        raise PopulationBindingError(f"{name} cannot contain duplicate IDs")


@dataclass(frozen=True, slots=True)
class PopulationDelta:
    """Stable-ID change from the preceding bound frame to the current one."""

    entered_pair_ids: tuple[str, ...]
    continuing_pair_ids: tuple[str, ...]
    exited_pair_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "entered_pair_ids",
            "continuing_pair_ids",
            "exited_pair_ids",
        ):
            _require_canonical_ids(getattr(self, name), name=name)
        entered = set(self.entered_pair_ids)
        continuing = set(self.continuing_pair_ids)
        exited = set(self.exited_pair_ids)
        if entered & continuing or entered & exited or continuing & exited:
            raise PopulationBindingError("population delta ID sets must be disjoint")


@dataclass(frozen=True, slots=True)
class BoundPopulationFrame:
    """One API observation and its stable-ID population change."""

    observation: FrameObservation
    delta: PopulationDelta

    def __post_init__(self) -> None:
        if not isinstance(self.observation, FrameObservation):
            raise PopulationBindingError("bound population requires a FrameObservation")
        if not isinstance(self.delta, PopulationDelta):
            raise PopulationBindingError("bound population requires a PopulationDelta")
        current = set(self.observation.pair_ids)
        represented = set(self.delta.entered_pair_ids) | set(
            self.delta.continuing_pair_ids
        )
        if represented != current:
            raise PopulationBindingError(
                "entered and continuing IDs must partition the current population"
            )
        if current & set(self.delta.exited_pair_ids):
            raise PopulationBindingError("exited IDs cannot remain in the current frame")


@dataclass(slots=True)
class VariablePopulationBinding:
    """Chronologically bind keyed actor rows to population observations."""

    api_schema: FrameAPISchema
    action_mask: ActionMask
    _trace_id: str | None = field(default=None, init=False, repr=False)
    _expected_frame_index: int | None = field(default=None, init=False, repr=False)
    _current_pair_ids: tuple[str, ...] = field(default=(), init=False, repr=False)
    _has_current_frame: bool = field(default=False, init=False, repr=False)
    _retired_pair_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.api_schema, FrameAPISchema):
            raise PopulationBindingError("population binding requires an API schema")
        if not isinstance(self.action_mask, ActionMask):
            raise PopulationBindingError("population binding requires an ActionMask")
        if len(self.action_mask.values) != self.api_schema.action_count:
            raise PopulationBindingError(
                "profile action mask does not match the API action width"
            )

    @classmethod
    def from_config(cls, config: ProjectConfig) -> VariablePopulationBinding:
        """Bind the frame schema and profile-wide hardware mask to one config."""

        if not isinstance(config, ProjectConfig):
            raise PopulationBindingError(
                "population binding requires a resolved ProjectConfig"
            )
        return cls(
            api_schema=FrameAPISchema.from_config(config),
            action_mask=MaskedActionSpace.from_config(
                config.environment,
                config.rf,
                config.vlc,
            ).mask,
        )

    @property
    def current_pair_ids(self) -> tuple[str, ...]:
        """Stable IDs in the most recently bound frame, empty before binding."""

        return self._current_pair_ids

    def reset(self, trace_id: str, *, start_frame_index: int = 0) -> None:
        """Start a sampled episode and clear all prior identity history."""

        if not isinstance(trace_id, str) or not trace_id.strip():
            raise PopulationBindingError("trace_id must be a non-empty string")
        if (
            not isinstance(start_frame_index, int)
            or isinstance(start_frame_index, bool)
            or start_frame_index < 0
        ):
            raise PopulationBindingError(
                "start_frame_index must be a non-negative integer"
            )
        self._trace_id = trace_id
        self._expected_frame_index = start_frame_index
        self._current_pair_ids = ()
        self._has_current_frame = False
        self._retired_pair_ids.clear()

    def bind_frame(
        self,
        frame: PopulationFrame,
        actor_rows: Mapping[str, Sequence[float]],
    ) -> BoundPopulationFrame:
        """Build one canonical observation without trusting mapping order."""

        if self._trace_id is None or self._expected_frame_index is None:
            raise PopulationBindingError("population binding must be reset first")
        if not isinstance(frame, PopulationFrame):
            raise PopulationBindingError("binding requires a PopulationFrame")
        if frame.trace_id != self._trace_id:
            raise PopulationBindingError(
                "population frame trace does not match the sampled episode",
                context={"actual": frame.trace_id, "expected": self._trace_id},
            )
        if frame.index != self._expected_frame_index:
            raise PopulationBindingError(
                "population frames must be bound without gaps or reordering",
                context={
                    "actual": frame.index,
                    "expected": self._expected_frame_index,
                },
            )
        if not isinstance(actor_rows, Mapping):
            raise PopulationBindingError("actor rows must be keyed by stable pair ID")

        keys = tuple(actor_rows.keys())
        if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in keys):
            raise PopulationBindingError(
                "actor-row mapping keys must be non-empty stable pair IDs"
            )
        expected_ids = frame.active_pair_ids
        actual_ids = set(keys)
        expected_set = set(expected_ids)
        if actual_ids != expected_set:
            raise PopulationBindingError(
                "actor-row IDs do not match the active population",
                context={
                    "missing_pair_ids": tuple(sorted(expected_set - actual_ids)),
                    "unexpected_pair_ids": tuple(sorted(actual_ids - expected_set)),
                },
            )

        reappeared = tuple(sorted(expected_set & self._retired_pair_ids))
        if reappeared:
            raise PopulationBindingError(
                "a retired pair ID cannot reappear without a new episode identity",
                context={"reappeared_pair_ids": reappeared},
            )

        actor_matrix = np.empty(
            (len(expected_ids), self.api_schema.actor_width),
            dtype=np.float32,
        )
        for index, pair_id in enumerate(expected_ids):
            try:
                row = np.asarray(actor_rows[pair_id], dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as error:
                raise PopulationBindingError(
                    "actor row must be a finite numeric sequence",
                    context={"pair_id": pair_id},
                ) from error
            if row.shape != (self.api_schema.actor_width,):
                raise PopulationBindingError(
                    "actor row width does not match the API schema",
                    context={
                        "pair_id": pair_id,
                        "actual": row.shape,
                        "expected": (self.api_schema.actor_width,),
                    },
                )
            if not bool(np.all(np.isfinite(row))):
                raise PopulationBindingError(
                    "actor row must contain only finite values",
                    context={"pair_id": pair_id},
                )
            actor_matrix[index] = row

        mask_row = np.asarray(self.action_mask.values, dtype=np.bool_)
        masks = np.broadcast_to(
            mask_row,
            (len(expected_ids), self.api_schema.action_count),
        ).copy()
        observation = FrameObservation(
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            pair_ids=expected_ids,
            actor_observations=actor_matrix,
            action_masks=masks,
        )
        self.api_schema.validate_observation(observation)

        previous_set = set(self._current_pair_ids) if self._has_current_frame else set()
        entered = tuple(sorted(expected_set - previous_set))
        continuing = tuple(sorted(expected_set & previous_set))
        exited = tuple(sorted(previous_set - expected_set))
        delta = PopulationDelta(
            entered_pair_ids=entered,
            continuing_pair_ids=continuing,
            exited_pair_ids=exited,
        )
        bound = BoundPopulationFrame(observation=observation, delta=delta)

        self._retired_pair_ids.update(exited)
        self._current_pair_ids = expected_ids
        self._has_current_frame = True
        self._expected_frame_index = frame.index + 1
        return bound


__all__ = [
    "BoundPopulationFrame",
    "PopulationBindingError",
    "PopulationDelta",
    "VariablePopulationBinding",
]
