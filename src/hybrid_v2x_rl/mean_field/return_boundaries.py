"""Pair-aligned lifecycle masks for value and advantage estimation.

Every active pair owns its final packet.  The packet reward and cost therefore
remain learnable on the row that carries a terminal or truncation flag; only
the value bootstrap and recursive advantage continuation change at that
boundary.  This module derives those decisions once from the action ledger and
the causal actor frame so later rollout code cannot infer them from a single
ambiguous ``done`` flag.

``bootstrap_valid`` identifies an internal truncation whose next physical
trace observation exists and must be supplied separately as
``final_observation``.  ``value_bootstrap_mask`` is broader: it is also true on
ordinary continuing rows, whose next observation arrives through the normal
step result.  ``gae_continuation_mask`` is false at every pair boundary, so a
recursive return never crosses into a reset episode even when the boundary has
a valid one-step value bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.actor_observations import CausalActorFrame

LifecycleArray: TypeAlias = NDArray[np.bool_]
LifecycleInfo: TypeAlias = Mapping[str, object]


class ReturnBoundaryError(HybridV2XError):
    """A lifecycle batch cannot define unambiguous return-estimation masks."""


def _freeze_bool(values: tuple[bool, ...]) -> LifecycleArray:
    array = np.asarray(values, dtype=np.bool_)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FrameReturnBoundary:
    """Immutable lifecycle decisions aligned with one acted population."""

    trace_id: str
    frame_index: int
    time_s: float
    pair_ids: tuple[str, ...]
    terminated: LifecycleArray
    truncated: LifecycleArray
    bootstrap_valid: LifecycleArray
    learn_mask: LifecycleArray
    value_bootstrap_mask: LifecycleArray
    gae_continuation_mask: LifecycleArray
    end_reasons: tuple[str | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ReturnBoundaryError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise ReturnBoundaryError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ReturnBoundaryError("time_s must be finite and non-negative")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(self.pair_ids) != len(
            set(self.pair_ids)
        ):
            raise ReturnBoundaryError("pair_ids must be unique and canonically ordered")
        if len(self.end_reasons) != len(self.pair_ids):
            raise ReturnBoundaryError("end_reasons must align with pair_ids")
        if any(
            reason is not None and (not isinstance(reason, str) or not reason.strip())
            for reason in self.end_reasons
        ):
            raise ReturnBoundaryError("end reasons must be non-empty strings when provided")

        population = len(self.pair_ids)
        arrays: dict[str, LifecycleArray] = {}
        for name in (
            "terminated",
            "truncated",
            "bootstrap_valid",
            "learn_mask",
            "value_bootstrap_mask",
            "gae_continuation_mask",
        ):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.bool_):
                raise ReturnBoundaryError(f"{name} must be a bool ndarray")
            if value.shape != (population,):
                raise ReturnBoundaryError(f"{name} must have shape (N_t,)")
            frozen = value.copy(order="C")
            frozen.setflags(write=False)
            arrays[name] = cast(LifecycleArray, frozen)

        terminated = arrays["terminated"]
        truncated = arrays["truncated"]
        bootstrap_valid = arrays["bootstrap_valid"]
        expected_value_bootstrap = (~terminated & ~truncated) | bootstrap_valid
        expected_gae_continuation = ~(terminated | truncated)
        if bool(np.any(terminated & truncated)):
            raise ReturnBoundaryError("a transition cannot terminate and truncate")
        if bool(np.any(bootstrap_valid & ~truncated)):
            raise ReturnBoundaryError("only a truncation may use a final bootstrap")
        if not np.array_equal(arrays["value_bootstrap_mask"], expected_value_bootstrap):
            raise ReturnBoundaryError("value_bootstrap_mask does not match lifecycle semantics")
        if not np.array_equal(arrays["gae_continuation_mask"], expected_gae_continuation):
            raise ReturnBoundaryError("gae_continuation_mask does not stop at every pair boundary")

        final = terminated | truncated
        reason_present = np.asarray(
            tuple(reason is not None for reason in self.end_reasons),
            dtype=np.bool_,
        )
        if not np.array_equal(final, reason_present):
            raise ReturnBoundaryError("end reasons must appear exactly on final transition rows")
        for name, frozen in arrays.items():
            object.__setattr__(self, name, frozen)

    @classmethod
    def from_frame(
        cls,
        ledger: FrameActionLedger,
        actor_frame: CausalActorFrame,
    ) -> FrameReturnBoundary:
        """Derive masks only after proving action and observation identity alignment."""

        if not isinstance(ledger, FrameActionLedger):
            raise ReturnBoundaryError("return boundary requires a frame action ledger")
        if not isinstance(actor_frame, CausalActorFrame):
            raise ReturnBoundaryError("return boundary requires a causal actor frame")
        mismatches: dict[str, object] = {}
        if ledger.trace_id != actor_frame.trace_id:
            mismatches["trace_id"] = (ledger.trace_id, actor_frame.trace_id)
        if ledger.frame_index != actor_frame.frame_index:
            mismatches["frame_index"] = (
                ledger.frame_index,
                actor_frame.frame_index,
            )
        if not math.isclose(
            ledger.time_s,
            actor_frame.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            mismatches["time_s"] = (ledger.time_s, actor_frame.time_s)
        if ledger.pair_ids != actor_frame.pair_ids:
            mismatches["pair_ids"] = (ledger.pair_ids, actor_frame.pair_ids)
        if mismatches:
            raise ReturnBoundaryError(
                "action and actor frames are not the same decision frame",
                context=mismatches,
            )

        terminated_values = tuple(row.lifecycle.terminated for row in ledger.reservations)
        truncated_values = tuple(row.lifecycle.truncated for row in ledger.reservations)
        bootstrap_values = tuple(row.lifecycle.bootstrap_valid for row in ledger.reservations)
        terminated = _freeze_bool(terminated_values)
        truncated = _freeze_bool(truncated_values)
        bootstrap_valid = _freeze_bool(bootstrap_values)
        return cls(
            trace_id=ledger.trace_id,
            frame_index=ledger.frame_index,
            time_s=ledger.time_s,
            pair_ids=ledger.pair_ids,
            terminated=terminated,
            truncated=truncated,
            bootstrap_valid=bootstrap_valid,
            learn_mask=_freeze_bool(actor_frame.usable_mask),
            value_bootstrap_mask=(~terminated & ~truncated) | bootstrap_valid,
            gae_continuation_mask=~(terminated | truncated),
            end_reasons=tuple(row.lifecycle.end_reason for row in ledger.reservations),
        )

    @property
    def final_pair_ids(self) -> tuple[str, ...]:
        final = self.terminated | self.truncated
        return tuple(
            pair_id
            for pair_id, is_final in zip(self.pair_ids, final, strict=True)
            if bool(is_final)
        )

    @property
    def bootstrap_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            pair_id
            for pair_id, is_valid in zip(self.pair_ids, self.bootstrap_valid, strict=True)
            if bool(is_valid)
        )

    @property
    def zero_bootstrap_final_pair_ids(self) -> tuple[str, ...]:
        final_without_bootstrap = (self.terminated | self.truncated) & (~self.bootstrap_valid)
        return tuple(
            pair_id
            for pair_id, uses_zero in zip(self.pair_ids, final_without_bootstrap, strict=True)
            if bool(uses_zero)
        )

    def as_step_info(
        self,
        *,
        final_observation: Mapping[str, object] | None = None,
    ) -> LifecycleInfo:
        """Build step metadata and require every valid final bootstrap observation."""

        supplied = {} if final_observation is None else dict(final_observation)
        invalid_ids = tuple(
            repr(pair_id)
            for pair_id in supplied
            if not isinstance(pair_id, str) or not pair_id.strip()
        )
        if invalid_ids:
            raise ReturnBoundaryError(
                "final_observation keys must be non-empty pair IDs",
                context={"invalid_pair_ids": invalid_ids},
            )
        expected_ids = set(self.bootstrap_pair_ids)
        supplied_ids = set(supplied)
        missing = tuple(
            pair_id for pair_id in self.bootstrap_pair_ids if pair_id not in supplied_ids
        )
        unexpected = tuple(sorted(supplied_ids - expected_ids))
        if missing or unexpected:
            raise ReturnBoundaryError(
                "final_observation must cover bootstrap-valid truncations exactly",
                context={
                    "missing_pair_ids": missing,
                    "unexpected_pair_ids": unexpected,
                },
            )
        if any(value is None for value in supplied.values()):
            raise ReturnBoundaryError("a final bootstrap observation cannot be None")
        return MappingProxyType(
            {
                "transition_pair_ids": self.pair_ids,
                "bootstrap_valid": self.bootstrap_valid,
                "learn_mask": self.learn_mask,
                "value_bootstrap_mask": self.value_bootstrap_mask,
                "gae_continuation_mask": self.gae_continuation_mask,
                "end_reason": self.end_reasons,
                "final_observation": MappingProxyType(supplied),
                "release_after_frame_pair_ids": self.final_pair_ids,
            }
        )


__all__ = [
    "FrameReturnBoundary",
    "LifecycleArray",
    "LifecycleInfo",
    "ReturnBoundaryError",
]
