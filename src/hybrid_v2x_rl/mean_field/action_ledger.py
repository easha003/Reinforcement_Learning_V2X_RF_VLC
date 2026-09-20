"""Identity-preserving joint actions and RF demand for one population frame.

The action-mask layer resolves raw policy output before this module is called.
Consequently, the aggregation boundary accepts only exact ``PolicyAction``
members.  It has no channel state or outcome inputs: every committed RF
attempt contributes to current offered demand, including reservations from
flows whose physical endpoints overlap.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources

if TYPE_CHECKING:
    from hybrid_v2x_rl.mean_field.frames import PopulationFrame


class ActionAggregationError(HybridV2XError):
    """A joint action cannot be bound exactly to one population frame."""


@dataclass(frozen=True, slots=True)
class PairActionReservation:
    """One active pair's validated action and committed RF reservation."""

    pair_id: str
    action: PolicyAction

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ActionAggregationError("pair_id must be a non-empty string")
        if type(self.action) is not PolicyAction:
            raise ActionAggregationError(
                "frame aggregation requires a mask-validated PolicyAction",
                context={"pair_id": self.pair_id, "action": repr(self.action)},
            )

    @property
    def reserved_rf_attempts(self) -> int:
        """All attempts committed by this pair, whether or not decoding succeeds."""

        return action_resources(self.action).reserved_rf_attempts


@dataclass(frozen=True, slots=True)
class FrameActionLedger:
    """Canonical joint-action rows and their exact current-frame RF demand."""

    trace_id: str
    frame_index: int
    time_s: float
    reservations: tuple[PairActionReservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ActionAggregationError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise ActionAggregationError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ActionAggregationError("time_s must be finite and non-negative")
        if any(
            not isinstance(reservation, PairActionReservation)
            for reservation in self.reservations
        ):
            raise ActionAggregationError(
                "reservations must contain PairActionReservation rows"
            )
        pair_ids = self.pair_ids
        if pair_ids != tuple(sorted(pair_ids)):
            raise ActionAggregationError(
                "pair reservations must follow canonical frame pair-ID order"
            )
        if len(pair_ids) != len(set(pair_ids)):
            raise ActionAggregationError(
                "a frame action ledger cannot contain duplicate pair IDs"
            )

    @classmethod
    def from_frame(
        cls,
        frame: PopulationFrame,
        actions_by_pair: Mapping[str, PolicyAction],
    ) -> FrameActionLedger:
        """Bind exactly one validated action to every active pair in ``frame``."""

        actions = dict(actions_by_pair)
        invalid_pair_ids = tuple(
            repr(pair_id)
            for pair_id in actions
            if not isinstance(pair_id, str) or not pair_id.strip()
        )
        if invalid_pair_ids:
            raise ActionAggregationError(
                "joint-action keys must be non-empty pair-ID strings",
                context={"invalid_pair_ids": invalid_pair_ids},
            )

        expected = frame.active_pair_ids
        expected_set = set(expected)
        missing = tuple(pair_id for pair_id in expected if pair_id not in actions)
        unexpected = tuple(
            sorted(pair_id for pair_id in actions if pair_id not in expected_set)
        )
        if missing or unexpected:
            raise ActionAggregationError(
                "joint action does not cover the active population exactly",
                context={
                    "trace_id": frame.trace_id,
                    "frame_index": frame.index,
                    "missing_pair_ids": missing,
                    "unexpected_pair_ids": unexpected,
                },
            )

        reservations = tuple(
            PairActionReservation(pair_id=pair_id, action=actions[pair_id])
            for pair_id in expected
        )
        return cls(
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            reservations=reservations,
        )

    @property
    def pair_ids(self) -> tuple[str, ...]:
        """Stable identity-to-row mapping inherited from the population frame."""

        return tuple(reservation.pair_id for reservation in self.reservations)

    @property
    def active_pairs(self) -> int:
        return len(self.reservations)

    @property
    def reserved_rf_attempts_by_pair(self) -> tuple[tuple[str, int], ...]:
        """Per-pair terms whose sum defines contract quantity ``D_t``."""

        return tuple(
            (reservation.pair_id, reservation.reserved_rf_attempts)
            for reservation in self.reservations
        )

    @property
    def total_reserved_rf_attempts(self) -> int:
        """Current offered RF demand ``D_t`` in reserved attempts per frame."""

        return sum(
            reservation.reserved_rf_attempts for reservation in self.reservations
        )


__all__ = [
    "ActionAggregationError",
    "FrameActionLedger",
    "PairActionReservation",
]
