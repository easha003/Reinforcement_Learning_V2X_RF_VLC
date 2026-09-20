"""Identity-preserving resource accounting for one population action frame.

The action-mask layer resolves raw policy output before this module is called.
Consequently, the aggregation boundary accepts only exact ``PolicyAction``
members.  It has no channel state or outcome inputs: every committed RF
attempt contributes to current offered demand, including reservations from
flows whose physical endpoints overlap. RF use, VLC use, duplication, cost,
and reward are all derived from that same authoritative action record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    ActionResourceMap,
    PolicyAction,
    action_resources,
)

if TYPE_CHECKING:
    from hybrid_v2x_rl.mean_field.frames import PopulationFrame


class ActionAggregationError(HybridV2XError):
    """A joint action cannot be bound exactly to one population frame."""


@dataclass(frozen=True, slots=True)
class PairActionReservation:
    """One active pair's validated action and committed resource selection."""

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

    @property
    def vlc_activations(self) -> int:
        return action_resources(self.action).vlc_activations

    @property
    def uses_rf(self) -> bool:
        return action_resources(self.action).uses_rf

    @property
    def uses_vlc(self) -> bool:
        return action_resources(self.action).uses_vlc

    @property
    def duplicates(self) -> bool:
        return action_resources(self.action).duplicates

    def account(self, resource_map: ActionResourceMap) -> PairResourceAccounting:
        """Materialize this packet's resources, configured cost, and reward."""

        activation_cost = resource_map.activation_cost(self.action)
        return PairResourceAccounting(
            pair_id=self.pair_id,
            action=self.action,
            reserved_rf_attempts=self.reserved_rf_attempts,
            vlc_activations=self.vlc_activations,
            uses_rf=self.uses_rf,
            uses_vlc=self.uses_vlc,
            duplicates=self.duplicates,
            activation_cost=activation_cost,
            reward=resource_map.reward(self.action),
        )


@dataclass(frozen=True, slots=True)
class PairResourceAccounting:
    """Complete action-level resource record for one generated packet."""

    pair_id: str
    action: PolicyAction
    reserved_rf_attempts: int
    vlc_activations: int
    uses_rf: bool
    uses_vlc: bool
    duplicates: bool
    activation_cost: float
    reward: float

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ActionAggregationError("pair_id must be a non-empty string")
        if type(self.action) is not PolicyAction:
            raise ActionAggregationError(
                "packet accounting requires a mask-validated PolicyAction",
                context={"pair_id": self.pair_id, "action": repr(self.action)},
            )
        if (
            not isinstance(self.reserved_rf_attempts, int)
            or isinstance(self.reserved_rf_attempts, bool)
            or not isinstance(self.vlc_activations, int)
            or isinstance(self.vlc_activations, bool)
            or self.vlc_activations not in (0, 1)
            or any(
                type(flag) is not bool
                for flag in (self.uses_rf, self.uses_vlc, self.duplicates)
            )
        ):
            raise ActionAggregationError(
                "packet resource fields have invalid types or ranges",
                context={"pair_id": self.pair_id},
            )
        spec = action_resources(self.action)
        expected = (
            spec.reserved_rf_attempts,
            spec.vlc_activations,
            spec.uses_rf,
            spec.uses_vlc,
            spec.duplicates,
        )
        actual = (
            self.reserved_rf_attempts,
            self.vlc_activations,
            self.uses_rf,
            self.uses_vlc,
            self.duplicates,
        )
        if actual != expected:
            raise ActionAggregationError(
                "packet resources do not match the authoritative action mapping",
                context={"pair_id": self.pair_id, "action": spec.name},
            )
        if not math.isfinite(self.activation_cost) or self.activation_cost <= 0.0:
            raise ActionAggregationError(
                "packet activation_cost must be finite and positive",
                context={"pair_id": self.pair_id},
            )
        if not math.isfinite(self.reward) or self.reward != -self.activation_cost:
            raise ActionAggregationError(
                "packet reward must equal negative activation cost",
                context={"pair_id": self.pair_id},
            )

    @property
    def action_index(self) -> int:
        return int(self.action)

    @property
    def action_name(self) -> str:
        return self.action.label


@dataclass(frozen=True, slots=True)
class FrameActionLedger:
    """Canonical per-packet rows and exact current-frame resource totals."""

    trace_id: str
    frame_index: int
    time_s: float
    resource_map: ActionResourceMap
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
        if not isinstance(self.resource_map, ActionResourceMap):
            raise ActionAggregationError(
                "resource_map must be a validated ActionResourceMap"
            )
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
        *,
        resource_map: ActionResourceMap,
    ) -> FrameActionLedger:
        """Bind one action and the configured resource prices to every active pair."""

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
            resource_map=resource_map,
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
    def pair_accounting(self) -> tuple[PairResourceAccounting, ...]:
        """Complete per-packet records in the frame's stable pair-ID order."""

        return tuple(
            reservation.account(self.resource_map)
            for reservation in self.reservations
        )

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

    @property
    def total_vlc_activations(self) -> int:
        return sum(record.vlc_activations for record in self.pair_accounting)

    @property
    def rf_using_pairs(self) -> int:
        return sum(record.uses_rf for record in self.pair_accounting)

    @property
    def vlc_using_pairs(self) -> int:
        return sum(record.uses_vlc for record in self.pair_accounting)

    @property
    def duplicated_pairs(self) -> int:
        return sum(record.duplicates for record in self.pair_accounting)

    @property
    def activation_costs_by_pair(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (record.pair_id, record.activation_cost)
            for record in self.pair_accounting
        )

    @property
    def rewards_by_pair(self) -> tuple[tuple[str, float], ...]:
        return tuple(
            (record.pair_id, record.reward) for record in self.pair_accounting
        )

    @property
    def total_activation_cost(self) -> float:
        return math.fsum(record.activation_cost for record in self.pair_accounting)

    @property
    def total_reward(self) -> float:
        return -self.total_activation_cost


__all__ = [
    "ActionAggregationError",
    "FrameActionLedger",
    "PairActionReservation",
    "PairResourceAccounting",
]
