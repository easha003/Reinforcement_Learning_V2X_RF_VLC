"""Identity-preserving resource accounting for one population action frame.

The action-mask layer resolves raw policy output before this module is called.
Consequently, the aggregation boundary accepts only exact ``PolicyAction``
members.  It has no channel state or outcome inputs: every committed RF
attempt contributes to current offered demand, including reservations from
flows whose physical endpoints overlap. RF use, VLC use, duplication, cost,
and reward are all derived from that same authoritative action record. A frame
is rebuilt only from its current actions: ``VLC`` records an explicit RF
reservation release, so no RF or DUP reservation can leak in from an earlier
frame. Lifecycle is copied from the population frame onto the same row: born
and final pairs act normally in that frame, while final rows explicitly direct
the environment to release pair state only after their packet is processed.
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
    from hybrid_v2x_rl.mean_field.frames import PopulationFrame, PopulationPair


class ActionAggregationError(HybridV2XError):
    """A joint action cannot be bound exactly to one population frame."""


@dataclass(frozen=True, slots=True)
class PairActionLifecycle:
    """Lifecycle snapshot aligned with one active pair's current packet."""

    episode_step: int
    born: bool
    terminated: bool
    truncated: bool
    bootstrap_valid: bool
    end_reason: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_step, int)
            or isinstance(self.episode_step, bool)
            or self.episode_step < 0
        ):
            raise ActionAggregationError(
                "episode_step must be a non-negative integer"
            )
        if any(
            type(flag) is not bool
            for flag in (
                self.born,
                self.terminated,
                self.truncated,
                self.bootstrap_valid,
            )
        ):
            raise ActionAggregationError("lifecycle flags must be booleans")
        if self.born != (self.episode_step == 0):
            raise ActionAggregationError(
                "born must be true exactly on episode step zero",
                context={"episode_step": self.episode_step, "born": self.born},
            )
        if self.terminated and self.truncated:
            raise ActionAggregationError(
                "a pair action cannot be both terminated and truncated"
            )
        if self.end_reason is not None and (
            not isinstance(self.end_reason, str) or not self.end_reason.strip()
        ):
            raise ActionAggregationError(
                "end_reason must be a non-empty string when provided"
            )
        if self.final != (self.end_reason is not None):
            raise ActionAggregationError(
                "final lifecycle flags and end_reason must appear together"
            )
        if self.bootstrap_valid and not self.truncated:
            raise ActionAggregationError(
                "only a truncated final packet may carry a valid bootstrap"
            )

    @classmethod
    def from_pair(cls, pair: PopulationPair) -> PairActionLifecycle:
        """Copy validated replay metadata without retaining mutable frame state."""

        return cls(
            episode_step=pair.episode_step,
            born=pair.lifecycle.born,
            terminated=pair.lifecycle.terminated,
            truncated=pair.lifecycle.truncated,
            bootstrap_valid=pair.lifecycle.bootstrap_valid,
            end_reason=pair.lifecycle.end_reason,
        )

    @property
    def continuing(self) -> bool:
        return not self.born

    @property
    def final(self) -> bool:
        return self.terminated or self.truncated

    @property
    def release_after_frame(self) -> bool:
        """Final packets act first; their pair state is released afterward."""

        return self.final


@dataclass(frozen=True, slots=True)
class PairActionReservation:
    """One active pair's validated action and committed resource selection."""

    pair_id: str
    action: PolicyAction
    lifecycle: PairActionLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ActionAggregationError("pair_id must be a non-empty string")
        if type(self.action) is not PolicyAction:
            raise ActionAggregationError(
                "frame aggregation requires a mask-validated PolicyAction",
                context={"pair_id": self.pair_id, "action": repr(self.action)},
            )
        if not isinstance(self.lifecycle, PairActionLifecycle):
            raise ActionAggregationError(
                "pair action must carry a validated lifecycle snapshot",
                context={"pair_id": self.pair_id},
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

    @property
    def rf_reservation_released(self) -> bool:
        """Whether this action returns the pair's current-frame RF reservation."""

        return not action_resources(self.action).uses_rf

    def account(self, resource_map: ActionResourceMap) -> PairResourceAccounting:
        """Materialize this packet's resources, configured cost, and reward."""

        activation_cost = resource_map.activation_cost(self.action)
        return PairResourceAccounting(
            pair_id=self.pair_id,
            action=self.action,
            lifecycle=self.lifecycle,
            reserved_rf_attempts=self.reserved_rf_attempts,
            vlc_activations=self.vlc_activations,
            uses_rf=self.uses_rf,
            uses_vlc=self.uses_vlc,
            duplicates=self.duplicates,
            rf_reservation_released=self.rf_reservation_released,
            activation_cost=activation_cost,
            reward=resource_map.reward(self.action),
        )


@dataclass(frozen=True, slots=True)
class PairResourceAccounting:
    """Complete action-level resource record for one generated packet."""

    pair_id: str
    action: PolicyAction
    lifecycle: PairActionLifecycle
    reserved_rf_attempts: int
    vlc_activations: int
    uses_rf: bool
    uses_vlc: bool
    duplicates: bool
    rf_reservation_released: bool
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
        if not isinstance(self.lifecycle, PairActionLifecycle):
            raise ActionAggregationError(
                "packet accounting must carry a validated lifecycle snapshot",
                context={"pair_id": self.pair_id},
            )
        if (
            not isinstance(self.reserved_rf_attempts, int)
            or isinstance(self.reserved_rf_attempts, bool)
            or not isinstance(self.vlc_activations, int)
            or isinstance(self.vlc_activations, bool)
            or self.vlc_activations not in (0, 1)
            or any(
                type(flag) is not bool
                for flag in (
                    self.uses_rf,
                    self.uses_vlc,
                    self.duplicates,
                    self.rf_reservation_released,
                )
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
            not spec.uses_rf,
        )
        actual = (
            self.reserved_rf_attempts,
            self.vlc_activations,
            self.uses_rf,
            self.uses_vlc,
            self.duplicates,
            self.rf_reservation_released,
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
class ResourceAccountingTotals:
    """Comparable per-agent or population resource-accounting totals."""

    active_pairs: int
    reserved_rf_attempts: int
    vlc_activations: int
    rf_using_pairs: int
    vlc_using_pairs: int
    duplicated_pairs: int
    rf_reservation_releases: int
    activation_cost: float
    reward: float

    def __post_init__(self) -> None:
        counts = {
            "active_pairs": self.active_pairs,
            "reserved_rf_attempts": self.reserved_rf_attempts,
            "vlc_activations": self.vlc_activations,
            "rf_using_pairs": self.rf_using_pairs,
            "vlc_using_pairs": self.vlc_using_pairs,
            "duplicated_pairs": self.duplicated_pairs,
            "rf_reservation_releases": self.rf_reservation_releases,
        }
        invalid_counts = tuple(
            name
            for name, value in counts.items()
            if not isinstance(value, int) or isinstance(value, bool) or value < 0
        )
        if invalid_counts:
            raise ActionAggregationError(
                "accounting totals require non-negative integer counts",
                context={"invalid_fields": invalid_counts},
            )
        if not math.isfinite(self.activation_cost) or self.activation_cost < 0.0:
            raise ActionAggregationError(
                "total activation cost must be finite and non-negative"
            )
        if not math.isfinite(self.reward) or self.reward > 0.0:
            raise ActionAggregationError("total reward must be finite and non-positive")

    @classmethod
    def from_records(
        cls,
        records: tuple[PairResourceAccounting, ...],
    ) -> ResourceAccountingTotals:
        """Sum immutable per-agent rows without rounding intermediate values."""

        if any(not isinstance(record, PairResourceAccounting) for record in records):
            raise ActionAggregationError(
                "accounting totals require PairResourceAccounting rows"
            )
        return cls(
            active_pairs=len(records),
            reserved_rf_attempts=sum(
                record.reserved_rf_attempts for record in records
            ),
            vlc_activations=sum(record.vlc_activations for record in records),
            rf_using_pairs=sum(record.uses_rf for record in records),
            vlc_using_pairs=sum(record.uses_vlc for record in records),
            duplicated_pairs=sum(record.duplicates for record in records),
            rf_reservation_releases=sum(
                record.rf_reservation_released for record in records
            ),
            activation_cost=math.fsum(record.activation_cost for record in records),
            reward=math.fsum(record.reward for record in records),
        )

    def as_dict(self) -> dict[str, int | float]:
        """Return stable machine-readable fields for diagnostics and artifacts."""

        return {
            "active_pairs": self.active_pairs,
            "reserved_rf_attempts": self.reserved_rf_attempts,
            "vlc_activations": self.vlc_activations,
            "rf_using_pairs": self.rf_using_pairs,
            "vlc_using_pairs": self.vlc_using_pairs,
            "duplicated_pairs": self.duplicated_pairs,
            "rf_reservation_releases": self.rf_reservation_releases,
            "activation_cost": self.activation_cost,
            "reward": self.reward,
        }


_ACCOUNTING_TOTAL_FIELDS = tuple(ResourceAccountingTotals.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class AccountingConservationAudit:
    """Exact reconciliation of summed agent rows and published frame totals."""

    trace_id: str
    frame_index: int
    per_agent_totals: ResourceAccountingTotals
    population_totals: ResourceAccountingTotals

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ActionAggregationError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise ActionAggregationError("frame_index must be a non-negative integer")
        if not isinstance(self.per_agent_totals, ResourceAccountingTotals) or not isinstance(
            self.population_totals, ResourceAccountingTotals
        ):
            raise ActionAggregationError(
                "conservation audit requires validated accounting totals"
            )

    @property
    def mismatched_fields(self) -> tuple[str, ...]:
        """Fields that fail exact per-agent-to-population reconciliation."""

        return tuple(
            field
            for field in _ACCOUNTING_TOTAL_FIELDS
            if getattr(self.per_agent_totals, field)
            != getattr(self.population_totals, field)
        )

    @property
    def passed(self) -> bool:
        return not self.mismatched_fields

    def require_conserved(self) -> AccountingConservationAudit:
        """Return this audit when valid; otherwise fail with reproducible context."""

        mismatches = self.mismatched_fields
        if mismatches:
            raise ActionAggregationError(
                "per-agent resource accounting does not conserve population totals",
                context={
                    "trace_id": self.trace_id,
                    "frame_index": self.frame_index,
                    "mismatched_fields": mismatches,
                    "per_agent_totals": self.per_agent_totals.as_dict(),
                    "population_totals": self.population_totals.as_dict(),
                },
            )
        return self


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
            PairActionReservation(
                pair_id=pair.pair_id,
                action=actions[pair.pair_id],
                lifecycle=PairActionLifecycle.from_pair(pair),
            )
            for pair in frame.pairs
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
    def born_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.lifecycle.born
        )

    @property
    def continuing_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.lifecycle.continuing
        )

    @property
    def terminated_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.lifecycle.terminated
        )

    @property
    def truncated_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.lifecycle.truncated
        )

    @property
    def bootstrap_valid_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.lifecycle.bootstrap_valid
        )

    @property
    def release_after_frame_pair_ids(self) -> tuple[str, ...]:
        """Pairs whose state must be removed after their current packet."""

        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.lifecycle.release_after_frame
        )

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
    def rf_reservation_releases_by_pair(self) -> tuple[tuple[str, bool], ...]:
        """Explicit current-frame release decision for every active pair."""

        return tuple(
            (record.pair_id, record.rf_reservation_released)
            for record in self.pair_accounting
        )

    @property
    def released_rf_pair_ids(self) -> tuple[str, ...]:
        """Pairs whose current action leaves no reservation in RF demand."""

        return tuple(
            record.pair_id
            for record in self.pair_accounting
            if record.rf_reservation_released
        )

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

    @property
    def population_totals(self) -> ResourceAccountingTotals:
        """Published frame aggregates in the same schema as summed agent rows."""

        return ResourceAccountingTotals(
            active_pairs=self.active_pairs,
            reserved_rf_attempts=self.total_reserved_rf_attempts,
            vlc_activations=self.total_vlc_activations,
            rf_using_pairs=self.rf_using_pairs,
            vlc_using_pairs=self.vlc_using_pairs,
            duplicated_pairs=self.duplicated_pairs,
            rf_reservation_releases=len(self.released_rf_pair_ids),
            activation_cost=self.total_activation_cost,
            reward=self.total_reward,
        )

    def audit_accounting_conservation(self) -> AccountingConservationAudit:
        """Reconcile every resource total against one captured set of agent rows."""

        audit = AccountingConservationAudit(
            trace_id=self.trace_id,
            frame_index=self.frame_index,
            per_agent_totals=ResourceAccountingTotals.from_records(
                self.pair_accounting
            ),
            population_totals=self.population_totals,
        )
        return audit.require_conserved()


__all__ = [
    "AccountingConservationAudit",
    "ActionAggregationError",
    "FrameActionLedger",
    "PairActionLifecycle",
    "PairActionReservation",
    "PairResourceAccounting",
    "ResourceAccountingTotals",
]
