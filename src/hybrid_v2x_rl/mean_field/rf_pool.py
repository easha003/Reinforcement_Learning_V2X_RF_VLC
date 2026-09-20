"""Action-coupled input boundary for the shared RF pool.

Phase 3 binds one complete joint action to a population frame.  This module is
the handoff into Phase 4: it captures the resulting committed RF reservations
only after that ledger is complete and its per-agent accounting conserves.

``offered_rf_attempts`` is the contract quantity ``D_t`` in reserved attempts
per frame.  It is deliberately not clipped and is not inferred from neighbour
count or a fixed RF-use fraction.  Later Phase 4 components will map this
snapshot to airtime demand, CBR, and collision risk without reopening the
action ledger or observing a partial joint action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import MAX_RESERVED_RF_ATTEMPTS
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger


class RFPoolError(HybridV2XError):
    """The action-coupled RF-pool input violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class RFPoolDemand:
    """Audited current-frame demand presented to the shared RF-pool model."""

    trace_id: str
    frame_index: int
    time_s: float
    active_pairs: int
    reserved_rf_attempts_by_pair: tuple[tuple[str, int], ...]
    offered_rf_attempts: int
    rf_using_pairs: int

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise RFPoolError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise RFPoolError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise RFPoolError("time_s must be finite and non-negative")

        for name, value in (
            ("active_pairs", self.active_pairs),
            ("offered_rf_attempts", self.offered_rf_attempts),
            ("rf_using_pairs", self.rf_using_pairs),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RFPoolError(
                    f"{name} must be a non-negative integer",
                    context={name: value},
                )

        if not isinstance(self.reserved_rf_attempts_by_pair, tuple):
            raise RFPoolError("per-pair RF reservations must be an immutable tuple")
        invalid_rows = tuple(
            repr(row)
            for row in self.reserved_rf_attempts_by_pair
            if (
                not isinstance(row, tuple)
                or len(row) != 2
                or not isinstance(row[0], str)
                or not row[0].strip()
                or not isinstance(row[1], int)
                or isinstance(row[1], bool)
                or not 0 <= row[1] <= MAX_RESERVED_RF_ATTEMPTS
            )
        )
        if invalid_rows:
            raise RFPoolError(
                "per-pair RF reservations contain invalid rows",
                context={"invalid_rows": invalid_rows},
            )

        pair_ids = tuple(pair_id for pair_id, _ in self.reserved_rf_attempts_by_pair)
        if pair_ids != tuple(sorted(pair_ids)):
            raise RFPoolError("per-pair RF reservations must use canonical pair-ID order")
        if len(pair_ids) != len(set(pair_ids)):
            raise RFPoolError("per-pair RF reservations cannot repeat a pair ID")
        if self.active_pairs != len(self.reserved_rf_attempts_by_pair):
            raise RFPoolError(
                "active-pair count does not match per-pair RF reservations",
                context={
                    "active_pairs": self.active_pairs,
                    "reservation_rows": len(self.reserved_rf_attempts_by_pair),
                },
            )

        summed_attempts = sum(
            attempts for _, attempts in self.reserved_rf_attempts_by_pair
        )
        if self.offered_rf_attempts != summed_attempts:
            raise RFPoolError(
                "offered RF attempts do not equal the per-pair reservation sum",
                context={
                    "offered_rf_attempts": self.offered_rf_attempts,
                    "summed_rf_attempts": summed_attempts,
                },
            )
        counted_rf_users = sum(
            attempts > 0 for _, attempts in self.reserved_rf_attempts_by_pair
        )
        if self.rf_using_pairs != counted_rf_users:
            raise RFPoolError(
                "RF-using pair count does not match per-pair reservations",
                context={
                    "rf_using_pairs": self.rf_using_pairs,
                    "counted_rf_users": counted_rf_users,
                },
            )

    @classmethod
    def from_ledger(cls, ledger: FrameActionLedger) -> RFPoolDemand:
        """Capture ``D_t`` only after the complete action ledger conserves."""

        if not isinstance(ledger, FrameActionLedger):
            raise RFPoolError("RF-pool demand requires a complete FrameActionLedger")
        audit = ledger.audit_accounting_conservation()
        totals = audit.population_totals
        return cls(
            trace_id=ledger.trace_id,
            frame_index=ledger.frame_index,
            time_s=ledger.time_s,
            active_pairs=totals.active_pairs,
            reserved_rf_attempts_by_pair=ledger.reserved_rf_attempts_by_pair,
            offered_rf_attempts=totals.reserved_rf_attempts,
            rf_using_pairs=totals.rf_using_pairs,
        )

    def as_dict(self) -> dict[str, object]:
        """Return stable diagnostics without losing zero-demand VLC rows."""

        return {
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "active_pairs": self.active_pairs,
            "reserved_rf_attempts_by_pair": [
                {"pair_id": pair_id, "reserved_rf_attempts": attempts}
                for pair_id, attempts in self.reserved_rf_attempts_by_pair
            ],
            "offered_rf_attempts": self.offered_rf_attempts,
            "rf_using_pairs": self.rf_using_pairs,
        }


__all__ = ["RFPoolDemand", "RFPoolError"]
