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
from dataclasses import dataclass, replace

from hybrid_v2x_rl.channels.rf.collision import (
    MODEL_NAME,
    CollisionParameters,
    SensitivityBand,
    channel_busy_ratio,
    collision_probability,
    headline_parameters,
    hidden_contenders,
    resource_demand,
)
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


@dataclass(frozen=True, slots=True)
class RFPoolResponse:
    """Auditable load and per-attempt contention response for one frame."""

    demand: RFPoolDemand
    sensitivity_band: SensitivityBand
    sensed_fraction: float
    parameters: CollisionParameters
    offered_airtime_s: float
    pool_capacity_airtime_s: float
    pool_utilization: float
    channel_busy_ratio: float
    contending_attempts: int
    hidden_contenders: float
    per_attempt_collision_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.demand, RFPoolDemand):
            raise RFPoolError("RF-pool response requires validated demand")
        if not isinstance(self.sensitivity_band, SensitivityBand):
            raise RFPoolError("RF-pool response requires a sensitivity band")
        if not isinstance(self.parameters, CollisionParameters):
            raise RFPoolError("RF-pool response requires collision parameters")
        if self.parameters.rf_usage_fraction != 1.0:
            raise RFPoolError(
                "action-coupled demand must not be scaled by an RF-use fraction"
            )
        if (
            not isinstance(self.contending_attempts, int)
            or isinstance(self.contending_attempts, bool)
            or self.contending_attempts < 0
        ):
            raise RFPoolError("contending_attempts must be a non-negative integer")

        if not math.isfinite(self.sensed_fraction) or not 0.0 <= self.sensed_fraction <= 1.0:
            raise RFPoolError(
                "sensed_fraction must be finite and lie in [0, 1]",
                context={"sensed_fraction": self.sensed_fraction},
            )
        if (
            not math.isfinite(self.pool_capacity_airtime_s)
            or self.pool_capacity_airtime_s <= 0.0
        ):
            raise RFPoolError(
                "pool_capacity_airtime_s must be finite and positive",
                context={"pool_capacity_airtime_s": self.pool_capacity_airtime_s},
            )
        for name in (
            "offered_airtime_s",
            "pool_utilization",
            "hidden_contenders",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise RFPoolError(
                    f"{name} must be finite and non-negative",
                    context={name: value},
                )
        for name in ("channel_busy_ratio", "per_attempt_collision_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RFPoolError(
                    f"{name} must be finite and lie in [0, 1]",
                    context={name: value},
                )

        expected_contenders = max(0, self.demand.offered_rf_attempts - 1)
        if self.contending_attempts != expected_contenders:
            raise RFPoolError(
                "a focal RF attempt must contend with every other offered attempt",
                context={
                    "contending_attempts": self.contending_attempts,
                    "expected": expected_contenders,
                },
            )
        expected_offered_airtime = self.demand.offered_rf_attempts * self.attempt_airtime_s
        expected_capacity = (
            self.parameters.subchannels * self.parameters.generation_period_s
        )
        expected_values = {
            "offered_airtime_s": expected_offered_airtime,
            "pool_capacity_airtime_s": expected_capacity,
            "pool_utilization": resource_demand(
                self.demand.offered_rf_attempts,
                self.parameters,
            ),
            "channel_busy_ratio": channel_busy_ratio(
                self.demand.offered_rf_attempts,
                self.parameters,
            ),
            "hidden_contenders": hidden_contenders(
                self.contending_attempts,
                self.sensed_fraction,
                self.parameters,
            ),
            "per_attempt_collision_probability": collision_probability(
                self.contending_attempts,
                self.parameters,
                sensed_fraction=self.sensed_fraction,
            ),
        }
        mismatches = tuple(
            name
            for name, expected in expected_values.items()
            if not math.isclose(
                getattr(self, name),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        if mismatches:
            raise RFPoolError(
                "RF-pool response fields do not reconcile",
                context={"mismatched_fields": mismatches},
            )

    @property
    def model_name(self) -> str:
        """Preserve the validated model's deliberately limited research claim."""

        return MODEL_NAME

    @property
    def sensing_reliability(self) -> float:
        return self.parameters.sensing_reliability

    @property
    def candidate_resources(self) -> int:
        return self.parameters.candidate_resources

    @property
    def attempt_airtime_s(self) -> float:
        return self.parameters.airtime_s

    @property
    def oversubscribed(self) -> bool:
        return self.pool_utilization > 1.0

    def as_dict(self) -> dict[str, object]:
        """Return stable diagnostics for rollout logs and validation artifacts."""

        return {
            "model_name": self.model_name,
            "sensitivity_band": self.sensitivity_band.value,
            "sensed_fraction": self.sensed_fraction,
            "sensing_reliability": self.sensing_reliability,
            "candidate_resources": self.candidate_resources,
            "attempt_airtime_s": self.attempt_airtime_s,
            "offered_airtime_s": self.offered_airtime_s,
            "pool_capacity_airtime_s": self.pool_capacity_airtime_s,
            "pool_utilization": self.pool_utilization,
            "channel_busy_ratio": self.channel_busy_ratio,
            "contending_attempts": self.contending_attempts,
            "hidden_contenders": self.hidden_contenders,
            "per_attempt_collision_probability": (
                self.per_attempt_collision_probability
            ),
            "oversubscribed": self.oversubscribed,
            "demand": self.demand.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RFPoolModel:
    """Map action-coupled attempts through the validated analytical model.

    ``parameters`` supplies the frozen resource-pool shape and one declared
    sensing-reliability band. ``attempt_airtime_s`` is intentionally separate:
    the legacy channel parameters describe airtime committed by one packet,
    whereas ``D_t`` already counts each reserved attempt. Replacing that
    airtime with the per-attempt value prevents retransmissions from being
    counted twice. Likewise, the adapter fixes ``rf_usage_fraction`` to one
    because VLC offload has already removed attempts from ``D_t``.
    """

    parameters: CollisionParameters
    sensitivity_band: SensitivityBand
    attempt_airtime_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, CollisionParameters):
            raise RFPoolError("RF-pool model requires CollisionParameters")
        if not isinstance(self.sensitivity_band, SensitivityBand):
            raise RFPoolError("RF-pool model requires a sensitivity band")
        if (
            not math.isfinite(self.attempt_airtime_s)
            or self.attempt_airtime_s <= 0.0
            or self.attempt_airtime_s > self.parameters.generation_period_s
        ):
            raise RFPoolError(
                "attempt_airtime_s must be finite, positive, and no longer than the frame",
                context={"attempt_airtime_s": self.attempt_airtime_s},
            )
        expected_reliability = headline_parameters(
            self.sensitivity_band
        ).sensing_reliability
        if self.parameters.sensing_reliability != expected_reliability:
            raise RFPoolError(
                "collision parameters do not match the declared sensitivity band",
                context={
                    "band": self.sensitivity_band.value,
                    "sensing_reliability": self.parameters.sensing_reliability,
                    "expected": expected_reliability,
                },
            )

    @property
    def attempt_parameters(self) -> CollisionParameters:
        """Normalize legacy packet parameters to one already-counted attempt."""

        return replace(
            self.parameters,
            airtime_s=self.attempt_airtime_s,
            rf_usage_fraction=1.0,
        )

    def evaluate(
        self,
        demand: RFPoolDemand,
        *,
        sensed_fraction: float = 1.0,
    ) -> RFPoolResponse:
        """Return unclipped load, clipped CBR, and focal-attempt collision risk."""

        if not isinstance(demand, RFPoolDemand):
            raise RFPoolError("RF-pool evaluation requires validated demand")
        parameters = self.attempt_parameters
        contenders = max(0, demand.offered_rf_attempts - 1)
        utilization = resource_demand(demand.offered_rf_attempts, parameters)
        cbr = channel_busy_ratio(demand.offered_rf_attempts, parameters)
        hidden = hidden_contenders(contenders, sensed_fraction, parameters)
        collision = collision_probability(
            contenders,
            parameters,
            sensed_fraction=sensed_fraction,
        )
        capacity_airtime = (
            parameters.subchannels * parameters.generation_period_s
        )
        return RFPoolResponse(
            demand=demand,
            sensitivity_band=self.sensitivity_band,
            sensed_fraction=sensed_fraction,
            parameters=parameters,
            offered_airtime_s=demand.offered_rf_attempts * self.attempt_airtime_s,
            pool_capacity_airtime_s=capacity_airtime,
            pool_utilization=utilization,
            channel_busy_ratio=cbr,
            contending_attempts=contenders,
            hidden_contenders=hidden,
            per_attempt_collision_probability=collision,
        )


__all__ = [
    "RFPoolDemand",
    "RFPoolError",
    "RFPoolModel",
    "RFPoolResponse",
]
