"""Endpoint-specific RF activity and half-duplex exposure boundary.

Service flows remain separate accounting identities, but a physical vehicle
has one half-duplex radio.  This module assigns every selected RF reservation
to its physical transmitter exactly once and derives each focal receiver's
transmit duty cycle from that endpoint, replacing the legacy population mean.

The schedule is an offered-airtime schedule over one packet-generation period,
not a realized Mode-2 slot allocation.  Collision slot selection and endpoint
activity phase remain statistical in the declared analytical model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import MAX_RESERVED_RF_ATTEMPTS
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.frames import PopulationFrame

ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION: Final = "1.0.0"
_TIME_TOLERANCE_S: Final = 1e-9


class EndpointRFScheduleError(HybridV2XError):
    """Endpoint activity or half-duplex exposure violates the contract."""


def _valid_timing(attempt_airtime_s: float, generation_period_s: float) -> bool:
    return (
        math.isfinite(attempt_airtime_s)
        and attempt_airtime_s > 0.0
        and math.isfinite(generation_period_s)
        and generation_period_s > 0.0
        and attempt_airtime_s <= generation_period_s
    )


@dataclass(frozen=True, slots=True)
class PairRFEndpointReservation:
    """One service flow's RF reservation bound to physical endpoints."""

    pair_id: str
    transmitter_id: str
    receiver_id: str
    reserved_rf_attempts: int

    def __post_init__(self) -> None:
        if not self.pair_id or not self.transmitter_id or not self.receiver_id:
            raise EndpointRFScheduleError(
                "pair and endpoint identities must be nonempty"
            )
        if self.transmitter_id == self.receiver_id:
            raise EndpointRFScheduleError(
                "a service flow cannot transmit to the same physical endpoint"
            )
        if (
            not isinstance(self.reserved_rf_attempts, int)
            or isinstance(self.reserved_rf_attempts, bool)
            or not 0 <= self.reserved_rf_attempts <= MAX_RESERVED_RF_ATTEMPTS
        ):
            raise EndpointRFScheduleError(
                "reserved RF attempts must follow the action contract"
            )

    @property
    def uses_rf(self) -> bool:
        return self.reserved_rf_attempts > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "transmitter_id": self.transmitter_id,
            "receiver_id": self.receiver_id,
            "reserved_rf_attempts": self.reserved_rf_attempts,
        }


@dataclass(frozen=True, slots=True)
class EndpointRFSerializedAttempt:
    """One logically serialized attempt on a physical transmitter."""

    endpoint_id: str
    pair_id: str
    pair_attempt_index: int
    endpoint_sequence_index: int

    def __post_init__(self) -> None:
        if not self.endpoint_id or not self.pair_id:
            raise EndpointRFScheduleError(
                "serialized attempt identities must be nonempty"
            )
        if (
            not isinstance(self.pair_attempt_index, int)
            or isinstance(self.pair_attempt_index, bool)
            or not 0 <= self.pair_attempt_index < MAX_RESERVED_RF_ATTEMPTS
        ):
            raise EndpointRFScheduleError(
                "pair attempt index is outside the action contract"
            )
        if (
            not isinstance(self.endpoint_sequence_index, int)
            or isinstance(self.endpoint_sequence_index, bool)
            or self.endpoint_sequence_index < 0
        ):
            raise EndpointRFScheduleError(
                "endpoint sequence index must be non-negative"
            )

    def as_dict(self, *, attempt_airtime_s: float) -> dict[str, object]:
        start = self.endpoint_sequence_index * attempt_airtime_s
        return {
            "endpoint_id": self.endpoint_id,
            "pair_id": self.pair_id,
            "pair_attempt_index": self.pair_attempt_index,
            "endpoint_sequence_index": self.endpoint_sequence_index,
            "serialization_start_offset_s": start,
            "serialization_end_offset_s": start + attempt_airtime_s,
        }


@dataclass(frozen=True, slots=True)
class EndpointRFTransmitSchedule:
    """All service reservations offered by one physical transmitter."""

    endpoint_id: str
    attempt_airtime_s: float
    generation_period_s: float
    reservations: tuple[PairRFEndpointReservation, ...]
    serialized_attempts: tuple[EndpointRFSerializedAttempt, ...]
    offered_rf_attempts: int
    offered_airtime_s: float
    endpoint_utilization: float
    transmit_duty_cycle: float

    def __post_init__(self) -> None:
        if not self.endpoint_id:
            raise EndpointRFScheduleError("schedule endpoint identity is empty")
        if not _valid_timing(self.attempt_airtime_s, self.generation_period_s):
            raise EndpointRFScheduleError(
                "endpoint RF timing must be finite, positive, and frame-bounded"
            )
        if any(
            not isinstance(reservation, PairRFEndpointReservation)
            for reservation in self.reservations
        ):
            raise EndpointRFScheduleError(
                "endpoint schedule requires pair RF reservations"
            )
        pair_ids = self.transmitting_pair_ids
        if pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(
            set(pair_ids)
        ):
            raise EndpointRFScheduleError(
                "endpoint reservations must be unique and canonical"
            )
        if any(
            reservation.transmitter_id != self.endpoint_id
            for reservation in self.reservations
        ):
            raise EndpointRFScheduleError(
                "endpoint schedule contains another transmitter's reservation"
            )
        if any(
            not isinstance(attempt, EndpointRFSerializedAttempt)
            for attempt in self.serialized_attempts
        ):
            raise EndpointRFScheduleError(
                "endpoint schedule requires serialized attempt rows"
            )
        expected_attempt_keys = tuple(
            (reservation.pair_id, pair_attempt_index)
            for reservation in self.reservations
            for pair_attempt_index in range(reservation.reserved_rf_attempts)
        )
        actual_attempt_keys = tuple(
            (attempt.pair_id, attempt.pair_attempt_index)
            for attempt in self.serialized_attempts
        )
        if (
            actual_attempt_keys != expected_attempt_keys
            or any(
                attempt.endpoint_id != self.endpoint_id
                or attempt.endpoint_sequence_index != sequence_index
                for sequence_index, attempt in enumerate(self.serialized_attempts)
            )
        ):
            raise EndpointRFScheduleError(
                "endpoint attempts must serialize every reservation exactly once"
            )
        expected_attempts = sum(
            reservation.reserved_rf_attempts
            for reservation in self.reservations
        )
        expected_airtime = expected_attempts * self.attempt_airtime_s
        expected_utilization = expected_airtime / self.generation_period_s
        expected_duty = min(1.0, expected_utilization)
        if (
            not isinstance(self.offered_rf_attempts, int)
            or isinstance(self.offered_rf_attempts, bool)
            or self.offered_rf_attempts != expected_attempts
            or len(self.serialized_attempts) != expected_attempts
        ):
            raise EndpointRFScheduleError(
                "endpoint RF attempt total does not conserve reservations"
            )
        expected_values = {
            "offered_airtime_s": expected_airtime,
            "endpoint_utilization": expected_utilization,
            "transmit_duty_cycle": expected_duty,
        }
        mismatches = tuple(
            name
            for name, expected in expected_values.items()
            if not math.isfinite(getattr(self, name))
            or not math.isclose(
                getattr(self, name),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        )
        if mismatches:
            raise EndpointRFScheduleError(
                "endpoint RF schedule fields do not reconcile",
                context={"mismatched_fields": mismatches},
            )

    @property
    def transmitting_pair_ids(self) -> tuple[str, ...]:
        return tuple(reservation.pair_id for reservation in self.reservations)

    @property
    def rf_using_pair_ids(self) -> tuple[str, ...]:
        return tuple(
            reservation.pair_id
            for reservation in self.reservations
            if reservation.uses_rf
        )

    @property
    def oversubscribed(self) -> bool:
        return self.endpoint_utilization > 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint_id": self.endpoint_id,
            "attempt_airtime_s": self.attempt_airtime_s,
            "generation_period_s": self.generation_period_s,
            "offered_rf_attempts": self.offered_rf_attempts,
            "offered_airtime_s": self.offered_airtime_s,
            "endpoint_utilization": self.endpoint_utilization,
            "transmit_duty_cycle": self.transmit_duty_cycle,
            "oversubscribed": self.oversubscribed,
            "reservations": [row.as_dict() for row in self.reservations],
            "serialized_attempts": [
                attempt.as_dict(attempt_airtime_s=self.attempt_airtime_s)
                for attempt in self.serialized_attempts
            ],
        }


@dataclass(frozen=True, slots=True)
class PairHalfDuplexExposure:
    """One focal receiver's exposure to its own endpoint's RF activity."""

    reservation: PairRFEndpointReservation
    receiver_schedule: EndpointRFTransmitSchedule
    half_duplex_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.reservation, PairRFEndpointReservation):
            raise EndpointRFScheduleError(
                "half-duplex exposure requires a pair RF reservation"
            )
        if not isinstance(self.receiver_schedule, EndpointRFTransmitSchedule):
            raise EndpointRFScheduleError(
                "half-duplex exposure requires a receiver schedule"
            )
        if self.reservation.receiver_id != self.receiver_schedule.endpoint_id:
            raise EndpointRFScheduleError(
                "half-duplex exposure is bound to the wrong receiver endpoint"
            )
        if (
            not math.isfinite(self.half_duplex_probability)
            or not 0.0 <= self.half_duplex_probability <= 1.0
            or not math.isclose(
                self.half_duplex_probability,
                self.receiver_schedule.transmit_duty_cycle,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            raise EndpointRFScheduleError(
                "half-duplex probability must equal receiver transmit duty cycle"
            )

    @property
    def pair_id(self) -> str:
        return self.reservation.pair_id

    @property
    def transmitter_id(self) -> str:
        return self.reservation.transmitter_id

    @property
    def receiver_id(self) -> str:
        return self.reservation.receiver_id

    @property
    def receiver_reserved_rf_attempts(self) -> int:
        return self.receiver_schedule.offered_rf_attempts

    @property
    def receiver_transmitting_pair_ids(self) -> tuple[str, ...]:
        return self.receiver_schedule.rf_using_pair_ids

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "transmitter_id": self.transmitter_id,
            "receiver_id": self.receiver_id,
            "focal_reserved_rf_attempts": self.reservation.reserved_rf_attempts,
            "receiver_reserved_rf_attempts": (
                self.receiver_reserved_rf_attempts
            ),
            "receiver_transmitting_pair_ids": list(
                self.receiver_transmitting_pair_ids
            ),
            "receiver_offered_airtime_s": (
                self.receiver_schedule.offered_airtime_s
            ),
            "receiver_endpoint_utilization": (
                self.receiver_schedule.endpoint_utilization
            ),
            "receiver_schedule_oversubscribed": (
                self.receiver_schedule.oversubscribed
            ),
            "half_duplex_probability": self.half_duplex_probability,
        }


@dataclass(frozen=True, slots=True)
class FrameEndpointRFSchedule:
    """Conserved RF activity and half-duplex exposure for one frame."""

    contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    attempt_airtime_s: float
    generation_period_s: float
    pair_ids: tuple[str, ...]
    reservations: tuple[PairRFEndpointReservation, ...]
    endpoint_schedules: tuple[EndpointRFTransmitSchedule, ...]
    exposures: tuple[PairHalfDuplexExposure, ...]
    total_reserved_rf_attempts: int

    def __post_init__(self) -> None:
        if self.contract_version != ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION:
            raise EndpointRFScheduleError("endpoint RF schedule version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise EndpointRFScheduleError("endpoint RF frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise EndpointRFScheduleError("endpoint RF frame time is invalid")
        if not _valid_timing(self.attempt_airtime_s, self.generation_period_s):
            raise EndpointRFScheduleError("endpoint RF frame timing is invalid")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise EndpointRFScheduleError(
                "endpoint RF pair IDs must be unique and canonical"
            )
        if len(self.reservations) != len(self.pair_ids) or tuple(
            reservation.pair_id for reservation in self.reservations
        ) != self.pair_ids:
            raise EndpointRFScheduleError(
                "endpoint RF reservations must align with frame pair IDs"
            )
        endpoint_ids = self.endpoint_ids
        if endpoint_ids != tuple(sorted(endpoint_ids)) or len(endpoint_ids) != len(
            set(endpoint_ids)
        ):
            raise EndpointRFScheduleError(
                "physical endpoint schedules must be unique and canonical"
            )
        expected_endpoint_ids = tuple(
            sorted(
                {
                    endpoint_id
                    for reservation in self.reservations
                    for endpoint_id in (
                        reservation.transmitter_id,
                        reservation.receiver_id,
                    )
                }
            )
        )
        if endpoint_ids != expected_endpoint_ids:
            raise EndpointRFScheduleError(
                "endpoint schedules must cover every active physical endpoint"
            )
        if any(
            schedule.attempt_airtime_s != self.attempt_airtime_s
            or schedule.generation_period_s != self.generation_period_s
            for schedule in self.endpoint_schedules
        ):
            raise EndpointRFScheduleError(
                "endpoint timing must match the frame schedule"
            )
        if len(self.exposures) != len(self.pair_ids) or tuple(
            exposure.pair_id for exposure in self.exposures
        ) != self.pair_ids:
            raise EndpointRFScheduleError(
                "half-duplex exposures must align with frame pair IDs"
            )

        scheduled_reservations = tuple(
            sorted(
                (
                    reservation
                    for schedule in self.endpoint_schedules
                    for reservation in schedule.reservations
                ),
                key=lambda reservation: reservation.pair_id,
            )
        )
        if scheduled_reservations != self.reservations:
            raise EndpointRFScheduleError(
                "each service reservation must appear in one transmitter schedule"
            )
        schedule_by_endpoint = {
            schedule.endpoint_id: schedule for schedule in self.endpoint_schedules
        }
        if any(
            exposure.reservation != reservation
            or exposure.receiver_schedule
            != schedule_by_endpoint[reservation.receiver_id]
            for exposure, reservation in zip(
                self.exposures,
                self.reservations,
                strict=True,
            )
        ):
            raise EndpointRFScheduleError(
                "half-duplex exposures must use the focal receiver schedule"
            )
        expected_total = sum(
            reservation.reserved_rf_attempts
            for reservation in self.reservations
        )
        endpoint_total = sum(
            schedule.offered_rf_attempts for schedule in self.endpoint_schedules
        )
        if (
            not isinstance(self.total_reserved_rf_attempts, int)
            or isinstance(self.total_reserved_rf_attempts, bool)
            or self.total_reserved_rf_attempts != expected_total
            or endpoint_total != expected_total
        ):
            raise EndpointRFScheduleError(
                "frame and endpoint RF attempt totals do not conserve"
            )

    @classmethod
    def from_frame_and_ledger(
        cls,
        frame: PopulationFrame,
        ledger: FrameActionLedger,
        *,
        attempt_airtime_s: float,
        generation_period_s: float,
    ) -> FrameEndpointRFSchedule:
        """Bind each selected reservation to one physical transmitter."""

        if not isinstance(frame, PopulationFrame):
            raise EndpointRFScheduleError(
                "endpoint RF scheduling requires PopulationFrame"
            )
        if not isinstance(ledger, FrameActionLedger):
            raise EndpointRFScheduleError(
                "endpoint RF scheduling requires FrameActionLedger"
            )
        if not _valid_timing(attempt_airtime_s, generation_period_s):
            raise EndpointRFScheduleError(
                "endpoint RF timing must be finite, positive, and frame-bounded"
            )
        if (
            frame.trace_id != ledger.trace_id
            or frame.index != ledger.frame_index
            or not math.isclose(
                frame.time_s,
                ledger.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or frame.active_pair_ids != ledger.pair_ids
        ):
            raise EndpointRFScheduleError(
                "population frame and action ledger do not identify the same frame"
            )
        ledger.audit_accounting_conservation()
        attempts_by_pair = dict(ledger.reserved_rf_attempts_by_pair)
        reservations = tuple(
            PairRFEndpointReservation(
                pair_id=pair.pair_id,
                transmitter_id=pair.transmitter.vehicle_id,
                receiver_id=pair.receiver.vehicle_id,
                reserved_rf_attempts=attempts_by_pair[pair.pair_id],
            )
            for pair in frame.pairs
        )
        endpoint_ids = tuple(
            sorted(
                {
                    endpoint_id
                    for reservation in reservations
                    for endpoint_id in (
                        reservation.transmitter_id,
                        reservation.receiver_id,
                    )
                }
            )
        )
        endpoint_schedules: list[EndpointRFTransmitSchedule] = []
        for endpoint_id in endpoint_ids:
            rows = tuple(
                reservation
                for reservation in reservations
                if reservation.transmitter_id == endpoint_id
            )
            offered_attempts = sum(
                reservation.reserved_rf_attempts for reservation in rows
            )
            offered_airtime = offered_attempts * attempt_airtime_s
            utilization = offered_airtime / generation_period_s
            attempt_keys = tuple(
                (reservation, pair_attempt_index)
                for reservation in rows
                for pair_attempt_index in range(
                    reservation.reserved_rf_attempts
                )
            )
            serialized_attempts = tuple(
                EndpointRFSerializedAttempt(
                    endpoint_id=endpoint_id,
                    pair_id=reservation.pair_id,
                    pair_attempt_index=pair_attempt_index,
                    endpoint_sequence_index=endpoint_sequence_index,
                )
                for endpoint_sequence_index, (
                    reservation,
                    pair_attempt_index,
                ) in enumerate(
                    attempt_keys
                )
            )
            endpoint_schedules.append(
                EndpointRFTransmitSchedule(
                    endpoint_id=endpoint_id,
                    attempt_airtime_s=attempt_airtime_s,
                    generation_period_s=generation_period_s,
                    reservations=rows,
                    serialized_attempts=serialized_attempts,
                    offered_rf_attempts=offered_attempts,
                    offered_airtime_s=offered_airtime,
                    endpoint_utilization=utilization,
                    transmit_duty_cycle=min(1.0, utilization),
                )
            )
        schedule_by_endpoint = {
            schedule.endpoint_id: schedule for schedule in endpoint_schedules
        }
        exposures = tuple(
            PairHalfDuplexExposure(
                reservation=reservation,
                receiver_schedule=schedule_by_endpoint[reservation.receiver_id],
                half_duplex_probability=(
                    schedule_by_endpoint[
                        reservation.receiver_id
                    ].transmit_duty_cycle
                ),
            )
            for reservation in reservations
        )
        return cls(
            contract_version=ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION,
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            attempt_airtime_s=attempt_airtime_s,
            generation_period_s=generation_period_s,
            pair_ids=frame.active_pair_ids,
            reservations=reservations,
            endpoint_schedules=tuple(endpoint_schedules),
            exposures=exposures,
            total_reserved_rf_attempts=ledger.total_reserved_rf_attempts,
        )

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        return tuple(schedule.endpoint_id for schedule in self.endpoint_schedules)

    @property
    def endpoint_offered_rf_attempts(self) -> int:
        return sum(
            schedule.offered_rf_attempts for schedule in self.endpoint_schedules
        )

    @property
    def oversubscribed_endpoint_ids(self) -> tuple[str, ...]:
        return tuple(
            schedule.endpoint_id
            for schedule in self.endpoint_schedules
            if schedule.oversubscribed
        )

    def schedule_for(self, endpoint_id: str) -> EndpointRFTransmitSchedule:
        try:
            index = self.endpoint_ids.index(endpoint_id)
        except ValueError as error:
            raise EndpointRFScheduleError(
                "physical endpoint is absent from the frame schedule",
                context={"endpoint_id": endpoint_id},
            ) from error
        return self.endpoint_schedules[index]

    def exposure_for(self, pair_id: str) -> PairHalfDuplexExposure:
        try:
            index = self.pair_ids.index(pair_id)
        except ValueError as error:
            raise EndpointRFScheduleError(
                "pair is absent from the half-duplex exposures",
                context={"pair_id": pair_id},
            ) from error
        return self.exposures[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "attempt_airtime_s": self.attempt_airtime_s,
            "generation_period_s": self.generation_period_s,
            "active_pairs": len(self.pair_ids),
            "physical_endpoints": len(self.endpoint_ids),
            "total_reserved_rf_attempts": self.total_reserved_rf_attempts,
            "endpoint_offered_rf_attempts": self.endpoint_offered_rf_attempts,
            "oversubscribed_endpoint_ids": list(
                self.oversubscribed_endpoint_ids
            ),
            "reservations": [row.as_dict() for row in self.reservations],
            "endpoint_schedules": [
                schedule.as_dict() for schedule in self.endpoint_schedules
            ],
            "exposures": [exposure.as_dict() for exposure in self.exposures],
        }


__all__ = [
    "ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION",
    "EndpointRFScheduleError",
    "EndpointRFSerializedAttempt",
    "EndpointRFTransmitSchedule",
    "FrameEndpointRFSchedule",
    "PairHalfDuplexExposure",
    "PairRFEndpointReservation",
]
