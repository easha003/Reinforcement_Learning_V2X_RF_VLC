"""Pair-specific analytical RF attempt-risk composition.

This boundary composes local collision, receiver-specific half-duplex exposure,
and deterministic propagation without sampling an outcome.  It intentionally
does not accept or inspect matched packet tapes; the later atomic rollout
migration will continue to use those existing identity-addressed draws.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from hybrid_v2x_rl.channels.rf.model import RFPropagationResult
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.endpoint_rf_schedule import (
    ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION,
    FrameEndpointRFSchedule,
    PairHalfDuplexExposure,
)
from hybrid_v2x_rl.mean_field.local_rf_response import (
    LOCAL_RF_RESPONSE_CONTRACT_VERSION,
    FrameLocalRFResponses,
    PairLocalRFResponse,
)

LOCAL_RF_RISK_CONTRACT_VERSION: Final = "1.0.0"
_TIME_TOLERANCE_S: Final = 1e-9


class LocalRFRiskError(HybridV2XError):
    """Pair-local RF risk inputs or derived probabilities are inconsistent."""


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _validate_exact_keys(
    supplied: Mapping[str, object],
    expected: tuple[str, ...],
) -> None:
    if not isinstance(supplied, Mapping):
        raise LocalRFRiskError("RF propagation results must be a pair-ID mapping")
    invalid = tuple(
        repr(pair_id)
        for pair_id in supplied
        if not isinstance(pair_id, str) or not pair_id
    )
    if invalid:
        raise LocalRFRiskError(
            "RF propagation results contain invalid pair IDs",
            context={"invalid_pair_ids": invalid},
        )
    actual = set(supplied)
    expected_set = set(expected)
    missing = tuple(pair_id for pair_id in expected if pair_id not in actual)
    unexpected = tuple(sorted(actual - expected_set))
    if missing or unexpected:
        raise LocalRFRiskError(
            "RF propagation results do not cover RF-using pairs exactly",
            context={
                "missing_pair_ids": missing,
                "unexpected_pair_ids": unexpected,
            },
        )


@dataclass(frozen=True, slots=True)
class PairLocalRFAttemptRisk:
    """Mechanism-separated failure probabilities for one selected RF attempt."""

    pair_id: str
    reserved_rf_attempts: int
    local_response: PairLocalRFResponse
    half_duplex_exposure: PairHalfDuplexExposure
    propagation: RFPropagationResult
    access_failure_probability: float
    total_failure_probability: float

    def __post_init__(self) -> None:
        if not self.pair_id:
            raise LocalRFRiskError("RF attempt-risk pair identity is empty")
        if not isinstance(self.local_response, PairLocalRFResponse):
            raise LocalRFRiskError(
                "RF attempt risk requires a pair-local collision response"
            )
        if not isinstance(self.half_duplex_exposure, PairHalfDuplexExposure):
            raise LocalRFRiskError(
                "RF attempt risk requires endpoint half-duplex exposure"
            )
        if not isinstance(self.propagation, RFPropagationResult):
            raise LocalRFRiskError(
                "RF attempt risk requires an RF propagation result"
            )
        if (
            self.local_response.focal_pair_id != self.pair_id
            or self.half_duplex_exposure.pair_id != self.pair_id
        ):
            raise LocalRFRiskError(
                "RF attempt-risk components identify different pairs"
            )
        expected_attempts = self.local_response.load.focal_rf_attempts
        exposure_attempts = (
            self.half_duplex_exposure.reservation.reserved_rf_attempts
        )
        if (
            not isinstance(self.reserved_rf_attempts, int)
            or isinstance(self.reserved_rf_attempts, bool)
            or self.reserved_rf_attempts <= 0
            or self.reserved_rf_attempts != expected_attempts
            or exposure_attempts != expected_attempts
        ):
            raise LocalRFRiskError(
                "RF attempt-risk reservation does not match its components"
            )
        receiver_schedule = self.half_duplex_exposure.receiver_schedule
        if not math.isclose(
            self.local_response.attempt_airtime_s,
            receiver_schedule.attempt_airtime_s,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            self.local_response.parameters.generation_period_s,
            receiver_schedule.generation_period_s,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise LocalRFRiskError(
                "collision and endpoint schedules use different RF timing"
            )
        decoding = self.propagation.decoding_failure_probability
        if not isinstance(
            self.propagation.propagation_state,
            RFPropagationState,
        ):
            raise LocalRFRiskError("RF propagation state is invalid")
        if any(
            not math.isfinite(value)
            for value in (
                self.propagation.pathloss_db,
                self.propagation.shadowing_db,
                self.propagation.sinr_db,
            )
        ) or (
            not math.isfinite(self.propagation.fading_gain_linear)
            or self.propagation.fading_gain_linear <= 0.0
        ):
            raise LocalRFRiskError("RF propagation diagnostics are invalid")
        if not _is_probability(decoding):
            raise LocalRFRiskError(
                "RF decoding failure probability must lie in [0, 1]"
            )
        if not _is_probability(self.access_failure_probability) or not _is_probability(
            self.total_failure_probability
        ):
            raise LocalRFRiskError(
                "composed RF failure probabilities must lie in [0, 1]"
            )
        collision = self.collision_probability
        half_duplex = self.half_duplex_probability
        expected_access = 1.0 - (1.0 - collision) * (1.0 - half_duplex)
        expected_total = 1.0 - (1.0 - expected_access) * (1.0 - decoding)
        mismatches = tuple(
            name
            for name, actual, expected in (
                (
                    "access_failure_probability",
                    self.access_failure_probability,
                    expected_access,
                ),
                (
                    "total_failure_probability",
                    self.total_failure_probability,
                    expected_total,
                ),
            )
            if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
        )
        if mismatches:
            raise LocalRFRiskError(
                "pair-local RF attempt-risk fields do not reconcile",
                context={"mismatched_fields": mismatches},
            )

    @property
    def collision_probability(self) -> float:
        return self.local_response.per_attempt_collision_probability

    @property
    def half_duplex_probability(self) -> float:
        return self.half_duplex_exposure.half_duplex_probability

    @property
    def decoding_failure_probability(self) -> float:
        return self.propagation.decoding_failure_probability

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "reserved_rf_attempts": self.reserved_rf_attempts,
            "collision_probability": self.collision_probability,
            "half_duplex_probability": self.half_duplex_probability,
            "access_failure_probability": self.access_failure_probability,
            "propagation_state": self.propagation.propagation_state.value,
            "pathloss_db": self.propagation.pathloss_db,
            "shadowing_db": self.propagation.shadowing_db,
            "fading_gain_linear": self.propagation.fading_gain_linear,
            "sinr_db": self.propagation.sinr_db,
            "decoding_failure_probability": self.decoding_failure_probability,
            "total_failure_probability": self.total_failure_probability,
            "local_response_contract_version": LOCAL_RF_RESPONSE_CONTRACT_VERSION,
            "endpoint_schedule_contract_version": (
                ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION
            ),
        }


@dataclass(frozen=True, slots=True)
class FrameLocalRFAttemptRisks:
    """Pair-aligned analytical risks for the RF-using subset of one frame."""

    contract_version: str
    local_response_contract_version: str
    endpoint_schedule_contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    pair_ids: tuple[str, ...]
    rf_pair_ids: tuple[str, ...]
    risks: tuple[PairLocalRFAttemptRisk, ...]
    local_responses: FrameLocalRFResponses
    endpoint_schedule: FrameEndpointRFSchedule

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_RISK_CONTRACT_VERSION:
            raise LocalRFRiskError("pair-local RF risk contract version is invalid")
        if self.local_response_contract_version != LOCAL_RF_RESPONSE_CONTRACT_VERSION:
            raise LocalRFRiskError("pair-local RF response contract version is invalid")
        if (
            self.endpoint_schedule_contract_version
            != ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION
        ):
            raise LocalRFRiskError("endpoint RF schedule contract version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise LocalRFRiskError("pair-local RF risk frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise LocalRFRiskError("pair-local RF risk frame time is invalid")
        if not isinstance(self.local_responses, FrameLocalRFResponses):
            raise LocalRFRiskError("frame RF risk requires local RF responses")
        if not isinstance(self.endpoint_schedule, FrameEndpointRFSchedule):
            raise LocalRFRiskError("frame RF risk requires an endpoint RF schedule")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise LocalRFRiskError("frame RF risk pair IDs are not canonical")
        if self.rf_pair_ids != tuple(sorted(self.rf_pair_ids)) or not set(
            self.rf_pair_ids
        ).issubset(self.pair_ids):
            raise LocalRFRiskError("RF-using pair IDs are not a canonical subset")
        if len(self.risks) != len(self.rf_pair_ids) or tuple(
            risk.pair_id for risk in self.risks
        ) != self.rf_pair_ids:
            raise LocalRFRiskError("RF attempt risks must align with RF-using pairs")
        if (
            self.local_responses.trace_id != self.trace_id
            or self.local_responses.frame_index != self.frame_index
            or not math.isclose(
                self.local_responses.time_s,
                self.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or self.local_responses.pair_ids != self.pair_ids
            or self.endpoint_schedule.trace_id != self.trace_id
            or self.endpoint_schedule.frame_index != self.frame_index
            or not math.isclose(
                self.endpoint_schedule.time_s,
                self.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or self.endpoint_schedule.pair_ids != self.pair_ids
        ):
            raise LocalRFRiskError(
                "risk inputs do not identify the same population frame"
            )

        expected_rf_pair_ids: list[str] = []
        for pair_id in self.pair_ids:
            response = self.local_responses.response_for(pair_id)
            exposure = self.endpoint_schedule.exposure_for(pair_id)
            response_attempts = response.load.focal_rf_attempts
            endpoint_attempts = exposure.reservation.reserved_rf_attempts
            if response_attempts != endpoint_attempts:
                raise LocalRFRiskError(
                    "local response and endpoint reservation counts differ",
                    context={"pair_id": pair_id},
                )
            if response_attempts > 0:
                expected_rf_pair_ids.append(pair_id)
        if self.rf_pair_ids != tuple(expected_rf_pair_ids):
            raise LocalRFRiskError(
                "RF-using subset does not follow selected reservations"
            )
        for risk in self.risks:
            if (
                risk.local_response
                != self.local_responses.response_for(risk.pair_id)
                or risk.half_duplex_exposure
                != self.endpoint_schedule.exposure_for(risk.pair_id)
            ):
                raise LocalRFRiskError(
                    "RF attempt risk does not use its frame components",
                    context={"pair_id": risk.pair_id},
                )

    @classmethod
    def from_components(
        cls,
        local_responses: FrameLocalRFResponses,
        endpoint_schedule: FrameEndpointRFSchedule,
        *,
        propagation_by_pair: Mapping[str, RFPropagationResult],
    ) -> FrameLocalRFAttemptRisks:
        """Compose probabilities without reading or advancing outcome tapes."""

        if not isinstance(local_responses, FrameLocalRFResponses):
            raise LocalRFRiskError(
                "RF risk composition requires FrameLocalRFResponses"
            )
        if not isinstance(endpoint_schedule, FrameEndpointRFSchedule):
            raise LocalRFRiskError(
                "RF risk composition requires FrameEndpointRFSchedule"
            )
        if (
            local_responses.trace_id != endpoint_schedule.trace_id
            or local_responses.frame_index != endpoint_schedule.frame_index
            or not math.isclose(
                local_responses.time_s,
                endpoint_schedule.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or local_responses.pair_ids != endpoint_schedule.pair_ids
        ):
            raise LocalRFRiskError(
                "local responses and endpoint schedule identify different frames"
            )
        rf_pair_ids: list[str] = []
        for pair_id in local_responses.pair_ids:
            response_attempts = local_responses.response_for(
                pair_id
            ).load.focal_rf_attempts
            endpoint_attempts = endpoint_schedule.exposure_for(
                pair_id
            ).reservation.reserved_rf_attempts
            if response_attempts != endpoint_attempts:
                raise LocalRFRiskError(
                    "local response and endpoint reservation counts differ",
                    context={"pair_id": pair_id},
                )
            if response_attempts > 0:
                rf_pair_ids.append(pair_id)
        canonical_rf_pair_ids = tuple(rf_pair_ids)
        _validate_exact_keys(propagation_by_pair, canonical_rf_pair_ids)

        risks: list[PairLocalRFAttemptRisk] = []
        for pair_id in canonical_rf_pair_ids:
            propagation = propagation_by_pair[pair_id]
            if not isinstance(propagation, RFPropagationResult):
                raise LocalRFRiskError(
                    "RF propagation mapping contains an invalid row",
                    context={"pair_id": pair_id},
                )
            response = local_responses.response_for(pair_id)
            exposure = endpoint_schedule.exposure_for(pair_id)
            collision = response.per_attempt_collision_probability
            half_duplex = exposure.half_duplex_probability
            decoding = propagation.decoding_failure_probability
            if not _is_probability(decoding):
                raise LocalRFRiskError(
                    "RF decoding failure probability must lie in [0, 1]",
                    context={"pair_id": pair_id},
                )
            access = 1.0 - (1.0 - collision) * (1.0 - half_duplex)
            total = 1.0 - (1.0 - access) * (1.0 - decoding)
            risks.append(
                PairLocalRFAttemptRisk(
                    pair_id=pair_id,
                    reserved_rf_attempts=response.load.focal_rf_attempts,
                    local_response=response,
                    half_duplex_exposure=exposure,
                    propagation=propagation,
                    access_failure_probability=access,
                    total_failure_probability=total,
                )
            )
        return cls(
            contract_version=LOCAL_RF_RISK_CONTRACT_VERSION,
            local_response_contract_version=local_responses.contract_version,
            endpoint_schedule_contract_version=endpoint_schedule.contract_version,
            trace_id=local_responses.trace_id,
            frame_index=local_responses.frame_index,
            time_s=local_responses.time_s,
            pair_ids=local_responses.pair_ids,
            rf_pair_ids=canonical_rf_pair_ids,
            risks=tuple(risks),
            local_responses=local_responses,
            endpoint_schedule=endpoint_schedule,
        )

    def risk_for(self, pair_id: str) -> PairLocalRFAttemptRisk:
        try:
            index = self.rf_pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFRiskError(
                "pair has no selected RF attempt risk",
                context={"pair_id": pair_id},
            ) from error
        return self.risks[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "local_response_contract_version": (
                self.local_response_contract_version
            ),
            "endpoint_schedule_contract_version": (
                self.endpoint_schedule_contract_version
            ),
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "active_pairs": len(self.pair_ids),
            "rf_using_pairs": len(self.rf_pair_ids),
            "rf_pair_ids": list(self.rf_pair_ids),
            "risks": [risk.as_dict() for risk in self.risks],
        }


__all__ = [
    "LOCAL_RF_RISK_CONTRACT_VERSION",
    "FrameLocalRFAttemptRisks",
    "LocalRFRiskError",
    "PairLocalRFAttemptRisk",
]
