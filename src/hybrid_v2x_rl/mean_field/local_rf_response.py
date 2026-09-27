"""Pair-specific collision and channel-load responses for local RF domains.

This module is an isolated migration boundary.  It consumes the validated
pair-local sensing/load partition, but the live rollout continues to use the
legacy frame-global pool until packet outcomes, half-duplex, and feedback can
move together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final

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
from hybrid_v2x_rl.mean_field.local_rf_sensing import (
    LOCAL_RF_SENSING_CONTRACT_VERSION,
    FrameLocalRFSensedLoads,
    PairLocalRFSensedLoad,
)

LOCAL_RF_RESPONSE_CONTRACT_VERSION: Final = "1.0.0"


class LocalRFResponseError(HybridV2XError):
    """Pair-local RF response fields or model inputs are inconsistent."""


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)


def _validate_declared_band(
    parameters: CollisionParameters,
    band: SensitivityBand,
) -> None:
    if not isinstance(parameters, CollisionParameters):
        raise LocalRFResponseError("local RF response requires CollisionParameters")
    if not isinstance(band, SensitivityBand):
        raise LocalRFResponseError("local RF response requires a sensitivity band")
    expected = headline_parameters(band).sensing_reliability
    if parameters.sensing_reliability != expected:
        raise LocalRFResponseError(
            "collision parameters do not match the declared sensitivity band",
            context={
                "band": band.value,
                "sensing_reliability": parameters.sensing_reliability,
                "expected": expected,
            },
        )


@dataclass(frozen=True, slots=True)
class PairLocalRFResponse:
    """Auditable load and external-collision response for one focal flow."""

    load: PairLocalRFSensedLoad
    sensitivity_band: SensitivityBand
    parameters: CollisionParameters
    offered_airtime_s: float
    pool_capacity_airtime_s: float
    pool_utilization: float
    channel_busy_ratio: float
    external_contending_attempts: int
    effective_sensed_external_attempts: float
    effective_hidden_external_attempts: float
    per_attempt_collision_probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.load, PairLocalRFSensedLoad):
            raise LocalRFResponseError(
                "pair-local RF response requires a validated sensed load"
            )
        _validate_declared_band(self.parameters, self.sensitivity_band)
        if self.parameters.rf_usage_fraction != 1.0:
            raise LocalRFResponseError(
                "action-coupled local demand must not use an RF-use multiplier"
            )
        if (
            not isinstance(self.external_contending_attempts, int)
            or isinstance(self.external_contending_attempts, bool)
            or self.external_contending_attempts < 0
        ):
            raise LocalRFResponseError(
                "external contending attempts must be a non-negative integer"
            )
        if (
            not math.isfinite(self.pool_capacity_airtime_s)
            or self.pool_capacity_airtime_s <= 0.0
        ):
            raise LocalRFResponseError(
                "local RF capacity airtime must be finite and positive"
            )
        for name in (
            "offered_airtime_s",
            "pool_utilization",
            "effective_sensed_external_attempts",
            "effective_hidden_external_attempts",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise LocalRFResponseError(
                    f"{name} must be finite and non-negative",
                    context={name: value},
                )
        for name in ("channel_busy_ratio", "per_attempt_collision_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise LocalRFResponseError(
                    f"{name} must be finite and lie in [0, 1]",
                    context={name: value},
                )

        expected_external = self.load.external_contending_rf_attempts
        geometric_sensed = self.load.geometrically_sensed_external_rf_attempts
        geometric_hidden = self.load.geometrically_hidden_external_rf_attempts
        reliability = self.parameters.sensing_reliability
        expected_sensed = geometric_sensed * reliability
        expected_hidden = geometric_hidden + geometric_sensed * (1.0 - reliability)
        helper_hidden = hidden_contenders(
            expected_external,
            self.load.sensed_fraction,
            self.parameters,
        )
        expected_capacity = (
            self.parameters.subchannels * self.parameters.generation_period_s
        )
        expected_values = {
            "offered_airtime_s": (
                self.load.local_offered_rf_attempts * self.parameters.airtime_s
            ),
            "pool_capacity_airtime_s": expected_capacity,
            "pool_utilization": resource_demand(
                self.load.local_offered_rf_attempts,
                self.parameters,
            ),
            "channel_busy_ratio": channel_busy_ratio(
                self.load.local_offered_rf_attempts,
                self.parameters,
            ),
            "effective_sensed_external_attempts": expected_sensed,
            "effective_hidden_external_attempts": expected_hidden,
            "per_attempt_collision_probability": collision_probability(
                expected_external,
                self.parameters,
                sensed_fraction=self.load.sensed_fraction,
            ),
        }
        if self.external_contending_attempts != expected_external:
            raise LocalRFResponseError(
                "collision contenders must be exactly the external attempts"
            )
        if not _close(helper_hidden, expected_hidden):
            raise LocalRFResponseError(
                "geometric and sensed-fraction hidden loads do not reconcile"
            )
        mismatches = tuple(
            name
            for name, expected in expected_values.items()
            if not _close(getattr(self, name), expected)
        )
        if mismatches:
            raise LocalRFResponseError(
                "pair-local RF response fields do not reconcile",
                context={"mismatched_fields": mismatches},
            )

    @property
    def focal_pair_id(self) -> str:
        return self.load.focal_pair_id

    @property
    def model_name(self) -> str:
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
    def has_focal_rf_reservation(self) -> bool:
        return self.load.focal_rf_attempts > 0

    @property
    def oversubscribed(self) -> bool:
        return self.pool_utilization > 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "focal_pair_id": self.focal_pair_id,
            "model_name": self.model_name,
            "sensitivity_band": self.sensitivity_band.value,
            "sensing_reliability": self.sensing_reliability,
            "sensed_fraction": self.load.sensed_fraction,
            "candidate_resources": self.candidate_resources,
            "attempt_airtime_s": self.attempt_airtime_s,
            "offered_airtime_s": self.offered_airtime_s,
            "pool_capacity_airtime_s": self.pool_capacity_airtime_s,
            "pool_utilization": self.pool_utilization,
            "channel_busy_ratio": self.channel_busy_ratio,
            "external_contending_attempts": self.external_contending_attempts,
            "effective_sensed_external_attempts": (
                self.effective_sensed_external_attempts
            ),
            "effective_hidden_external_attempts": (
                self.effective_hidden_external_attempts
            ),
            "per_attempt_collision_probability": (
                self.per_attempt_collision_probability
            ),
            "has_focal_rf_reservation": self.has_focal_rf_reservation,
            "oversubscribed": self.oversubscribed,
            "load": self.load.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class FrameLocalRFResponses:
    """Pair-aligned local RF responses for one complete population frame."""

    contract_version: str
    sensing_contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    sensitivity_band: SensitivityBand
    parameters: CollisionParameters
    pair_ids: tuple[str, ...]
    responses: tuple[PairLocalRFResponse, ...]

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_RESPONSE_CONTRACT_VERSION:
            raise LocalRFResponseError("local RF response contract version is invalid")
        if self.sensing_contract_version != LOCAL_RF_SENSING_CONTRACT_VERSION:
            raise LocalRFResponseError("local RF sensing contract version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise LocalRFResponseError("local RF response frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise LocalRFResponseError("local RF response frame time is invalid")
        _validate_declared_band(self.parameters, self.sensitivity_band)
        if self.parameters.rf_usage_fraction != 1.0:
            raise LocalRFResponseError(
                "action-coupled frame responses require unit RF usage fraction"
            )
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise LocalRFResponseError(
                "local RF response pair IDs must be unique and canonical"
            )
        if len(self.responses) != len(self.pair_ids) or tuple(
            response.focal_pair_id for response in self.responses
        ) != self.pair_ids:
            raise LocalRFResponseError(
                "local RF responses must align exactly with frame pair IDs"
            )
        if any(
            response.sensitivity_band is not self.sensitivity_band
            or response.parameters != self.parameters
            for response in self.responses
        ):
            raise LocalRFResponseError(
                "all pair-local responses must use the frame model"
            )

    def response_for(self, pair_id: str) -> PairLocalRFResponse:
        try:
            index = self.pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFResponseError(
                "pair is absent from frame-local RF responses",
                context={"pair_id": pair_id},
            ) from error
        return self.responses[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "sensing_contract_version": self.sensing_contract_version,
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "sensitivity_band": self.sensitivity_band.value,
            "active_pairs": len(self.pair_ids),
            "responses": [response.as_dict() for response in self.responses],
        }


@dataclass(frozen=True, slots=True)
class LocalRFResponseModel:
    """Apply one declared analytical sensing band to every local domain."""

    parameters: CollisionParameters
    sensitivity_band: SensitivityBand
    attempt_airtime_s: float

    def __post_init__(self) -> None:
        _validate_declared_band(self.parameters, self.sensitivity_band)
        if (
            not math.isfinite(self.attempt_airtime_s)
            or self.attempt_airtime_s <= 0.0
            or self.attempt_airtime_s > self.parameters.generation_period_s
        ):
            raise LocalRFResponseError(
                "attempt airtime must be finite, positive, and no longer than the frame"
            )

    @property
    def attempt_parameters(self) -> CollisionParameters:
        """Normalize packet-level inputs to one already-counted reservation."""

        return replace(
            self.parameters,
            airtime_s=self.attempt_airtime_s,
            rf_usage_fraction=1.0,
        )

    def evaluate(self, loads: FrameLocalRFSensedLoads) -> FrameLocalRFResponses:
        """Return one collision/CBR response for every focal pair."""

        if not isinstance(loads, FrameLocalRFSensedLoads):
            raise LocalRFResponseError(
                "local RF evaluation requires FrameLocalRFSensedLoads"
            )
        parameters = self.attempt_parameters
        capacity = parameters.subchannels * parameters.generation_period_s
        responses: list[PairLocalRFResponse] = []
        for load in loads.rows:
            external = load.external_contending_rf_attempts
            geometric_sensed = load.geometrically_sensed_external_rf_attempts
            geometric_hidden = load.geometrically_hidden_external_rf_attempts
            effective_sensed = (
                geometric_sensed * parameters.sensing_reliability
            )
            effective_hidden = geometric_hidden + geometric_sensed * (
                1.0 - parameters.sensing_reliability
            )
            responses.append(
                PairLocalRFResponse(
                    load=load,
                    sensitivity_band=self.sensitivity_band,
                    parameters=parameters,
                    offered_airtime_s=(
                        load.local_offered_rf_attempts * self.attempt_airtime_s
                    ),
                    pool_capacity_airtime_s=capacity,
                    pool_utilization=resource_demand(
                        load.local_offered_rf_attempts,
                        parameters,
                    ),
                    channel_busy_ratio=channel_busy_ratio(
                        load.local_offered_rf_attempts,
                        parameters,
                    ),
                    external_contending_attempts=external,
                    effective_sensed_external_attempts=effective_sensed,
                    effective_hidden_external_attempts=effective_hidden,
                    per_attempt_collision_probability=collision_probability(
                        external,
                        parameters,
                        sensed_fraction=load.sensed_fraction,
                    ),
                )
            )
        return FrameLocalRFResponses(
            contract_version=LOCAL_RF_RESPONSE_CONTRACT_VERSION,
            sensing_contract_version=loads.contract_version,
            trace_id=loads.trace_id,
            frame_index=loads.frame_index,
            time_s=loads.time_s,
            sensitivity_band=self.sensitivity_band,
            parameters=parameters,
            pair_ids=loads.pair_ids,
            responses=tuple(responses),
        )


__all__ = [
    "LOCAL_RF_RESPONSE_CONTRACT_VERSION",
    "FrameLocalRFResponses",
    "LocalRFResponseError",
    "LocalRFResponseModel",
    "PairLocalRFResponse",
]
