"""One authoritative assembler for pair-local RF frame physics.

Topology and building visibility are action independent and are materialized
once per population frame.  Every realized or counterfactual action ledger is
then projected through the same local-load, sensing, collision, endpoint, and
attempt-risk contracts.  Keeping this orchestration in one module prevents the
rollout, baselines, and feasibility oracle from silently using different RF
physics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.channels.rf.model import RFPropagationResult
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.geometry import OrientedRectangle
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.endpoint_rf_schedule import FrameEndpointRFSchedule
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.mean_field.local_rf_domain import (
    LOCAL_RF_CONTENTION_RADIUS_M,
    FrameLocalRFLoads,
    FrameLocalRFTopology,
)
from hybrid_v2x_rl.mean_field.local_rf_response import (
    FrameLocalRFResponses,
    LocalRFResponseModel,
)
from hybrid_v2x_rl.mean_field.local_rf_risk import FrameLocalRFAttemptRisks
from hybrid_v2x_rl.mean_field.local_rf_sensing import (
    FrameLocalRFSensedLoads,
    FrameLocalRFSensing,
    sensing_buildings_from_config,
)

LOCAL_RF_PIPELINE_CONTRACT_VERSION: Final = "1.0.0"
_TIME_TOLERANCE_S: Final = 1e-9


class LocalRFPipelineError(HybridV2XError):
    """Frame-local RF assembly inputs do not share one physical contract."""


def _validate_exact_pair_mapping(
    supplied: Mapping[str, object],
    expected: tuple[str, ...],
) -> None:
    if not isinstance(supplied, Mapping):
        raise LocalRFPipelineError("RF propagation truth must be a pair-ID mapping")
    actual = set(supplied)
    expected_set = set(expected)
    invalid = tuple(
        repr(pair_id)
        for pair_id in supplied
        if not isinstance(pair_id, str) or not pair_id
    )
    missing = tuple(pair_id for pair_id in expected if pair_id not in actual)
    unexpected = tuple(sorted(actual - expected_set))
    if invalid or missing or unexpected:
        raise LocalRFPipelineError(
            "RF propagation truth must cover the population exactly",
            context={
                "invalid_pair_ids": invalid,
                "missing_pair_ids": missing,
                "unexpected_pair_ids": unexpected,
            },
        )


@dataclass(frozen=True, slots=True)
class FrameLocalRFContext:
    """Action-independent local topology and sensing truth for one frame."""

    contract_version: str
    frame: PopulationFrame
    topology: FrameLocalRFTopology
    sensing: FrameLocalRFSensing

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_PIPELINE_CONTRACT_VERSION:
            raise LocalRFPipelineError("local RF pipeline contract version is invalid")
        if not isinstance(self.frame, PopulationFrame):
            raise LocalRFPipelineError("local RF context requires a PopulationFrame")
        if not isinstance(self.topology, FrameLocalRFTopology):
            raise LocalRFPipelineError("local RF context requires pair-local topology")
        if not isinstance(self.sensing, FrameLocalRFSensing):
            raise LocalRFPipelineError("local RF context requires pair-local sensing")
        for source in (self.topology, self.sensing):
            if (
                source.trace_id != self.frame.trace_id
                or source.frame_index != self.frame.index
                or not math.isclose(
                    source.time_s,
                    self.frame.time_s,
                    rel_tol=0.0,
                    abs_tol=_TIME_TOLERANCE_S,
                )
                or source.pair_ids != self.frame.active_pair_ids
            ):
                raise LocalRFPipelineError(
                    "local RF context components identify different frames"
                )

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return self.frame.active_pair_ids


@dataclass(frozen=True, slots=True)
class FrameLocalRFPhysics:
    """Complete analytical RF state for one selected population action."""

    contract_version: str
    context: FrameLocalRFContext
    ledger: FrameActionLedger
    loads: FrameLocalRFLoads
    sensed_loads: FrameLocalRFSensedLoads
    responses: FrameLocalRFResponses
    endpoint_schedule: FrameEndpointRFSchedule
    attempt_risks: FrameLocalRFAttemptRisks

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_PIPELINE_CONTRACT_VERSION:
            raise LocalRFPipelineError("local RF physics contract version is invalid")
        if not isinstance(self.context, FrameLocalRFContext):
            raise LocalRFPipelineError("local RF physics requires a frame context")
        if not isinstance(self.ledger, FrameActionLedger):
            raise LocalRFPipelineError("local RF physics requires an action ledger")
        expected = (
            self.context.frame.trace_id,
            self.context.frame.index,
            self.context.frame.time_s,
            self.context.pair_ids,
        )
        sources = (
            self.ledger,
            self.loads,
            self.sensed_loads,
            self.responses,
            self.endpoint_schedule,
            self.attempt_risks,
        )
        for source in sources:
            actual = (
                source.trace_id,
                source.frame_index,
                source.time_s,
                source.pair_ids,
            )
            if (
                actual[:2] != expected[:2]
                or not math.isclose(
                    actual[2],
                    expected[2],
                    rel_tol=0.0,
                    abs_tol=_TIME_TOLERANCE_S,
                )
                or actual[3] != expected[3]
            ):
                raise LocalRFPipelineError(
                    "local RF physics components identify different frames"
                )
        if self.attempt_risks.local_responses != self.responses:
            raise LocalRFPipelineError("attempt risks do not retain the assembled responses")
        if self.attempt_risks.endpoint_schedule != self.endpoint_schedule:
            raise LocalRFPipelineError(
                "attempt risks do not retain the assembled endpoint schedule"
            )

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return self.context.pair_ids

    @property
    def mean_local_pool_utilization(self) -> float:
        if not self.responses.responses:
            return 0.0
        return math.fsum(
            row.pool_utilization for row in self.responses.responses
        ) / len(self.responses.responses)

    @property
    def max_local_pool_utilization(self) -> float:
        return max(
            (row.pool_utilization for row in self.responses.responses),
            default=0.0,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "trace_id": self.context.frame.trace_id,
            "frame_index": self.context.frame.index,
            "time_s": self.context.frame.time_s,
            "active_pairs": len(self.pair_ids),
            "total_reserved_rf_attempts": self.ledger.total_reserved_rf_attempts,
            "mean_local_pool_utilization": self.mean_local_pool_utilization,
            "max_local_pool_utilization": self.max_local_pool_utilization,
            "responses": self.responses,
            "endpoint_schedule": self.endpoint_schedule,
            "attempt_risks": self.attempt_risks,
        }


@dataclass(frozen=True, slots=True)
class LocalRFPhysicsModel:
    """Configuration-bound evaluator shared by rollout and counterfactuals."""

    response_model: LocalRFResponseModel
    buildings: tuple[OrientedRectangle, ...]
    antenna_height_m: float
    contention_radius_m: float = LOCAL_RF_CONTENTION_RADIUS_M

    def __post_init__(self) -> None:
        if not isinstance(self.response_model, LocalRFResponseModel):
            raise LocalRFPipelineError("local RF physics requires a response model")
        if any(not isinstance(building, OrientedRectangle) for building in self.buildings):
            raise LocalRFPipelineError("local RF buildings must be rectangles")
        if not math.isfinite(self.antenna_height_m) or self.antenna_height_m <= 0.0:
            raise LocalRFPipelineError("local RF antenna height must be positive")
        if (
            not math.isfinite(self.contention_radius_m)
            or self.contention_radius_m <= 0.0
        ):
            raise LocalRFPipelineError("local RF contention radius must be positive")

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        *,
        sensitivity_band: SensitivityBand = SensitivityBand.NOMINAL,
    ) -> LocalRFPhysicsModel:
        if not isinstance(config, ProjectConfig):
            raise LocalRFPipelineError("local RF physics requires ProjectConfig")
        rf = build_rf_channel(config, band=sensitivity_band)
        return cls(
            response_model=LocalRFResponseModel(
                parameters=rf.collision,
                sensitivity_band=sensitivity_band,
                attempt_airtime_s=float(config.rf.timing.airtime_s),
            ),
            buildings=sensing_buildings_from_config(config),
            antenna_height_m=float(config.geometry.rf_antenna_height_m),
        )

    @property
    def attempt_airtime_s(self) -> float:
        return self.response_model.attempt_airtime_s

    @property
    def generation_period_s(self) -> float:
        return self.response_model.parameters.generation_period_s

    def context_for(self, frame: PopulationFrame) -> FrameLocalRFContext:
        topology = FrameLocalRFTopology.from_frame(
            frame,
            radius_m=self.contention_radius_m,
        )
        sensing = FrameLocalRFSensing.from_frame_and_topology(
            frame,
            topology,
            buildings=self.buildings,
            antenna_height_m=self.antenna_height_m,
        )
        return FrameLocalRFContext(
            contract_version=LOCAL_RF_PIPELINE_CONTRACT_VERSION,
            frame=frame,
            topology=topology,
            sensing=sensing,
        )

    def evaluate(
        self,
        context: FrameLocalRFContext,
        ledger: FrameActionLedger,
        *,
        propagation_by_pair: Mapping[str, RFPropagationResult],
    ) -> FrameLocalRFPhysics:
        if not isinstance(context, FrameLocalRFContext):
            raise LocalRFPipelineError("local RF evaluation requires a frame context")
        if not isinstance(ledger, FrameActionLedger):
            raise LocalRFPipelineError("local RF evaluation requires an action ledger")
        if (
            ledger.trace_id != context.frame.trace_id
            or ledger.frame_index != context.frame.index
            or not math.isclose(
                ledger.time_s,
                context.frame.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or ledger.pair_ids != context.pair_ids
        ):
            raise LocalRFPipelineError(
                "local RF context and ledger identify different frames"
            )
        _validate_exact_pair_mapping(propagation_by_pair, context.pair_ids)
        invalid = tuple(
            pair_id
            for pair_id in context.pair_ids
            if not isinstance(propagation_by_pair[pair_id], RFPropagationResult)
        )
        if invalid:
            raise LocalRFPipelineError(
                "RF propagation truth contains invalid rows",
                context={"pair_ids": invalid},
            )

        loads = FrameLocalRFLoads.from_topology_and_ledger(
            context.topology,
            ledger,
        )
        sensed_loads = FrameLocalRFSensedLoads.from_sensing_and_loads(
            context.sensing,
            loads,
        )
        responses = self.response_model.evaluate(sensed_loads)
        endpoint_schedule = FrameEndpointRFSchedule.from_frame_and_ledger(
            context.frame,
            ledger,
            attempt_airtime_s=self.attempt_airtime_s,
            generation_period_s=self.generation_period_s,
        )
        rf_pair_ids = tuple(
            row.pair_id for row in ledger.pair_accounting if row.uses_rf
        )
        attempt_risks = FrameLocalRFAttemptRisks.from_components(
            responses,
            endpoint_schedule,
            propagation_by_pair={
                pair_id: propagation_by_pair[pair_id]
                for pair_id in rf_pair_ids
            },
        )
        return FrameLocalRFPhysics(
            contract_version=LOCAL_RF_PIPELINE_CONTRACT_VERSION,
            context=context,
            ledger=ledger,
            loads=loads,
            sensed_loads=sensed_loads,
            responses=responses,
            endpoint_schedule=endpoint_schedule,
            attempt_risks=attempt_risks,
        )


__all__ = [
    "LOCAL_RF_PIPELINE_CONTRACT_VERSION",
    "FrameLocalRFContext",
    "FrameLocalRFPhysics",
    "LocalRFPhysicsModel",
    "LocalRFPipelineError",
]
