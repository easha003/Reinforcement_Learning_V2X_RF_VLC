"""Authoritative frame-local RF assembly shared by every live consumer."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.channels.rf.model import RFPropagationResult
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
)
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.local_rf_pipeline import (
    LOCAL_RF_PIPELINE_CONTRACT_VERSION,
    LocalRFPhysicsModel,
    LocalRFPipelineError,
)
from hybrid_v2x_rl.mean_field.local_rf_response import LocalRFResponseModel
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d10-train-000"
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=1.0,
    vlc_activation_cost=1.0,
)


def _vehicle(vehicle_id: str, x_m: float, *, time_s: float) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=time_s,
        vehicle_id=vehicle_id,
        x_m=x_m,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


def _frame(
    positions: tuple[tuple[str, float], ...],
    *,
    index: int = 0,
    time_s: float = 0.0,
) -> PopulationFrame:
    vehicles: list[VehicleTraceRecord] = []
    pairs: list[PopulationPair] = []
    for pair_id, transmitter_x in positions:
        transmitter = _vehicle(f"tx-{pair_id}", transmitter_x, time_s=time_s)
        receiver = _vehicle(f"rx-{pair_id}", transmitter_x + 10.0, time_s=time_s)
        vehicles.extend((transmitter, receiver))
        pairs.append(
            PopulationPair(
                pair_id=pair_id,
                episode_step=index,
                transmitter=transmitter,
                receiver=receiver,
                lifecycle=PairLifecycle(born=index == 0),
            )
        )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="train",
            density=10.0,
            replicate=0,
        ),
        index=index,
        time_s=time_s,
        vehicles=tuple(sorted(vehicles, key=lambda row: row.vehicle_id)),
        pairs=tuple(sorted(pairs, key=lambda row: row.pair_id)),
    )


def _model() -> LocalRFPhysicsModel:
    config = load_headline_config(PROJECT_ROOT)
    rf = build_rf_channel(config, band=SensitivityBand.NOMINAL)
    return LocalRFPhysicsModel(
        response_model=LocalRFResponseModel(
            parameters=rf.collision,
            sensitivity_band=SensitivityBand.NOMINAL,
            attempt_airtime_s=float(config.rf.timing.airtime_s),
        ),
        buildings=(),
        antenna_height_m=float(config.geometry.rf_antenna_height_m),
    )


def _propagation(probability: float) -> RFPropagationResult:
    return RFPropagationResult(
        propagation_state=RFPropagationState.LOS,
        pathloss_db=80.0,
        shadowing_db=0.0,
        fading_gain_linear=1.0,
        sinr_db=20.0,
        decoding_failure_probability=probability,
    )


def _ledger(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
) -> FrameActionLedger:
    return FrameActionLedger.from_frame(frame, actions, resource_map=RESOURCE_MAP)


def _truth(frame: PopulationFrame) -> dict[str, RFPropagationResult]:
    return {pair_id: _propagation(0.1) for pair_id in frame.active_pair_ids}


def test_pipeline_assembles_one_identity_preserving_local_rf_state() -> None:
    frame = _frame((("pair-a", 0.0), ("pair-b", 40.0), ("pair-c", 80.0)))
    actions = {
        "pair-a": PolicyAction.RF_2,
        "pair-b": PolicyAction.DUP_1,
        "pair-c": PolicyAction.VLC,
    }
    model = _model()
    context = model.context_for(frame)
    ledger = _ledger(frame, actions)

    physics = model.evaluate(context, ledger, propagation_by_pair=_truth(frame))

    assert physics.contract_version == LOCAL_RF_PIPELINE_CONTRACT_VERSION
    assert physics.context is context
    assert physics.ledger is ledger
    assert physics.pair_ids == frame.active_pair_ids
    assert physics.loads.load_for("pair-a").offered_rf_attempts == 3
    assert physics.sensed_loads.row_for("pair-a").focal_rf_attempts == 2
    assert physics.responses.response_for("pair-a").load.focal_rf_attempts == 2
    assert (
        physics.endpoint_schedule.exposure_for(
            "pair-a"
        ).reservation.reserved_rf_attempts
        == 2
    )
    assert physics.attempt_risks.rf_pair_ids == ("pair-a", "pair-b")
    assert physics.attempt_risks.risk_for("pair-a").local_response is (
        physics.responses.response_for("pair-a")
    )
    assert physics.mean_local_pool_utilization <= physics.max_local_pool_utilization


def test_action_independent_context_is_reused_across_counterfactual_ledgers() -> None:
    frame = _frame((("pair-a", 0.0), ("pair-b", 40.0)))
    model = _model()
    context = model.context_for(frame)
    truth = _truth(frame)

    quiet = model.evaluate(
        context,
        _ledger(frame, {"pair-a": PolicyAction.VLC, "pair-b": PolicyAction.VLC}),
        propagation_by_pair=truth,
    )
    loaded = model.evaluate(
        context,
        _ledger(frame, {"pair-a": PolicyAction.RF_1, "pair-b": PolicyAction.RF_4}),
        propagation_by_pair=truth,
    )

    assert quiet.context is context
    assert loaded.context is context
    assert quiet.responses.response_for("pair-a").load.focal_rf_attempts == 0
    assert loaded.responses.response_for("pair-a").load.focal_rf_attempts == 1
    assert loaded.responses.response_for("pair-a").load.local_offered_rf_attempts == 5


def test_spatial_reuse_excludes_distant_rf_reservations() -> None:
    frame = _frame((("pair-a", 0.0), ("pair-b", 1_000.0)))
    model = _model()
    context = model.context_for(frame)
    physics = model.evaluate(
        context,
        _ledger(frame, {"pair-a": PolicyAction.RF_1, "pair-b": PolicyAction.RF_4}),
        propagation_by_pair=_truth(frame),
    )

    assert context.topology.domain_for("pair-a").member_pair_ids == ("pair-a",)
    assert physics.loads.load_for("pair-a").offered_rf_attempts == 1
    assert physics.loads.load_for("pair-b").offered_rf_attempts == 4
    assert physics.responses.response_for("pair-a").per_attempt_collision_probability == 0.0


def test_empty_frame_has_a_complete_empty_local_rf_state() -> None:
    frame = _frame(())
    model = _model()
    context = model.context_for(frame)

    physics = model.evaluate(
        context,
        _ledger(frame, {}),
        propagation_by_pair={},
    )

    assert physics.pair_ids == ()
    assert physics.responses.responses == ()
    assert physics.attempt_risks.risks == ()
    assert physics.mean_local_pool_utilization == 0.0
    assert physics.max_local_pool_utilization == 0.0


def test_pipeline_fails_closed_on_incomplete_truth_or_mismatched_frame() -> None:
    frame = _frame((("pair-a", 0.0),))
    later = _frame((("pair-a", 0.0),), index=1, time_s=0.1)
    model = _model()
    context = model.context_for(frame)

    with pytest.raises(LocalRFPipelineError, match="cover the population exactly"):
        model.evaluate(
            context,
            _ledger(frame, {"pair-a": PolicyAction.VLC}),
            propagation_by_pair={},
        )
    with pytest.raises(LocalRFPipelineError, match="different frames"):
        model.evaluate(
            context,
            _ledger(later, {"pair-a": PolicyAction.RF_1}),
            propagation_by_pair=_truth(frame),
        )
