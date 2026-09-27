"""Pair-specific RF attempt-risk composition without outcome sampling."""

from __future__ import annotations

from dataclasses import replace
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
    action_resources,
)
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.endpoint_rf_schedule import FrameEndpointRFSchedule
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.local_rf_domain import (
    FrameLocalRFLoads,
    FrameLocalRFTopology,
)
from hybrid_v2x_rl.mean_field.local_rf_response import (
    FrameLocalRFResponses,
    LocalRFResponseModel,
)
from hybrid_v2x_rl.mean_field.local_rf_risk import (
    LOCAL_RF_RISK_CONTRACT_VERSION,
    FrameLocalRFAttemptRisks,
    LocalRFRiskError,
)
from hybrid_v2x_rl.mean_field.local_rf_sensing import (
    FrameLocalRFSensedLoads,
    FrameLocalRFSensing,
)
from hybrid_v2x_rl.mean_field.random_tape import (
    MATCHED_TAPE_SCHEMA,
    MatchedPacketTapeFactory,
    PacketRandomnessIdentity,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

TRACE_ID = "synthetic-d10-train-000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATTEMPT_AIRTIME_S = 0.0005
GENERATION_PERIOD_S = 0.1
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=1.0,
    vlc_activation_cost=1.0,
)


def _vehicle(
    vehicle_id: str,
    x_m: float,
    *,
    time_s: float,
) -> VehicleTraceRecord:
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
    specs: tuple[tuple[str, str, float, str, float], ...],
    *,
    index: int = 0,
    time_s: float = 0.0,
) -> PopulationFrame:
    positions: dict[str, float] = {}
    for _, transmitter_id, transmitter_x, receiver_id, receiver_x in specs:
        for vehicle_id, x_m in (
            (transmitter_id, transmitter_x),
            (receiver_id, receiver_x),
        ):
            existing = positions.setdefault(vehicle_id, x_m)
            assert existing == x_m
    vehicles = tuple(
        _vehicle(vehicle_id, x_m, time_s=time_s)
        for vehicle_id, x_m in sorted(positions.items())
    )
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=index,
            transmitter=by_id[transmitter_id],
            receiver=by_id[receiver_id],
            lifecycle=PairLifecycle(born=index == 0),
        )
        for pair_id, transmitter_id, _, receiver_id, _ in sorted(specs)
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
        vehicles=vehicles,
        pairs=pairs,
    )


def _ledger(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
) -> FrameActionLedger:
    return FrameActionLedger.from_frame(frame, actions, resource_map=RESOURCE_MAP)


def _responses(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
    *,
    attempt_airtime_s: float = ATTEMPT_AIRTIME_S,
) -> FrameLocalRFResponses:
    ledger = _ledger(frame, actions)
    topology = FrameLocalRFTopology.from_frame(frame)
    loads = FrameLocalRFLoads.from_topology_and_ledger(topology, ledger)
    sensing = FrameLocalRFSensing.from_frame_and_topology(
        frame,
        topology,
        buildings=(),
        antenna_height_m=1.5,
    )
    sensed = FrameLocalRFSensedLoads.from_sensing_and_loads(sensing, loads)
    config = load_headline_config(PROJECT_ROOT)
    model = LocalRFResponseModel(
        parameters=build_rf_channel(
            config,
            band=SensitivityBand.NOMINAL,
        ).collision,
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=attempt_airtime_s,
    )
    return model.evaluate(sensed)


def _endpoint_schedule(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
) -> FrameEndpointRFSchedule:
    return FrameEndpointRFSchedule.from_frame_and_ledger(
        frame,
        _ledger(frame, actions),
        attempt_airtime_s=ATTEMPT_AIRTIME_S,
        generation_period_s=GENERATION_PERIOD_S,
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


def _propagation_for_actions(
    actions: dict[str, PolicyAction],
    *,
    probability: float = 0.1,
) -> dict[str, RFPropagationResult]:
    return {
        pair_id: _propagation(probability)
        for pair_id, action in actions.items()
        if action_resources(action).uses_rf
    }


def _compose(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
    *,
    probability: float = 0.1,
) -> FrameLocalRFAttemptRisks:
    return FrameLocalRFAttemptRisks.from_components(
        _responses(frame, actions),
        _endpoint_schedule(frame, actions),
        propagation_by_pair=_propagation_for_actions(
            actions,
            probability=probability,
        ),
    )


def _chain_frame() -> PopulationFrame:
    return _frame(
        (
            ("pair-a", "veh-1", 0.0, "veh-2", 10.0),
            ("pair-b", "veh-2", 10.0, "veh-3", 20.0),
            ("pair-c", "veh-4", 30.0, "veh-5", 40.0),
        )
    )


def test_risk_composes_pair_local_collision_receiver_activity_and_propagation() -> None:
    actions = {
        "pair-a": PolicyAction.RF_2,
        "pair-b": PolicyAction.RF_4,
        "pair-c": PolicyAction.VLC,
    }
    frame = _chain_frame()
    responses = _responses(frame, actions)
    schedule = _endpoint_schedule(frame, actions)
    propagations = {
        "pair-a": _propagation(0.1),
        "pair-b": _propagation(0.2),
    }

    risks = FrameLocalRFAttemptRisks.from_components(
        responses,
        schedule,
        propagation_by_pair=propagations,
    )
    pair_a = risks.risk_for("pair-a")
    expected_access = 1.0 - (
        1.0 - pair_a.collision_probability
    ) * (1.0 - 0.02)
    expected_total = 1.0 - (1.0 - expected_access) * (1.0 - 0.1)

    assert risks.contract_version == LOCAL_RF_RISK_CONTRACT_VERSION
    assert risks.rf_pair_ids == ("pair-a", "pair-b")
    assert pair_a.local_response is responses.response_for("pair-a")
    assert pair_a.half_duplex_exposure is schedule.exposure_for("pair-a")
    assert pair_a.reserved_rf_attempts == 2
    assert pair_a.half_duplex_probability == pytest.approx(0.02)
    assert pair_a.decoding_failure_probability == pytest.approx(0.1)
    assert pair_a.access_failure_probability == pytest.approx(expected_access)
    assert pair_a.total_failure_probability == pytest.approx(expected_total)
    assert risks.risk_for("pair-b").half_duplex_probability == 0.0
    assert risks.as_dict()["rf_using_pairs"] == 2


def test_pair_risks_retain_distinct_local_collision_responses() -> None:
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_4,
        "pair-c": PolicyAction.VLC,
    }
    risks = _compose(_chain_frame(), actions, probability=0.0)

    pair_a = risks.risk_for("pair-a")
    pair_b = risks.risk_for("pair-b")
    assert pair_a.collision_probability > pair_b.collision_probability
    assert pair_a.collision_probability == (
        pair_a.local_response.per_attempt_collision_probability
    )
    assert pair_b.collision_probability == (
        pair_b.local_response.per_attempt_collision_probability
    )


def test_receiver_activity_not_population_mean_enters_access_risk() -> None:
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_4,
        "pair-c": PolicyAction.VLC,
    }
    risks = _compose(_chain_frame(), actions, probability=0.0)

    assert risks.risk_for("pair-a").half_duplex_probability == pytest.approx(
        0.02
    )
    assert risks.risk_for("pair-b").half_duplex_probability == 0.0


def test_all_nine_actions_create_risks_only_for_the_eight_rf_actions() -> None:
    specs = tuple(
        (
            f"pair-{index}",
            f"tx-{index}",
            float(2 * index),
            f"rx-{index}",
            float(2 * index + 1),
        )
        for index in range(9)
    )
    actions = {
        f"pair-{index}": action for index, action in enumerate(PolicyAction)
    }

    risks = _compose(_frame(specs), actions)

    assert risks.rf_pair_ids == tuple(f"pair-{index}" for index in range(1, 9))
    assert tuple(risk.reserved_rf_attempts for risk in risks.risks) == (
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
    )
    with pytest.raises(LocalRFRiskError, match="no selected RF"):
        risks.risk_for("pair-0")


def test_zero_mechanism_limits_produce_zero_or_unit_total_risk() -> None:
    frame = _frame((("pair-a", "veh-1", 0.0, "veh-2", 10.0),))
    actions = {"pair-a": PolicyAction.RF_1}
    responses = _responses(frame, actions)
    schedule = _endpoint_schedule(frame, actions)

    zero = FrameLocalRFAttemptRisks.from_components(
        responses,
        schedule,
        propagation_by_pair={"pair-a": _propagation(0.0)},
    ).risk_for("pair-a")
    certain_decode_failure = FrameLocalRFAttemptRisks.from_components(
        responses,
        schedule,
        propagation_by_pair={"pair-a": _propagation(1.0)},
    ).risk_for("pair-a")

    assert zero.collision_probability == 0.0
    assert zero.half_duplex_probability == 0.0
    assert zero.access_failure_probability == 0.0
    assert zero.total_failure_probability == 0.0
    assert certain_decode_failure.total_failure_probability == 1.0


def test_vlc_only_and_empty_frames_produce_no_rf_risks() -> None:
    vlc_frame = _frame((("pair-a", "veh-1", 0.0, "veh-2", 10.0),))
    vlc = _compose(vlc_frame, {"pair-a": PolicyAction.VLC})
    empty = _compose(_frame(()), {})

    assert vlc.pair_ids == ("pair-a",)
    assert vlc.rf_pair_ids == ()
    assert vlc.risks == ()
    assert empty.pair_ids == ()
    assert empty.rf_pair_ids == ()
    assert empty.risks == ()


def test_propagation_mapping_must_cover_rf_using_pairs_exactly() -> None:
    frame = _chain_frame()
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_2,
        "pair-c": PolicyAction.VLC,
    }
    responses = _responses(frame, actions)
    schedule = _endpoint_schedule(frame, actions)

    with pytest.raises(LocalRFRiskError, match="cover RF-using pairs exactly"):
        FrameLocalRFAttemptRisks.from_components(
            responses,
            schedule,
            propagation_by_pair={"pair-a": _propagation(0.1)},
        )
    with pytest.raises(LocalRFRiskError, match="cover RF-using pairs exactly"):
        FrameLocalRFAttemptRisks.from_components(
            responses,
            schedule,
            propagation_by_pair={
                "pair-a": _propagation(0.1),
                "pair-b": _propagation(0.1),
                "pair-c": _propagation(0.1),
            },
        )
    with pytest.raises(LocalRFRiskError, match="invalid row"):
        FrameLocalRFAttemptRisks.from_components(
            responses,
            schedule,
            propagation_by_pair={
                "pair-a": _propagation(0.1),
                "pair-b": object(),  # type: ignore[dict-item]
            },
        )


def test_frame_identity_and_selected_reservations_must_match() -> None:
    frame = _chain_frame()
    later = _frame(
        (
            ("pair-a", "veh-1", 0.0, "veh-2", 10.0),
            ("pair-b", "veh-2", 10.0, "veh-3", 20.0),
            ("pair-c", "veh-4", 30.0, "veh-5", 40.0),
        ),
        index=1,
        time_s=0.1,
    )
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_2,
        "pair-c": PolicyAction.VLC,
    }

    with pytest.raises(LocalRFRiskError, match="different frames"):
        FrameLocalRFAttemptRisks.from_components(
            _responses(frame, actions),
            _endpoint_schedule(later, actions),
            propagation_by_pair=_propagation_for_actions(actions),
        )

    changed_actions = dict(actions)
    changed_actions["pair-a"] = PolicyAction.RF_2
    with pytest.raises(LocalRFRiskError, match="counts differ"):
        FrameLocalRFAttemptRisks.from_components(
            _responses(frame, actions),
            _endpoint_schedule(frame, changed_actions),
            propagation_by_pair=_propagation_for_actions(actions),
        )


def test_collision_and_endpoint_timing_must_match() -> None:
    frame = _frame((("pair-a", "veh-1", 0.0, "veh-2", 10.0),))
    actions = {"pair-a": PolicyAction.RF_1}

    with pytest.raises(LocalRFRiskError, match="different RF timing"):
        FrameLocalRFAttemptRisks.from_components(
            _responses(frame, actions, attempt_airtime_s=0.001),
            _endpoint_schedule(frame, actions),
            propagation_by_pair={"pair-a": _propagation(0.1)},
        )


def test_invalid_propagation_and_derived_probability_drift_fail_closed() -> None:
    frame = _frame((("pair-a", "veh-1", 0.0, "veh-2", 10.0),))
    actions = {"pair-a": PolicyAction.RF_1}
    responses = _responses(frame, actions)
    schedule = _endpoint_schedule(frame, actions)
    valid = FrameLocalRFAttemptRisks.from_components(
        responses,
        schedule,
        propagation_by_pair={"pair-a": _propagation(0.1)},
    )

    with pytest.raises(LocalRFRiskError, match="decoding failure"):
        FrameLocalRFAttemptRisks.from_components(
            responses,
            schedule,
            propagation_by_pair={"pair-a": _propagation(float("nan"))},
        )
    with pytest.raises(LocalRFRiskError, match="diagnostics"):
        FrameLocalRFAttemptRisks.from_components(
            responses,
            schedule,
            propagation_by_pair={
                "pair-a": replace(_propagation(0.1), sinr_db=float("inf"))
            },
        )
    with pytest.raises(LocalRFRiskError, match="do not reconcile"):
        replace(valid.risks[0], total_failure_probability=0.9)
    with pytest.raises(LocalRFRiskError, match="align"):
        replace(valid, rf_pair_ids=("pair-a",), risks=())


def test_composition_is_invariant_to_propagation_mapping_order() -> None:
    frame = _chain_frame()
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_2,
        "pair-c": PolicyAction.VLC,
    }
    responses = _responses(frame, actions)
    schedule = _endpoint_schedule(frame, actions)
    forward = {
        "pair-a": _propagation(0.1),
        "pair-b": _propagation(0.2),
    }
    reverse = dict(reversed(tuple(forward.items())))

    first = FrameLocalRFAttemptRisks.from_components(
        responses,
        schedule,
        propagation_by_pair=forward,
    )
    second = FrameLocalRFAttemptRisks.from_components(
        responses,
        schedule,
        propagation_by_pair=reverse,
    )

    assert first == second


def test_risk_composition_does_not_consume_or_change_matched_packet_tapes() -> None:
    frame = _frame((("pair-a", "veh-1", 0.0, "veh-2", 10.0),))
    actions = {"pair-a": PolicyAction.RF_1}
    factory = MatchedPacketTapeFactory(root_seed=101)
    identity = PacketRandomnessIdentity(
        trace_id=TRACE_ID,
        pair_episode_id="pair-a",
        packet_index=0,
    )
    before = factory.build(identity)

    risk = _compose(frame, actions).risk_for("pair-a")
    after = factory.build(identity)
    diagnostics = risk.as_dict()

    assert MATCHED_TAPE_SCHEMA == "hybrid-rf-vlc-rl.matched-packet-tape.v1"
    assert before == after
    assert "success" not in diagnostics
    assert "failure_cause" not in diagnostics
    assert not any("draw" in key for key in diagnostics)
