"""Endpoint-specific RF activity and half-duplex exposure contract."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
)
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.endpoint_rf_schedule import (
    ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION,
    EndpointRFScheduleError,
    FrameEndpointRFSchedule,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
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
    endpoints: tuple[tuple[str, str, str], ...],
    *,
    index: int = 0,
    time_s: float = 0.0,
) -> PopulationFrame:
    vehicle_ids = tuple(
        sorted(
            {
                endpoint
                for _, transmitter_id, receiver_id in endpoints
                for endpoint in (transmitter_id, receiver_id)
            }
        )
    )
    vehicles = tuple(
        _vehicle(vehicle_id, position * 10.0, time_s=time_s)
        for position, vehicle_id in enumerate(vehicle_ids)
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
        for pair_id, transmitter_id, receiver_id in sorted(endpoints)
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


def _schedule(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
) -> FrameEndpointRFSchedule:
    return FrameEndpointRFSchedule.from_frame_and_ledger(
        frame,
        _ledger(frame, actions),
        attempt_airtime_s=ATTEMPT_AIRTIME_S,
        generation_period_s=GENERATION_PERIOD_S,
    )


def test_receiver_exposure_uses_that_physical_endpoints_transmit_activity() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-2", "veh-3"),
            ("pair-c", "veh-4", "veh-2"),
        )
    )
    schedule = _schedule(
        frame,
        {
            "pair-a": PolicyAction.RF_1,
            "pair-b": PolicyAction.RF_4,
            "pair-c": PolicyAction.VLC,
        },
    )

    assert schedule.contract_version == ENDPOINT_RF_SCHEDULE_CONTRACT_VERSION
    assert schedule.total_reserved_rf_attempts == 5
    assert schedule.endpoint_offered_rf_attempts == 5
    assert schedule.schedule_for("veh-2").offered_rf_attempts == 4
    assert schedule.schedule_for("veh-2").transmit_duty_cycle == pytest.approx(
        0.02
    )
    assert schedule.exposure_for("pair-a").half_duplex_probability == pytest.approx(
        0.02
    )
    assert schedule.exposure_for("pair-c").half_duplex_probability == pytest.approx(
        0.02
    )
    assert schedule.exposure_for("pair-a").receiver_transmitting_pair_ids == (
        "pair-b",
    )
    assert schedule.exposure_for("pair-b").half_duplex_probability == 0.0
    assert schedule.exposure_for("pair-b").receiver_transmitting_pair_ids == ()


def test_shared_transmitter_serializes_distinct_service_reservations() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-shared", "rx-a"),
            ("pair-b", "veh-shared", "rx-b"),
            ("pair-c", "veh-shared", "rx-c"),
        )
    )
    schedule = _schedule(
        frame,
        {
            "pair-a": PolicyAction.RF_4,
            "pair-b": PolicyAction.DUP_3,
            "pair-c": PolicyAction.VLC,
        },
    )

    shared = schedule.schedule_for("veh-shared")
    assert shared.transmitting_pair_ids == ("pair-a", "pair-b", "pair-c")
    assert shared.rf_using_pair_ids == ("pair-a", "pair-b")
    assert tuple(
        row.reserved_rf_attempts for row in shared.reservations
    ) == (4, 3, 0)
    assert tuple(
        (
            attempt.pair_id,
            attempt.pair_attempt_index,
            attempt.endpoint_sequence_index,
        )
        for attempt in shared.serialized_attempts
    ) == (
        ("pair-a", 0, 0),
        ("pair-a", 1, 1),
        ("pair-a", 2, 2),
        ("pair-a", 3, 3),
        ("pair-b", 0, 4),
        ("pair-b", 1, 5),
        ("pair-b", 2, 6),
    )
    assert shared.offered_rf_attempts == 7
    assert shared.offered_airtime_s == pytest.approx(0.0035)
    assert shared.endpoint_utilization == pytest.approx(0.035)
    assert shared.transmit_duty_cycle == pytest.approx(0.035)
    assert not shared.oversubscribed
    last_attempt = shared.serialized_attempts[-1].as_dict(
        attempt_airtime_s=shared.attempt_airtime_s
    )
    assert last_attempt["serialization_end_offset_s"] == pytest.approx(0.0035)
    assert schedule.total_reserved_rf_attempts == 7
    assert schedule.endpoint_offered_rf_attempts == 7


def test_endpoint_exposure_is_not_the_population_mean_duty_cycle() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-2", "veh-3"),
            ("pair-c", "veh-4", "veh-5"),
        )
    )
    schedule = _schedule(
        frame,
        {
            "pair-a": PolicyAction.RF_1,
            "pair-b": PolicyAction.RF_4,
            "pair-c": PolicyAction.VLC,
        },
    )
    legacy_population_mean = (
        schedule.total_reserved_rf_attempts
        / len(schedule.pair_ids)
        * ATTEMPT_AIRTIME_S
        / GENERATION_PERIOD_S
    )

    assert legacy_population_mean == pytest.approx(1.0 / 120.0)
    assert schedule.exposure_for("pair-a").half_duplex_probability == pytest.approx(
        0.02
    )
    assert schedule.exposure_for("pair-b").half_duplex_probability == 0.0
    assert schedule.exposure_for("pair-c").half_duplex_probability == 0.0
    assert all(
        exposure.half_duplex_probability != pytest.approx(legacy_population_mean)
        for exposure in schedule.exposures
    )


def test_inbound_reservations_do_not_make_a_receiver_transmit() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-hub"),
            ("pair-b", "veh-2", "veh-hub"),
        )
    )
    schedule = _schedule(
        frame,
        {"pair-a": PolicyAction.RF_4, "pair-b": PolicyAction.DUP_4},
    )

    hub = schedule.schedule_for("veh-hub")
    assert hub.offered_rf_attempts == 0
    assert hub.transmitting_pair_ids == ()
    assert hub.serialized_attempts == ()
    assert schedule.exposure_for("pair-a").half_duplex_probability == 0.0
    assert schedule.exposure_for("pair-b").half_duplex_probability == 0.0


def test_multiple_outgoing_flows_sum_for_every_shared_receiver_exposure() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-hub"),
            ("pair-b", "veh-hub", "veh-3"),
            ("pair-c", "veh-hub", "veh-4"),
            ("pair-d", "veh-5", "veh-hub"),
        )
    )
    schedule = _schedule(
        frame,
        {
            "pair-a": PolicyAction.RF_1,
            "pair-b": PolicyAction.RF_2,
            "pair-c": PolicyAction.DUP_3,
            "pair-d": PolicyAction.VLC,
        },
    )

    hub = schedule.schedule_for("veh-hub")
    assert hub.offered_rf_attempts == 5
    assert hub.rf_using_pair_ids == ("pair-b", "pair-c")
    for pair_id in ("pair-a", "pair-d"):
        exposure = schedule.exposure_for(pair_id)
        assert exposure.receiver_reserved_rf_attempts == 5
        assert exposure.receiver_transmitting_pair_ids == ("pair-b", "pair-c")
        assert exposure.half_duplex_probability == pytest.approx(0.025)
    assert schedule.total_reserved_rf_attempts == 6


@pytest.mark.parametrize(
    ("flows", "expected_utilization", "expected_overload"),
    ((50, 1.0, False), (51, 1.02, True)),
)
def test_endpoint_capacity_boundary_preserves_offered_overload(
    flows: int,
    expected_utilization: float,
    expected_overload: bool,
) -> None:
    endpoints = tuple(
        (f"pair-{index:03d}", "veh-shared", f"rx-{index:03d}")
        for index in range(flows)
    )
    frame = _frame(endpoints)
    schedule = _schedule(
        frame,
        {pair_id: PolicyAction.RF_4 for pair_id, _, _ in endpoints},
    )

    shared = schedule.schedule_for("veh-shared")
    assert shared.offered_rf_attempts == 4 * flows
    assert len(shared.serialized_attempts) == 4 * flows
    assert shared.endpoint_utilization == pytest.approx(expected_utilization)
    assert shared.transmit_duty_cycle == pytest.approx(1.0)
    assert shared.oversubscribed is expected_overload
    assert ("veh-shared" in schedule.oversubscribed_endpoint_ids) is (
        expected_overload
    )
    assert schedule.endpoint_offered_rf_attempts == 4 * flows


def test_all_nine_actions_conserve_at_pair_endpoint_and_frame_boundaries() -> None:
    endpoints = tuple(
        (f"pair-{index}", f"tx-{index}", f"rx-{index}")
        for index in range(9)
    )
    frame = _frame(endpoints)
    schedule = _schedule(
        frame,
        {f"pair-{index}": action for index, action in enumerate(PolicyAction)},
    )

    assert tuple(
        reservation.reserved_rf_attempts
        for reservation in schedule.reservations
    ) == (0, 1, 2, 3, 4, 1, 2, 3, 4)
    assert schedule.total_reserved_rf_attempts == 20
    assert schedule.endpoint_offered_rf_attempts == 20
    assert sum(
        schedule.schedule_for(f"tx-{index}").offered_rf_attempts
        for index in range(9)
    ) == 20


def test_vlc_release_is_current_frame_activity_not_carried_history() -> None:
    endpoints = (("pair-a", "veh-1", "veh-2"),)
    previous_frame = _frame(endpoints, index=1, time_s=0.1)
    current_frame = _frame(endpoints, index=2, time_s=0.2)

    previous = _schedule(previous_frame, {"pair-a": PolicyAction.DUP_4})
    current = _schedule(current_frame, {"pair-a": PolicyAction.VLC})

    assert previous.schedule_for("veh-1").offered_rf_attempts == 4
    assert current.schedule_for("veh-1").offered_rf_attempts == 0
    assert current.total_reserved_rf_attempts == 0


def test_empty_population_has_empty_activity_and_exposure() -> None:
    frame = _frame(())
    schedule = _schedule(frame, {})

    assert schedule.pair_ids == ()
    assert schedule.endpoint_ids == ()
    assert schedule.reservations == ()
    assert schedule.endpoint_schedules == ()
    assert schedule.exposures == ()
    assert schedule.total_reserved_rf_attempts == 0
    assert schedule.as_dict()["physical_endpoints"] == 0


def test_schedule_is_invariant_to_input_mapping_and_specification_order() -> None:
    endpoints = (
        ("pair-a", "veh-1", "veh-2"),
        ("pair-b", "veh-2", "veh-3"),
        ("pair-c", "veh-1", "veh-4"),
    )
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_2,
        "pair-c": PolicyAction.DUP_3,
    }
    reversed_actions = dict(reversed(tuple(actions.items())))

    first = _schedule(_frame(endpoints), actions)
    second = _schedule(_frame(tuple(reversed(endpoints))), reversed_actions)

    assert first == second


def test_frame_and_ledger_must_identify_the_same_population_frame() -> None:
    endpoints = (("pair-a", "veh-1", "veh-2"),)
    first = _frame(endpoints)
    later = _frame(endpoints, index=1, time_s=0.1)
    later_ledger = _ledger(later, {"pair-a": PolicyAction.RF_1})

    with pytest.raises(EndpointRFScheduleError, match="same frame"):
        FrameEndpointRFSchedule.from_frame_and_ledger(
            first,
            later_ledger,
            attempt_airtime_s=ATTEMPT_AIRTIME_S,
            generation_period_s=GENERATION_PERIOD_S,
        )


def test_schedule_and_exposure_fields_fail_closed_on_drift() -> None:
    frame = _frame(
        (
            ("pair-a", "veh-1", "veh-2"),
            ("pair-b", "veh-2", "veh-3"),
        )
    )
    schedule = _schedule(
        frame,
        {"pair-a": PolicyAction.RF_1, "pair-b": PolicyAction.RF_4},
    )
    endpoint = schedule.schedule_for("veh-2")
    exposure = schedule.exposure_for("pair-a")

    with pytest.raises(EndpointRFScheduleError, match="does not conserve"):
        replace(endpoint, offered_rf_attempts=3)
    with pytest.raises(EndpointRFScheduleError, match="duty cycle"):
        replace(exposure, half_duplex_probability=0.5)
    with pytest.raises(EndpointRFScheduleError, match="totals do not conserve"):
        replace(schedule, total_reserved_rf_attempts=99)
    with pytest.raises(EndpointRFScheduleError, match="absent"):
        schedule.schedule_for("missing")
    with pytest.raises(EndpointRFScheduleError, match="absent"):
        schedule.exposure_for("missing")


@pytest.mark.parametrize(
    ("attempt_airtime_s", "generation_period_s"),
    ((0.0, 0.1), (0.0005, 0.0), (0.2, 0.1)),
)
def test_invalid_endpoint_timing_is_rejected(
    attempt_airtime_s: float,
    generation_period_s: float,
) -> None:
    frame = _frame((("pair-a", "veh-1", "veh-2"),))
    ledger = _ledger(frame, {"pair-a": PolicyAction.RF_1})

    with pytest.raises(EndpointRFScheduleError, match="timing"):
        FrameEndpointRFSchedule.from_frame_and_ledger(
            frame,
            ledger,
            attempt_airtime_s=attempt_airtime_s,
            generation_period_s=generation_period_s,
        )


def test_headline_timing_matches_endpoint_contract_units() -> None:
    config = load_headline_config(PROJECT_ROOT)

    assert float(config.rf.timing.airtime_s) == pytest.approx(
        ATTEMPT_AIRTIME_S
    )
    assert float(config.service.generation_period_s) == pytest.approx(
        GENERATION_PERIOD_S
    )
