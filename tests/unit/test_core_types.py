"""Contract tests for immutable cross-module value objects."""

from dataclasses import FrozenInstanceError, fields

import pytest

from hybrid_v2x_rl.core.enums import Action, FailureCause, Link
from hybrid_v2x_rl.core.types import (
    LinkAttemptResult,
    PacketOutcome,
    PairState,
    ServiceProfile,
    VehicleState,
)


def _vehicle(vehicle_id: str) -> VehicleState:
    return VehicleState(
        vehicle_id=vehicle_id,
        time_s=1.5,
        x_m=10.0,
        y_m=20.0,
        heading_rad=0.0,
        speed_mps=11.18,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge_0_0",
        edge_id="edge_0",
        route_id="route_0",
    )


def test_vehicle_state_schema_is_stable_and_immutable() -> None:
    state = _vehicle("veh-1")
    assert [field.name for field in fields(VehicleState)] == [
        "vehicle_id",
        "time_s",
        "x_m",
        "y_m",
        "heading_rad",
        "speed_mps",
        "acceleration_mps2",
        "length_m",
        "width_m",
        "height_m",
        "lane_id",
        "edge_id",
        "route_id",
    ]
    assert not hasattr(state, "__dict__")
    with pytest.raises(FrozenInstanceError):
        state.x_m = 99.0  # type: ignore[misc]


def test_pair_and_service_profiles_are_value_objects() -> None:
    tx = _vehicle("tx")
    rx = _vehicle("rx")
    pair = PairState(time_s=1.5, tx=tx, rx=rx, distance_m=22.0, same_route=True)
    service = ServiceProfile(
        name="headline",
        payload_bytes=300,
        generation_period_s=0.1,
        deadline_s=0.003,
        miss_budget=1e-4,
    )
    assert pair.tx is tx
    assert pair.rx is rx
    assert service.payload_bytes == 300
    with pytest.raises(FrozenInstanceError):
        pair.same_route = False  # type: ignore[misc]


def test_packet_outcome_uses_stable_enum_types() -> None:
    rf = LinkAttemptResult(
        link=Link.RF,
        selected=True,
        success=True,
        arrival_time_s=2.0008,
        conditional_failure_probability=1e-5,
        failure_cause=FailureCause.NONE,
    )
    vlc = LinkAttemptResult(
        link=Link.VLC,
        selected=False,
        success=False,
        arrival_time_s=None,
        conditional_failure_probability=0.2,
        failure_cause=FailureCause.NONE,
    )
    outcome = PacketOutcome(
        packet_id="packet-0",
        generation_time_s=2.0,
        action=Action.RF,
        delivered=True,
        deadline_missed=False,
        delivery_time_s=rf.arrival_time_s,
        rf=rf,
        vlc=vlc,
        activation_cost=1.0,
    )
    assert outcome.action is Action.RF
    assert outcome.rf.link is Link.RF
    with pytest.raises(FrozenInstanceError):
        outcome.delivered = False  # type: ignore[misc]
