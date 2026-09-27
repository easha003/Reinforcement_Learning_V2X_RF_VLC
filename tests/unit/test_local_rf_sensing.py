"""Pair-local geometric reservation sensing and attempt partition contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.geometry import OrientedRectangle, Point
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
)
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
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
from hybrid_v2x_rl.mean_field.local_rf_sensing import (
    LOCAL_RF_SENSING_CONTRACT_VERSION,
    FrameLocalRFSensedLoads,
    FrameLocalRFSensing,
    LocalRFSensingError,
    sensing_buildings_from_config,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

TRACE_ID = "synthetic-d10-train-000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANTENNA_HEIGHT_M = 1.5
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=1.0,
    vlc_activation_cost=1.0,
)
WALL = OrientedRectangle(
    centre=Point(50.0, 0.0),
    heading_rad=0.0,
    length_m=10.0,
    width_m=100.0,
)


def _vehicle(
    vehicle_id: str,
    x_m: float,
    y_m: float,
    *,
    time_s: float,
) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=time_s,
        vehicle_id=vehicle_id,
        x_m=x_m,
        y_m=y_m,
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
    specs: tuple[tuple[str, str, float, float, str, float, float], ...],
    *,
    index: int = 0,
    time_s: float = 0.0,
) -> PopulationFrame:
    positions: dict[str, tuple[float, float]] = {}
    for _, transmitter_id, tx_x, tx_y, receiver_id, rx_x, rx_y in specs:
        for vehicle_id, position in (
            (transmitter_id, (tx_x, tx_y)),
            (receiver_id, (rx_x, rx_y)),
        ):
            existing = positions.setdefault(vehicle_id, position)
            assert existing == position
    vehicles = tuple(
        _vehicle(vehicle_id, x_m, y_m, time_s=time_s)
        for vehicle_id, (x_m, y_m) in sorted(positions.items())
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
        for pair_id, transmitter_id, _, _, receiver_id, _, _ in sorted(specs)
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


def _shared_transmitter_frame() -> PopulationFrame:
    return _frame(
        (
            ("pair-a", "tx-shared", 0.0, 0.0, "rx-a", 10.0, 0.0),
            ("pair-b", "tx-b", 100.0, 0.0, "rx-b", 110.0, 0.0),
            ("pair-c", "tx-shared", 0.0, 0.0, "rx-c", 20.0, 0.0),
        )
    )


def _sensing(
    frame: PopulationFrame,
    *,
    buildings: tuple[OrientedRectangle, ...] = (WALL,),
) -> FrameLocalRFSensing:
    return FrameLocalRFSensing.from_frame_and_topology(
        frame,
        FrameLocalRFTopology.from_frame(frame),
        buildings=buildings,
        antenna_height_m=ANTENNA_HEIGHT_M,
    )


def _sensed_loads(
    frame: PopulationFrame,
    actions: dict[str, PolicyAction],
    *,
    buildings: tuple[OrientedRectangle, ...] = (WALL,),
) -> FrameLocalRFSensedLoads:
    topology = FrameLocalRFTopology.from_frame(frame)
    sensing = FrameLocalRFSensing.from_frame_and_topology(
        frame,
        topology,
        buildings=buildings,
        antenna_height_m=ANTENNA_HEIGHT_M,
    )
    loads = FrameLocalRFLoads.from_topology_and_ledger(
        topology,
        _ledger(frame, actions),
    )
    return FrameLocalRFSensedLoads.from_sensing_and_loads(sensing, loads)


def test_headline_building_geometry_is_complete_and_deterministic() -> None:
    config = load_headline_config(PROJECT_ROOT)

    first = sensing_buildings_from_config(config)
    second = sensing_buildings_from_config(config)
    first_report = FrameLocalRFSensing.from_frame_and_topology(
        _frame(()),
        FrameLocalRFTopology.from_frame(_frame(())),
        buildings=first,
        antenna_height_m=float(config.geometry.rf_antenna_height_m),
    )
    second_report = FrameLocalRFSensing.from_frame_and_topology(
        _frame(()),
        FrameLocalRFTopology.from_frame(_frame(())),
        buildings=second,
        antenna_height_m=float(config.geometry.rf_antenna_height_m),
    )

    assert len(first) == 55
    assert first == second
    assert first_report.building_count == 55
    assert first_report.building_geometry_sha256 == (
        second_report.building_geometry_sha256
    )


def test_building_visibility_is_pair_local_and_colocated_flows_remain_known() -> None:
    sensing = _sensing(_shared_transmitter_frame())

    pair_a = sensing.sensing_for("pair-a")
    assert pair_a.member_for("pair-a").colocated_with_focal_transmitter
    assert pair_a.member_for("pair-a").geometrically_decodable
    assert pair_a.member_for("pair-c").colocated_with_focal_transmitter
    assert pair_a.member_for("pair-c").geometrically_decodable
    assert pair_a.member_for("pair-b").building_blocked
    assert not pair_a.member_for("pair-b").geometrically_decodable

    pair_b = sensing.sensing_for("pair-b")
    assert pair_b.member_for("pair-b").colocated_with_focal_transmitter
    assert pair_b.member_for("pair-b").geometrically_decodable
    assert pair_b.member_for("pair-a").building_blocked
    assert pair_b.member_for("pair-c").building_blocked
    assert sensing.contract_version == LOCAL_RF_SENSING_CONTRACT_VERSION
    with pytest.raises(LocalRFSensingError, match="absent"):
        sensing.sensing_for("missing")


def test_sensed_load_partitions_focal_colocated_and_hidden_attempts() -> None:
    sensed = _sensed_loads(
        _shared_transmitter_frame(),
        {
            "pair-a": PolicyAction.RF_2,
            "pair-b": PolicyAction.RF_4,
            "pair-c": PolicyAction.RF_1,
        },
    )

    pair_a = sensed.row_for("pair-a")
    assert pair_a.local_offered_rf_attempts == 7
    assert pair_a.focal_rf_attempts == 2
    assert pair_a.colocated_other_rf_attempts == 1
    assert pair_a.external_contending_rf_attempts == 4
    assert pair_a.geometrically_sensed_external_rf_attempts == 0
    assert pair_a.geometrically_hidden_external_rf_attempts == 4
    assert pair_a.sensed_fraction == 0.0
    assert tuple(reservation.role for reservation in pair_a.reservations) == (
        "focal",
        "external_geometrically_hidden",
        "colocated_other_flow",
    )

    pair_b = sensed.row_for("pair-b")
    assert pair_b.local_offered_rf_attempts == 7
    assert pair_b.focal_rf_attempts == 4
    assert pair_b.colocated_other_rf_attempts == 0
    assert pair_b.external_contending_rf_attempts == 3
    assert pair_b.geometrically_sensed_external_rf_attempts == 0
    assert pair_b.geometrically_hidden_external_rf_attempts == 3
    assert pair_b.sensed_fraction == 0.0


def test_sensed_fraction_is_weighted_by_external_attempts_not_flow_count() -> None:
    frame = _frame(
        (
            ("pair-a", "tx-a", 0.0, 0.0, "rx-a", 10.0, 0.0),
            ("pair-b", "tx-hidden", 100.0, 0.0, "rx-b", 110.0, 0.0),
            ("pair-c", "tx-visible", 20.0, 0.0, "rx-c", 30.0, 0.0),
            ("pair-d", "tx-zero", 25.0, 0.0, "rx-d", 35.0, 0.0),
        )
    )
    sensed = _sensed_loads(
        frame,
        {
            "pair-a": PolicyAction.VLC,
            "pair-b": PolicyAction.RF_4,
            "pair-c": PolicyAction.RF_2,
            "pair-d": PolicyAction.VLC,
        },
    )

    pair_a = sensed.row_for("pair-a")
    assert pair_a.external_contending_rf_attempts == 6
    assert pair_a.geometrically_sensed_external_rf_attempts == 2
    assert pair_a.geometrically_hidden_external_rf_attempts == 4
    assert pair_a.sensed_fraction == pytest.approx(1.0 / 3.0)


def test_no_external_rf_attempts_has_neutral_sensed_fraction() -> None:
    frame = _shared_transmitter_frame()
    sensed = _sensed_loads(
        frame,
        {
            "pair-a": PolicyAction.RF_2,
            "pair-b": PolicyAction.VLC,
            "pair-c": PolicyAction.RF_1,
        },
    )

    pair_a = sensed.row_for("pair-a")
    assert pair_a.external_contending_rf_attempts == 0
    assert pair_a.sensed_fraction == 1.0


def test_no_buildings_makes_every_external_reservation_geometrically_decodable() -> None:
    frame = _shared_transmitter_frame()
    sensed = _sensed_loads(
        frame,
        {
            "pair-a": PolicyAction.RF_1,
            "pair-b": PolicyAction.RF_4,
            "pair-c": PolicyAction.VLC,
        },
        buildings=(),
    )

    pair_a = sensed.row_for("pair-a")
    assert pair_a.external_contending_rf_attempts == 4
    assert pair_a.geometrically_sensed_external_rf_attempts == 4
    assert pair_a.geometrically_hidden_external_rf_attempts == 0
    assert pair_a.sensed_fraction == 1.0


def test_empty_frame_produces_empty_sensing_and_sensed_loads() -> None:
    frame = _frame(())
    topology = FrameLocalRFTopology.from_frame(frame)
    sensing = FrameLocalRFSensing.from_frame_and_topology(
        frame,
        topology,
        buildings=(WALL,),
        antenna_height_m=ANTENNA_HEIGHT_M,
    )
    loads = FrameLocalRFLoads.from_topology_and_ledger(
        topology,
        _ledger(frame, {}),
    )
    sensed = FrameLocalRFSensedLoads.from_sensing_and_loads(sensing, loads)

    assert sensing.pair_ids == ()
    assert sensing.sensing == ()
    assert sensed.rows == ()
    assert sensed.as_dict()["active_pairs"] == 0


def test_sensing_and_load_inputs_must_identify_the_same_frame() -> None:
    first = _shared_transmitter_frame()
    later = _frame(
        (
            ("pair-a", "tx-shared", 0.0, 0.0, "rx-a", 10.0, 0.0),
            ("pair-b", "tx-b", 100.0, 0.0, "rx-b", 110.0, 0.0),
            ("pair-c", "tx-shared", 0.0, 0.0, "rx-c", 20.0, 0.0),
        ),
        index=1,
        time_s=0.1,
    )
    first_topology = FrameLocalRFTopology.from_frame(first)
    later_topology = FrameLocalRFTopology.from_frame(later)

    with pytest.raises(LocalRFSensingError, match="same frame"):
        FrameLocalRFSensing.from_frame_and_topology(
            later,
            first_topology,
            buildings=(),
            antenna_height_m=ANTENNA_HEIGHT_M,
        )

    sensing = FrameLocalRFSensing.from_frame_and_topology(
        first,
        first_topology,
        buildings=(),
        antenna_height_m=ANTENNA_HEIGHT_M,
    )
    later_loads = FrameLocalRFLoads.from_topology_and_ledger(
        later_topology,
        _ledger(later, {pair_id: PolicyAction.VLC for pair_id in later.active_pair_ids}),
    )
    with pytest.raises(LocalRFSensingError, match="same frame"):
        FrameLocalRFSensedLoads.from_sensing_and_loads(sensing, later_loads)
