"""Pair-local RF topology and selected-reservation accounting contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
)
from hybrid_v2x_rl.env.perception import CONTENTION_RADIUS_M as ACTOR_RADIUS_M
from hybrid_v2x_rl.env.rollout import CONTENTION_RADIUS_M as LEGACY_RADIUS_M
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.local_rf_domain import (
    LOCAL_RF_CONTENTION_RADIUS_M,
    LOCAL_RF_DOMAIN_CONTRACT_VERSION,
    FrameLocalRFLoads,
    FrameLocalRFTopology,
    LocalRFDomainError,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

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


def _spatial_frame() -> PopulationFrame:
    return _frame(
        (
            ("pair-a", "tx-a", 0.0, "rx-a", 10.0),
            ("pair-b", "tx-b", 200.0, "rx-b", 210.0),
            ("pair-c", "tx-c", 200.001, "rx-c", 220.0),
            ("pair-d", "tx-d", 500.0, "rx-d", 510.0),
        )
    )


def test_local_radius_matches_existing_actor_and_physical_contracts() -> None:
    assert LOCAL_RF_CONTENTION_RADIUS_M == ACTOR_RADIUS_M == LEGACY_RADIUS_M


def test_topology_is_transmitter_centred_reciprocal_and_boundary_inclusive() -> None:
    topology = FrameLocalRFTopology.from_frame(_spatial_frame())

    assert topology.contract_version == LOCAL_RF_DOMAIN_CONTRACT_VERSION
    assert topology.radius_m == LOCAL_RF_CONTENTION_RADIUS_M
    assert topology.domain_for("pair-a").member_pair_ids == ("pair-a", "pair-b")
    assert topology.domain_for("pair-b").member_pair_ids == (
        "pair-a",
        "pair-b",
        "pair-c",
    )
    assert topology.domain_for("pair-c").member_pair_ids == ("pair-b", "pair-c")
    assert topology.domain_for("pair-d").member_pair_ids == ("pair-d",)
    with pytest.raises(LocalRFDomainError, match="absent"):
        topology.domain_for("missing")


def test_local_load_sums_each_member_flow_reservation_once() -> None:
    frame = _spatial_frame()
    topology = FrameLocalRFTopology.from_frame(frame)
    ledger = _ledger(
        frame,
        {
            "pair-a": PolicyAction.VLC,
            "pair-b": PolicyAction.RF_2,
            "pair-c": PolicyAction.DUP_4,
            "pair-d": PolicyAction.RF_1,
        },
    )

    loads = FrameLocalRFLoads.from_topology_and_ledger(topology, ledger)

    assert loads.load_for("pair-a").reserved_rf_attempts_by_pair == (
        ("pair-a", 0),
        ("pair-b", 2),
    )
    assert loads.load_for("pair-a").offered_rf_attempts == 2
    assert loads.load_for("pair-a").rf_using_pairs == 1
    assert loads.load_for("pair-b").offered_rf_attempts == 6
    assert loads.load_for("pair-c").offered_rf_attempts == 6
    assert loads.load_for("pair-d").offered_rf_attempts == 1


def test_distant_action_change_does_not_change_focal_local_demand() -> None:
    frame = _spatial_frame()
    topology = FrameLocalRFTopology.from_frame(frame)
    actions = {
        "pair-a": PolicyAction.RF_1,
        "pair-b": PolicyAction.RF_1,
        "pair-c": PolicyAction.VLC,
        "pair-d": PolicyAction.RF_1,
    }
    first = FrameLocalRFLoads.from_topology_and_ledger(
        topology, _ledger(frame, actions)
    )
    actions["pair-d"] = PolicyAction.RF_4
    second = FrameLocalRFLoads.from_topology_and_ledger(
        topology, _ledger(frame, actions)
    )

    assert first.load_for("pair-a") == second.load_for("pair-a")
    assert first.load_for("pair-d").offered_rf_attempts == 1
    assert second.load_for("pair-d").offered_rf_attempts == 4


def test_shared_transmitter_flows_remain_distinct_and_conserve() -> None:
    frame = _frame(
        (
            ("pair-a", "tx-shared", 0.0, "rx-a", 10.0),
            ("pair-b", "tx-shared", 0.0, "rx-b", 20.0),
            ("pair-c", "tx-other", 100.0, "rx-c", 110.0),
        )
    )
    topology = FrameLocalRFTopology.from_frame(frame)
    loads = FrameLocalRFLoads.from_topology_and_ledger(
        topology,
        _ledger(
            frame,
            {
                "pair-a": PolicyAction.RF_1,
                "pair-b": PolicyAction.RF_2,
                "pair-c": PolicyAction.VLC,
            },
        ),
    )

    domain = topology.domain_for("pair-a")
    assert domain.member_pair_ids == ("pair-a", "pair-b", "pair-c")
    assert domain.member_transmitter_ids == ("tx-shared", "tx-shared", "tx-other")
    assert loads.load_for("pair-a").reserved_rf_attempts_by_pair == (
        ("pair-a", 1),
        ("pair-b", 2),
        ("pair-c", 0),
    )
    assert loads.load_for("pair-a").offered_rf_attempts == 3


def test_empty_frame_produces_empty_topology_and_loads() -> None:
    frame = _frame(())
    topology = FrameLocalRFTopology.from_frame(frame)
    loads = FrameLocalRFLoads.from_topology_and_ledger(
        topology,
        _ledger(frame, {}),
    )

    assert topology.pair_ids == ()
    assert topology.domains == ()
    assert loads.loads == ()
    assert loads.as_dict()["active_pairs"] == 0


def test_topology_and_ledger_must_identify_the_same_frame() -> None:
    topology = FrameLocalRFTopology.from_frame(_spatial_frame())
    later = _frame(
        (
            ("pair-a", "tx-a", 0.0, "rx-a", 10.0),
            ("pair-b", "tx-b", 200.0, "rx-b", 210.0),
            ("pair-c", "tx-c", 200.001, "rx-c", 220.0),
            ("pair-d", "tx-d", 500.0, "rx-d", 510.0),
        ),
        index=1,
        time_s=0.1,
    )
    ledger = _ledger(later, {pair_id: PolicyAction.VLC for pair_id in later.active_pair_ids})

    with pytest.raises(LocalRFDomainError, match="same frame"):
        FrameLocalRFLoads.from_topology_and_ledger(topology, ledger)


def test_radius_must_be_finite_and_positive() -> None:
    with pytest.raises(LocalRFDomainError, match="radius"):
        FrameLocalRFTopology.from_frame(_spatial_frame(), radius_m=0.0)
