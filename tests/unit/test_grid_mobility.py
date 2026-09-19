"""Analytic Manhattan-grid mobility tests.

These assert the four properties that `research/MOBILITY_MODEL_V2_PROPOSAL.md`
section 2 declares load-bearing, plus the two additional Gate-1 checks that
section 7 proposes and the SUMO backend could not satisfy.
"""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.core.geometry import (
    OrientedRectangle,
    Point,
    Segment,
    rectangle_from_front,
    rectangle_penetration_m,
    rectangles_overlap,
    vehicle_rectangle,
)
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex
from hybrid_v2x_rl.mobility.car_following import (
    IDMParameters,
    MOBILParameters,
    acceleration_mps2,
    integrate,
    mobil_accepts,
    stop_line_gap,
)
from hybrid_v2x_rl.mobility.grid_network import Direction, GridNetwork, GridNetworkSpec
from hybrid_v2x_rl.mobility.grid_signals import SignalController, SignalPhase, SignalProgram
from hybrid_v2x_rl.mobility.grid_simulator import (
    GridMobilitySimulator,
    GridMobilitySpec,
    MobilityModelError,
    TurnProbabilities,
    _Vehicle,
)

# --------------------------------------------------------------------------
# network topology
# --------------------------------------------------------------------------


def test_topology_matches_the_frozen_synthetic_grid() -> None:
    """The analytic grid must reproduce the SUMO network it replaces."""

    network = GridNetwork()

    assert len(network.junctions) == 72
    assert len(network.signalized_junction_ids) == 40
    # One directed edge per segment: the streets are one-way.
    assert len(network.edges) == 126
    reversed_ids = {f"{edge.to_junction}{edge.from_junction}" for edge in network.edges}
    assert not reversed_ids & {edge.edge_id for edge in network.edges}
    # Two lanes one-way is the same total as one lane each way, so lane length
    # is unchanged and density comparisons carry across the conversion.
    assert network.total_lane_length_m == pytest.approx(37_332.0)


def test_edge_geometry_and_headings_are_consistent() -> None:
    network = GridNetwork()
    edge = network.edge("B6B5")

    assert edge.direction is Direction.SOUTH
    assert edge.length_m == pytest.approx(61.0)
    assert edge.heading_rad == pytest.approx(1.5 * math.pi)

    start = edge.position_at(0.0, lane_index=1)
    end = edge.position_at(edge.length_m, lane_index=1)
    # One-way lanes straddle the centreline; lane 1 is the right-hand one,
    # and right of southbound is west, so it sits west of the axis.
    assert start == pytest.approx((244.0 - 1.75, 366.0))
    # The edge and the spec must report the same lane offsets: they are
    # separate implementations, and position_at uses the edge's.
    assert edge.lane_offset_m(1) == pytest.approx(
        network.spec.lane_centre_offset_m(1)
    )
    assert end[0] == pytest.approx(start[0])
    assert end[1] - start[1] == pytest.approx(-61.0)


def test_no_street_carries_both_directions() -> None:
    """One-way removes the §4.6.3 defect at its root rather than separating it.

    Both directions used to share one centreline, so vehicles passed through
    one another and an oncoming vehicle sat exactly on the optical path between
    a tagged pair -- 94% of all measured V-VLC blockage.  Lateral separation
    fixed that; one-way means there is no oncoming traffic on a street at all.
    """

    network = GridNetwork()
    edge_ids = {edge.edge_id for edge in network.edges}

    for edge in network.edges:
        assert f"{edge.to_junction}{edge.from_junction}" not in edge_ids


def test_adjacent_streets_run_opposite_ways() -> None:
    """Alternating one-way is what makes a green wave possible.

    A two-way street can only carry one at v = 2L/(kC), which is 1.36 m/s for
    61 m blocks on a 90 s cycle.
    """

    network = GridNetwork()
    avenue_dirs = {}
    for edge in network.edges:
        if edge.direction.is_avenue:
            avenue = network.junction(edge.from_junction).avenue_index
            avenue_dirs.setdefault(avenue, set()).add(edge.direction)

    for avenue, directions in avenue_dirs.items():
        assert len(directions) == 1, f"avenue {avenue} carries {directions}"
    assert len({tuple(d) for d in avenue_dirs.values()}) == 2, "avenues must alternate"


def test_wider_lanes_spread_the_carriageway_further() -> None:
    narrow = GridNetworkSpec(lane_width_m=3.0)
    wide = GridNetworkSpec(lane_width_m=4.0)

    # One-way lanes straddle the centreline, so lane 0 sits left of it.
    assert narrow.lane_centre_offset_m(0) == pytest.approx(-1.5)
    assert wide.lane_centre_offset_m(0) == pytest.approx(-2.0)
    assert wide.carriageway_width_m > narrow.carriageway_width_m


def test_a_carriageway_wider_than_the_block_is_rejected() -> None:
    """Cross streets are 61 m apart; enough lanes leave no street left."""

    with pytest.raises(ValueError, match="carriageways are wider than the block"):
        GridNetworkSpec(lanes_per_direction=20)


def test_a_one_way_junction_offers_at_most_straight_and_one_turn() -> None:
    """The crossing street decides which way you may turn, not the driver.

    That is a defining constraint of a one-way grid and it halves the turn
    choices a two-way junction offers.
    """

    network = GridNetwork()
    for edge in network.edges:
        options = network.turn_options(edge)
        assert len(options) <= 2, f"{edge.edge_id} offers {sorted(options)}"
        assert not ({"left", "right"} <= set(options)), "cannot offer both turns"
        for target in options.values():
            assert target.direction is not _OPPOSITE_DIRECTION[edge.direction]


def test_turns_that_leave_the_grid_are_unavailable() -> None:
    network = GridNetwork()
    for edge in network.edges:
        for target in network.turn_options(edge).values():
            assert target.edge_id in {other.edge_id for other in network.edges}


def test_lane_length_scales_with_geometry() -> None:
    small = GridNetwork(GridNetworkSpec(avenues=3, cross_streets=3))
    assert len(small.junctions) == 9
    assert len(small.signalized_junction_ids) == 1


# --------------------------------------------------------------------------
# car-following
# --------------------------------------------------------------------------


def test_free_flow_converges_to_the_desired_speed() -> None:
    parameters = IDMParameters()
    speed = 0.0
    for _ in range(2000):
        speed, _ = integrate(speed, acceleration_mps2(speed, None, None, parameters), 0.05)

    assert speed == pytest.approx(parameters.desired_speed_mps, abs=1e-3)


def test_vehicle_stops_short_of_a_red_signal() -> None:
    """The stop line behaves as a stationary leader, so a queue can form."""

    parameters = IDMParameters()
    speed = parameters.desired_speed_mps
    distance = 60.0
    for _ in range(800):
        gap, leader_speed = stop_line_gap(distance)
        speed, travelled = integrate(
            speed, acceleration_mps2(speed, gap, leader_speed, parameters), 0.05
        )
        distance -= travelled

    assert speed == pytest.approx(0.0, abs=1e-6)
    assert distance > 0.0, "vehicle must not cross the stop line"
    assert distance == pytest.approx(parameters.minimum_gap_m, abs=0.1)


def test_speed_never_goes_negative_under_hard_braking() -> None:
    speed, _ = integrate(1.0, -50.0, 0.05)
    assert speed == 0.0

    _, travelled = integrate(1.0, -50.0, 0.05)
    assert travelled >= 0.0, "a braking vehicle must never move backwards"


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------


def test_signal_phases_cycle_and_are_mutually_exclusive() -> None:
    network = GridNetwork()
    controller = SignalController(network)

    for time_s in range(0, 180, 7):
        north = controller.may_enter("B5", Direction.NORTH, float(time_s))
        east = controller.may_enter("B5", Direction.EAST, float(time_s))
        assert not (north and east), "conflicting movements must never both be permitted"


def test_alternating_junctions_are_staggered() -> None:
    network = GridNetwork()
    program = SignalProgram()

    even = program.phase_at(network.junction("B5"), 0.0)
    odd = program.phase_at(network.junction("B6"), 0.0)

    assert even is SignalPhase.NORTH_SOUTH_GREEN
    assert odd is SignalPhase.EAST_WEST_GREEN
    # Half a cycle, derived rather than written out: the literal that used to
    # stand here was 45.0, which was cycle/2 of a 90 s default and silently
    # became the whole cycle when the default was corrected to match the
    # configuration.
    assert program.offset_s(network.junction("B6")) == pytest.approx(
        0.5 * program.cycle_s
    )


def test_boundary_junctions_are_unsignalized() -> None:
    network = GridNetwork()
    controller = SignalController(network)

    assert not network.junction("A0").signalized
    assert controller.may_enter("A0", Direction.NORTH, 12.3)


# --------------------------------------------------------------------------
# simulator
# --------------------------------------------------------------------------


def _run(density: float = 40.0, seed: int = 7, duration: float = 5.0, warmup: float = 30.0):
    simulator = GridMobilitySimulator()
    spec = GridMobilitySpec(
        trace_id="test",
        target_density_veh_per_lane_km=density,
        seed=seed,
        warmup_s=warmup,
        duration_s=duration,
    )
    return simulator, spec


def test_density_is_exact_and_needs_no_calibration() -> None:
    """The whole closed-loop density search disappears: N is set, not searched."""

    simulator, spec = _run(density=30.0)
    expected = round(30.0 * simulator.network.total_lane_length_m / 1000.0)

    for _, vehicles in simulator.run(spec):
        assert len(vehicles) == expected


def test_runs_are_deterministic_for_a_fixed_seed() -> None:
    def positions() -> list[tuple[str, float]]:
        simulator, spec = _run(duration=2.0, warmup=10.0)
        snapshot: list = []
        for _, vehicles in simulator.run(spec):
            snapshot = list(vehicles)
        return sorted((v.vehicle_id, round(v.offset_m, 9)) for v in snapshot)

    assert positions() == positions()


def test_different_seeds_diverge() -> None:
    def final(seed: int) -> list[float]:
        simulator, spec = _run(seed=seed, duration=2.0, warmup=10.0)
        snapshot: list = []
        for _, vehicles in simulator.run(spec):
            snapshot = list(vehicles)
        return sorted(round(v.offset_m, 6) for v in snapshot)

    assert final(1) != final(2)


def test_a_tail_straddling_a_junction_holds_the_queue_behind_it() -> None:
    """Work plan section 4.7 item 7, and the defect the published traces caught.

    A body whose front has crossed a junction still occupies the edge it came
    from until it has travelled its own length. An 11 m bus clearing the 3 m
    junction box leaves 8 m of tail behind the junction centre, which is 5 m
    *past* the stop line at ``length_m - half_box`` where a follower waits.

    The cross-junction leader search looked only down the follower's own route,
    so a body that took a different exit from the same junction was invisible.
    That produced every footprint overlap deeper than 0.5 m in the regenerated
    campaign -- 1.8 to 2.5 m against a 0.5 m declared bound, always a long
    vehicle, always same-heading, always inside a junction box.

    This asserts the constraint directly rather than waiting for it to surface
    as an overlap, because at 3% heavy vehicles it takes a 900 s trace to show
    up and a unit-length run will not find it.
    """

    simulator = GridMobilitySimulator()
    network = simulator.network
    approach = network.edge("D10D9")
    straight, turning = network.edge("D9D8"), network.edge("D9C9")

    car = next(v for v in simulator.distribution.vehicle_types if v.type_id == "passenger_car")
    bus = next(v for v in simulator.distribution.vehicle_types if v.type_id == "bus_truck")

    def footprint(vehicle: _Vehicle) -> OrientedRectangle:
        x_m, y_m = vehicle.edge.position_at(vehicle.offset_m, vehicle.lane_index)
        return rectangle_from_front(
            Point(x_m, y_m),
            vehicle.edge.heading_rad,
            vehicle.vehicle_type.length_m,
            vehicle.vehicle_type.width_m,
        )

    # The exit the bus takes is varied deliberately.  Going straight puts its
    # tail along the approach; turning swings the body across approaches it
    # never travels, and the second case is the one measured at junction C5
    # that a route-based lookup cannot see.
    for exit_edge, label in ((straight, "straight through"), (turning, "turning")):
        crossed = _Vehicle(
            vehicle_id="bus",
            vehicle_type=bus,
            route=[approach, exit_edge],
            route_index=1,
            offset_m=3.0,   # front 3 m past the junction; 8 m of tail behind it
            speed_mps=5.0,
            lane_index=0,
        )
        waiting = _Vehicle(
            vehicle_id="car",
            vehicle_type=car,
            route=[approach, straight],
            route_index=0,
            offset_m=approach.length_m - 3.0,   # exactly at the stop line
            speed_mps=0.0,
            lane_index=0,
        )

        simulator._enforce_following_distance([crossed, waiting])

        assert not rectangles_overlap(footprint(crossed), footprint(waiting)), (
            f"a follower at the stop line was left inside the body of a bus "
            f"{label} at the junction"
        )


def test_no_two_vehicles_overlap_on_the_same_edge() -> None:
    """Car following must never let a vehicle drive into the one ahead.

    SUMO logged 347 junction collisions under --collision.action warn, which
    recorded overlapping vehicles. This is the *in-lane* half of the invariant,
    checked in edge-local coordinates because that is the arithmetic
    `car_following.py` actually performs. The general statement lives in
    :func:`test_vehicle_footprints_barely_overlap_at_generated_densities`.
    """

    simulator, spec = _run(density=60.0, duration=5.0, warmup=30.0)

    for _, vehicles in simulator.run(spec):
        by_lane: dict[tuple[str, int], list[tuple[float, float]]] = {}
        for vehicle in vehicles:
            by_lane.setdefault(vehicle.lane_key, []).append(
                (vehicle.offset_m, vehicle.vehicle_type.length_m)
            )
        for occupants in by_lane.values():
            occupants.sort()
            for (back, _), (front, front_length) in zip(occupants, occupants[1:], strict=False):
                assert front - back >= front_length - 1e-6, "vehicle footprints overlap"


#: Declared residual tolerance for footprint overlap; work plan section 4.7
#: item 7 records how it was reached.  Junction arbitration removed 99.9% of the
#: overlaps the shared-centreline and point-junction models produced; what
#: remains is a rare same-step race between two crossings, bounded in rate and
#: in depth.  Zero is the intent, not the achievement, so the achievement is
#: asserted rather than assumed.
#:
#: Measured on the generated traces, 300 sampled frames each: 0.093, 0.130 and
#: 0.097 pairs per frame at 10, 20 and 30 veh/lane-km, deepest 0.42 m.  An
#: earlier reading of 0.00/0.01/0.02 came from one seed run directly against
#: the simulator and was optimistic by roughly ten times, so the bound is set
#: from the traces rather than from this test's own configuration.
#:
#: The rate bound carries headroom because a 20-frame window is noisy and this
#: test runs one seed; the depth bound does not, because depth is a property of
#: the collision rather than of the sample.
MAX_OVERLAPS_PER_FRAME = 0.30
MAX_PENETRATION_M = 0.5


def _overlapping_pairs(simulator, spec) -> tuple[int, int, float]:
    """Return ``(frames, overlapping_pairs, deepest_penetration_m)``."""

    def circumscribed_radius(record) -> float:
        return 0.5 * math.hypot(record.length_m, record.width_m)

    frames = pairs = 0
    deepest = 0.0
    for time_s, vehicles in simulator.run(spec):
        records = list(simulator.vehicle_records(spec, time_s, vehicles))
        index = SpatialIndex.build(records)
        rectangles = {record.vehicle_id: vehicle_rectangle(record) for record in records}
        widest = max(circumscribed_radius(record) for record in records)

        seen: set[tuple[str, ...]] = set()
        for record in records:
            centre = rectangles[record.vehicle_id].centre
            # A body overlapping this one must share a point within its own
            # circumscribed radius of this centre, so it is bucketed in a cell
            # the query below reaches. The extra `widest` is slack, not need.
            reach = circumscribed_radius(record) + widest
            for other in index.candidates(Segment(centre, centre), margin_m=reach):
                if other.vehicle_id == record.vehicle_id:
                    continue
                key = tuple(sorted((record.vehicle_id, other.vehicle_id)))
                if key in seen:
                    continue
                seen.add(key)
                first, second = rectangles[key[0]], rectangles[key[1]]
                if rectangles_overlap(first, second):
                    pairs += 1
                    deepest = max(deepest, rectangle_penetration_m(first, second))
        frames += 1

    return frames, pairs, deepest


@pytest.mark.parametrize("density", [10.0, 20.0, 30.0])
def test_vehicle_footprints_barely_overlap_at_generated_densities(density: float) -> None:
    """Work plan section 16.3, in world coordinates and over *all* pairs.

    The same-edge test above compares vehicles sharing an ``edge_id``, and a
    street's two directions carry different ids, ``E7F7`` against ``F7E7``, so
    an opposing pair was never a candidate. Under the shared centreline of work
    plan section 4.6.3 that let roughly 74 pairs per frame occupy the same
    ground, head to head, while every mobility test passed.

    This assertion makes no assumption about lanes, headings or edge identity:
    it asks only whether any body is inside any other, so it keeps holding when
    those change. A wider vehicle class, a narrower lane, a second lane per
    direction with the offset miscomputed, or lane changing would each dissolve
    the current clearance margin silently.

    It runs at the densities traces are actually generated at. Above 30
    veh/lane-km the model is past capacity and measures queueing rather than
    communication (section 4.5.1), so its junction behaviour there does not
    reach a recorded frame.

    Deliberately uses the geometry engine rather than reimplementing the
    predicate, since geometry is the consumer an overlap actually breaks.
    """

    # The production warm-up, because the residual is sensitive to how settled
    # the network is: at 120 s it is several times higher than at the 300 s
    # traces are generated with, so a cheaper warm-up would measure a regime
    # that never reaches a trace.
    simulator, spec = _run(density=density, duration=1.0, warmup=300.0)
    frames, pairs, deepest = _overlapping_pairs(simulator, spec)

    assert frames > 0, "the run produced no frames to check"
    assert pairs <= MAX_OVERLAPS_PER_FRAME * frames, (
        f"{pairs} overlapping pairs over {frames} frames at rho={density:g} "
        f"exceeds the declared bound of {MAX_OVERLAPS_PER_FRAME}/frame"
    )
    assert deepest < MAX_PENETRATION_M, (
        f"deepest penetration {deepest:.3f} m at rho={density:g} exceeds "
        f"the declared bound of {MAX_PENETRATION_M} m"
    )


def test_vehicles_queue_near_intersections() -> None:
    """Hypothesis H3 needs bunching at intersection approaches to exist."""

    simulator, spec = _run(density=40.0, duration=5.0, warmup=120.0)
    near = [0, 0]
    far = [0, 0]

    for _, vehicles in simulator.run(spec):
        for vehicle in vehicles:
            distance = vehicle.edge.length_m - vehicle.offset_m
            bucket = near if distance <= 40.0 else far
            bucket[0] += 1
            bucket[1] += vehicle.speed_mps <= 0.1

    near_fraction = near[1] / near[0]
    far_fraction = far[1] / far[0]
    assert near_fraction > far_fraction, "queues must concentrate near junctions"


def test_trace_records_carry_the_fields_downstream_modules_need() -> None:
    simulator, spec = _run(duration=1.0, warmup=10.0)

    for time_s, vehicles in simulator.run(spec):
        records = list(simulator.vehicle_records(spec, time_s, vehicles))
        assert len(records) == len(vehicles)
        sample = records[0]
        # Positions, headings and dimensions are what VLC occlusion needs.
        assert math.isfinite(sample.x_m) and math.isfinite(sample.y_m)
        assert 0.0 <= sample.heading_rad < 2.0 * math.pi
        assert sample.length_m > 0.0 and sample.width_m > 0.0 and sample.height_m > 0.0
        assert sample.trace_id == "test"
        break


def test_planned_route_is_exposed_as_an_edge_sequence() -> None:
    """The SUMO backend could not do this, which is why pair extraction failed."""

    simulator, spec = _run(duration=1.0, warmup=5.0)

    for _, vehicles in simulator.run(spec):
        route = vehicles[0].planned_route_ids
        assert len(route) >= 1
        assert all(isinstance(edge_id, str) and edge_id for edge_id in route)
        break


def test_signal_records_cover_every_signalized_junction() -> None:
    simulator, spec = _run(duration=1.0, warmup=5.0)
    records = list(simulator.signal_records(spec, 0.0))

    assert len(records) == len(simulator.network.signalized_junction_ids)
    assert {record.signal_id for record in records} == set(
        simulator.network.signalized_junction_ids
    )


def test_zero_density_is_rejected() -> None:
    simulator = GridMobilitySimulator()
    spec = GridMobilitySpec(
        trace_id="t",
        target_density_veh_per_lane_km=1e-9,
        seed=1,
        warmup_s=0.0,
        duration_s=0.1,
    )
    with pytest.raises(MobilityModelError, match="fewer than one vehicle"):
        next(iter(simulator.run(spec)))


def test_turn_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        TurnProbabilities(left=0.5, straight=0.5, right=0.5)


def test_initial_population_matches_the_target_exactly() -> None:
    """The initial state must not start short of the requested density.

    An earlier implementation dropped surplus vehicles when an edge's spacing
    was too tight, so a run began roughly 2% below target before any traffic
    dynamics occurred.
    """

    for density in (20.0, 40.0, 60.0):
        simulator, spec = _run(density=density, duration=1.0, warmup=0.0)
        expected = round(density * simulator.network.total_lane_length_m / 1000.0)
        _, vehicles = next(iter(simulator.run(spec)))
        assert len(vehicles) == expected


def test_realized_density_meets_the_gate_one_tolerance() -> None:
    """Gate 1 requires realised density within 5% of target.

    At the highest density every boundary entry saturates, so injections are
    deferred rather than placed overlapping and realised density settles a few
    percent low.  That is a property of congested boundary inflow, not a defect,
    and it must stay inside tolerance.
    """

    simulator, spec = _run(density=60.0, duration=10.0, warmup=120.0)
    lane_km = simulator.network.total_lane_length_m / 1000.0

    samples = [len(vehicles) / lane_km for _, vehicles in simulator.run(spec)]
    mean_density = sum(samples) / len(samples)

    assert abs(mean_density - 60.0) / 60.0 <= 0.05


def test_density_beyond_holding_capacity_is_rejected() -> None:
    """An impossible density must fail loudly rather than silently under-fill."""

    simulator = GridMobilitySimulator()
    spec = GridMobilitySpec(
        trace_id="t",
        target_density_veh_per_lane_km=500.0,
        seed=1,
        warmup_s=0.0,
        duration_s=0.1,
    )
    with pytest.raises(MobilityModelError, match="holding capacity"):
        next(iter(simulator.run(spec)))


# --------------------------------------------------------------------------
# multi-lane geometry (work plan §4.9 stage 1)
# --------------------------------------------------------------------------


def test_two_way_lanes_stay_to_one_side_of_the_centreline() -> None:
    """Kept for comparison: two-way must never let directions share a line."""

    spec = GridNetworkSpec(lanes_per_direction=1, one_way=False)
    assert spec.lane_centre_offset_m(0) == pytest.approx(1.75)


def test_one_way_lanes_straddle_the_centreline() -> None:
    """The whole carriageway runs one way, so the lanes spread across it."""

    spec = GridNetworkSpec(lanes_per_direction=3)
    offsets = [spec.lane_centre_offset_m(index) for index in range(3)]

    assert offsets == pytest.approx([-3.5, 0.0, 3.5])
    assert spec.carriageway_width_m == pytest.approx(3 * 3.5)


def test_adjacent_lanes_clear_a_vehicle_width() -> None:
    """Two vehicles side by side must not overlap."""

    edge = GridNetwork().edge("B6B5")
    inner = edge.position_at(30.0, lane_index=0)
    outer = edge.position_at(30.0, lane_index=1)

    separation = math.hypot(outer[0] - inner[0], outer[1] - inner[1])
    assert separation == pytest.approx(3.5)
    assert separation > 2.5, "wider than the widest configured body"


def test_the_junction_box_reaches_the_outermost_crossing_lane() -> None:
    """A stop line must clear the kerb-side lane, not the carriageway centre."""

    two_lane = GridNetworkSpec(lanes_per_direction=2)
    four_lane = GridNetworkSpec(lanes_per_direction=4)

    assert two_lane.outermost_lane_centre_offset_m == pytest.approx(1.75)
    assert four_lane.outermost_lane_centre_offset_m == pytest.approx(5.25)


def test_a_lane_index_outside_the_carriageway_is_rejected() -> None:
    spec = GridNetworkSpec()
    edge = GridNetwork(spec).edge("B6B5")

    for bad in (-1, 2, 99):
        with pytest.raises(ValueError, match="lane_index"):
            spec.lane_centre_offset_m(bad)
        with pytest.raises(ValueError, match="lane_index"):
            edge.position_at(10.0, lane_index=bad)


# --------------------------------------------------------------------------
# lane changing (work plan §4.9 stage 1)
# --------------------------------------------------------------------------


def test_safety_vetoes_a_change_however_much_it_gains() -> None:
    """Braking harder than the safe bound refuses the change outright.

    Safety is never traded against incentive: it is checked first and returns.
    """

    parameters = MOBILParameters(safe_deceleration_mps2=4.0)
    assert not mobil_accepts(
        own_before=0.0, own_after=10.0,          # enormous gain
        new_follower_before=0.0, new_follower_after=-9.0,   # unacceptable braking
        old_follower_before=0.0, old_follower_after=0.0,
        parameters=parameters,
    )


def test_a_change_needs_to_clear_the_threshold() -> None:
    parameters = MOBILParameters(politeness=0.0, threshold_mps2=0.2)
    common = dict(
        new_follower_before=0.0, new_follower_after=0.0,
        old_follower_before=0.0, old_follower_after=0.0,
        parameters=parameters,
    )

    assert not mobil_accepts(own_before=0.0, own_after=0.1, **common)
    assert mobil_accepts(own_before=0.0, own_after=0.5, **common)


def test_politeness_lets_others_veto_a_selfish_gain() -> None:
    """A vehicle must not buy a small gain with a large loss imposed on others."""

    selfish = MOBILParameters(politeness=0.0, threshold_mps2=0.1)
    polite = MOBILParameters(politeness=1.0, threshold_mps2=0.1)
    case = dict(
        own_before=0.0, own_after=0.5,
        new_follower_before=0.0, new_follower_after=-2.0,
        old_follower_before=0.0, old_follower_after=0.0,
    )

    assert mobil_accepts(**case, parameters=selfish)
    assert not mobil_accepts(**case, parameters=polite)


def test_the_keep_right_bias_favours_moving_towards_the_kerb() -> None:
    parameters = MOBILParameters(politeness=0.0, threshold_mps2=0.2)
    neutral = dict(
        own_before=0.0, own_after=0.0,
        new_follower_before=0.0, new_follower_after=0.0,
        old_follower_before=0.0, old_follower_after=0.0,
        parameters=parameters,
    )

    assert mobil_accepts(**neutral, bias_mps2=0.3), "outward: bias alone suffices"
    assert not mobil_accepts(**neutral, bias_mps2=-0.3), "inward: bias opposes"


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_an_unusable_politeness_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="politeness"):
        MOBILParameters(politeness=bad)


def test_a_zero_manoeuvre_duration_is_rejected() -> None:
    """Without a duration the criterion re-fires at 20 Hz and vehicles oscillate."""

    with pytest.raises(ValueError, match="manoeuvre_duration_s"):
        MOBILParameters(manoeuvre_duration_s=0.0)


def test_a_single_lane_road_produces_no_lane_changes() -> None:
    simulator = GridMobilitySimulator(GridNetwork(GridNetworkSpec(lanes_per_direction=1)))
    spec = GridMobilitySpec(
        trace_id="single", target_density_veh_per_lane_km=20.0,
        seed=7, warmup_s=60.0, duration_s=1.0,
    )
    seen: dict[tuple[str, str], int] = {}
    changes = 0

    for _, vehicles in simulator.run(spec):
        for vehicle in vehicles:
            key = (vehicle.vehicle_id, vehicle.edge.edge_id)
            if key in seen and seen[key] != vehicle.lane_index:
                changes += 1
            seen[key] = vehicle.lane_index

    assert changes == 0


_OPPOSITE_DIRECTION = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


def test_two_lanes_produce_changes_at_a_plausible_rate() -> None:
    """The blockage that is *not* derivable from a map (§4.9).

    Rate matters as much as existence: without the manoeuvre duration the
    criterion re-fires every 50 ms and yields 937 changes per vehicle-hour,
    which is a vehicle swapping lanes every four seconds.
    """

    simulator = GridMobilitySimulator(GridNetwork())
    spec = GridMobilitySpec(
        trace_id="lanes", target_density_veh_per_lane_km=20.0,
        seed=7, warmup_s=60.0, duration_s=5.0,
    )

    seen: dict[tuple[str, str], int] = {}
    changes = frames = population = 0
    for _, vehicles in simulator.run(spec):
        for vehicle in vehicles:
            key = (vehicle.vehicle_id, vehicle.edge.edge_id)
            if key in seen and seen[key] != vehicle.lane_index:
                changes += 1
            seen[key] = vehicle.lane_index
        frames += 1
        population += len(vehicles)

    vehicle_hours = (population / frames) * (frames * spec.step_s) / 3600.0
    per_vehicle_hour = changes / vehicle_hours

    assert changes > 0, "two lanes must produce overtaking"
    assert 5.0 < per_vehicle_hour < 250.0, f"{per_vehicle_hour:.0f}/veh-hour is not plausible"


def test_both_lanes_are_used() -> None:
    simulator = GridMobilitySimulator(GridNetwork())
    spec = GridMobilitySpec(
        trace_id="lanes", target_density_veh_per_lane_km=20.0,
        seed=7, warmup_s=60.0, duration_s=1.0,
    )

    occupancy = [0, 0]
    for _, vehicles in simulator.run(spec):
        for vehicle in vehicles:
            occupancy[vehicle.lane_index] += 1

    share = occupancy[0] / sum(occupancy)
    assert 0.3 < share < 0.7, f"lane use is lopsided: {occupancy}"
