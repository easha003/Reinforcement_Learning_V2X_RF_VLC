"""Junction admission: would two crossings put bodies on the same ground?

Written after three rounds of tightening a hand-made movement matrix, then a
fourth round replacing that matrix with centrelines inflated by a margin.  Each
round fixed something real and left something else, because all four were
*proxies* for the question the footprint invariant actually asks -- do two
bodies occupy the same ground -- and a proxy keeps diverging from it whenever
the layout changes underneath.

The rule no longer approximates.  It samples the real body along the curve it
follows and asks :func:`hybrid_v2x_rl.core.geometry.rectangles_overlap`, which is the
same function the mobility acceptance invariant applies to generated traces.
Admission and acceptance can no longer disagree, because there is nothing left
to disagree about.

These tests pin the geometry against the real network rather than against
synthetic segments, so they keep testing the layout actually in force.
"""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.mobility.grid_network import GridNetwork
from hybrid_v2x_rl.mobility.grid_simulator import (
    GridMobilitySimulator,
    GridMobilitySpec,
    _bodies_conflict,
    _swept_bodies,
)
from hybrid_v2x_rl.mobility.vehicle_types import headline_vehicle_distribution

#: An interior junction, so both approaches and both exits exist.
JUNCTION = "D9"


@pytest.fixture(scope="module")
def network() -> GridNetwork:
    return GridNetwork()


@pytest.fixture(scope="module")
def half_box() -> float:
    return GridMobilitySimulator()._junction_half_box_m


def vehicle(type_id: str):
    types = headline_vehicle_distribution().vehicle_types
    return next(v for v in types if v.type_id == type_id)


def crossing(
    network: GridNetwork,
    half_box: float,
    approach_direction: str,
    lane_in: int,
    exit_direction: str,
    lane_out: int,
    type_id: str = "passenger_car",
):
    """Bodies swept by one movement through :data:`JUNCTION`."""

    approach = next(
        e
        for e in network.edges
        if e.to_junction == JUNCTION and e.direction.name == approach_direction
    )
    exit_edge = next(
        e
        for e in network.edges
        if e.from_junction == JUNCTION and e.direction.name == exit_direction
    )
    return _swept_bodies(
        approach, lane_in, exit_edge, lane_out, vehicle(type_id), half_box
    )


# -- what a swept path is -----------------------------------------------------


def test_a_straight_crossing_spans_the_whole_box(network, half_box) -> None:
    """Entry to exit covers both sides of the junction centre.

    A path that stopped at the centre would let a following vehicle be granted
    ground the leader's tail still occupies.
    """

    bodies = crossing(network, half_box, "SOUTH", 0, "SOUTH", 0)
    first, last = bodies[0].centre, bodies[-1].centre
    assert math.hypot(last.x_m - first.x_m, last.y_m - first.y_m) >= 2.0 * half_box


def test_a_straight_crossing_keeps_one_heading(network, half_box) -> None:
    bodies = crossing(network, half_box, "SOUTH", 0, "SOUTH", 0)
    headings = {round(body.heading_rad, 9) for body in bodies}
    assert len(headings) == 1


def test_a_turn_rotates_the_body_through_a_right_angle(network, half_box) -> None:
    """The body follows the curve, which is the whole point of the rewrite.

    Modelling a turn as a straight chord left the body pointing the wrong way
    for the entire crossing, and the error was papered over by inflating the
    body's half-width by half its length -- 3.15 m against a 3.5 m lane
    spacing, which serialised every adjacent pair whenever either turned.
    """

    bodies = crossing(network, half_box, "SOUTH", 1, "WEST", 1)
    swing = bodies[-1].heading_rad - bodies[0].heading_rad
    # Headings come from atan2, so wrap to the shortest signed angle before
    # comparing: a right turn otherwise reads as 270 degrees of left turn.
    turned = abs((swing + math.pi) % (2.0 * math.pi) - math.pi)
    assert turned == pytest.approx(0.5 * math.pi, abs=math.radians(2.0))


def test_a_turn_is_sampled_finely_enough_to_be_continuous(network, half_box) -> None:
    """Consecutive poses must overlap, or a body could slip through the gap."""

    bodies = crossing(network, half_box, "SOUTH", 0, "WEST", 0)
    assert len(bodies) >= 5
    for earlier, later in zip(bodies, bodies[1:], strict=False):
        step = math.hypot(
            later.centre.x_m - earlier.centre.x_m,
            later.centre.y_m - earlier.centre.y_m,
        )
        assert step < earlier.length_m


# -- what conflicts, and what does not ----------------------------------------


def test_adjacent_lanes_running_straight_are_compatible(network, half_box) -> None:
    """The case the inflated-margin rule got wrong, and the reason for the rewrite.

    Two cars abreast in 3.5 m lanes leave 1.7 m between their flanks, which the
    0.5 m of junction clearance does not close.
    """

    assert not _bodies_conflict(
        crossing(network, half_box, "SOUTH", 0, "SOUTH", 0),
        crossing(network, half_box, "SOUTH", 1, "SOUTH", 1),
    )


def test_a_bus_beside_a_car_still_clears_the_adjacent_lane(network, half_box) -> None:
    """Buses were over-represented in measured overlaps, but from the footprint
    convention rather than from width: 2.5 m against 3.5 m lanes still fits."""

    assert not _bodies_conflict(
        crossing(network, half_box, "SOUTH", 0, "SOUTH", 0, type_id="bus_truck"),
        crossing(network, half_box, "SOUTH", 1, "SOUTH", 1, type_id="bus_truck"),
    )


def test_perpendicular_crossings_conflict(network, half_box) -> None:
    """Two approaches of a one-way junction meet inside the box by construction."""

    assert _bodies_conflict(
        crossing(network, half_box, "SOUTH", 0, "SOUTH", 0),
        crossing(network, half_box, "WEST", 0, "WEST", 0),
    )


def test_one_lane_of_one_approach_is_a_queue(network, half_box) -> None:
    """Identical movements share all their ground, so the follower waits."""

    mine = crossing(network, half_box, "SOUTH", 1, "SOUTH", 1)
    assert _bodies_conflict(mine, mine)


def test_a_turn_out_of_the_inner_lane_crosses_the_outer_lane(network, half_box) -> None:
    """The case a movement matrix called compatible and measurement refuted.

    Southbound lane 0 is the left lane, so turning right out of it necessarily
    cuts across lane 1.  It was 23 of 36 measured overlaps.
    """

    assert _bodies_conflict(
        crossing(network, half_box, "SOUTH", 0, "WEST", 0),
        crossing(network, half_box, "SOUTH", 1, "SOUTH", 1),
    )


def test_a_tight_turn_sweeps_its_tail_into_the_adjacent_lane(network, half_box) -> None:
    """Measured, and the reason turning pairs are not simply waved through.

    A right turn from the outer lane into the outer lane of the crossing street
    has a radius near 1.25 m at this box size.  A rigid 4.5 m body cannot take
    that without its tail swinging wide, so the conflict is real geometry
    rather than residual conservatism -- and asking the footprint directly is
    what distinguishes the two.
    """

    assert _bodies_conflict(
        crossing(network, half_box, "SOUTH", 1, "WEST", 1),
        crossing(network, half_box, "SOUTH", 0, "SOUTH", 0),
    )


def test_conflict_is_symmetric(network, half_box) -> None:
    a = crossing(network, half_box, "SOUTH", 0, "WEST", 0)
    b = crossing(network, half_box, "WEST", 1, "SOUTH", 1)
    assert _bodies_conflict(a, b) == _bodies_conflict(b, a)


def test_an_empty_sweep_conflicts_with_nothing(network, half_box) -> None:
    assert not _bodies_conflict((), crossing(network, half_box, "SOUTH", 0, "SOUTH", 0))


# -- the simulator's own view -------------------------------------------------


def test_a_vehicle_far_from_any_junction_has_no_crossing() -> None:
    simulator = GridMobilitySimulator()
    spec = GridMobilitySpec(
        trace_id="paths",
        target_density_veh_per_lane_km=20.0,
        seed=7,
        warmup_s=60.0,
        duration_s=0.05,
    )
    frame = next(vehicles for _, vehicles in simulator.run(spec))
    assert frame
    # "A0" is a corner of the grid; nothing in a sampled frame is crossing it
    # on an edge that touches it, so the key must be absent rather than guessed.
    assert simulator._swept_key(frame[0], "A0") is None


def test_the_conflict_cache_agrees_with_recomputation(network, half_box) -> None:
    """Memoisation is what makes a footprint-level rule affordable at 20 Hz.

    A key that failed to capture some part of the geometry would return a
    cached answer for a crossing it does not describe, which is silent and
    would show up only as an overlap much later.
    """

    simulator = GridMobilitySimulator()
    keys = [
        ("D10D9", 0, "D9D8", 0, "passenger_car"),
        ("D10D9", 1, "D9C9", 1, "passenger_car"),
        ("E9D9", 0, "D9C9", 0, "bus_truck"),
    ]
    for first in keys:
        for second in keys:
            cached = simulator._keys_conflict(first, second, half_box)
            direct = _bodies_conflict(
                _swept_bodies(
                    network.edge(first[0]),
                    first[1],
                    network.edge(first[2]),
                    first[3],
                    vehicle(first[4]),
                    half_box,
                ),
                _swept_bodies(
                    network.edge(second[0]),
                    second[1],
                    network.edge(second[2]),
                    second[3],
                    vehicle(second[4]),
                    half_box,
                ),
            )
            assert cached == direct, f"{first} vs {second}"
