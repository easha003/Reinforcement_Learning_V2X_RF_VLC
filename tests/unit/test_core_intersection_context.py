from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.core.geometry import Point, Segment
from hybrid_v2x_rl.core.intersection_context import (
    JunctionGrid,
    intersection_context,
)
from hybrid_v2x_rl.core.link_endpoints import optical_link_path
from hybrid_v2x_rl.geometry.building_geometry import LANE_WIDTH_M

EAST = 0.0
NORTH = 0.5 * math.pi
WEST = math.pi
SOUTH = 1.5 * math.pi

AVENUE_SPACING_M = 244.0
CROSS_STREET_SPACING_M = 61.0
CAR_LENGTH_M = 4.5


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = CAR_LENGTH_M
    width_m: float = 1.8
    height_m: float = 1.5


def headline_grid() -> JunctionGrid:
    return JunctionGrid.from_grid(
        avenues=6,
        cross_streets=12,
        avenue_spacing_m=AVENUE_SPACING_M,
        cross_street_spacing_m=CROSS_STREET_SPACING_M,
        road_half_width_m=LANE_WIDTH_M,
    )


def context_for(tx: FakeVehicle, rx: FakeVehicle):
    grid = headline_grid()
    return intersection_context(
        tx.x_m,
        tx.y_m,
        tx.heading_rad,
        rx.x_m,
        rx.y_m,
        rx.heading_rad,
        optical_link_path(tx, rx).segment,
        grid,
    )


# -- the lattice --------------------------------------------------------------


def test_headline_grid_has_seventy_two_junctions() -> None:
    assert headline_grid().junction_count == 72


def test_a_junction_box_is_the_overlap_of_two_carriageways() -> None:
    box = headline_grid().box(1, 3)

    assert box.centre.x_m == pytest.approx(AVENUE_SPACING_M)
    assert box.centre.y_m == pytest.approx(3 * CROSS_STREET_SPACING_M)
    assert box.length_m == pytest.approx(2 * LANE_WIDTH_M)
    assert box.width_m == pytest.approx(2 * LANE_WIDTH_M)


# -- distance to the next junction -------------------------------------------


@pytest.mark.parametrize(
    ("y_m", "expected_m"),
    [
        (0.0, 0.0),  # already on the crossing line
        (1.0, CROSS_STREET_SPACING_M - 1.0),
        (60.0, 1.0),
        (CROSS_STREET_SPACING_M, 0.0),  # on the next one
        (CROSS_STREET_SPACING_M + 5.0, CROSS_STREET_SPACING_M - 5.0),
    ],
)
def test_northbound_distance_counts_down_to_the_next_cross_street(
    y_m: float, expected_m: float
) -> None:
    grid = headline_grid()
    assert grid.distance_ahead_to_junction_m(0.0, y_m, NORTH) == pytest.approx(expected_m)


def test_southbound_distance_counts_down_the_other_way() -> None:
    grid = headline_grid()
    assert grid.distance_ahead_to_junction_m(0.0, 1.0, SOUTH) == pytest.approx(1.0)
    assert grid.distance_ahead_to_junction_m(0.0, 60.0, SOUTH) == pytest.approx(60.0)


def test_travel_along_an_avenue_uses_the_avenue_pitch() -> None:
    """Eastbound traffic crosses avenues, 244 m apart, not cross streets."""

    grid = headline_grid()
    assert grid.distance_ahead_to_junction_m(10.0, 0.0, EAST) == pytest.approx(
        AVENUE_SPACING_M - 10.0
    )
    assert grid.distance_ahead_to_junction_m(10.0, 0.0, WEST) == pytest.approx(10.0)


# -- what the forecast actually keys on ---------------------------------------


def test_a_pair_between_junctions_spans_nothing() -> None:
    """Mid-block, cross-traffic has no way to reach the optical path."""

    follower = FakeVehicle("tx", 0.0, 20.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 45.0, NORTH)

    context = context_for(follower, leader)
    assert not context.spans_junction
    assert context.spanned_junction_ids == ()


def test_a_pair_straddling_a_junction_spans_it() -> None:
    """The configuration that lets a crossing vehicle cut the beam."""

    follower = FakeVehicle("tx", 0.0, CROSS_STREET_SPACING_M - 10.0, NORTH)
    leader = FakeVehicle("rx", 0.0, CROSS_STREET_SPACING_M + 15.0, NORTH)

    context = context_for(follower, leader)
    assert context.spans_junction
    assert context.spanned_junction_ids == ("A1",)
    assert context.spanned_junction_count == 1


def test_a_long_link_can_span_two_junctions() -> None:
    """Cross streets are 61 m apart, so a 40 m window reaches at most two."""

    grid = headline_grid()
    spanning = Segment(
        Point(0.0, CROSS_STREET_SPACING_M - 5.0), Point(0.0, 2 * CROSS_STREET_SPACING_M + 5.0)
    )
    found = grid.junctions_overlapping(spanning)

    assert tuple(ref.junction_id for ref in found) == ("A1", "A2")


def test_junction_identifiers_match_the_trace_convention() -> None:
    """Junction naming must line up with routes.json and network.json."""

    grid = headline_grid()
    found = grid.junctions_overlapping(
        Segment(
            Point(AVENUE_SPACING_M, 7 * CROSS_STREET_SPACING_M - 2.0),
            Point(AVENUE_SPACING_M, 7 * CROSS_STREET_SPACING_M + 2.0),
        )
    )
    assert tuple(ref.junction_id for ref in found) == ("B7",)


def test_distance_to_junction_is_reported_for_both_endpoints() -> None:
    follower = FakeVehicle("tx", 0.0, 40.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 55.0, NORTH)

    context = context_for(follower, leader)
    assert context.transmitter_distance_to_junction_m == pytest.approx(21.0)
    assert context.receiver_distance_to_junction_m == pytest.approx(6.0)
    assert context.nearest_distance_to_junction_m == pytest.approx(6.0)


def test_a_pair_just_short_of_a_junction_does_not_span_it_yet() -> None:
    """The near-miss the forecast has to distinguish from a hit.

    The optical path ends at the leader's rear, so a leader whose *front* has
    entered the junction may still have the path stop short of it.
    """

    follower = FakeVehicle("tx", 0.0, 30.0, NORTH)
    leader = FakeVehicle("rx", 0.0, CROSS_STREET_SPACING_M - LANE_WIDTH_M - 0.5, NORTH)

    context = context_for(follower, leader)
    assert not context.spans_junction
    assert context.receiver_distance_to_junction_m < 5.0


# -- the leakage boundary -----------------------------------------------------


def test_intersection_context_reads_no_blocker_state() -> None:
    """These features may enter the policy observation; occlusion truth may not.

    Everything here derives from own pose plus a road map, which a real vehicle
    has.  If this module could see other vehicles' footprints it would become
    hidden state and could not be observed, which would remove the forecast's
    only legitimate input.
    """

    module = importlib.import_module("hybrid_v2x_rl.core.intersection_context")

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as stream:
        text = stream.read()

    for forbidden in ("vehicle_occlusion", "hybrid_v2x_rl.channels", "pathloss"):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text
