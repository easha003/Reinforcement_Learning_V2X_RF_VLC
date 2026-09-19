from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.core.geometry import (
    OrientedRectangle,
    Point,
    Segment,
    rectangle_from_front,
    segment_intersects_rectangle,
    vehicle_rear,
    vehicle_rectangle,
)

EAST = 0.0
NORTH = 0.5 * math.pi


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    """Minimal stand-in for a trace record's footprint fields."""

    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = 4.5
    width_m: float = 1.8


def vertical_segment(x_m: float, half_span_m: float = 5.0) -> Segment:
    return Segment(Point(x_m, -half_span_m), Point(x_m, half_span_m))


# -- the position convention -------------------------------------------------


def test_vehicle_rectangle_places_the_stored_position_at_the_front_bumper() -> None:
    """The trace stores the front, not the centre.

    ``GridEdge.position_at(offset_m)`` is evaluated at the vehicle's
    ``offset_m``, which the car-following gap arithmetic defines as the front:
    ``gap = leader.offset_m - follower.offset_m - leader.length_m``.  Treating
    the stored point as a centre displaces the footprint forward by half a
    vehicle length.
    """

    vehicle = FakeVehicle(x_m=10.0, y_m=0.0, heading_rad=EAST)
    rectangle = vehicle_rectangle(vehicle)

    assert rectangle.centre.x_m == pytest.approx(7.75)
    assert rectangle.centre.y_m == pytest.approx(0.0)
    # The stored position is on the leading edge, not inside the body's middle.
    assert rectangle.contains(Point(10.0, 0.0))
    assert rectangle.contains(Point(5.5, 0.0))
    assert not rectangle.contains(Point(10.5, 0.0))


def test_footprint_occupies_the_road_behind_the_stored_position() -> None:
    """Regression: the body must extend backwards, never forwards.

    A segment at x = 6 m crosses a vehicle whose front is at x = 10 m.  If the
    stored position were mistaken for the centre the body would span
    7.75-12.25 m and this segment would be reported clear -- silently
    under-reporting blockage everywhere.
    """

    rectangle = vehicle_rectangle(FakeVehicle(x_m=10.0, y_m=0.0, heading_rad=EAST))

    assert segment_intersects_rectangle(vertical_segment(6.0), rectangle)
    assert segment_intersects_rectangle(vertical_segment(9.9), rectangle)
    assert not segment_intersects_rectangle(vertical_segment(11.0), rectangle)


def test_vehicle_rear_is_one_full_length_behind_the_stored_position() -> None:
    """The photodiode faces rearward, so a link ends here on the receiver."""

    rear = vehicle_rear(FakeVehicle(x_m=10.0, y_m=0.0, heading_rad=EAST))
    assert rear.x_m == pytest.approx(5.5)
    assert rear.y_m == pytest.approx(0.0)

    heading_north = vehicle_rear(FakeVehicle(x_m=0.0, y_m=10.0, heading_rad=NORTH))
    assert heading_north.x_m == pytest.approx(0.0)
    assert heading_north.y_m == pytest.approx(5.5)


# -- the cross-traffic case, which dominates real blockage -------------------


def test_perpendicular_crosser_blocks_across_its_length_not_its_width() -> None:
    """Regression on the approximation both earlier probes used.

    98.6-99.5% of measured optical blockage is cross-traffic.  A vehicle
    crossing perpendicular presents its 4.5 m *length* to the path, not its
    1.8 m width, so a point-plus-half-width test rejects blockers that in fact
    obstruct.  Here the crosser's centre is 2.0 m off the path: beyond half its
    width, well inside half its length.
    """

    crosser = rectangle_from_front(Point(4.25, 0.0), heading_rad=EAST, length_m=4.5, width_m=1.8)
    assert crosser.centre.x_m == pytest.approx(2.0)

    optical_path = vertical_segment(0.0)
    assert segment_intersects_rectangle(optical_path, crosser)

    # The discarded shortcut, stated explicitly so the difference is visible.
    lateral_offset_m = abs(crosser.centre.x_m)
    assert lateral_offset_m > 0.5 * crosser.width_m
    assert lateral_offset_m < 0.5 * crosser.length_m


def test_a_crosser_clear_of_the_path_does_not_block() -> None:
    crosser = rectangle_from_front(Point(7.0, 0.0), heading_rad=EAST, length_m=4.5, width_m=1.8)
    assert not segment_intersects_rectangle(vertical_segment(0.0), crosser)


# -- section 5.1 required cases ----------------------------------------------


def test_boundary_contact_counts_as_intersection() -> None:
    """A grazing segment reports blocked; see the module's stated convention."""

    rectangle = OrientedRectangle(Point(0.0, 0.0), EAST, length_m=4.0, width_m=2.0)

    # Tangent along the full length of the +y face.
    assert segment_intersects_rectangle(Segment(Point(-3.0, 1.0), Point(3.0, 1.0)), rectangle)
    # Touching exactly one corner.
    assert segment_intersects_rectangle(Segment(Point(2.0, 3.0), Point(2.0, 1.0)), rectangle)
    # A hair outside the same face is clear.
    assert not segment_intersects_rectangle(
        Segment(Point(-3.0, 1.001), Point(3.0, 1.001)), rectangle
    )


def test_partial_intersection_with_an_endpoint_inside() -> None:
    rectangle = OrientedRectangle(Point(0.0, 0.0), EAST, length_m=4.0, width_m=2.0)

    assert segment_intersects_rectangle(Segment(Point(0.0, 0.0), Point(10.0, 0.0)), rectangle)
    assert segment_intersects_rectangle(Segment(Point(-10.0, 0.0), Point(0.5, 0.5)), rectangle)
    # Fully contained.
    assert segment_intersects_rectangle(Segment(Point(-1.0, -0.5), Point(1.0, 0.5)), rectangle)


def test_heading_rotation_changes_the_verdict() -> None:
    """The same segment and centre, blocked or clear depending on rotation."""

    segment = Segment(Point(-5.0, 1.4), Point(5.0, 1.4))
    centre = Point(0.0, 0.0)

    # Long axis east-west: the body reaches only 1.0 m across, so it is clear.
    aligned = OrientedRectangle(centre, EAST, length_m=4.0, width_m=2.0)
    assert not segment_intersects_rectangle(segment, aligned)

    # Rotated a quarter turn, the 4 m length now spans y and it blocks.
    rotated = OrientedRectangle(centre, NORTH, length_m=4.0, width_m=2.0)
    assert segment_intersects_rectangle(segment, rotated)


def test_diagonal_heading_is_handled_exactly() -> None:
    rectangle = OrientedRectangle(Point(0.0, 0.0), 0.25 * math.pi, length_m=4.0, width_m=2.0)
    # Along the diagonal the body reaches 2.0 m from the centre.
    assert rectangle.contains(Point(1.35, 1.35))
    assert not rectangle.contains(Point(1.5, 1.5))


# -- degenerate and parallel cases -------------------------------------------


def test_parallel_segment_outside_the_slab_is_clear() -> None:
    rectangle = OrientedRectangle(Point(0.0, 0.0), EAST, length_m=4.0, width_m=2.0)
    assert not segment_intersects_rectangle(Segment(Point(-5.0, 5.0), Point(5.0, 5.0)), rectangle)


def test_degenerate_segment_behaves_as_a_point_test() -> None:
    rectangle = OrientedRectangle(Point(0.0, 0.0), EAST, length_m=4.0, width_m=2.0)
    inside = Point(1.0, 0.5)
    outside = Point(5.0, 0.0)

    assert Segment(inside, inside).is_degenerate
    assert segment_intersects_rectangle(Segment(inside, inside), rectangle)
    assert not segment_intersects_rectangle(Segment(outside, outside), rectangle)


def test_segment_short_of_the_rectangle_does_not_intersect() -> None:
    """Intersection is bounded by the segment, not by its infinite line."""

    rectangle = OrientedRectangle(Point(10.0, 0.0), EAST, length_m=4.0, width_m=2.0)
    assert not segment_intersects_rectangle(Segment(Point(-5.0, 0.0), Point(0.0, 0.0)), rectangle)


# -- structure ---------------------------------------------------------------


def test_corners_are_ordered_from_the_leading_face() -> None:
    rectangle = vehicle_rectangle(FakeVehicle(x_m=10.0, y_m=0.0, heading_rad=EAST))
    front_left, front_right, rear_right, rear_left = rectangle.corners()

    assert front_left.x_m == pytest.approx(10.0)
    assert front_left.y_m == pytest.approx(0.9)
    assert front_right.x_m == pytest.approx(10.0)
    assert front_right.y_m == pytest.approx(-0.9)
    assert rear_right.x_m == pytest.approx(5.5)
    assert rear_left.x_m == pytest.approx(5.5)


def test_to_local_round_trips_the_centre_to_the_origin() -> None:
    rectangle = OrientedRectangle(Point(3.0, -7.0), 1.1, length_m=4.0, width_m=2.0)
    local = rectangle.to_local(rectangle.centre)
    assert local.x_m == pytest.approx(0.0)
    assert local.y_m == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("length_m", "width_m", "heading_rad"),
    [(0.0, 2.0, 0.0), (-1.0, 2.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, math.nan)],
)
def test_degenerate_rectangles_are_rejected(
    length_m: float, width_m: float, heading_rad: float
) -> None:
    with pytest.raises(ValueError):
        OrientedRectangle(Point(0.0, 0.0), heading_rad, length_m, width_m)
