"""Two-dimensional primitives for obstruction geometry.

Oriented rectangles, segments, and the segment-rectangle intersection test from
which both RF propagation class and V-VLC occlusion are derived.  It is the
foundation of the geometry engine required by work plan section 5.

**Why this lives in ``core`` and not in ``hybrid_v2x_rl.geometry``.**  Everything in
that package reasons about *exact* vehicle state and is therefore oracle-only:
``hybrid_v2x_rl.observation`` may not import any of it.  This module reasons about
nothing.  It is pure convex mathematics over whatever coordinates it is handed,
with no more access to hidden state than :mod:`math` has, so the observation
layer may use it on *noisy predicted* positions without leaking anything.

The distinction matters because it is the difference between a rule and a
proxy.  "Do not import the geometry package" was standing in for "do not use
exact state", and the two diverged the moment the blockage predictor needed a
segment-rectangle test.  The choice was to weaken the guard, duplicate tested
code, or move the boundary; moving it leaves the guard absolute and exception
free.  Nothing here may import from ``hybrid_v2x_rl`` -- a test enforces it, because
one convenience import would quietly re-couple the layers.

Two conventions are fixed here rather than left to callers, because getting
either wrong displaces every downstream measurement without failing loudly.

**Vehicle position is the front bumper.**  The mobility trace stores
``(x_m, y_m)`` at ``GridEdge.position_at(offset_m)``, and ``offset_m`` is the
vehicle's *front*, not its centre -- the car-following gap is computed as
``leader.offset_m - follower.offset_m - leader.length_m``.  A rectangle built
by treating the stored point as a centre is displaced forward by half a vehicle
length, roughly 2.25 m for a passenger car.  :func:`vehicle_rectangle` is the
only supported way to build a vehicle footprint, and it applies the correction.

**Boundary contact counts as intersection.**  A segment exactly tangent to a
rectangle's edge or corner is reported as intersecting.  The case has measure
zero, but the choice is deliberate: for occlusion it over-reports blockage
rather than under-reporting it, which understates V-VLC availability and so
understates the benefit claimed for the hybrid scheme.

No NumPy: these operate on scalars, are called from tight loops with tiny
inputs, and array overhead per call would dominate.  Vectorisation, if it is
ever needed, belongs in the spatial index rather than here.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

#: Tolerance for treating a direction component as parallel to a rectangle
#: axis.  Positions are metres, so this is far below any physical scale.
_PARALLEL_EPS = 1e-12


class VehicleFootprintSource(Protocol):
    """Structural type of anything carrying a vehicle's planar footprint.

    Satisfied by ``VehicleState``, the trace's ``VehicleTraceRecord``, and any
    equivalent record, so the geometry layer needs no import from mobility.
    """

    @property
    def x_m(self) -> float: ...
    @property
    def y_m(self) -> float: ...
    @property
    def heading_rad(self) -> float: ...
    @property
    def length_m(self) -> float: ...
    @property
    def width_m(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Point:
    """A planar position in metres."""

    x_m: float
    y_m: float

    def distance_to(self, other: Point) -> float:
        return math.hypot(other.x_m - self.x_m, other.y_m - self.y_m)


@dataclass(frozen=True, slots=True)
class Segment:
    """A straight line segment between two planar positions."""

    start: Point
    end: Point

    @property
    def length_m(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def is_degenerate(self) -> bool:
        """Whether the segment has collapsed to a point."""

        return self.length_m <= _PARALLEL_EPS


@dataclass(frozen=True, slots=True)
class OrientedRectangle:
    """An axis-aligned rectangle rotated about its own centre.

    ``length_m`` runs along ``heading_rad`` and ``width_m`` across it, matching
    the vehicle convention: a car heading east is 4.5 m along x and 1.8 m
    along y.
    """

    centre: Point
    heading_rad: float
    length_m: float
    width_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.heading_rad):
            raise ValueError("heading_rad must be finite")
        if not math.isfinite(self.length_m) or self.length_m <= 0.0:
            raise ValueError("length_m must be finite and positive")
        if not math.isfinite(self.width_m) or self.width_m <= 0.0:
            raise ValueError("width_m must be finite and positive")

    @property
    def half_length_m(self) -> float:
        return 0.5 * self.length_m

    @property
    def half_width_m(self) -> float:
        return 0.5 * self.width_m

    def to_local(self, point: Point) -> Point:
        """Express ``point`` in the rectangle's frame.

        In the returned frame the rectangle is axis-aligned and centred at the
        origin, spanning ``+/-half_length_m`` in x and ``+/-half_width_m``
        in y.
        """

        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)
        dx = point.x_m - self.centre.x_m
        dy = point.y_m - self.centre.y_m
        return Point(dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h)

    def contains(self, point: Point) -> bool:
        """Whether ``point`` lies inside or on the rectangle."""

        local = self.to_local(point)
        return abs(local.x_m) <= self.half_length_m and abs(local.y_m) <= self.half_width_m

    def corners(self) -> tuple[Point, Point, Point, Point]:
        """Return the corners in order, starting from front-left.

        "Front" is the ``+length`` end, so the first two corners are the
        leading face of a vehicle travelling along ``heading_rad``.
        """

        cos_h = math.cos(self.heading_rad)
        sin_h = math.sin(self.heading_rad)
        half_l = self.half_length_m
        half_w = self.half_width_m

        def place(along: float, across: float) -> Point:
            return Point(
                self.centre.x_m + along * cos_h - across * sin_h,
                self.centre.y_m + along * sin_h + across * cos_h,
            )

        return (
            place(half_l, half_w),
            place(half_l, -half_w),
            place(-half_l, -half_w),
            place(-half_l, half_w),
        )


def rectangle_from_front(
    front: Point,
    heading_rad: float,
    length_m: float,
    width_m: float,
) -> OrientedRectangle:
    """Build a rectangle whose *leading edge midpoint* is ``front``.

    This is the correction described in the module docstring: the centre sits
    half a length behind the stored position, along the heading.
    """

    half_l = 0.5 * length_m
    centre = Point(
        front.x_m - half_l * math.cos(heading_rad),
        front.y_m - half_l * math.sin(heading_rad),
    )
    return OrientedRectangle(
        centre=centre,
        heading_rad=heading_rad,
        length_m=length_m,
        width_m=width_m,
    )


def vehicle_rectangle(vehicle: VehicleFootprintSource) -> OrientedRectangle:
    """Return the planar footprint of ``vehicle`` from a trace record.

    The record's ``(x_m, y_m)`` is the front bumper, so this is the only
    correct way to obtain a vehicle footprint from trace data.
    """

    return rectangle_from_front(
        Point(vehicle.x_m, vehicle.y_m),
        vehicle.heading_rad,
        vehicle.length_m,
        vehicle.width_m,
    )


def vehicle_rear(vehicle: VehicleFootprintSource) -> Point:
    """Return the midpoint of ``vehicle``'s trailing edge.

    The V-VLC photodiode faces rearward, so a link terminates here on the
    receiving vehicle rather than at its stored front-bumper position.
    """

    return Point(
        vehicle.x_m - vehicle.length_m * math.cos(vehicle.heading_rad),
        vehicle.y_m - vehicle.length_m * math.sin(vehicle.heading_rad),
    )


def segment_intersects_rectangle(segment: Segment, rectangle: OrientedRectangle) -> bool:
    """Whether ``segment`` touches or crosses ``rectangle``.

    Implemented as a slab clip in the rectangle's own frame, which turns the
    oriented test into an axis-aligned one and handles rotation exactly rather
    than by approximating the rectangle with a radius.  Boundary contact counts
    as intersection; see the module docstring.
    """

    start = rectangle.to_local(segment.start)
    end = rectangle.to_local(segment.end)
    direction_x = end.x_m - start.x_m
    direction_y = end.y_m - start.y_m

    enter = 0.0
    exit_ = 1.0
    for origin, direction, half_extent in (
        (start.x_m, direction_x, rectangle.half_length_m),
        (start.y_m, direction_y, rectangle.half_width_m),
    ):
        if abs(direction) <= _PARALLEL_EPS:
            # Parallel to this pair of faces: it can only intersect if it
            # already lies between them.
            if abs(origin) > half_extent:
                return False
            continue
        near = (-half_extent - origin) / direction
        far = (half_extent - origin) / direction
        if near > far:
            near, far = far, near
        enter = max(enter, near)
        exit_ = min(exit_, far)
        if enter > exit_:
            return False
    return True


def _projection_gaps(
    a: OrientedRectangle, b: OrientedRectangle
) -> Iterator[float]:
    """Overlap of both bodies' shadows on each of the four face normals.

    A separating axis between two convex polygons is always normal to an edge
    of one of them, and a rectangle has two distinct edge normals, so these
    four axes decide the question exactly.  A negative gap on any one of them
    proves the bodies are apart; the smallest positive gap is how far one would
    have to be pushed to separate them.
    """

    for rectangle in (a, b):
        for axis_rad in (rectangle.heading_rad, rectangle.heading_rad + 0.5 * math.pi):
            unit_x, unit_y = math.cos(axis_rad), math.sin(axis_rad)
            spans = []
            for target in (a, b):
                projected = [c.x_m * unit_x + c.y_m * unit_y for c in target.corners()]
                spans.append((min(projected), max(projected)))
            (low_a, high_a), (low_b, high_b) = spans
            yield min(high_a, high_b) - max(low_a, low_b)


def rectangles_overlap(a: OrientedRectangle, b: OrientedRectangle) -> bool:
    """Do two bodies occupy the same ground?

    This is the question the mobility acceptance invariant asks of generated
    traces, and it is deliberately the same function junction admission asks of
    a crossing it is about to permit.  Those two were previously separate: the
    invariant compared footprints while admission compared centreline paths
    inflated by a margin.  Every junction defect so far came from that gap --
    the margin is a *proxy* for body overlap, and each time the layout changed,
    the proxy and the invariant disagreed about a case neither had been tested
    on.  Sharing one implementation makes disagreement impossible rather than
    unlikely.

    Boundary contact counts as overlap, matching
    :func:`segment_intersects_rectangle`.
    """

    return all(gap >= 0.0 for gap in _projection_gaps(a, b))


def rectangle_penetration_m(a: OrientedRectangle, b: OrientedRectangle) -> float:
    """Minimum translation distance separating two overlapping bodies.

    Zero when they are apart or merely touching.  Depth is a property of the
    collision rather than of the sampling, so it is the more stable of the two
    quantities the overlap invariant bounds.
    """

    deepest = math.inf
    for gap in _projection_gaps(a, b):
        if gap <= 0.0:
            return 0.0
        deepest = min(deepest, gap)
    return deepest


__all__ = [
    "OrientedRectangle",
    "Point",
    "Segment",
    "VehicleFootprintSource",
    "rectangle_from_front",
    "rectangle_penetration_m",
    "rectangles_overlap",
    "segment_intersects_rectangle",
    "vehicle_rear",
    "vehicle_rectangle",
]
