"""Where a link sits relative to the junctions that can obstruct it.

This is the module the 200 ms blockage forecast of work plan section 6.1 rests
on.  Cross-traffic can only reach a link where the link crosses a junction, so
"does this path span a junction box, and if not how far away is one" is the
leading indicator, and the forecast becomes a kinematics problem rather than a
guess.  Measured on the published traces, a path spanning a junction is blocked
3.5 to 4.9 times as often as one mid-block.

Note that those traces carry the defect of work plan section 4.6.3: opposing
carriageways share a centreline, so most of their measured blockage is oncoming
traffic sitting on the optical axis rather than genuine cross-traffic.  That
inflates the mid-block baseline and therefore *understates* the lift computed
above.  The junction signal is real either way; its magnitude is not yet
settled.

**These features are legitimately observable.**  Unlike the occlusion truth in
:mod:`hybrid_v2x_rl.geometry.vehicle_occlusion`, everything here derives from the
vehicle's own pose and a road map: a real vehicle knows where it is and where
the next intersection is, and V2X already carries signal phase in SPaT.  They
may therefore appear in the policy observation, subject to the same position
noise and ageing as any other tracked quantity.  Nothing in this module reads a
blocker's state, which is what keeps that permissible.

**Cardinal headings are assumed** for the distance-to-junction calculation.
The analytic grid only ever produces headings along the two street axes, so
projecting to the next lattice line on the dominant axis is exact rather than
approximate.  A curved or skewed network would need a real map projection here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.geometry import (
    OrientedRectangle,
    Point,
    Segment,
    segment_intersects_rectangle,
)
from hybrid_v2x_rl.core.grid_naming import junction_id

#: Tolerance for deciding a position already sits on a lattice line, in metres.
_LATTICE_EPS_M = 1e-9


@dataclass(frozen=True, slots=True)
class JunctionRef:
    """One intersection of the lattice."""

    avenue_index: int
    cross_street_index: int
    x_m: float
    y_m: float

    @property
    def junction_id(self) -> str:
        """Canonical identifier, matching the trace's naming."""

        return junction_id(self.avenue_index, self.cross_street_index)


@dataclass(frozen=True, slots=True)
class IntersectionContext:
    """A link's relationship to the junctions near it.

    ``spanned_junction_ids`` is the set the optical or radio path actually
    passes through, and is the feature that carries the forecast: an empty
    tuple means cross-traffic has no way to reach the path at this instant.
    """

    transmitter_distance_to_junction_m: float
    receiver_distance_to_junction_m: float
    spanned_junction_ids: tuple[str, ...]

    @property
    def spans_junction(self) -> bool:
        return bool(self.spanned_junction_ids)

    @property
    def spanned_junction_count(self) -> int:
        return len(self.spanned_junction_ids)

    @property
    def nearest_distance_to_junction_m(self) -> float:
        """Whichever endpoint is closer to reaching an intersection."""

        return min(
            self.transmitter_distance_to_junction_m,
            self.receiver_distance_to_junction_m,
        )


def _distance_to_next_lattice_line(position_m: float, pitch_m: float, forward: bool) -> float:
    """Distance from ``position_m`` to the next lattice line in the direction of travel.

    A position already on a line returns zero rather than a full pitch.
    """

    index = position_m / pitch_m
    if forward:
        boundary = math.ceil(index - _LATTICE_EPS_M) * pitch_m
        return max(boundary - position_m, 0.0)
    boundary = math.floor(index + _LATTICE_EPS_M) * pitch_m
    return max(position_m - boundary, 0.0)


@dataclass(frozen=True)
class JunctionGrid:
    """The lattice of intersections implied by a Manhattan street grid.

    ``road_half_width_m`` matches
    :class:`hybrid_v2x_rl.geometry.building_geometry.BuildingLayout`, so a junction
    box is exactly the square where two carriageways overlap.
    """

    avenues: int
    cross_streets: int
    avenue_spacing_m: float
    cross_street_spacing_m: float
    road_half_width_m: float

    def __post_init__(self) -> None:
        if self.avenues < 1 or self.cross_streets < 1:
            raise ValueError("a grid needs at least one avenue and one cross street")
        if self.road_half_width_m <= 0.0:
            raise ValueError("road_half_width_m must be positive")

    @classmethod
    def from_grid(
        cls,
        *,
        avenues: int,
        cross_streets: int,
        avenue_spacing_m: float,
        cross_street_spacing_m: float,
        road_half_width_m: float,
    ) -> JunctionGrid:
        return cls(
            avenues=avenues,
            cross_streets=cross_streets,
            avenue_spacing_m=avenue_spacing_m,
            cross_street_spacing_m=cross_street_spacing_m,
            road_half_width_m=road_half_width_m,
        )

    @property
    def junction_count(self) -> int:
        return self.avenues * self.cross_streets

    def box(self, avenue_index: int, cross_street_index: int) -> OrientedRectangle:
        """The square where the two carriageways cross."""

        side = 2.0 * self.road_half_width_m
        return OrientedRectangle(
            centre=Point(
                avenue_index * self.avenue_spacing_m,
                cross_street_index * self.cross_street_spacing_m,
            ),
            heading_rad=0.0,
            length_m=side,
            width_m=side,
        )

    def junctions_overlapping(self, segment: Segment) -> tuple[JunctionRef, ...]:
        """Every junction whose box the segment meets.

        Only the lattice cells within the segment's bounding box are tested, so
        this stays cheap without approximating: a box outside that envelope
        cannot intersect the segment.
        """

        pad = self.road_half_width_m
        lo_x = min(segment.start.x_m, segment.end.x_m) - pad
        hi_x = max(segment.start.x_m, segment.end.x_m) + pad
        lo_y = min(segment.start.y_m, segment.end.y_m) - pad
        hi_y = max(segment.start.y_m, segment.end.y_m) + pad

        first_avenue = max(0, math.floor(lo_x / self.avenue_spacing_m))
        last_avenue = min(self.avenues - 1, math.ceil(hi_x / self.avenue_spacing_m))
        first_cross = max(0, math.floor(lo_y / self.cross_street_spacing_m))
        last_cross = min(self.cross_streets - 1, math.ceil(hi_y / self.cross_street_spacing_m))

        found: list[JunctionRef] = []
        for avenue in range(first_avenue, last_avenue + 1):
            for cross in range(first_cross, last_cross + 1):
                if segment_intersects_rectangle(segment, self.box(avenue, cross)):
                    found.append(
                        JunctionRef(
                            avenue_index=avenue,
                            cross_street_index=cross,
                            x_m=avenue * self.avenue_spacing_m,
                            y_m=cross * self.cross_street_spacing_m,
                        )
                    )
        return tuple(found)

    def distance_ahead_to_junction_m(self, x_m: float, y_m: float, heading_rad: float) -> float:
        """Distance a vehicle must travel to reach the next intersection.

        Zero when the vehicle is already on the crossing line.  See the module
        docstring on the cardinal-heading assumption.
        """

        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)
        if abs(cos_h) >= abs(sin_h):
            return _distance_to_next_lattice_line(x_m, self.avenue_spacing_m, cos_h >= 0.0)
        return _distance_to_next_lattice_line(y_m, self.cross_street_spacing_m, sin_h >= 0.0)


def intersection_context(
    transmitter_x_m: float,
    transmitter_y_m: float,
    transmitter_heading_rad: float,
    receiver_x_m: float,
    receiver_y_m: float,
    receiver_heading_rad: float,
    path: Segment,
    grid: JunctionGrid,
) -> IntersectionContext:
    """Summarise a link's position relative to nearby junctions.

    ``path`` is passed explicitly rather than derived, because the optical and
    radio links have different endpoints and either may be the one in question.
    """

    spanned = grid.junctions_overlapping(path)
    return IntersectionContext(
        transmitter_distance_to_junction_m=grid.distance_ahead_to_junction_m(
            transmitter_x_m, transmitter_y_m, transmitter_heading_rad
        ),
        receiver_distance_to_junction_m=grid.distance_ahead_to_junction_m(
            receiver_x_m, receiver_y_m, receiver_heading_rad
        ),
        spanned_junction_ids=tuple(ref.junction_id for ref in spanned),
    )


__all__ = [
    "IntersectionContext",
    "JunctionGrid",
    "JunctionRef",
    "intersection_context",
]
