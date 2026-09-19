"""City blocks as the complement of the street grid.

Work plan section 4.2 builds the road network analytically, and buildings
follow from it rather than from an imported polygon file: a block is whatever
rectangle is left between four surrounding street centrelines once the
carriageway is taken out.  Nothing here needs to be fitted or downloaded, and a
link around a corner is obstructed by construction.

**Road width comes from the road model.**  ``GridConfig.lane_width_m`` sets it,
because lane width moves vehicles and so belongs to mobility rather than to an
interpretation applied afterwards.  A block is inset by ``lanes_per_direction``
lanes on each side of the centreline, which for the headline single-lane grid
leaves 237 x 54 m blocks inside the 244 x 61 m street pitch.

**Buildings are opaque at every link height.**  No height is modelled because
every plausible building exceeds both the 1.5 m antenna path and the 0.7 m
optical path; a height field would be carried and never consulted.  This is a
declared simplification, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from hybrid_v2x_rl.core.geometry import OrientedRectangle, Point
from hybrid_v2x_rl.mobility.grid_network import LANE_WIDTH_M as _LANE_WIDTH_M

#: Re-exported from the road model, which owns it: lane width moves vehicles,
#: so it is part of the mobility configuration rather than an interpretation
#: applied afterwards.
LANE_WIDTH_M = _LANE_WIDTH_M

#: Buildings are axis-aligned, so every rectangle is built on this heading and
#: ``length_m`` runs east while ``width_m`` runs north.
_BUILDING_HEADING_RAD = 0.0


@dataclass(frozen=True)
class BuildingLayout:
    """The rectangular blocks enclosed by a Manhattan street grid.

    ``road_half_width_m`` is measured from a street centreline to the building
    line, so the full carriageway is twice that.

    Unlike the rest of the geometry layer this dataclass is not ``slots=True``:
    :attr:`rectangles` is a ``cached_property`` and needs an instance
    ``__dict__``.  One layout is built per run and queried once per obstruction
    test, so caching the 55 blocks matters and the missing slots do not.
    """

    avenues: int
    cross_streets: int
    avenue_spacing_m: float
    cross_street_spacing_m: float
    road_half_width_m: float

    def __post_init__(self) -> None:
        if self.avenues < 2 or self.cross_streets < 2:
            raise ValueError("a block needs at least two avenues and two cross streets")
        if self.road_half_width_m <= 0.0:
            raise ValueError("road_half_width_m must be positive")
        if self.avenue_spacing_m <= 2.0 * self.road_half_width_m:
            raise ValueError("avenue spacing leaves no room for a block between carriageways")
        if self.cross_street_spacing_m <= 2.0 * self.road_half_width_m:
            raise ValueError("cross-street spacing leaves no room for a block")

    @classmethod
    def from_grid(
        cls,
        *,
        avenues: int,
        cross_streets: int,
        avenue_spacing_m: float,
        cross_street_spacing_m: float,
        lanes_per_direction: int = 1,
        lane_width_m: float = LANE_WIDTH_M,
    ) -> BuildingLayout:
        """Build a layout from the same parameters that define the road network."""

        return cls(
            avenues=avenues,
            cross_streets=cross_streets,
            avenue_spacing_m=avenue_spacing_m,
            cross_street_spacing_m=cross_street_spacing_m,
            road_half_width_m=lanes_per_direction * lane_width_m,
        )

    @property
    def block_length_m(self) -> float:
        """East-west extent of one block, after the carriageway is removed."""

        return self.avenue_spacing_m - 2.0 * self.road_half_width_m

    @property
    def block_width_m(self) -> float:
        """North-south extent of one block."""

        return self.cross_street_spacing_m - 2.0 * self.road_half_width_m

    @property
    def block_count(self) -> int:
        return (self.avenues - 1) * (self.cross_streets - 1)

    @cached_property
    def rectangles(self) -> tuple[OrientedRectangle, ...]:
        """Every block, ordered by avenue index then cross-street index."""

        blocks: list[OrientedRectangle] = []
        for avenue in range(self.avenues - 1):
            for cross in range(self.cross_streets - 1):
                centre = Point(
                    (avenue + 0.5) * self.avenue_spacing_m,
                    (cross + 0.5) * self.cross_street_spacing_m,
                )
                blocks.append(
                    OrientedRectangle(
                        centre=centre,
                        heading_rad=_BUILDING_HEADING_RAD,
                        length_m=self.block_length_m,
                        width_m=self.block_width_m,
                    )
                )
        return tuple(blocks)


__all__ = [
    "LANE_WIDTH_M",
    "BuildingLayout",
]
