"""Rectangular Manhattan-grid topology for the analytic V2V mobility model.

Implements the geometry of `research/MOBILITY_MODEL_V2_PROPOSAL.md` section 4.1.
This module is pure geometry: it owns junctions, directed edges, coordinates,
headings, and turn resolution.  It launches no external process and has no
SUMO dependency.

Naming follows the SUMO convention already used by the project's traces so the
two mobility backends stay comparable: a junction is a letter for its avenue
and a number for its cross street (``B7``), and an edge concatenates its
endpoints (``B7B8``).

Coordinate convention:

- avenues run north-south and are spaced ``avenue_spacing_m`` apart in ``x``;
- cross streets run east-west and are spaced ``cross_street_spacing_m`` in ``y``;
- junction ``(a, c)`` sits at ``(a * avenue_spacing, c * cross_street_spacing)``;
- headings are mathematical: east is 0 and angles increase counter-clockwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

from hybrid_v2x_rl.core.grid_naming import junction_id

#: Turn names, matching the configured turn-probability fields.
TurnName = str

#: Conventional urban lane width in metres.  Owned here rather than in the
#: geometry layer because it changes vehicle positions and is therefore part of
#: the road model, not an interpretation of it.
LANE_WIDTH_M = 3.5


class Direction(Enum):
    """Travel direction of a one-way directed edge."""

    EAST = "east"
    NORTH = "north"
    WEST = "west"
    SOUTH = "south"

    @property
    def heading_rad(self) -> float:
        """Mathematical heading: east is 0, increasing counter-clockwise."""

        return {
            Direction.EAST: 0.0,
            Direction.NORTH: 0.5 * math.pi,
            Direction.WEST: math.pi,
            Direction.SOUTH: 1.5 * math.pi,
        }[self]

    @property
    def is_avenue(self) -> bool:
        """Whether travel is along an avenue (north-south)."""

        return self in (Direction.NORTH, Direction.SOUTH)

    @property
    def delta(self) -> tuple[int, int]:
        """Change in ``(avenue_index, cross_street_index)`` along this edge."""

        return {
            Direction.EAST: (1, 0),
            Direction.WEST: (-1, 0),
            Direction.NORTH: (0, 1),
            Direction.SOUTH: (0, -1),
        }[self]


#: Outgoing direction produced by each turn, per incoming direction.
_TURNS: dict[Direction, dict[TurnName, Direction]] = {
    Direction.NORTH: {"left": Direction.WEST, "straight": Direction.NORTH, "right": Direction.EAST},
    Direction.EAST: {"left": Direction.NORTH, "straight": Direction.EAST, "right": Direction.SOUTH},
    Direction.SOUTH: {"left": Direction.EAST, "straight": Direction.SOUTH, "right": Direction.WEST},
    Direction.WEST: {"left": Direction.SOUTH, "straight": Direction.WEST, "right": Direction.NORTH},
}


@dataclass(frozen=True, slots=True)
class GridNetworkSpec:
    """Frozen topology parameters.

    Defaults reproduce the project's frozen synthetic grid: the characteristic
    long, rectangular Manhattan block rather than a square abstraction.
    """

    avenues: int = 6
    cross_streets: int = 12
    avenue_spacing_m: float = 244.0
    cross_street_spacing_m: float = 61.0
    #: Travel lanes per street.  One-way, so the whole carriageway runs one
    #: way and every lane is available to it: a street of a given width carries
    #: twice the same-direction lanes it would if it were two-way.
    lanes_per_direction: int = 2
    lane_width_m: float = LANE_WIDTH_M
    #: Manhattan's grid is predominantly one-way, alternating street by street
    #: and avenue by avenue.  It is not incidental: the one-way conversion is
    #: what makes avenue progression possible, because a two-way street can
    #: only carry a green wave at ``v = 2L/(kC)`` -- 1.36 m/s for 61 m blocks
    #: on a 90 s cycle, which is no speed at all.  Setting this false restores
    #: two directed edges per segment and is kept for comparison only.
    one_way: bool = True
    speed_limit_mps: float = 11.18

    def __post_init__(self) -> None:
        if not 2 <= self.avenues <= 26:
            raise ValueError("avenues must be between 2 and 26")
        if self.cross_streets < 2:
            raise ValueError("cross_streets must be at least 2")
        if self.avenue_spacing_m <= 0.0 or self.cross_street_spacing_m <= 0.0:
            raise ValueError("grid spacings must be positive")
        if self.lanes_per_direction < 1:
            raise ValueError("lanes_per_direction must be at least one")
        if self.lane_width_m <= 0.0:
            raise ValueError("lane_width_m must be positive")
        if self.speed_limit_mps <= 0.0:
            raise ValueError("speed_limit_mps must be positive")
        if self.carriageway_width_m >= 0.5 * min(
            self.avenue_spacing_m, self.cross_street_spacing_m
        ):
            raise ValueError("carriageways are wider than the block they separate")

    @property
    def carriageway_width_m(self) -> float:
        """Total paved width of one direction of travel."""

        return self.lanes_per_direction * self.lane_width_m

    def lane_centre_offset_m(self, lane_index: int = 0) -> float:
        """Lateral distance from the street centreline to lane ``lane_index``.

        Higher indices lie further right in the direction of travel, so MOBIL's
        keep-right bias points towards higher indices either way.

        One-way streets spread their lanes across the whole carriageway, so the
        offsets straddle the centreline.  Two-way streets keep each direction
        entirely to the right of it, which is what stops opposing traffic
        sharing a line -- the defect recorded in work plan section 4.6.3.
        """

        if not 0 <= lane_index < self.lanes_per_direction:
            raise ValueError(
                f"lane_index {lane_index} outside 0..{self.lanes_per_direction - 1}"
            )
        if self.one_way:
            centred = lane_index + 0.5 - 0.5 * self.lanes_per_direction
            return centred * self.lane_width_m
        return (lane_index + 0.5) * self.lane_width_m

    def travel_direction(self, direction: Direction) -> Direction:
        """The direction a street of this orientation actually runs.

        Avenues alternate northbound and southbound by avenue index, cross
        streets eastbound and westbound by cross-street index, which is how
        Manhattan is laid out.  Under ``one_way`` a segment offered in the
        opposite sense simply does not exist.
        """

        return direction

    @property
    def outermost_lane_centre_offset_m(self) -> float:
        """Offset of the kerb-side lane, which sets how far a crossing body reaches.

        A stop line must stand clear of *this* plus half a vehicle width, not
        of the carriageway centre: it is the outermost crossing lane whose
        traffic a waiting vehicle must not intrude upon.
        """

        return max(
            abs(self.lane_centre_offset_m(index)) for index in range(self.lanes_per_direction)
        )


@dataclass(frozen=True, slots=True)
class Junction:
    """One grid intersection."""

    junction_id: str
    avenue_index: int
    cross_street_index: int
    x_m: float
    y_m: float
    signalized: bool


@dataclass(frozen=True, slots=True)
class GridEdge:
    """One directed, single-carriageway road segment between two junctions."""

    edge_id: str
    from_junction: str
    to_junction: str
    direction: Direction
    length_m: float
    start_x_m: float
    start_y_m: float
    lanes: int = 1
    """Travel lanes in this direction; higher indices lie further right."""

    lane_width_m: float = LANE_WIDTH_M
    one_way: bool = True
    """One-way lanes straddle the centreline; two-way keep to one side of it."""

    @property
    def heading_rad(self) -> float:
        return self.direction.heading_rad

    @property
    def right_unit(self) -> tuple[float, float]:
        """Unit vector 90 degrees clockwise from the heading."""

        heading = self.heading_rad
        return (math.sin(heading), -math.cos(heading))

    def lane_id(self, lane_index: int = 0) -> str:
        """SUMO-style lane identifier."""

        return f"{self.edge_id}_{lane_index}"

    def lane_offset_m(self, lane_index: int = 0) -> float:
        """Lateral offset of ``lane_index`` from the street centreline.

        Must agree with ``GridNetworkSpec.lane_centre_offset_m``.  It did not
        when one-way arrived, and because *this* is the copy ``position_at``
        uses, every vehicle sat at the two-way offsets while the spec reported
        the one-way ones.
        """

        if not 0 <= lane_index < self.lanes:
            raise ValueError(f"lane_index {lane_index} outside 0..{self.lanes - 1}")
        if self.one_way:
            return (lane_index + 0.5 - 0.5 * self.lanes) * self.lane_width_m
        return (lane_index + 0.5) * self.lane_width_m

    def position_at(self, offset_m: float, lane_index: int = 0) -> tuple[float, float]:
        """Cartesian position ``offset_m`` along this edge, in ``lane_index``.

        The returned point is on a travel lane, offset to the right of the
        street centreline, never on the centreline itself.  Opposing edges
        therefore never overlap, and neither do adjacent lanes.
        """

        heading = self.heading_rad
        right_x, right_y = self.right_unit
        lateral = self.lane_offset_m(lane_index)
        return (
            self.start_x_m + offset_m * math.cos(heading) + lateral * right_x,
            self.start_y_m + offset_m * math.sin(heading) + lateral * right_y,
        )


def _runs(direction: Direction, node: Junction) -> bool:
    """Whether a one-way street at ``node`` carries ``direction``.

    Avenues alternate by avenue index and cross streets by cross-street index,
    so adjacent parallel streets always run opposite ways.  A vehicle therefore
    never meets oncoming traffic on its own street, which removes the dominant
    junction conflict -- a left turn across an opposing lane -- and is what lets
    a green wave exist at all: a two-way street can only carry one at
    ``v = 2L/(kC)``, which is 1.36 m/s for 61 m blocks on a 90 s cycle.
    """

    if direction.is_avenue:
        northbound = node.avenue_index % 2 == 0
        return direction is (Direction.NORTH if northbound else Direction.SOUTH)
    eastbound = node.cross_street_index % 2 == 0
    return direction is (Direction.EAST if eastbound else Direction.WEST)


class GridNetwork:
    """Directed rectangular grid with turn resolution and boundary detection."""

    def __init__(self, spec: GridNetworkSpec | None = None) -> None:
        self.spec = spec or GridNetworkSpec()
        self._junctions: dict[str, Junction] = {}
        self._edges: dict[str, GridEdge] = {}
        self._outgoing: dict[str, list[GridEdge]] = {}
        self._build()

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        spec = self.spec
        for a in range(spec.avenues):
            for c in range(spec.cross_streets):
                interior = 0 < a < spec.avenues - 1 and 0 < c < spec.cross_streets - 1
                node = Junction(
                    junction_id=junction_id(a, c),
                    avenue_index=a,
                    cross_street_index=c,
                    x_m=a * spec.avenue_spacing_m,
                    y_m=c * spec.cross_street_spacing_m,
                    signalized=interior,
                )
                self._junctions[node.junction_id] = node
                self._outgoing[node.junction_id] = []

        for node in self._junctions.values():
            for direction in Direction:
                if spec.one_way and not _runs(direction, node):
                    continue
                da, dc = direction.delta
                a2 = node.avenue_index + da
                c2 = node.cross_street_index + dc
                if not (0 <= a2 < spec.avenues and 0 <= c2 < spec.cross_streets):
                    continue
                target = junction_id(a2, c2)
                length = (
                    spec.cross_street_spacing_m if direction.is_avenue else spec.avenue_spacing_m
                )
                edge = GridEdge(
                    edge_id=f"{node.junction_id}{target}",
                    from_junction=node.junction_id,
                    to_junction=target,
                    direction=direction,
                    length_m=length,
                    start_x_m=node.x_m,
                    start_y_m=node.y_m,
                    lanes=spec.lanes_per_direction,
                    lane_width_m=spec.lane_width_m,
                    one_way=spec.one_way,
                )
                self._edges[edge.edge_id] = edge
                self._outgoing[node.junction_id].append(edge)

    # -- lookups ----------------------------------------------------------

    @property
    def junctions(self) -> tuple[Junction, ...]:
        return tuple(self._junctions[key] for key in sorted(self._junctions))

    @property
    def edges(self) -> tuple[GridEdge, ...]:
        return tuple(self._edges[key] for key in sorted(self._edges))

    def junction(self, junction_id_: str) -> Junction:
        return self._junctions[junction_id_]

    def edge(self, edge_id: str) -> GridEdge:
        return self._edges[edge_id]

    def outgoing(self, junction_id_: str) -> tuple[GridEdge, ...]:
        return tuple(self._outgoing[junction_id_])

    # -- derived quantities -----------------------------------------------

    @cached_property
    def total_lane_length_m(self) -> float:
        """Sum of every directed lane, the denominator of lane density.

        Unlike a generated SUMO network there is no junction radius, so this
        equals the nominal geometric length exactly.
        """

        per_direction = sum(edge.length_m for edge in self._edges.values())
        return per_direction * self.spec.lanes_per_direction

    @cached_property
    def signalized_junction_ids(self) -> tuple[str, ...]:
        return tuple(node.junction_id for node in self.junctions if node.signalized)

    @cached_property
    def entry_edges(self) -> tuple[GridEdge, ...]:
        """Edges that begin at a boundary junction, where vehicles are injected."""

        return tuple(
            edge for edge in self.edges if self._is_boundary(self._junctions[edge.from_junction])
        )

    def _is_boundary(self, node: Junction) -> bool:
        spec = self.spec
        return node.avenue_index in (0, spec.avenues - 1) or node.cross_street_index in (
            0,
            spec.cross_streets - 1,
        )

    def is_exit_junction(self, junction_id_: str) -> bool:
        """Whether a vehicle reaching this junction may leave the network."""

        return self._is_boundary(self._junctions[junction_id_])

    # -- routing ----------------------------------------------------------

    def turn_options(self, edge: GridEdge) -> dict[TurnName, GridEdge]:
        """Return the available ``{turn: edge}`` continuations, excluding U-turns.

        A turn is unavailable when it would leave the grid.  Callers must
        renormalise turn probabilities over whatever remains.
        """

        options: dict[TurnName, GridEdge] = {}
        for turn, direction in _TURNS[edge.direction].items():
            for candidate in self._outgoing[edge.to_junction]:
                if candidate.direction is direction:
                    options[turn] = candidate
                    break
        return options


__all__ = [
    "Direction",
    "GridEdge",
    "GridNetwork",
    "GridNetworkSpec",
    "Junction",
    "TurnName",
    "junction_id",
]
