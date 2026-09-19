"""Geometric obstruction of a link by other vehicles.

This module answers one question exactly: which vehicles stand between the two
ends of a link.  It is deliberately the *only* place that decision is made, for
both V-VLC occlusion and the RF NLOSv classification, so the two cannot drift
apart.

**It must never import a path-loss model.**  Work plan section 5.3 makes this a
structural rule rather than a matter of style.  TR 37.885 models vehicle
blockage stochastically, as an extra loss term.  Adopting that treatment here
would make blockage something other than a function of where a blocker
actually is: it could not be predicted from tracked positions, the 200 ms
forecast of section 6.1 would carry no information, and RQ3 would collapse.
Path loss *given* a class belongs in ``channels/rf/pathloss_37885.py``; the
class itself is decided here, from rectangles.

Because both supported links are horizontal (see
:mod:`hybrid_v2x_rl.core.link_endpoints`), obstruction is exactly a planar
footprint intersection combined with a scalar height test.  A vehicle shorter
than the path passes beneath it; every vehicle class in the configured mixture
is at least 1.5 m tall, so nothing passes beneath a 0.7 m optical path, but the
test is applied rather than assumed.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol

from hybrid_v2x_rl.core.geometry import (
    Point,
    Segment,
    VehicleFootprintSource,
    segment_intersects_rectangle,
    vehicle_rectangle,
)
from hybrid_v2x_rl.core.link_endpoints import LinkPath


class OccludingVehicle(VehicleFootprintSource, Protocol):
    """A vehicle that can obstruct a link: a footprint, an identity, a height."""

    @property
    def vehicle_id(self) -> str: ...
    @property
    def height_m(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Occlusion:
    """Which vehicles obstruct a link, nearest to the transmitter first."""

    blocker_ids: tuple[str, ...]

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocker_ids)

    @property
    def blocker_count(self) -> int:
        return len(self.blocker_ids)

    @property
    def nearest_blocker_id(self) -> str | None:
        """The obstruction closest to the transmitting end, if any."""

        return self.blocker_ids[0] if self.blocker_ids else None


CLEAR = Occlusion(blocker_ids=())


def _distance_along(segment: Segment, point: Point) -> float:
    """Projection of ``point`` onto the segment's direction, from its start."""

    dx = segment.end.x_m - segment.start.x_m
    dy = segment.end.y_m - segment.start.y_m
    norm = math.hypot(dx, dy)
    if norm <= 0.0:
        return 0.0
    return ((point.x_m - segment.start.x_m) * dx + (point.y_m - segment.start.y_m) * dy) / norm


def occluding_vehicles(
    path: LinkPath,
    candidates: Iterable[OccludingVehicle],
    *,
    exclude_ids: Collection[str] = (),
) -> Occlusion:
    """Return the vehicles obstructing ``path``.

    ``exclude_ids`` must name the link's own endpoints; a transmitter and
    receiver always meet their own link and would otherwise be reported as
    blocking it.

    A candidate obstructs when its footprint meets the path *and* it stands at
    least as tall as the path.  Ties at exactly the path height count as
    obstructing, consistent with the boundary convention in
    :mod:`hybrid_v2x_rl.core.geometry`.
    """

    excluded = frozenset(exclude_ids)
    found: list[tuple[float, str]] = []
    for candidate in candidates:
        if candidate.vehicle_id in excluded:
            continue
        if candidate.height_m < path.height_m:
            continue
        rectangle = vehicle_rectangle(candidate)
        if not segment_intersects_rectangle(path.segment, rectangle):
            continue
        found.append((_distance_along(path.segment, rectangle.centre), candidate.vehicle_id))
    found.sort()
    return Occlusion(blocker_ids=tuple(vehicle_id for _, vehicle_id in found))


def is_obstructed(
    path: LinkPath,
    candidates: Iterable[OccludingVehicle],
    *,
    exclude_ids: Collection[str] = (),
) -> bool:
    """Whether any vehicle obstructs ``path``.

    Short-circuits on the first obstruction, so prefer this to
    :func:`occluding_vehicles` when the blocker's identity is not needed.
    """

    excluded = frozenset(exclude_ids)
    for candidate in candidates:
        if candidate.vehicle_id in excluded:
            continue
        if candidate.height_m < path.height_m:
            continue
        if segment_intersects_rectangle(path.segment, vehicle_rectangle(candidate)):
            return True
    return False


__all__ = [
    "CLEAR",
    "Occlusion",
    "OccludingVehicle",
    "is_obstructed",
    "occluding_vehicles",
]
