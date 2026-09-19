"""Uniform-grid index over the vehicles present in one frame.

Obstruction queries are the geometry layer's hot loop: every pair-instant tests
a link against whatever might stand in it, and a naive query scans all 1,120
vehicles present at 30 veh/lane-km.  This narrows that to the handful sharing a
neighbourhood with the link.

**It is a filter, not a decision.**  :meth:`SpatialIndex.candidates` returns a
*superset* of the vehicles whose footprints could meet a segment; the exact
test still runs in :mod:`hybrid_v2x_rl.geometry.vehicle_occlusion`.  Returning extra
candidates costs time, never correctness.  Returning too few would silently
lose blockers, so the index is built last in M2 and tested by asserting it
agrees with the brute-force scan it replaces.

The cell size defaults to the largest footprint diagonal in the frame, which
keeps a vehicle inside a small number of cells while leaving buckets large
enough that a short link touches few of them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hybrid_v2x_rl.core.geometry import Segment, vehicle_rectangle
from hybrid_v2x_rl.geometry.vehicle_occlusion import OccludingVehicle

#: Floor on the cell size, so a frame of tiny footprints cannot produce a
#: pathologically fine grid.
_MIN_CELL_SIZE_M = 5.0

_Cell = tuple[int, int]


def _footprint_bounds(vehicle: OccludingVehicle) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a vehicle's rotated footprint."""

    corners = vehicle_rectangle(vehicle).corners()
    xs = [corner.x_m for corner in corners]
    ys = [corner.y_m for corner in corners]
    return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True)
class SpatialIndex:
    """Vehicles bucketed by the cells their footprints occupy."""

    cell_size_m: float
    buckets: dict[_Cell, tuple[OccludingVehicle, ...]]

    @classmethod
    def build(
        cls,
        vehicles: Iterable[OccludingVehicle],
        *,
        cell_size_m: float | None = None,
    ) -> SpatialIndex:
        """Index one frame's vehicles.

        A vehicle is placed in every cell its footprint bounds touch, so a body
        straddling a cell edge is found from either side.
        """

        present: Sequence[OccludingVehicle] = tuple(vehicles)
        if cell_size_m is None:
            cell_size_m = _MIN_CELL_SIZE_M
            for vehicle in present:
                diagonal = math.hypot(vehicle.length_m, vehicle.width_m)
                cell_size_m = max(cell_size_m, diagonal)
        if not math.isfinite(cell_size_m) or cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and positive")

        collected: dict[_Cell, list[OccludingVehicle]] = {}
        for vehicle in present:
            min_x, min_y, max_x, max_y = _footprint_bounds(vehicle)
            for cell_x in range(
                math.floor(min_x / cell_size_m), math.floor(max_x / cell_size_m) + 1
            ):
                for cell_y in range(
                    math.floor(min_y / cell_size_m), math.floor(max_y / cell_size_m) + 1
                ):
                    collected.setdefault((cell_x, cell_y), []).append(vehicle)

        return cls(
            cell_size_m=cell_size_m,
            buckets={cell: tuple(found) for cell, found in collected.items()},
        )

    @property
    def cell_count(self) -> int:
        return len(self.buckets)

    def candidates(
        self, segment: Segment, *, margin_m: float = 0.0
    ) -> tuple[OccludingVehicle, ...]:
        """Vehicles that might meet ``segment``, in stable order.

        ``margin_m`` widens the search, for callers that need everything within
        a distance of the segment rather than only what touches it.  A vehicle
        appearing in several cells is returned once.
        """

        if margin_m < 0.0:
            raise ValueError("margin_m must be non-negative")

        min_x = min(segment.start.x_m, segment.end.x_m) - margin_m
        max_x = max(segment.start.x_m, segment.end.x_m) + margin_m
        min_y = min(segment.start.y_m, segment.end.y_m) - margin_m
        max_y = max(segment.start.y_m, segment.end.y_m) + margin_m

        seen: dict[str, OccludingVehicle] = {}
        for cell_x in range(
            math.floor(min_x / self.cell_size_m), math.floor(max_x / self.cell_size_m) + 1
        ):
            for cell_y in range(
                math.floor(min_y / self.cell_size_m), math.floor(max_y / self.cell_size_m) + 1
            ):
                for vehicle in self.buckets.get((cell_x, cell_y), ()):
                    seen.setdefault(vehicle.vehicle_id, vehicle)
        return tuple(seen.values())


__all__ = [
    "SpatialIndex",
]
