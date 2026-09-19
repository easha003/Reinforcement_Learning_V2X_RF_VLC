"""RF propagation class, decided from geometry alone.

Work plan section 5.1 classifies the radio link as **LOS**, **NLOSv** when a
vehicle obstructs it, or **NLOS** when a building does.  This module decides
that class and stops there.

**It must never import a path-loss model**, for the reason section 5.3 gives:
TR 37.885 supplies the loss, shadowing and fading *given* a class, and merging
the two would make the class itself stochastic.  A class that cannot be derived
from where obstacles are cannot be predicted from tracked positions, and the
observation vector's forecast would carry no information.  Path loss belongs in
``channels/rf/pathloss_37885.py``.

**A building outranks a vehicle.**  When both obstruct, the state is NLOS: a
facade is the more severe obstruction and the one TR 37.885 treats with the
separate NLOS model.  Vehicle blockers are still reported alongside, because
the count is useful for diagnostics and for the neighbour visibility that feeds
the RF congestion model.

**Where NLOS actually occurs.**  For a tagged pair it essentially never does.
Pairs share a heading and a planned route, so the link runs along a street and
no block lies between them.  NLOS matters for *neighbour* links instead: it is
what stops a vehicle two streets away from contributing interference, and so it
shapes the channel-busy ratio rather than the tagged link's own budget.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.geometry import (
    OrientedRectangle,
    VehicleFootprintSource,
    segment_intersects_rectangle,
)
from hybrid_v2x_rl.core.link_endpoints import LinkPath, rf_link_path
from hybrid_v2x_rl.geometry.vehicle_occlusion import OccludingVehicle, occluding_vehicles


@dataclass(frozen=True, slots=True)
class RFVisibility:
    """The propagation class of one radio link and what decided it."""

    state: RFPropagationState
    vehicle_blocker_ids: tuple[str, ...]
    building_blocked: bool

    @property
    def is_line_of_sight(self) -> bool:
        return self.state is RFPropagationState.LOS

    @property
    def vehicle_blocker_count(self) -> int:
        return len(self.vehicle_blocker_ids)


def building_obstructs(path: LinkPath, buildings: Iterable[OrientedRectangle]) -> bool:
    """Whether any block stands across ``path``.

    Buildings are opaque at every link height, so no height test applies; see
    :mod:`hybrid_v2x_rl.geometry.building_geometry`.
    """

    return any(segment_intersects_rectangle(path.segment, block) for block in buildings)


def classify_path(
    path: LinkPath,
    vehicles: Iterable[OccludingVehicle],
    buildings: Iterable[OrientedRectangle],
    *,
    exclude_ids: Collection[str] = (),
) -> RFVisibility:
    """Classify an arbitrary radio path as LOS, NLOSv or NLOS."""

    blocked_by_building = building_obstructs(path, buildings)
    occlusion = occluding_vehicles(path, vehicles, exclude_ids=exclude_ids)

    if blocked_by_building:
        state = RFPropagationState.NLOS
    elif occlusion.is_blocked:
        state = RFPropagationState.NLOSV
    else:
        state = RFPropagationState.LOS

    return RFVisibility(
        state=state,
        vehicle_blocker_ids=occlusion.blocker_ids,
        building_blocked=blocked_by_building,
    )


def classify_pair(
    transmitter: VehicleFootprintSource,
    receiver: VehicleFootprintSource,
    vehicles: Iterable[OccludingVehicle],
    buildings: Iterable[OrientedRectangle],
    *,
    exclude_ids: Collection[str] | None = None,
    antenna_height_m: float | None = None,
) -> RFVisibility:
    """Classify the antenna-to-antenna link between two vehicles.

    ``exclude_ids`` defaults to the pair's own identifiers when both carry
    them, so a caller cannot accidentally have the endpoints obstruct their own
    link.
    """

    path = (
        rf_link_path(transmitter, receiver)
        if antenna_height_m is None
        else rf_link_path(transmitter, receiver, height_m=antenna_height_m)
    )

    if exclude_ids is None:
        endpoints = [getattr(vehicle, "vehicle_id", None) for vehicle in (transmitter, receiver)]
        exclude_ids = tuple(name for name in endpoints if isinstance(name, str))

    return classify_path(path, vehicles, buildings, exclude_ids=exclude_ids)


__all__ = [
    "RFVisibility",
    "building_obstructs",
    "classify_pair",
    "classify_path",
]
