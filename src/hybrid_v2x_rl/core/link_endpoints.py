"""Where each link physically begins and ends on a vehicle.

The RF and V-VLC links do not share endpoints, and neither of them sits at the
position the trace stores.  Getting this wrong shortens or lengthens every
obstruction query, so the offsets live here rather than at each call site.

**V-VLC** runs from the transmitting vehicle's headlamp to the receiving
vehicle's photodiode.  In the longitudinal following geometry of work plan
section 4.6 the transmitter is the follower and the receiver is its leader, so
the beam leaves the follower's *front* and arrives at the leader's *rear*.  The
stored position is already the front bumper, so the transmit end needs no
correction while the receive end must be moved back one full vehicle length.

**RF** runs roof to roof, taken at the centre of each vehicle.

The RF path is horizontal: both antennas sit at 1.5 m.  The optical path is
**not** -- the photodiode sits 0.3 m below the headlamp, which is where rear
lamps and reflectors actually are.  The two were equal until it was noticed that
co-height endpoints sample an ECE low beam exactly along its cut-off, the one
elevation a headlamp is engineered to keep dark.  That was a modelling
convenience, not a design choice, and it cost 7 to 14 dB depending on the gap.

Occlusion is unaffected by the change: the shortest vehicle class is 1.5 m,
which is taller than either endpoint, so every body blocks either path and the
scalar height comparison returns the same answer at 0.4 m as at 0.7 m.  A
sloped path therefore still reduces to a planar footprint test plus a scalar
height comparison, with no interpolation
along the path.  :mod:`hybrid_v2x_rl.geometry.vehicle_occlusion` relies on this.
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_v2x_rl.core.geometry import (
    Point,
    Segment,
    VehicleFootprintSource,
    vehicle_rear,
    vehicle_rectangle,
)

#: Defaults matching ``configs/project/default.yaml`` under ``geometry``.
DEFAULT_HEADLAMP_HEIGHT_M = 0.7
#: Rear lamps and reflectors sit at 0.35-0.6 m on real vehicles; 0.4 m is the
#: ordinary choice rather than a favourable one.  See the module docstring for
#: why this is no longer equal to the headlamp height.
DEFAULT_PHOTODIODE_HEIGHT_M = 0.55
DEFAULT_RF_ANTENNA_HEIGHT_M = 1.5


@dataclass(frozen=True, slots=True)
class LinkPath:
    """The planar path of one link, with the height it is carried at.

    ``height_m`` is the common height of both endpoints.  Because the path is
    horizontal, a vehicle obstructs it when its footprint meets ``segment`` and
    it stands at least ``height_m`` tall.
    """

    segment: Segment
    height_m: float

    @property
    def length_m(self) -> float:
        return self.segment.length_m


def headlamp_position(transmitter: VehicleFootprintSource) -> Point:
    """Return the transmitting headlamp's planar position.

    This is the vehicle's front bumper, which is what the trace stores.
    """

    return Point(transmitter.x_m, transmitter.y_m)


def photodiode_position(receiver: VehicleFootprintSource) -> Point:
    """Return the receiving photodiode's planar position.

    The photodiode faces rearward to see a following transmitter, so it sits on
    the trailing edge, one full vehicle length behind the stored position.
    """

    return vehicle_rear(receiver)


def antenna_position(vehicle: VehicleFootprintSource) -> Point:
    """Return the roof antenna's planar position, taken at the body centre."""

    return vehicle_rectangle(vehicle).centre


def optical_link_path(
    transmitter: VehicleFootprintSource,
    receiver: VehicleFootprintSource,
    *,
    height_m: float = DEFAULT_HEADLAMP_HEIGHT_M,
) -> LinkPath:
    """Return the headlamp-to-photodiode path for a following pair."""

    return LinkPath(
        segment=Segment(headlamp_position(transmitter), photodiode_position(receiver)),
        height_m=height_m,
    )


def rf_link_path(
    transmitter: VehicleFootprintSource,
    receiver: VehicleFootprintSource,
    *,
    height_m: float = DEFAULT_RF_ANTENNA_HEIGHT_M,
) -> LinkPath:
    """Return the antenna-to-antenna path for a pair."""

    return LinkPath(
        segment=Segment(antenna_position(transmitter), antenna_position(receiver)),
        height_m=height_m,
    )


__all__ = [
    "DEFAULT_HEADLAMP_HEIGHT_M",
    "DEFAULT_PHOTODIODE_HEIGHT_M",
    "DEFAULT_RF_ANTENNA_HEIGHT_M",
    "LinkPath",
    "antenna_position",
    "headlamp_position",
    "optical_link_path",
    "photodiode_position",
    "rf_link_path",
]
