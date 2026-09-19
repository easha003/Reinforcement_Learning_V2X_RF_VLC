"""Relative pose of a tagged pair, and the angles each link cares about.

Work plan section 5.2 lists what the optical link depends on: transmitter and
receiver pose, headlamp radiation direction, photodiode incidence angle and
field of view, and pair separation.  This module produces those quantities as
*angles*.  It does not convert them into optical power.

That division is the same one section 5.3 draws for blockage.  Geometry decides
where things point; the channel decides what that costs.  The measured
non-Lambertian headlamp pattern under ``vlc.pattern_artifact`` maps an emission
angle to a gain, and the concentrator maps an incidence angle to a gain; both
belong in ``channels/vlc``, not here.

**Field-of-view convention.**  ``vlc.receiver_fov_deg`` is read as the receiver
*semi-angle*, following the Kahn-Barry optical front-end model this project's
Lambertian and non-Lambertian treatment is built on, where the concentrator
admits incidence angles ``0 <= psi <= psi_c``.  A configured 60 deg therefore
means the photodiode accepts light arriving up to 60 deg off its boresight.
Functions here take an explicit half-angle so the interpretation is applied in
one visible place rather than assumed at each call site.

**Why this is not inert.**  Tagged pairs are formed only when the two headings
agree within 30 deg, and in a rectangular grid that means they agree exactly.
The heading test is applied at formation only; the continuation loop in
``mobility/tagged_pairs.py`` never re-checks it.  A pair therefore survives its
leader turning at a junction, at which point the photodiode has swung a quarter
turn away and receives nothing even though the path may be geometrically clear.
That is an outage mechanism no amount of occlusion testing would find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.geometry import Point, VehicleFootprintSource
from hybrid_v2x_rl.core.link_endpoints import (
    headlamp_position,
    optical_link_path,
    photodiode_position,
)

#: ``vlc.receiver_fov_deg`` default, read as a semi-angle; see the module docstring.
DEFAULT_FOV_HALF_ANGLE_RAD = math.radians(60.0)

#: Angular slack on the acceptance test.  The layer's stated convention is that
#: a ray exactly on the cone is accepted, and without a tolerance that is not
#: achievable: an incidence angle built from ``atan2`` lands within an ulp or
#: two of the boundary and falls on either side of it arbitrarily.  1e-12 rad is
#: roughly 6e-11 degrees, far below any physical or sensed angle.
_FOV_EPS_RAD = 1e-12


def wrap_to_pi(angle_rad: float) -> float:
    """Return ``angle_rad`` mapped into ``(-pi, pi]``."""

    wrapped = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    # ``fmod`` maps exactly -pi to -pi; normalise so the interval is half-open.
    return math.pi if wrapped == -math.pi else wrapped


def _bearing(origin: Point, target: Point) -> float:
    return math.atan2(target.y_m - origin.y_m, target.x_m - origin.x_m)


@dataclass(frozen=True, slots=True)
class PairGeometry:
    """Exact relative pose of one transmitter/receiver pair at one instant.

    This is hidden simulator state.  The policy sees a noisy, aged
    reconstruction of some of these fields and never this object.
    """

    separation_m: float
    """Front-to-front distance, matching how tagged-pair eligibility measures it."""

    optical_path_length_m: float
    """Headlamp-to-photodiode distance, shorter by the receiver's length."""

    bearing_rad: float
    """World-frame bearing from the headlamp to the photodiode."""

    relative_bearing_rad: float
    """Bearing to the receiver expressed in the transmitter's frame."""

    relative_heading_rad: float
    """Receiver heading minus transmitter heading, wrapped."""

    emission_angle_rad: float
    """Angle off the headlamp boresight, which points along the transmitter."""

    incidence_angle_rad: float
    """Angle off the photodiode boresight, which points behind the receiver."""

    fov_half_angle_rad: float

    @property
    def within_field_of_view(self) -> bool:
        """Whether the arriving ray falls inside the photodiode's acceptance cone.

        Equality counts as inside, matching the boundary convention used
        throughout the geometry layer, to within ``_FOV_EPS_RAD``.
        """

        return self.incidence_angle_rad <= self.fov_half_angle_rad + _FOV_EPS_RAD

    @property
    def fov_margin_rad(self) -> float:
        """Headroom before the receiver loses alignment; negative once lost."""

        return self.fov_half_angle_rad - self.incidence_angle_rad


def pair_geometry(
    transmitter: VehicleFootprintSource,
    receiver: VehicleFootprintSource,
    *,
    fov_half_angle_rad: float = DEFAULT_FOV_HALF_ANGLE_RAD,
) -> PairGeometry:
    """Return the exact relative pose of a following pair.

    ``transmitter`` is the follower and ``receiver`` its leader, per the
    longitudinal following geometry of work plan section 4.6.
    """

    lamp = headlamp_position(transmitter)
    diode = photodiode_position(receiver)
    bearing = _bearing(lamp, diode)

    # The photodiode looks backwards, so its boresight is the receiver's
    # heading reversed.  The ray arrives from the transmitter, along
    # ``bearing + pi``; the two pi terms cancel.
    incidence = abs(wrap_to_pi(bearing - receiver.heading_rad))

    return PairGeometry(
        separation_m=math.hypot(receiver.x_m - transmitter.x_m, receiver.y_m - transmitter.y_m),
        optical_path_length_m=optical_link_path(transmitter, receiver).length_m,
        bearing_rad=bearing,
        relative_bearing_rad=wrap_to_pi(bearing - transmitter.heading_rad),
        relative_heading_rad=wrap_to_pi(receiver.heading_rad - transmitter.heading_rad),
        emission_angle_rad=abs(wrap_to_pi(bearing - transmitter.heading_rad)),
        incidence_angle_rad=incidence,
        fov_half_angle_rad=fov_half_angle_rad,
    )


__all__ = [
    "DEFAULT_FOV_HALF_ANGLE_RAD",
    "PairGeometry",
    "pair_geometry",
    "wrap_to_pi",
]
