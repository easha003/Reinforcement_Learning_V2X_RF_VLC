"""Direct-path received optical power, from a measured beam and a geometry.

Implementation spec section 12.3.  One expression carries the module:

    P_r = I(theta_h, theta_v) * A_eff(psi) / d^2

with the beam supplied by a :class:`~hybrid_v2x_rl.channels.vlc.headlamp_pattern.HeadlampPattern`,
the effective area by :class:`~hybrid_v2x_rl.channels.vlc.receiver.OpticalReceiver`, and
``d`` the headlamp-to-photodiode distance.  Occlusion collapses it to zero, or
to a calibrated residual floor when one exists.

**The artifact now exists and this module runs.**  ``headlamp-ece-r112-v1``
supplies the beam as a regulatory envelope derived from ECE R112 test-point
geometry -- a bound rather than a measured lamp, and labelled as such in its own
manifest.  Nothing in this file constructs a Lambertian lobe to fill any gap:
spec 12.1 permits Lambertian only as an explicitly named comparison model, and a
cosine lobe standing in for a measured beam produces exactly the claim W17 warns
against while looking identical to a real result in every log.

Five consequences of this project's geometry meeting this expression were
worked out rather than assumed.  Each narrows what the optical link can mean
here, and each is pinned in ``tests/unit/test_vlc_optical_gain.py``.

**1. The distance in the inverse-square law is the bumper gap, not the pair
separation.**  The photodiode sits one full vehicle length behind the stored
position, so for a same-lane pair ``d`` is exactly the car-following gap.  The
IDM holds that gap at ``s0 + v T`` with ``s0 = 2.5 m`` and ``T = 1.2 s``, so the
optical path *shortens as traffic slows*: 10.1 m at 10 veh/lane-km, 7.4 m at
20, 5.8 m at 30 on the desired gap, and 10.6 / 7.5 / 5.8 m once the IDM's
free-flow correction is included.  Received power goes as ``1/d^2``, so the
link budget **improves by 4.8 dB optical and 9.5 dB electrical** from the
lightest to the heaviest density, 5.2 / 10.4 dB on the corrected gaps -- the
same span as the entire 60-to-30-degree concentrator
sensitivity, and pointing the same way as the measured geometric outage, which
also falls with density (17.4 / 14.4 / 10.3% over the campaign; an earlier
reading of 3.4 / 1.9 / 1.0% counted only the receiver's cone and only the trace's
seeded opening formation).  Both V-VLC mechanisms therefore
strengthen exactly where RF congestion is worst.  Note what that makes of the
time headway: work plan section 4.7 item 1 flags it as undeclared and treats it
as a *capacity* parameter, and it is in fact the leading term in the optical
link budget.  Sourcing it is no longer only a realism question.

**2. Leader length is a link-budget lever the policy cannot see.**  At a fixed
20 m separation the optical path is 15.5 m behind a passenger car and 9.0 m
behind an 11 m bus -- 4.7 dB optical, 9.4 dB electrical, again the size of the
whole concentrator sensitivity.  The observation vector carries
``pair_distance`` and no vehicle class, so this variation is unobservable to
the policy and is not represented as a source of spread in the section 14.1
frontier tables, which read a single PER off a single separation.

**3. The vertical axis is sampled, and moving the photodiode is what made it
so.**  This finding originally read that the vertical angle was identically
zero, because both endpoints sat at 0.7 m on a planar road.  That pinning is
what hid the fact that the link was sampling an ECE low beam along its cut-off.
With the photodiode at 0.55 m the beam is read at ``-atan2(0.15, d)``, which is
-0.86 deg at a 10 m path -- the 50V/50R hot-spot row -- and steepens as the pair
closes, reaching ECE R112's 4D limit only at 2.15 m, under the IDM's 2.5 m
minimum gap.  An earlier 0.40 m photodiode gave a 0.30 m drop, reached 4D at
4.29 m, and put 56% of density-30 pair-instants below anything the regulation
describes.  The vertical axis
carries the legally mandated cutoff -- the single feature that most
distinguishes a real automotive beam from a cosine lobe -- so it is now
exercised rather than bypassed.  Roads are still planar and there is still no
pitch or grade model, which is the gap section 4.7 item 6 records.

**4. The obvious geometry field is the wrong one.**  ``PairGeometry`` offers
``emission_angle_rad``, which is an absolute value.  Feeding it to an
asymmetric beam folds the lamp's left half onto its right and silently
symmetrizes the pattern whose asymmetry is the reason for having it.  The
signed quantity is ``relative_bearing_rad``, and it needs a sign flip:
headings are measured counter-clockwise, so a positive relative bearing is a
target to the vehicle's *left*, while the pattern's horizontal axis is positive
to the right.  :func:`horizontal_emission_angle_rad` is that one-line
conversion, named so it can be tested rather than inlined at call sites.

**5. Both ends of the link limit alignment, and the regulation limits what can
be said about either.**  ECE R112 specifies the passing beam only within
9L..9R by 4U..4D.  The transmitter-side share of geometric outage therefore
depends on what is assumed outside that box, which is why the pattern artifact
ships in ``wide`` and ``narrow`` variants and why results are reported as a band
across them.  An earlier reading -- that the +/-9 deg horizontal limit dominates
the 60 deg receiver cone by threefold -- was measured against an artifact whose
own angular extent produced the cliff, and does not survive a smooth compliant
beam.  See the alignment short-circuit in :mod:`hybrid_v2x_rl.channels.vlc.model`.

**Validity has a near end, and unlike the radio's it is inside the configured
window.**  The point-source form ``I / d^2`` assumes the far field of an
extended source; a headlamp lens is 0.1-0.3 m across, so it holds from roughly
1 m outward.  The tagged-pair window admits separations from 5 m, which behind
a 4.5 m car is a 0.5 m optical path and behind a 5.2 m van is a *negative* one
-- overlapping footprints.  The RF module's equivalent guard never fires for a
tagged link; this one guards a region the admission window nominally permits,
and only the IDM's 2.5 m minimum gap keeps real pairs above it, with a factor
of 2.5 to spare.  Refusing is the right response *in the primitive*: ``1/d^2``
at 0.5 m returns a large, entirely fictional number.

The campaign then showed the guard firing for real, because queued traffic at
density 30 closes below the IDM's steady-state gap.  So the two layers differ
deliberately: :func:`direct_received_power_w` still refuses, and
:func:`received_power_from_geometry` holds the distance at the floor and says
why.  Holding is conservative -- it awards *less* power than the pair has, into
a 40 dB margin -- and it cannot turn a failure into a success.  Below roughly
0.8 m the question does not arise: the vertical emission angle has by then
steepened past the pattern's envelope and the pair is an alignment failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.channels.vlc.headlamp_pattern import HeadlampPattern
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.link_endpoints import (
    DEFAULT_HEADLAMP_HEIGHT_M,
    DEFAULT_PHOTODIODE_HEIGHT_M,
)
from hybrid_v2x_rl.core.pair_geometry import PairGeometry

#: Below this headlamp-to-photodiode distance the point-source radiant-intensity
#: form is outside its far-field validity and is refused rather than
#: extrapolated.  Five times a 0.2 m lens aperture.  UNVERIFIED -- the aperture
#: is a plausible automotive figure and the factor of five is the conventional
#: photometric-distance rule, neither taken from a source.
MIN_VALID_PATH_LENGTH_M = 1.0

#: The vertical emission angle a configuration with both endpoints at one
#: height produces on a planar road.  Retained as the explicit
#: no-height-difference case; the headline profile is *not* it, since the
#: photodiode sits at 0.55 m against the headlamp's 0.7 m.  See finding 3.
PLANAR_VERTICAL_EMISSION_RAD = 0.0


class OpticalGainError(HybridV2XError):
    """Received optical power was requested outside the model's validity."""


@dataclass(frozen=True, slots=True)
class BlockageModel:
    """What an occluded optical path delivers.

    The headline configuration sets ``complete_blockage_main: true`` and
    ``residual_optical_floor_enabled: false``, which is this class's default: a
    blocked path delivers nothing.  That is the physically defensible position
    for a 0.7 m beam cut by a vehicle body, and it is what makes the
    RF/V-VLC asymmetry the contribution rests on real -- the same body that
    costs the radio a few decibels costs the optical link everything.

    The residual floor is a *sensitivity*, and work plan section 8.1 calls it a
    **calibrated** residual floor.  Enabling it without a calibrated value is
    therefore refused: an uncalibrated floor is a free parameter that converts
    every geometric outage into a survivable link, which is the one direction
    the result must not be nudged by an invented number.
    """

    complete: bool = True
    residual_floor_w: float | None = None

    def __post_init__(self) -> None:
        if self.complete and self.residual_floor_w is not None:
            raise OpticalGainError(
                "complete blockage and a residual floor are contradictory; choose one",
                context={"residual_floor_w": self.residual_floor_w},
            )
        if not self.complete:
            if self.residual_floor_w is None:
                raise OpticalGainError(
                    "the residual optical floor is a calibrated sensitivity (work plan "
                    "section 8.1); enabling it without a calibrated value would turn "
                    "every geometric outage into a survivable link on an invented number"
                )
            if not math.isfinite(self.residual_floor_w) or self.residual_floor_w < 0.0:
                raise OpticalGainError("residual floor must be finite and non-negative",
                                       context={"w": self.residual_floor_w})

    @property
    def occluded_power_w(self) -> float:
        """Power delivered when the direct path is severed."""

        if self.complete or self.residual_floor_w is None:
            return 0.0
        return self.residual_floor_w


#: The configured model: a blocked path delivers nothing.
COMPLETE_BLOCKAGE = BlockageModel()


def horizontal_emission_angle_rad(relative_bearing_rad: float) -> float:
    """Convert a geometry-frame relative bearing into the pattern's frame.

    A sign flip and nothing else, but the sign is load-bearing.  Headings and
    bearings are counter-clockwise-positive, so a target at ``+30 degrees``
    relative bearing lies to the transmitter's **left**; the headlamp pattern's
    horizontal axis is positive toward the vehicle's **right**, because that is
    the axis a real beam is asymmetric about when it is aimed away from oncoming
    traffic.  Getting this backwards mirrors the beam, which is invisible on a
    symmetric comparison model and wrong by whatever the measured asymmetry is
    on a real one.
    """

    if not math.isfinite(relative_bearing_rad):
        raise OpticalGainError("relative bearing must be finite",
                               context={"relative_bearing_rad": relative_bearing_rad})
    return -relative_bearing_rad


def direct_received_power_w(
    *,
    pattern: HeadlampPattern,
    receiver: OpticalReceiver,
    path_length_m: float,
    horizontal_emission_rad: float,
    incidence_angle_rad: float,
    occluded: bool = False,
    vertical_emission_rad: float = PLANAR_VERTICAL_EMISSION_RAD,
    blockage: BlockageModel = COMPLETE_BLOCKAGE,
) -> float:
    """Optical power reaching the photodiode over the direct path, in watts.

    Occlusion is decided by the caller and passed in, exactly as the RF module
    takes its propagation class rather than deriving it: a channel that decided
    its own blockage from vehicle positions could not be predicted from tracked
    positions either, and the observation forecast would have nothing to
    forecast.

    When ``occluded`` the pattern is not sampled at all.  That is not an
    optimization -- it is the statement that a severed path delivers the
    blockage model's power regardless of how brightly the lamp was pointing,
    and sampling the beam first would invite a later edit that lets a bright
    beam leak through a bus.

    No field-of-view test is applied here or in the effective area.  The
    acceptance cone is tested in exactly one place, and the returned power is
    conditional on that test having passed.
    """

    if occluded:
        return blockage.occluded_power_w

    if not math.isfinite(path_length_m) or path_length_m < MIN_VALID_PATH_LENGTH_M:
        raise OpticalGainError(
            "optical path length is below the far-field validity of the point-source "
            "radiant-intensity form; note that this is reachable inside the configured "
            "tagged-pair separation window, unlike the RF module's equivalent guard",
            context={
                "path_length_m": path_length_m,
                "minimum_m": MIN_VALID_PATH_LENGTH_M,
            },
        )

    intensity = pattern.radiant_intensity_w_per_sr(horizontal_emission_rad, vertical_emission_rad)
    if intensity < 0.0:
        raise OpticalGainError("radiant intensity cannot be negative",
                               context={"w_per_sr": intensity})

    area = receiver.effective_area_m2(incidence_angle_rad)
    return intensity * area / (path_length_m * path_length_m)


def _emission_elevation(
    geometry: PairGeometry, tx_height_m: float, rx_height_m: float
) -> tuple[float, float]:
    """The height drop and the resulting downward emission angle."""

    drop_m = tx_height_m - rx_height_m
    return drop_m, -math.atan2(drop_m, max(geometry.optical_path_length_m, 1e-9))


def emission_direction_rad(
    geometry: PairGeometry,
    *,
    tx_height_m: float = DEFAULT_HEADLAMP_HEIGHT_M,
    rx_height_m: float = DEFAULT_PHOTODIODE_HEIGHT_M,
) -> tuple[float, float]:
    """Where the lamp has to point for this pose, in the pattern's own frame.

    Exposed so a caller can ask the pattern whether it covers that direction
    *before* sampling it, without reimplementing the sign flip and the elevation
    -- the two conversions the adapter exists to centralize. Duplicating them at
    a call site is how the beam ends up sampled at the absolute bearing again.
    """

    _, elevation_rad = _emission_elevation(geometry, tx_height_m, rx_height_m)
    return horizontal_emission_angle_rad(geometry.relative_bearing_rad), elevation_rad


def beam_covers_pair(
    pattern: HeadlampPattern,
    geometry: PairGeometry,
    *,
    tx_height_m: float = DEFAULT_HEADLAMP_HEIGHT_M,
    rx_height_m: float = DEFAULT_PHOTODIODE_HEIGHT_M,
) -> bool:
    """Whether the lamp's pattern can answer for this pair's direction."""

    horizontal, vertical = emission_direction_rad(
        geometry, tx_height_m=tx_height_m, rx_height_m=rx_height_m
    )
    return pattern.covers(horizontal, vertical)


def received_power_from_geometry(
    *,
    pattern: HeadlampPattern,
    receiver: OpticalReceiver,
    geometry: PairGeometry,
    occluded: bool = False,
    blockage: BlockageModel = COMPLETE_BLOCKAGE,
    tx_height_m: float = DEFAULT_HEADLAMP_HEIGHT_M,
    rx_height_m: float = DEFAULT_PHOTODIODE_HEIGHT_M,
) -> float:
    """Received optical power for one exact pair pose.

    The adapter exists so the two easy mistakes are made in one place instead of
    at every call site: the distance is ``optical_path_length_m`` and not
    ``separation_m``, and the beam is sampled at the *signed*
    ``relative_bearing_rad`` and not at the absolute ``emission_angle_rad``.
    See findings 1, 2 and 4 in the module docstring.

    The vertical angle is now a real argument.  It was pinned to zero while both
    endpoints sat at 0.7 m, and that pinning is what hid the fact that the link
    was sampling an ECE low beam along its cut-off.  With the photodiode at
    0.4 m the beam is sampled below the horizon, where a headlamp actually puts
    its light, and the pattern's cut-off axis starts to matter.

    Roads are still planar and there is still no pitch model, so the elevation
    comes only from the fixed height difference and the gap.  The slant range is
    ignored: 0.3 m of drop over a 5 m gap is 0.2% of the distance, which is far
    inside every other uncertainty in this budget.
    """

    _, elevation_rad = _emission_elevation(geometry, tx_height_m, rx_height_m)

    # A pair closer than the photometric distance is held *at* it rather than
    # refused. The primitive still refuses, because extrapolating a point-source
    # form into the near field is a modelling error there; here it is a traffic
    # state. At density 30 vehicles queue bumper to bumper and the tenth
    # percentile optical path is 2.5 m, so sub-metre pairs are a routine
    # occurrence and raising would end a campaign run partway through.
    #
    # Holding the distance is conservative in the only direction that matters:
    # 1/d^2 means a pair reported at 1.0 m when it is really at 0.9 m is
    # credited with 0.9 dB *less* power than it has, against a link margin of
    # roughly 40 dB at that range. The clamp cannot turn a failure into a
    # success -- it can only fail to award power that is already superfluous.
    path_length_m = max(geometry.optical_path_length_m, MIN_VALID_PATH_LENGTH_M)
    return direct_received_power_w(
        pattern=pattern,
        receiver=receiver,
        path_length_m=path_length_m,
        horizontal_emission_rad=horizontal_emission_angle_rad(geometry.relative_bearing_rad),
        incidence_angle_rad=geometry.incidence_angle_rad,
        occluded=occluded,
        vertical_emission_rad=elevation_rad,
        blockage=blockage,
    )


__all__ = [
    "COMPLETE_BLOCKAGE",
    "MIN_VALID_PATH_LENGTH_M",
    "PLANAR_VERTICAL_EMISSION_RAD",
    "BlockageModel",
    "OpticalGainError",
    "beam_covers_pair",
    "direct_received_power_w",
    "emission_direction_rad",
    "horizontal_emission_angle_rad",
    "received_power_from_geometry",
]
