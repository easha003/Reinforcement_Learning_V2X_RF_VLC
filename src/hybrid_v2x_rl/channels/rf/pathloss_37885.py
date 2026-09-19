"""TR 37.885 urban-grid path loss, given a propagation class.

Work plan section 7.1 fixes the large-scale channel to the 3GPP TR 37.885
urban-grid LOS / NLOSv / NLOS family.  This module supplies the loss *given* a
class; it never decides the class.  :mod:`hybrid_v2x_rl.geometry.rf_visibility` does
that from geometry alone, and its docstring records why the two must not merge:
a class that cannot be derived from where obstacles are cannot be predicted
from tracked positions, and the observation vector's forecast would then carry
no information.

**The frequency-flat / frequency-selective split is load-bearing.**  Everything
here is *flat across the 10 MHz carrier*: median path loss is a function of
distance alone, and vehicle blockage is a shadowing term set by the geometry of
one obstructing body.  Neither is improved by hopping a retransmission to
another subchannel.  Small-scale fading and Mode-2 collision are the
frequency-selective mechanisms and live elsewhere, because that is the
distinction the paper's central ablation rests on:

* if RF failure is one aggregate BLER, a second hopped attempt improves it
  uniformly and DUP-versus-RF×2 at equal cost measures nothing;
* if the two mechanisms are separable, a hopped attempt fixes the fading tail
  and leaves the geometry tail, which is exactly the asymmetry that makes
  cross-medium diversity different from within-medium diversity.

**Correlation with V-VLC enters here, not by assumption.**  NLOSv is the same
physical event that occludes the optical path, so RF and VLC failure are
coupled through this module's input rather than through a fitted correlation
coefficient.  Section 8.3 requires the joint failure probability to be logged
rather than assumed independent, and this is where the dependence originates.

**Sourcing status: verified.**  Every coefficient below was checked against the
ns-3 implementation of Table 6.2.1-1 and against an independent quotation of the
same expressions.  The check was worth doing: it found the NLOS coefficients
cyclically permuted, a 4.6 dB understatement at 5 m turning into a 4.3 dB
overstatement at 100 m, and the NLOSv blockage spread applied as one value where
the document gives two.  Naming the constants rather than inlining them is what
made the correction a five-line edit.

Two consequences of those constants meeting this configuration were measured
rather than assumed, and both narrow what NLOSv can mean here.  Both survive the
correction, because the blockage coefficients were right all along.

**The distance term is inert, and that is correct rather than a transcription
error.**  ``max(0, 15 log10(d) - 41)`` turns positive
only beyond **541 m**, which is outside the 5-100 m tagged-pair window and
outside any V2V range that matters.  The coefficients 15 and 41 are confirmed,
so the term genuinely targets ranges this study never reaches.  Within the window, vehicle blockage is a
flat offset with no distance dependence, so blockage *severity* cannot be
inferred from separation -- only blockage *presence* can.  A feature that tries
to read severity off pair distance is reading nothing.

**A passenger car is free.**  Antennas sit at 1.5 m and a car is exactly
1.5 m, so it lands on the ``<=`` boundary and contributes zero RF loss, while
the same body severs the 0.7 m optical path completely.  Cars are 90% of the
mixture.  Read one way that is the contribution's mechanism stated in decibels:
the most common blocker is a total outage for V-VLC and free for the radio.
Read another way it is a boundary case carrying a great deal of weight, because
a body at exactly antenna height obstructs a real Fresnel zone substantially.
Both readings are live; the tie is a property of the fleet and the antenna
height, not of TR 37.885, and it should be broken deliberately -- by sourcing
the constants, by raising the antenna, or by giving cars a distribution of
heights rather than one value -- rather than left to a ``<=``.

A third case is unreachable: with Tx and Rx at equal heights the "between the
two antennas" branch cannot fire, so the 5 dB partial-obstruction mean is dead
in this profile.  It is kept because unequal heights are a legitimate future
configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError

#: Below this separation the log-distance forms are extrapolated beyond the
#: measurement range they were fitted over, so callers are refused rather than
#: silently handed a number.  The tagged-pair window starts at 5 m, so this
#: never fires for a tagged link; it guards neighbour links, which can be
#: arbitrarily close.
MIN_VALID_DISTANCE_M = 1.0

#: TR 37.885 urban LOS: ``PL = 38.77 + 16.7 log10(d) + 18.2 log10(fc)``,
#: distance in metres, carrier in GHz.
LOS_INTERCEPT_DB = 38.77
LOS_DISTANCE_COEFFICIENT = 16.7
LOS_FREQUENCY_COEFFICIENT = 18.2
LOS_SHADOWING_SIGMA_DB = 3.0

#: TR 37.885 urban NLOS: ``PL = 36.85 + 30 log10(d) + 18.9 log10(fc)``.
#: VERIFIED against the ns-3 implementation of Table 6.2.1-1 and an independent
#: quotation of the same expression.  An earlier revision of this module had all
#: three coefficients cyclically permuted -- intercept 18.9, distance 36.85,
#: frequency 30.0 -- which understated NLOS loss by 4.6 dB at 5 m and overstated
#: it by 4.3 dB at 100 m.  The tell was physical rather than numerical: an NLOS
#: intercept of 18.9 dB against a LOS intercept of 38.77 dB is a 20 dB gap
#: between two quantities that are both "loss at one metre" and should sit in
#: the same range.
NLOS_INTERCEPT_DB = 36.85
NLOS_DISTANCE_COEFFICIENT = 30.0
NLOS_FREQUENCY_COEFFICIENT = 18.9
NLOS_SHADOWING_SIGMA_DB = 4.0

#: NLOSv is LOS propagation plus a blockage term, so it inherits the LOS
#: shadowing spread and adds the variate below.
NLOSV_SHADOWING_SIGMA_DB = LOS_SHADOWING_SIGMA_DB

#: Additional vehicle blockage loss, in dB, as a normal variate.  VERIFIED.
#: The spread differs by case: 4.5 dB when both antennas sit below the blocker,
#: 4.0 dB when the blocker falls between them.  The earlier revision used 4.5
#: for both, which widened the mixed case by half a decibel.
NLOSV_BLOCKAGE_MEAN_SHORT_DB = 5.0
NLOSV_BLOCKAGE_MEAN_TALL_DB = 9.0
NLOSV_BLOCKAGE_DISTANCE_COEFFICIENT = 15.0
NLOSV_BLOCKAGE_DISTANCE_OFFSET_DB = 41.0
NLOSV_BLOCKAGE_SIGMA_TALL_DB = 4.5
NLOSV_BLOCKAGE_SIGMA_MIXED_DB = 4.0


class PathLossError(HybridV2XError):
    """A path loss was requested outside the model's validity."""


@dataclass(frozen=True, slots=True)
class LargeScaleLoss:
    """The frequency-flat part of one link's budget.

    Separated from the frequency-selective part deliberately; see the module
    docstring.  ``shadowing_sigma_db`` is *not* a realized sample: it is the
    spread the spatially-correlated shadowing process must use for this class,
    and section 7.1 requires that process to be correlated rather than redrawn
    per packet.
    """

    state: RFPropagationState
    median_db: float
    blockage_db: float
    shadowing_sigma_db: float

    @property
    def total_db(self) -> float:
        """Median loss plus vehicle blockage, before shadowing and fading."""

        return self.median_db + self.blockage_db

    @property
    def improves_with_frequency_hopping(self) -> bool:
        """Whether a retransmission on another subchannel helps this term.

        Always false.  Stated as a property rather than left implicit because
        the retransmission model must ask it, and because the answer being
        uniformly false for this module is the whole point of the split.
        """

        return False


def median_path_loss_db(
    distance_m: float, carrier_hz: float, state: RFPropagationState
) -> float:
    """Median TR 37.885 urban path loss for ``state`` at ``distance_m``.

    NLOSv shares the LOS median: the blockage is an additive term, not a
    different propagation exponent, which is precisely why TR 37.885 prices it
    as extra loss the radio usually survives while an occluded optical path
    delivers nothing at all.
    """

    if not math.isfinite(distance_m) or distance_m < MIN_VALID_DISTANCE_M:
        raise PathLossError(
            "path loss requested below the model's validity range",
            context={"distance_m": distance_m, "minimum_m": MIN_VALID_DISTANCE_M},
        )
    if not math.isfinite(carrier_hz) or carrier_hz <= 0.0:
        raise PathLossError("carrier frequency must be finite and positive",
                            context={"carrier_hz": carrier_hz})

    carrier_ghz = carrier_hz / 1e9
    log_distance = math.log10(distance_m)
    log_carrier = math.log10(carrier_ghz)

    if state is RFPropagationState.NLOS:
        return (
            NLOS_INTERCEPT_DB
            + NLOS_DISTANCE_COEFFICIENT * log_distance
            + NLOS_FREQUENCY_COEFFICIENT * log_carrier
        )
    return (
        LOS_INTERCEPT_DB
        + LOS_DISTANCE_COEFFICIENT * log_distance
        + LOS_FREQUENCY_COEFFICIENT * log_carrier
    )


def shadowing_sigma_db(state: RFPropagationState) -> float:
    """Spread of the spatially-correlated shadowing process for ``state``."""

    if state is RFPropagationState.NLOS:
        return NLOS_SHADOWING_SIGMA_DB
    if state is RFPropagationState.NLOSV:
        return NLOSV_SHADOWING_SIGMA_DB
    return LOS_SHADOWING_SIGMA_DB


def blockage_mean_db(
    distance_m: float,
    blocker_height_m: float,
    tx_height_m: float,
    rx_height_m: float,
) -> float:
    """Mean additional loss from one obstructing vehicle, in dB.

    Three cases, decided by how the blocker's roof compares with the two
    antennas:

    * below both -- the first Fresnel zone is essentially clear, so no loss;
    * between them -- partial obstruction;
    * above both -- full obstruction, the larger mean.

    The distance term is floored at zero, so a nearby blocker contributes only
    the base mean.  Note the interaction with the vehicle mixture: antennas sit
    at 1.5 m and the *shortest* vehicle class is also 1.5 m, so in this model
    the "below both" case never fires for a real blocker.  That is a property
    of the fleet, not of the propagation model, and it is why the height test
    almost never discriminates -- section 3 of the system model records the
    same fact for the optical link.
    """

    lower, upper = sorted((tx_height_m, rx_height_m))
    if blocker_height_m <= lower:
        return 0.0

    base = (
        NLOSV_BLOCKAGE_MEAN_TALL_DB
        if blocker_height_m > upper
        else NLOSV_BLOCKAGE_MEAN_SHORT_DB
    )
    growth = max(
        0.0,
        NLOSV_BLOCKAGE_DISTANCE_COEFFICIENT * math.log10(distance_m)
        - NLOSV_BLOCKAGE_DISTANCE_OFFSET_DB,
    )
    return base + growth


def blockage_sigma_db(
    blocker_height_m: float, tx_height_m: float, rx_height_m: float
) -> float:
    """Spread of the blockage variate, which differs by case.

    Wider when the blocker stands above both antennas than when it falls
    between them: a fully obstructing body admits a broader range of
    diffraction outcomes than a partially obstructing one.
    """

    lower, upper = sorted((tx_height_m, rx_height_m))
    if blocker_height_m <= lower:
        return 0.0
    return (
        NLOSV_BLOCKAGE_SIGMA_TALL_DB
        if blocker_height_m > upper
        else NLOSV_BLOCKAGE_SIGMA_MIXED_DB
    )


def large_scale_loss(
    *,
    distance_m: float,
    carrier_hz: float,
    state: RFPropagationState,
    blockage_db: float = 0.0,
) -> LargeScaleLoss:
    """Assemble the frequency-flat budget for one link.

    ``blockage_db`` is a *realized* sample for NLOSv, drawn by the caller from
    :func:`blockage_mean_db` and :data:`NLOSV_BLOCKAGE_SIGMA_DB`.  It is passed
    in rather than drawn here so that this module stays deterministic and the
    randomness lives with the named-stream generator, which is what makes a
    trace reproducible bit-exactly.
    """

    if state is not RFPropagationState.NLOSV and blockage_db:
        raise PathLossError(
            "vehicle blockage loss applies only to NLOSv",
            context={"state": str(state), "blockage_db": blockage_db},
        )
    return LargeScaleLoss(
        state=state,
        median_db=median_path_loss_db(distance_m, carrier_hz, state),
        blockage_db=blockage_db,
        shadowing_sigma_db=shadowing_sigma_db(state),
    )


__all__ = [
    "LargeScaleLoss",
    "MIN_VALID_DISTANCE_M",
    "NLOSV_BLOCKAGE_SIGMA_MIXED_DB",
    "NLOSV_BLOCKAGE_SIGMA_TALL_DB",
    "blockage_sigma_db",
    "PathLossError",
    "blockage_mean_db",
    "large_scale_loss",
    "median_path_loss_db",
    "shadowing_sigma_db",
]
