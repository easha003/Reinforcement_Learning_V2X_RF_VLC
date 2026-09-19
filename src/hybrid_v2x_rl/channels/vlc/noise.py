"""Electrical noise at the optical receiver.

Implementation spec section 12.4, and one instruction in it shapes the module:

    Night and high-ambient/day conditions are separate configurations, not a
    runtime boolean with undocumented constants.

So :class:`AmbientCondition` carries its own photocurrent and its own source
note, and there is no ``is_daytime`` flag anywhere. The reason is that the two
regimes are not a switch on one model -- they are different *limiting physics*.
At night the receiver is thermal-limited by its own load resistor; in daylight
the ambient shot noise from sunlight exceeds the signal by orders of magnitude
and the front end may not even stay in its linear range. A boolean would let a
caller flip between those with no record of which constants moved.

**The measured regime, stated because it decides which levers work.** At the
configured 5 MHz bandwidth with a 7.5 mm^2 detector on a 250 ohm load, thermal
noise exceeds signal shot noise by roughly three orders of magnitude across the
whole pair window. That is what makes an avalanche photodiode the effective
lever and a larger PIN detector a poor one: internal gain multiplies the signal
*before* the thermal term, while area buys signal and capacitance together and
the bandwidth constraint then forces the load resistance back down.

**This file is optical only.** It must not import
:mod:`hybrid_v2x_rl.channels.rf.bler`; a guard test enforces that. IM/DD detection is
not a complex-AWGN channel and the finite-blocklength expression does not
transfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.core.errors import HybridV2XError

#: Charge on the electron, exact by SI definition since 2019.
ELEMENTARY_CHARGE_C = 1.602176634e-19

#: Boltzmann's constant, exact by SI definition since 2019.
BOLTZMANN_J_PER_K = 1.380649e-23


class NoiseError(HybridV2XError):
    """Noise was evaluated with an invalid argument."""


@dataclass(frozen=True, slots=True)
class AmbientCondition:
    """One named lighting condition and the background photocurrent it induces.

    Named rather than parameterized so that a result always records *which*
    condition produced it. ``photocurrent_a`` is the DC current the background
    scene drives through the detector; it contributes shot noise but carries no
    information, and a real receiver removes its mean with AC coupling while
    keeping its variance.
    """

    name: str
    photocurrent_a: float
    source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.photocurrent_a) or self.photocurrent_a < 0.0:
            raise NoiseError("ambient photocurrent must be finite and non-negative",
                             context={"condition": self.name,
                                      "photocurrent_a": self.photocurrent_a})


#: The headline condition.  UNSOURCED order of magnitude: streetlights, other
#: vehicles' lamps and shop fronts on a lit urban street, not moonlight and not
#: sunlight.  It sits far below the thermal term at the configured bandwidth,
#: so the headline result is insensitive to it -- which is the only reason an
#: unsourced value is tolerable here, and the tolerance ends the moment an
#: avalanche detector makes the shot term competitive.
CLEAR_NIGHT = AmbientCondition(
    name="clear_night",
    photocurrent_a=1.0e-7,
    source="DECLARED order of magnitude for a lit urban street; see module docstring",
)

#: Retained so the day case is a *configuration* rather than an argument, and so
#: its absence from the headline is visible.  Derived rather than guessed: full
#: sun is about 1 kW/m^2, and 7.5 mm^2 at 0.4 A/W collects 7.5 mW and passes
#: 3 mA if nothing rejects the out-of-band power.  An optical filter cuts that,
#: by a factor this profile has not specified, so 3 mA is the unfiltered bound.
#:
#: It matters that this is ten times the 308 uA at which shot noise overtakes
#: the thermal term: **daylight changes which physics limits the receiver**, it
#: does not merely add to it.  A first attempt at this constant used 100 uA and
#: did not flip the regime, which would have quietly understated the daylight
#: case.  Nothing in the frozen profile selects it; work plan section 8.1 lists
#: daylight as a sensitivity that has not been run.
CLEAR_DAY = AmbientCondition(
    name="clear_day",
    photocurrent_a=3.0e-3,
    source="DERIVED from 1 kW/m^2 on an unfiltered 7.5 mm^2 detector at 0.4 A/W",
)

CONDITIONS = {condition.name: condition for condition in (CLEAR_NIGHT, CLEAR_DAY)}


def ambient_condition(name: str) -> AmbientCondition:
    """Look up a named condition, refusing an unknown one."""

    try:
        return CONDITIONS[name]
    except KeyError:
        raise NoiseError("unknown ambient condition",
                         context={"name": name, "known": sorted(CONDITIONS)}) from None


@dataclass(frozen=True, slots=True)
class NoisePower:
    """The variance budget at the detector output, in amperes squared."""

    signal_shot_a2: float
    ambient_shot_a2: float
    thermal_a2: float

    @property
    def total_a2(self) -> float:
        return self.signal_shot_a2 + self.ambient_shot_a2 + self.thermal_a2

    @property
    def is_thermal_limited(self) -> bool:
        """Whether the receiver's own electronics dominate the scene.

        Reported rather than assumed, because which term dominates decides
        which design lever works: internal gain helps a thermal-limited
        receiver and does nothing for a shot-limited one.
        """

        return self.thermal_a2 > self.signal_shot_a2 + self.ambient_shot_a2


def shot_noise_a2(photocurrent_a: float, bandwidth_hz: float) -> float:
    """``2 q I B`` -- the Poisson arrival of photoelectrons."""

    if photocurrent_a < 0.0 or not math.isfinite(photocurrent_a):
        raise NoiseError("photocurrent must be finite and non-negative",
                         context={"photocurrent_a": photocurrent_a})
    if bandwidth_hz <= 0.0 or not math.isfinite(bandwidth_hz):
        raise NoiseError("bandwidth must be finite and positive",
                         context={"bandwidth_hz": bandwidth_hz})
    return 2.0 * ELEMENTARY_CHARGE_C * photocurrent_a * bandwidth_hz


def thermal_noise_a2(receiver: OpticalReceiver, bandwidth_hz: float) -> float:
    """``4 k T F B / R_L`` -- Johnson noise in the load, scaled by the preamp.

    The preamplifier is one lumped excess-noise factor rather than a
    transconductance model with its own six constants. That is a declared
    simplification: at three orders below nothing it would change no result,
    and inventing six unsourced values to look more detailed than the evidence
    supports is the failure this project keeps a parameter table to avoid.
    """

    if bandwidth_hz <= 0.0 or not math.isfinite(bandwidth_hz):
        raise NoiseError("bandwidth must be finite and positive",
                         context={"bandwidth_hz": bandwidth_hz})
    return (
        4.0
        * BOLTZMANN_J_PER_K
        * receiver.noise_temperature_k
        * receiver.preamplifier_noise_factor
        * bandwidth_hz
        / receiver.load_resistance_ohm
    )


def noise_power(
    *,
    receiver: OpticalReceiver,
    received_optical_power_w: float,
    bandwidth_hz: float,
    ambient: AmbientCondition = CLEAR_NIGHT,
) -> NoisePower:
    """Assemble the three noise terms for one received power."""

    signal_a = receiver.photocurrent_a(received_optical_power_w)
    return NoisePower(
        signal_shot_a2=shot_noise_a2(signal_a, bandwidth_hz),
        ambient_shot_a2=shot_noise_a2(ambient.photocurrent_a, bandwidth_hz),
        thermal_a2=thermal_noise_a2(receiver, bandwidth_hz),
    )


def electrical_snr(
    *,
    receiver: OpticalReceiver,
    received_optical_power_w: float,
    bandwidth_hz: float,
    ambient: AmbientCondition = CLEAR_NIGHT,
) -> float:
    """``(R P)^2 / sigma^2`` -- the ratio OOK detection actually sees.

    Squared in the numerator because an IM/DD receiver detects *power*, so a
    decibel of optical loss costs two decibels of electrical SNR. That factor
    of two is why the optical link is so much more distance-sensitive than the
    radio: received power falls as 1/d^2 and electrical SNR as 1/d^4.
    """

    signal_a = receiver.photocurrent_a(received_optical_power_w)
    noise = noise_power(
        receiver=receiver,
        received_optical_power_w=received_optical_power_w,
        bandwidth_hz=bandwidth_hz,
        ambient=ambient,
    )
    if noise.total_a2 <= 0.0:
        raise NoiseError("noise power vanished; the receiver is unphysical")
    return signal_a * signal_a / noise.total_a2


__all__ = [
    "BOLTZMANN_J_PER_K",
    "CLEAR_DAY",
    "CLEAR_NIGHT",
    "CONDITIONS",
    "ELEMENTARY_CHARGE_C",
    "AmbientCondition",
    "NoiseError",
    "NoisePower",
    "ambient_condition",
    "electrical_snr",
    "noise_power",
    "shot_noise_a2",
    "thermal_noise_a2",
]
