"""RF block error from a realized SINR, by the finite-blocklength approximation.

Work plan section 7.3 permits either calibrated transport-block BLER curves or
"a finite-blocklength approximation whose channel-use, modulation, coding and
validity assumptions are explicitly documented". No calibrated curves exist, so
this is the approximation, and the documentation is the point of the module.

**The assumptions, stated before the formula.**

1. *Channel*: complex AWGN at the realized instantaneous SINR. Fading is
   applied by the caller as a power gain before the packet is evaluated, so the
   BLER here is conditional on one fading realization and is *not* averaged
   over the fading distribution. Averaging is the caller's business because the
   deadline permits at most two attempts and their correlation matters -- see
   :mod:`hybrid_v2x_rl.channels.rf.fading`.
2. *Channel uses*: resource elements carrying data, from the configured grid
   after the declared overhead. The blocklength is a count of complex symbols,
   not of bits.
3. *Modulation and coding*: the approximation is information-theoretic and
   assumes an optimal code at the given rate. It therefore **understates** the
   error of a real QPSK rate-1/3 transport block, by an implementation margin
   that is not modelled. Any reliability claim built on it is optimistic and
   must say so.
4. *Validity*: the normal approximation is accurate for blocklengths of a few
   hundred channel uses upward. The configured grid gives roughly 4,800, which
   is comfortably inside. Below :data:`MIN_BLOCKLENGTH` the call is refused
   rather than extrapolated.

**The dispersion term's convention**, which section 7.3 singles out. In
Polyanskiy, Poor and Verdu's normal approximation the maximum number of
information bits at error ``eps`` is

    k = n C - sqrt(n V) Q^-1(eps) + (1/2) log2(n)

so the ``+ (1/2) log2 n`` is a bonus *added to the achievable payload*, in
bits, for the whole block -- not a per-channel-use rate term and not a penalty.
Inverting for ``eps`` puts it inside the numerator with a positive sign, which
is what :func:`block_error_probability` does. Getting its sign or its
normalization wrong shifts BLER by a factor that grows as the blocklength
shrinks, which is precisely why the work plan asks for the source and the
convention rather than the number.

**This file is radio only.** IM/DD optical detection is not a complex-AWGN
channel and a generic finite-blocklength expression does not apply to it;
section 7.3 forbids the transfer explicitly, and a guard test asserts that no
VLC module imports this one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.core.errors import HybridV2XError

#: Thermal noise density at 290 K, dBm/Hz: ``10 log10(k T) + 30``.
THERMAL_NOISE_DENSITY_DBM_PER_HZ = -173.975

#: Receiver noise figure, dB.  A vehicular UE figure, DECLARED not sourced.
DEFAULT_NOISE_FIGURE_DB = 9.0

#: Below this many channel uses the normal approximation is not trustworthy.
MIN_BLOCKLENGTH = 100

#: Numerical floor and ceiling, so a caller never sees exactly 0 or 1 and then
#: takes a logarithm of it.
_MIN_BLER = 1e-15
_MAX_BLER = 1.0 - 1e-15


class BLERError(HybridV2XError):
    """Block error was requested outside the approximation's validity."""


def gaussian_tail(x: float) -> float:
    """``Q(x)``, the upper tail of the standard normal.

    Via ``erfc`` from the standard library rather than SciPy, for the same
    reason :func:`hybrid_v2x_rl.channels.rf.fading.bessel_j0` is hand-rolled.
    """

    return 0.5 * math.erfc(x / math.sqrt(2.0))


def thermal_noise_dbm(bandwidth_hz: float, noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB) -> float:
    """Receiver noise floor over ``bandwidth_hz``."""

    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise BLERError("bandwidth must be finite and positive",
                        context={"bandwidth_hz": bandwidth_hz})
    return (
        THERMAL_NOISE_DENSITY_DBM_PER_HZ
        + 10.0 * math.log10(bandwidth_hz)
        + noise_figure_db
    )


@dataclass(frozen=True, slots=True)
class LinkBudget:
    """One link's realized budget, in decibels except the fading gain."""

    tx_power_dbm: float
    path_loss_db: float
    shadowing_db: float
    fading_power_gain: float
    noise_dbm: float

    @property
    def received_power_dbm(self) -> float:
        if self.fading_power_gain <= 0.0:
            return -math.inf
        return (
            self.tx_power_dbm
            - self.path_loss_db
            - self.shadowing_db
            + 10.0 * math.log10(self.fading_power_gain)
        )

    @property
    def snr_db(self) -> float:
        return self.received_power_dbm - self.noise_dbm

    @property
    def snr_linear(self) -> float:
        return 10.0 ** (self.snr_db / 10.0)


def shannon_capacity(snr_linear: float) -> float:
    """Capacity of the complex AWGN channel, bits per channel use."""

    if snr_linear < 0.0 or not math.isfinite(snr_linear):
        raise BLERError("SNR must be finite and non-negative",
                        context={"snr_linear": snr_linear})
    return math.log2(1.0 + snr_linear)


def channel_dispersion(snr_linear: float) -> float:
    """Dispersion ``V`` of the complex AWGN channel, bits^2 per channel use.

    ``V = (1 - 1/(1+snr)^2) (log2 e)^2``.  It vanishes at zero SNR, where the
    channel is useless and there is nothing to disperse, and saturates at
    ``(log2 e)^2`` when the SNR is large.
    """

    if snr_linear < 0.0 or not math.isfinite(snr_linear):
        raise BLERError("SNR must be finite and non-negative",
                        context={"snr_linear": snr_linear})
    return (1.0 - 1.0 / (1.0 + snr_linear) ** 2) * (math.log2(math.e) ** 2)


def block_error_probability(
    snr_linear: float, blocklength: int, information_bits: int
) -> float:
    """Normal-approximation BLER for ``information_bits`` over ``blocklength`` uses.

    Inverts ``k = nC - sqrt(nV) Q^-1(eps) + (1/2) log2 n`` for ``eps``; see the
    module docstring for the convention on the last term.
    """

    if blocklength < MIN_BLOCKLENGTH:
        raise BLERError(
            "blocklength below the normal approximation's validity range",
            context={"blocklength": blocklength, "minimum": MIN_BLOCKLENGTH},
        )
    if information_bits <= 0:
        raise BLERError("information bits must be positive",
                        context={"information_bits": information_bits})

    capacity = shannon_capacity(snr_linear)
    dispersion = channel_dispersion(snr_linear)
    if dispersion <= 0.0:
        # Zero SNR: nothing is carried, so every block is in error.
        return _MAX_BLER

    achievable = blocklength * capacity + 0.5 * math.log2(blocklength)
    numerator = achievable - information_bits
    denominator = math.sqrt(blocklength * dispersion)
    return min(_MAX_BLER, max(_MIN_BLER, gaussian_tail(numerator / denominator)))


def required_snr_db(
    target_bler: float, blocklength: int, information_bits: int,
    *, tolerance_db: float = 1e-6,
) -> float:
    """SNR at which :func:`block_error_probability` reaches ``target_bler``.

    Bisection rather than an inverse, because the forward direction is the one
    that is exact and the inverse is only ever needed for reporting -- "how much
    margin does this profile need" is a design question, not an inner-loop one.
    """

    if not 0.0 < target_bler < 1.0:
        raise BLERError("target BLER must lie strictly in (0, 1)",
                        context={"target_bler": target_bler})

    low, high = -40.0, 60.0
    while high - low > tolerance_db:
        middle = 0.5 * (low + high)
        value = block_error_probability(
            10.0 ** (middle / 10.0), blocklength, information_bits
        )
        if value > target_bler:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


__all__ = [
    "BLERError",
    "DEFAULT_NOISE_FIGURE_DB",
    "LinkBudget",
    "MIN_BLOCKLENGTH",
    "THERMAL_NOISE_DENSITY_DBM_PER_HZ",
    "block_error_probability",
    "channel_dispersion",
    "gaussian_tail",
    "required_snr_db",
    "shannon_capacity",
    "thermal_noise_dbm",
]
