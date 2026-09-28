"""Small-scale fading, correlated in **both** time and frequency.

This is the frequency-selective half of the split that
:mod:`hybrid_v2x_rl.channels.rf.pathloss_37885` opens. Everything there is flat
across the 10 MHz carrier and immune to hopping; everything here moves within a
packet's lifetime and *is* recoverable by hopping. The paper's equal-cost
DUP-versus-RF x2 ablation measures cross-medium diversity only if that boundary
is real, so both correlations are modelled explicitly rather than assumed.

**Which speed sets the Doppler.** For a tagged pair travelling the same way the
*relative* speed is near zero, and using it would freeze the fading and make
retransmission look useless. That would be wrong: the fading is driven by each
terminal moving through a scattering environment that is largely static --
facades, parked cars, street furniture -- so the composite Doppler spread scales
with the sum of the two ground speeds, not their difference. This is the single
most consequential modelling choice in the module, because it decides whether
time diversity inside the 3 ms deadline exists at all.

**And the answer points the wrong way.** Measured, the correlation between two
attempts 1 ms apart is:

===========  ==========  =========  =========  ===========
density      speed       f_d        T_c        R(1 ms)
===========  ==========  =========  =========  ===========
free flow    11.18 m/s   440 Hz     0.96 ms    -0.17
rho = 10      6.30 m/s   248 Hz     1.71 ms     0.48
rho = 20      4.08 m/s   161 Hz     2.63 ms     0.76
rho = 30      2.77 m/s   109 Hz     3.88 ms     0.89
===========  ==========  =========  =========  ===========

Slower traffic means a smaller Doppler spread, a longer coherence time, and
*less* time diversity. So the mechanism decays exactly as density rises --
which is where the reliability constraint binds hardest and where RF congestion
is simultaneously worst. Genuine decorrelation inside the deadline happens only
in free flow, which is the regime that already has link margin.

**Frequency diversity requires another full carrier.** The configured block
consumes all 24 RB of the 10 MHz carrier, so repeated attempts in the headline
pool share one carrier centre and do not receive an invented intra-carrier hop.
A 20 or 40 MHz system can supply two or four adjacent 10 MHz allocations;
their centres are at least 10 MHz apart, well beyond the approximately 1.4 MHz
coherence bandwidth. This module represents that optional inter-carrier
diversity while leaving the 10 MHz baseline frequency-correlated.

Sourcing status, as elsewhere in M3: the Rician K-factor and the RMS delay
spread are **not yet verified** against TR 37.885 and carry the same status as
the NLOSv blockage constants. The correlation *forms* are standard: Clarke's
isotropic-scattering autocorrelation ``J0(2 pi f_d tau)`` in time, and the
Fourier pair of an exponential power delay profile in frequency.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError

SPEED_OF_LIGHT_MPS = 299_792_458.0

#: RMS delay spread of the urban V2V channel, seconds.  UNVERIFIED.
URBAN_RMS_DELAY_SPREAD_S = 200e-9

#: Rician K-factor for a line-of-sight link, dB.  NLOSv and NLOS are treated as
#: Rayleigh (K = -inf), because a blocked direct path is what "no specular
#: component" means.  UNVERIFIED.
LOS_RICIAN_K_DB = 9.0

#: Clarke's autocorrelation first crosses zero at ``f_d tau = 0.3830``; the
#: conventional coherence time is the 0.5-correlation point, ``0.423 / f_d``.
COHERENCE_TIME_COEFFICIENT = 0.423

GeneratorFactory: TypeAlias = Callable[[str], np.random.Generator]


class FadingError(HybridV2XError):
    """Fading was evaluated with an invalid argument."""


def bessel_j0(x: float) -> float:
    """Bessel function of the first kind, order zero.

    Abramowitz & Stegun 9.4.1 and 9.4.3, absolute error below 1.6e-8 over the
    whole range. Implemented here rather than taken from SciPy because it is
    the only special function M3 needs and the project's runtime dependencies
    are deliberately four packages.
    """

    magnitude = abs(x)
    if magnitude < 3.0:
        t = (magnitude / 3.0) ** 2
        return (
            1.0
            + t * (-2.2499997
            + t * (1.2656208
            + t * (-0.3163866
            + t * (0.0444479
            + t * (-0.0039444
            + t * 0.0002100)))))
        )
    t = 3.0 / magnitude
    amplitude = (
        0.79788456
        + t * (-0.00000077
        + t * (-0.00552740
        + t * (-0.00009512
        + t * (0.00137237
        + t * (-0.00072805
        + t * 0.00014476)))))
    )
    phase = (
        magnitude
        - 0.78539816
        + t * (-0.04166397
        + t * (-0.00003954
        + t * (0.00262573
        + t * (-0.00054125
        + t * (-0.00029333
        + t * 0.00013558)))))
    )
    return amplitude * math.cos(phase) / math.sqrt(magnitude)


def wavelength_m(carrier_hz: float) -> float:
    if not math.isfinite(carrier_hz) or carrier_hz <= 0.0:
        raise FadingError("carrier must be finite and positive",
                          context={"carrier_hz": carrier_hz})
    return SPEED_OF_LIGHT_MPS / carrier_hz


def doppler_spread_hz(
    tx_speed_mps: float, rx_speed_mps: float, carrier_hz: float
) -> float:
    """Composite Doppler spread from two terminals moving through static scatter.

    The **sum** of ground speeds, not the difference; see the module docstring.
    """

    for speed in (tx_speed_mps, rx_speed_mps):
        if not math.isfinite(speed) or speed < 0.0:
            raise FadingError("speeds must be finite and non-negative",
                              context={"tx": tx_speed_mps, "rx": rx_speed_mps})
    return (tx_speed_mps + rx_speed_mps) / wavelength_m(carrier_hz)


def coherence_time_s(doppler_hz: float) -> float:
    """Time over which the fading stays correlated at 0.5.

    Infinite when nothing moves, which is the honest answer: a stationary pair
    in a static environment has no small-scale time variation at all.
    """

    if doppler_hz < 0.0 or not math.isfinite(doppler_hz):
        raise FadingError("doppler must be finite and non-negative",
                          context={"doppler_hz": doppler_hz})
    if doppler_hz == 0.0:
        return math.inf
    return COHERENCE_TIME_COEFFICIENT / doppler_hz


def temporal_correlation(elapsed_s: float, doppler_hz: float) -> float:
    """Clarke's isotropic-scattering autocorrelation, ``J0(2 pi f_d tau)``."""

    if elapsed_s < 0.0 or not math.isfinite(elapsed_s):
        raise FadingError("elapsed time must be finite and non-negative",
                          context={"elapsed_s": elapsed_s})
    return bessel_j0(2.0 * math.pi * doppler_hz * elapsed_s)


def frequency_correlation(separation_hz: float, rms_delay_spread_s: float) -> float:
    """Correlation between two subchannels ``separation_hz`` apart.

    The magnitude of the Fourier transform of an exponential power delay
    profile: ``1 / sqrt(1 + (2 pi df tau_rms)^2)``.
    """

    if separation_hz < 0.0 or not math.isfinite(separation_hz):
        raise FadingError("subchannel separation must be finite and non-negative",
                          context={"separation_hz": separation_hz})
    if rms_delay_spread_s <= 0.0 or not math.isfinite(rms_delay_spread_s):
        raise FadingError("delay spread must be finite and positive",
                          context={"rms_delay_spread_s": rms_delay_spread_s})
    return 1.0 / math.sqrt(1.0 + (2.0 * math.pi * separation_hz * rms_delay_spread_s) ** 2)


def coherence_bandwidth_hz(rms_delay_spread_s: float, correlation_level: float = 0.5) -> float:
    """Separation at which :func:`frequency_correlation` falls to ``correlation_level``."""

    if not 0.0 < correlation_level < 1.0:
        raise FadingError("correlation level must lie strictly in (0, 1)",
                          context={"correlation_level": correlation_level})
    inverse = math.sqrt(1.0 / correlation_level**2 - 1.0)
    return inverse / (2.0 * math.pi * rms_delay_spread_s)


def rician_k_linear(state: RFPropagationState) -> float:
    """Ratio of specular to diffuse power for ``state``.

    Zero for NLOSv and NLOS: an obstructed direct path has no specular
    component, which makes the envelope Rayleigh rather than Rician. That is
    the same event that severs the optical link, so the two media's fading
    statistics change together at exactly the moment section 8.3 cares about.
    """

    if state is RFPropagationState.LOS:
        return float(10.0 ** (LOS_RICIAN_K_DB / 10.0))
    return 0.0


@dataclass(slots=True)
class FadingProcess:
    """Per-link, per-subchannel complex gains with separable correlation.

    The realized correlation is ``R(dt, df) = R_t(dt) * R_f(df)``. Separability
    is an approximation and is stated as one: a jointly correlated scattering
    model would not factor exactly, but the factored form reproduces both
    marginals, which is what the retransmission model reads.
    """

    rng: np.random.Generator | None
    carrier_hz: float
    subchannel_separations_hz: tuple[float, ...]
    rms_delay_spread_s: float = URBAN_RMS_DELAY_SPREAD_S
    generator_factory: GeneratorFactory | None = None
    _gains: dict[str, np.ndarray] = field(default_factory=dict)
    _generators: dict[str, np.random.Generator] = field(
        default_factory=dict,
        repr=False,
    )
    _cholesky: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (self.rng is None) == (self.generator_factory is None):
            raise FadingError(
                "fading requires exactly one shared RNG or keyed generator factory"
            )
        if self.rng is not None and not isinstance(self.rng, np.random.Generator):
            raise FadingError("fading rng must be a NumPy Generator")
        if not self.subchannel_separations_hz:
            raise FadingError("at least one subchannel is required")
        offsets = np.asarray(self.subchannel_separations_hz, dtype=float)
        gaps = np.abs(offsets[:, None] - offsets[None, :])
        matrix = np.array(
            [[frequency_correlation(float(g), self.rms_delay_spread_s) for g in row]
             for row in gaps]
        )
        # A correlation matrix built pointwise is not guaranteed positive
        # definite, so nudge the diagonal rather than let Cholesky fail on a
        # configuration a user is entitled to choose.
        self._cholesky = np.linalg.cholesky(
            matrix + 1e-12 * np.eye(len(offsets))
        )

    @property
    def subchannel_count(self) -> int:
        return len(self.subchannel_separations_hz)

    def _generator(self, key: str) -> np.random.Generator:
        """Return one persistent stream per link when a factory is configured."""

        if self.rng is not None:
            return self.rng
        generator = self._generators.get(key)
        if generator is None:
            factory = self.generator_factory
            if factory is None:  # pragma: no cover - rejected in __post_init__.
                raise FadingError("fading generator factory is unavailable")
            generator = factory(key)
            if not isinstance(generator, np.random.Generator):
                raise FadingError(
                    "fading generator factory must return a NumPy Generator"
                )
            self._generators[key] = generator
        return generator

    def _draw_innovation(self, generator: np.random.Generator) -> np.ndarray:
        """Unit-power complex Gaussian, correlated across subchannels."""

        white = (
            generator.standard_normal(self.subchannel_count)
            + 1j * generator.standard_normal(self.subchannel_count)
        ) / math.sqrt(2.0)
        cholesky = self._cholesky
        if cholesky is None:  # pragma: no cover - initialized in __post_init__
            raise FadingError("fading correlation matrix is not initialized")
        return cholesky @ white

    def advance(
        self,
        key: str,
        *,
        elapsed_s: float,
        tx_speed_mps: float,
        rx_speed_mps: float,
        state: RFPropagationState,
    ) -> np.ndarray:
        """Advance ``key`` and return per-subchannel **power** gains.

        Gains are normalized to unit mean power, so they multiply a budget
        rather than displace it. The Rician specular component is added at read
        time for the same reason shadowing scales at read time: an obstruction
        appearing should remove the direct path without discontinuously
        resetting the diffuse field the link is sitting in.
        """

        doppler = doppler_spread_hz(tx_speed_mps, rx_speed_mps, self.carrier_hz)
        generator = self._generator(key)
        previous = self._gains.get(key)
        if previous is None:
            diffuse = self._draw_innovation(generator)
        else:
            rho = temporal_correlation(elapsed_s, doppler)
            # J0 is oscillatory and goes negative; an AR(1) needs |rho| <= 1 and
            # a non-negative innovation variance, which holds for any J0 value.
            diffuse = rho * previous + math.sqrt(
                max(0.0, 1.0 - rho * rho)
            ) * self._draw_innovation(generator)
        self._gains[key] = diffuse

        k = rician_k_linear(state)
        specular = math.sqrt(k / (k + 1.0))
        scatter = math.sqrt(1.0 / (k + 1.0))
        envelope = specular + scatter * diffuse
        return np.abs(envelope) ** 2

    def forget(self, key: str) -> None:
        self._gains.pop(key, None)
        self._generators.pop(key, None)

    def live_links(self) -> int:
        return len(self._gains)


__all__ = [
    "COHERENCE_TIME_COEFFICIENT",
    "FadingError",
    "FadingProcess",
    "GeneratorFactory",
    "LOS_RICIAN_K_DB",
    "SPEED_OF_LIGHT_MPS",
    "URBAN_RMS_DELAY_SPREAD_S",
    "bessel_j0",
    "coherence_bandwidth_hz",
    "coherence_time_s",
    "doppler_spread_hz",
    "frequency_correlation",
    "rician_k_linear",
    "temporal_correlation",
    "wavelength_m",
]
