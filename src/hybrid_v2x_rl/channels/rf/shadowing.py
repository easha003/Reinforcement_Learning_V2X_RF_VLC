"""Spatially correlated log-normal shadowing.

Work plan section 7.1: "Shadowing must be spatially correlated." That word is
doing real work. Shadowing redrawn independently per packet is a fast fade
under another name -- it would average out over the 3 ms deadline, make a
retransmission look far more useful than it is, and hand the policy a quantity
with no persistence to learn. Correlated shadowing persists over metres of
travel, which is tens of packets, so a shadowed link *stays* shadowed and the
policy's choice to switch medium has a consequence that lasts.

The model is Gudmundson's: an exponential autocorrelation in travelled
distance, realized as a first-order autoregression along the trajectory.

    rho(dd) = exp(-dd / d_corr)
    s <- rho * s + sqrt(1 - rho^2) * N(0, 1)

The state is kept at **unit variance** and scaled by the class-dependent sigma
only when read. That is deliberate: a link crossing from LOS into NLOSv has not
teleported into a different neighbourhood, it has had a van pull in front of
it. The obstruction changes how much the surroundings matter, not which
surroundings they are. Storing sigma-scaled state instead would make every
class transition a discontinuity in the shadowing itself, which is the wrong
physics and would also inject an artificial correlation between the class
sequence and the shadowing sequence -- precisely the correlation section 8.3
exists to measure rather than manufacture.

Sourcing status: the decorrelation distances below have the same status as the
NLOSv blockage constants in :mod:`hybrid_v2x_rl.channels.rf.pathloss_37885` -- the
structure is standard, the values are **not yet verified** against TR 37.885
and must be checked before Gate 2.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError

#: Distance over which shadowing decorrelates to 1/e, in metres.  UNVERIFIED.
LOS_DECORRELATION_M = 10.0
NLOSV_DECORRELATION_M = 10.0
NLOS_DECORRELATION_M = 13.0

#: Beyond this many decorrelation lengths the AR coefficient underflows to
#: zero in double precision anyway, so the sample is drawn fresh.
_INDEPENDENCE_THRESHOLD = 40.0

GeneratorFactory: TypeAlias = Callable[[str], np.random.Generator]


class ShadowingError(HybridV2XError):
    """Shadowing was advanced with an invalid displacement."""


def decorrelation_distance_m(state: RFPropagationState) -> float:
    """Gudmundson decorrelation length for ``state``."""

    if state is RFPropagationState.NLOS:
        return NLOS_DECORRELATION_M
    if state is RFPropagationState.NLOSV:
        return NLOSV_DECORRELATION_M
    return LOS_DECORRELATION_M


def correlation(displacement_m: float, decorrelation_m: float) -> float:
    """Autocorrelation of shadowing over ``displacement_m`` of travel."""

    if not math.isfinite(displacement_m) or displacement_m < 0.0:
        raise ShadowingError(
            "shadowing displacement must be finite and non-negative",
            context={"displacement_m": displacement_m},
        )
    if not math.isfinite(decorrelation_m) or decorrelation_m <= 0.0:
        raise ShadowingError(
            "decorrelation distance must be finite and positive",
            context={"decorrelation_m": decorrelation_m},
        )
    ratio = displacement_m / decorrelation_m
    if ratio > _INDEPENDENCE_THRESHOLD:
        return 0.0
    return math.exp(-ratio)


@dataclass(slots=True)
class ShadowingProcess:
    """Per-link shadowing state, advanced along each link's trajectory.

    One instance holds every live link's normalized state. Links are keyed by
    an opaque string so the caller decides identity -- a tagged pair keys on
    the ordered vehicle pair, a neighbour link on whatever the congestion model
    needs -- and this module never has to know about vehicles.
    """

    rng: np.random.Generator | None
    generator_factory: GeneratorFactory | None = None
    _normalized: dict[str, float] = field(default_factory=dict)
    _generators: dict[str, np.random.Generator] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if (self.rng is None) == (self.generator_factory is None):
            raise ShadowingError(
                "shadowing requires exactly one shared RNG or keyed generator factory"
            )
        if self.rng is not None and not isinstance(self.rng, np.random.Generator):
            raise ShadowingError("shadowing rng must be a NumPy Generator")

    def _generator(self, key: str) -> np.random.Generator:
        """Return one persistent stream per link when a factory is configured."""

        if self.rng is not None:
            return self.rng
        generator = self._generators.get(key)
        if generator is None:
            factory = self.generator_factory
            if factory is None:  # pragma: no cover - rejected in __post_init__.
                raise ShadowingError("shadowing generator factory is unavailable")
            generator = factory(key)
            if not isinstance(generator, np.random.Generator):
                raise ShadowingError(
                    "shadowing generator factory must return a NumPy Generator"
                )
            self._generators[key] = generator
        return generator

    def advance(self, key: str, displacement_m: float, state: RFPropagationState) -> float:
        """Move ``key``'s shadowing forward and return its unit-variance value.

        A link seen for the first time is initialized from the stationary
        distribution rather than at zero, so it does not start unshadowed and
        converge -- an artefact that would make every newly formed tagged pair
        briefly and wrongly optimistic.
        """

        generator = self._generator(key)
        previous = self._normalized.get(key)
        if previous is None:
            value = float(generator.standard_normal())
        else:
            rho = correlation(displacement_m, decorrelation_distance_m(state))
            value = rho * previous + math.sqrt(1.0 - rho * rho) * float(
                generator.standard_normal()
            )
        self._normalized[key] = value
        return value

    def forget(self, key: str) -> None:
        """Drop a link's state when its episode ends.

        Without this the dictionary grows for the life of a run, and a pair
        that re-forms would inherit shadowing from an unrelated earlier
        encounter at a different place on the map.
        """

        self._normalized.pop(key, None)
        self._generators.pop(key, None)

    def live_links(self) -> int:
        return len(self._normalized)


def shadowing_db(normalized: float, state: RFPropagationState, sigma_db: float) -> float:
    """Scale a unit-variance state to decibels for ``state``.

    ``sigma_db`` comes from :func:`hybrid_v2x_rl.channels.rf.pathloss_37885.shadowing_sigma_db`
    and is passed in rather than imported so the two modules stay independent:
    path loss owns the spread per class, this module owns the correlation.
    """

    if not math.isfinite(sigma_db) or sigma_db < 0.0:
        raise ShadowingError(
            "shadowing sigma must be finite and non-negative",
            context={"sigma_db": sigma_db, "state": str(state)},
        )
    return normalized * sigma_db


__all__ = [
    "LOS_DECORRELATION_M",
    "NLOSV_DECORRELATION_M",
    "NLOS_DECORRELATION_M",
    "ShadowingError",
    "ShadowingProcess",
    "GeneratorFactory",
    "correlation",
    "decorrelation_distance_m",
    "shadowing_db",
]
