"""Analytical resource-collision model with declared sensitivity bands.

Work plan section 7.2 is unusually prescriptive here, and for a reason. The
original neighbour-count-to-collision mapping was rejected as arbitrary; the
5G-LENA Mode-2 calibration that would have replaced it is deferred under v2's
own fallback clause. What remains is required to be a **clearly scoped
analytical model with sensitivity bands**, with Mode-2 realism claims weakened
*from the start*.

So: **this is not NR Mode 2.** It is a birthday-collision model over a resource
pool, with sensing represented as a scalar and its uncertainty represented as a
band rather than a point. The paper must say "analytical collision model with
sensitivity bands" and never "NR Mode 2". Nothing in this module licenses the
stronger claim, and :data:`MODEL_NAME` exists so reports quote it rather than
paraphrase.

**Why this mechanism matters to the contribution.** It is the one RF failure
mode genuinely *decoupled* from optical blockage. NLOSv is the same geometric
event that severs the optical path, so diversity arguments built on it are
circular. Collision is driven by a headcount, and the measured campaign shows
the two moving in opposite directions across the training band: neighbours
within 200 m rise 44 -> 100 -> 159 while V-VLC unavailability falls
9.59% -> 9.17% -> 7.26%. At rho = 10 the optical link is the weaker one; at
rho = 30 the radio is. That opposition is what RQ4 needs, and it lives here.

**Where coupling does re-enter, stated rather than hidden.** The *sensed
fraction* is geometric: a neighbour whose reservation cannot be decoded is one
that contends blind, and whether it can be decoded depends on RF visibility.
The caller supplies that fraction, so the coupling is visible at the call site
instead of buried in a constant. Section 8.3 requires the joint failure
probability to be logged rather than assumed independent; a module that hid
this would make that measurement meaningless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum, unique

from hybrid_v2x_rl.core.errors import HybridV2XError

#: Quote this, not a paraphrase, wherever the model is described.
MODEL_NAME = "analytical collision model with sensitivity bands"


class CollisionError(HybridV2XError):
    """The collision model was evaluated outside its declared scope."""


@unique
class SensitivityBand(StrEnum):
    """Which end of the declared uncertainty to evaluate.

    The bands are *declared uncertainty*, not measurements. Reporting only the
    nominal would assert a precision the deferred calibration has not earned,
    so every headline number that depends on collision must be reported across
    all three.
    """

    OPTIMISTIC = "optimistic"
    NOMINAL = "nominal"
    PESSIMISTIC = "pessimistic"


@dataclass(frozen=True, slots=True)
class CollisionParameters:
    """Scope of the analytical model, and the knob its band moves.

    ``sensing_reliability`` is the fraction of contending neighbours whose
    reservation is successfully decoded and therefore excluded from the
    candidate pool. It is the single quantity the sensitivity band moves,
    because it is the single quantity the deferred Mode-2 calibration would
    have pinned down.
    """

    subchannels: int
    selection_window_slots: int
    sensing_reliability: float
    #: Airtime one transmission occupies, and how often it repeats.  Together
    #: these set channel occupancy per vehicle.
    airtime_s: float
    generation_period_s: float
    #: Share of generated packets that actually put a copy on the radio.
    #:
    #: One means every vehicle transmits every packet, which is what a
    #: single-medium profile does and why it is the default. Below one the
    #: radio is carrying only part of the offered traffic because the rest
    #: went by light, and **three separate quantities have to fall together**:
    #: how many neighbours contend, how often the intended receiver is busy
    #: transmitting, and how much of the pool is claimed. They are one
    #: behaviour seen three ways, so they scale from one field rather than
    #: from three call sites that could disagree.
    rf_usage_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.subchannels < 1:
            raise CollisionError("at least one subchannel is required",
                                 context={"subchannels": self.subchannels})
        if self.selection_window_slots < 1:
            raise CollisionError("selection window must span at least one slot",
                                 context={"slots": self.selection_window_slots})
        if not 0.0 <= self.sensing_reliability <= 1.0:
            raise CollisionError("sensing reliability must lie in [0, 1]",
                                 context={"value": self.sensing_reliability})
        if not 0.0 <= self.rf_usage_fraction <= 1.0:
            raise CollisionError("rf usage fraction must lie in [0, 1]",
                                 context={"value": self.rf_usage_fraction})
        for name in ("airtime_s", "generation_period_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise CollisionError(f"{name} must be finite and positive",
                                     context={name: value})
        if self.airtime_s > self.generation_period_s:
            raise CollisionError(
                "a transmission cannot outlast its own generation period",
                context={"airtime_s": self.airtime_s,
                         "generation_period_s": self.generation_period_s},
            )

    @property
    def candidate_resources(self) -> int:
        """Distinct resources a contender may select from."""

        return self.subchannels * self.selection_window_slots


#: Airtime one vehicle commits per generation period under the headline
#: profile: three pre-reserved 0.5 ms attempts. It is a **default, not a
#: constant** -- the profile owns it, and :func:`headline_parameters` takes it
#: as an argument so a changed service profile cannot leave this behind. An
#: earlier version hardcoded 1.0 ms here while the profile committed 1.5 ms,
#: which fed a channel-busy-ratio a third too low into the observation vector.
DEFAULT_COMMITTED_AIRTIME_S = 1.5e-3


#: The frozen profile: two 12-RB subchannels on 24 RB, a 100 ms selection
#: window at 0.5 ms slots.
def headline_parameters(
    band: SensitivityBand = SensitivityBand.NOMINAL,
    *,
    committed_airtime_s: float = DEFAULT_COMMITTED_AIRTIME_S,
    rf_usage_fraction: float = 1.0,
) -> CollisionParameters:
    """Parameters for the headline profile at one end of the declared band.

    The reliability values are a declared range, not a measurement. They are
    wide on purpose: a deferred calibration is uncertainty, and narrowing the
    band without doing the calibration would be asserting the result of work
    that has not been done.
    """

    reliability = {
        SensitivityBand.OPTIMISTIC: 0.95,
        SensitivityBand.NOMINAL: 0.85,
        SensitivityBand.PESSIMISTIC: 0.70,
    }[band]
    return CollisionParameters(
        subchannels=2,
        selection_window_slots=200,
        sensing_reliability=reliability,
        airtime_s=committed_airtime_s,
        generation_period_s=0.1,
        rf_usage_fraction=rf_usage_fraction,
    )


def resource_demand(neighbour_count: int, parameters: CollisionParameters) -> float:
    """Airtime the contending population claims, as a multiple of what exists.

    The pool supplies ``subchannels * generation_period_s`` of airtime per
    period. Each of ``neighbour_count`` vehicles transmits once per period and
    commits ``airtime_s``, so the demand is their ratio. Where the selection
    window spans the generation period this is exactly
    ``neighbour_count * attempts / candidate_resources`` -- the counting form --
    but it is written as airtime because that needs no slot duration and so
    cannot disagree with :attr:`CollisionParameters.airtime_s`.

    **Above one the pool is oversubscribed and the profile is not deliverable
    at any collision probability**, because no assignment of contenders to
    resources gives every vehicle what it asked for.
    :func:`collision_probability` cannot see this: the birthday model asks where
    *one* selection lands, not whether every selection can be honoured, so an
    oversubscribed profile still reports a modest per-attempt collision. That is
    how the headline profile came to sit at 1.19x demand at rho = 30 -- three
    attempts against a pool that supplies 2.5 -- with nothing in the model
    objecting.

    Deliberately **not** clipped. :func:`channel_busy_ratio` clips because it
    feeds an observation vector, where a fraction above one is meaningless; this
    is the physical quantity, and the overflow is the entire signal.
    """

    if neighbour_count < 0:
        raise CollisionError("neighbour count cannot be negative",
                             context={"neighbour_count": neighbour_count})
    supply = parameters.generation_period_s * parameters.subchannels
    offered = neighbour_count * parameters.rf_usage_fraction
    return offered * parameters.airtime_s / supply


def channel_busy_ratio(neighbour_count: int, parameters: CollisionParameters) -> float:
    """Fraction of resources occupied by neighbours, clipped to one.

    Each neighbour occupies ``airtime / period`` of one subchannel, so the pool
    it consumes is that divided by the subchannel count. This is the quantity
    the observation vector carries as ``rf_channel_busy_ratio``, and deriving
    it here rather than measuring it off a trace is what keeps M3 independent
    of the campaign that will be regenerated.

    The clip is what separates this from :func:`resource_demand`, which is the
    same arithmetic left unbounded. Delegating rather than repeating the
    division keeps a saturated observation and an oversubscribed pool as one
    fact reported two ways, instead of two numbers free to drift apart.
    """

    return min(1.0, resource_demand(neighbour_count, parameters))


def hidden_contenders(neighbour_count: int, sensed_fraction: float,
                      parameters: CollisionParameters) -> float:
    """Expected contenders that select without seeing this link's reservation.

    ``sensed_fraction`` is geometric -- the share of neighbours whose
    reservations are decodable, which the caller derives from RF visibility --
    and it multiplies the model's own ``sensing_reliability``. Two separate
    things, kept separate: one is where the neighbours are, the other is how
    well sensing works when they can be heard.
    """

    if neighbour_count < 0:
        raise CollisionError("neighbour count cannot be negative",
                             context={"neighbour_count": neighbour_count})
    if not 0.0 <= sensed_fraction <= 1.0:
        raise CollisionError("sensed fraction must lie in [0, 1]",
                             context={"sensed_fraction": sensed_fraction})
    effective = parameters.sensing_reliability * sensed_fraction
    offered = neighbour_count * parameters.rf_usage_fraction
    return offered * (1.0 - effective)


def collision_probability(
    neighbour_count: int,
    parameters: CollisionParameters,
    *,
    sensed_fraction: float = 1.0,
) -> float:
    """Probability that at least one hidden contender picks the same resource.

    The birthday form: with ``M`` candidate resources and ``n`` contenders
    selecting uniformly and independently, a given selection survives with
    ``(1 - 1/M)^n``.

    Independence is the model's main simplification and is stated as such:
    real Mode-2 selections are correlated through shared sensing history, which
    would make collisions burstier than this predicts. That is one of the
    things the deferred calibration would have quantified, and it is why the
    band exists.
    """

    hidden = hidden_contenders(neighbour_count, sensed_fraction, parameters)
    survival = (1.0 - 1.0 / parameters.candidate_resources) ** hidden
    return 1.0 - survival


def half_duplex_probability(parameters: CollisionParameters) -> float:
    """Probability the intended receiver is transmitting when this arrives.

    A sidelink terminal cannot receive while it transmits. With independent
    phases this is just the receiver's own duty cycle, and it is a floor no
    amount of resource selection removes -- which makes it a genuine, if small,
    argument for a second medium rather than a second attempt.

    The duty cycle is the receiver's *radio* duty cycle, so it carries
    ``rf_usage_fraction``: a receiver that sent this packet's period by light
    was not transmitting and could hear. Charging the full cycle regardless
    would overstate the term by ``1 / rf_usage_fraction``, and at the attempt
    counts this profile grants the term is the same order as collision -- so it
    is not a rounding error, it decides the answer.
    """

    return (
        parameters.rf_usage_fraction
        * parameters.airtime_s
        / parameters.generation_period_s
    )


def failure_probability(
    neighbour_count: int,
    parameters: CollisionParameters,
    *,
    sensed_fraction: float = 1.0,
) -> float:
    """Probability the access layer loses the packet, from either mechanism.

    Collision and half-duplex are independent events here: whether the receiver
    happens to be talking has nothing to do with which resource a third party
    selected.
    """

    collision = collision_probability(
        neighbour_count, parameters, sensed_fraction=sensed_fraction
    )
    half_duplex = half_duplex_probability(parameters)
    return 1.0 - (1.0 - collision) * (1.0 - half_duplex)


__all__ = [
    "DEFAULT_COMMITTED_AIRTIME_S",
    "MODEL_NAME",
    "CollisionError",
    "CollisionParameters",
    "SensitivityBand",
    "channel_busy_ratio",
    "collision_probability",
    "failure_probability",
    "half_duplex_probability",
    "headline_parameters",
    "hidden_contenders",
    "resource_demand",
]
