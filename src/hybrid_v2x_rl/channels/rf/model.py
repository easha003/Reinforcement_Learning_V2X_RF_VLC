"""The NR sidelink channel, assembled into one outcome per packet.

Implementation spec section 10.7. Everything above this module answers one
question each -- which propagation class, how much median loss, how much
shadowing, how much fading, how likely a collision, how likely a decoding
failure -- and this is where they compose into a delivered-or-not.

**The composition order is the physics and is not interchangeable.** Class
comes from geometry alone; median loss and shadowing spread follow from the
class; fading statistics follow from the class as well, because an obstructed
direct path has no specular component; the budget then yields a decoding
failure probability; and collision is evaluated independently of all of it,
because which resource a third party selected has nothing to do with how far
away this receiver is. Reordering any of that would either make the class
stochastic -- which strips the observation forecast of anything to predict --
or would couple collision to geometry, which is the coupling section 8.3 exists
to measure rather than manufacture.

**Two failure mechanisms, kept separable to the outcome.** The result reports
``collision_probability`` and ``decoding_failure_probability`` separately, and
records which one actually fired. Section 7.2's mechanism is decoupled from
optical blockage; section 7.1's is the same geometric event that severs the
optical path. A single aggregate probability would let a diversity claim rest
on whichever one the reader assumed, and measurement so far says the two differ
by two to four orders of magnitude -- so the aggregate would be collision under
another name while looking like a channel result.

**Randomness is supplied, never drawn here.** The caller passes the uniform
draws for this packet, so a trace regenerates bit-exactly and so a counterfactual
-- what the *other* action would have produced on the same packet -- is
evaluated against the same tape. Matched tapes across actions are what make the
oracle's advantage a measurement rather than a sampling artefact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.channels.rf.bler import (
    LinkBudget,
    block_error_probability,
    thermal_noise_dbm,
)
from hybrid_v2x_rl.channels.rf.collision import (
    CollisionParameters,
    collision_probability,
    half_duplex_probability,
)
from hybrid_v2x_rl.channels.rf.pathloss_37885 import large_scale_loss
from hybrid_v2x_rl.channels.rf.shadowing import shadowing_db
from hybrid_v2x_rl.core.enums import FailureCause, RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError


class RFChannelError(HybridV2XError):
    """The RF channel was invoked with an inconsistent request."""


@dataclass(frozen=True, slots=True)
class RFPacketRandomness:
    """The uniform draws one packet consumes, supplied by the caller.

    Three independent draws rather than one, so that adding a mechanism later
    does not shift the others' realizations and silently invalidate every
    published trace.
    """

    collision_draw: float
    decoding_draw: float
    half_duplex_draw: float

    def __post_init__(self) -> None:
        for name in ("collision_draw", "decoding_draw", "half_duplex_draw"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise RFChannelError(f"{name} must be a uniform draw in [0, 1]",
                                     context={name: value})


@dataclass(frozen=True, slots=True)
class RFPropagationRequest:
    """Policy-independent physical state for one RF transmission attempt.

    ``shadowing_normalized`` and ``fading_power_gain`` come from the correlated
    processes rather than being drawn here, because their whole value is that
    they carry state between packets.
    """

    distance_m: float
    propagation_state: RFPropagationState
    blockage_db: float
    shadowing_normalized: float
    fading_power_gain: float


@dataclass(frozen=True, slots=True)
class RFChannelRequest:
    """Legacy propagation, contention, and randomness inputs for one attempt."""

    distance_m: float
    propagation_state: RFPropagationState
    blockage_db: float
    shadowing_normalized: float
    fading_power_gain: float
    neighbour_count: int
    sensed_fraction: float
    randomness: RFPacketRandomness

    @property
    def propagation(self) -> RFPropagationRequest:
        """Return the action-independent subset consumed by the physical link."""

        return RFPropagationRequest(
            distance_m=self.distance_m,
            propagation_state=self.propagation_state,
            blockage_db=self.blockage_db,
            shadowing_normalized=self.shadowing_normalized,
            fading_power_gain=self.fading_power_gain,
        )


@dataclass(frozen=True, slots=True)
class RFPropagationResult:
    """Deterministic link-budget result with no action or contention inputs."""

    propagation_state: RFPropagationState
    pathloss_db: float
    shadowing_db: float
    fading_gain_linear: float
    sinr_db: float
    decoding_failure_probability: float


@dataclass(frozen=True, slots=True)
class RFChannelResult:
    """Hidden truth and outcome for one attempt.

    Everything here is oracle-side. The observation layer reconstructs a noisy,
    aged view of a subset of it, and the leakage guards keep the two apart.
    """

    propagation_state: RFPropagationState
    pathloss_db: float
    shadowing_db: float
    fading_gain_linear: float
    sinr_db: float
    collision_probability: float
    decoding_failure_probability: float
    total_failure_probability: float
    success: bool
    failure_cause: FailureCause

    @property
    def collision_dominates(self) -> bool:
        """Whether access contention outweighs the link budget for this attempt.

        Reported per attempt rather than assumed, because the measured answer
        is "almost always" and an assumption that flipped silently would move
        the diversity argument from non-circular to circular.
        """

        return self.collision_probability > self.decoding_failure_probability


@dataclass(frozen=True, slots=True)
class NRV2XChannel:
    """Composes the M3 modules into a per-packet outcome."""

    carrier_hz: float
    bandwidth_hz: float
    tx_power_dbm: float
    blocklength: int
    information_bits: int
    collision: CollisionParameters
    noise_figure_db: float | None = None

    @property
    def noise_dbm(self) -> float:
        if self.noise_figure_db is None:
            return thermal_noise_dbm(self.bandwidth_hz)
        return thermal_noise_dbm(self.bandwidth_hz, self.noise_figure_db)

    def evaluate_propagation(
        self,
        request: RFPropagationRequest,
    ) -> RFPropagationResult:
        """Evaluate only policy-independent propagation and decoding state."""

        if not isinstance(request, RFPropagationRequest):
            raise RFChannelError(
                "propagation evaluation requires an RFPropagationRequest"
            )
        loss = large_scale_loss(
            distance_m=request.distance_m,
            carrier_hz=self.carrier_hz,
            state=request.propagation_state,
            blockage_db=request.blockage_db,
        )
        shadow_db = shadowing_db(
            request.shadowing_normalized,
            request.propagation_state,
            loss.shadowing_sigma_db,
        )
        budget = LinkBudget(
            tx_power_dbm=self.tx_power_dbm,
            path_loss_db=loss.total_db,
            shadowing_db=shadow_db,
            fading_power_gain=request.fading_power_gain,
            noise_dbm=self.noise_dbm,
        )
        decoding = block_error_probability(
            budget.snr_linear, self.blocklength, self.information_bits
        )
        return RFPropagationResult(
            propagation_state=request.propagation_state,
            pathloss_db=loss.total_db,
            shadowing_db=shadow_db,
            fading_gain_linear=request.fading_power_gain,
            sinr_db=budget.snr_db,
            decoding_failure_probability=decoding,
        )

    def evaluate(self, request: RFChannelRequest) -> RFChannelResult:
        """Deliver or lose one packet, and say which mechanism decided it."""

        propagation = self.evaluate_propagation(request.propagation)
        decoding = propagation.decoding_failure_probability
        contention = collision_probability(
            request.neighbour_count,
            self.collision,
            sensed_fraction=request.sensed_fraction,
        )
        half_duplex = half_duplex_probability(self.collision)

        # Resolved in the order the mechanisms occur, so the recorded cause is
        # the one that actually stopped the packet rather than the first one
        # tested.  A receiver that is transmitting never hears the collision.
        draws = request.randomness
        if draws.half_duplex_draw < half_duplex:
            cause, success = FailureCause.RF_COLLISION, False
        elif draws.collision_draw < contention:
            cause, success = FailureCause.RF_COLLISION, False
        elif draws.decoding_draw < decoding:
            cause, success = FailureCause.RF_CHANNEL, False
        else:
            cause, success = FailureCause.NONE, True

        access = 1.0 - (1.0 - contention) * (1.0 - half_duplex)
        total = 1.0 - (1.0 - access) * (1.0 - decoding)
        return RFChannelResult(
            propagation_state=propagation.propagation_state,
            pathloss_db=propagation.pathloss_db,
            shadowing_db=propagation.shadowing_db,
            fading_gain_linear=propagation.fading_gain_linear,
            sinr_db=propagation.sinr_db,
            collision_probability=access,
            decoding_failure_probability=propagation.decoding_failure_probability,
            total_failure_probability=total,
            success=success,
            failure_cause=cause,
        )

    def failure_probability(self, request: RFChannelRequest) -> float:
        """Total failure probability without consuming the draws.

        The oracle needs the probability of every action on the same packet,
        and evaluating the unused ones through :meth:`evaluate` would burn tape
        and shift the used one's realization.
        """

        return self.evaluate(request).total_failure_probability


def marginal_and_joint(
    rf_failure: float, vlc_failure: float, joint_failure: float
) -> tuple[float, float, float, float]:
    """Return ``(p_rf, p_vlc, p_joint, dependence_ratio)`` for section 8.3.

    The ratio ``p_joint / (p_rf p_vlc)`` is one under independence, above one
    when the media fail together, and below one when they complement. Logging
    it per scenario is what stops the diversity result silently assuming
    independent failures -- an assumption this model has concrete reason to
    doubt, since NLOSv is the same geometric event that severs the optical
    path.
    """

    for name, value in (("rf", rf_failure), ("vlc", vlc_failure), ("joint", joint_failure)):
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise RFChannelError(f"{name} failure probability must lie in [0, 1]",
                                 context={name: value})
    product = rf_failure * vlc_failure
    ratio = math.inf if product == 0.0 else joint_failure / product
    return rf_failure, vlc_failure, joint_failure, ratio


__all__ = [
    "NRV2XChannel",
    "RFChannelError",
    "RFChannelRequest",
    "RFChannelResult",
    "RFPacketRandomness",
    "RFPropagationRequest",
    "RFPropagationResult",
    "marginal_and_joint",
]
