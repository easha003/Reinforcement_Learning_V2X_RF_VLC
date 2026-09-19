"""The V-VLC channel, assembled into one outcome per packet.

The optical counterpart of :mod:`hybrid_v2x_rl.channels.rf.model`, and deliberately
the same shape: supplied randomness, a per-attempt outcome that names which
mechanism fired, and a failure probability computable without consuming the
tape. The environment then treats the two media identically and the packet
lifecycle contains no channel equations, which is the layering rule the
implementation spec sets out.

**The composition order is the physics, as it is on the radio side.** Occlusion
and field of view come from geometry alone; the beam is sampled only if the path
survives both; noise follows from the received power and the named ambient
condition; and the coded block error follows from the SNR. Deciding blockage
inside the channel would make it unpredictable from tracked positions and strip
the observation forecast of anything to forecast.

**Where this differs from the radio, and why it matters.** The RF channel has
two failure mechanisms that are separable and very unequal -- collision
dominates the budget by orders of magnitude. The optical channel has no
equivalent: there is no contention, because the link is directional and
point-to-point, so *every* optical failure is geometric or budgetary. That is
the asymmetry the whole contribution rests on, and it is visible here as a
structural difference between two files rather than as a claim in prose:

* an occluded optical path delivers nothing, and no power fixes it;
* a congested radio delivers nothing either, and no geometry fixes it.

Neither medium's dominant failure is the other's, which is what makes the pair
worth having.

**Retransmission is not modelled here.** The frozen profile grants the optical
leg one attempt, because a blocked path stays blocked for hundreds of
milliseconds -- retransmitting into it inside a 3 ms deadline is spending
airtime on a certainty. The radio's three attempts exist for the opposite
reason: collision is redrawn each time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.channels.vlc.headlamp_pattern import HeadlampPattern
from hybrid_v2x_rl.channels.vlc.noise import CLEAR_NIGHT, AmbientCondition, electrical_snr
from hybrid_v2x_rl.channels.vlc.ook import DEFAULT_CORRECTABLE_FRACTION, evaluate
from hybrid_v2x_rl.channels.vlc.optical_gain import (
    COMPLETE_BLOCKAGE,
    BlockageModel,
    beam_covers_pair,
    received_power_from_geometry,
)
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.core.enums import FailureCause
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.pair_geometry import PairGeometry


class VLCChannelError(HybridV2XError):
    """The optical channel was invoked with an inconsistent request."""


@dataclass(frozen=True, slots=True)
class VLCPacketRandomness:
    """The uniform draw one optical packet consumes, supplied by the caller.

    One draw rather than the radio's three, because the optical link has one
    stochastic step: whether the coded block decodes. Occlusion and field of
    view are decided by geometry and carry no randomness of their own.
    """

    decoding_draw: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.decoding_draw <= 1.0:
            raise VLCChannelError("decoding_draw must be a uniform draw in [0, 1]",
                                  context={"decoding_draw": self.decoding_draw})


@dataclass(frozen=True, slots=True)
class VLCChannelRequest:
    """One optical transmission attempt's inputs.

    ``occluded`` is supplied rather than derived, exactly as the radio takes its
    propagation class: the geometry engine owns that decision and the channel
    must not duplicate it.
    """

    geometry: PairGeometry
    occluded: bool
    randomness: VLCPacketRandomness


@dataclass(frozen=True, slots=True)
class VLCChannelResult:
    """Hidden truth and outcome for one optical attempt."""

    received_power_w: float
    electrical_snr: float
    within_field_of_view: bool
    occluded: bool
    bit_error_rate: float
    decoding_failure_probability: float
    total_failure_probability: float
    success: bool
    failure_cause: FailureCause
    #: Whether the transmitter's beam envelope covers this direction at all.
    #:
    #: Separate from ``within_field_of_view`` because the two are different
    #: design levers pointing at opposite ends of the link: a wider acceptance
    #: cone fixes one, a wider beam the other, and a P_out that merged them
    #: would say which link to build without saying which end to build it at.
    beam_aimed: bool = True

    @property
    def snr_db(self) -> float:
        return 10.0 * math.log10(self.electrical_snr) if self.electrical_snr > 0.0 else -math.inf

    @property
    def is_geometric_failure(self) -> bool:
        """Whether geometry alone decided this, with no appeal to the budget.

        The quantity that caps the hybrid gain at ``1 / P_out``: no optical
        power reaches a blocked path, one outside the acceptance cone, or one
        the beam does not illuminate, so these failures are immune to every
        transmitter and receiver *power* improvement. Reported per attempt so
        the cap is measured rather than assumed.
        """

        return self.occluded or not self.within_field_of_view or not self.beam_aimed


@dataclass(frozen=True, slots=True)
class VVLCChannel:
    """Composes the M4 modules into a per-packet outcome."""

    pattern: HeadlampPattern
    receiver: OpticalReceiver
    electrical_bandwidth_hz: float
    payload_bytes: int
    framing_bytes: int
    code_rate: float
    ambient: AmbientCondition = CLEAR_NIGHT
    blockage: BlockageModel = COMPLETE_BLOCKAGE
    correctable_fraction: float = DEFAULT_CORRECTABLE_FRACTION

    def evaluate(self, request: VLCChannelRequest) -> VLCChannelResult:
        """Deliver or lose one packet, and say which mechanism decided it."""

        geometry = request.geometry

        # Alignment has two halves and both are geometric. The receiver's is
        # the acceptance cone; the transmitter's is the beam's own envelope,
        # and it is the narrower of the two: the ECE R112 low-beam test points
        # span +/-9 degrees horizontally, so a pair further off the
        # transmitter's heading than that is a direction the lamp is not
        # required to illuminate and the artifact cannot answer for.
        #
        # Crediting it zero is a conservative choice, deliberately made and
        # deliberately visible: a real lamp does emit some stray light out
        # there, so this understates the optical link at exactly the junction
        # geometries where the two media are least alike. It is not a clamp --
        # the pattern still refuses to extrapolate, and this asks first.
        #
        # The consequence is that P_out, the geometric outage that caps the
        # hybrid gain at 1/P_out, now counts transmitter-side misalignment as
        # well as receiver-side. That is a larger P_out than the acceptance
        # cone alone gives, and it is the honest one.
        aimed = beam_covers_pair(self.pattern, geometry)
        in_view = geometry.within_field_of_view

        # Geometry first, and it short-circuits. A path outside the acceptance
        # cone is not a dim link, it is no link, and sampling the beam for it
        # would invite a later edit that lets a bright lamp leak through the
        # cone's edge.
        if request.occluded or not in_view or not aimed:
            cause = (
                FailureCause.VLC_OCCLUSION if request.occluded
                else FailureCause.VLC_ALIGNMENT
            )
            return VLCChannelResult(
                received_power_w=self.blockage.occluded_power_w if request.occluded else 0.0,
                electrical_snr=0.0,
                within_field_of_view=in_view,
                beam_aimed=aimed,
                occluded=request.occluded,
                bit_error_rate=0.5,
                decoding_failure_probability=1.0,
                total_failure_probability=1.0,
                success=False,
                failure_cause=cause,
            )

        power = received_power_from_geometry(
            pattern=self.pattern,
            receiver=self.receiver,
            geometry=geometry,
            occluded=False,
            blockage=self.blockage,
        )
        snr = electrical_snr(
            receiver=self.receiver,
            received_optical_power_w=power,
            bandwidth_hz=self.electrical_bandwidth_hz,
            ambient=self.ambient,
        )
        outcome = evaluate(
            snr,
            payload_bytes=self.payload_bytes,
            framing_bytes=self.framing_bytes,
            code_rate=self.code_rate,
            correctable_fraction=self.correctable_fraction,
        )
        failed = request.randomness.decoding_draw < outcome.packet_error_rate
        return VLCChannelResult(
            received_power_w=power,
            electrical_snr=snr,
            within_field_of_view=True,
            beam_aimed=True,
            occluded=False,
            bit_error_rate=outcome.bit_error_rate,
            decoding_failure_probability=outcome.packet_error_rate,
            total_failure_probability=outcome.packet_error_rate,
            success=not failed,
            failure_cause=FailureCause.VLC_CHANNEL if failed else FailureCause.NONE,
        )

    def failure_probability(self, request: VLCChannelRequest) -> float:
        """Failure probability without consuming the draw.

        The oracle needs every action's probability on the same packet, and
        evaluating an unused one through :meth:`evaluate` would burn tape and
        shift the used one's realization.
        """

        return self.evaluate(request).total_failure_probability


__all__ = [
    "VLCChannelError",
    "VLCChannelRequest",
    "VLCChannelResult",
    "VLCPacketRandomness",
    "VVLCChannel",
]
