"""One packet's life: chosen action in, delivered-or-missed out.

Work plan section 9. This is the layer that composes the two channels and
contains, deliberately, **no channel equations of its own**. Both façades answer
the same two questions -- evaluate this attempt, and give me its probability
without consuming the tape -- so the lifecycle only has to sequence them and
apply the deadline.

**Both legs always pay.** Section 9 is explicit: there is no early cancellation,
and a DUP packet incurs both activation costs even when the radio arrives first
and the optical copy is discarded. That is what makes the cost model honest --
duplication is not free just because one leg happened to win -- and it is why
the constraint has something to trade against.

**The random tape is matched across actions.** One tape is drawn per packet and
every action reads from it, so RF-only, VLC-only and DUP are evaluated against
*the same* realized channel. Without that, the difference between two actions on
one packet carries a sampling artefact, and at a 1e-4 miss budget no amount of
averaging separates that artefact from a real effect. It is also what makes the
oracle's advantage a measurement: the oracle picks the best action on a packet
whose outcomes were all drawn from one tape, rather than the luckiest draw.

**Retransmission is asymmetric, and the asymmetry is physical.** The radio gets
three hopped attempts because collision is redrawn on every attempt. The optical
leg gets one because a blocked path stays blocked for hundreds of milliseconds,
so a second attempt inside a 3 ms deadline would spend airtime on a certainty.
Both numbers come from the frozen service profile rather than from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hybrid_v2x_rl.channels.rf.model import NRV2XChannel, RFChannelRequest, RFPacketRandomness
from hybrid_v2x_rl.channels.vlc.model import (
    VLCChannelRequest,
    VLCPacketRandomness,
    VVLCChannel,
)
from hybrid_v2x_rl.core.enums import FailureCause
from hybrid_v2x_rl.core.errors import HybridV2XError


class PacketError(HybridV2XError):
    """A packet was evaluated with an inconsistent request."""


@dataclass(frozen=True, slots=True)
class Action:
    """One of the three link selections, with its activation cost.

    Cost is carried on the action rather than looked up, so a caller cannot
    evaluate DUP and then charge for a single leg.
    """

    name: str
    uses_rf: bool
    uses_vlc: bool
    activation_cost: float

    def __post_init__(self) -> None:
        if not (self.uses_rf or self.uses_vlc):
            raise PacketError("an action must use at least one medium",
                              context={"action": self.name})


RF_ONLY = Action("RF", uses_rf=True, uses_vlc=False, activation_cost=1.0)
VLC_ONLY = Action("VLC", uses_rf=False, uses_vlc=True, activation_cost=1.0)
DUP = Action("DUP", uses_rf=True, uses_vlc=True, activation_cost=2.0)

ACTIONS = (RF_ONLY, VLC_ONLY, DUP)


@dataclass(frozen=True, slots=True)
class PacketTape:
    """Every uniform draw one packet consumes, for every action.

    Drawn once and shared, so the counterfactual "what would the other action
    have done on *this* packet" is a measurement rather than a resample. The RF
    attempts get their own tapes because each is an independent resource
    selection; the optical leg gets one because it has one stochastic step.
    """

    rf_attempts: tuple[RFPacketRandomness, ...]
    vlc: VLCPacketRandomness
    #: Small-scale fading power gain realized on each RF attempt.
    #:
    #: On the tape rather than in the request because it is part of the
    #: packet's realized channel, and the matched-tape rule says RF-only and
    #: DUP must see the *same* fade on the same packet. It is per attempt
    #: because the attempts hop: that is the entire reason the profile grants
    #: three of them rather than repeating one.
    #:
    #: Empty means "no per-attempt realization supplied" and the request's
    #: single ``fading_power_gain`` applies to every attempt -- which is a link
    #: with no hopping diversity, not a link with no fading.
    rf_fading_power_gains: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.rf_attempts:
            raise PacketError("at least one RF attempt tape is required")
        if self.rf_fading_power_gains and len(self.rf_fading_power_gains) != len(self.rf_attempts):
            raise PacketError(
                "fading gains must be supplied for every RF attempt or for none",
                context={
                    "attempts": len(self.rf_attempts),
                    "gains": len(self.rf_fading_power_gains),
                },
            )


@dataclass(frozen=True, slots=True)
class Timing:
    """The deadline arithmetic, from the frozen service and channel profiles."""

    deadline_s: float
    predecision_lead_s: float
    rf_airtime_s: float
    vlc_airtime_s: float
    rf_attempts: int

    @property
    def available_s(self) -> float:
        return self.deadline_s - self.predecision_lead_s

    @property
    def rf_total_airtime_s(self) -> float:
        return self.rf_attempts * self.rf_airtime_s

    def rf_arrival_s(self, attempt_index: int) -> float:
        """When attempt ``attempt_index`` completes, zero-based."""

        return (attempt_index + 1) * self.rf_airtime_s

    @property
    def vlc_arrival_s(self) -> float:
        return self.vlc_airtime_s

    def check_feasible(self) -> None:
        """Refuse a profile whose committed airtime cannot meet the deadline.

        The two media run concurrently -- different physical channels -- so the
        binding quantity is the longer leg, not the sum. A profile that failed
        this would produce misses attributable to arithmetic rather than to
        physics, and they would look identical in the logs.
        """

        longest = max(self.rf_total_airtime_s, self.vlc_airtime_s)
        if longest > self.available_s:
            raise PacketError(
                "committed airtime exceeds the deadline",
                context={
                    "rf_total_s": self.rf_total_airtime_s,
                    "vlc_s": self.vlc_airtime_s,
                    "available_s": self.available_s,
                },
            )


@dataclass(frozen=True, slots=True)
class PacketOutcome:
    """What happened to one packet, and why."""

    action: Action
    delivered: bool
    delivery_time_s: float | None
    failure_cause: FailureCause
    activation_cost: float
    rf_attempts_used: int
    rf_delivered: bool
    vlc_delivered: bool
    #: Marginal failure probabilities for this packet, for section 8.3.
    rf_failure_probability: float
    vlc_failure_probability: float
    #: Realized signal quality on each leg the action actually spent, in dB.
    #:
    #: ``None`` for a leg the action did not use, and that asymmetry is the
    #: point: section 6.3's sequential claim is that choosing a link buys a
    #: fresh reading of it while the other leg's estimate ages. A value here is
    #: the *oracle* quantity; the observation layer is handed a noisy, aged
    #: version of it, and :mod:`hybrid_v2x_rl.env.feedback` is the only thing allowed
    #: to make that conversion.
    rf_quality_db: float | None = None
    vlc_quality_db: float | None = None

    @property
    def joint_failure_probability(self) -> float:
        """The product the independence assumption would predict.

        Reported alongside the realized outcome so section 8.3's dependence
        ratio can be accumulated without recomputing anything: the numerator is
        the measured joint failure rate, this is the denominator.
        """

        return self.rf_failure_probability * self.vlc_failure_probability


@dataclass(slots=True)
class PacketLifecycle:
    """Applies an action to one pair pose and returns the outcome."""

    rf: NRV2XChannel
    vlc: VVLCChannel
    timing: Timing

    def __post_init__(self) -> None:
        self.timing.check_feasible()

    def _run_rf(
        self, request: RFChannelRequest, tape: PacketTape
    ) -> tuple[bool, float | None, FailureCause, int, float, float | None]:
        """Up to ``rf_attempts`` hopped attempts, stopping at the first success.

        Stopping early is not an optimization and does not save cost: the
        opportunity is pre-reserved, so the airtime is committed whether or not
        it is used. It only decides the arrival time.

        The SINR returned is the *last attempt actually made*, because that is
        the freshest thing a receiver could report back. The attempts hop, so
        they do not share a fade, and averaging them would report a channel
        that no single attempt saw.
        """

        attempts = min(self.timing.rf_attempts, len(tape.rf_attempts))
        probability = 1.0
        cause = FailureCause.NONE
        sinr_db: float | None = None
        for index in range(attempts):
            attempt = RFChannelRequest(
                distance_m=request.distance_m,
                propagation_state=request.propagation_state,
                blockage_db=request.blockage_db,
                shadowing_normalized=request.shadowing_normalized,
                fading_power_gain=(
                    tape.rf_fading_power_gains[index]
                    if tape.rf_fading_power_gains
                    else request.fading_power_gain
                ),
                neighbour_count=request.neighbour_count,
                sensed_fraction=request.sensed_fraction,
                randomness=tape.rf_attempts[index],
            )
            result = self.rf.evaluate(attempt)
            probability *= result.total_failure_probability
            sinr_db = result.sinr_db
            if result.success:
                return (True, self.timing.rf_arrival_s(index), FailureCause.NONE,
                        index + 1, probability, sinr_db)
            cause = result.failure_cause
        return False, None, cause, attempts, probability, sinr_db

    def _rf_failure_probability(self, request: RFChannelRequest, tape: PacketTape) -> float:
        """The **packet's** RF failure probability, over every granted attempt.

        Not one attempt's. Section 8.3 forms ``p_joint / (p_RF p_VLC)``, and
        p_VLC is already a packet-level quantity because the optical leg gets
        one attempt. Pairing it with a single-attempt radio probability would
        overstate the denominator by roughly the diversity order and make the
        ratio read far below one -- an apparent complementarity that is pure
        bookkeeping.

        The product form assumes attempts fail independently. That is the same
        assumption :meth:`_run_rf` realizes by drawing a fresh collision and a
        fresh hop per attempt, and it is the optimistic reading: correlated
        collisions across attempts would raise this. It is stated here because
        it is the single assumption the three-attempt profile rests on.
        """

        attempts = min(self.timing.rf_attempts, len(tape.rf_attempts))
        probability = 1.0
        for index in range(attempts):
            probability *= self.rf.failure_probability(
                RFChannelRequest(
                    distance_m=request.distance_m,
                    propagation_state=request.propagation_state,
                    blockage_db=request.blockage_db,
                    shadowing_normalized=request.shadowing_normalized,
                    fading_power_gain=(
                        tape.rf_fading_power_gains[index]
                        if tape.rf_fading_power_gains
                        else request.fading_power_gain
                    ),
                    neighbour_count=request.neighbour_count,
                    sensed_fraction=request.sensed_fraction,
                    randomness=tape.rf_attempts[index],
                )
            )
        return probability

    def run(
        self,
        *,
        action: Action,
        rf_request: RFChannelRequest,
        vlc_request: VLCChannelRequest,
        tape: PacketTape,
    ) -> PacketOutcome:
        """Deliver or miss one packet under ``action``.

        Both legs are evaluated whenever the action selects them, and the
        marginal probabilities of *both* media are always reported, even for a
        single-medium action. Section 8.3 needs the marginals on every packet to
        form the dependence ratio, and computing them only when an action
        happened to select the medium would condition the statistic on the
        policy -- which is exactly the bias it exists to detect.
        """

        rf_probability = self._rf_failure_probability(rf_request, tape)
        vlc_probability = self.vlc.failure_probability(vlc_request)

        rf_delivered = False
        rf_time: float | None = None
        rf_cause = FailureCause.NONE
        attempts_used = 0
        rf_quality_db: float | None = None
        if action.uses_rf:
            (rf_delivered, rf_time, rf_cause, attempts_used, _,
             rf_quality_db) = self._run_rf(rf_request, tape)

        vlc_delivered = False
        vlc_time: float | None = None
        vlc_cause = FailureCause.NONE
        vlc_quality_db: float | None = None
        if action.uses_vlc:
            result = self.vlc.evaluate(
                VLCChannelRequest(
                    geometry=vlc_request.geometry,
                    occluded=vlc_request.occluded,
                    randomness=tape.vlc,
                )
            )
            vlc_delivered = result.success
            vlc_cause = result.failure_cause
            vlc_quality_db = result.snr_db
            if vlc_delivered:
                vlc_time = self.timing.vlc_arrival_s

        arrivals = [t for t in (rf_time, vlc_time) if t is not None]
        delivered = bool(arrivals)
        delivery_time = min(arrivals) if arrivals else None

        if delivered:
            cause = FailureCause.NONE
        elif action.uses_rf and action.uses_vlc:
            # Both legs were spent and both failed. Named as such rather than
            # attributed to whichever happened to be tested last, because the
            # whole point of duplication is that this case is supposed to be
            # rare and it must be countable.
            cause = FailureCause.JOINT_FAILURE
        else:
            cause = rf_cause if action.uses_rf else vlc_cause

        return PacketOutcome(
            action=action,
            delivered=delivered,
            delivery_time_s=delivery_time,
            failure_cause=cause,
            activation_cost=action.activation_cost,
            rf_attempts_used=attempts_used,
            rf_delivered=rf_delivered,
            vlc_delivered=vlc_delivered,
            rf_failure_probability=rf_probability,
            vlc_failure_probability=vlc_probability,
            rf_quality_db=rf_quality_db,
            vlc_quality_db=vlc_quality_db,
        )

    def counterfactuals(
        self,
        *,
        rf_request: RFChannelRequest,
        vlc_request: VLCChannelRequest,
        tape: PacketTape,
    ) -> dict[str, PacketOutcome]:
        """Every action's outcome on the same packet and the same tape.

        This is what the oracle reads and what the matched-tape design exists
        for. Evaluating the actions separately with fresh randomness would make
        the oracle's advantage partly a sampling artefact, and at 1e-4 that is
        indistinguishable from a real effect.
        """

        return {
            action.name: self.run(
                action=action, rf_request=rf_request, vlc_request=vlc_request, tape=tape
            )
            for action in ACTIONS
        }


@dataclass(slots=True)
class DependenceAccumulator:
    """Section 8.3's joint-failure statistic, accumulated over packets.

    The ratio ``p_joint / (p_RF p_VLC)`` is one under independence, above one
    when the media fail together and below one when they complement. The
    contribution is a claim about which side of one it falls on, so it is
    measured rather than assumed -- and the analytical argument that it should
    sit near one (RF failure is collision-dominated, and collision does not care
    about the geometry that blocks light) is exactly the kind of argument that
    deserves a number.
    """

    packets: int = 0
    rf_failures: int = 0
    vlc_failures: int = 0
    joint_failures: int = 0
    expected_rf: float = 0.0
    expected_vlc: float = 0.0
    expected_joint: float = 0.0

    def observe(self, outcome: PacketOutcome) -> None:
        """Record one packet, using the marginals rather than the taken action.

        Both media's probabilities are available on every packet precisely so
        this statistic is not conditioned on what the policy chose.
        """

        self.packets += 1
        self.expected_rf += outcome.rf_failure_probability
        self.expected_vlc += outcome.vlc_failure_probability
        self.expected_joint += outcome.joint_failure_probability
        if not outcome.rf_delivered and outcome.action.uses_rf:
            self.rf_failures += 1
        if not outcome.vlc_delivered and outcome.action.uses_vlc:
            self.vlc_failures += 1
        if outcome.failure_cause is FailureCause.JOINT_FAILURE:
            self.joint_failures += 1

    @property
    def realized_joint_rate(self) -> float:
        return self.joint_failures / self.packets if self.packets else 0.0

    @property
    def predicted_joint_rate(self) -> float:
        """What independence would predict, averaged over the same packets."""

        return self.expected_joint / self.packets if self.packets else 0.0

    @property
    def dependence_ratio(self) -> float:
        """Realized over predicted. One means independent."""

        predicted = self.predicted_joint_rate
        if predicted <= 0.0:
            return math.inf if self.realized_joint_rate > 0.0 else 1.0
        return self.realized_joint_rate / predicted


__all__ = [
    "ACTIONS",
    "DUP",
    "RF_ONLY",
    "VLC_ONLY",
    "Action",
    "DependenceAccumulator",
    "PacketError",
    "PacketLifecycle",
    "PacketOutcome",
    "PacketTape",
    "Timing",
]
