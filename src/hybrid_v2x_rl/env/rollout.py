"""Drive tagged-pair episodes from a trace through the packet lifecycle.

This is the loop that turns a stored campaign into packet outcomes: for each
sampled instant of each tagged pair it rebuilds the exact geometry, asks the
geometry engine what is obstructed and how, draws one random tape, and evaluates
whichever actions the caller wants.

**It contains no policy.** A caller supplies an action-chooser, so the same loop
serves a fixed baseline, the oracle reading counterfactuals, and later a learned
policy. That is what keeps the comparison honest: every policy sees the same
packets, the same geometry and the same tapes, and differs only in what it picks.

**Randomness is derived from packet identity, not from a running stream.** The
tape for a packet is seeded from the trace, the pair and the packet index, so it
does not depend on how many packets were evaluated before it or on how many
actions were tried. Two runs that evaluate different action sets still see
identical channels on the same packet, which is what makes an ablation a
comparison rather than two separate experiments.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from hybrid_v2x_rl.channels.rf.fading import FadingProcess
from hybrid_v2x_rl.channels.rf.model import (
    RFChannelRequest,
    RFPacketRandomness,
    RFPropagationRequest,
    RFPropagationResult,
)
from hybrid_v2x_rl.channels.rf.pathloss_37885 import blockage_mean_db, blockage_sigma_db
from hybrid_v2x_rl.channels.rf.shadowing import ShadowingProcess
from hybrid_v2x_rl.channels.vlc.model import (
    VLCChannelRequest,
    VLCChannelResult,
    VLCPacketRandomness,
)
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.geometry import OrientedRectangle, Segment
from hybrid_v2x_rl.core.link_endpoints import (
    DEFAULT_RF_ANTENNA_HEIGHT_M,
    optical_link_path,
    rf_link_path,
)
from hybrid_v2x_rl.core.pair_geometry import (
    DEFAULT_FOV_HALF_ANGLE_RAD,
    PairGeometry,
    pair_geometry,
)
from hybrid_v2x_rl.core.randomness import derive_seed, make_generator
from hybrid_v2x_rl.env.episodes import VehiclePose
from hybrid_v2x_rl.env.packet import Action, PacketLifecycle, PacketOutcome, PacketTape
from hybrid_v2x_rl.geometry.rf_visibility import classify_path
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex
from hybrid_v2x_rl.geometry.vehicle_occlusion import is_obstructed

#: Neighbours within this range contend for the same sidelink resources.
CONTENTION_RADIUS_M = 200.0

#: Share of contenders whose resource reservations this transmitter can
#: decode. It multiplies the collision model's own ``sensing_reliability``:
#: one is *where* the neighbours are, the other is how well sensing works once
#: they are heard.
#:
#: Held at 1.0 -- every contender assumed decodable -- and that is optimistic.
#: A contender behind a building is NLOS and its reservation is not decodable,
#: so real hidden-terminal counts are higher and collisions worse. Deriving it
#: would mean a visibility classification per contender per packet, several
#: hundred ray casts where the model currently does two, and it is deferred on
#: that basis rather than because it does not matter. The pessimistic
#: sensitivity band, which drops sensing reliability to 0.7, covers part of the
#: same ground.
#:
#: It lives here as a named constant rather than a literal in the request so
#: that it is visible to a reader and changeable in one place.
DEFAULT_SENSED_FRACTION = 1.0

#: Suffix separating a pair's blockage state from its shadowing state.
#:
#: The NLOSv blockage loss carries its own spread on top of the class
#: shadowing, so the two need independent draws. Reusing one normal for both
#: would make a deeply shadowed link *necessarily* a deeply blocked one, which
#: manufactures exactly the kind of correlation section 8.3 exists to measure.
#: Keying the same correlated process under a different namespace gives the
#: independence for free and keeps the blockage residual persistent in travelled
#: distance, which it should be: the van in front stays in front.
_BLOCKAGE_KEY_SUFFIX = "|blockage"


class RolloutSeedError(HybridV2XError):
    """A rollout seed or trace transition would contaminate channel streams."""


@dataclass(frozen=True, slots=True)
class PacketContext:
    """Everything one packet needed, kept so a result can be conditioned on it.

    Section 8.3 requires the joint-failure statistic conditioned on density,
    separation, intersection distance and propagation class, so those travel
    with the outcome rather than being recomputed from it.
    """

    trace_id: str
    pair_id: str
    time_s: float
    density: float
    separation_m: float
    optical_path_m: float
    propagation_state: RFPropagationState
    occluded: bool
    within_field_of_view: bool
    neighbour_count: int


@dataclass(frozen=True, slots=True)
class PairChannelEvaluation:
    """Action-independent RF and VLC truth after one physical-state advance."""

    context: PacketContext
    rf_propagation: RFPropagationResult
    vlc_result: VLCChannelResult


@dataclass(frozen=True, slots=True)
class _AdvancedPairState:
    """Internal physical state shared by legacy and population evaluators."""

    context: PacketContext
    rf_propagation_request: RFPropagationRequest
    geometry: PairGeometry
    occluded: bool
    fading_power_gains: tuple[float, ...]


ActionChooser = Callable[[PacketContext], Action]


def always(action: Action) -> ActionChooser:
    """A fixed-action baseline."""

    return lambda _context: action


@dataclass(slots=True)
class Rollout:
    """Replays a campaign through the lifecycle under a chosen policy."""

    lifecycle: PacketLifecycle
    buildings: Sequence[OrientedRectangle]
    root_seed: int = 0
    fov_half_angle_rad: float = DEFAULT_FOV_HALF_ANGLE_RAD
    sensed_fraction: float = DEFAULT_SENSED_FRACTION
    shadowing: ShadowingProcess = field(init=False)
    fading: FadingProcess = field(init=False)
    _last_time_s: dict[str, float] = field(default_factory=dict, repr=False)
    _active_trace_id: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            make_generator(self.root_seed, "rollout.root-validation")
        except (TypeError, ValueError) as error:
            raise RolloutSeedError("root_seed is not a valid unsigned seed") from error
        self.shadowing = ShadowingProcess(
            rng=None,
            generator_factory=lambda key: self._channel_generator("shadowing", key),
        )
        self.fading = FadingProcess(
            rng=None,
            carrier_hz=self.lifecycle.rf.carrier_hz,
            subchannel_separations_hz=self._hop_offsets_hz(),
            generator_factory=lambda key: self._channel_generator("fading", key),
        )

    def _channel_generator(
        self,
        stream_name: str,
        pair_state_key: str,
    ) -> np.random.Generator:
        """Derive one persistent channel stream from trace and pair identity."""

        trace_id = self._active_trace_id
        if trace_id is None:
            raise RolloutSeedError(
                "a trace must be bound before channel randomness is requested"
            )
        return make_generator(
            self.root_seed,
            stream_name,
            trace_id=trace_id,
            episode_id=pair_state_key,
        )

    def _bind_trace(self, trace_id: str) -> None:
        """Bind stateful channels to one trace until every live pair is released."""

        if not isinstance(trace_id, str) or not trace_id.strip():
            raise RolloutSeedError("trace_id must be a non-empty string")
        if self._active_trace_id in (None, trace_id):
            self._active_trace_id = trace_id
            return
        if (
            self.shadowing.live_links() > 0
            or self.fading.live_links() > 0
            or self._last_time_s
        ):
            raise RolloutSeedError(
                "cannot change traces while correlated pair state is live",
                context={
                    "active_trace_id": self._active_trace_id,
                    "requested_trace_id": trace_id,
                },
            )
        self._active_trace_id = trace_id

    def _hop_offsets_hz(self) -> tuple[float, ...]:
        """Where in the band each granted attempt lands.

        Attempts are spread evenly across the configured bandwidth rather than
        stacked, because that is what makes three attempts worth more than one
        repetition. With a 200 ns urban delay spread the subchannels decorrelate
        to 0.5 within about 1.4 MHz, so a 10 MHz band gives three near-
        independent hops -- and the frequency correlation is computed from that
        spacing rather than asserted, so a narrower band would correctly show
        less diversity instead of the same amount.
        """

        attempts = self.lifecycle.timing.rf_attempts
        if attempts == 1:
            return (0.0,)
        span = self.lifecycle.rf.bandwidth_hz
        return tuple(
            -0.5 * span + span * index / (attempts - 1) for index in range(attempts)
        )

    def _tape(
        self, trace_id: str, pair_id: str, index: int, fading_gains: Iterable[float]
    ) -> PacketTape:
        """A tape derived from packet identity so it cannot drift with order."""

        seed = derive_seed(
            self.root_seed,
            "decoding",
            trace_id=trace_id,
            episode_id=pair_id,
            packet_index=index,
        )
        rng = np.random.default_rng(seed)
        attempts = tuple(
            RFPacketRandomness(*rng.random(3))
            for _ in range(self.lifecycle.timing.rf_attempts)
        )
        return PacketTape(
            rf_attempts=attempts,
            vlc=VLCPacketRandomness(float(rng.random())),
            rf_fading_power_gains=tuple(float(g) for g in fading_gains),
        )

    def evaluate_instant(
        self,
        *,
        trace_id: str,
        pair_id: str,
        index: int,
        density: float,
        time_s: float,
        transmitter: VehiclePose,
        receiver: VehiclePose,
        neighbours: Sequence[VehiclePose],
        index_of_frame: SpatialIndex,
        choose: ActionChooser,
        counterfactual: bool = False,
    ) -> tuple[PacketOutcome, PacketContext, dict[str, PacketOutcome] | None]:
        """One packet at one pair pose."""

        state = self._advance_pair_state(
            trace_id=trace_id,
            pair_id=pair_id,
            density=density,
            time_s=time_s,
            transmitter=transmitter,
            receiver=receiver,
            neighbours=neighbours,
            index_of_frame=index_of_frame,
        )
        tape = self._tape(trace_id, pair_id, index, state.fading_power_gains)

        propagation = state.rf_propagation_request
        rf_request = RFChannelRequest(
            distance_m=propagation.distance_m,
            propagation_state=propagation.propagation_state,
            blockage_db=propagation.blockage_db,
            shadowing_normalized=propagation.shadowing_normalized,
            # Superseded per attempt by the tape's hopped gains; carried so a
            # caller that supplies no gains still gets an unfaded budget rather
            # than a zero one.
            fading_power_gain=1.0,
            neighbour_count=state.context.neighbour_count,
            sensed_fraction=self.sensed_fraction,
            randomness=tape.rf_attempts[0],
        )
        vlc_request = VLCChannelRequest(
            geometry=state.geometry,
            occluded=state.occluded,
            randomness=tape.vlc,
        )

        alternatives = None
        if counterfactual:
            alternatives = self.lifecycle.counterfactuals(
                rf_request=rf_request, vlc_request=vlc_request, tape=tape
            )
        outcome = self.lifecycle.run(
            action=choose(state.context),
            rf_request=rf_request,
            vlc_request=vlc_request,
            tape=tape,
        )
        return outcome, state.context, alternatives

    def evaluate_channels(
        self,
        *,
        trace_id: str,
        pair_id: str,
        density: float,
        time_s: float,
        transmitter: VehiclePose,
        receiver: VehiclePose,
        neighbours: Sequence[VehiclePose],
        index_of_frame: SpatialIndex,
        vlc_randomness: VLCPacketRandomness,
    ) -> PairChannelEvaluation:
        """Advance one pair once and expose action-independent channel truth.

        The population environment supplies its identity-addressed optical draw.
        RF access randomness remains in the matched packet tape and is consumed
        only after the complete joint action fixes shared-pool contention.
        """

        if not isinstance(vlc_randomness, VLCPacketRandomness):
            raise TypeError("vlc_randomness must be VLCPacketRandomness")
        state = self._advance_pair_state(
            trace_id=trace_id,
            pair_id=pair_id,
            density=density,
            time_s=time_s,
            transmitter=transmitter,
            receiver=receiver,
            neighbours=neighbours,
            index_of_frame=index_of_frame,
        )
        return PairChannelEvaluation(
            context=state.context,
            rf_propagation=self.lifecycle.rf.evaluate_propagation(
                state.rf_propagation_request
            ),
            vlc_result=self.lifecycle.vlc.evaluate(
                VLCChannelRequest(
                    geometry=state.geometry,
                    occluded=state.occluded,
                    randomness=vlc_randomness,
                )
            ),
        )

    def _advance_pair_state(
        self,
        *,
        trace_id: str,
        pair_id: str,
        density: float,
        time_s: float,
        transmitter: VehiclePose,
        receiver: VehiclePose,
        neighbours: Sequence[VehiclePose],
        index_of_frame: SpatialIndex,
    ) -> _AdvancedPairState:
        """Advance correlated channel state without selecting or sampling an action."""

        self._bind_trace(trace_id)

        geometry = pair_geometry(
            transmitter, receiver, fov_half_angle_rad=self.fov_half_angle_rad
        )
        optical = optical_link_path(transmitter, receiver)
        radio = rf_link_path(transmitter, receiver)
        widest = (
            max(0.5 * math.hypot(v.length_m, v.width_m) for v in neighbours) if neighbours else 3.0
        )

        exclude = (transmitter.vehicle_id, receiver.vehicle_id)
        optical_blockers = index_of_frame.candidates(optical.segment, margin_m=widest)
        occluded = is_obstructed(optical, optical_blockers, exclude_ids=exclude)

        # The radio path runs centre to centre and the optical path headlamp to
        # photodiode, so their candidate sets are *not* the same. Looking a
        # radio blocker's height up in the optical set is how a blocking truck
        # goes missing and an NLOSv link is charged the default height's ~0 dB.
        radio_blockers = index_of_frame.candidates(radio.segment, margin_m=widest)
        visibility = classify_path(
            radio, radio_blockers, self.buildings, exclude_ids=exclude
        )

        centre = Segment(optical.segment.start, optical.segment.start)
        contenders = sum(
            1
            for other in index_of_frame.candidates(centre, margin_m=CONTENTION_RADIUS_M)
            if other.vehicle_id != transmitter.vehicle_id
            and math.hypot(other.x_m - transmitter.x_m, other.y_m - transmitter.y_m)
            <= CONTENTION_RADIUS_M
        )

        # Shadowing persists along the pair's own trajectory, so it is keyed by
        # the pair rather than redrawn: a shadowed link must stay shadowed for
        # tens of packets or it is a fast fade under another name. The elapsed
        # time comes from the caller rather than a nominal period, so a trace
        # sampled at an irregular cadence decorrelates by how far the pair
        # actually travelled.
        elapsed_s = max(0.0, time_s - self._last_time_s.get(pair_id, time_s))
        self._last_time_s[pair_id] = time_s
        travelled = transmitter.speed_mps * elapsed_s
        normalized = self.shadowing.advance(pair_id, travelled, visibility.state)

        blockage_db = 0.0
        if visibility.state is RFPropagationState.NLOSV and visibility.vehicle_blocker_ids:
            tallest = max(
                (
                    v.height_m
                    for v in radio_blockers
                    if v.vehicle_id in visibility.vehicle_blocker_ids
                ),
                default=DEFAULT_RF_ANTENNA_HEIGHT_M,
            )
            residual = self.shadowing.advance(
                pair_id + _BLOCKAGE_KEY_SUFFIX, travelled, visibility.state
            )
            mean = blockage_mean_db(
                geometry.separation_m,
                tallest,
                DEFAULT_RF_ANTENNA_HEIGHT_M,
                DEFAULT_RF_ANTENNA_HEIGHT_M,
            )
            sigma = blockage_sigma_db(
                tallest, DEFAULT_RF_ANTENNA_HEIGHT_M, DEFAULT_RF_ANTENNA_HEIGHT_M
            )
            blockage_db = max(0.0, mean + sigma * residual)

        gains = self.fading.advance(
            pair_id,
            elapsed_s=elapsed_s,
            tx_speed_mps=transmitter.speed_mps,
            rx_speed_mps=receiver.speed_mps,
            state=visibility.state,
        )

        context = PacketContext(
            trace_id=trace_id,
            pair_id=pair_id,
            time_s=time_s,
            density=density,
            separation_m=geometry.separation_m,
            optical_path_m=geometry.optical_path_length_m,
            propagation_state=visibility.state,
            occluded=occluded,
            within_field_of_view=geometry.within_field_of_view,
            neighbour_count=contenders,
        )
        rf_request = RFPropagationRequest(
            distance_m=max(geometry.separation_m, 1.0),
            propagation_state=visibility.state,
            blockage_db=blockage_db,
            shadowing_normalized=normalized,
            # The shared-pool model currently exposes one propagation risk for
            # every reserved attempt.  Use the first frequency-hop state as that
            # contract's focal attempt; legacy replay below still consumes the
            # complete hopped sequence from ``fading_power_gains``.
            fading_power_gain=float(gains[0]),
        )
        return _AdvancedPairState(
            context=context,
            rf_propagation_request=rf_request,
            geometry=geometry,
            occluded=occluded,
            fading_power_gains=tuple(float(gain) for gain in gains),
        )

    def release(self, pair_id: str) -> None:
        """Drop a finished pair's shadowing state.

        Without this a re-formed pair would inherit shadowing from an unrelated
        earlier encounter somewhere else on the map, and the dictionary would
        grow for the life of the run. Every per-pair state is dropped together:
        one of them surviving would be worse than none of them being dropped,
        because the stale one would be silently correlated with a fresh one.
        """

        self.shadowing.forget(pair_id)
        self.shadowing.forget(pair_id + _BLOCKAGE_KEY_SUFFIX)
        self.fading.forget(pair_id)
        self._last_time_s.pop(pair_id, None)


def best_action(alternatives: dict[str, PacketOutcome]) -> Action:
    """The oracle's choice: cheapest action that delivers on this exact packet.

    Cheapest rather than most reliable, because every action that delivers is
    equally successful on a packet that has already been drawn. Preferring the
    cheapest is what makes the oracle a *bound on cost at zero misses* rather
    than a policy that duplicates everything -- and duplicating everything is
    already available as a baseline.
    """

    delivering = [o for o in alternatives.values() if o.delivered]
    if not delivering:
        # Nothing works; take the cheapest so the miss is not also expensive.
        return min(alternatives.values(), key=lambda o: o.activation_cost).action
    return min(delivering, key=lambda o: o.activation_cost).action


__all__ = [
    "CONTENTION_RADIUS_M",
    "DEFAULT_SENSED_FRACTION",
    "ActionChooser",
    "PairChannelEvaluation",
    "PacketContext",
    "Rollout",
    "RolloutSeedError",
    "always",
    "best_action",
]
