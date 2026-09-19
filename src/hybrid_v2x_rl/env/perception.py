"""Turn a pair instant into what a deployable radio could actually know.

This is the causal half of the fork described in work plan section 5. The
channels are driven by exact geometry because they are physics; a policy is
driven by *this*, because it has to run on a vehicle. The two must never meet,
and the separation is structural rather than advisory: nothing in this module
imports the geometry package or :mod:`hybrid_v2x_rl.core.pair_geometry`, and a test
asserts that by parsing the imports rather than trusting the reader.

The chain is sensing, tracking, forecasting, blockage estimation, and assembly:

* positions are measured on a \\SI{50}{\\milli\\second} grid with
  \\SI{50}{\\milli\\second} of latency and metre-scale noise;
* tracks age and are forgotten, so a neighbour that has not been heard from
  recently simply is not there;
* states are propagated \\SI{200}{\\milli\\second} forward, because a decision
  taken now applies to a packet sent later;
* blockage is a *probability* with a confidence, computed from forecast
  positions rather than read off the simulator.

**Sensing carries no vehicle dimensions, and that is not an oversight.** A
:class:`~hybrid_v2x_rl.observation.sensing.TrackSample` holds an identity, a time, a
position, a speed and a heading. It does not hold length, width or height,
because those are not what a position-reporting sensor reports. Blockage
estimation therefore uses one nominal body for every blocker, which means the
policy cannot tell a hatchback from a bus.

That limitation has already been measured from the other side: finding 2 in
:mod:`hybrid_v2x_rl.channels.vlc.optical_gain` records that leader length moves the
optical budget by \\SI{4.7}{\\decibel} optical and \\SI{9.4}{\\decibel}
electrical between a car and an \\SI{11}{\\meter} bus. So the single largest
unobservable in this system is a quantity the channel cares about a great deal
and the observation cannot see at all. Giving the policy true dimensions would
close that gap by handing it hidden state; supplying a nominal body keeps the
gap honest and leaves it visible as a limitation.

**The neighbour count is sensed, not counted.** The rollout tells the channel
how many contenders are really within range; this module tells the policy how
many it can currently hear. They differ by exactly the tracks that have aged
out or were never seen, which is the point.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

from hybrid_v2x_rl.channels.rf.collision import CollisionParameters, channel_busy_ratio
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.enums import Action as ObservedAction
from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.geometry import Point, Segment
from hybrid_v2x_rl.core.intersection_context import JunctionGrid, intersection_context
from hybrid_v2x_rl.env.episodes import PairInstant
from hybrid_v2x_rl.env.packet import Action
from hybrid_v2x_rl.observation.blockage import BlockerShape, blockage_probability
from hybrid_v2x_rl.observation.builder import ObservationBuilder, PairObservationInputs
from hybrid_v2x_rl.observation.forecast import ConstantVelocityForecaster, PredictedState
from hybrid_v2x_rl.observation.link_state import LinkStateTracker
from hybrid_v2x_rl.observation.sensing import SensorModel, sense_frame
from hybrid_v2x_rl.observation.tracks import TrackStore

#: Radius within which a track is treated as a contender, matching the
#: channel's own contention radius so the sensed count and the true count are
#: the same quantity measured two ways.
CONTENTION_RADIUS_M = 200.0

#: The body assumed for every blocker. Sensing reports where a vehicle is, not
#: how large it is, so one shape stands for all of them. These are the
#: passenger-car dimensions of the fleet's modal class -- the commonest guess,
#: not the safest one, because a policy that assumed every blocker was a bus
#: would refuse the optical link everywhere.
NOMINAL_BLOCKER = BlockerShape(length_m=4.5, width_m=1.8, height_m=1.5)

#: The environment and the observation layer name the three actions with
#: different types, and this is the seam between them. M5's
#: :class:`~hybrid_v2x_rl.env.packet.Action` is a dataclass carrying an activation
#: cost, because the lifecycle has to charge for it; M2's
#: :class:`~hybrid_v2x_rl.core.enums.Action` is an enum, because the link-state
#: tracker only needs to key a dictionary by it.
#:
#: Translating here rather than unifying them is deliberate. The cost belongs
#: to the action in the environment and would be meaningless in an enum used as
#: a dictionary key, and the observation layer must not acquire a notion of
#: cost -- a policy that could read an action's price off its own observation
#: would be reading the objective rather than the state.
_OBSERVED_ACTION = {
    "RF": ObservedAction.RF,
    "VLC": ObservedAction.VLC,
    "DUP": ObservedAction.DUP,
}

#: Cell size of the spatial hash over forecast positions. Set equal to the
#: contention radius so a neighbour query touches nine cells and misses
#: nothing: any point within the radius of a cell's occupant lies in that cell
#: or one adjacent to it.
_CELL_M = CONTENTION_RADIUS_M

#: Tracks staler than this are not offered to the blockage estimator. It is the
#: forecast horizon: a position propagated further than it was measured ahead
#: is mostly the propagation.
_BLOCKER_FRESHNESS_S = 1.0


def _near_segment(
    x: float, y: float, tx: PredictedState, rx: PredictedState, margin_m: float
) -> bool:
    """Whether a point lies within ``margin_m`` of the transmitter-receiver line."""

    dx, dy = rx.x_m - tx.x_m, rx.y_m - tx.y_m
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(x - tx.x_m, y - tx.y_m) <= margin_m
    along = ((x - tx.x_m) * dx + (y - tx.y_m) * dy) / length_sq
    if along < 0.0 or along > 1.0:
        return False
    perpendicular = abs((x - tx.x_m) * dy - (y - tx.y_m) * dx) / math.sqrt(length_sq)
    return perpendicular <= margin_m


@dataclass(slots=True)
class Perception:
    """The causal view of one running trace.

    Holds the sensor, the track store, the forecaster and one link-state
    tracker per pair. It is stateful in the way a vehicle is stateful: what it
    knows depends on what it has recently heard.
    """

    sensor: SensorModel
    tracks: TrackStore
    forecaster: ConstantVelocityForecaster
    builder: ObservationBuilder
    grid: JunctionGrid
    collision: CollisionParameters
    fov_half_angle_rad: float
    root_seed: int = 0
    _links: dict[str, LinkStateTracker] = field(default_factory=dict, repr=False)
    _sensed_at: float | None = field(default=None, repr=False)
    _predicted: dict[str, PredictedState] = field(default_factory=dict, repr=False)
    _cells: dict[tuple[int, int], list[PredictedState]] = field(
        default_factory=dict, repr=False)
    _neighbours: dict[str, int] = field(default_factory=dict, repr=False)

    def _link_state(self, pair_id: str) -> LinkStateTracker:
        state = self._links.get(pair_id)
        if state is None:
            state = self._links[pair_id] = LinkStateTracker(
                history_packets=self.builder.schema.history_packets
            )
        return state

    def _sense(self, instant: PairInstant) -> None:
        """Refresh the track store once per frame, not once per pair.

        Every pair in a frame shares one sensor sweep, exactly as they share
        one spatial index on the exact side. Sensing per pair would give a
        vehicle a fresh measurement for each of its neighbours and quietly
        remove the latency this module exists to model.
        """

        if self._sensed_at == instant.time_s:
            return
        samples = sense_frame(
            instant.neighbours,
            now_s=instant.time_s,
            sensor=self.sensor,
            root_seed=self.root_seed,
            trace_id=instant.trace_id,
        )
        self.tracks.update(samples, now_s=instant.time_s)
        # Forecast every live track once per frame. A predicted state depends
        # on the track and the clock, not on which pair is asking, so doing it
        # per pair repeats the same arithmetic once for every pair in the
        # frame -- which at density 20 is several hundred times.
        self._predicted = {
            track.vehicle_id: self.forecaster.predict(track, now_s=instant.time_s)
            for track in self.tracks.fresh_tracks(
                instant.time_s, limit_s=_BLOCKER_FRESHNESS_S)
        }
        # One spatial hash per frame over the forecasts. Both the contender
        # count and the blocker filter were scanning every live track for every
        # pair, which at density 20 is a few hundred tracks times a few hundred
        # pairs and dominated the whole chain. Bucketing once per frame turns
        # both into local queries.
        self._cells = {}
        for state in self._predicted.values():
            key = (int(state.x_m // _CELL_M), int(state.y_m // _CELL_M))
            self._cells.setdefault(key, []).append(state)
        self._neighbours = {}
        self._sensed_at = instant.time_s

    def _cells_near(
        self, x0: float, y0: float, x1: float, y1: float, margin_m: float
    ) -> Iterator[PredictedState]:
        """Forecast states in every cell the box touches."""

        lo_x = int((min(x0, x1) - margin_m) // _CELL_M)
        hi_x = int((max(x0, x1) + margin_m) // _CELL_M)
        lo_y = int((min(y0, y1) - margin_m) // _CELL_M)
        hi_y = int((max(y0, y1) + margin_m) // _CELL_M)
        for cx in range(lo_x, hi_x + 1):
            for cy in range(lo_y, hi_y + 1):
                yield from self._cells.get((cx, cy), ())

    def _contenders(self, centre: PredictedState) -> int:
        """Live tracks within the contention radius, at the forecast horizon.

        Counted over *forecast* positions rather than the tracks' own, and that
        is a deliberate choice rather than a consequence of how this is
        computed. Every other quantity in the observation -- the pair's own
        geometry, the blockage estimate -- is evaluated at the horizon, because
        the decision applies to a packet sent later. A contender count taken at
        measurement time while blockage is taken at the horizon would describe
        two different instants and invite the policy to correlate them.

        Measured against the track-based count over 1,316 instants at density
        20, the two agree on 51% and differ by -0.19 on average against counts
        near a hundred, so the choice is defensible either way and is recorded
        here rather than left implicit.

        Cached per transmitter because it depends on the transmitter alone, and
        a frame holds many pairs that share one.
        """

        cached = self._neighbours.get(centre.vehicle_id)
        if cached is not None:
            return cached
        radius_sq = CONTENTION_RADIUS_M * CONTENTION_RADIUS_M
        count = sum(
            1
            for other in self._cells_near(
                centre.x_m, centre.y_m, centre.x_m, centre.y_m, CONTENTION_RADIUS_M)
            if other.vehicle_id != centre.vehicle_id
            and (other.x_m - centre.x_m) ** 2 + (other.y_m - centre.y_m) ** 2 <= radius_sq
        )
        self._neighbours[centre.vehicle_id] = count
        return count

    def observe(self, instant: PairInstant) -> tuple[float, ...] | None:
        """The observation vector for this packet, or ``None`` if unusable.

        ``None`` means the transmitter or receiver has no live track: the
        sensor has not yet reported them, or their last report has aged out.
        A deployable radio in that position has nothing to condition on, and
        the caller must decide what to do about it rather than being handed a
        vector of plausible-looking defaults.
        """

        self._sense(instant)
        now = instant.time_s

        tx_track = self.tracks.get(instant.transmitter.vehicle_id)
        rx_track = self.tracks.get(instant.receiver.vehicle_id)
        if tx_track is None or rx_track is None:
            return None

        tx = self._predicted.get(tx_track.vehicle_id) or self.forecaster.predict(
            tx_track, now_s=now)
        rx = self._predicted.get(rx_track.vehicle_id) or self.forecaster.predict(
            rx_track, now_s=now)

        # Only tracks near the path can obstruct it. Offering every live track
        # to the estimator is correct but quadratic in fleet size, and the
        # estimator then rejects almost all of them on the same geometry this
        # test applies once. The margin is generous -- half a nominal body plus
        # the forecast's own position uncertainty -- so the filter removes work
        # rather than candidates.
        margin = 0.5 * NOMINAL_BLOCKER.length_m + 3.0 * max(
            tx.cross_track_std_m, rx.cross_track_std_m, 1.0)
        blockers = [
            (state, NOMINAL_BLOCKER)
            for state in self._cells_near(tx.x_m, tx.y_m, rx.x_m, rx.y_m, margin)
            if state.vehicle_id not in (tx.vehicle_id, rx.vehicle_id)
            and _near_segment(state.x_m, state.y_m, tx, rx, margin)
        ]
        blockage = blockage_probability(
            tx, rx, blockers, horizon_s=self.forecaster.horizon_s
        )

        neighbours = self._contenders(tx)
        busy = channel_busy_ratio(neighbours, self.collision)

        junctions = intersection_context(
            transmitter_x_m=tx.x_m, transmitter_y_m=tx.y_m,
            transmitter_heading_rad=tx.heading_rad,
            receiver_x_m=rx.x_m, receiver_y_m=rx.y_m,
            receiver_heading_rad=rx.heading_rad,
            path=Segment(Point(tx.x_m, tx.y_m), Point(rx.x_m, rx.y_m)),
            grid=self.grid,
        )

        return self.builder.build(PairObservationInputs(
            now_s=now,
            transmitter=tx,
            receiver=rx,
            transmitter_track=tx_track,
            receiver_track=rx_track,
            blockage=blockage,
            links=self._link_state(instant.pair_id),
            neighbour_count=neighbours,
            channel_busy_ratio=busy,
            fov_half_angle_rad=self.fov_half_angle_rad,
            intersection=junctions,
        ))

    def record(
        self,
        pair_id: str,
        *,
        action: Action,
        at_s: float,
        delivered: bool,
        measurements: dict[Link, float] | None = None,
    ) -> None:
        """Feed one packet's outcome back into the link history.

        Feedback is action-dependent by construction: a packet that used only
        the radio teaches nothing about the optical link, so ``measurements``
        should carry only the legs the action actually spent. That asymmetry is
        why the observation carries a quality *age* beside every quality.
        """

        try:
            observed = _OBSERVED_ACTION[action.name]
        except KeyError:  # pragma: no cover - guarded by test
            raise KeyError(
                f"no observation-layer action for {action.name!r}; extend "
                f"_OBSERVED_ACTION when the action set grows"
            ) from None
        self._link_state(pair_id).record(
            action=observed, at_s=at_s, delivered=delivered, measurements=measurements
        )

    def release(self, pair_id: str) -> None:
        """Drop a finished pair's link history."""

        self._links.pop(pair_id, None)


def build_perception(config: ProjectConfig, *, root_seed: int = 0) -> Perception:
    """Assemble the causal view from a resolved configuration."""

    from hybrid_v2x_rl.env.assembly import build_rf_channel

    grid = config.mobility.grid
    if grid is None:  # pragma: no cover - rejected by configuration validation
        raise ValueError("perception requires a configured mobility grid")
    return Perception(
        sensor=SensorModel.from_config(config.observation),
        tracks=TrackStore.from_config(config.observation),
        forecaster=ConstantVelocityForecaster.from_config(config.observation),
        builder=ObservationBuilder.from_config(config.observation),
        grid=JunctionGrid(
            avenues=grid.avenues,
            cross_streets=grid.cross_streets,
            avenue_spacing_m=grid.avenue_spacing_m,
            cross_street_spacing_m=grid.cross_street_spacing_m,
            road_half_width_m=grid.lanes_per_direction * grid.lane_width_m,
        ),
        collision=build_rf_channel(config).collision,
        fov_half_angle_rad=math.radians(config.vlc.receiver_fov_deg),
        root_seed=root_seed,
    )


__all__ = [
    "CONTENTION_RADIUS_M",
    "NOMINAL_BLOCKER",
    "Perception",
    "build_perception",
]
