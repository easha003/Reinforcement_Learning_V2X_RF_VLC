"""The environment loop: a stored trace in, packet outcomes out.

The load-bearing tests here are the ones pinning that a packet's randomness
comes from its *identity* rather than from where it fell in an iteration, and
that every per-pair correlated state is independent of every other. Both exist
because the whole comparison -- baseline against oracle against a learned
policy -- is only a comparison if all three see the same packets.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.fading import frequency_correlation
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.geometry import OrientedRectangle, Point
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.env.packet import DUP, RF_ONLY, VLC_ONLY, PacketError, PacketTape
from hybrid_v2x_rl.env.rollout import (
    CONTENTION_RADIUS_M,
    PacketContext,
    RolloutSeedError,
    always,
    best_action,
)
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex

CAR_LENGTH_M = 4.5
NORTH = 0.5 * math.pi


class FakeVehicle:
    """Minimal stand-in carrying only what the geometry engine reads."""

    def __init__(
        self,
        vid: str,
        x: float,
        y: float,
        *,
        heading: float = NORTH,
        speed: float = 12.0,
        height: float = 1.5,
    ) -> None:
        self.vehicle_id, self.x_m, self.y_m, self.heading_rad = vid, x, y, heading
        self.length_m, self.width_m, self.height_m = CAR_LENGTH_M, 1.8, height
        self.speed_mps = speed


@pytest.fixture(scope="module")
def config():
    return load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd())


@pytest.fixture
def rollout(config):
    return build_rollout(config, buildings=(), root_seed=7)


def platoon(gap_m: float = 18.0, *, extra: int = 0, blocker_height: float | None = None):
    """A follower, a leader, and optionally a vehicle between them."""

    tx = FakeVehicle("tx", 0.0, 0.0)
    rx = FakeVehicle("rx", 0.0, gap_m + CAR_LENGTH_M)
    fleet = [tx, rx]
    if blocker_height is not None:
        fleet.append(
            FakeVehicle("mid", 0.0, 0.5 * (gap_m + CAR_LENGTH_M), height=blocker_height)
        )
    fleet += [FakeVehicle(f"n{i}", 3.5, 20.0 * (i + 1)) for i in range(extra)]
    return tx, rx, fleet


def evaluate(rollout, tx, rx, fleet, *, index: int = 0, time_s: float = 0.0,
             action=RF_ONLY, counterfactual: bool = False, density: float = 20.0):
    return rollout.evaluate_instant(
        trace_id="trace-a",
        pair_id="tx>rx",
        index=index,
        density=density,
        time_s=time_s,
        transmitter=tx,
        receiver=rx,
        neighbours=fleet,
        index_of_frame=SpatialIndex.build(fleet),
        choose=always(action),
        counterfactual=counterfactual,
    )


# -- randomness is keyed to the packet, not to the iteration ------------------


def test_the_tape_depends_only_on_packet_identity(rollout):
    """The property the whole matched-tape design rests on.

    If the tape came from a running stream, evaluating one extra action -- or
    evaluating pairs in a different order -- would shift every subsequent
    packet's channel, and an ablation would be two experiments rather than a
    comparison.
    """

    gains = (1.0, 1.0, 1.0)
    first = rollout._tape("trace-a", "tx>rx", 5, gains)
    rollout._tape("trace-a", "tx>rx", 6, gains)  # consume "later" packets
    rollout._tape("trace-b", "other", 5, gains)
    again = rollout._tape("trace-a", "tx>rx", 5, gains)
    assert first == again


def test_different_packets_get_different_tapes(rollout):
    gains = (1.0, 1.0, 1.0)
    tapes = {
        rollout._tape(trace, pair, index, gains).rf_attempts[0].collision_draw
        for trace in ("trace-a", "trace-b")
        for pair in ("tx>rx", "rx>tx")
        for index in (0, 1)
    }
    assert len(tapes) == 8


def test_a_tape_carries_one_draw_set_per_granted_attempt(rollout):
    tape = rollout._tape("trace-a", "tx>rx", 0, (1.0,) * rollout.lifecycle.timing.rf_attempts)
    assert len(tape.rf_attempts) == rollout.lifecycle.timing.rf_attempts
    assert len(tape.rf_fading_power_gains) == rollout.lifecycle.timing.rf_attempts


def test_correlated_channel_state_is_independent_of_pair_iteration_order(config):
    tx, rx, fleet = platoon()

    def run(order):
        rollout = build_rollout(config, buildings=(), root_seed=73)
        realized = {}
        for pair_id in order:
            outcome, _, _ = rollout.evaluate_instant(
                trace_id="trace-a",
                pair_id=pair_id,
                index=0,
                density=20.0,
                time_s=0.0,
                transmitter=tx,
                receiver=rx,
                neighbours=fleet,
                index_of_frame=SpatialIndex.build(fleet),
                choose=always(RF_ONLY),
            )
            realized[pair_id] = (
                rollout.shadowing._normalized[pair_id],
                tuple(rollout.fading._gains[pair_id]),
                outcome.delivered,
                outcome.rf_failure_probability,
            )
        return realized

    assert run(("pair-a", "pair-b")) == run(("pair-b", "pair-a"))


def test_trace_identity_changes_correlated_channel_streams(config):
    tx, rx, fleet = platoon()

    def initial_state(trace_id):
        rollout = build_rollout(config, buildings=(), root_seed=73)
        rollout.evaluate_instant(
            trace_id=trace_id,
            pair_id="pair-a",
            index=0,
            density=20.0,
            time_s=0.0,
            transmitter=tx,
            receiver=rx,
            neighbours=fleet,
            index_of_frame=SpatialIndex.build(fleet),
            choose=always(RF_ONLY),
        )
        return (
            rollout.shadowing._normalized["pair-a"],
            tuple(rollout.fading._gains["pair-a"]),
        )

    assert initial_state("trace-a") != initial_state("trace-b")


def test_trace_cannot_change_until_all_correlated_state_is_released(config):
    tx, rx, fleet = platoon()
    rollout = build_rollout(config, buildings=(), root_seed=73)
    common = {
        "index": 0,
        "density": 20.0,
        "time_s": 0.0,
        "transmitter": tx,
        "receiver": rx,
        "neighbours": fleet,
        "index_of_frame": SpatialIndex.build(fleet),
        "choose": always(RF_ONLY),
    }
    rollout.evaluate_instant(trace_id="trace-a", pair_id="pair-a", **common)

    with pytest.raises(RolloutSeedError, match="correlated pair state is live"):
        rollout.evaluate_instant(trace_id="trace-b", pair_id="pair-b", **common)

    rollout.release("pair-a")
    rollout.evaluate_instant(trace_id="trace-b", pair_id="pair-b", **common)


# -- the correlated per-pair states -------------------------------------------


def test_blockage_residual_is_not_the_shadowing_draw(rollout):
    """Independent draws, or a deeply shadowed link is necessarily a deeply
    blocked one -- a correlation manufactured by bookkeeping, in exactly the
    statistic section 8.3 exists to measure."""

    tx, rx, fleet = platoon(blocker_height=3.5)
    _, _, _ = evaluate(rollout, tx, rx, fleet)
    shadow = rollout.shadowing._normalized["tx>rx"]
    blockage = rollout.shadowing._normalized["tx>rx|blockage"]
    assert shadow != blockage


def test_a_tall_blocker_makes_the_pair_nlosv_and_costs_it_decibels(rollout):
    tx, rx, fleet = platoon(blocker_height=3.5)
    _, context, _ = evaluate(rollout, tx, rx, fleet)
    assert context.propagation_state is RFPropagationState.NLOSV


def test_shadowing_persists_across_packets_of_one_pair(rollout):
    """Over 3 ms a vehicle travels centimetres against a 10 m decorrelation
    length, so consecutive packets must be almost the same draw. If this ever
    fails, shadowing has become a fast fade and retransmission looks free."""

    tx, rx, fleet = platoon()
    evaluate(rollout, tx, rx, fleet, index=0, time_s=0.0)
    first = rollout.shadowing._normalized["tx>rx"]
    evaluate(rollout, tx, rx, fleet, index=1, time_s=0.003)
    second = rollout.shadowing._normalized["tx>rx"]
    assert abs(second - first) < 0.15 * max(1.0, abs(first))


def test_release_drops_every_per_pair_state(rollout):
    """All of them together: one surviving is worse than none being dropped,
    because the stale one would be silently correlated with a fresh one."""

    tx, rx, fleet = platoon(blocker_height=3.5)
    evaluate(rollout, tx, rx, fleet)
    assert rollout.shadowing.live_links() > 0
    rollout.release("tx>rx")
    assert rollout.shadowing.live_links() == 0
    assert rollout.fading.live_links() == 0
    assert not rollout._last_time_s


# -- full-carrier allocation ---------------------------------------------------


def test_headline_attempts_repeat_on_the_same_full_carrier(rollout, config):
    offsets = rollout._hop_offsets_hz()
    assert len(offsets) == config.service.rf_attempts_per_packet
    assert offsets == (0.0,) * config.service.rf_attempts_per_packet


def test_two_carrier_pool_hops_between_ten_megahertz_centres(config):
    multi_carrier = build_rollout(
        config,
        buildings=(),
        root_seed=7,
        collision_subchannels=2,
    )
    offsets = multi_carrier._hop_offsets_hz()
    assert offsets == (-5e6, 5e6, -5e6)
    adjacent = frequency_correlation(abs(offsets[1] - offsets[0]), 200e-9)
    assert adjacent < 0.1


def test_a_single_attempt_profile_does_not_hop(rollout):
    rollout.lifecycle.timing = type(rollout.lifecycle.timing)(
        deadline_s=rollout.lifecycle.timing.deadline_s,
        predecision_lead_s=rollout.lifecycle.timing.predecision_lead_s,
        rf_airtime_s=rollout.lifecycle.timing.rf_airtime_s,
        vlc_airtime_s=rollout.lifecycle.timing.vlc_airtime_s,
        rf_attempts=1,
    )
    assert rollout._hop_offsets_hz() == (0.0,)


def test_fading_gains_reach_the_attempts(rollout):
    """The 10 MHz baseline repeats one full-carrier fading realization."""

    tx, rx, fleet = platoon()
    _, _, _ = evaluate(rollout, tx, rx, fleet)
    gains = rollout.fading.advance(
        "probe", elapsed_s=0.0, tx_speed_mps=12.0, rx_speed_mps=12.0,
        state=RFPropagationState.LOS,
    )
    assert len(gains) == rollout.lifecycle.timing.rf_attempts
    assert gains == pytest.approx(
        [float(gains[0])] * rollout.lifecycle.timing.rf_attempts,
        abs=1e-5,
    )


def test_mismatched_fading_gains_are_refused():
    from hybrid_v2x_rl.channels.rf.model import RFPacketRandomness
    from hybrid_v2x_rl.channels.vlc.model import VLCPacketRandomness

    with pytest.raises(PacketError, match="every RF attempt"):
        PacketTape(
            rf_attempts=(RFPacketRandomness(0.5, 0.5, 0.5),) * 3,
            vlc=VLCPacketRandomness(0.5),
            rf_fading_power_gains=(1.0, 1.0),
        )


# -- contention ---------------------------------------------------------------


def test_contenders_exclude_the_transmitter_and_respect_the_radius(rollout):
    tx, rx, fleet = platoon(extra=4)
    far = FakeVehicle("far", 0.0, CONTENTION_RADIUS_M + 50.0)
    fleet = [*fleet, far]
    _, context, _ = evaluate(rollout, tx, rx, fleet)
    assert context.neighbour_count == len(fleet) - 2, "self excluded, far one excluded"


# -- the field of view is configuration, not a default ------------------------


def test_the_acceptance_cone_comes_from_the_profile(config):
    """A pair off the boresight is in view under a wide cone and out under a
    narrow one. If the rollout silently took pair_geometry's default, the
    30 deg exploratory profile would be measured at 60 deg."""

    tx = FakeVehicle("tx", 0.0, 0.0)
    rx = FakeVehicle("rx", 12.0, 12.0, heading=0.0)
    fleet = [tx, rx]

    def in_view(half_angle_deg: float) -> bool:
        r = build_rollout(config, buildings=(), root_seed=1)
        r.fov_half_angle_rad = math.radians(half_angle_deg)
        _, context, _ = r.evaluate_instant(
            trace_id="t", pair_id="p", index=0, density=10.0, time_s=0.0,
            transmitter=tx, receiver=rx, neighbours=fleet,
            index_of_frame=SpatialIndex.build(fleet), choose=always(VLC_ONLY),
        )
        return context.within_field_of_view

    assert in_view(80.0)
    assert not in_view(10.0)


def test_the_configured_cone_is_the_one_that_gets_used(config, rollout):
    assert rollout.fov_half_angle_rad == pytest.approx(
        math.radians(config.vlc.receiver_fov_deg)
    )


# -- counterfactuals and the oracle -------------------------------------------


def test_counterfactuals_cover_every_action_on_one_packet(rollout):
    tx, rx, fleet = platoon()
    _, _, alternatives = evaluate(rollout, tx, rx, fleet, counterfactual=True)
    assert set(alternatives) == {"RF", "VLC", "DUP"}


def test_the_taken_action_matches_its_own_counterfactual(rollout):
    """The same packet, the same tape: choosing an action must not perturb it."""

    tx, rx, fleet = platoon()
    outcome, _, alternatives = evaluate(
        rollout, tx, rx, fleet, action=DUP, counterfactual=True
    )
    assert outcome.delivered == alternatives["DUP"].delivered


def test_a_counterfactual_run_does_not_change_the_taken_outcome(rollout, config):
    """Order independence again, at the level a caller sees it."""

    tx, rx, fleet = platoon()
    plain = build_rollout(config, buildings=(), root_seed=7)
    both = build_rollout(config, buildings=(), root_seed=7)
    a, _, _ = evaluate(plain, tx, rx, fleet, action=RF_ONLY)
    b, _, _ = evaluate(both, tx, rx, fleet, action=RF_ONLY, counterfactual=True)
    assert a.delivered == b.delivered
    assert a.rf_failure_probability == pytest.approx(b.rf_failure_probability)


def test_the_oracle_takes_the_cheapest_action_that_delivers(rollout):
    tx, rx, fleet = platoon()
    _, _, alternatives = evaluate(rollout, tx, rx, fleet, counterfactual=True)
    chosen = best_action(alternatives)
    if any(o.delivered for o in alternatives.values()):
        assert chosen.activation_cost == 1.0, "duplication is never the oracle's choice \
when one leg suffices"


def test_the_oracle_duplicates_only_when_neither_single_leg_works():
    """Constructed rather than sampled, because the case is meant to be rare."""

    from hybrid_v2x_rl.core.enums import FailureCause
    from hybrid_v2x_rl.env.packet import PacketOutcome

    def outcome(action, delivered):
        return PacketOutcome(
            action=action, delivered=delivered, delivery_time_s=0.001 if delivered else None,
            failure_cause=FailureCause.NONE if delivered else FailureCause.RF_COLLISION,
            activation_cost=action.activation_cost, rf_attempts_used=1,
            rf_delivered=delivered, vlc_delivered=False,
            rf_failure_probability=0.1, vlc_failure_probability=0.1,
        )

    only_dup = {
        "RF": outcome(RF_ONLY, False),
        "VLC": outcome(VLC_ONLY, False),
        "DUP": outcome(DUP, True),
    }
    assert best_action(only_dup) is DUP

    nothing = {name: outcome(o.action, False) for name, o in only_dup.items()}
    assert best_action(nothing).activation_cost == 1.0, \
        "a miss should not also be expensive"


# -- what travels with the outcome --------------------------------------------


def test_the_context_carries_what_section_8_3_conditions_on(rollout):
    tx, rx, fleet = platoon(extra=2)
    _, context, _ = evaluate(rollout, tx, rx, fleet, density=30.0)
    assert isinstance(context, PacketContext)
    assert context.density == 30.0
    assert context.separation_m > 0.0
    assert context.optical_path_m < context.separation_m, \
        "the optical path is headlamp to photodiode, shorter by the receiver's length"


def test_an_occluded_optical_path_is_reported_and_costs_the_vlc_leg(rollout):
    """Occlusion is the geometry engine's decision, and it must reach the leg."""

    tx, rx, fleet = platoon(gap_m=30.0, blocker_height=3.5)
    outcome, context, _ = evaluate(rollout, tx, rx, fleet, action=VLC_ONLY)
    assert context.occluded
    assert not outcome.vlc_delivered
    assert outcome.vlc_failure_probability == 1.0


def test_a_building_makes_the_pair_nlos(config):
    """Buildings are held by the rollout, not passed per packet, because the
    layout does not change within a trace."""

    tx = FakeVehicle("tx", 0.0, 0.0)
    rx = FakeVehicle("rx", 40.0, 40.0, heading=0.0)
    wall = OrientedRectangle(
        centre=Point(20.0, 20.0), heading_rad=0.0, length_m=20.0, width_m=20.0
    )
    r = build_rollout(config, buildings=(wall,), root_seed=3)
    fleet = [tx, rx]
    _, context, _ = r.evaluate_instant(
        trace_id="t", pair_id="p", index=0, density=10.0, time_s=0.0,
        transmitter=tx, receiver=rx, neighbours=fleet,
        index_of_frame=SpatialIndex.build(fleet), choose=always(RF_ONLY),
    )
    assert context.propagation_state is RFPropagationState.NLOS


# -- the marginal probabilities are packet-level ------------------------------


def test_the_rf_marginal_covers_every_attempt_not_just_one(rollout):
    """Section 8.3 pairs p_RF with a packet-level p_VLC, so a single-attempt
    p_RF would overstate the denominator by roughly the diversity order and
    make the ratio read far below one for purely clerical reasons."""

    tx, rx, fleet = platoon(extra=30)
    outcome, _, _ = evaluate(rollout, tx, rx, fleet, action=RF_ONLY)
    single = rollout.lifecycle.rf.failure_probability(
        __import__("hybrid_v2x_rl.channels.rf.model", fromlist=["RFChannelRequest"]).RFChannelRequest(
            distance_m=max(1.0, 18.0),
            propagation_state=RFPropagationState.LOS,
            blockage_db=0.0,
            shadowing_normalized=0.0,
            fading_power_gain=1.0,
            neighbour_count=30,
            sensed_fraction=1.0,
            randomness=rollout._tape("trace-a", "tx>rx", 0, (1.0,) * 3).rf_attempts[0],
        )
    )
    assert outcome.rf_failure_probability < single, \
        "three attempts must beat one, or the profile bought nothing"


def test_both_marginals_are_recorded_whatever_the_action_selected(rollout):
    tx, rx, fleet = platoon()
    for action in (RF_ONLY, VLC_ONLY, DUP):
        outcome, _, _ = evaluate(rollout, tx, rx, fleet, action=action)
        assert outcome.rf_failure_probability > 0.0
        assert 0.0 <= outcome.vlc_failure_probability <= 1.0
