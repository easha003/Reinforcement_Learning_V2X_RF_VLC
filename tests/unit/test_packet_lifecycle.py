"""The packet lifecycle: action in, delivered-or-missed out.

The load-bearing tests are the ones pinning that both legs always pay, that the
random tape is shared across actions, and that the marginal probabilities are
recorded on every packet regardless of what the action selected -- because that
last one is what stops section 8.3's dependence statistic being conditioned on
the policy it exists to check.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.collision import headline_parameters
from hybrid_v2x_rl.channels.rf.model import NRV2XChannel, RFChannelRequest, RFPacketRandomness
from hybrid_v2x_rl.channels.vlc.headlamp_pattern import load_pattern
from hybrid_v2x_rl.channels.vlc.model import VLCChannelRequest, VLCPacketRandomness, VVLCChannel
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.core.enums import FailureCause, RFPropagationState
from hybrid_v2x_rl.core.pair_geometry import pair_geometry
from hybrid_v2x_rl.env.packet import (
    ACTIONS,
    DUP,
    RF_ONLY,
    VLC_ONLY,
    Action,
    DependenceAccumulator,
    PacketError,
    PacketLifecycle,
    PacketTape,
    Timing,
)

CAR_LENGTH_M = 4.5
NORTH = 0.5 * math.pi


class FakeVehicle:
    def __init__(self, vid: str, x: float, y: float) -> None:
        self.vehicle_id, self.x_m, self.y_m, self.heading_rad = vid, x, y, NORTH
        self.length_m, self.width_m, self.height_m, self.speed_mps = CAR_LENGTH_M, 1.8, 1.5, 10.0


@pytest.fixture(scope="module")
def lifecycle() -> PacketLifecycle:
    config = load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd())
    rf = NRV2XChannel(
        carrier_hz=config.rf.carrier_hz,
        bandwidth_hz=config.rf.bandwidth_hz,
        tx_power_dbm=config.rf.tx_power_dbm,
        blocklength=int(config.rf.available_coded_bits() / 4),
        information_bits=(300 + config.rf.timing.framing_overhead_bytes) * 8,
        collision=headline_parameters(),
    )
    vlc = VVLCChannel(
        pattern=load_pattern(config.vlc.pattern_artifact),
        receiver=OpticalReceiver(),
        electrical_bandwidth_hz=config.vlc.electrical_bandwidth_hz,
        payload_bytes=300,
        framing_bytes=config.vlc.timing.framing_overhead_bytes,
        code_rate=config.vlc.timing.code_rate,
    )
    timing = Timing(
        deadline_s=config.service.deadline_s,
        predecision_lead_s=config.service.predecision_lead_s,
        rf_airtime_s=config.rf.timing.airtime_s,
        vlc_airtime_s=config.vlc.timing.airtime_s,
        rf_attempts=config.service.rf_attempts_per_packet,
    )
    return PacketLifecycle(rf=rf, vlc=vlc, timing=timing)


def requests(gap_m: float = 18.0, *, occluded: bool = False, neighbours: int = 100):
    follower, leader = FakeVehicle("tx", 0.0, 0.0), FakeVehicle("rx", 0.0, gap_m + CAR_LENGTH_M)
    geometry = pair_geometry(follower, leader)
    rf = RFChannelRequest(
        distance_m=geometry.separation_m,
        propagation_state=RFPropagationState.LOS,
        blockage_db=0.0,
        shadowing_normalized=0.0,
        fading_power_gain=1.0,
        neighbour_count=neighbours,
        sensed_fraction=1.0,
        randomness=RFPacketRandomness(0.5, 0.5, 0.5),
    )
    vlc = VLCChannelRequest(
        geometry=geometry, occluded=occluded,
        randomness=VLCPacketRandomness(decoding_draw=0.5),
    )
    return rf, vlc


def tape(rf_draws=((0.9, 0.9, 0.9),) * 3, vlc_draw: float = 0.9) -> PacketTape:
    return PacketTape(
        rf_attempts=tuple(RFPacketRandomness(*d) for d in rf_draws),
        vlc=VLCPacketRandomness(decoding_draw=vlc_draw),
    )


# -- the deadline -------------------------------------------------------------


def test_the_frozen_profile_fits_its_own_deadline(lifecycle) -> None:
    """Three 0.5 ms RF attempts and one 2.4 ms optical leg, concurrent."""

    timing = lifecycle.timing
    assert timing.rf_total_airtime_s == pytest.approx(1.5e-3)
    assert timing.vlc_arrival_s == pytest.approx(2.4e-3)
    assert max(timing.rf_total_airtime_s, timing.vlc_airtime_s) <= timing.available_s


def test_the_media_are_concurrent_not_sequential(lifecycle) -> None:
    """Different physical channels, so the binding quantity is the longer leg.

    If they were summed, 1.5 + 2.4 = 3.9 ms would exceed the deadline and DUP
    would be infeasible by arithmetic rather than by physics.
    """

    timing = lifecycle.timing
    assert timing.rf_total_airtime_s + timing.vlc_airtime_s > timing.available_s
    timing.check_feasible()


def test_an_overcommitted_profile_is_refused() -> None:
    greedy = Timing(deadline_s=0.003, predecision_lead_s=0.0001,
                    rf_airtime_s=0.001, vlc_airtime_s=0.0024, rf_attempts=6)
    with pytest.raises(PacketError, match="exceeds the deadline"):
        greedy.check_feasible()


# -- both legs always pay -----------------------------------------------------


def test_dup_pays_twice_even_when_the_radio_wins(lifecycle) -> None:
    """Section 9: no early cancellation, both legs incur their cost.

    This is what makes the constraint have something to trade against. If a
    winning RF leg refunded the optical activation, duplication would be nearly
    free and the policy would have no reason to ever choose a single medium.
    """

    rf, vlc = requests()
    outcome = lifecycle.run(action=DUP, rf_request=rf, vlc_request=vlc, tape=tape())
    assert outcome.delivered
    assert outcome.activation_cost == 2.0
    assert outcome.rf_delivered and outcome.vlc_delivered


def test_delivery_time_is_the_earlier_of_the_two_legs(lifecycle) -> None:
    """RF's first attempt lands at 0.5 ms, the optical leg at 2.4 ms."""

    rf, vlc = requests()
    outcome = lifecycle.run(action=DUP, rf_request=rf, vlc_request=vlc, tape=tape())
    assert outcome.delivery_time_s == pytest.approx(0.5e-3)


def test_the_optical_leg_alone_arrives_later(lifecycle) -> None:
    rf, vlc = requests()
    outcome = lifecycle.run(action=VLC_ONLY, rf_request=rf, vlc_request=vlc, tape=tape())
    assert outcome.delivered
    assert outcome.delivery_time_s == pytest.approx(2.4e-3)
    assert outcome.activation_cost == 1.0


# -- retransmission -----------------------------------------------------------

def test_the_radio_retries_and_the_optical_leg_does_not(lifecycle) -> None:
    """Asymmetric because the physics is: collision is redrawn every attempt,
    a blocked optical path is not."""

    rf, vlc = requests()
    # First two RF attempts collide, the third succeeds.
    unlucky = tape(rf_draws=((0.0, 0.9, 0.9), (0.0, 0.9, 0.9), (0.9, 0.9, 0.9)))
    outcome = lifecycle.run(action=RF_ONLY, rf_request=rf, vlc_request=vlc, tape=unlucky)
    assert outcome.delivered
    assert outcome.rf_attempts_used == 3
    assert outcome.delivery_time_s == pytest.approx(1.5e-3)


def test_a_packet_delivered_on_the_first_attempt_stops_there(lifecycle) -> None:
    rf, vlc = requests()
    outcome = lifecycle.run(action=RF_ONLY, rf_request=rf, vlc_request=vlc, tape=tape())
    assert outcome.rf_attempts_used == 1


# -- causes -------------------------------------------------------------------


def test_a_dup_packet_that_loses_both_legs_is_named_a_joint_failure(lifecycle) -> None:
    """Not attributed to whichever leg was tested last. Duplication exists to
    make this case rare, so it has to be countable."""

    rf, vlc = requests(occluded=True)
    doomed = tape(rf_draws=((0.0, 0.9, 0.9),) * 3, vlc_draw=0.0)
    outcome = lifecycle.run(action=DUP, rf_request=rf, vlc_request=vlc, tape=doomed)
    assert not outcome.delivered
    assert outcome.failure_cause is FailureCause.JOINT_FAILURE


def test_a_single_medium_failure_keeps_its_own_cause(lifecycle) -> None:
    rf, vlc = requests(occluded=True)
    outcome = lifecycle.run(action=VLC_ONLY, rf_request=rf, vlc_request=vlc, tape=tape())
    assert outcome.failure_cause is FailureCause.VLC_OCCLUSION


# -- matched tapes ------------------------------------------------------------


def test_every_action_is_evaluated_against_the_same_tape(lifecycle) -> None:
    """The design the oracle depends on.

    Fresh randomness per action would make the oracle's advantage partly a
    sampling artefact, and at a 1e-4 budget that is indistinguishable from a
    real effect however long you average.
    """

    rf, vlc = requests()
    shared = tape()
    outcomes = lifecycle.counterfactuals(rf_request=rf, vlc_request=vlc, tape=shared)
    assert set(outcomes) == {action.name for action in ACTIONS}
    # DUP's legs must agree with the single-medium runs on the same tape.
    assert outcomes["DUP"].rf_delivered == outcomes["RF"].rf_delivered
    assert outcomes["DUP"].vlc_delivered == outcomes["VLC"].vlc_delivered


def test_counterfactuals_are_reproducible(lifecycle) -> None:
    rf, vlc = requests()
    shared = tape()
    first = lifecycle.counterfactuals(rf_request=rf, vlc_request=vlc, tape=shared)
    second = lifecycle.counterfactuals(rf_request=rf, vlc_request=vlc, tape=shared)
    assert first == second


# -- section 8.3 --------------------------------------------------------------


def test_both_marginals_are_recorded_whatever_action_was_taken(lifecycle) -> None:
    """Otherwise the dependence statistic is conditioned on the policy.

    Section 8.3 exists to detect a diversity result that silently assumed
    independent failures. Computing the marginals only for the medium an action
    happened to select would make the statistic depend on the very choice it is
    supposed to audit.
    """

    rf, vlc = requests()
    for action in ACTIONS:
        outcome = lifecycle.run(action=action, rf_request=rf, vlc_request=vlc, tape=tape())
        assert outcome.rf_failure_probability > 0.0
        assert outcome.vlc_failure_probability > 0.0


def test_the_dependence_ratio_is_one_under_independence() -> None:
    """Constructed rather than simulated, so the estimator itself is checked."""

    accumulator = DependenceAccumulator()
    rng = np.random.default_rng(3)
    p_rf, p_vlc = 0.2, 0.3
    for _ in range(40000):
        rf_fail = rng.random() < p_rf
        vlc_fail = rng.random() < p_vlc
        accumulator.packets += 1
        accumulator.expected_rf += p_rf
        accumulator.expected_vlc += p_vlc
        accumulator.expected_joint += p_rf * p_vlc
        accumulator.joint_failures += rf_fail and vlc_fail
    assert accumulator.dependence_ratio == pytest.approx(1.0, abs=0.05)


def test_the_dependence_ratio_detects_media_failing_together() -> None:
    accumulator = DependenceAccumulator()
    rng = np.random.default_rng(5)
    p_rf, p_vlc = 0.2, 0.3
    for _ in range(40000):
        # Perfectly coupled: a shared cause fails both.
        shared = rng.random()
        accumulator.packets += 1
        accumulator.expected_rf += p_rf
        accumulator.expected_vlc += p_vlc
        accumulator.expected_joint += p_rf * p_vlc
        accumulator.joint_failures += shared < min(p_rf, p_vlc)
    assert accumulator.dependence_ratio > 2.5


def test_an_empty_accumulator_reports_independence_rather_than_dividing_by_zero() -> None:
    assert DependenceAccumulator().dependence_ratio == 1.0


# -- actions ------------------------------------------------------------------


def test_the_action_set_matches_the_frozen_cost_model() -> None:
    assert RF_ONLY.activation_cost == 1.0
    assert VLC_ONLY.activation_cost == 1.0
    assert DUP.activation_cost == 2.0


def test_an_action_using_no_medium_is_refused() -> None:
    with pytest.raises(PacketError, match="at least one medium"):
        Action("NONE", uses_rf=False, uses_vlc=False, activation_cost=0.0)
