"""The assembled V-VLC channel.

Pins the composition, and pins the ways it deliberately differs from the radio
façade -- because the environment is about to treat the two identically, and the
places where they must *not* be identical are where the contribution lives.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.vlc.headlamp_pattern import load_pattern
from hybrid_v2x_rl.channels.vlc.model import (
    VLCChannelError,
    VLCChannelRequest,
    VLCPacketRandomness,
    VVLCChannel,
)
from hybrid_v2x_rl.channels.vlc.noise import CLEAR_DAY, CLEAR_NIGHT
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.core.enums import FailureCause
from hybrid_v2x_rl.core.pair_geometry import pair_geometry

CAR_LENGTH_M = 4.5
NORTH = 0.5 * math.pi


class FakeVehicle:
    """Minimal stand-in carrying only what pair_geometry reads."""

    def __init__(self, vehicle_id: str, x_m: float, y_m: float, heading_rad: float) -> None:
        self.vehicle_id = vehicle_id
        self.x_m = x_m
        self.y_m = y_m
        self.heading_rad = heading_rad
        self.length_m = CAR_LENGTH_M
        self.width_m = 1.8
        self.height_m = 1.5
        self.speed_mps = 10.0


@pytest.fixture(scope="module")
def channel() -> VVLCChannel:
    config = load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd())
    return VVLCChannel(
        pattern=load_pattern(config.vlc.pattern_artifact),
        receiver=OpticalReceiver(),
        electrical_bandwidth_hz=config.vlc.electrical_bandwidth_hz,
        payload_bytes=300,
        framing_bytes=config.vlc.timing.framing_overhead_bytes,
        code_rate=config.vlc.timing.code_rate,
    )


def request_at(gap_m: float, *, occluded: bool = False, draw: float = 0.5) -> VLCChannelRequest:
    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 0.0, gap_m + CAR_LENGTH_M, NORTH)
    return VLCChannelRequest(
        geometry=pair_geometry(follower, leader),
        occluded=occluded,
        randomness=VLCPacketRandomness(decoding_draw=draw),
    )


# -- the tape -----------------------------------------------------------------


def test_a_draw_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(VLCChannelError, match="uniform draw"):
        VLCPacketRandomness(decoding_draw=1.5)


def test_one_draw_not_three_because_geometry_carries_no_randomness() -> None:
    """The radio takes three draws; the optical link has one stochastic step.

    Occlusion and field of view are decided by the geometry engine and passed
    in. If this module ever grew its own blockage draw, the class would stop
    being predictable from tracked positions and the observation forecast would
    have nothing left to forecast.
    """

    assert VLCPacketRandomness.__dataclass_fields__.keys() == {"decoding_draw"}


def test_the_same_tape_gives_the_same_outcome(channel) -> None:
    assert channel.evaluate(request_at(20.0)) == channel.evaluate(request_at(20.0))


def test_failure_probability_does_not_consume_the_draw(channel) -> None:
    lucky = channel.failure_probability(request_at(20.0, draw=0.999))
    unlucky = channel.failure_probability(request_at(20.0, draw=0.0))
    assert lucky == pytest.approx(unlucky)


# -- geometry short-circuits --------------------------------------------------


def test_an_occluded_path_delivers_nothing_and_names_occlusion(channel) -> None:
    """No power fixes a blocked path -- that is the whole asymmetry with RF."""

    result = channel.evaluate(request_at(20.0, occluded=True, draw=0.999))
    assert not result.success
    assert result.failure_cause is FailureCause.VLC_OCCLUSION
    assert result.total_failure_probability == 1.0
    assert result.is_geometric_failure


def test_occlusion_is_decided_by_the_caller_not_the_channel(channel) -> None:
    """Same geometry, opposite outcome, because the geometry engine owns it."""

    clear = channel.evaluate(request_at(20.0, occluded=False, draw=0.999))
    blocked = channel.evaluate(request_at(20.0, occluded=True, draw=0.999))
    assert clear.success
    assert not blocked.success


def test_a_blocked_path_never_samples_the_beam(channel) -> None:
    """Not an optimization: a severed path delivers the blockage model's power
    however brightly the lamp was pointing, and sampling first would invite an
    edit that lets a bright beam leak through a bus."""

    result = channel.evaluate(request_at(20.0, occluded=True))
    assert result.received_power_w == channel.blockage.occluded_power_w
    assert result.electrical_snr == 0.0


# -- the budget ---------------------------------------------------------------


def test_the_link_closes_across_the_whole_pair_window(channel) -> None:
    """The feasibility result, pinned where a profile change would break it.

    Before the photodiode was moved to 0.4 m this failed at the two longer
    gaps: the link was sampling an ECE low beam along its cut-off.
    """

    for gap_m in (10.9, 18.1, 30.5):
        result = channel.evaluate(request_at(gap_m, draw=0.5))
        assert result.success, f"the optical link should close at a {gap_m} m gap"
        assert result.decoding_failure_probability < 1e-6


def test_snr_falls_steeply_with_the_gap(channel) -> None:
    """1/d^4 in electrical SNR: 12 dB per doubling, which is why the link is
    close to binary in range where the radio is not."""

    near = channel.evaluate(request_at(12.0)).snr_db
    far = channel.evaluate(request_at(24.0)).snr_db
    assert near - far > 10.0


def test_short_gaps_look_below_the_test_point_grid_and_still_resolve(channel) -> None:
    """The boundary the façade found, and why the artifact was extended.

    A 0.3 m height difference makes the link look 1.72 deg down at a 10 m gap
    and 3.43 deg at 5 m, while the ECE test-point grid stops at 1.718 deg. Every
    gap under 10 m was therefore outside the tabulated envelope -- and rho=30's
    mean gap is 10.9 m, so the trained band sits right on it.

    Rows below the last test point carry the zone IV minimum, a regulatory floor
    rather than a measured shape. It is conservative: 6.25 W/sr against the
    40.2 W/sr the 50V hot spot delivers, so a short-gap pair is credited with
    less light than a compliant lamp must actually emit there.
    """

    for gap_m in (2.0, 5.0, 8.0, 10.0):
        result = channel.evaluate(request_at(gap_m, draw=0.5))
        assert result.success, f"a {gap_m} m gap must resolve, not raise"


def test_daylight_degrades_the_link(channel) -> None:
    day = VVLCChannel(
        pattern=channel.pattern,
        receiver=channel.receiver,
        electrical_bandwidth_hz=channel.electrical_bandwidth_hz,
        payload_bytes=channel.payload_bytes,
        framing_bytes=channel.framing_bytes,
        code_rate=channel.code_rate,
        ambient=CLEAR_DAY,
    )
    assert day.evaluate(request_at(30.5)).electrical_snr < \
        channel.evaluate(request_at(30.5)).electrical_snr
    assert channel.ambient is CLEAR_NIGHT


def test_a_decoding_failure_is_named_as_a_channel_failure(channel) -> None:
    """Reached only when geometry allowed the packet through."""

    result = channel.evaluate(request_at(200.0, draw=0.0))
    assert not result.success
    assert result.failure_cause is FailureCause.VLC_CHANNEL
    assert not result.is_geometric_failure


def test_a_delivered_packet_records_no_cause(channel) -> None:
    result = channel.evaluate(request_at(15.0, draw=0.999))
    assert result.success
    assert result.failure_cause is FailureCause.NONE


# -- the shape the environment depends on -------------------------------------


def test_both_channel_facades_present_the_same_shape() -> None:
    """M5 treats the two media identically, so the contracts must match.

    Not a style preference: the packet lifecycle is required to contain no
    channel equations, which is only possible if both façades answer the same
    two questions -- evaluate this attempt, and tell me its probability without
    consuming the tape.
    """

    from hybrid_v2x_rl.channels.rf.model import NRV2XChannel, RFChannelResult

    for facade in (VVLCChannel, NRV2XChannel):
        assert hasattr(facade, "evaluate")
        assert hasattr(facade, "failure_probability")

    from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult

    shared = {"success", "failure_cause", "total_failure_probability"}
    assert shared <= set(VLCChannelResult.__dataclass_fields__)
    assert shared <= set(RFChannelResult.__dataclass_fields__)


def test_the_optical_link_has_no_contention_mechanism() -> None:
    """The structural asymmetry, asserted rather than described.

    The radio reports a collision probability separate from its decoding
    failure, because contention dominates it by orders of magnitude. A
    directional point-to-point optical link has no equivalent: every optical
    failure is geometric or budgetary. Neither medium's dominant failure is the
    other's, which is what makes the pair worth having -- and if this file ever
    grows a collision field, that claim needs re-examining.
    """

    from hybrid_v2x_rl.channels.rf.model import RFChannelResult
    from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult

    assert "collision_probability" in RFChannelResult.__dataclass_fields__
    assert not any(
        "collision" in name for name in VLCChannelResult.__dataclass_fields__
    )
