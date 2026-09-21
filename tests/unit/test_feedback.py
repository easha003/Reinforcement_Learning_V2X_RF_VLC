"""The feedback channel: what a receiver reports, and what it must not.

Two tests here are load-bearing for different reasons. One pins that a leg the
action did not spend produces no reading, because that asymmetry is the whole
sequential claim -- without it the problem is contextual and the discount
factor models structure that does not exist. The other pins that the exact
oracle SINR never crosses, because the observation's other inputs are all
carefully degraded and a clean channel here would undo that work through a
route no leakage guard is watching.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from test_packet_lifecycle import requests, tape  # noqa: F401

from hybrid_v2x_rl.channels.rf.collision import headline_parameters
from hybrid_v2x_rl.channels.rf.model import NRV2XChannel
from hybrid_v2x_rl.channels.vlc.headlamp_pattern import load_pattern
from hybrid_v2x_rl.channels.vlc.model import VVLCChannel
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.core.enums import FailureCause, Link
from hybrid_v2x_rl.env.feedback import (
    MEASUREMENT_ERROR_DB,
    REPORTING_STEP_DB,
    RF_QUALITY_SPAN_DB,
    measurements,
    reported_quality,
)
from hybrid_v2x_rl.env.packet import DUP, RF_ONLY, VLC_ONLY, PacketLifecycle, PacketOutcome, Timing

SEED = 4242
IDENTITY = {"root_seed": SEED, "trace_id": "t", "pair_id": "p", "packet_index": 7}


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
    return PacketLifecycle(
        rf=rf, vlc=vlc,
        timing=Timing(
            deadline_s=config.service.deadline_s,
            predecision_lead_s=config.service.predecision_lead_s,
            rf_airtime_s=config.rf.timing.airtime_s,
            vlc_airtime_s=config.vlc.timing.airtime_s,
            rf_attempts=config.service.rf_attempts_per_packet,
        ),
    )


def outcome(**overrides) -> PacketOutcome:
    fields = dict(
        action=RF_ONLY, delivered=True, delivery_time_s=1e-3,
        failure_cause=FailureCause.NONE, activation_cost=1.0, rf_attempts_used=1,
        rf_delivered=True, vlc_delivered=False,
        rf_failure_probability=1e-3, vlc_failure_probability=1e-2,
    )
    fields.update(overrides)
    return PacketOutcome(**fields)


# -- the sequential claim -----------------------------------------------------


def test_only_the_legs_the_action_spent_report_back(lifecycle) -> None:
    """A radio-only packet teaches nothing about the light, and vice versa.

    If this ever returned both legs, the unused leg's reading would refresh for
    free, ages would stop growing, and the action would no longer change what
    the policy knows next -- which is section 6.3's entire premise.
    """

    rf_request, vlc_request = requests()
    spent = {
        RF_ONLY: {Link.RF},
        VLC_ONLY: {Link.VLC},
        DUP: {Link.RF, Link.VLC},
    }
    for action, expected in spent.items():
        result = lifecycle.run(action=action, rf_request=rf_request,
                               vlc_request=vlc_request, tape=tape())
        assert set(measurements(result, **IDENTITY)) == expected, action.name


def test_an_unused_leg_carries_no_quality_at_all(lifecycle) -> None:
    """Not zero, not a floor -- absent. ``record`` refreshes the keys present,
    so a zero would read as a measured dead link rather than a stale one."""

    rf_request, vlc_request = requests()
    result = lifecycle.run(action=RF_ONLY, rf_request=rf_request,
                           vlc_request=vlc_request, tape=tape())
    assert result.vlc_quality_db is None
    assert Link.VLC not in measurements(result, **IDENTITY)


# -- the barrier --------------------------------------------------------------


def test_the_exact_oracle_sinr_never_crosses() -> None:
    """The reported value is degraded, and this checks it actually moved.

    Every other observation input is noised or aged on purpose. A quality
    channel that passed the channel model's own SINR through unchanged would
    hand the policy a cleaner view of the radio than of anything else, and no
    existing leakage guard would catch it -- they watch imports and geometry,
    not floats.

    Asserted over many packets rather than one, because a single coincidence
    is not a leak: an error under half a step rounds back to where it started,
    which is quantization behaving correctly. The failure being guarded is a
    reading that is *always* exact.
    """

    low, high = RF_QUALITY_SPAN_DB
    exact = 20.0
    normalized_exact = (exact - low) / (high - low)
    reports = [
        measurements(outcome(rf_quality_db=exact), root_seed=SEED, trace_id="t",
                     pair_id="p", packet_index=i)[Link.RF]
        for i in range(200)
    ]
    moved = sum(1 for r in reports if abs(r - normalized_exact) > 1e-9)
    assert moved > 0.2 * len(reports), (
        f"only {moved}/{len(reports)} readings moved off the exact value"
    )


def test_readings_are_quantized_to_the_reporting_step() -> None:
    """Feedback formats carry steps, not real numbers."""

    low, high = RF_QUALITY_SPAN_DB
    span = high - low
    for value in (3.3, 7.9, 12.1, 25.6):
        reported = measurements(outcome(rf_quality_db=value), **IDENTITY)[Link.RF]
        steps = reported * span / REPORTING_STEP_DB
        assert steps == pytest.approx(round(steps), abs=1e-9)


def test_the_error_is_not_negligible_beside_the_step() -> None:
    """A noise far below the quantization step would round away every time and
    leave the barrier decorative."""

    assert MEASUREMENT_ERROR_DB >= 0.5 * REPORTING_STEP_DB


# -- matched across actions ---------------------------------------------------


def test_the_same_packet_reports_the_same_error_to_every_action(lifecycle) -> None:
    """The matched-tape rule, extended to the observation.

    RF-only and DUP spend the same radio leg on the same packet. If their
    reported qualities differed, the difference between the two actions would
    carry a measurement artefact, and at a 1e-4 budget that is not separable
    from a real effect.
    """

    rf_request, vlc_request = requests()
    single = lifecycle.run(action=RF_ONLY, rf_request=rf_request,
                           vlc_request=vlc_request, tape=tape())
    both = lifecycle.run(action=DUP, rf_request=rf_request,
                         vlc_request=vlc_request, tape=tape())
    assert single.rf_quality_db == pytest.approx(both.rf_quality_db)
    assert (measurements(single, **IDENTITY)[Link.RF]
            == pytest.approx(measurements(both, **IDENTITY)[Link.RF]))


def test_different_packets_draw_different_errors() -> None:
    """Seeded per packet, so the error is reproducible but not a constant
    offset the policy could learn to subtract."""

    reports = {
        measurements(outcome(rf_quality_db=20.0), root_seed=SEED, trace_id="t",
                     pair_id="p", packet_index=i)[Link.RF]
        for i in range(40)
    }
    assert len(reports) > 1


def test_public_single_link_report_uses_the_same_identity_address() -> None:
    packet = outcome(rf_quality_db=20.0)

    assert reported_quality(
        20.0,
        link=Link.RF,
        **IDENTITY,
    ) == measurements(packet, **IDENTITY)[Link.RF]


def test_the_two_legs_draw_independent_errors() -> None:
    """One shared draw would correlate the legs' readings and quietly make a
    radio measurement partly informative about the light."""

    pairs = [
        measurements(outcome(action=DUP, rf_quality_db=20.0, vlc_quality_db=20.0),
                     root_seed=SEED, trace_id="t", pair_id="p", packet_index=i)
        for i in range(40)
    ]
    assert any(p[Link.RF] != p[Link.VLC] for p in pairs)


# -- the degenerate reading ---------------------------------------------------


def test_a_fully_blocked_optical_link_reports_the_bottom_of_the_scale() -> None:
    """Zero received power is ``-inf`` dB, which is a real reading and not an
    error. Unclipped it would poison the feature column for every packet the
    normalizer ever sees."""

    reported = measurements(
        outcome(action=DUP, rf_quality_db=10.0, vlc_quality_db=-math.inf), **IDENTITY
    )[Link.VLC]
    assert reported == pytest.approx(0.0)


@pytest.mark.parametrize("value", [-1e6, -50.0, 0.0, 20.0, 500.0, 1e6])
def test_every_reading_lands_inside_the_unit_interval(value) -> None:
    """The network sees these directly, so an out-of-range value is a training
    failure that shows up as a bad policy rather than as an error."""

    reported = measurements(outcome(rf_quality_db=value), **IDENTITY)[Link.RF]
    assert 0.0 <= reported <= 1.0
