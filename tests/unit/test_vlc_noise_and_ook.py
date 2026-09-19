"""Optical noise and OOK error, and what regime the receiver is actually in.

The load-bearing tests here are the ones pinning *which noise term dominates*
and *how steeply the link falls with range*, because those two decide which
design levers work and why the optical link behaves so differently from the
radio.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.vlc.noise import (
    BOLTZMANN_J_PER_K,
    CLEAR_DAY,
    CLEAR_NIGHT,
    ELEMENTARY_CHARGE_C,
    AmbientCondition,
    NoiseError,
    ambient_condition,
    electrical_snr,
    noise_power,
    shot_noise_a2,
    thermal_noise_a2,
)
from hybrid_v2x_rl.channels.vlc.ook import (
    OOKError,
    bit_error_rate,
    coded_bits,
    coded_packet_error,
    evaluate,
    uncoded_packet_error,
)
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config

BANDWIDTH_HZ = 5.0e6
#: Received power at the three measured mean gaps, from the feasibility spike.
MEASURED_POWER_W = {10: 1.34e-7, 20: 1.78e-6, 30: 2.52e-6}


@pytest.fixture(scope="module")
def vlc():
    return load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd()).vlc


# -- conditions are configurations, not a boolean -----------------------------


def test_ambient_conditions_are_named_and_carry_their_own_source() -> None:
    """Spec 12.4 forbids a runtime boolean with undocumented constants."""

    for condition in (CLEAR_NIGHT, CLEAR_DAY):
        assert condition.name
        assert condition.source
        assert condition.photocurrent_a > 0.0
    assert CLEAR_DAY.photocurrent_a > CLEAR_NIGHT.photocurrent_a


def test_the_headline_profile_selects_night(vlc) -> None:
    assert vlc.ambient_condition == "clear_night"
    assert ambient_condition(vlc.ambient_condition) is CLEAR_NIGHT


def test_an_unknown_condition_is_refused() -> None:
    with pytest.raises(NoiseError, match="unknown ambient"):
        ambient_condition("twilight")


def test_a_negative_ambient_photocurrent_is_refused() -> None:
    with pytest.raises(NoiseError, match="non-negative"):
        AmbientCondition(name="bad", photocurrent_a=-1.0, source="x")


# -- the regime, which decides which levers work ------------------------------


@pytest.mark.parametrize("density", [10, 20, 30])
def test_the_receiver_is_thermal_limited_across_the_whole_window(density: int) -> None:
    """Measured, and it is why an APD is the lever and a bigger PIN is not.

    Internal gain multiplies the signal *before* the thermal term, so it helps
    a thermal-limited receiver directly. Detector area buys signal and junction
    capacitance together, and the bandwidth constraint then forces the load
    resistance down, so thermal noise rises with it.
    """

    noise = noise_power(
        receiver=OpticalReceiver(),
        received_optical_power_w=MEASURED_POWER_W[density],
        bandwidth_hz=BANDWIDTH_HZ,
    )
    assert noise.is_thermal_limited
    shot_total = noise.signal_shot_a2 + noise.ambient_shot_a2
    assert noise.thermal_a2 > 100.0 * shot_total, (
        f"thermal {noise.thermal_a2:.2e} should dominate shot {shot_total:.2e} "
        "by orders of magnitude, not marginally"
    )


def test_the_night_ambient_barely_matters_which_is_why_it_may_stay_unsourced() -> None:
    """The tolerance for an unsourced constant, and the condition on it.

    Night ambient sits far below the thermal term, so the headline result is
    insensitive to it. That is the only reason the value is acceptable
    unsourced -- and the tolerance ends if an avalanche detector ever makes the
    shot term competitive.
    """

    receiver = OpticalReceiver()
    quiet = electrical_snr(receiver=receiver, received_optical_power_w=MEASURED_POWER_W[20],
                           bandwidth_hz=BANDWIDTH_HZ, ambient=CLEAR_NIGHT)
    tenfold = electrical_snr(
        receiver=receiver, received_optical_power_w=MEASURED_POWER_W[20],
        bandwidth_hz=BANDWIDTH_HZ,
        ambient=AmbientCondition("ten_times_night", 10 * CLEAR_NIGHT.photocurrent_a, "x"),
    )
    assert 10.0 * math.log10(quiet / tenfold) < 0.1, "ten times the ambient should barely move it"


def test_daylight_changes_which_physics_limits_the_receiver() -> None:
    """Not merely worse -- a different limiting term, which is the point of
    spec 12.4 insisting the two conditions are separate configurations.

    Shot noise overtakes thermal at 308 uA. Full sun on an unfiltered 7.5 mm^2
    detector passes 3 mA, ten times that. A first attempt at the constant used
    100 uA, which does *not* flip the regime and would have quietly understated
    the daylight case while looking like it had modelled it.
    """

    receiver = OpticalReceiver()
    night = electrical_snr(receiver=receiver, received_optical_power_w=MEASURED_POWER_W[20],
                           bandwidth_hz=BANDWIDTH_HZ, ambient=CLEAR_NIGHT)
    day = electrical_snr(receiver=receiver, received_optical_power_w=MEASURED_POWER_W[20],
                         bandwidth_hz=BANDWIDTH_HZ, ambient=CLEAR_DAY)
    assert day < night
    day_noise = noise_power(receiver=receiver, received_optical_power_w=MEASURED_POWER_W[20],
                            bandwidth_hz=BANDWIDTH_HZ, ambient=CLEAR_DAY)
    assert not day_noise.is_thermal_limited, "daylight should flip the receiver to shot-limited"


def test_shot_and_thermal_use_the_textbook_forms() -> None:
    assert shot_noise_a2(1e-6, 1e6) == pytest.approx(2 * ELEMENTARY_CHARGE_C * 1e-6 * 1e6)
    receiver = OpticalReceiver()
    expected = (
        4 * BOLTZMANN_J_PER_K * receiver.noise_temperature_k
        * receiver.preamplifier_noise_factor * 1e6 / receiver.load_resistance_ohm
    )
    assert thermal_noise_a2(receiver, 1e6) == pytest.approx(expected)


def test_noise_scales_with_bandwidth() -> None:
    receiver = OpticalReceiver()
    narrow = noise_power(receiver=receiver, received_optical_power_w=1e-6, bandwidth_hz=1e6)
    wide = noise_power(receiver=receiver, received_optical_power_w=1e-6, bandwidth_hz=2e6)
    assert wide.total_a2 == pytest.approx(2.0 * narrow.total_a2)


# -- the square, and what it does to range ------------------------------------


def test_a_decibel_of_optical_loss_costs_two_decibels_of_snr() -> None:
    """IM/DD detects power, so the photocurrent is squared in the numerator.

    This is the single most consequential difference from the radio: the
    optical link falls as 1/d^4 in SNR where the radio falls as 1/d^2 in power.
    It is why V-VLC here is close to binary in range.
    """

    receiver = OpticalReceiver()
    strong = electrical_snr(receiver=receiver, received_optical_power_w=1e-6,
                            bandwidth_hz=BANDWIDTH_HZ)
    halved = electrical_snr(receiver=receiver, received_optical_power_w=0.5e-6,
                            bandwidth_hz=BANDWIDTH_HZ)
    assert 10.0 * math.log10(strong / halved) == pytest.approx(6.0, abs=0.15)


def test_doubling_the_range_costs_twelve_decibels() -> None:
    """1/d^2 in power, squared into SNR, is 1/d^4 -- 12 dB per doubling."""

    receiver = OpticalReceiver()
    near = electrical_snr(receiver=receiver, received_optical_power_w=1e-6,
                          bandwidth_hz=BANDWIDTH_HZ)
    far = electrical_snr(receiver=receiver, received_optical_power_w=1e-6 / 4.0,
                         bandwidth_hz=BANDWIDTH_HZ)
    assert 10.0 * math.log10(near / far) == pytest.approx(12.0, abs=0.3)


# -- OOK --------------------------------------------------------------------


def test_bit_error_is_a_half_at_zero_snr_and_falls_steeply() -> None:
    assert bit_error_rate(0.0) == pytest.approx(0.5, abs=1e-9)
    previous = 0.5
    for snr_db in (0, 5, 10, 13, 16):
        value = bit_error_rate(10 ** (snr_db / 10))
        assert value < previous
        previous = value
    # Q(sqrt(gamma)) at 13 dB: sqrt(19.95) = 4.467, Q(4.467) = 3.97e-6.
    assert bit_error_rate(10 ** 1.3) == pytest.approx(3.97e-6, rel=0.05)


def test_the_coded_block_matches_the_configured_rate_and_framing(vlc) -> None:
    block = coded_bits(300, vlc.timing.framing_overhead_bytes, vlc.timing.code_rate)
    assert block == 10624
    assert vlc.timing.code_rate == pytest.approx(0.25)


def test_coding_helps_and_the_uncoded_shortcut_is_the_pessimistic_bound() -> None:
    ber = 1e-4
    block = 10624
    assert coded_packet_error(ber, block) < uncoded_packet_error(ber, block)


def test_packet_error_falls_as_the_bit_error_does() -> None:
    block = 10624
    previous = 1.0
    for ber in (1e-2, 1e-3, 1e-4, 1e-5):
        value = coded_packet_error(ber, block)
        assert value <= previous
        previous = value


def test_the_correctable_fraction_is_a_declared_placeholder_and_moves_the_answer() -> None:
    """Stated as a sensitivity because the profile fixes a rate but no code.

    If this had no effect it would not need declaring; it has a large one,
    which is exactly why the value must be reported rather than buried.
    """

    ber, block = 3e-2, 10624
    weak = coded_packet_error(ber, block, correctable_fraction=0.02)
    strong = coded_packet_error(ber, block, correctable_fraction=0.08)
    assert weak > strong
    assert weak / max(strong, 1e-300) > 10.0


def test_evaluate_assembles_the_outcome(vlc) -> None:
    outcome = evaluate(
        10 ** 3.0,
        payload_bytes=300,
        framing_bytes=vlc.timing.framing_overhead_bytes,
        code_rate=vlc.timing.code_rate,
    )
    assert outcome.block_bits == 10624
    assert outcome.snr_db == pytest.approx(30.0)
    assert 0.0 <= outcome.packet_error_rate <= 1.0


def test_invalid_arguments_are_refused() -> None:
    with pytest.raises(OOKError, match="non-negative"):
        bit_error_rate(-1.0)
    with pytest.raises(OOKError, match="code rate"):
        coded_bits(300, 32, 0.0)
    with pytest.raises(OOKError, match="block length"):
        coded_packet_error(1e-3, 0)
