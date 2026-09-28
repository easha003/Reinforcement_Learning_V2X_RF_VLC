"""RF block error by the finite-blocklength approximation.

The last group of tests pins what the assembled budget *implies*: that this
profile is not thermal-noise limited anywhere in the tagged-pair window, and
that LOS RF failure is therefore collision-dominated by roughly one to two
orders of magnitude. That is a structural result about the contribution, not
a property of the formula, so it is asserted here where it will break loudly
if a parameter moves.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.bler import (
    DEFAULT_NOISE_FIGURE_DB,
    MIN_BLOCKLENGTH,
    BLERError,
    LinkBudget,
    block_error_probability,
    channel_dispersion,
    gaussian_tail,
    required_snr_db,
    shannon_capacity,
    thermal_noise_dbm,
)
from hybrid_v2x_rl.channels.rf.collision import (
    SensitivityBand,
    failure_probability,
    headline_parameters,
)
from hybrid_v2x_rl.channels.rf.pathloss_37885 import median_path_loss_db
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.config.models import BITS_PER_RESOURCE_ELEMENT
from hybrid_v2x_rl.core.enums import RFPropagationState

BLOCKLENGTH = 2419
INFORMATION_BITS = 2784


@pytest.fixture(scope="module")
def rf():
    return load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd()).rf


# -- primitives ---------------------------------------------------------------


def test_gaussian_tail_matches_known_values() -> None:
    assert gaussian_tail(0.0) == pytest.approx(0.5)
    assert gaussian_tail(1.0) == pytest.approx(0.158655, abs=1e-6)
    assert gaussian_tail(1.959964) == pytest.approx(0.025, abs=1e-6)
    assert gaussian_tail(-1.0) == pytest.approx(1.0 - 0.158655, abs=1e-6)


def test_noise_floor_is_minus_ninety_five_dbm_over_ten_megahertz() -> None:
    """-174 dBm/Hz + 70 dB + 9 dB noise figure."""

    assert thermal_noise_dbm(10e6, DEFAULT_NOISE_FIGURE_DB) == pytest.approx(-95.0, abs=0.05)


def test_capacity_and_dispersion_have_the_right_limits() -> None:
    assert shannon_capacity(0.0) == 0.0
    assert shannon_capacity(1.0) == pytest.approx(1.0)
    # Dispersion vanishes at zero SNR and saturates at (log2 e)^2.
    assert channel_dispersion(0.0) == pytest.approx(0.0)
    assert channel_dispersion(1e9) == pytest.approx(math.log2(math.e) ** 2, rel=1e-6)


# -- the approximation --------------------------------------------------------


def test_bler_falls_monotonically_with_snr() -> None:
    previous = 1.0
    for snr_db in (-6, -4, -3, -2, 0, 5):
        value = block_error_probability(10 ** (snr_db / 10), BLOCKLENGTH, INFORMATION_BITS)
        assert value <= previous
        previous = value


def test_bler_rises_when_more_bits_are_pushed_through_the_same_block() -> None:
    snr = 10 ** (2.0 / 10.0)
    light = block_error_probability(snr, BLOCKLENGTH, INFORMATION_BITS)
    heavy = block_error_probability(snr, BLOCKLENGTH, 2 * INFORMATION_BITS)
    assert heavy > light


def test_required_snr_inverts_the_forward_direction() -> None:
    for target in (1e-1, 1e-3, 1e-5):
        snr_db = required_snr_db(target, BLOCKLENGTH, INFORMATION_BITS)
        achieved = block_error_probability(
            10 ** (snr_db / 10), BLOCKLENGTH, INFORMATION_BITS
        )
        assert achieved == pytest.approx(target, rel=0.05)


def test_the_waterfall_is_steep_at_this_blocklength() -> None:
    """Four decades of BLER inside half a decibel.

    A consequence of 2,419 channel uses: the normal approximation's transition
    sharpens as sqrt(n), so the link is essentially binary in SNR. That is why
    the margin table below is the interesting object and the BLER curve is not.
    """

    loose = required_snr_db(1e-1, BLOCKLENGTH, INFORMATION_BITS)
    tight = required_snr_db(1e-5, BLOCKLENGTH, INFORMATION_BITS)
    assert tight - loose < 0.5


def test_a_short_block_is_refused_rather_than_extrapolated() -> None:
    with pytest.raises(BLERError, match="validity range"):
        block_error_probability(1.0, MIN_BLOCKLENGTH - 1, 100)


def test_zero_snr_loses_every_block() -> None:
    assert block_error_probability(0.0, BLOCKLENGTH, INFORMATION_BITS) > 0.999


# -- the assembled budget -----------------------------------------------------


def test_the_configured_grid_gives_the_expected_blocklength_and_rate(rf) -> None:
    uses = rf.available_coded_bits() / BITS_PER_RESOURCE_ELEMENT[rf.modulation]
    bits = (300 + rf.timing.framing_overhead_bytes) * 8
    assert uses == pytest.approx(BLOCKLENGTH, abs=1)
    assert bits == INFORMATION_BITS
    assert bits / uses == pytest.approx(1.151, abs=0.005)


def snr_db_at(rf, distance_m: float, state: RFPropagationState, blockage_db: float = 0.0) -> float:
    loss = median_path_loss_db(distance_m, rf.carrier_hz, state) + blockage_db
    return LinkBudget(
        tx_power_dbm=rf.tx_power_dbm,
        path_loss_db=loss,
        shadowing_db=0.0,
        fading_power_gain=1.0,
        noise_dbm=thermal_noise_dbm(rf.bandwidth_hz),
    ).snr_db


def test_the_link_is_not_thermal_noise_limited_anywhere_in_the_window(rf) -> None:
    """The finding that reorders M3's priorities.

    1e-5 needs 1.46 dB. The median budget delivers 53.5 dB at 5 m and 31.8 dB
    at 100 m, so there is 30 to 52 dB of margin over what the code requires.
    Even a building-blocked link at 100 m clears it. Thermal noise is simply
    not the mechanism, and a reliability story told through the link budget
    would be telling the wrong story.
    """

    need = required_snr_db(1e-5, BLOCKLENGTH, INFORMATION_BITS)
    assert need == pytest.approx(1.46, abs=0.1)

    for distance, expected in ((5.0, 53.5), (100.0, 31.8)):
        snr = snr_db_at(rf, distance, RFPropagationState.LOS)
        assert snr == pytest.approx(expected, abs=0.2)
        assert snr - need > 30.0

    # Even the worst class at the far edge of the window still closes.
    assert snr_db_at(rf, 100.0, RFPropagationState.NLOS) - need > 5.0


def rayleigh_outage(margin_db: float) -> float:
    """P(a Rayleigh power gain fails to cover ``margin_db``)."""

    return 1.0 - math.exp(-(10 ** (-margin_db / 10.0)))


@pytest.mark.parametrize(
    ("state", "blockage_db", "distance_m", "expected"),
    [
        (RFPropagationState.LOS, 0.0, 100.0, 9.29e-4),
        (RFPropagationState.NLOSV, 9.0, 100.0, 7.36e-3),
        (RFPropagationState.NLOS, 0.0, 100.0, 2.66e-1),
        (RFPropagationState.NLOS, 0.0, 50.0, 3.79e-2),
    ],
)
def test_deep_fade_outage_is_where_the_budget_actually_fails(
    rf, state: RFPropagationState, blockage_db: float, distance_m: float, expected: float
) -> None:
    """With this much margin, only the tail of the fading distribution matters.

    So the budget's contribution to failure is an outage probability, not a
    BLER curve -- and it is strongly ordered by propagation class, which is the
    coupling to optical blockage that section 8.3 measures.
    """

    need = required_snr_db(1e-5, BLOCKLENGTH, INFORMATION_BITS)
    margin = snr_db_at(rf, distance_m, state, blockage_db) - need
    assert rayleigh_outage(margin) == pytest.approx(expected, rel=0.15)


def test_rf_failure_is_collision_dominated_not_budget_dominated(rf) -> None:
    """The structural conclusion, and it favours the contribution.

    At 100 m in LOS the budget fails about 9.3e-4 of the time from deep fades.
    The analytical access model loses about 2% to 13% of packets over the same
    band. Access failure is at least twenty times larger at every density and
    every end of the declared band.

    That matters because collision is the RF failure mode *decoupled* from
    optical blockage, while the budget's failures are strongly ordered by
    propagation class and therefore correlated with it. The dominant RF
    mechanism is the one that makes the diversity argument non-circular.
    """

    need = required_snr_db(1e-5, BLOCKLENGTH, INFORMATION_BITS)
    budget = rayleigh_outage(snr_db_at(rf, 100.0, RFPropagationState.LOS) - need)

    for density, neighbours in ((10, 44), (20, 100), (30, 159)):
        for band in SensitivityBand:
            access = failure_probability(neighbours, headline_parameters(band))
            assert access > 20 * budget, (
                f"at rho={density} band={band.value} collision {access:.3f} "
                f"should dominate budget outage {budget:.2e}"
            )
