"""Small-scale fading, correlated in time and in frequency.

The properties pinned here decide whether the paper's retransmission argument
holds: how much time diversity the 3 ms deadline actually contains, and how
much frequency diversity a hop between full 10 MHz carriers actually buys.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.fading import (
    LOS_RICIAN_K_DB,
    URBAN_RMS_DELAY_SPREAD_S,
    FadingError,
    FadingProcess,
    bessel_j0,
    coherence_bandwidth_hz,
    coherence_time_s,
    doppler_spread_hz,
    frequency_correlation,
    rician_k_linear,
    temporal_correlation,
    wavelength_m,
)
from hybrid_v2x_rl.core.enums import RFPropagationState

CARRIER_HZ = 5.9e9
#: Adjacent full 10 MHz carrier-allocation centres in a wider system pool.
SUBCHANNELS_HZ = (0.0, 10e6)


def fading(seed: int = 7, subchannels: tuple[float, ...] = SUBCHANNELS_HZ) -> FadingProcess:
    return FadingProcess(
        rng=np.random.default_rng(seed),
        carrier_hz=CARRIER_HZ,
        subchannel_separations_hz=subchannels,
    )


# -- the Bessel approximation -------------------------------------------------


def test_bessel_j0_matches_known_values() -> None:
    assert bessel_j0(0.0) == pytest.approx(1.0, abs=1e-8)
    assert bessel_j0(1.0) == pytest.approx(0.7651976866, abs=1e-7)
    assert bessel_j0(2.4048255577) == pytest.approx(0.0, abs=1e-7)  # first zero
    assert bessel_j0(5.0) == pytest.approx(-0.1775967713, abs=1e-7)
    assert bessel_j0(10.0) == pytest.approx(-0.2459357644, abs=1e-7)


def test_bessel_j0_is_even() -> None:
    for x in (0.5, 3.0, 7.5):
        assert bessel_j0(-x) == pytest.approx(bessel_j0(x), abs=1e-12)


def test_bessel_j0_is_continuous_across_the_branch_at_three() -> None:
    """The approximation switches form at |x| = 3; a step there would alias.

    Compared against the true value at the switch rather than across it: the
    difference between samples either side is dominated by the function's own
    slope, ``|J0'(3)| = J1(3) = 0.3391``, so a naive equality test would fail on
    honest curvature rather than on a seam.
    """

    assert bessel_j0(3.0) == pytest.approx(-0.2600519549, abs=1e-7)

    below, above = bessel_j0(2.9999), bessel_j0(3.0001)
    slope_over_gap = 0.3391 * 2e-4
    assert abs(above - below) == pytest.approx(slope_over_gap, rel=0.05)


# -- Doppler and coherence time -----------------------------------------------


def test_wavelength_at_the_carrier() -> None:
    assert wavelength_m(CARRIER_HZ) == pytest.approx(0.0508, abs=1e-4)


def test_doppler_uses_the_sum_of_speeds_not_the_difference() -> None:
    """The choice that decides whether time diversity exists at all.

    A tagged pair travels the same way, so the relative speed is near zero.
    Using it would freeze the fading and make retransmission look useless. The
    fading is driven by each terminal moving through largely static scatter, so
    the spread scales with the sum of ground speeds.
    """

    same_speed = doppler_spread_hz(11.18, 11.18, CARRIER_HZ)
    assert same_speed > 0.0
    assert same_speed == pytest.approx(2 * 11.18 / 0.0508, rel=0.01)

    # If it used the difference, an equal-speed pair would have zero Doppler.
    assert same_speed != pytest.approx(0.0)


def test_coherence_time_at_the_speed_limit_is_about_one_millisecond() -> None:
    """The number the system model quotes in prose, now measured."""

    doppler = doppler_spread_hz(11.18, 11.18, CARRIER_HZ)
    assert coherence_time_s(doppler) == pytest.approx(0.96e-3, abs=0.1e-3)


@pytest.mark.parametrize(
    ("speed_mps", "density", "expected"),
    [(6.30, 10, 0.479), (4.08, 20, 0.761), (2.77, 30, 0.886)],
)
def test_time_diversity_gets_worse_as_density_rises(
    speed_mps: float, density: int, expected: float
) -> None:
    """Measured, and it points the wrong way.

    A retransmission 1 ms later sees a channel correlated at 0.48 / 0.76 / 0.89
    at rho = 10 / 20 / 30. Slower traffic means a smaller Doppler spread, a
    longer coherence time, and *less* time diversity -- so the mechanism decays
    exactly as density rises, which is where the reliability constraint binds
    hardest. Repeating on the same subchannel is worth least when it is needed
    most.
    """

    doppler = doppler_spread_hz(speed_mps, speed_mps, CARRIER_HZ)
    assert temporal_correlation(1.0e-3, doppler) == pytest.approx(expected, abs=0.01)
    assert coherence_time_s(doppler) > 1.5e-3, f"rho={density} should be slow-fading"


def test_only_free_flow_decorrelates_inside_the_deadline() -> None:
    """At the speed limit a 1 ms gap crosses the first zero of J0.

    So genuine time diversity exists only in free traffic -- 0.96 ms coherence
    time, correlation -0.17 after 1 ms -- and that is the regime where the link
    budget already has margin. Taken with the test above, the conclusion is
    that the retransmission model must hop in frequency rather than repeat in
    time, because frequency diversity does not depend on how fast traffic is
    moving.
    """

    doppler = doppler_spread_hz(11.18, 11.18, CARRIER_HZ)
    assert coherence_time_s(doppler) == pytest.approx(0.96e-3, abs=0.05e-3)
    assert temporal_correlation(1.0e-3, doppler) == pytest.approx(-0.170, abs=0.02)

    congested = doppler_spread_hz(2.77, 2.77, CARRIER_HZ)
    assert temporal_correlation(1.0e-3, congested) > 0.8


def test_a_stationary_pair_has_no_time_variation() -> None:
    assert coherence_time_s(doppler_spread_hz(0.0, 0.0, CARRIER_HZ)) == math.inf


def test_temporal_correlation_starts_at_one_and_decays() -> None:
    doppler = doppler_spread_hz(11.18, 11.18, CARRIER_HZ)
    assert temporal_correlation(0.0, doppler) == pytest.approx(1.0, abs=1e-8)
    assert temporal_correlation(0.5e-3, doppler) < 1.0
    assert abs(temporal_correlation(5.0e-3, doppler)) < 0.5


def test_negative_speeds_and_times_are_refused() -> None:
    with pytest.raises(FadingError):
        doppler_spread_hz(-1.0, 5.0, CARRIER_HZ)
    with pytest.raises(FadingError):
        temporal_correlation(-1.0, 100.0)


# -- frequency correlation ----------------------------------------------------


def test_zero_separation_is_perfectly_correlated() -> None:
    assert frequency_correlation(0.0, URBAN_RMS_DELAY_SPREAD_S) == pytest.approx(1.0)


def test_frequency_correlation_decays_monotonically() -> None:
    previous = 1.0
    for separation in (0.5e6, 1e6, 2e6, 4e6, 8e6):
        value = frequency_correlation(separation, URBAN_RMS_DELAY_SPREAD_S)
        assert value < previous
        previous = value


def test_coherence_bandwidth_inverts_the_correlation() -> None:
    for level in (0.3, 0.5, 0.9):
        separation = coherence_bandwidth_hz(URBAN_RMS_DELAY_SPREAD_S, level)
        assert frequency_correlation(separation, URBAN_RMS_DELAY_SPREAD_S) == pytest.approx(
            level, abs=1e-9
        )


def test_two_full_carriers_are_far_enough_apart_to_be_worth_hopping() -> None:
    """A wider system can hop between adjacent 10 MHz carrier centres."""

    coherence = coherence_bandwidth_hz(URBAN_RMS_DELAY_SPREAD_S)
    assert coherence == pytest.approx(1.38e6, rel=0.05)

    hop = SUBCHANNELS_HZ[1] - SUBCHANNELS_HZ[0]
    assert hop / coherence > 7.0
    assert frequency_correlation(hop, URBAN_RMS_DELAY_SPREAD_S) < 0.1


# -- distribution by class ----------------------------------------------------


def test_los_is_rician_and_obstructed_classes_are_rayleigh() -> None:
    """A blocked direct path has no specular component, by definition.

    This is the same event that severs the optical link, so both media's
    statistics change at the same instant -- the coupling section 8.3 measures.
    """

    assert rician_k_linear(RFPropagationState.LOS) == pytest.approx(
        10 ** (LOS_RICIAN_K_DB / 10)
    )
    assert rician_k_linear(RFPropagationState.NLOSV) == 0.0
    assert rician_k_linear(RFPropagationState.NLOS) == 0.0


def test_rayleigh_power_is_exponential_with_unit_mean() -> None:
    process = fading()
    powers = np.array([
        process.advance(f"l{i}", elapsed_s=1.0, tx_speed_mps=10.0,
                        rx_speed_mps=10.0, state=RFPropagationState.NLOS)[0]
        for i in range(20000)
    ])
    assert powers.mean() == pytest.approx(1.0, abs=0.05)
    # Exponential: std equals mean, and the median is ln(2) of it.
    assert powers.std() == pytest.approx(1.0, abs=0.05)
    assert np.median(powers) == pytest.approx(math.log(2.0), abs=0.05)


def test_rician_fades_less_deeply_than_rayleigh() -> None:
    """A 9 dB K-factor should make deep fades much rarer."""

    def deep_fraction(state: RFPropagationState) -> float:
        process = fading()
        powers = np.array([
            process.advance(f"l{i}", elapsed_s=1.0, tx_speed_mps=10.0,
                            rx_speed_mps=10.0, state=state)[0]
            for i in range(20000)
        ])
        assert powers.mean() == pytest.approx(1.0, abs=0.06)
        return float((powers < 0.1).mean())   # 10 dB below mean

    assert deep_fraction(RFPropagationState.LOS) < 0.2 * deep_fraction(
        RFPropagationState.NLOS
    )


# -- the process's correlations ----------------------------------------------


def test_successive_samples_are_temporally_correlated_as_specified() -> None:
    step_s = 0.5e-3
    doppler = doppler_spread_hz(11.18, 11.18, CARRIER_HZ)
    expected = temporal_correlation(step_s, doppler) ** 2   # power, not amplitude

    process = fading()
    powers = np.array([
        process.advance("pair", elapsed_s=step_s, tx_speed_mps=11.18,
                        rx_speed_mps=11.18, state=RFPropagationState.NLOS)[0]
        for _ in range(40000)
    ])
    measured = float(np.corrcoef(powers[:-1], powers[1:])[0, 1])
    assert measured == pytest.approx(expected, abs=0.06)


def test_subchannels_are_correlated_at_the_specified_strength() -> None:
    """The property a hopped retransmission depends on."""

    process = fading()
    powers = np.array([
        process.advance(f"l{i}", elapsed_s=1.0, tx_speed_mps=10.0,
                        rx_speed_mps=10.0, state=RFPropagationState.NLOS)
        for i in range(20000)
    ])
    measured = float(np.corrcoef(powers[:, 0], powers[:, 1])[0, 1])
    amplitude = frequency_correlation(
        SUBCHANNELS_HZ[1] - SUBCHANNELS_HZ[0], URBAN_RMS_DELAY_SPREAD_S
    )
    assert measured == pytest.approx(amplitude**2, abs=0.05)
    assert measured < 0.1, "a hop must deliver near-independent fading"


def test_a_single_subchannel_configuration_is_allowed() -> None:
    process = fading(subchannels=(0.0,))
    assert process.subchannel_count == 1
    gains = process.advance("pair", elapsed_s=1e-3, tx_speed_mps=5.0,
                            rx_speed_mps=5.0, state=RFPropagationState.LOS)
    assert gains.shape == (1,)


def test_an_empty_subchannel_list_is_refused() -> None:
    with pytest.raises(FadingError, match="at least one subchannel"):
        FadingProcess(rng=np.random.default_rng(0), carrier_hz=CARRIER_HZ,
                      subchannel_separations_hz=())


def test_forgetting_a_link_releases_its_state() -> None:
    process = fading()
    process.advance("pair", elapsed_s=1e-3, tx_speed_mps=5.0,
                    rx_speed_mps=5.0, state=RFPropagationState.LOS)
    assert process.live_links() == 1
    process.forget("pair")
    assert process.live_links() == 0
