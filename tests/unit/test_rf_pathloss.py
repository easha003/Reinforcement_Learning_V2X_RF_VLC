"""TR 37.885 urban path loss, given a propagation class.

These pin the properties the rest of M3 depends on, not just the arithmetic of
the published formulas. The load-bearing ones are that NLOSv shares the LOS
median and adds a term, and that nothing in this module is improved by
frequency hopping -- the paper's equal-cost DUP-versus-RF×2 ablation rests on
that separation.
"""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.channels.rf.pathloss_37885 import (
    LOS_SHADOWING_SIGMA_DB,
    MIN_VALID_DISTANCE_M,
    NLOS_SHADOWING_SIGMA_DB,
    PathLossError,
    blockage_mean_db,
    large_scale_loss,
    median_path_loss_db,
    shadowing_sigma_db,
)
from hybrid_v2x_rl.core.enums import RFPropagationState

CARRIER_HZ = 5.9e9
ANTENNA_M = 1.5


# -- median path loss ---------------------------------------------------------


def test_los_matches_the_published_urban_form() -> None:
    """PL = 38.77 + 16.7 log10(d) + 18.2 log10(fc), d in m, fc in GHz."""

    expected = 38.77 + 16.7 * math.log10(100.0) + 18.2 * math.log10(5.9)
    assert median_path_loss_db(100.0, CARRIER_HZ, RFPropagationState.LOS) == pytest.approx(expected)


def test_nlos_matches_the_published_urban_form() -> None:
    """PL = 36.85 + 30 log10(d) + 18.9 log10(fc). VERIFIED against Table 6.2.1-1.

    This test previously encoded a cyclic permutation of the three
    coefficients -- intercept 18.9, distance 36.85, frequency 30.0 -- and passed,
    because it restated the implementation rather than the document. It is the
    reason a test that only echoes the code is worth very little: it locked the
    error in place instead of catching it.

    The physical tell was available all along. An NLOS intercept of 18.9 dB
    against a LOS intercept of 38.77 dB is a 20 dB gap between two quantities
    that are both "loss at one metre", and a distance exponent of 3.685 with
    that intercept is not a shape any measurement campaign produces.
    """

    expected = 36.85 + 30.0 * math.log10(100.0) + 18.9 * math.log10(5.9)
    assert median_path_loss_db(100.0, CARRIER_HZ, RFPropagationState.NLOS) == pytest.approx(expected)


def test_the_two_intercepts_sit_in_the_same_range() -> None:
    """A cheap invariant that would have caught the permutation.

    Both intercepts are loss at one metre, so they cannot differ by 20 dB. This
    is not a restatement of the formula -- it is a statement about what kind of
    quantity an intercept is, and it fails for any permutation of the three
    coefficients.
    """

    at_one_metre_los = median_path_loss_db(1.0, CARRIER_HZ, RFPropagationState.LOS)
    at_one_metre_nlos = median_path_loss_db(1.0, CARRIER_HZ, RFPropagationState.NLOS)
    assert abs(at_one_metre_nlos - at_one_metre_los) < 10.0


def test_nlosv_shares_the_los_median() -> None:
    """The defining property: blockage is an additive term, not an exponent.

    This is what makes TR 37.885 price NLOSv as extra loss the radio usually
    survives, while the same obstruction delivers nothing at all on the optical
    link. If NLOSv ever grew its own distance exponent, the asymmetry the
    contribution rests on would be silently rewritten.
    """

    for distance in (5.0, 20.0, 60.0, 100.0):
        assert median_path_loss_db(
            distance, CARRIER_HZ, RFPropagationState.NLOSV
        ) == median_path_loss_db(distance, CARRIER_HZ, RFPropagationState.LOS)


def test_nlos_is_always_worse_than_los_across_the_pair_window() -> None:
    for distance in (5.0, 20.0, 50.0, 100.0):
        assert median_path_loss_db(
            distance, CARRIER_HZ, RFPropagationState.NLOS
        ) > median_path_loss_db(distance, CARRIER_HZ, RFPropagationState.LOS)


def test_loss_grows_with_distance_and_with_carrier() -> None:
    near = median_path_loss_db(10.0, CARRIER_HZ, RFPropagationState.LOS)
    far = median_path_loss_db(100.0, CARRIER_HZ, RFPropagationState.LOS)
    assert far > near

    low = median_path_loss_db(50.0, 2.0e9, RFPropagationState.LOS)
    high = median_path_loss_db(50.0, 5.9e9, RFPropagationState.LOS)
    assert high > low


def test_the_pair_window_spans_the_range_the_optics_care_about() -> None:
    """5 to 100 m is 20x in distance; the radio pays 16.7 log10(20) = 21.7 dB.

    Recorded because the window was widened to 100 m precisely so the far end
    is a regime where V-VLC degrades as 1/d^2 and the radio keeps margin. The
    numbers have to be in the same file as the claim.
    """

    near = median_path_loss_db(5.0, CARRIER_HZ, RFPropagationState.LOS)
    far = median_path_loss_db(100.0, CARRIER_HZ, RFPropagationState.LOS)
    assert far - near == pytest.approx(16.7 * math.log10(20.0), abs=1e-9)
    assert far - near == pytest.approx(21.7, abs=0.05)


# -- validity -----------------------------------------------------------------


def test_a_distance_below_the_validity_range_is_refused() -> None:
    """Extrapolating a log-distance fit below its range is silent nonsense."""

    with pytest.raises(PathLossError, match="validity range"):
        median_path_loss_db(0.5 * MIN_VALID_DISTANCE_M, CARRIER_HZ, RFPropagationState.LOS)


def test_a_nonfinite_distance_is_refused() -> None:
    with pytest.raises(PathLossError):
        median_path_loss_db(math.nan, CARRIER_HZ, RFPropagationState.LOS)


def test_a_nonpositive_carrier_is_refused() -> None:
    with pytest.raises(PathLossError, match="carrier"):
        median_path_loss_db(50.0, 0.0, RFPropagationState.LOS)


# -- vehicle blockage ---------------------------------------------------------


def test_a_blocker_below_both_antennas_costs_nothing() -> None:
    assert blockage_mean_db(50.0, 1.0, ANTENNA_M, ANTENNA_M) == 0.0


def test_a_taller_blocker_costs_more_than_a_shorter_one() -> None:
    between = blockage_mean_db(50.0, 1.8, 1.5, 2.5)
    above = blockage_mean_db(50.0, 3.25, 1.5, 2.5)
    assert above > between > 0.0


def test_the_distance_term_is_inert_across_the_whole_pair_window() -> None:
    """Measured, and it changes what NLOSv means here.

    The growth term ``max(0, 15 log10(d) - 41)`` only turns positive beyond
    **541 m**, far outside the 5-100 m tagged-pair window and outside any V2V
    range that matters. Within the window, vehicle blockage is therefore a
    **flat offset with no distance dependence at all**.

    That is a modelling consequence worth asserting rather than discovering
    later: it means NLOSv cannot be predicted from separation, only from
    whether a blocker is present. Any feature that tries to infer blockage
    severity from pair distance is inferring nothing.
    """

    flat = blockage_mean_db(5.0, 3.25, ANTENNA_M, ANTENNA_M)
    for distance in (5.0, 20.0, 50.0, 100.0, 200.0, 500.0):
        assert blockage_mean_db(distance, 3.25, ANTENNA_M, ANTENNA_M) == pytest.approx(flat)

    turning_point = 10.0 ** (41.0 / 15.0)
    assert turning_point == pytest.approx(541.2, abs=0.5)
    assert blockage_mean_db(1000.0, 3.25, ANTENNA_M, ANTENNA_M) > flat


def test_equal_antenna_heights_make_the_middle_branch_unreachable() -> None:
    """With Tx and Rx both at 1.5 m there is no "between" case to fall into.

    ``lower == upper``, so a blocker is either at or below antenna height (no
    loss) or above it (the tall mean). The 5 dB partial-obstruction branch is
    dead code in this configuration. It would come alive only if the two
    antennas sat at different heights, which the frozen profile does not do.
    """

    below = blockage_mean_db(50.0, 1.4, ANTENNA_M, ANTENNA_M)
    above = blockage_mean_db(50.0, 1.6, ANTENNA_M, ANTENNA_M)
    assert below == 0.0
    assert above == pytest.approx(9.0)

    # Unequal heights do reach the middle branch, so the code is not wrong --
    # only unexercised by this profile.
    middle = blockage_mean_db(50.0, 1.8, 1.5, 2.5)
    assert middle == pytest.approx(5.0)


def test_ninety_percent_of_the_fleet_blocks_the_optics_but_not_the_radio() -> None:
    """The sharpest consequence of the fleet meeting the antenna height.

    A passenger car is exactly 1.5 m, the antenna height, so it lands on the
    ``<=`` boundary and contributes **zero** RF blockage loss -- while the same
    body severs the 0.7 m optical path completely. Cars are 90% of the mixture.

    Read carefully this is the contribution's mechanism stated in decibels:
    the most common blocker is a total outage for V-VLC and free for the radio.
    Read sceptically it is a boundary case doing a great deal of work, since a
    body at exactly antenna height obstructs a real Fresnel zone substantially.
    Both readings are recorded in the module docstring; the number is pinned
    here so that a change to either the fleet or the antenna height surfaces
    it rather than absorbing it.
    """

    from hybrid_v2x_rl.mobility.vehicle_types import headline_vehicle_distribution

    types = {v.type_id: v for v in headline_vehicle_distribution().vehicle_types}
    assert types["passenger_car"].height_m == pytest.approx(ANTENNA_M)

    free = blockage_mean_db(50.0, types["passenger_car"].height_m, ANTENNA_M, ANTENNA_M)
    assert free == 0.0
    assert types["passenger_car"].share == pytest.approx(0.90)

    for heavy in ("van_suv", "bus_truck"):
        assert blockage_mean_db(
            50.0, types[heavy].height_m, ANTENNA_M, ANTENNA_M
        ) == pytest.approx(9.0)


# -- assembled budget ---------------------------------------------------------


def test_shadowing_spread_differs_by_class() -> None:
    assert shadowing_sigma_db(RFPropagationState.LOS) == LOS_SHADOWING_SIGMA_DB
    assert shadowing_sigma_db(RFPropagationState.NLOS) == NLOS_SHADOWING_SIGMA_DB
    # NLOSv is LOS propagation plus a blockage term, so it inherits LOS spread.
    assert shadowing_sigma_db(RFPropagationState.NLOSV) == LOS_SHADOWING_SIGMA_DB


def test_total_is_median_plus_blockage() -> None:
    loss = large_scale_loss(
        distance_m=50.0,
        carrier_hz=CARRIER_HZ,
        state=RFPropagationState.NLOSV,
        blockage_db=7.0,
    )
    assert loss.total_db == pytest.approx(loss.median_db + 7.0)


def test_blockage_outside_nlosv_is_refused() -> None:
    """A blockage term on a LOS link would be a class that decided itself."""

    with pytest.raises(PathLossError, match="NLOSv"):
        large_scale_loss(
            distance_m=50.0,
            carrier_hz=CARRIER_HZ,
            state=RFPropagationState.LOS,
            blockage_db=7.0,
        )


def test_nothing_here_is_helped_by_frequency_hopping() -> None:
    """The split the equal-cost ablation rests on.

    Everything in this module is flat across the 10 MHz carrier, so a hopped
    retransmission cannot recover any of it. Fading and collision are the
    mechanisms hopping does help, and they live elsewhere. If this ever returns
    true, DUP-versus-RF×2 at matched cost stops measuring cross-medium
    diversity and starts measuring coding gain.
    """

    for state in RFPropagationState:
        loss = large_scale_loss(
            distance_m=50.0, carrier_hz=CARRIER_HZ, state=state
        )
        assert loss.improves_with_frequency_hopping is False
