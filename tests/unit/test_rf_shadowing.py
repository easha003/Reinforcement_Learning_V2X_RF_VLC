"""Spatially correlated shadowing.

The properties pinned here are the ones that make shadowing *shadowing* rather
than a slow fade: it persists over metres of travel, it has the right
stationary spread, and a class change does not discontinuously reset it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.pathloss_37885 import shadowing_sigma_db
from hybrid_v2x_rl.channels.rf.shadowing import (
    LOS_DECORRELATION_M,
    ShadowingError,
    ShadowingProcess,
    correlation,
    decorrelation_distance_m,
    shadowing_db,
)
from hybrid_v2x_rl.core.enums import RFPropagationState


def process(seed: int = 7) -> ShadowingProcess:
    return ShadowingProcess(rng=np.random.default_rng(seed))


# -- the correlation function -------------------------------------------------


def test_zero_displacement_is_perfectly_correlated() -> None:
    assert correlation(0.0, LOS_DECORRELATION_M) == pytest.approx(1.0)


def test_one_decorrelation_length_falls_to_one_over_e() -> None:
    assert correlation(LOS_DECORRELATION_M, LOS_DECORRELATION_M) == pytest.approx(1 / math.e)


def test_correlation_decays_monotonically() -> None:
    previous = 1.0
    for distance in (1.0, 5.0, 10.0, 25.0, 50.0):
        value = correlation(distance, LOS_DECORRELATION_M)
        assert value < previous
        previous = value


def test_nlos_decorrelates_over_a_longer_distance_than_los() -> None:
    assert decorrelation_distance_m(RFPropagationState.NLOS) > decorrelation_distance_m(
        RFPropagationState.LOS
    )


def test_a_negative_displacement_is_refused() -> None:
    with pytest.raises(ShadowingError, match="non-negative"):
        correlation(-1.0, LOS_DECORRELATION_M)


# -- the process --------------------------------------------------------------


def test_a_new_link_starts_from_the_stationary_distribution() -> None:
    """Not from zero.

    Starting at zero would make every newly formed tagged pair briefly and
    wrongly unshadowed, then converge -- a transient that correlates with pair
    formation, which is exactly when the policy is deciding.
    """

    shadow = process()
    firsts = [
        shadow.advance(f"link-{index}", 0.0, RFPropagationState.LOS)
        for index in range(4000)
    ]
    assert np.mean(firsts) == pytest.approx(0.0, abs=0.05)
    assert np.std(firsts) == pytest.approx(1.0, abs=0.05)
    assert any(abs(value) > 0.5 for value in firsts[:20]), "should not start near zero"


def test_the_state_is_stationary_at_unit_variance() -> None:
    """The AR recursion must not drift or shrink over a long trajectory."""

    shadow = process()
    values = [shadow.advance("pair", 2.0, RFPropagationState.LOS) for _ in range(20000)]
    tail = values[1000:]
    assert np.mean(tail) == pytest.approx(0.0, abs=0.05)
    assert np.std(tail) == pytest.approx(1.0, abs=0.05)


def test_successive_samples_are_correlated_at_the_expected_strength() -> None:
    """The whole point: a shadowed link stays shadowed for tens of packets."""

    step_m = 2.0
    expected = correlation(step_m, LOS_DECORRELATION_M)

    shadow = process()
    values = np.array(
        [shadow.advance("pair", step_m, RFPropagationState.LOS) for _ in range(40000)]
    )
    measured = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    assert measured == pytest.approx(expected, abs=0.02)
    assert measured > 0.75, "2 m of travel must not decorrelate the link"


def test_a_long_jump_decorrelates() -> None:
    shadow = process()
    step_m = 40.0 * LOS_DECORRELATION_M
    values = np.array(
        [shadow.advance("pair", step_m, RFPropagationState.LOS) for _ in range(4000)]
    )
    measured = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    assert measured == pytest.approx(0.0, abs=0.05)


def test_links_are_independent_of_one_another() -> None:
    shadow = process()
    first, second = [], []
    for _ in range(6000):
        first.append(shadow.advance("a", 2.0, RFPropagationState.LOS))
        second.append(shadow.advance("b", 2.0, RFPropagationState.LOS))
    assert float(np.corrcoef(first, second)[0, 1]) == pytest.approx(0.0, abs=0.05)


def test_forgetting_a_link_releases_its_state() -> None:
    shadow = process()
    shadow.advance("pair", 0.0, RFPropagationState.LOS)
    assert shadow.live_links() == 1
    shadow.forget("pair")
    assert shadow.live_links() == 0


# -- the class interaction ----------------------------------------------------


def test_a_class_change_does_not_reset_the_shadowing_itself() -> None:
    """A van pulling in front does not move the link to a new neighbourhood.

    State is unit-variance and sigma is applied at read time, so crossing from
    LOS to NLOSv rescales the shadowing without discontinuity in the underlying
    process. Storing sigma-scaled state would jump the value at every class
    transition and manufacture a correlation between the class sequence and the
    shadowing sequence -- which is the dependence section 8.3 must *measure*.
    """

    shadow = process()
    shadow.advance("pair", 0.0, RFPropagationState.LOS)
    before = shadow.advance("pair", 0.01, RFPropagationState.LOS)
    after = shadow.advance("pair", 0.01, RFPropagationState.NLOSV)

    # Over 1 cm of travel the state is essentially unchanged despite the class
    # flipping underneath it.
    assert after == pytest.approx(before, abs=0.15)


def test_decibels_scale_with_the_class_spread() -> None:
    normalized = 1.5
    los = shadowing_db(
        normalized, RFPropagationState.LOS, shadowing_sigma_db(RFPropagationState.LOS)
    )
    nlos = shadowing_db(
        normalized, RFPropagationState.NLOS, shadowing_sigma_db(RFPropagationState.NLOS)
    )
    assert los == pytest.approx(1.5 * 3.0)
    assert nlos == pytest.approx(1.5 * 4.0)
    assert abs(nlos) > abs(los), "NLOS shadowing must be the wider process"


def test_a_negative_sigma_is_refused() -> None:
    with pytest.raises(ShadowingError, match="sigma"):
        shadowing_db(1.0, RFPropagationState.LOS, -1.0)


def test_shadowing_is_reproducible_from_a_seed() -> None:
    """Bit-exact reproduction is what lets a trace be regenerated, not stored."""

    first = [process(11).advance("pair", 2.0, RFPropagationState.LOS) for _ in range(1)]
    second = [process(11).advance("pair", 2.0, RFPropagationState.LOS) for _ in range(1)]
    assert first == second
