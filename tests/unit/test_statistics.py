"""Confidence bounds that respect the trajectory structure.

The load-bearing test in this file is the one showing the bootstrap widens when
packets are correlated and a binomial bound does not. That gap is the entire
reason this module exists: at a 1e-4 budget it decides whether the upper bound
clears the constraint, which is the claim the paper rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_v2x_rl.env.campaign import TargetingBound
from hybrid_v2x_rl.env.statistics import (
    MIN_CLUSTERS,
    ClusteredRate,
    StatisticsError,
    _normal_quantile,
    poisson_interval,
)


def fill(rate: ClusteredRate, clusters: int, per_cluster: int, misses_per_cluster=0):
    """Populate with a deterministic pattern rather than a sampled one."""

    for c in range(clusters):
        m = misses_per_cluster(c) if callable(misses_per_cluster) else misses_per_cluster
        for k in range(per_cluster):
            rate.observe(f"c{c}", missed=k < m, expected_failure=1e-4)
    return rate


# -- the primitives -----------------------------------------------------------


def test_the_normal_quantile_matches_published_values() -> None:
    assert _normal_quantile(0.95) == pytest.approx(1.6448536, abs=1e-6)
    assert _normal_quantile(0.975) == pytest.approx(1.9599640, abs=1e-6)
    assert _normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _normal_quantile(0.001) == pytest.approx(-3.0902323, abs=1e-5)


def test_a_quantile_outside_the_open_unit_interval_is_refused() -> None:
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(StatisticsError, match="quantile"):
            _normal_quantile(bad)


def test_poisson_intervals_match_garwood() -> None:
    """Published exact values, so an approximation drifting is caught."""

    for count, low, high in ((0, 0.0, 3.689), (6, 2.202, 13.059), (13, 6.922, 22.230)):
        got_low, got_high = poisson_interval(count)
        assert got_low == pytest.approx(low, abs=2e-3)
        assert got_high == pytest.approx(high, abs=2e-3)


def test_a_zero_count_has_no_lower_bound_but_a_real_upper_one() -> None:
    """Seeing nothing is not evidence of nothing."""

    low, high = poisson_interval(0)
    assert low == 0.0
    assert high > 3.0


# -- the bound under independence ---------------------------------------------


def test_the_bootstrap_agrees_with_wilson_when_packets_are_independent() -> None:
    """One packet per cluster is the independent case, and the two estimators
    must not disagree there -- if they do, the bootstrap is measuring something
    other than sampling variation."""

    rate = ClusteredRate()
    rng = np.random.default_rng(0)
    for i in range(20_000):
        rate.observe(f"c{i}", missed=bool(rng.random() < 1e-3), expected_failure=1e-3)
    boot = rate.bootstrap_upper(replicates=2_000, seed=1)
    assert boot == pytest.approx(rate.wilson_upper(), rel=0.25)


# -- the bound under correlation, which is the point --------------------------


def test_the_bootstrap_widens_as_failures_become_bursty() -> None:
    """Same packet count, same miss count, increasing concentration within
    episodes. A binomial bound cannot see this; the cluster bootstrap must."""

    ratios = []
    for burst in (1, 10, 50):
        rate = ClusteredRate()
        clusters, per_cluster = 400, 50
        bad = 20_000 // (burst * 100)  # hold total misses roughly constant
        fill(rate, clusters, per_cluster, lambda c, b=burst, n=bad: b if c < n else 0)
        ratios.append(
            rate.bootstrap_upper(replicates=2_000, seed=2) / rate.wilson_upper()
        )
    assert ratios[0] < ratios[-1], "burstier failures must widen the interval"
    assert ratios[-1] > 1.2, "a strongly clustered failure pattern must inflate the bound"


def test_a_binomial_bound_understates_a_clustered_rate() -> None:
    """Stated as the reason the module exists, and asserted so it stays true."""

    rate = ClusteredRate()
    fill(rate, 300, 100, lambda c: 100 if c < 3 else 0)  # 3 episodes fail entirely
    assert rate.bootstrap_upper(replicates=2_000, seed=3) > rate.wilson_upper()


# -- estimator shape ----------------------------------------------------------


def test_the_rate_is_a_ratio_of_sums_not_a_mean_of_cluster_rates() -> None:
    """Episodes differ in length by an order of magnitude. Averaging their
    rates would weight a five-packet episode like a five-hundred-packet one."""

    rate = ClusteredRate()
    for c in range(100):
        fill_long = 1000
        for _k in range(fill_long):
            rate.observe(f"long{c}", missed=False, expected_failure=0.0)
        for k in range(10):
            rate.observe(f"short{c}", missed=k < 5, expected_failure=0.0)

    assert rate.realized_rate == pytest.approx(500 / 101_000, rel=1e-9)
    mean_of_rates = 0.5 * (0.0 + 0.5)
    assert rate.realized_rate < 0.1 * mean_of_rates


def test_the_bound_is_reproducible_from_its_seed() -> None:
    """Bit-identical across runs, because a published interval that moves
    between invocations is not a published interval.

    Seed *sensitivity* is deliberately not asserted: the resampled rate is a
    ratio of integer counts over a fixed cluster set, so it takes discrete
    values and two seeds routinely land on the same quantile. That is the
    statistic being discrete, not the resampling failing to happen.
    """

    rate = fill(ClusteredRate(), 300, 20, lambda c: 1 if c % 7 == 0 else 0)
    assert rate.bootstrap_upper(replicates=1_000, seed=11) == \
           rate.bootstrap_upper(replicates=1_000, seed=11)
    assert rate.bootstrap_upper(replicates=1_000, seed=11) >= rate.realized_rate


def test_the_resampling_actually_varies_the_statistic() -> None:
    """The companion to the above: with heterogeneous episodes the bound does
    move with the seed, which is how we know the draw is doing work."""

    rate = ClusteredRate()
    rng = np.random.default_rng(9)
    for c in range(400):
        length = int(rng.integers(5, 400))
        misses = int(rng.integers(0, 4))
        for k in range(length):
            rate.observe(f"c{c}", missed=k < misses, expected_failure=1e-4)
    bounds = {rate.bootstrap_upper(replicates=500, seed=s) for s in range(6)}
    assert len(bounds) > 1


def test_zero_observed_misses_still_produces_a_usable_upper_bound() -> None:
    """The case that matters most: at 1e-4 most runs see no miss at all, and a
    bound that collapsed to zero would certify a budget on no evidence."""

    rate = fill(ClusteredRate(), 500, 100, 0)
    assert rate.realized_rate == 0.0
    assert rate.bootstrap_upper(replicates=1_000, seed=4) == 0.0
    assert rate.wilson_upper() > 0.0


def test_the_expected_rate_bound_survives_when_no_miss_was_realized() -> None:
    """Because the realized bootstrap legitimately collapses at zero misses,
    the expectation is what carries the constraint check there."""

    rate = fill(ClusteredRate(), 500, 100, 0)
    upper = rate.bootstrap_upper(replicates=1_000, seed=5, statistic="expected")
    assert upper == pytest.approx(1e-4, rel=0.05)


# -- guards -------------------------------------------------------------------


def test_too_few_clusters_is_reported_rather_than_silently_bounded() -> None:
    assert not fill(ClusteredRate(), 10, 100).sufficient()
    assert fill(ClusteredRate(), MIN_CLUSTERS, 10).sufficient()


def test_an_empty_rate_bounds_at_one_rather_than_dividing_by_zero() -> None:
    empty = ClusteredRate()
    assert empty.bootstrap_upper() == 1.0
    assert empty.wilson_upper() == 1.0
    assert empty.realized_rate == 0.0


def test_a_nonsense_confidence_level_is_refused() -> None:
    rate = fill(ClusteredRate(), 50, 10)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(StatisticsError, match="confidence"):
            rate.bootstrap_upper(confidence=bad)


# -- the frontier -------------------------------------------------------------


def test_a_tighter_budget_never_costs_less() -> None:
    """Monotonicity is what makes the curve a frontier rather than a scatter."""

    bound = TargetingBound()
    rng = np.random.default_rng(6)
    for _ in range(5_000):
        rf = float(10 ** rng.uniform(-6, -2))
        bound.observe(rf, rf * 0.25)
    costs = [cost for _, cost, _, _ in bound.frontier((1e-3, 1e-4, 1e-5, 1e-6))]
    assert costs == sorted(costs)


def test_the_frontier_flags_a_budget_no_mix_can_reach() -> None:
    """Duplicating everything is the most it can do; past that it says so
    rather than reporting a cost above 2."""

    bound = TargetingBound()
    for _ in range(1_000):
        bound.observe(1e-2, 1e-3)
    rows = {target: (cost, feasible) for target, cost, _, feasible in
            bound.frontier((1e-2, 1e-3, 1e-9))}
    assert rows[1e-2][1] is True
    assert rows[1e-9] == (pytest.approx(2.0), False)


def test_the_frontier_reuses_one_ordering() -> None:
    """Each budget's solution is a prefix of the same descending order, so the
    sort happens once however many budgets are swept."""

    bound = TargetingBound()
    for index in range(1_000):
        bound.observe(1e-3 * (index + 1) / 1000, 1e-7)
    bound.frontier((1e-3, 1e-4))
    first = bound._descending()
    bound.frontier((1e-5,))
    assert bound._descending() is first


def test_observing_after_a_solve_invalidates_the_cached_order() -> None:
    bound = TargetingBound()
    for _ in range(10):
        bound.observe(1e-3, 1e-6)
    bound.solve(1e-4)
    bound.observe(0.9, 1e-6)
    assert bound._descending()[0] == pytest.approx(0.9 - 1e-6)


# -- the zero-miss failure mode, and the estimator that survives it -----------


def test_the_realized_bootstrap_cannot_bound_a_zero_count() -> None:
    """Stated as a limitation rather than hidden.

    Resampling clusters that all contain no miss can only ever produce no miss,
    so the bound collapses to zero. Printed as "95% upper bound = 0" that reads
    as certainty, which is why the report must not use it here.
    """

    rate = fill(ClusteredRate(), 500, 100, 0)
    assert rate.bootstrap_upper(replicates=500, seed=7) == 0.0
    assert not rate.realized_bound_is_informative


def test_the_expected_bound_still_works_with_no_observed_miss() -> None:
    """The reason the constraint is estimated as an expectation.

    Every packet carries a failure probability whether or not it failed, so
    the statistic varies across episodes and the resampling has something to
    work with.
    """

    rate = ClusteredRate()
    rng = np.random.default_rng(8)
    for c in range(400):
        level = float(10 ** rng.uniform(-5, -3))
        for _ in range(int(rng.integers(20, 300))):
            rate.observe(f"c{c}", missed=False, expected_failure=level)
    upper = rate.expected_upper(replicates=1_000, seed=2)
    assert upper > rate.expected_rate > 0.0
    assert rate.bootstrap_upper(replicates=1_000, seed=2) == 0.0


def test_a_miss_makes_the_realized_bound_informative_again() -> None:
    rate = fill(ClusteredRate(), 400, 50, lambda c: 1 if c < 4 else 0)
    assert rate.realized_bound_is_informative
    assert rate.bootstrap_upper(replicates=1_000, seed=9) > rate.realized_rate
