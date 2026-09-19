"""Confidence bounds that respect how the packets were actually generated.

**Packets are not independent samples, and treating them as such is the single
easiest way to publish a reliability claim that is not supported.** Every packet
in one tagged-pair episode shares that pair's shadowing state, its fading
trajectory, its geometry and very nearly its contention. Three hundred packets
from one episode are worth far less than three hundred packets from three
hundred episodes, so the effective sample size is the number of *episodes*.

A Wilson interval -- or any binomial interval -- assumes independent Bernoulli
trials, so in general it reports an interval that is too narrow. **On this
campaign it turned out not to**: measured like for like, the cluster bootstrap
and Wilson agree on the realized rate to within 1% at every density. Misses do
not bunch inside episodes here, and there is a physical reason -- RF failure is
collision-dominated, and the collision draw is independent per packet even
though the contender count it depends on varies slowly.

That is a measurement, not an assumption, and it is worth having either way:
the correct estimator now says the correction is negligible, rather than nobody
having checked. The real tightening came from elsewhere -- bounding the
*expectation* rather than the realized count, which is roughly ten times
tighter because it is a smooth statistic rather than a rare-event tally.

The fix is the one the evaluation profile already specifies: resample whole
**pair-episode clusters** with replacement, recompute the rate as a ratio of
sums over the resampled clusters, and read the one-sided upper quantile. The
constraint is ``miss rate <= budget``, so only the upper tail matters.

Both estimators are kept. Reporting them side by side is itself a result worth
showing: it says how much correlation the trajectory structure carries, and it
is the evidence that the narrower number was not chosen because it was
narrower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hybrid_v2x_rl.core.errors import HybridV2XError

#: Below this many independent clusters a resampling interval is not
#: trustworthy -- the bootstrap can only redistribute the variation it was
#: given, and a handful of episodes cannot represent the population. The
#: evaluation profile names the same figure.
MIN_CLUSTERS = 200


class StatisticsError(HybridV2XError):
    """A confidence bound was requested from evidence that cannot support it."""


@dataclass(slots=True)
class ClusterTally:
    """One pair episode's sums -- the unit that gets resampled."""

    packets: int = 0
    misses: int = 0
    expected_misses: float = 0.0
    cost: float = 0.0


@dataclass(slots=True)
class ClusteredRate:
    """A rate statistic that knows its own correlation structure.

    Accumulates per-episode rather than per-packet, so the bootstrap has
    something to resample. The point estimate is identical either way; only the
    interval differs, and the interval is the part that was wrong.
    """

    clusters: dict[str, ClusterTally] = field(default_factory=dict)

    def observe(
        self,
        cluster_id: str,
        *,
        missed: bool,
        expected_failure: float,
        cost: float = 0.0,
    ) -> None:
        tally = self.clusters.get(cluster_id)
        if tally is None:
            tally = self.clusters[cluster_id] = ClusterTally()
        tally.packets += 1
        tally.expected_misses += expected_failure
        tally.cost += cost
        if missed:
            tally.misses += 1

    # -- totals ---------------------------------------------------------------

    @property
    def cluster_count(self) -> int:
        return len(self.clusters)

    @property
    def packets(self) -> int:
        return sum(t.packets for t in self.clusters.values())

    @property
    def misses(self) -> int:
        return sum(t.misses for t in self.clusters.values())

    @property
    def realized_rate(self) -> float:
        return self.misses / self.packets if self.packets else 0.0

    @property
    def expected_rate(self) -> float:
        total = math.fsum(t.expected_misses for t in self.clusters.values())
        return total / self.packets if self.packets else 0.0

    @property
    def mean_cost(self) -> float:
        total = math.fsum(t.cost for t in self.clusters.values())
        return total / self.packets if self.packets else 0.0

    # -- intervals ------------------------------------------------------------

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tallies = list(self.clusters.values())
        return (
            np.fromiter((t.packets for t in tallies), dtype=np.int64, count=len(tallies)),
            np.fromiter((t.misses for t in tallies), dtype=np.int64, count=len(tallies)),
            np.fromiter(
                (t.expected_misses for t in tallies), dtype=np.float64, count=len(tallies)
            ),
        )

    def bootstrap_upper(
        self,
        *,
        replicates: int = 10_000,
        confidence: float = 0.95,
        seed: int = 0,
        statistic: str = "realized",
    ) -> float:
        """One-sided upper bound from resampling whole episodes.

        Clusters are drawn with replacement and the rate is recomputed as a
        **ratio of sums** -- total misses over total packets across the
        resampled clusters -- rather than as a mean of per-cluster rates. That
        matters because episodes differ in length by an order of magnitude, and
        averaging their rates would weight a five-packet episode the same as a
        five-hundred-packet one.
        """

        if not self.clusters:
            return 1.0
        if not 0.0 < confidence < 1.0:
            raise StatisticsError(
                "confidence must lie strictly between 0 and 1",
                context={"confidence": confidence},
            )
        if replicates < 1:
            raise StatisticsError(
                "at least one bootstrap replicate is required",
                context={"replicates": replicates},
            )

        packets, misses, expected = self._arrays()
        numerator = misses.astype(np.float64) if statistic == "realized" else expected
        rng = np.random.default_rng(seed)
        n = len(packets)

        # Resample in one shot: (replicates x n) index matrix, then sum along
        # the cluster axis. Vectorised because 10,000 replicates over a
        # thousand-odd clusters is otherwise the slowest thing in the report.
        draws = rng.integers(0, n, size=(replicates, n))
        resampled_num = numerator[draws].sum(axis=1)
        resampled_den = packets[draws].sum(axis=1)
        rates = np.divide(
            resampled_num,
            resampled_den,
            out=np.zeros_like(resampled_num, dtype=np.float64),
            where=resampled_den > 0,
        )
        return float(np.quantile(rates, confidence))

    def expected_upper(
        self, *, replicates: int = 10_000, confidence: float = 0.95, seed: int = 0
    ) -> float:
        """Upper bound on the **expected** miss rate -- the reportable one.

        The realized-rate bootstrap has a hard limitation that matters at
        exactly this budget: with zero observed misses it returns zero, because
        resampling clusters that all contain no miss can only ever produce no
        miss. Printed as a 95% bound that says the rate is certainly zero, which
        is the opposite of what the evidence supports.

        The expectation does not have that failure mode. Every packet carries a
        failure probability whether or not it failed, so the statistic varies
        across episodes and the bootstrap has something to resample. It is also
        the quantity the constraint is actually written against.

        So: this bound carries the constraint, and
        :meth:`bootstrap_upper` on the realized rate is the *check* that the
        expectation is not lying -- informative when misses were observed,
        and honestly uninformative when none were.
        """

        return self.bootstrap_upper(
            replicates=replicates, confidence=confidence, seed=seed, statistic="expected"
        )

    @property
    def realized_bound_is_informative(self) -> bool:
        """Whether the realized-rate bootstrap can bound anything at all."""

        return self.misses > 0

    def wilson_upper(self, *, confidence: float = 0.95) -> float:
        """Wilson score interval, upper limit, **one-sided at ``confidence``**.

        No continuity correction::

            upper = [p + z^2/2n + z sqrt(p(1-p)/n + z^2/4n^2)] / (1 + z^2/n)

        ``z`` is the one-sided quantile -- 1.6449 at 0.95, **not** 1.96. That
        distinction is decision-relevant rather than pedantic: at density 30 the
        realized duplication rate bounds to 9.70e-5 with z = 1.6449 and 1.005e-4
        with z = 1.96, so the same data passes or fails the budget depending on
        which convention is meant. An earlier version of this function hardcoded
        1.96 and labelled the result a 95% bound, which is a 97.5% bound.

        Assumes independent trials, which packets within an episode are not.
        Retained because the gap between it and the cluster bootstrap measures
        how much correlation the trajectory structure actually carries -- which
        on this campaign turned out to be almost none.
        """

        n = self.packets
        if not n:
            return 1.0
        z = _normal_quantile(confidence)
        p = self.misses / n
        denominator = 1.0 + z * z / n
        centre = p + z * z / (2 * n)
        spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
        return (centre + spread) / denominator

    def sufficient(self) -> bool:
        """Whether there are enough independent episodes to bound anything."""

        return self.cluster_count >= MIN_CLUSTERS


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, Acklam's rational approximation.

    Good to about 1.15e-9 across the whole domain, which is far beyond what a
    confidence level needs, and it avoids a SciPy dependency for one call.
    """

    if not 0.0 < p < 1.0:
        raise StatisticsError("quantile argument must lie in (0, 1)", context={"p": p})
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low, high = 0.02425, 1.0 - 0.02425
    if p < low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def poisson_interval(count: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided interval on a Poisson count, for the dependence ratio.

    The realized joint-failure count is a handful of events -- 6, 13, 36 across
    the campaign -- so its uncertainty is Poisson, not normal, and quoting the
    ratio without it invites a reader to over-read the difference between 0.85
    and 0.96. Garwood's exact interval via the chi-square relationship, computed
    from the gamma quantile.
    """

    if count < 0:
        raise StatisticsError("a count cannot be negative", context={"count": count})
    alpha = 1.0 - confidence
    lower = 0.0 if count == 0 else _gamma_quantile(alpha / 2.0, count)
    upper = _gamma_quantile(1.0 - alpha / 2.0, count + 1)
    return lower, upper


def _gamma_quantile(p: float, shape: int, *, tolerance: float = 1e-10) -> float:
    """Quantile of Gamma(shape, 1) by bisection on its regularized CDF."""

    if shape <= 0:
        return 0.0
    low, high = 0.0, max(10.0, 2.0 * shape + 10.0)
    while _gamma_cdf(high, shape) < p:
        high *= 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _gamma_cdf(mid, shape) < p:
            low = mid
        else:
            high = mid
        if high - low < tolerance * max(1.0, high):
            break
    return 0.5 * (low + high)


def _gamma_cdf(x: float, shape: int) -> float:
    """Regularized lower incomplete gamma for integer shape.

    For integer shape this is exactly one minus a Poisson tail, so it needs no
    series expansion -- ``P(X <= x) = 1 - sum_{k<shape} e^-x x^k / k!``.
    """

    if x <= 0.0:
        return 0.0
    term = math.exp(-x)
    total = term
    for k in range(1, shape):
        term *= x / k
        total += term
    return max(0.0, min(1.0, 1.0 - total))


__all__ = [
    "MIN_CLUSTERS",
    "ClusterTally",
    "ClusteredRate",
    "StatisticsError",
    "poisson_interval",
]
