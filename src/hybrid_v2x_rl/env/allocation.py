"""Who carries each packet, and what that does to the radio it left behind.

The existing frontier in :mod:`hybrid_v2x_rl.env.campaign` answers "which packets are
worth duplicating", and it answers it against a fixed baseline: every packet
starts on the radio and duplication is the only upgrade. That is the right
question for a diversity argument and the wrong one here, because it cannot
express the move this module exists to price -- **sending a packet by light and
not touching the radio at all**.

That move matters because RF attempts are not free and not private. The mode-2
pool supplies a fixed number of resources per generation period, every vehicle
draws from the same pool, and a packet that goes optical returns its share.
So the offered radio load is a *consequence* of the allocation, and the
collision each surviving RF packet then sees is a consequence of that. The
allocation and the channel determine each other.

**So this is a fixed point, not a solve.** Start with every vehicle on the
radio, price the packets, let the cheap-by-light ones leave, recompute
contention against the smaller offered load, and repeat until the fraction
stops moving. What comes back is an equilibrium under the declared assumption
that every vehicle runs the same allocation -- a mean field. Stated plainly
because it is load-bearing: one vehicle offloading changes nothing, and the
result is about what happens when the fleet does.

**Two conservatisms, both pointing the same way.** Per-packet RF risk here is
access-layer only -- collision and half-duplex, no decoding term -- because the
completed campaign attributes every RF failure at every trained density to
``RF_COLLISION``. And attempts are treated as failing independently, which is
the most generous possible reading of retransmission diversity. Both make the
radio look *better* than it is. That is deliberate: this module is used to
argue the radio runs out, and an argument that survives its own optimism is
worth more than one tuned to win.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from hybrid_v2x_rl.channels.rf.collision import (
    CollisionParameters,
    failure_probability,
    resource_demand,
)
from hybrid_v2x_rl.core.errors import HybridV2XError

#: Default initial RF-load fractions used to expose distinct fixed-point branches.
DEFAULT_EQUILIBRIUM_STARTS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)

#: Relative miss-rate and absolute activation-cost settling tolerance.
DEFAULT_EQUILIBRIUM_TOLERANCE = 1e-2

#: Iterations the reported quantities must remain settled over.
EQUILIBRIUM_SETTLED_WINDOW = 8

#: Delayed noisy-geometry rules screened in the paper's deployment analysis.
SELECTOR_THRESHOLD_RULES: tuple[tuple[float, float], ...] = (
    (8.0, 0.95),
    (9.5, 0.90),
    (11.0, 0.90),
    (13.0, 0.80),
    (16.0, 0.80),
    (25.0, 0.60),
    (float("inf"), 0.0),
)


class AllocationError(HybridV2XError):
    """An allocation was requested with inconsistent inputs."""


@dataclass(frozen=True, slots=True)
class Allocation:
    """What one assignment of packets to media costs and achieves."""

    mean_cost: float
    miss_rate: float
    #: Share of packets that put a copy on the radio: RF-only plus duplicated.
    #: This is what the pool actually sees, and what the fixed point iterates.
    rf_fraction: float
    #: Share carried by light alone -- the packets that return their resources.
    vlc_only_fraction: float
    dup_fraction: float
    feasible: bool


def fit_risk_estimate(
    feature: np.ndarray, truth: np.ndarray, *, edges: np.ndarray
) -> np.ndarray:
    """Mean true risk per bin of one observable, fitted on a held-out split.

    The estimator a real vehicle could actually run. It sees a noisy tracked
    separation, not the optical channel, so the best it can say is what packets
    at this separation have historically cost. Deliberately the dumbest thing
    that could work: a binned conditional mean has no capacity to memorise the
    packets it was fitted on, so a gap between it and the oracle is the
    estimator's honest loss rather than an overfitting artefact.

    Returns one risk per bin, with empty bins carrying the global mean -- the
    pessimistic filling, since an unseen separation is not evidence of a good
    one.
    """

    values = np.asarray(feature, dtype=np.float64)
    outcome = np.asarray(truth, dtype=np.float64)
    if values.shape != outcome.shape:
        raise AllocationError("feature and truth must describe the same packets")
    index = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    totals = np.bincount(index, weights=outcome, minlength=len(edges) - 1)
    counts = np.bincount(index, minlength=len(edges) - 1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), outcome.mean())


def apply_risk_estimate(
    feature: np.ndarray, *, edges: np.ndarray, table: np.ndarray
) -> np.ndarray:
    """Score packets with an estimate fitted elsewhere."""

    values = np.asarray(feature, dtype=np.float64)
    index = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    return table[index]


def solve_allocation(
    rf_risk: np.ndarray,
    vlc_risk: np.ndarray,
    budget: float,
    *,
    vlc_belief: np.ndarray | None = None,
) -> Allocation:
    """Cheapest per-packet assignment whose mean miss meets ``budget``.

    Each packet's baseline is its **cheaper single leg**, which costs one
    activation and misses ``min(rf, vlc)``. That one word is the whole
    difference from ``PolicyStatistics.solve``: a packet whose optical path is
    short and clear takes the light, and the radio never hears about it.

    Duplication is the only upgrade, costs exactly one more activation, and
    reduces the packet's miss to ``rf * vlc``. Because every upgrade costs the
    same, taking them in order of decreasing reduction is not a heuristic -- it
    is optimal, since any feasible set of ``k`` upgrades is beaten by the ``k``
    largest reductions.

    Independence across the two media is assumed in the ``rf * vlc`` term. The
    campaign measures the dependence ratio at 0.93, so the real joint failure is
    slightly *better* than this predicts and the cost reported here is, again,
    marginally conservative.

    **Decisions are made on ``vlc_belief`` and scored on ``vlc_risk``.** A
    vehicle chooses with what it can observe and then lives with what the
    channel actually does, and collapsing the two would report an allocation
    nobody could have made. Defaulting the belief to the truth recovers the
    oracle, which is the right upper bound to quote a realistic estimator
    against -- the gap between them is the cost of not knowing.
    """

    rf = np.asarray(rf_risk, dtype=np.float64)
    vlc = np.asarray(vlc_risk, dtype=np.float64)
    if rf.shape != vlc.shape:
        raise AllocationError(
            "risk arrays must describe the same packets",
            context={"rf": rf.shape, "vlc": vlc.shape},
        )
    n = rf.size
    if n == 0:
        raise AllocationError("an allocation needs at least one packet")
    if not 0.0 < budget < 1.0:
        raise AllocationError("budget must lie in (0, 1)", context={"budget": budget})

    belief = vlc if vlc_belief is None else np.asarray(vlc_belief, dtype=np.float64)
    if belief.shape != vlc.shape:
        raise AllocationError(
            "belief must describe the same packets as the outcome",
            context={"belief": belief.shape, "vlc": vlc.shape},
        )

    # Chosen on the belief, paid on the truth.
    on_rf = rf <= belief
    single = np.where(on_rf, rf, vlc)
    joint = rf * vlc
    # The upgrade is ranked by the reduction the vehicle *expects*, since that
    # is the number it has, but every reported total is the realized one.
    expected_single = np.where(on_rf, rf, belief)

    total = float(np.sum(single))
    if total / n <= budget:
        return Allocation(
            mean_cost=1.0,
            miss_rate=total / n,
            rf_fraction=float(np.count_nonzero(on_rf)) / n,
            vlc_only_fraction=float(np.count_nonzero(~on_rf)) / n,
            dup_fraction=0.0,
            feasible=True,
        )

    # Upgrade in order of decreasing benefit until the mean clears the budget.
    reduction = expected_single - rf * belief
    order = np.argsort(reduction)[::-1]
    running = total - np.cumsum((single - joint)[order])
    cleared = np.flatnonzero(running <= budget * n)
    if cleared.size == 0:
        upgrades = n
        achieved = float(np.sum(joint)) / n
    else:
        upgrades = int(cleared[0]) + 1
        achieved = float(running[cleared[0]]) / n

    duplicated = np.zeros(n, dtype=bool)
    duplicated[order[:upgrades]] = True
    return Allocation(
        mean_cost=1.0 + upgrades / n,
        miss_rate=achieved,
        rf_fraction=float(np.count_nonzero(on_rf | duplicated)) / n,
        vlc_only_fraction=float(np.count_nonzero(~on_rf & ~duplicated)) / n,
        dup_fraction=upgrades / n,
        feasible=achieved <= budget * (1.0 + 1e-9),
    )


def rf_packet_risk(
    neighbour_counts: np.ndarray,
    parameters: CollisionParameters,
    *,
    attempts: int,
    sensed_fraction: float = 1.0,
) -> np.ndarray:
    """Packet-level RF failure for each packet's own contender headcount.

    Evaluated through :func:`~hybrid_v2x_rl.channels.rf.collision.failure_probability`
    on each *distinct* headcount rather than each packet: the counts are small
    integers and the packets number millions, so a lookup over the unique values
    is the same arithmetic a few thousand times instead of a few million. Going
    through the module's own function rather than inlining the birthday form is
    what keeps this from becoming a second collision model.
    """

    if attempts < 1:
        raise AllocationError("a packet needs at least one attempt",
                              context={"attempts": attempts})
    counts = np.asarray(neighbour_counts)
    if np.any(counts < 0):
        raise AllocationError("neighbour counts cannot be negative")
    unique, inverse = np.unique(np.rint(counts).astype(np.int64), return_inverse=True)
    table = np.array(
        [
            failure_probability(int(c), parameters, sensed_fraction=sensed_fraction)
            for c in unique
        ],
        dtype=np.float64,
    )
    return (table**attempts)[inverse]


#: Separation bins the occlusion onset rate is resolved over, in metres.
OCCLUSION_SEPARATION_EDGES_M: tuple[float, ...] = (0.0, 9.5, 12.0, 16.0, 25.0)

#: Onset rate per second of clear optical path, by density then by separation
#: bin. Measured at 50 ms over 6,585 / 9,941 / 11,741 s of clear path.
#:
#: The first entry of each row is the same **pooled bound** rather than a
#: per-density measurement: below 9.5 m the three densities together supply
#: 17,071 s of clear path and one onset, which is too few to resolve by density
#: and too few to call zero. The Poisson 95% upper bound for one event gives
#: 4.744/17071, and that is what is carried.
OCCLUSION_ONSET_RATE_HZ: dict[float, tuple[float, ...]] = {
    10.0: (4.744 / 17071, 0.0833, 0.0319, 0.0151, 0.0454),
    20.0: (4.744 / 17071, 0.0332, 0.0362, 0.0248, 0.0796),
    30.0: (4.744 / 17071, 0.0241, 0.0333, 0.0441, 0.0800),
}


def occlusion_in_flight_risk(
    separation_m: np.ndarray, *, airtime_s: float, density: float
) -> np.ndarray:
    """Chance a vehicle occludes the optical path *during* the transmission.

    The static optical PER answers whether the path is clear at the instant of
    the decision. A packet is in flight for its whole airtime and the path can
    close underneath it, so the two are different questions and only the first
    is in the cached risk.

    **Why the answer is negligible for the packets that matter, and why that is
    a measurement rather than an assumption.** Occluding a following pair's
    optical path needs a third vehicle between them, and inserting one needs
    9.5 m of longitudinal gap -- a 4.5 m body plus the 2.5 m minimum spacing the
    car-following model enforces at each end. Below that separation there is
    nowhere to go. Measured across all three densities: 17,071 s of clear path
    below 9.5 m and **one** onset, against 396 above it.

    That one event is why this is a bound and not a zero. It sits at 8.0-9.5 m,
    just under the threshold, and it is a reminder that the 9.5 m argument
    covers longitudinal insertion only -- a vehicle in an adjacent lane crossing
    the path diagonally needs no gap between the pair at all. So the rate below
    the threshold is not zero, it is roughly seven hundred times smaller than
    above it, and the Poisson bound for a single event is carried instead of the
    point estimate.

    **Resolved by density above the threshold, because it varies and not
    monotonically.** In the 9.5-12 m bin the rate *falls* with density, 0.083 to
    0.024, as gaps close and insertion gets harder; in the 16-25 m bin it
    *rises*, 0.015 to 0.044, as there are simply more vehicles to do it. A rate
    measured at one density and applied at another is wrong in both directions
    by up to 3.5x. These bins govern the packets that are *not* offloaded, so
    getting them right sharpens the boundary rather than moving it.

    Two conservatisms remain. The occlusion model this was measured through is
    binary (``complete_blockage_main``), so a vehicle edge entering the beam
    kills the link instantly; a real beam has spatial extent and would fade over
    some milliseconds, which makes the measured rate an upper bound on the real
    one. And the sub-threshold bound is a 95% upper limit rather than the 1.4e-7
    the single event actually implies.
    """

    if airtime_s <= 0.0:
        raise AllocationError("airtime must be positive",
                              context={"airtime_s": airtime_s})
    known = np.asarray(sorted(OCCLUSION_ONSET_RATE_HZ))
    nearest = float(known[int(np.argmin(np.abs(known - float(density))))])
    rates = np.asarray(OCCLUSION_ONSET_RATE_HZ[nearest])
    edges = np.asarray(OCCLUSION_SEPARATION_EDGES_M)
    index = np.clip(np.digitize(np.asarray(separation_m), edges) - 1, 0, rates.size - 1)
    return np.asarray(rates[index] * airtime_s, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Equilibrium:
    """The allocation and the channel that produce each other."""

    allocation: Allocation
    parameters: CollisionParameters
    attempts: int
    #: Pool claim at the equilibrium load. Above one the profile is not
    #: deliverable however small the collision probability looks.
    resource_demand: float
    iterations: int
    converged: bool
    #: Distinct equilibria the starting sweep found. More than one means the
    #: system is bistable and which one it settles in is a deployment question,
    #: not a property of the physics.
    equilibria: tuple[float, ...] = ()

    @property
    def deliverable(self) -> bool:
        """Feasible on reliability, affordable in resources, and actually solved.

        Convergence is part of the verdict rather than a caveat beside it. A
        ringing iterate reports an allocation and a contention level that were
        never each other's answer -- the numbers are individually well formed and
        jointly meaningless -- and reading ``feasible`` off one of those is how a
        non-result gets published as a result.
        """

        return (
            self.converged
            and self.allocation.feasible
            and self.resource_demand <= 1.0
        )


def solve_equilibrium(
    *,
    neighbour_counts: np.ndarray,
    vlc_risk: np.ndarray,
    budget: float,
    attempts: int,
    attempt_airtime_s: float,
    template: CollisionParameters,
    sensed_fraction: float = 1.0,
    vlc_belief: np.ndarray | None = None,
    damping: float = 1.0,
    tolerance: float = DEFAULT_EQUILIBRIUM_TOLERANCE,
    max_iterations: int = 50,
    starts: tuple[float, ...] = DEFAULT_EQUILIBRIUM_STARTS,
) -> Equilibrium:
    """Iterate allocation and contention to their common fixed point.

    **Swept from several starting loads, because the map has more than one fixed
    point.** Started at full radio use -- the profile as frozen -- a congested
    pool makes duplication look necessary, duplicating holds the pool congested,
    and the iteration settles there: self-consistent, cost 2.0, and infeasible.
    Started lower, the same inputs settle on most packets going by light, the
    offered load collapsing, and the radio comfortably carrying the rest at cost
    1.0. Both are equilibria of the same system. Measured on the pessimistic
    band at rho = 30, N = 4, the first gives 7.2e-5 and the second 2.2e-7.

    So a single start does not answer the question asked of this function --
    whether the profile *can* meet the budget -- it answers whether one
    particular initial condition happens to reach it. The sweep returns the
    cheapest feasible equilibrium found, and records the rest in ``equilibria``,
    because bistability is a real property here: a fleet that begins congested
    stays congested without something to coordinate the move to light, and that
    is a deployment finding rather than a solver detail.

    Undamped by default, because measured on the trained caches the map is
    strongly contracting: ``rf_fraction`` reaches its fixed point in three
    iterations at every density, to the last digit, with no ringing. Damping
    below one is kept for the case that stops being true -- a crowded pool
    pushing packets to light and the emptied pool pulling them back is the
    obvious way this map could oscillate -- and averaging successive iterates
    would converge that without moving the fixed point, since a fixed point of
    the damped map is a fixed point of the map.

    Convergence is reported rather than assumed. The map is not a contraction by
    construction: the allocation is a discrete argsort, so ``rf_fraction`` moves
    in steps of one packet, and a caller that treated a ringing result as an
    equilibrium would be reporting an artefact of the iteration count.
    """

    if not 0.0 < damping <= 1.0:
        raise AllocationError("damping must lie in (0, 1]", context={"damping": damping})

    counts = np.asarray(neighbour_counts)
    committed = attempts * attempt_airtime_s
    if not starts:
        raise AllocationError("at least one starting load is required")

    found: list[tuple[Allocation, int, bool]] = []
    for start in starts:
        if not 0.0 <= start <= 1.0:
            raise AllocationError("a starting load must lie in [0, 1]",
                                  context={"start": start})
        fraction = start
        recent: list[tuple[float, float]] = []
        allocation: Allocation | None = None
        converged = False
        used = 0
        while used < max_iterations:
            used += 1
            rf_risk = rf_packet_risk(
                counts,
                replace(template, airtime_s=committed, rf_usage_fraction=fraction),
                attempts=attempts,
                sensed_fraction=sensed_fraction,
            )
            allocation = solve_allocation(
                rf_risk, vlc_risk, budget, vlc_belief=vlc_belief
            )
            # Convergence is a property of the *state*, not of the allocator's
            # reply to it. Near the fixed point the reply is set-valued: the
            # allocation is a discrete argsort, so whole groups of packets flip
            # together as the radio's risk crosses their optical risk, and the
            # reply alternates within a narrow band rather than settling on a
            # value. Measured at rho = 30 with four attempts that band is 0.005
            # wide, and testing the reply gap against it never terminates while
            # the state itself has been fixed to five figures for thirty
            # iterations, at one miss rate and one cost. Testing the reply gap
            # reported that settled answer as ringing.
            fraction = (1.0 - damping) * fraction + damping * allocation.rf_fraction
            recent.append((allocation.miss_rate, allocation.mean_cost))
            if len(recent) > EQUILIBRIUM_SETTLED_WINDOW:
                recent.pop(0)
            # Settled is a statement about the answer, not about the state.
            # The map is a step function -- whole groups of packets flip together
            # as the radio's risk crosses their optical risk -- so an exact fixed
            # point need not exist, and testing for one never terminates: at
            # rho = 30 with four attempts the state ends up alternating inside a
            # band 1.1e-3 wide, forever, while the miss rate it reports moves by
            # 0.6% and the cost not at all. Choosing a tolerance that admits that
            # band would be tuning a threshold to a single cell. Asking whether
            # the reported quantities have stopped moving needs no such choice,
            # and a genuine limit cycle fails it easily -- undamped at the same
            # point the miss rate swings by a factor of 2.4.
            if len(recent) == EQUILIBRIUM_SETTLED_WINDOW:
                misses = [m for m, _ in recent]
                costs = [c for _, c in recent]
                scale = max(misses) or 1.0
                if ((max(misses) - min(misses)) / scale < tolerance
                        and max(costs) - min(costs) < tolerance):
                    converged = True
                    break
        assert allocation is not None
        found.append((allocation, used, converged))

    # Cheapest equilibrium that actually met the budget, and only among the ones
    # that settled -- an infeasible or ringing branch is not a cheaper answer, it
    # is a different question's answer.
    settled = [entry for entry in found if entry[2]]
    feasible = [entry for entry in settled if entry[0].feasible]
    allocation, used, converged = min(
        feasible or settled or found,
        key=lambda entry: (entry[0].mean_cost, entry[0].miss_rate),
    )
    distinct = tuple(
        sorted({round(entry[0].rf_fraction, 6) for entry in settled})
    )
    # Demand is reported against the fraction the allocation *returned*, not the
    # one it was handed. They are the same number at a fixed point and differ
    # everywhere else, and quoting the input would describe a load nobody
    # offered -- which is how a non-converged cell came to report full radio use
    # and a comfortable pool claim in the same row.
    at_equilibrium = replace(
        template, airtime_s=committed, rf_usage_fraction=allocation.rf_fraction
    )
    return Equilibrium(
        allocation=allocation,
        parameters=at_equilibrium,
        attempts=attempts,
        resource_demand=resource_demand(
            int(round(float(np.mean(counts)))), at_equilibrium
        ),
        iterations=used,
        converged=converged,
        equilibria=distinct,
    )


__all__ = [
    "Allocation",
    "AllocationError",
    "DEFAULT_EQUILIBRIUM_STARTS",
    "DEFAULT_EQUILIBRIUM_TOLERANCE",
    "Equilibrium",
    "EQUILIBRIUM_SETTLED_WINDOW",
    "SELECTOR_THRESHOLD_RULES",
    "apply_risk_estimate",
    "OCCLUSION_ONSET_RATE_HZ",
    "OCCLUSION_SEPARATION_EDGES_M",
    "calibrate_threshold",
    "fit_risk_estimate",
    "occlusion_in_flight_risk",
    "rf_packet_risk",
    "threshold_selector",
    "solve_allocation",
    "solve_equilibrium",
]


def threshold_selector(
    reach_m: float, fov_margin: float, *, believed: float, otherwise: float = 1.0
) -> Callable[[np.ndarray, dict[str, int]], np.ndarray]:
    """A belief a vehicle could actually evaluate, as a decision not an estimate.

    Returns a :data:`~hybrid_v2x_rl.env.campaign.BeliefFn` that offers the optical
    link to packets whose *tracked* separation is under ``reach_m`` and whose
    field-of-view margin is over ``fov_margin``, and refuses it to every other
    packet by believing the light certain to fail.

    **Why a threshold and not an estimate.** The obvious belief is the expected
    optical risk given the observation, and it cannot work here however well it
    is fitted. Optical risk is bimodal -- around 1e-15 with a clear path and
    around 1 without -- so any conditional mean returns the blockage *rate* in
    its neighbourhood, which is a few percent at best and never below the
    radio's failure probability. An allocator handed that belief keeps every
    packet on the radio, correctly. A threshold sidesteps the averaging by
    committing to a decision, which is why it is the shape that gets anywhere.

    ``believed`` is the risk asserted for the packets it accepts, and it must be
    calibrated on a split the selector did not choose from: it is the rate at
    which this rule is wrong, and quoting anything lower is how an evaluation
    starts believing its own selector.
    """

    if not 0.0 <= believed <= 1.0:
        raise AllocationError("a believed risk must be a probability",
                              context={"believed": believed})

    def belief(observations: np.ndarray, columns: dict[str, int]) -> np.ndarray:
        try:
            separation = observations[:, columns["pair_distance"]]
            margin = observations[:, columns["optical_fov_margin"]]
        except KeyError as exc:                       # pragma: no cover - config
            raise AllocationError(
                "the observation does not carry the features this selector reads",
                context={"missing": str(exc)},
            ) from exc
        accepted = (separation < reach_m) & (margin > fov_margin)
        return np.where(accepted, believed, otherwise)

    return belief


def calibrate_threshold(
    observations: np.ndarray, columns: dict[str, int], truth: np.ndarray,
    *, reach_m: float, fov_margin: float,
) -> tuple[float, float]:
    """Coverage and error of one threshold rule, measured on a held-out split.

    The second return value is what :func:`threshold_selector` should be told to
    believe. It is measured rather than assumed because the rule's precision is
    the whole question: the budget wants five nines from it and geometry
    supplies three.
    """

    separation = observations[:, columns["pair_distance"]]
    margin = observations[:, columns["optical_fov_margin"]]
    accepted = (separation < reach_m) & (margin > fov_margin)
    if not accepted.any():
        return 0.0, 1.0
    return float(accepted.mean()), float(np.asarray(truth)[accepted].mean())
