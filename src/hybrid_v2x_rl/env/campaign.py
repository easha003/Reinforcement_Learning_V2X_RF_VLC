"""Run every policy over a trace in one pass and accumulate what the gate needs.

**One pass, not four.** Every action's outcome on a packet is already available
from the lifecycle's counterfactuals, computed against the same tape. Running
the baselines as four separate sweeps would cost four times as much and would
only be a fair comparison if all four happened to consume randomness in exactly
the same order -- a property that holds today and that nothing enforces. Reading
them from one packet makes the matching structural.

**The constraint is on an expectation, so it is estimated as one.** Counting
realized misses to verify a 1e-4 budget needs on the order of a million packets
per density before the count is even stable. The per-packet failure probability
is available exactly, and its mean is the same quantity with a variance smaller
by orders of magnitude. Both are reported: the expectation is the estimate, and
the realized count is the check that the expectation is not lying.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from hybrid_v2x_rl.channels.rf.collision import CollisionParameters, SensitivityBand
from hybrid_v2x_rl.channels.rf.collision import resource_demand as _pool_claim
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.enums import FailureCause
from hybrid_v2x_rl.env.episodes import PairInstant, TraceSource
from hybrid_v2x_rl.env.packet import ACTIONS, DUP, RF_ONLY, VLC_ONLY, PacketOutcome
from hybrid_v2x_rl.env.perception import Perception, build_perception
from hybrid_v2x_rl.env.rollout import Rollout, always, best_action
from hybrid_v2x_rl.env.statistics import (
    MIN_CLUSTERS,
    ClusteredRate,
    _normal_quantile,
    poisson_interval,
)

#: The fixed-action baselines, by the name the lifecycle uses for each action.
BASELINES = tuple(action.name for action in ACTIONS)

#: The oracle is not an action, it is a per-packet choice among them.
ORACLE = "ORACLE"

#: The mean-field allocation, scored like any other policy.  Not an action
#: either: it is a per-packet choice made against the contention the choice
#: itself produces.
EQUILIBRIUM = "EQUILIBRIUM"

#: The same allocation chosen on observables rather than on the truth.
PREDICTED = "PREDICTED"

#: Fixed-point controls used by the paper's two-pass evaluation campaign.
CAMPAIGN_EQUILIBRIUM_DAMPING = 0.25
CAMPAIGN_EQUILIBRIUM_MAX_ITERATIONS = 400

#: A belief maps a surveyed population's observation vectors, and the column
#: index of each feature within them, to a believed optical risk per packet.
#:
#: It is handed the observation and nothing else on purpose. An earlier
#: signature also passed the pair separations, which came from
#: ``PacketContext.separation_m`` and is *true* geometry -- a belief written
#: against it would have been reading the answer while appearing to estimate
#: it. Everything a vehicle knows is already in the observation, including the
#: tracked separation, so the narrower signature is also the sufficient one.
BeliefFn = Callable[[np.ndarray, dict[str, int]], np.ndarray]

#: Budgets the frontier is swept over: the headline 1e-4, the exploratory
#: 1e-5 frontier, and enough points either side to draw a curve.
DEFAULT_FRONTIER_BUDGETS: tuple[float, ...] = (
    1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5,
)

#: Optical failures no amount of power can fix, so they set the floor.
_GEOMETRIC_CAUSES = frozenset(
    {FailureCause.VLC_OCCLUSION, FailureCause.VLC_ALIGNMENT}
)


@dataclass(slots=True)
class PolicyStatistics:
    """What one policy did over one density group.

    Counts are kept per pair episode rather than per packet, because that is
    the unit the confidence bound has to resample -- see
    :mod:`hybrid_v2x_rl.env.statistics`. The point estimates are identical either
    way; only the interval changes, and the interval is the part a binomial
    bound gets wrong.
    """

    rate: ClusteredRate = field(default_factory=ClusteredRate)
    causes: Counter[str] = field(default_factory=Counter)
    #: How often the oracle-style chooser picked each action. Constant for a
    #: fixed baseline, and the whole result for the oracle.
    choices: Counter[str] = field(default_factory=Counter)

    def observe(
        self, outcome: PacketOutcome, expected_failure: float, cluster_id: str = ""
    ) -> None:
        self.rate.observe(
            cluster_id,
            missed=not outcome.delivered,
            expected_failure=expected_failure,
            cost=outcome.activation_cost,
        )
        self.choices[outcome.action.name] += 1
        if not outcome.delivered:
            self.causes[outcome.failure_cause.name] += 1

    @property
    def packets(self) -> int:
        return self.rate.packets

    @property
    def misses(self) -> int:
        return self.rate.misses

    @property
    def mean_cost(self) -> float:
        return self.rate.mean_cost

    @property
    def realized_miss_rate(self) -> float:
        return self.rate.realized_rate

    @property
    def expected_miss_rate(self) -> float:
        """The estimator the constraint is actually checked against."""

        return self.rate.expected_rate

    def upper_bound(self, *, replicates: int = 10_000, confidence: float = 0.95) -> float:
        """The reportable bound: the expectation, resampled over episodes.

        Not the realized rate. At a 1e-4 budget most runs observe no miss at
        all, and a realized-rate bootstrap returns zero there -- a bound that
        certifies the constraint on no evidence. See
        :meth:`~hybrid_v2x_rl.env.statistics.ClusteredRate.expected_upper`.
        """

        return self.rate.expected_upper(replicates=replicates, confidence=confidence)

    def realized_upper(
        self, *, replicates: int = 10_000, confidence: float = 0.95
    ) -> float | None:
        """The realized-rate bound, or ``None`` when no miss was observed.

        ``None`` rather than zero: seeing nothing is not evidence of nothing,
        and the caller has to decide how to say so rather than being handed a
        number that reads as certainty.
        """

        if not self.rate.realized_bound_is_informative:
            return None
        return self.rate.bootstrap_upper(replicates=replicates, confidence=confidence)

    def wilson_upper(self, *, confidence: float = 0.95) -> float:
        """The binomial bound, for comparison only. Do not report this alone."""

        return self.rate.wilson_upper(confidence=confidence)


def expected_failure_for(action_name: str, outcome: PacketOutcome) -> float:
    """The marginal this policy is judged on for this packet.

    Duplication is scored on the independence product rather than on the
    realized joint, and the gap between the two is exactly section 8.3's
    dependence ratio -- which is accumulated separately, so the assumption is
    visible as a number instead of buried in this line.

    **The oracle is scored on the joint too, and that is the point.** A
    clairvoyant chooser misses a packet only when no action would have
    delivered it, and the event "no action delivers" is exactly "both legs
    fail" -- duplication's failure event. So the oracle's reliability *is*
    duplication's, and its cost is what separates them. Scoring it on whichever
    action it happened to pick would instead report the marginal of a choice
    that was made *because* it was going to succeed, which is not a probability
    of anything.
    """

    if action_name == "RF":
        return outcome.rf_failure_probability
    if action_name == "VLC":
        return outcome.vlc_failure_probability
    return outcome.joint_failure_probability


@dataclass(slots=True)
class TargetingBound:
    """The cheapest a causal policy could be, given perfect risk prediction.

    The realization oracle answers "what if you knew the outcome", which no
    policy can approach and which therefore bounds nothing a learner can reach.
    This answers the question that matters instead: **what if you knew each
    packet's failure probability, but still had to choose before the draw?**

    That is a knapsack. Duplicating a packet costs one extra unit and buys
    ``p_RF - p_joint`` of expected miss. Every packet costs the same, so the
    greedy solution is exact: sort by how much duplication buys, duplicate from
    the top until the mean expected miss falls under the budget.

    The number this produces is the one to compare a trained policy against. If
    it sits near 2.0, per-packet risk is too flat to target and no policy beats
    duplicating everything -- the contribution would be the constraint
    machinery, not the selection. If it sits near 1.0, risk is concentrated in
    a few identifiable packets and there is real headroom to learn.
    """

    #: ``(reduction, rf_probability)`` per packet, kept to solve the knapsack.
    _packets: list[tuple[float, float]] = field(default_factory=list)
    _sorted: list[float] | None = field(default=None, repr=False)

    def observe(self, rf_probability: float, joint_probability: float) -> None:
        self._packets.append((rf_probability - joint_probability, rf_probability))
        self._sorted = None

    def _descending(self) -> list[float]:
        """Reductions, largest first, computed once and reused.

        The frontier evaluates many budgets against the same packets, and each
        one is a prefix of this same ordering -- so sorting per budget would
        make an O(n log n) job O(bn log n) for no reason.
        """

        if self._sorted is None:
            self._sorted = sorted((reduction for reduction, _ in self._packets), reverse=True)
        return self._sorted

    def solve(self, budget: float) -> tuple[float, float]:
        """Return ``(mean cost, achieved miss rate)`` for the cheapest feasible mix.

        A cost of 1.0 means RF alone already met the budget; 2.0 means even
        duplicating everything did not, and the returned rate says by how much.
        """

        n = len(self._packets)
        if not n:
            return 1.0, 0.0
        total = math.fsum(rf for _, rf in self._packets)
        if total / n <= budget:
            return 1.0, total / n
        duplicated = 0
        for reduction in self._descending():
            if total / n <= budget:
                break
            total -= reduction
            duplicated += 1
        return 1.0 + duplicated / n, total / n

    def frontier(
        self, budgets: Sequence[float]
    ) -> tuple[tuple[float, float, float, bool], ...]:
        """Cost of the cheapest feasible mix across a range of budgets.

        The reliability-cost Pareto frontier, which is what turns "our method
        beats the baselines" into "here is the feasible region and here is
        where every strategy sits in it". Returns one
        ``(budget, cost, achieved, feasible)`` row per budget; ``feasible`` is
        false where even duplicating every packet misses the target, and there
        the cost saturates at 2.0 rather than climbing past it.
        """

        rows = []
        for budget in budgets:
            cost, achieved = self.solve(budget)
            rows.append((budget, cost, achieved, achieved <= budget * (1.0 + 1e-9)))
        return tuple(rows)

    @property
    def packets(self) -> int:
        return len(self._packets)

    def risk_concentration(self, top_fraction: float = 0.01) -> float:
        """Share of total expected RF misses carried by the riskiest packets.

        A flat link puts ``top_fraction`` of its misses in the top fraction; a
        link whose risk is concentrated puts far more. This is the same
        information the knapsack uses, in a form that can be quoted.
        """

        if not self._packets:
            return 0.0
        risks = sorted((rf for _, rf in self._packets), reverse=True)
        cut = max(1, int(top_fraction * len(risks)))
        total = math.fsum(risks)
        return math.fsum(risks[:cut]) / total if total > 0.0 else 0.0


@dataclass(slots=True)
class DensityReport:
    """Every policy's statistics for one density group, plus the joint check."""

    density: float
    policies: dict[str, PolicyStatistics] = field(default_factory=dict)
    #: Realized joint failures against what independence predicted.
    joint_failures: int = 0
    predicted_joint: float = 0.0
    packets: int = 0
    #: Optical failures geometry alone decided -- occlusion or either end's
    #: misalignment. No transmitter or receiver power improvement reaches them,
    #: so they are the optical leg's floor.
    optical_geometric_failures: int = 0
    #: Every optical failure, geometric or budgetary. Larger than the floor by
    #: however many pairs sit beyond the link's range, which at low density is
    #: most of them.
    optical_failures: int = 0
    #: Packets the radio lost and the light saved, and the converse. Together
    #: they are the entire case for carrying two media: if the first is zero,
    #: duplication buys nothing a further RF attempt would not buy cheaper.
    rf_lost_vlc_saved: int = 0
    vlc_lost_rf_saved: int = 0
    #: Contenders summed over packets, so the report can state what the
    #: population asked of the resource pool rather than only what it achieved.
    contender_total: int = 0
    #: The collision profile these packets were evaluated under, recorded by
    #: :func:`run` because the pool claim is a property of the profile and the
    #: traffic together, and the report only has the traffic.
    collision: CollisionParameters | None = None
    #: The bound a learned policy is actually competing against.
    targeting: TargetingBound = field(default_factory=TargetingBound)

    @property
    def mean_contenders(self) -> float:
        return self.contender_total / self.packets if self.packets else 0.0

    @property
    def resource_demand(self) -> float:
        """Airtime this population claims, as a multiple of what the pool has.

        **Above one no reliability number on this report is deliverable**, and
        that is not visible anywhere else in it. The collision model asks where
        one selection lands, not whether every selection can be honoured, so an
        oversubscribed profile still reports a modest per-attempt failure and a
        plausible miss rate. The frozen headline profile sits at 1.19 here at
        the densest trained condition, which no previous version of this table
        said out loud.
        """

        if self.collision is None or not self.packets:
            return 0.0
        return _pool_claim(int(round(self.mean_contenders)), self.collision)

    @property
    def deliverable(self) -> bool:
        """Whether the pool can supply what this profile committed."""

        return self.resource_demand <= 1.0

    def policy(self, name: str) -> PolicyStatistics:
        return self.policies.setdefault(name, PolicyStatistics())

    @property
    def dependence_ratio(self) -> float:
        """Realized joint failure rate over the independence prediction.

        One means the media fail independently. Above one means they fail
        together and duplication buys less than the product suggests; below one
        means they complement. The contribution claims this sits near one
        because RF failure is collision-dominated and collision does not care
        about the geometry that blocks light -- which is an argument, and this
        is the number that settles it.
        """

        if self.predicted_joint <= 0.0:
            return math.inf if self.joint_failures > 0 else 1.0
        return self.joint_failures / self.predicted_joint

    @property
    def cluster_count(self) -> int:
        """Independent pair episodes -- the real sample size for an interval."""

        for stats in self.policies.values():
            if stats.packets:
                return stats.rate.cluster_count
        return 0

    def dependence_interval_text(self) -> str:
        """The ratio with the Poisson uncertainty on its numerator.

        A handful of events decides this number -- 6, 13, 36 across the
        campaign -- so quoting 0.853 without an interval invites a reader to
        over-read the gap between densities that the counts cannot support.
        """

        if self.predicted_joint <= 0.0:
            return f'({self.joint_failures} realized, none predicted)'
        low, high = poisson_interval(self.joint_failures)
        return (
            f'(1.0 = independent; {self.joint_failures} realized '
            f'[{low / self.predicted_joint:.2f}, {high / self.predicted_joint:.2f}] '
            f'vs {self.predicted_joint:.2f} predicted)'
        )

    @property
    def optical_outage(self) -> float:
        """P_out: the share of packets geometry alone denies the optical leg."""

        return self.optical_geometric_failures / self.packets if self.packets else 0.0

    @property
    def optical_failure_rate(self) -> float:
        return self.optical_failures / self.packets if self.packets else 0.0

    @property
    def complementarity(self) -> float:
        """Share of packets where exactly one medium worked and it was the light.

        The hybrid's reason to exist, stated as a rate. It is not the same as
        the reliability gain -- most of these packets the radio would have
        recovered on a later attempt -- but if it were zero the reliability
        gain could not exist at all.
        """

        return self.rf_lost_vlc_saved / self.packets if self.packets else 0.0


def run_equilibrium(
    config: ProjectConfig,
    *,
    sources: Sequence[TraceSource],
    density: float,
    budget: float,
    root_seed: int,
    band: SensitivityBand | None = None,
    max_packets: int | None = None,
    warmup_s: float = 0.0,
    generation_period_s: float,
    belief: BeliefFn | None = None,
    offload_attempts: int = 0,
    policy_name: str = EQUILIBRIUM,
    into: DensityReport | None = None,
) -> DensityReport:
    r"""Score the mean-field allocation, which needs two passes over the traces.

    Every other policy here is a rule the rollout can apply as it goes. This one
    is not, for two reasons that both force a second pass.

    The allocation and the channel determine each other. A packet carried by
    light returns its share of the pool, so the contention the *remaining* radio
    packets face is a consequence of how many left -- and
    :meth:`Rollout.evaluate_instant` draws its outcomes against collision
    parameters fixed when the rollout was built. Scoring the equilibrium against
    the profile's contention would report an allocation nobody made, on a
    channel it would never have produced.

    And the duplication set is global: the cheapest feasible mix upgrades
    packets in order of the risk they shed, which cannot be decided while
    streaming, because the ordering is not known until the last packet is seen.

    So: one pass at full radio use to collect each packet's risk, the fixed
    point solved on those, then the rollout rebuilt at the settled usage
    fraction and a second pass that replays the same packets and scores what the
    allocation actually delivered. The tape is derived from packet identity, so
    the second pass sees the same realized channel as the first up to the
    contention that legitimately changed.

    **``offload_attempts`` grants the offloaded packets a reduced radio
    reservation instead of none.** With it at zero an offloaded packet is
    carried by light alone, which relieves the pool completely and exposes every
    selector error as a miss. At $k > 0$ the packet reserves $k$ attempts rather
    than the profile's $N$, so a selector error costs a joint failure instead of
    a miss while the pool still recovers $N - k$ resources. Reservation is
    pre-emptive, so the saving is real: the airtime is committed by the
    reservation, not by its use. The realized outcome is derived from the same
    tape -- an attempt that succeeded at index $j$ succeeds under any cap
    $k \ge j$ -- so no channel is re-evaluated and the matched tape is intact.

    **What the allocation is allowed to know.** With ``belief`` unset the
    offload is chosen on the packet's *true* optical risk, which makes this an
    upper bound rather than a policy: no vehicle knows that number. It is still
    weaker than :data:`ORACLE`, which knows the realization and not merely the
    risk, and it is the right bound to quote a realistic selector against.
    Passing ``belief`` supplies one -- a function from the causal observation
    vector and the noisy tracked separation to a believed optical risk -- and
    the allocation is then chosen on what a vehicle could actually see, while
    every reported number still comes from what the channel did. The gap between
    the two runs is the cost of not knowing, and it is the number worth
    reporting.
    """

    from hybrid_v2x_rl.env.allocation import (
        occlusion_in_flight_risk,
        rf_packet_risk,
        solve_allocation,
        solve_equilibrium,
    )
    from hybrid_v2x_rl.env.assembly import build_rollout
    from hybrid_v2x_rl.env.episodes import iter_pair_instants
    from hybrid_v2x_rl.observation.builder import ObservationSchema

    report = into if into is not None else DensityReport(density=density)

    def passes(
        usage: float, mean_attempts: float | None = None,
    ) -> Iterator[tuple[Rollout, Perception | None, PairInstant]]:
        """One rollout per trace, at a fixed radio usage fraction.

        A fresh rollout per source because shadowing and fading state must not
        carry across traces, and ``pair_id`` is only unique within one -- which
        is why clusters are keyed by ``trace_id|pair_id``. Iterating every
        source through a single rollout would let one trace's fading state
        decide another trace's outcomes. Every trace receives the same
        experiment root; each component derives its stream from that root and
        the immutable trace ID. Source reordering therefore cannot change a
        trace's draws, and the usage fraction remains shared across the pool.
        """

        for source in sources:
            # A mixed reservation is expressed as the population's mean
            # committed airtime, which is what the contention model reads.
            share = (usage if mean_attempts is None
                     else mean_attempts / config.service.rf_attempts_per_packet)
            rollout = build_rollout(
                config, buildings=(), root_seed=root_seed, band=band,
                rf_usage_fraction=share,
            )
            # Perception only where a belief will read it: it maintains noisy
            # tracks for every vehicle in frame and costs more than the channel
            # evaluation it accompanies.
            seeing = (
                build_perception(config, root_seed=root_seed)
                if belief
                else None
            )
            for instant in iter_pair_instants(
                source, generation_period_s=generation_period_s,
                max_packets=max_packets, warmup_s=warmup_s,
            ):
                yield rollout, seeing, instant
                if instant.final:
                    rollout.release(instant.pair_id)
                    if seeing is not None:
                        seeing.release(instant.pair_id)

    # -- pass one: what each packet costs, at full radio use ------------------
    neighbours: list[int] = []
    separations: list[float] = []
    optical: list[float] = []
    keys: list[tuple[str, str, int]] = []
    template: CollisionParameters | None = None
    seen: list[np.ndarray] = []
    schema = ObservationSchema(
        features=tuple(config.observation.features),
        history_packets=config.observation.history_packets,
    )
    columns = {name: index for index, name in enumerate(schema.columns)}
    for survey, seeing, instant in passes(1.0):
        template = survey.lifecycle.rf.collision
        if seeing is not None:
            observation = seeing.observe(instant)
            if observation is None:
                # No track yet, so no belief can be formed; the packet stays on
                # the radio rather than being offloaded on a guess.
                continue
            seen.append(np.asarray(observation, dtype=np.float64))
        _, context, alternatives = survey.evaluate_instant(
            trace_id=instant.trace_id, pair_id=instant.pair_id, index=instant.index,
            density=density, time_s=instant.time_s,
            transmitter=instant.transmitter, receiver=instant.receiver,
            neighbours=instant.neighbours, index_of_frame=instant.index_of_frame,
            choose=always(ACTIONS[0]), counterfactual=True,
        )
        assert alternatives is not None
        neighbours.append(context.neighbour_count)
        separations.append(context.separation_m)
        optical.append(alternatives["VLC"].vlc_failure_probability)
        keys.append((instant.trace_id, instant.pair_id, instant.index))

    if not keys:
        return report
    assert template is not None

    counts = np.asarray(neighbours)
    vlc_risk = 1.0 - (1.0 - np.asarray(optical)) * (
        1.0 - occlusion_in_flight_risk(
            np.asarray(separations),
            airtime_s=config.vlc.timing.airtime_s,
            density=density,
        )
    )

    believed = (
        vlc_risk if belief is None
        else np.asarray(belief(np.vstack(seen), columns), dtype=np.float64)
    )
    if believed.shape != vlc_risk.shape:
        raise ValueError("a belief must cover exactly the surveyed packets")
    # The belief has to enter here as well as at the decision below. Solving the
    # fixed point on the truth while deciding on the belief gives the policy the
    # contention an oracle's offload would have produced and then lets it make a
    # deployable selector's choices -- a channel and a policy that disagree. It
    # showed up as an all-radio allocation reporting a pool claim of 0.50 where
    # all-radio is 1.22.
    settled = solve_equilibrium(
        neighbour_counts=counts, vlc_risk=vlc_risk, budget=budget,
        attempts=config.service.rf_attempts_per_packet,
        attempt_airtime_s=config.rf.timing.airtime_s,
        template=template, vlc_belief=None if belief is None else believed,
        damping=CAMPAIGN_EQUILIBRIUM_DAMPING,
        max_iterations=CAMPAIGN_EQUILIBRIUM_MAX_ITERATIONS,
    )

    # -- the decision, taken once the whole population is known ---------------
    fraction = settled.allocation.rf_fraction
    at_equilibrium = rf_packet_risk(
        counts,
        replace(template,
                airtime_s=config.rf.timing.airtime_s
                * config.service.rf_attempts_per_packet,
                rf_usage_fraction=fraction),
        attempts=config.service.rf_attempts_per_packet,
    )
    final = solve_allocation(
        at_equilibrium, vlc_risk, budget,
        vlc_belief=None if belief is None else believed,
    )
    on_rf = at_equilibrium <= believed
    reduction = np.where(on_rf, at_equilibrium, believed) - at_equilibrium * believed
    upgrades = int(round(final.dup_fraction * len(keys)))
    duplicated = np.zeros(len(keys), dtype=bool)
    if upgrades:
        duplicated[np.argsort(reduction)[::-1][:upgrades]] = True
    decision = {
        key: (DUP if dup else (RF_ONLY if rf else VLC_ONLY))
        for key, rf, dup in zip(keys, on_rf, duplicated, strict=True)
    }
    # Offloaded packets keep a reduced reservation rather than none. The action
    # evaluated is DUP -- both legs on the matched tape -- and the cap is applied
    # to the realized radio outcome afterwards.
    capped = {key for key, rf, dup in zip(keys, on_rf, duplicated, strict=True)
              if not rf and not dup} if offload_attempts else set()
    mean_attempts = (
        config.service.rf_attempts_per_packet
        - (1.0 - fraction) * (config.service.rf_attempts_per_packet - offload_attempts)
        if offload_attempts else None
    )

    # -- pass two: what it delivered, on the channel it produced --------------
    expected = dict(zip(keys, at_equilibrium * vlc_risk, strict=True))
    # A capped packet's radio risk is q^k, and q is recoverable from the
    # uncapped marginal because the attempts are independent draws.
    attempts_full = config.service.rf_attempts_per_packet
    capped_expected = {
        key: float(vlc_risk[i]) * float(at_equilibrium[i]) ** (
            offload_attempts / attempts_full)
        for i, key in enumerate(keys) if key in capped
    }
    usage = fraction if not offload_attempts else 1.0
    for scored, _, instant in passes(usage, mean_attempts):
        key = (instant.trace_id, instant.pair_id, instant.index)
        action = DUP if key in capped else decision[key]
        outcome, _, _ = scored.evaluate_instant(
            trace_id=instant.trace_id, pair_id=instant.pair_id, index=instant.index,
            density=density, time_s=instant.time_s,
            transmitter=instant.transmitter, receiver=instant.receiver,
            neighbours=instant.neighbours, index_of_frame=instant.index_of_frame,
            choose=always(action),
        )
        if key in capped:
            # Delivered if the light arrived, or the radio arrived inside the
            # reduced reservation. Cost is both media, as for any duplication;
            # what the cap buys is pool, not activations.
            within = (outcome.rf_delivered
                      and outcome.rf_attempts_used <= offload_attempts)
            outcome = replace(outcome, delivered=bool(outcome.vlc_delivered or within))
            risk = capped_expected[key]
        else:
            risk = (expected[key] if action is DUP
                    else (outcome.rf_failure_probability if action is RF_ONLY
                          else outcome.vlc_failure_probability))
        report.policy(policy_name).observe(
            outcome, risk, f"{instant.trace_id}|{instant.pair_id}"
        )
        report.packets += 1
        if report.collision is None:
            report.collision = scored.lifecycle.rf.collision

    # Contenders come from the survey pass, which measured the count *within
    # range* through the geometry. The frame's neighbour list is every vehicle
    # the source handed over, several times larger, and using it reported a pool
    # claim of 5.38 where the truth was 0.77.
    report.contender_total += int(counts.sum())

    return report


def run(
    rollout: Rollout,
    instants: Iterable[PairInstant],
    *,
    density: float,
    release_finished_pairs: bool = True,
    into: DensityReport | None = None,
) -> DensityReport:
    """Evaluate every policy on every instant and accumulate the report.

    ``into`` pools a further trace into an existing report. Replicates of one
    density are independent by construction -- different seeds, disjoint
    vehicle populations -- so their pair episodes join one cluster set rather
    than producing three intervals that then have to be combined by hand.
    Each replicate still needs its own rollout, because shadowing and fading
    state must not carry across traces.
    """

    report = into if into is not None else DensityReport(density=density)
    if report.density != density:
        raise ValueError(
            f'cannot pool a density-{density:g} trace into a '
            f'density-{report.density:g} report'
        )
    for instant in instants:
        _, context, alternatives = rollout.evaluate_instant(
            trace_id=instant.trace_id,
            pair_id=instant.pair_id,
            index=instant.index,
            density=density,
            time_s=instant.time_s,
            transmitter=instant.transmitter,
            receiver=instant.receiver,
            neighbours=instant.neighbours,
            index_of_frame=instant.index_of_frame,
            # The taken action is irrelevant: every action's outcome comes back
            # in the counterfactuals, and the report scores all of them.
            choose=always(ACTIONS[0]),
            counterfactual=True,
        )
        assert alternatives is not None
        report.packets += 1
        report.contender_total += context.neighbour_count
        if report.collision is None:
            report.collision = rollout.lifecycle.rf.collision
        elif report.collision != rollout.lifecycle.rf.collision:
            raise ValueError(
                "cannot pool traces evaluated under different collision profiles"
            )

        cluster = f"{instant.trace_id}|{instant.pair_id}"
        for name, outcome in alternatives.items():
            report.policy(name).observe(
                outcome, expected_failure_for(name, outcome), cluster
            )

        chosen = best_action(alternatives)
        oracle_outcome = alternatives[chosen.name]
        report.policy(ORACLE).observe(
            oracle_outcome, expected_failure_for(ORACLE, oracle_outcome), cluster
        )

        rf, vlc, dup = alternatives["RF"], alternatives["VLC"], alternatives["DUP"]
        report.predicted_joint += dup.joint_failure_probability
        if dup.failure_cause is FailureCause.JOINT_FAILURE:
            report.joint_failures += 1
        if vlc.failure_cause in _GEOMETRIC_CAUSES:
            report.optical_geometric_failures += 1
        if not vlc.delivered:
            report.optical_failures += 1
        # The complementarity the hybrid is for: a packet the radio lost and
        # the light saved. If this is zero, duplication is buying nothing that
        # a third RF attempt would not have bought more cheaply.
        if not rf.delivered and vlc.delivered:
            report.rf_lost_vlc_saved += 1
        if rf.delivered and not vlc.delivered:
            report.vlc_lost_rf_saved += 1
        report.targeting.observe(
            rf.rf_failure_probability, dup.joint_failure_probability
        )

        if release_finished_pairs and instant.final:
            rollout.release(instant.pair_id)

    return report


def format_report(
    report: DensityReport,
    budget: float = 1e-4,
    *,
    bootstrap_replicates: int = 10_000,
    frontier_budgets: Sequence[float] | None = None,
    confidence: float = 0.95,
) -> str:
    """A table a reader can check the constraint against directly."""

    cap = (
        f"caps the optical leg's contribution at {1.0 / report.optical_outage:,.1f}x"
        if report.optical_outage > 0
        else "no geometric floor observed"
    )
    claim = report.resource_demand
    verdict = (
        "within the pool"
        if claim <= 1.0
        else "OVERSUBSCRIBED -- no miss rate below is deliverable"
    )
    lines = [
        f"density {report.density:g} veh/km   packets {report.packets:,}",
        f"  pool claim       demand = {claim:>7.2f} from {report.mean_contenders:,.0f} "
        f"mean contenders ({verdict})",
        f"  optical outage   P_out = {report.optical_outage:>7.2%} geometric ({cap})",
        f"                         = {report.optical_failure_rate:>7.2%} including "
        f"out-of-range, which is what duplication actually pays for",
        # Never a percentage: this rate lives at 1e-5 by construction, and
        # "0.00%" is how the one statistic that justifies the second medium
        # gets read as a zero.
        f"  complementarity        = {report.complementarity:>7.3e} "
        f"({report.rf_lost_vlc_saved:,} packets the radio lost and the light saved; "
        f"{report.vlc_lost_rf_saved:,} the other way)",
        f"  dependence ratio (8.3) = {report.dependence_ratio:>7.3f}   "
        f"{report.dependence_interval_text()}",
    ]

    clusters = report.cluster_count
    if clusters:
        enough = "" if clusters >= MIN_CLUSTERS else f"  BELOW the {MIN_CLUSTERS} required"
        lines.append(f"  pair-episode clusters  = {clusters:>7,}{enough}")

    lines += [
        "",
        f"  bounds are ONE-SIDED upper at {confidence:.0%}, z = "
        f"{_normal_quantile(confidence):.4f}  (z = 1.96 would be a 97.5% "
        f"one-sided bound, not this one)",
        "",
        f"  {'policy':<8} {'cost':>6} {'E[miss]':>10} {'E up':>10} "
        f"{'holds':>6} {'realized':>10} {'R up':>10}",
    ]
    # EQUILIBRIUM only when it was scored -- the run is optional, and an
    # empty row would read as a policy that missed nothing.
    ordered = (*BASELINES, ORACLE)
    if EQUILIBRIUM in report.policies:
        ordered = (*ordered, EQUILIBRIUM)
    for name in ordered:
        stats = report.policies.get(name)
        if stats is None or not stats.packets:
            continue
        bound = stats.upper_bound(replicates=bootstrap_replicates)
        realized_bound = stats.realized_upper(replicates=bootstrap_replicates)
        # The constraint is declared met only if the *bound* clears it, not the
        # point estimate. That is what the evaluation profile requires, and it
        # is the difference between "we measured 7.5e-5" and "we showed 1e-4".
        holds = "yes" if bound <= budget else "NO"
        shown = f"{realized_bound:>10.3e}" if realized_bound is not None else f"{'--':>10}"
        lines.append(
            f"  {name:<8} {stats.mean_cost:>6.2f} {stats.expected_miss_rate:>10.3e} "
            f"{bound:>10.3e} {holds:>6} {stats.realized_miss_rate:>10.3e} {shown}"
        )
    lines.append(
        "  (E 95%up carries the constraint -- it bounds the expectation the budget "
        "is written against.)"
    )
    lines.append(
        "  (R 95%up is the check that the expectation is not lying; '--' means no "
        "miss was observed, so it bounds nothing.)"
    )

    if report.targeting.packets:
        cost, achieved = report.targeting.solve(budget)
        lines += [
            "",
            f"  targeting bound: mean cost {cost:.3f} at {achieved:.3e} "
            f"-- the cheapest a policy could be if it knew each packet's risk "
            f"but not its outcome",
            f"  risk concentration: the riskiest 1% of packets carry "
            f"{report.targeting.risk_concentration():.1%} of expected RF misses",
            "",
            f"  {'budget':>10} {'cost':>7} {'achieved':>10} {'feasible':>9}",
        ]
        for target, mix_cost, achieved_rate, feasible in report.targeting.frontier(
            frontier_budgets or DEFAULT_FRONTIER_BUDGETS
        ):
            lines.append(
                f"  {target:>10.1e} {mix_cost:>7.3f} {achieved_rate:>10.3e} "
                f"{'yes' if feasible else 'NO':>9}"
            )
    oracle = report.policies.get(ORACLE)
    if oracle and oracle.packets:
        picks = ", ".join(
            f"{action}={oracle.choices.get(action, 0) / oracle.packets:.1%}"
            for action in BASELINES
        )
        lines.append(f"  oracle picks: {picks}")
    return "\n".join(lines)


__all__ = [
    "BASELINES",
    "CAMPAIGN_EQUILIBRIUM_DAMPING",
    "CAMPAIGN_EQUILIBRIUM_MAX_ITERATIONS",
    "EQUILIBRIUM",
    "PREDICTED",
    "DEFAULT_FRONTIER_BUDGETS",
    "ORACLE",
    "DensityReport",
    "PolicyStatistics",
    "TargetingBound",
    "expected_failure_for",
    "format_report",
    "run",
]
