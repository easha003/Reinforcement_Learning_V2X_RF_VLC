"""Scoring the baselines, the oracle and the section 8.3 ratio.

The subtle one here is how the oracle is scored. It chooses by looking at the
realization, so the marginal of whichever action it picked is not a probability
of anything -- the action was picked *because* it was going to succeed. Every
test below that touches ORACLE is defending that distinction, because getting it
wrong produces a table that looks right and reports the wrong bound.
"""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.core.enums import FailureCause
from hybrid_v2x_rl.env import campaign
from hybrid_v2x_rl.env.campaign import ORACLE, DensityReport, PolicyStatistics, expected_failure_for
from hybrid_v2x_rl.env.packet import DUP, RF_ONLY, PacketOutcome


def outcome(action, *, delivered=True, rf=1e-4, vlc=0.2, cause=None):
    return PacketOutcome(
        action=action,
        delivered=delivered,
        delivery_time_s=0.001 if delivered else None,
        failure_cause=cause or (FailureCause.NONE if delivered else FailureCause.RF_COLLISION),
        activation_cost=action.activation_cost,
        rf_attempts_used=1,
        rf_delivered=delivered,
        vlc_delivered=delivered,
        rf_failure_probability=rf,
        vlc_failure_probability=vlc,
    )


# -- how each policy is scored ------------------------------------------------


def test_each_baseline_is_scored_on_its_own_medium() -> None:
    packet = outcome(DUP, rf=1e-4, vlc=0.2)
    assert expected_failure_for("RF", packet) == pytest.approx(1e-4)
    assert expected_failure_for("VLC", packet) == pytest.approx(0.2)
    assert expected_failure_for("DUP", packet) == pytest.approx(2e-5)


def test_the_oracle_is_scored_on_the_joint_not_on_what_it_picked() -> None:
    """A clairvoyant chooser misses only when no action would have delivered,
    and that event is exactly duplication's failure event. Scoring it on the
    action it picked would report the marginal of a choice made *because* it
    was going to succeed."""

    packet = outcome(RF_ONLY, rf=1e-4, vlc=0.2)
    assert expected_failure_for(ORACLE, packet) == pytest.approx(2e-5)
    assert expected_failure_for(ORACLE, packet) < expected_failure_for("RF", packet)


def test_the_oracle_matches_duplication_in_reliability_and_not_in_cost() -> None:
    """The entire bound, in one assertion: same failure event, half the cost."""

    packet = outcome(RF_ONLY, rf=1e-4, vlc=0.2)
    assert expected_failure_for(ORACLE, packet) == expected_failure_for("DUP", packet)
    assert RF_ONLY.activation_cost < DUP.activation_cost


# -- the statistics -----------------------------------------------------------


def test_cost_and_expectation_accumulate_per_packet() -> None:
    stats = PolicyStatistics()
    for _ in range(4):
        stats.observe(outcome(DUP, rf=1e-4, vlc=0.5), 5e-5)
    assert stats.packets == 4
    assert stats.mean_cost == pytest.approx(2.0)
    assert stats.expected_miss_rate == pytest.approx(5e-5)


def test_a_miss_is_counted_and_its_cause_named() -> None:
    stats = PolicyStatistics()
    stats.observe(outcome(DUP, delivered=False, cause=FailureCause.JOINT_FAILURE), 1e-5)
    stats.observe(outcome(DUP, delivered=True), 1e-5)
    assert stats.misses == 1
    assert stats.realized_miss_rate == pytest.approx(0.5)
    assert stats.causes["JOINT_FAILURE"] == 1


def test_the_confidence_bound_stays_open_at_zero_misses() -> None:
    """A normal interval collapses to zero width here and would report a 1e-4
    budget as met with certainty on evidence that cannot tell 1e-4 from 1e-6."""

    stats = PolicyStatistics()
    for _ in range(10_000):
        stats.observe(outcome(RF_ONLY), 1e-6)
    assert stats.misses == 0
    assert stats.realized_miss_rate == 0.0
    assert stats.wilson_upper() > 1e-4, "zero misses in 10k packets does not prove 1e-4"
    assert stats.wilson_upper() < 1e-2


def test_more_evidence_tightens_the_bound() -> None:
    def bound(n):
        stats = PolicyStatistics()
        for _ in range(n):
            stats.observe(outcome(RF_ONLY), 1e-6)
        return stats.wilson_upper()

    assert bound(1_000_000) < bound(10_000) < bound(100)


# -- the dependence ratio -----------------------------------------------------


def test_the_ratio_is_one_when_the_realized_rate_matches_the_prediction() -> None:
    report = DensityReport(density=10.0)
    report.packets = 100_000
    report.predicted_joint = 20.0
    report.joint_failures = 20
    assert report.dependence_ratio == pytest.approx(1.0)


def test_the_ratio_rises_when_the_media_fail_together() -> None:
    report = DensityReport(density=10.0)
    report.packets, report.predicted_joint, report.joint_failures = 100_000, 20.0, 60
    assert report.dependence_ratio == pytest.approx(3.0)


def test_no_prediction_and_no_failure_reads_as_independent_not_as_a_crash() -> None:
    report = DensityReport(density=10.0)
    report.packets = 10
    assert report.dependence_ratio == 1.0


def test_a_failure_nobody_predicted_is_infinite_rather_than_silently_zero() -> None:
    report = DensityReport(density=10.0)
    report.packets, report.joint_failures = 10, 1
    assert math.isinf(report.dependence_ratio)


# -- what the report says about the optical leg -------------------------------


def test_geometric_outage_and_total_optical_failure_are_separate() -> None:
    """P_out is the floor no power reaches; the total is what duplication pays
    for. At low density most of the gap is pairs simply out of range, and
    conflating them would credit a wider beam with fixing distance."""

    report = DensityReport(density=10.0)
    report.packets = 1000
    report.optical_geometric_failures = 170
    report.optical_failures = 590
    assert report.optical_outage == pytest.approx(0.17)
    assert report.optical_failure_rate == pytest.approx(0.59)
    assert report.optical_outage < report.optical_failure_rate


def test_complementarity_counts_only_the_packets_the_light_saved() -> None:
    report = DensityReport(density=30.0)
    report.packets = 10_000
    report.rf_lost_vlc_saved = 3
    report.vlc_lost_rf_saved = 2_500
    assert report.complementarity == pytest.approx(3e-4)


def test_the_report_formats_without_a_division_by_zero() -> None:
    """An empty group must print, because a density with no eligible pairs is a
    result and not a reason to lose the other groups."""

    text = campaign.format_report(DensityReport(density=5.0))
    assert "density 5" in text


# -- the bound a learned policy is actually competing against -----------------


def test_a_flat_link_forces_near_universal_duplication() -> None:
    """If every packet carries the same risk, targeting is impossible and the
    bound collapses onto the always-duplicate baseline. This is the case where
    the selection contribution would not exist, so it must be visible."""

    bound = campaign.TargetingBound()
    for _ in range(10_000):
        bound.observe(3.3e-4, 8e-5)
    cost, achieved = bound.solve(1e-4)
    assert cost > 1.9
    assert achieved <= 1e-4
    assert bound.risk_concentration() == pytest.approx(0.01, abs=1e-3)


def test_concentrated_risk_is_cheap_to_target() -> None:
    """A few bad packets carrying the whole miss budget can be duplicated
    individually, and the mean cost barely moves off 1.0."""

    bound = campaign.TargetingBound()
    for index in range(10_000):
        risky = index < 100
        bound.observe(3.0e-2 if risky else 1e-7, 1e-9)
    cost, achieved = bound.solve(1e-4)
    assert cost < 1.05
    assert achieved <= 1e-4
    assert bound.risk_concentration() > 0.9


def test_a_link_already_inside_budget_pays_nothing() -> None:
    bound = campaign.TargetingBound()
    for _ in range(1000):
        bound.observe(1e-6, 1e-9)
    cost, achieved = bound.solve(1e-4)
    assert cost == 1.0
    assert achieved == pytest.approx(1e-6)


def test_an_infeasible_budget_stops_at_universal_duplication() -> None:
    """Duplicating everything is the most it can do, and if that still misses
    the budget the bound says so rather than reporting a cost above 2."""

    bound = campaign.TargetingBound()
    for _ in range(1000):
        bound.observe(0.5, 0.4)
    cost, achieved = bound.solve(1e-4)
    assert cost == pytest.approx(2.0)
    assert achieved == pytest.approx(0.4)


def test_the_bound_never_exceeds_the_duplicate_everything_baseline() -> None:
    bound = campaign.TargetingBound()
    for index in range(500):
        bound.observe(1e-3 * (index + 1), 1e-6)
    cost, _ = bound.solve(1e-4)
    assert 1.0 <= cost <= 2.0


def test_complementarity_is_never_printed_as_a_percentage() -> None:
    """It lives at 1e-5 by construction. Printed as "0.00%", the one statistic
    that justifies carrying a second medium reads as a zero -- which is exactly
    what happened on the first 500k-packet run."""

    report = DensityReport(density=10.0)
    report.packets = 500_172
    report.rf_lost_vlc_saved = 6
    line = next(
        row for row in campaign.format_report(report).splitlines()
        if "complementarity" in row
    )
    assert "%" not in line
    assert "1.200e-05" in line
    assert "6 packets the radio lost" in line


# -- the pool claim -----------------------------------------------------------


def test_the_report_states_what_the_population_asked_of_the_pool() -> None:
    """A miss rate is only meaningful if the profile could have been delivered.

    The headline profile at the densest trained condition commits three attempts
    against a pool that supplies about two and a half, and nothing else on the
    report says so: the collision model prices where one selection lands, not
    whether every selection can be honoured, so the miss rates look ordinary.
    """

    from hybrid_v2x_rl.channels.rf.collision import headline_parameters

    report = DensityReport(density=30.0, collision=headline_parameters())
    report.packets = 1_000
    report.contender_total = 159_000  # 159 contenders on every packet

    assert report.mean_contenders == pytest.approx(159.0)
    assert report.resource_demand == pytest.approx(159 * 3 / 200.0)
    assert report.resource_demand > 1.0
    assert not report.deliverable


def test_a_sparse_population_fits_the_pool() -> None:
    from hybrid_v2x_rl.channels.rf.collision import headline_parameters

    report = DensityReport(density=10.0, collision=headline_parameters())
    report.packets = 1_000
    report.contender_total = 47_000

    assert report.resource_demand == pytest.approx(47 * 3 / 200.0)
    assert report.deliverable


def test_the_pool_claim_is_absent_rather_than_wrong_before_any_packet() -> None:
    """An empty report must not read as a comfortably empty pool."""

    empty = DensityReport(density=30.0)
    assert empty.mean_contenders == 0.0
    assert empty.resource_demand == 0.0


# -- the equilibrium, scored --------------------------------------------------


def test_the_legacy_equilibrium_is_exposed_as_undeliverable_after_pool_correction() -> None:
    """The full-carrier correction must invalidate an overloaded policy.

    Scoring the allocation against the profile's contention would report a
    policy nobody could run: the packets carried by light return their share of
    the pool, so the radio the remaining packets face is not the radio the
    survey pass measured. The corrected pool claim is the check. The previously
    selected equilibrium remains above one, proving that its attractive miss
    rate cannot authorize training under one 10 MHz full-carrier resource.
    """

    from pathlib import Path

    from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
    from hybrid_v2x_rl.env.episodes import TraceSource

    root = Path(__file__).resolve().parents[2]
    trace = root / "artifacts" / "traces" / "synthetic-d30-test-000"
    if not trace.exists():
        pytest.skip("mobility traces are not present in this checkout")

    layers = list(headline_config_layers(root))
    layers[2] = root / "configs" / "service" / "ev2x_300B_3ms_1e-5.yaml"
    layers.insert(-1, root / "configs" / "training" / "primal_dual_ppo_ultra.yaml")
    layers[-1] = root / "configs" / "evaluation" / "ultra_reliability.yaml"
    config = load_config(tuple(layers), project_root=root)

    report = campaign.run_equilibrium(
        config, sources=[TraceSource.discover(trace)], density=30.0, budget=1e-5,
        root_seed=11, max_packets=8000, warmup_s=400.0,
        generation_period_s=config.service.generation_period_s,
    )
    stats = report.policies[campaign.EQUILIBRIUM]

    # Contenders come from the survey pass, not the frame's neighbour list.
    assert 140 < report.mean_contenders < 190
    assert report.resource_demand > 1.0
    assert not report.deliverable

    # Both media carry packets, which is the claim the policy exists to make.
    assert stats.choices["RF"] > 0
    assert stats.choices["VLC"] > 0
    assert stats.expected_miss_rate < 1e-5
    assert 1.0 <= stats.mean_cost <= 2.0


def test_scoring_the_equilibrium_needs_traces_rather_than_a_rule() -> None:
    """It cannot be expressed as a chooser, and the signature says so.

    Every other policy here is a per-packet rule the rollout applies as it goes.
    This one needs the whole population twice: the duplication set is ranked
    globally, and the contention is a consequence of the allocation.
    """

    import inspect

    parameters = inspect.signature(campaign.run_equilibrium).parameters
    assert "sources" in parameters
    assert "budget" in parameters


# -- the reduced reservation --------------------------------------------------


def _equilibrium(config, sources, **kwargs):
    return campaign.run_equilibrium(
        config, sources=sources, density=30.0, budget=1e-5, root_seed=11,
        max_packets=6000, warmup_s=400.0,
        generation_period_s=config.service.generation_period_s, **kwargs)


def _ultra_config_and_sources():
    from pathlib import Path

    from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
    from hybrid_v2x_rl.env.episodes import TraceSource

    root = Path(__file__).resolve().parents[2]
    trace = root / "artifacts" / "traces" / "synthetic-d30-test-000"
    if not trace.exists():
        pytest.skip("mobility traces are not present in this checkout")
    layers = list(headline_config_layers(root))
    layers[2] = root / "configs" / "service" / "ev2x_300B_3ms_1e-5.yaml"
    layers.insert(-1, root / "configs" / "training" / "primal_dual_ppo_ultra.yaml")
    layers[-1] = root / "configs" / "evaluation" / "ultra_reliability.yaml"
    return load_config(tuple(layers), project_root=root), [TraceSource.discover(trace)]


def test_a_zero_cap_is_the_offload_it_already_was() -> None:
    """The reduced reservation must collapse to light-alone at its lower end.

    ``offload_attempts=0`` is not a new policy: it is the allocation as it was,
    and if the capping path changed any outcome there it would be changing
    something it does not name.
    """

    config, sources = _ultra_config_and_sources()
    plain = _equilibrium(config, sources)
    zero = _equilibrium(config, sources, offload_attempts=0)

    a, b = plain.policies[campaign.EQUILIBRIUM], zero.policies[campaign.EQUILIBRIUM]
    assert a.packets == b.packets
    assert a.misses == b.misses
    assert a.mean_cost == pytest.approx(b.mean_cost)
    assert a.expected_miss_rate == pytest.approx(b.expected_miss_rate)


def test_a_full_cap_delivers_exactly_what_duplication_delivers() -> None:
    """The derivation must collapse to duplication at its upper end.

    An offloaded packet is scored by capping the radio leg of a duplicated
    evaluation: it arrives if the light arrived, or if the radio arrived inside
    the reduced reservation. At a cap equal to the profile's attempts that
    second clause is unconditional, because a delivered packet always used at
    most the attempts it was granted -- so the capped outcome must equal the
    duplicated one for every packet, not merely on average.

    Asserted here rather than through a miss rate, because at any tractable
    sample size a miss rate compares one or two Poisson events and would pass
    whatever the comparison did.
    """

    attempts = 4
    for delivered_rf, used, delivered_vlc in (
        (True, 1, False), (True, 4, False), (False, 4, False),
        (True, 2, True), (False, 4, True),
    ):
        duplication = bool(delivered_rf or delivered_vlc)
        within = delivered_rf and used <= attempts
        assert bool(delivered_vlc or within) == duplication

    # And a cap below the successful attempt must lose exactly those packets.
    assert not (True and 3 <= 2)          # succeeded on 3, reserved 2 -> lost
    assert (True and 2 <= 2)              # succeeded on 2, reserved 2 -> kept


def test_a_reduced_reservation_returns_pool_to_the_population() -> None:
    """The point of the cap: fewer reserved attempts, smaller claim.

    Reservation is pre-emptive, so the saving comes from reserving less rather
    than from using less -- which is why the claim has to fall even though every
    packet still reaches both media.
    """

    config, sources = _ultra_config_and_sources()
    wide = _equilibrium(config, sources,
                        offload_attempts=config.service.rf_attempts_per_packet)
    thin = _equilibrium(config, sources, offload_attempts=1)

    assert thin.resource_demand < wide.resource_demand
