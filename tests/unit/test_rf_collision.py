"""Analytical resource-collision model with declared sensitivity bands.

Two kinds of test here. The first pin the model's own behaviour. The last few
pin what it *implies* for the reliability target, because that implication is
the reason section 7.2 insisted on bands rather than a point estimate -- and
because it decides a service parameter that is currently frozen at the wrong
value.
"""

from __future__ import annotations

import pytest

from hybrid_v2x_rl.channels.rf.collision import (
    MODEL_NAME,
    CollisionError,
    CollisionParameters,
    SensitivityBand,
    channel_busy_ratio,
    collision_probability,
    failure_probability,
    half_duplex_probability,
    headline_parameters,
    hidden_contenders,
    resource_demand,
)

#: Median contenders within 200 m, from
#: ``artifacts/evaluations/geometry-distributions.json``.
#:
#: Copied rather than read, so these tests run without the artifact tree, and
#: guarded by ``test_the_pinned_campaign_figures_still_match_the_artifacts``
#: below -- which is what stops a copy from quietly ageing past the thing it
#: copied. The previous values, 44/100/159, predated the re-run that produced
#: the current campaign and had drifted by up to 8%.
MEASURED_NEIGHBOURS = {10: 47, 20: 108, 30: 162}
#: V-VLC geometric outage from ``campaign-test-wide-1000000.json``, for the
#: comparisons below. The RF conclusions this file draws are insensitive to the
#: exact value -- it enters as a scale factor -- but the absolute magnitudes
#: move with it. The previous values, 0.0959/0.0917/0.0726, were pinned against
#: the superseded test-point headlamp artifact and were already flagged here as
#: superseded; the rebuilt R112-compliant beam and the 0.55 m photodiode roughly
#: doubled them.
MEASURED_VLC_OUTAGE = {10: 0.1560, 20: 0.1462, 30: 0.1187}


# -- scope --------------------------------------------------------------------


def test_the_model_names_itself_and_does_not_claim_mode_two() -> None:
    """Section 7.2 forbids the stronger claim, so the name is a constant."""

    assert MODEL_NAME == "analytical collision model with sensitivity bands"
    assert "mode 2" not in MODEL_NAME.lower()
    assert "nr" not in MODEL_NAME.lower().split()


def test_the_headline_pool_is_one_full_carrier_over_a_hundred_millisecond_window() -> None:
    parameters = headline_parameters()
    assert parameters.subchannels == 1
    assert parameters.selection_window_slots == 200
    assert parameters.candidate_resources == 200


def test_invalid_scopes_are_refused() -> None:
    with pytest.raises(CollisionError, match="subchannel"):
        CollisionParameters(0, 200, 0.85, 1e-3, 0.1)
    with pytest.raises(CollisionError, match="reliability"):
        CollisionParameters(2, 200, 1.5, 1e-3, 0.1)
    with pytest.raises(CollisionError, match="outlast"):
        CollisionParameters(2, 200, 0.85, 0.2, 0.1)


# -- channel busy ratio -------------------------------------------------------


@pytest.mark.parametrize(
    ("density", "expected"), [(10, 0.705), (20, 1.0), (30, 1.0)]
)
def test_channel_busy_ratio_tracks_the_measured_neighbour_counts(
    density: int, expected: float
) -> None:
    """This is the ``rf_channel_busy_ratio`` the observation vector carries.

    Derived from the model rather than measured off a trace, which is what
    keeps M3 independent of a campaign that will be regenerated.
    """

    assert channel_busy_ratio(
        MEASURED_NEIGHBOURS[density], headline_parameters()
    ) == pytest.approx(expected, abs=0.005)


def test_busy_ratio_saturates_rather_than_exceeding_one() -> None:
    assert channel_busy_ratio(100_000, headline_parameters()) == 1.0


def test_a_negative_neighbour_count_is_refused() -> None:
    with pytest.raises(CollisionError, match="negative"):
        channel_busy_ratio(-1, headline_parameters())


# -- contention ---------------------------------------------------------------


def test_perfect_sensing_leaves_no_hidden_contenders() -> None:
    parameters = CollisionParameters(2, 200, 1.0, 1e-3, 0.1)
    assert hidden_contenders(100, 1.0, parameters) == pytest.approx(0.0)
    assert collision_probability(100, parameters) == pytest.approx(0.0)


def test_geometry_and_sensing_reliability_multiply() -> None:
    """Two separate things: where neighbours are, and how well sensing works.

    Halving the share of neighbours whose reservations are decodable must have
    the same effect as halving sensing reliability, because the model treats
    them as independent causes of a blind contender.
    """

    parameters = headline_parameters()
    from_geometry = hidden_contenders(100, 0.5, parameters)
    halved = CollisionParameters(2, 200, parameters.sensing_reliability * 0.5, 1e-3, 0.1)
    from_sensing = hidden_contenders(100, 1.0, halved)
    assert from_geometry == pytest.approx(from_sensing)


def test_collision_rises_with_density() -> None:
    parameters = headline_parameters()
    probabilities = [
        collision_probability(MEASURED_NEIGHBOURS[d], parameters) for d in (10, 20, 30)
    ]
    assert probabilities == sorted(probabilities)


def test_half_duplex_is_the_receiver_duty_cycle() -> None:
    """A floor no resource selection removes, which is a small but genuine
    argument for a second *medium* rather than a second attempt."""

    assert half_duplex_probability(headline_parameters()) == pytest.approx(0.015)


# -- the bands ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("density", "optimistic", "nominal", "pessimistic"),
    [(10, 2.65, 4.92, 8.22), (20, 4.13, 9.18, 16.27), (30, 5.42, 12.80, 22.80)],
)
def test_the_declared_band_at_each_trained_density(
    density: int, optimistic: float, nominal: float, pessimistic: float
) -> None:
    """Access-layer loss, in percent, across the declared uncertainty."""

    neighbours = MEASURED_NEIGHBOURS[density]
    for band, expected in (
        (SensitivityBand.OPTIMISTIC, optimistic),
        (SensitivityBand.NOMINAL, nominal),
        (SensitivityBand.PESSIMISTIC, pessimistic),
    ):
        measured = 100.0 * failure_probability(neighbours, headline_parameters(band))
        assert measured == pytest.approx(expected, abs=0.05)


def test_the_band_is_wide_enough_to_span_the_weaker_link_crossover() -> None:
    """Why bands are mandatory rather than decorative.

    At rho = 30 the optical link is unavailable 11.87% of the time. RF
    access-layer loss is 5.42% optimistic and 22.80% pessimistic, so *which
    medium is the weaker one at the top of the band* is not determined by the
    model -- it is determined by which end of the declared uncertainty is
    taken. Reporting only the nominal would silently pick a side of that
    crossover, which is exactly the failure section 7.2 legislates against.
    """

    optical = MEASURED_VLC_OUTAGE[30]
    neighbours = MEASURED_NEIGHBOURS[30]

    optimistic = failure_probability(neighbours, headline_parameters(SensitivityBand.OPTIMISTIC))
    pessimistic = failure_probability(neighbours, headline_parameters(SensitivityBand.PESSIMISTIC))

    assert optimistic < optical, "optimistic band makes the radio the stronger link"
    assert pessimistic > optical, "pessimistic band makes the radio the weaker link"


def test_collision_and_optical_outage_move_in_opposite_directions() -> None:
    """The complementarity axis, and the reason this mechanism carries RQ4.

    NLOSv is the same geometric event that severs the optical path, so a
    diversity argument built on it is circular. Collision is a headcount, and
    it rises with density while optical outage falls.
    """

    parameters = headline_parameters()
    radio = [collision_probability(MEASURED_NEIGHBOURS[d], parameters) for d in (10, 20, 30)]
    optical = [MEASURED_VLC_OUTAGE[d] for d in (10, 20, 30)]

    assert radio == sorted(radio), "radio must worsen with density"
    assert optical == sorted(optical, reverse=True), "optical must improve with density"


# -- what it implies for the target -------------------------------------------


def test_a_single_rf_attempt_cannot_meet_the_budget_at_any_density() -> None:
    """Collision alone is 150x to 1200x the 10^-4 miss budget.

    And this is before any link-budget BLER is added. A single-attempt RF leg
    is not a candidate for the target at any density or any band.
    """

    budget = 1e-4
    for density in (10, 20, 30):
        for band in SensitivityBand:
            loss = failure_probability(
                MEASURED_NEIGHBOURS[density], headline_parameters(band)
            )
            assert loss > 150 * budget


def test_duplication_needs_more_than_two_reserved_rf_attempts() -> None:
    """The measurement that settles a frozen service parameter.

    With one RF attempt, DUP at rho = 20 nominal gives 9.18% x 14.62% = 1.34e-2,
    which is 134x over budget. A second independently selected time-frequency
    reservation makes the RF access term 8.43e-3 and DUP lands near 1.23e-3.
    Still over.

    Independence across attempts is the optimistic reading, and even it does
    not clear 10^-4 at the nominal band. The conclusion is not that two
    attempts suffice; it is that ``rf_attempts_per_packet: 1`` is certainly
    wrong, and that the budget is tight enough that the link-budget BLER
    arriving next cannot be assumed negligible.
    """

    parameters = headline_parameters()
    optical = MEASURED_VLC_OUTAGE[20]
    single = failure_probability(MEASURED_NEIGHBOURS[20], parameters)

    one_attempt = single * optical
    assert one_attempt == pytest.approx(1.34e-2, rel=0.1)
    assert one_attempt > 1e-4, "one RF attempt cannot reach the budget under DUP"

    two_attempts = single**2 * optical
    assert two_attempts == pytest.approx(1.23e-3, rel=0.15)
    assert two_attempts > 1e-4, "even two attempts do not clear it at nominal"
    assert two_attempts < one_attempt / 10.0, "but a second attempt is worth about 11x"


def test_half_duplex_is_tied_to_the_profile_rather_than_to_a_constant() -> None:
    """The invariant that would have caught a hardcoded airtime.

    ``half_duplex_probability`` is the receiver's own duty cycle, so it must
    equal the airtime the *service profile* commits divided by its generation
    period -- not a number that happens to sit in this module. An earlier
    version hardcoded 1.0 ms while the profile committed three pre-reserved
    0.5 ms attempts, and every test in this file agreed with it because they
    all read the same wrong constant. Deriving the expectation from the
    configuration instead is what makes this test able to fail.
    """

    from pathlib import Path

    from hybrid_v2x_rl.config.loader import headline_config_layers, load_config

    config = load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd())
    committed = config.rf.timing.airtime_s * config.service.rf_attempts_per_packet
    expected = committed / config.service.generation_period_s

    from hybrid_v2x_rl.env.assembly import build_rf_channel

    assert half_duplex_probability(build_rf_channel(config).collision) == pytest.approx(expected)


def test_the_channel_busy_ratio_uses_the_same_committed_airtime() -> None:
    """The observation feature and the half-duplex term share one airtime.

    They are the same physical quantity seen from two sides -- what this
    vehicle occupies, and what its neighbours occupy -- so a profile change
    that moved one without the other would be incoherent.
    """

    from pathlib import Path

    from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
    from hybrid_v2x_rl.env.assembly import build_rf_channel

    config = load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd())
    parameters = build_rf_channel(config).collision
    one_vehicle = channel_busy_ratio(1, parameters) * parameters.subchannels
    assert one_vehicle == pytest.approx(half_duplex_probability(parameters))


# -- resource demand ----------------------------------------------------------


def test_the_busy_ratio_is_the_clipped_resource_demand() -> None:
    """One fact reported two ways, so they cannot drift apart.

    The observation feature has to be a fraction; the physical quantity has no
    such obligation. Deriving one from the other is what stops a future edit
    from making the saturated observation and the oversubscribed pool disagree.
    """

    parameters = headline_parameters()
    for neighbours in (0, 1, 44, 100, 133, 159, 1_000):
        assert channel_busy_ratio(neighbours, parameters) == pytest.approx(
            min(1.0, resource_demand(neighbours, parameters))
        )


def test_resource_demand_exposes_what_the_busy_ratio_saturates_away() -> None:
    """Above one the pool is oversubscribed, and the clip hides exactly that."""

    parameters = headline_parameters()
    assert channel_busy_ratio(100_000, parameters) == 1.0
    assert resource_demand(100_000, parameters) > 1.0


def test_the_headline_profile_oversubscribes_the_pool_at_the_densest_condition() -> None:
    """Three attempts ask for more airtime than rho = 30 has to give.

    The pool supplies ``subchannels * generation_period`` of airtime per period,
    which under one full carrier and 0.5 ms slots is 200 resources. With 162
    measured contenders each committing three, demand is 2.43 -- the profile is
    not deliverable, and no collision probability says so, because the birthday
    model asks where one selection lands rather than whether every selection can
    be honoured.

    This is a property of the *frozen* profile, recorded here so the arithmetic
    is a measured fact rather than a claim in a commit message.
    """

    parameters = headline_parameters()

    assert resource_demand(MEASURED_NEIGHBOURS[10], parameters) < 1.0
    assert resource_demand(MEASURED_NEIGHBOURS[30], parameters) > 1.0
    assert resource_demand(MEASURED_NEIGHBOURS[30], parameters) == pytest.approx(
        162 * 3 / 200.0
    )

    # And the collision model is untroubled by it, which is the point.
    assert failure_probability(MEASURED_NEIGHBOURS[30], parameters) < 0.15


def test_resource_demand_counts_attempts_against_the_candidate_pool() -> None:
    """The airtime form and the counting form are the same number.

    ``resource_demand`` is written in airtime because that needs no slot
    duration. Where the selection window spans the generation period -- which
    the headline profile's 200 slots at 0.5 ms do exactly -- it must agree with
    ``neighbours * attempts / candidate_resources``, and this pins the two.
    """

    slot_s = 0.0005
    for attempts in (1, 3, 5):
        parameters = headline_parameters(committed_airtime_s=attempts * slot_s)
        assert parameters.selection_window_slots * slot_s == pytest.approx(
            parameters.generation_period_s
        )
        for neighbours in (44, 100, 159):
            assert resource_demand(neighbours, parameters) == pytest.approx(
                neighbours * attempts / parameters.candidate_resources
            )


def test_resource_demand_rejects_a_negative_headcount() -> None:
    with pytest.raises(CollisionError):
        resource_demand(-1, headline_parameters())


def test_the_pinned_campaign_figures_still_match_the_artifacts() -> None:
    """The guard that keeps a copied measurement from ageing past its source.

    ``MEASURED_NEIGHBOURS`` and ``MEASURED_VLC_OUTAGE`` are copied into this
    file so the tests run without the artifact tree. That copy is only safe
    while it agrees with what the campaign actually measured, and the previous
    values had drifted 8% and a factor of two respectively without anything
    noticing.
    """

    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "artifacts" / "evaluations"
    geometry = root / "geometry-distributions.json"
    campaign_file = root / "campaign-test-wide-1000000.json"
    if not (geometry.exists() and campaign_file.exists()):
        pytest.skip("evaluation artifacts are not present in this checkout")

    measured = json.loads(geometry.read_text())["densities"]
    for density, pinned in MEASURED_NEIGHBOURS.items():
        actual = measured[str(density)]["contenders"]["quantiles"]["50"]
        assert round(actual) == pinned

    blocks = json.loads(campaign_file.read_text())["densities"]
    for block in blocks:
        density = int(block["density_veh_per_lane_km"])
        assert block["optical"]["geometric_outage"] == pytest.approx(
            MEASURED_VLC_OUTAGE[density], abs=5e-4
        )
