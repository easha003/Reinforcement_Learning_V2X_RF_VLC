"""Work plan §6.2: the predictor emits a probability, not a future truth."""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.observation.blockage import (
    BlockageForecast,
    BlockerShape,
    blockage_probability,
)
from hybrid_v2x_rl.observation.forecast import PredictedState

EAST = 0.0
NORTH = 0.5 * math.pi

CAR = BlockerShape(length_m=4.5, width_m=1.8, height_m=1.5)
BUS = BlockerShape(length_m=11.0, width_m=2.5, height_m=3.25)
LOW = BlockerShape(length_m=4.5, width_m=1.8, height_m=0.4)


def state(
    vehicle_id: str,
    x_m: float,
    y_m: float,
    *,
    heading_rad: float = EAST,
    std_m: float = 0.5,
    propagated_s: float = 0.25,
) -> PredictedState:
    return PredictedState(
        vehicle_id=vehicle_id,
        at_s=1.25,
        x_m=x_m,
        y_m=y_m,
        heading_rad=heading_rad,
        speed_mps=10.0,
        propagated_s=propagated_s,
        along_track_std_m=std_m,
        cross_track_std_m=std_m,
        base_std_m=0.5,
    )


def predict(blockers, tx=None, rx=None, **kwargs):
    return blockage_probability(
        tx or state("tx", 0.0, 0.0),
        rx or state("rx", 20.0, 0.0),
        blockers,
        horizon_s=0.2,
        **kwargs,
    )


# -- it is a probability, and it lives strictly inside the unit interval ------


def test_a_clear_path_is_near_zero() -> None:
    far = state("blk", 10.0, 60.0, heading_rad=NORTH)
    assert predict([(far, CAR)]).probability < 1e-6


def test_a_body_squarely_across_the_path_is_near_one() -> None:
    across = state("blk", 10.0, 0.0, heading_rad=NORTH)
    assert predict([(across, CAR)]).probability > 0.99


def test_a_grazing_body_is_neither_zero_nor_one() -> None:
    """The point of a probability: near-misses are not decided, they are priced."""

    grazing = state("blk", 10.0, 1.4, heading_rad=NORTH)
    probability = predict([(grazing, CAR)]).probability

    assert 0.05 < probability < 0.95


def test_probability_falls_monotonically_with_lateral_offset() -> None:
    values = [
        predict([(state("blk", 10.0, offset, heading_rad=NORTH), CAR)]).probability
        for offset in (0.0, 1.0, 2.0, 3.0, 5.0, 10.0)
    ]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("offset", [0.0, 1.0, 2.5, 4.0, 20.0])
def test_probability_stays_in_the_unit_interval(offset: float) -> None:
    probability = predict([(state("blk", 10.0, offset, heading_rad=NORTH), CAR)]).probability
    assert 0.0 <= probability <= 1.0


def test_an_out_of_range_probability_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="probability"):
        BlockageForecast(probability=1.5, confidence=1.0, horizon_s=0.2, considered=1)
    with pytest.raises(ValueError, match="confidence"):
        BlockageForecast(probability=0.5, confidence=-0.1, horizon_s=0.2, considered=1)


# -- uncertainty is what turns geometry into probability ----------------------


def test_a_more_uncertain_blocker_is_pulled_towards_one_half() -> None:
    """Uncertainty must blur the verdict in both directions.

    A confident near-miss is nearly zero; the same near-miss seen through a
    poor track cannot stay nearly zero, or the noise model has not reached the
    feature that RQ3 depends on.
    """

    near_miss_confident = predict(
        [(state("blk", 10.0, 3.0, heading_rad=NORTH, std_m=0.3), CAR)]
    ).probability
    near_miss_vague = predict(
        [(state("blk", 10.0, 3.0, heading_rad=NORTH, std_m=3.0), CAR)]
    ).probability

    assert near_miss_vague > near_miss_confident

    hit_confident = predict(
        [(state("blk", 10.0, 0.0, heading_rad=NORTH, std_m=0.3), CAR)]
    ).probability
    hit_vague = predict(
        [(state("blk", 10.0, 0.0, heading_rad=NORTH, std_m=3.0), CAR)]
    ).probability

    assert hit_vague < hit_confident


def test_endpoint_uncertainty_reaches_the_verdict() -> None:
    """A link whose ends are poorly known is itself poorly located."""

    blocker = state("blk", 10.0, 3.0, heading_rad=NORTH, std_m=0.3)
    tight = predict(
        [(blocker, CAR)],
        tx=state("tx", 0.0, 0.0, std_m=0.1),
        rx=state("rx", 20.0, 0.0, std_m=0.1),
    ).probability
    loose = predict(
        [(blocker, CAR)],
        tx=state("tx", 0.0, 0.0, std_m=4.0),
        rx=state("rx", 20.0, 0.0, std_m=4.0),
    ).probability

    assert loose > tight


def test_confidence_is_the_weaker_of_the_two_endpoints() -> None:
    forecast = predict(
        [],
        tx=state("tx", 0.0, 0.0, std_m=0.5),
        rx=state("rx", 20.0, 0.0, std_m=4.0),
    )
    assert forecast.confidence == pytest.approx(0.5 / 4.0)


# -- geometry the engine already relies on ------------------------------------


def test_a_crossing_vehicle_presents_its_length_not_its_width() -> None:
    """The correction that superseded the point-with-tolerance approximation.

    A body crossing perpendicular blocks over its 4.5 m length, not its 1.8 m
    width, and understating that understated cross-traffic blockage.
    """

    crossing = state("blk", 10.0, 2.0, heading_rad=NORTH, std_m=0.01)
    aligned = state("blk", 10.0, 2.0, heading_rad=EAST, std_m=0.01)

    assert predict([(crossing, CAR)]).probability > predict([(aligned, CAR)]).probability


def test_a_longer_body_blocks_a_wider_band() -> None:
    offset = 4.0
    car = predict([(state("b", 10.0, offset, heading_rad=NORTH, std_m=0.01), CAR)])
    bus = predict([(state("b", 10.0, offset, heading_rad=NORTH, std_m=0.01), BUS)])

    assert bus.probability > car.probability


def test_a_body_shorter_than_the_beam_passes_under_it() -> None:
    under = state("blk", 10.0, 0.0, heading_rad=NORTH, std_m=0.01)
    assert predict([(under, LOW)]).probability == pytest.approx(0.0)
    assert predict([(under, CAR)]).probability > 0.99


def test_a_body_beyond_the_endpoints_does_not_block() -> None:
    beyond = state("blk", 60.0, 0.0, heading_rad=NORTH, std_m=0.01)
    behind = state("blk", -40.0, 0.0, heading_rad=NORTH, std_m=0.01)

    assert predict([(beyond, CAR)]).probability < 1e-6
    assert predict([(behind, CAR)]).probability < 1e-6


def test_the_endpoints_never_block_their_own_link() -> None:
    tx = state("tx", 0.0, 0.0)
    rx = state("rx", 20.0, 0.0)
    assert predict([(tx, CAR), (rx, CAR)], tx=tx, rx=rx).probability == pytest.approx(0.0)


def test_a_degenerate_path_yields_no_forecast_rather_than_dividing_by_zero() -> None:
    coincident = state("rx", 0.0, 0.0)
    forecast = predict([(state("b", 0.0, 0.0), CAR)], rx=coincident)

    assert forecast.probability == 0.0
    assert forecast.considered == 0


# -- combining blockers -------------------------------------------------------


def test_two_blockers_are_at_least_as_bad_as_either_alone() -> None:
    first = (state("b1", 6.0, 1.5, heading_rad=NORTH), CAR)
    second = (state("b2", 14.0, 1.5, heading_rad=NORTH), CAR)

    alone = predict([first]).probability
    both = predict([first, second]).probability

    assert both >= alone
    assert both <= 1.0


def test_every_candidate_is_counted_even_when_it_cannot_block() -> None:
    forecast = predict(
        [
            (state("near", 10.0, 0.0, heading_rad=NORTH), CAR),
            (state("far", 10.0, 80.0, heading_rad=NORTH), CAR),
        ]
    )
    assert forecast.considered == 2


def test_no_candidates_means_a_clear_path() -> None:
    forecast = predict([])
    assert forecast.probability == 0.0
    assert forecast.considered == 0


def test_the_horizon_is_reported_back() -> None:
    assert predict([]).horizon_s == pytest.approx(0.2)


# -- the support function -----------------------------------------------------


@pytest.mark.parametrize(
    "heading,bearing,expected",
    [
        (EAST, EAST, 2.25),
        (EAST, NORTH, 0.9),
        (NORTH, EAST, 0.9),
        (NORTH, NORTH, 2.25),
    ],
)
def test_half_extent_matches_the_rectangle(
    heading: float, bearing: float, expected: float
) -> None:
    assert CAR.half_extent_towards(heading, bearing) == pytest.approx(expected, abs=1e-9)


def test_half_extent_is_largest_along_the_diagonal() -> None:
    diagonal = CAR.half_extent_towards(EAST, math.atan2(1.8, 4.5))
    assert diagonal > CAR.half_extent_towards(EAST, EAST)
