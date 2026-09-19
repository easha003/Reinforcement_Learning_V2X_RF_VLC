"""Work plan §6.1: the vector a deployable policy receives."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import Action, Link
from hybrid_v2x_rl.core.geometry import Point, Segment
from hybrid_v2x_rl.core.intersection_context import (
    IntersectionContext,
    JunctionGrid,
    intersection_context,
)
from hybrid_v2x_rl.observation.blockage import BlockageForecast
from hybrid_v2x_rl.observation.builder import (
    UNMEASURED_AGE_S,
    UNMEASURED_QUALITY,
    ObservationBuilder,
    ObservationSchema,
    PairObservationInputs,
)
from hybrid_v2x_rl.observation.forecast import PredictedState
from hybrid_v2x_rl.observation.link_state import LinkStateTracker
from hybrid_v2x_rl.observation.sensing import TrackSample
from hybrid_v2x_rl.observation.tracks import Track

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EAST = 0.0
NORTH = 0.5 * math.pi


def predicted(vehicle_id: str, x_m: float, y_m: float, *, heading_rad: float = EAST,
              speed_mps: float = 10.0) -> PredictedState:
    return PredictedState(
        vehicle_id=vehicle_id, at_s=1.25, x_m=x_m, y_m=y_m, heading_rad=heading_rad,
        speed_mps=speed_mps, propagated_s=0.25, along_track_std_m=0.5,
        cross_track_std_m=0.5, base_std_m=0.5,
    )


def tracked(vehicle_id: str, measured_at_s: float = 1.0) -> Track:
    return Track(
        sample=TrackSample(vehicle_id=vehicle_id, measured_at_s=measured_at_s,
                           observed_at_s=measured_at_s + 0.05, x_m=0.0, y_m=0.0,
                           speed_mps=10.0, heading_rad=EAST),
        first_seen_s=measured_at_s,
        update_count=1,
    )



def grid() -> JunctionGrid:
    return JunctionGrid.from_grid(
        avenues=6, cross_streets=12, avenue_spacing_m=244.0,
        cross_street_spacing_m=61.0, road_half_width_m=3.5,
    )


def context(tx: PredictedState, rx: PredictedState) -> IntersectionContext:
    return intersection_context(
        tx.x_m, tx.y_m, tx.heading_rad, rx.x_m, rx.y_m, rx.heading_rad,
        Segment(Point(tx.x_m, tx.y_m), Point(rx.x_m, rx.y_m)), grid(),
    )

def inputs(**overrides) -> PairObservationInputs:
    base = dict(
        now_s=1.05,
        transmitter=predicted("tx", 0.0, 0.0),
        receiver=predicted("rx", 20.0, 0.0),
        transmitter_track=tracked("tx"),
        receiver_track=tracked("rx"),
        blockage=BlockageForecast(probability=0.25, confidence=0.8, horizon_s=0.2, considered=3),
        links=LinkStateTracker(history_packets=8),
        neighbour_count=42,
        channel_busy_ratio=0.3,
        fov_half_angle_rad=math.radians(60.0),
        intersection=context(predicted("tx", 0.0, 0.0), predicted("rx", 20.0, 0.0)),
    )
    base.update(overrides)
    return PairObservationInputs(**base)  # type: ignore[arg-type]


def headline_builder() -> ObservationBuilder:
    config = load_headline_config(PROJECT_ROOT)
    return ObservationBuilder.from_config(config.observation)


# -- the schema is a contract -------------------------------------------------


def test_the_headline_schema_matches_the_configured_features() -> None:
    config = load_headline_config(PROJECT_ROOT)
    schema = headline_builder().schema

    assert schema.features == tuple(config.observation.features)
    assert schema.history_packets == config.observation.history_packets


def test_histories_expand_to_one_column_each_and_the_rest_do_not() -> None:
    schema = headline_builder().schema
    columns = schema.columns

    assert "rf_quality_history[0]" in columns
    assert "rf_quality_history[7]" in columns
    assert "rf_quality" in columns
    # 21 named features, two of which expand to eight columns each.
    scalars = len(schema.features) - 2
    assert schema.width == scalars + 2 * schema.history_packets == len(columns)
    assert schema.width == 35


def test_the_vector_width_matches_the_schema() -> None:
    builder = headline_builder()
    assert len(builder.build(inputs())) == builder.schema.width


def test_column_order_follows_configuration_not_construction_order() -> None:
    """A silent reordering leaves a checkpoint reading the wrong columns."""

    reversed_features = tuple(reversed(headline_builder().schema.features))
    builder = ObservationBuilder(
        schema=ObservationSchema(features=reversed_features, history_packets=8)
    )

    forward = headline_builder().build(inputs())
    backward = builder.build(inputs())

    assert backward[-1] == pytest.approx(forward[0])
    assert len(backward) == len(forward)


def test_a_configured_feature_with_no_implementation_is_refused() -> None:
    builder = ObservationBuilder(
        schema=ObservationSchema(features=("rf_quality", "invented_feature"), history_packets=8)
    )
    with pytest.raises(KeyError, match="invented_feature"):
        builder.build(inputs())


# -- absent is encoded, not defaulted -----------------------------------------


def test_an_unmeasured_leg_is_marked_rather_than_given_a_plausible_age() -> None:
    """"Never measured" and "very stale" are different states.

    Collapsing them onto one large number would let the policy treat them
    alike, which is exactly the distinction §6.3 makes actionable.
    """

    builder = headline_builder()
    columns = builder.schema.columns
    vector = builder.build(inputs())

    assert vector[columns.index("rf_quality_age")] == UNMEASURED_AGE_S
    assert vector[columns.index("rf_quality")] == UNMEASURED_QUALITY
    assert UNMEASURED_AGE_S < 0.0, "no real age can be negative"


def test_a_measured_leg_reports_a_real_age() -> None:
    links = LinkStateTracker(history_packets=8)
    links.record(action=Action.RF, at_s=0.55, delivered=True, measurements={Link.RF: 3.0})

    builder = headline_builder()
    columns = builder.schema.columns
    vector = builder.build(inputs(links=links))

    assert vector[columns.index("rf_quality")] == pytest.approx(3.0)
    assert vector[columns.index("rf_quality_age")] == pytest.approx(0.5)
    assert vector[columns.index("vlc_quality_age")] == UNMEASURED_AGE_S


def test_the_first_packet_still_produces_a_full_width_vector() -> None:
    """History is padded, so the policy has fixed input from packet one."""

    builder = headline_builder()
    vector = builder.build(inputs())

    assert len(vector) == builder.schema.width
    assert all(math.isfinite(value) for value in vector)


def test_an_absent_previous_action_is_distinguishable_from_action_rf() -> None:
    """Action.RF is 0, so a zero default would look like a real first action."""

    builder = headline_builder()
    columns = builder.schema.columns

    fresh = builder.build(inputs())[columns.index("previous_action")]
    assert fresh == -1.0

    links = LinkStateTracker(history_packets=8)
    links.record(action=Action.RF, at_s=1.0, delivered=True, measurements={Link.RF: 1.0})
    used_rf = builder.build(inputs(links=links))[columns.index("previous_action")]
    assert used_rf == float(Action.RF) == 0.0
    assert used_rf != fresh


# -- geometry features --------------------------------------------------------


def test_pair_distance_is_the_predicted_separation() -> None:
    builder = headline_builder()
    columns = builder.schema.columns
    vector = builder.build(inputs())

    assert vector[columns.index("pair_distance")] == pytest.approx(20.0)


def test_bearing_is_relative_so_identical_geometry_reads_the_same_on_any_street() -> None:
    """An absolute bearing would make a northbound pair look different."""

    builder = headline_builder()
    index = builder.schema.columns.index("pair_bearing")

    eastbound = builder.build(
        inputs(transmitter=predicted("tx", 0.0, 0.0, heading_rad=EAST),
               receiver=predicted("rx", 20.0, 0.0, heading_rad=EAST))
    )[index]
    northbound = builder.build(
        inputs(transmitter=predicted("tx", 0.0, 0.0, heading_rad=NORTH),
               receiver=predicted("rx", 0.0, 20.0, heading_rad=NORTH))
    )[index]

    assert eastbound == pytest.approx(northbound, abs=1e-9)
    assert eastbound == pytest.approx(0.0, abs=1e-9)


def test_fov_margin_is_positive_when_aligned_and_negative_once_the_leader_turns() -> None:
    """The mechanism behind §4.6.1's dominant failure mode.

    A leader that turns at a junction swings its rearward photodiode away from
    the follower, and the margin goes negative before anything blocks the path.
    """

    builder = headline_builder()
    index = builder.schema.columns.index("optical_fov_margin")

    aligned = builder.build(
        inputs(transmitter=predicted("tx", 0.0, 0.0, heading_rad=EAST),
               receiver=predicted("rx", 20.0, 0.0, heading_rad=EAST))
    )[index]
    turned = builder.build(
        inputs(transmitter=predicted("tx", 0.0, 0.0, heading_rad=EAST),
               receiver=predicted("rx", 20.0, 0.0, heading_rad=NORTH))
    )[index]

    assert aligned == pytest.approx(math.radians(60.0))
    assert turned == pytest.approx(math.radians(60.0) - 0.5 * math.pi)
    assert turned < 0.0


def test_relative_speed_is_positive_when_the_gap_opens() -> None:
    builder = headline_builder()
    index = builder.schema.columns.index("relative_speed")

    opening = builder.build(
        inputs(transmitter=predicted("tx", 0.0, 0.0, speed_mps=8.0),
               receiver=predicted("rx", 20.0, 0.0, speed_mps=12.0))
    )[index]
    closing = builder.build(
        inputs(transmitter=predicted("tx", 0.0, 0.0, speed_mps=12.0),
               receiver=predicted("rx", 20.0, 0.0, speed_mps=8.0))
    )[index]

    assert opening == pytest.approx(4.0)
    assert closing == pytest.approx(-4.0)


def test_track_age_is_the_worse_tracked_end() -> None:
    """A pair is only as well known as its stalest endpoint."""

    builder = headline_builder()
    index = builder.schema.columns.index("track_age")

    vector = builder.build(
        inputs(transmitter_track=tracked("tx", measured_at_s=1.0),
               receiver_track=tracked("rx", measured_at_s=0.5))
    )
    assert vector[index] == pytest.approx(1.05 - 0.5)


# -- forecast passthrough -----------------------------------------------------


def test_the_blockage_probability_and_confidence_are_carried_through() -> None:
    builder = headline_builder()
    columns = builder.schema.columns
    forecast = BlockageForecast(probability=0.42, confidence=0.6, horizon_s=0.2, considered=5)
    vector = builder.build(inputs(blockage=forecast))

    assert vector[columns.index("predicted_blockage_probability")] == pytest.approx(0.42)
    assert vector[columns.index("predictor_confidence")] == pytest.approx(0.6)


def test_neighbour_count_and_busy_ratio_are_carried_through() -> None:
    builder = headline_builder()
    columns = builder.schema.columns
    vector = builder.build(inputs(neighbour_count=137, channel_busy_ratio=0.71))

    assert vector[columns.index("neighbor_count")] == pytest.approx(137.0)
    assert vector[columns.index("rf_channel_busy_ratio")] == pytest.approx(0.71)


def test_consecutive_misses_reach_the_vector() -> None:
    links = LinkStateTracker(history_packets=8)
    for index in range(4):
        links.record(action=Action.RF, at_s=float(index), delivered=False,
                     measurements={Link.RF: 1.0})

    builder = headline_builder()
    vector = builder.build(inputs(links=links))
    assert vector[builder.schema.columns.index("consecutive_miss_count")] == pytest.approx(4.0)


# -- refusals -----------------------------------------------------------------


def test_a_non_finite_value_is_refused_rather_than_shipped() -> None:
    builder = headline_builder()
    with pytest.raises(ValueError, match="non-finite"):
        builder.build(inputs(channel_busy_ratio=math.inf))


def test_a_history_of_the_wrong_length_is_refused() -> None:
    """The schema and the tracker must agree on the packet count."""

    builder = ObservationBuilder(
        schema=ObservationSchema(features=("rf_quality_history",), history_packets=4)
    )
    with pytest.raises(ValueError, match="expected 4"):
        builder.build(inputs(links=LinkStateTracker(history_packets=8)))


# -- junction geometry, the strongest predictor available ---------------------


def test_the_junction_features_are_present_and_lawful() -> None:
    """Work plan §4.6.1 measured a 65-121x lift on junction spanning.

    They derive from own pose plus a road map, which a real vehicle carries,
    so they are observable rather than oracle state.
    """

    columns = headline_builder().schema.columns
    assert "distance_to_junction" in columns
    assert "path_spans_junction" in columns


def test_a_path_across_a_junction_is_flagged_and_one_mid_block_is_not() -> None:
    builder = headline_builder()
    index = builder.schema.columns.index("path_spans_junction")

    # Junction B5 sits at x = 244, y = 305; a pair straddling it spans.
    spanning = builder.build(
        inputs(
            transmitter=predicted("tx", 234.0, 305.0),
            receiver=predicted("rx", 254.0, 305.0),
            intersection=context(predicted("tx", 234.0, 305.0), predicted("rx", 254.0, 305.0)),
        )
    )[index]
    mid_block = builder.build(
        inputs(
            transmitter=predicted("tx", 300.0, 305.0),
            receiver=predicted("rx", 320.0, 305.0),
            intersection=context(predicted("tx", 300.0, 305.0), predicted("rx", 320.0, 305.0)),
        )
    )[index]

    assert spanning == 1.0
    assert mid_block == 0.0


def test_distance_to_junction_shrinks_as_the_pair_approaches_one() -> None:
    """Continuous as well as binary: the network picks its own threshold."""

    builder = headline_builder()
    index = builder.schema.columns.index("distance_to_junction")

    distances = []
    for start in (140.0, 180.0, 210.0, 230.0):
        tx = predicted("tx", start, 305.0)
        rx = predicted("rx", start + 10.0, 305.0)
        distances.append(builder.build(inputs(transmitter=tx, receiver=rx,
                                              intersection=context(tx, rx)))[index])

    assert distances == sorted(distances, reverse=True)
    assert all(value >= 0.0 for value in distances)
