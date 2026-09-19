from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.geometry import Point, Segment
from hybrid_v2x_rl.core.link_endpoints import rf_link_path
from hybrid_v2x_rl.geometry.building_geometry import LANE_WIDTH_M, BuildingLayout
from hybrid_v2x_rl.geometry.rf_visibility import (
    building_obstructs,
    classify_pair,
    classify_path,
)

EAST = 0.0
NORTH = 0.5 * math.pi

# Headline grid, from configs/mobility/synthetic_manhattan.yaml.
AVENUE_SPACING_M = 244.0
CROSS_STREET_SPACING_M = 61.0


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = 4.5
    width_m: float = 1.8
    height_m: float = 1.5


def headline_layout() -> BuildingLayout:
    return BuildingLayout.from_grid(
        avenues=6,
        cross_streets=12,
        avenue_spacing_m=AVENUE_SPACING_M,
        cross_street_spacing_m=CROSS_STREET_SPACING_M,
        lanes_per_direction=1,
    )


# -- the block layout --------------------------------------------------------


def test_headline_grid_encloses_the_expected_blocks() -> None:
    """Blocks are the street pitch less one carriageway in each direction."""

    layout = headline_layout()

    assert layout.block_count == 5 * 11
    assert len(layout.rectangles) == 55
    assert layout.road_half_width_m == pytest.approx(LANE_WIDTH_M)
    assert layout.block_length_m == pytest.approx(AVENUE_SPACING_M - 2 * LANE_WIDTH_M)
    assert layout.block_width_m == pytest.approx(CROSS_STREET_SPACING_M - 2 * LANE_WIDTH_M)


def test_blocks_sit_between_street_centrelines_and_never_on_them() -> None:
    """A vehicle travelling any centreline must never be inside a building."""

    layout = headline_layout()

    for avenue in range(6):
        for offset in (0.0, 30.0, 60.0):
            on_avenue = Point(avenue * AVENUE_SPACING_M, offset)
            assert not any(block.contains(on_avenue) for block in layout.rectangles)

    for cross in range(12):
        for offset in (0.0, 120.0, 244.0):
            on_cross_street = Point(offset, cross * CROSS_STREET_SPACING_M)
            assert not any(block.contains(on_cross_street) for block in layout.rectangles)


def test_the_block_interior_is_inside_a_building() -> None:
    layout = headline_layout()
    interior = Point(0.5 * AVENUE_SPACING_M, 0.5 * CROSS_STREET_SPACING_M)
    assert sum(block.contains(interior) for block in layout.rectangles) == 1


def test_wider_roads_shrink_the_blocks() -> None:
    narrow = headline_layout()
    wide = BuildingLayout.from_grid(
        avenues=6,
        cross_streets=12,
        avenue_spacing_m=AVENUE_SPACING_M,
        cross_street_spacing_m=CROSS_STREET_SPACING_M,
        lanes_per_direction=3,
    )
    assert wide.block_width_m < narrow.block_width_m
    assert wide.block_count == narrow.block_count


def test_a_carriageway_wider_than_the_pitch_is_rejected() -> None:
    """Cross streets are only 61 m apart; enough lanes leave no block at all."""

    with pytest.raises(ValueError, match="cross-street spacing leaves no room"):
        BuildingLayout.from_grid(
            avenues=6,
            cross_streets=12,
            avenue_spacing_m=AVENUE_SPACING_M,
            cross_street_spacing_m=CROSS_STREET_SPACING_M,
            lanes_per_direction=9,
        )


# -- section 5.1: building corners -------------------------------------------


def test_a_link_along_a_street_clears_every_building() -> None:
    """The tagged-pair case: same heading, same street, no facade between."""

    layout = headline_layout()
    follower = FakeVehicle("tx", 0.0, 10.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 40.0, NORTH)

    assert not building_obstructs(rf_link_path(follower, leader), layout.rectangles)


def test_a_link_around_a_corner_is_cut_by_the_block() -> None:
    """Diagonal across a block interior: the defining NLOS case."""

    layout = headline_layout()
    on_avenue = FakeVehicle("a", 0.0, 10.0, NORTH)
    on_cross_street = FakeVehicle("b", 120.0, CROSS_STREET_SPACING_M, EAST)

    visibility = classify_pair(on_avenue, on_cross_street, [], layout.rectangles)
    assert visibility.state is RFPropagationState.NLOS
    assert visibility.building_blocked


def test_a_link_grazing_a_corner_counts_as_obstructed() -> None:
    """Boundary contact blocks, matching the convention in primitives."""

    layout = headline_layout()
    corner = Point(LANE_WIDTH_M, LANE_WIDTH_M)
    grazing = Segment(Point(LANE_WIDTH_M, 0.0), Point(LANE_WIDTH_M, 40.0))

    assert any(block.contains(corner) for block in layout.rectangles)
    assert building_obstructs(
        rf_link_path(
            FakeVehicle("a", grazing.start.x_m, grazing.start.y_m, NORTH),
            FakeVehicle("b", grazing.end.x_m, grazing.end.y_m, NORTH),
        ),
        layout.rectangles,
    )


def test_vehicles_inside_the_junction_box_still_see_each_other() -> None:
    """Close enough to the corner, the diagonal never leaves the carriageway."""

    layout = headline_layout()
    approaching = FakeVehicle("a", 0.0, CROSS_STREET_SPACING_M - 4.0, NORTH)
    departing = FakeVehicle("b", 5.0, CROSS_STREET_SPACING_M, EAST)

    visibility = classify_pair(approaching, departing, [], layout.rectangles)
    assert visibility.state is RFPropagationState.LOS


def test_backing_away_from_the_junction_loses_sight_around_the_corner() -> None:
    """The corner cut-off: the same two streets, a little further out.

    The carriageway is only 3.5 m either side of a centreline, so a diagonal
    between two vehicles on perpendicular streets leaves the road almost as
    soon as either one backs away from the junction.
    """

    layout = headline_layout()
    approaching = FakeVehicle("a", 0.0, CROSS_STREET_SPACING_M - 12.0, NORTH)
    departing = FakeVehicle("b", 14.0, CROSS_STREET_SPACING_M, EAST)

    visibility = classify_pair(approaching, departing, [], layout.rectangles)
    assert visibility.state is RFPropagationState.NLOS
    assert visibility.building_blocked


# -- the three states ---------------------------------------------------------


def test_a_clear_street_link_is_line_of_sight() -> None:
    layout = headline_layout()
    follower = FakeVehicle("tx", 0.0, 10.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 40.0, NORTH)

    visibility = classify_pair(follower, leader, [follower, leader], layout.rectangles)
    assert visibility.state is RFPropagationState.LOS
    assert visibility.is_line_of_sight
    assert visibility.vehicle_blocker_count == 0


def test_a_vehicle_across_the_path_gives_nlosv() -> None:
    layout = headline_layout()
    follower = FakeVehicle("tx", 0.0, 10.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 40.0, NORTH)
    crosser = FakeVehicle("cross", 4.25, 25.0, EAST)

    visibility = classify_pair(follower, leader, [follower, leader, crosser], layout.rectangles)
    assert visibility.state is RFPropagationState.NLOSV
    assert visibility.vehicle_blocker_ids == ("cross",)
    assert not visibility.building_blocked


def test_a_low_vehicle_passes_under_the_antenna_path() -> None:
    """The antenna path sits at 1.5 m; a shorter body does not obstruct it."""

    layout = headline_layout()
    follower = FakeVehicle("tx", 0.0, 10.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 40.0, NORTH)
    low = FakeVehicle("low", 4.25, 25.0, EAST, height_m=1.0)

    visibility = classify_pair(follower, leader, [follower, leader, low], layout.rectangles)
    assert visibility.state is RFPropagationState.LOS


def test_a_building_outranks_a_vehicle_when_both_obstruct() -> None:
    """NLOS is the more severe class and TR 37.885 models it separately."""

    layout = headline_layout()
    on_avenue = FakeVehicle("a", 0.0, 10.0, NORTH)
    on_cross_street = FakeVehicle("b", 120.0, CROSS_STREET_SPACING_M, EAST)
    crosser = FakeVehicle("cross", 30.0, 20.0, EAST)

    visibility = classify_pair(
        on_avenue, on_cross_street, [on_avenue, on_cross_street, crosser], layout.rectangles
    )
    assert visibility.state is RFPropagationState.NLOS
    assert visibility.building_blocked
    # The vehicle blocker is still reported for diagnostics.
    assert "cross" in visibility.vehicle_blocker_ids


def test_the_endpoints_never_obstruct_their_own_link() -> None:
    layout = headline_layout()
    follower = FakeVehicle("tx", 0.0, 10.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 40.0, NORTH)

    auto = classify_pair(follower, leader, [follower, leader], layout.rectangles)
    assert auto.vehicle_blocker_count == 0

    # Passing an empty exclusion set is the caller's choice, and then the
    # endpoints do meet their own path.
    explicit = classify_path(
        rf_link_path(follower, leader), [follower, leader], layout.rectangles, exclude_ids=()
    )
    assert explicit.state is RFPropagationState.NLOSV


# -- section 5.3: the module boundary ----------------------------------------


def test_rf_visibility_does_not_depend_on_any_path_loss_model() -> None:
    """Work plan section 5.3, enforced rather than merely stated.

    TR 37.885 supplies loss, shadowing and fading *given* a class.  If this
    module could reach that code the class itself could become stochastic, and
    a class that is not a function of where obstacles are cannot be predicted
    from tracked positions.
    """

    module = importlib.import_module("hybrid_v2x_rl.geometry.rf_visibility")

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as stream:
        text = stream.read()

    for forbidden in ("hybrid_v2x_rl.channels", "pathloss", "path_loss", "shadowing", "fading"):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text
