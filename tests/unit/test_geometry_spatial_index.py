from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.core.geometry import (
    Point,
    Segment,
    segment_intersects_rectangle,
    vehicle_rectangle,
)
from hybrid_v2x_rl.core.link_endpoints import LinkPath
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex
from hybrid_v2x_rl.geometry.vehicle_occlusion import is_obstructed, occluding_vehicles

EAST = 0.0
NORTH = 0.5 * math.pi


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = 4.5
    width_m: float = 1.8
    height_m: float = 1.5


def random_scene(seed: int, count: int) -> list[FakeVehicle]:
    """A frame of vehicles spread over a few blocks, with mixed classes."""

    rng = random.Random(seed)
    classes = ((4.5, 1.8, 1.5), (5.2, 2.0, 2.0), (11.0, 2.5, 3.25))
    scene: list[FakeVehicle] = []
    for index in range(count):
        length, width, height = classes[rng.randrange(len(classes))]
        scene.append(
            FakeVehicle(
                vehicle_id=f"v{index:04d}",
                x_m=rng.uniform(-50.0, 300.0),
                y_m=rng.uniform(-50.0, 300.0),
                heading_rad=rng.choice((0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)),
                length_m=length,
                width_m=width,
                height_m=height,
            )
        )
    return scene


def brute_force_blockers(segment: Segment, scene: list[FakeVehicle]) -> set[str]:
    return {
        vehicle.vehicle_id
        for vehicle in scene
        if segment_intersects_rectangle(segment, vehicle_rectangle(vehicle))
    }


# -- the property that matters ------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_index_never_loses_a_blocker_that_brute_force_finds(seed: int) -> None:
    """The index is a filter: its candidates must be a superset of the truth.

    A missing candidate would silently drop an obstruction, and nothing
    downstream could detect it.  This is the test the whole optimisation rests
    on, so it runs over many random scenes rather than one hand-built case.
    """

    scene = random_scene(seed, 400)
    index = SpatialIndex.build(scene)
    rng = random.Random(seed * 977)

    for _ in range(60):
        segment = Segment(
            Point(rng.uniform(-40.0, 290.0), rng.uniform(-40.0, 290.0)),
            Point(rng.uniform(-40.0, 290.0), rng.uniform(-40.0, 290.0)),
        )
        expected = brute_force_blockers(segment, scene)
        found = {vehicle.vehicle_id for vehicle in index.candidates(segment)}
        assert expected <= found, f"index lost {sorted(expected - found)}"


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_indexed_occlusion_matches_the_unindexed_scan(seed: int) -> None:
    """End to end: filtering first must not change any verdict."""

    scene = random_scene(seed, 300)
    index = SpatialIndex.build(scene)
    rng = random.Random(seed * 31)

    for _ in range(40):
        start = Point(rng.uniform(0.0, 250.0), rng.uniform(0.0, 250.0))
        end = Point(start.x_m + rng.uniform(-40.0, 40.0), start.y_m + rng.uniform(-40.0, 40.0))
        path = LinkPath(segment=Segment(start, end), height_m=0.7)

        full = occluding_vehicles(path, scene)
        narrowed = occluding_vehicles(path, index.candidates(path.segment))

        assert set(narrowed.blocker_ids) == set(full.blocker_ids)
        assert narrowed.is_blocked == full.is_blocked
        assert is_obstructed(path, index.candidates(path.segment)) == full.is_blocked


def test_the_index_actually_narrows_the_search() -> None:
    """A filter that returns everything would pass the tests above and be useless."""

    scene = random_scene(7, 500)
    index = SpatialIndex.build(scene)
    segment = Segment(Point(100.0, 100.0), Point(100.0, 130.0))

    narrowed = index.candidates(segment)
    assert len(narrowed) < len(scene) / 10


# -- construction -------------------------------------------------------------


def test_cell_size_defaults_to_the_largest_footprint_diagonal() -> None:
    bus = FakeVehicle("bus", 0.0, 0.0, EAST, length_m=11.0, width_m=2.5, height_m=3.25)
    index = SpatialIndex.build([bus])
    assert index.cell_size_m == pytest.approx(math.hypot(11.0, 2.5))


def test_a_frame_of_small_footprints_still_gets_a_sane_cell_size() -> None:
    tiny = FakeVehicle("tiny", 0.0, 0.0, EAST, length_m=1.0, width_m=0.5)
    assert SpatialIndex.build([tiny]).cell_size_m == pytest.approx(5.0)


def test_a_vehicle_straddling_a_cell_edge_is_found_from_either_side() -> None:
    """Bucketing by footprint bounds, not by a single point."""

    cell = 10.0
    # Long body centred on a cell boundary, so it occupies two cells.
    straddler = FakeVehicle("long", 24.0, 5.0, EAST, length_m=11.0, width_m=2.5)
    index = SpatialIndex.build([straddler], cell_size_m=cell)

    from_left = index.candidates(Segment(Point(15.0, 0.0), Point(15.0, 10.0)))
    from_right = index.candidates(Segment(Point(22.0, 0.0), Point(22.0, 10.0)))
    assert "long" in {v.vehicle_id for v in from_left}
    assert "long" in {v.vehicle_id for v in from_right}


def test_a_vehicle_is_returned_once_even_when_it_spans_cells() -> None:
    bus = FakeVehicle("bus", 30.0, 5.0, EAST, length_m=11.0, width_m=2.5)
    index = SpatialIndex.build([bus], cell_size_m=4.0)
    found = index.candidates(Segment(Point(0.0, 5.0), Point(60.0, 5.0)))
    assert [v.vehicle_id for v in found] == ["bus"]


def test_margin_widens_the_search() -> None:
    away = FakeVehicle("away", 60.0, 0.0, EAST)
    index = SpatialIndex.build([away], cell_size_m=5.0)
    segment = Segment(Point(0.0, 0.0), Point(10.0, 0.0))

    assert index.candidates(segment) == ()
    assert {v.vehicle_id for v in index.candidates(segment, margin_m=60.0)} == {"away"}


def test_an_empty_frame_yields_no_candidates() -> None:
    index = SpatialIndex.build([])
    assert index.cell_count == 0
    assert index.candidates(Segment(Point(0.0, 0.0), Point(10.0, 10.0))) == ()


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_an_unusable_cell_size_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="cell_size_m"):
        SpatialIndex.build([FakeVehicle("a", 0.0, 0.0, EAST)], cell_size_m=bad)


def test_a_negative_margin_is_rejected() -> None:
    index = SpatialIndex.build([FakeVehicle("a", 0.0, 0.0, EAST)])
    with pytest.raises(ValueError, match="margin_m"):
        index.candidates(Segment(Point(0.0, 0.0), Point(1.0, 0.0)), margin_m=-1.0)
