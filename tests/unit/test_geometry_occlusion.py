from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.core.link_endpoints import (
    DEFAULT_HEADLAMP_HEIGHT_M,
    DEFAULT_RF_ANTENNA_HEIGHT_M,
    antenna_position,
    headlamp_position,
    optical_link_path,
    photodiode_position,
    rf_link_path,
)
from hybrid_v2x_rl.geometry.vehicle_occlusion import (
    is_obstructed,
    occluding_vehicles,
)

EAST = 0.0
NORTH = 0.5 * math.pi

CAR_LENGTH_M = 4.5
CAR_WIDTH_M = 1.8
CAR_HEIGHT_M = 1.5


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    """Stand-in carrying everything the occlusion protocol requires."""

    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = CAR_LENGTH_M
    width_m: float = CAR_WIDTH_M
    height_m: float = CAR_HEIGHT_M


def following_pair(separation_m: float) -> tuple[FakeVehicle, FakeVehicle]:
    """A follower and its leader on a northbound avenue.

    ``separation_m`` is front-to-front, matching how tagged-pair eligibility
    measures distance in ``mobility/tagged_pairs.py``.
    """

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 0.0, separation_m, NORTH)
    return follower, leader


# -- endpoints ---------------------------------------------------------------


def test_optical_path_runs_from_the_follower_front_to_the_leader_rear() -> None:
    """The beam leaves a front bumper and lands on a trailing edge.

    Using the leader's stored position instead would run the path through the
    leader's own body and overstate the path length by a vehicle length.
    """

    follower, leader = following_pair(20.0)

    lamp = headlamp_position(follower)
    assert lamp.x_m == pytest.approx(0.0)
    assert lamp.y_m == pytest.approx(0.0)
    assert photodiode_position(leader).y_m == pytest.approx(20.0 - CAR_LENGTH_M)

    path = optical_link_path(follower, leader)
    assert path.height_m == pytest.approx(DEFAULT_HEADLAMP_HEIGHT_M)
    assert path.length_m == pytest.approx(20.0 - CAR_LENGTH_M)


def test_rf_path_runs_centre_to_centre_and_is_shorter_than_front_to_front() -> None:
    follower, leader = following_pair(20.0)

    assert antenna_position(follower).y_m == pytest.approx(-CAR_LENGTH_M / 2.0)
    assert antenna_position(leader).y_m == pytest.approx(20.0 - CAR_LENGTH_M / 2.0)

    path = rf_link_path(follower, leader)
    assert path.height_m == pytest.approx(DEFAULT_RF_ANTENNA_HEIGHT_M)
    assert path.length_m == pytest.approx(20.0)


# -- the endpoints never block their own link --------------------------------


def test_the_pair_itself_is_excluded_from_its_own_link() -> None:
    follower, leader = following_pair(20.0)
    path = optical_link_path(follower, leader)

    assert not is_obstructed(path, [follower, leader], exclude_ids=("tx", "rx"))
    # Without the exclusion the transmitter meets its own beam origin.
    assert is_obstructed(path, [follower, leader])


def test_a_clear_following_pair_reports_no_blockers() -> None:
    follower, leader = following_pair(20.0)
    result = occluding_vehicles(
        optical_link_path(follower, leader), [follower, leader], exclude_ids=("tx", "rx")
    )
    assert not result.is_blocked
    assert result.blocker_ids == ()
    assert result.nearest_blocker_id is None


# -- cross-traffic, the dominant real mechanism ------------------------------


def test_cross_traffic_at_a_junction_obstructs_the_optical_path() -> None:
    """The measured mechanism: 98.6-99.5% of blockage is a different edge.

    The crosser travels east through the junction between the pair.  Its
    centre is 2.0 m off the beam line -- outside half its width, inside half
    its length -- which the earlier point-plus-half-width probe would have
    missed.
    """

    follower, leader = following_pair(20.0)
    crosser = FakeVehicle("cross", x_m=4.25, y_m=8.0, heading_rad=EAST)

    path = optical_link_path(follower, leader)
    result = occluding_vehicles(path, [follower, leader, crosser], exclude_ids=("tx", "rx"))

    assert result.is_blocked
    assert result.blocker_ids == ("cross",)
    assert result.nearest_blocker_id == "cross"


def test_cross_traffic_clear_of_the_path_does_not_obstruct() -> None:
    follower, leader = following_pair(20.0)
    crosser = FakeVehicle("cross", x_m=7.0, y_m=8.0, heading_rad=EAST)

    assert not is_obstructed(
        optical_link_path(follower, leader),
        [follower, leader, crosser],
        exclude_ids=("tx", "rx"),
    )


def test_a_crosser_beyond_the_leader_does_not_obstruct() -> None:
    """Obstruction is bounded by the link, not by its infinite line."""

    follower, leader = following_pair(20.0)
    beyond = FakeVehicle("beyond", x_m=4.25, y_m=30.0, heading_rad=EAST)

    assert not is_obstructed(
        optical_link_path(follower, leader), [follower, leader, beyond], exclude_ids=("tx", "rx")
    )


# -- height -------------------------------------------------------------------


def test_a_body_shorter_than_the_path_passes_beneath_it() -> None:
    """Height is tested, not assumed, even though no configured class is short."""

    follower, leader = following_pair(20.0)
    low = FakeVehicle("low", x_m=4.25, y_m=8.0, heading_rad=EAST, height_m=0.4)
    tall = FakeVehicle("tall", x_m=4.25, y_m=12.0, heading_rad=EAST, height_m=0.8)

    result = occluding_vehicles(
        optical_link_path(follower, leader), [follower, leader, low, tall], exclude_ids=("tx", "rx")
    )
    assert result.blocker_ids == ("tall",)


def test_a_car_is_exactly_tall_enough_to_be_an_rf_blocker() -> None:
    """A 1.5 m car meets a 1.5 m antenna path; boundary contact blocks.

    This is the NLOSv case, and it is decided here geometrically. What that
    class costs in dB is TR 37.885's business, not this module's.
    """

    follower, leader = following_pair(20.0)
    crosser = FakeVehicle("cross", x_m=4.25, y_m=8.0, heading_rad=EAST, height_m=CAR_HEIGHT_M)

    assert is_obstructed(
        rf_link_path(follower, leader), [follower, leader, crosser], exclude_ids=("tx", "rx")
    )


def test_the_same_geometry_can_block_optics_while_clearing_the_antennas() -> None:
    """The asymmetry the contribution rests on, at its simplest.

    A low obstruction cuts a 0.7 m beam and passes under a 1.5 m antenna path.
    """

    follower, leader = following_pair(20.0)
    low = FakeVehicle("low", x_m=4.25, y_m=8.0, heading_rad=EAST, height_m=1.0)

    assert is_obstructed(
        optical_link_path(follower, leader), [follower, leader, low], exclude_ids=("tx", "rx")
    )
    assert not is_obstructed(
        rf_link_path(follower, leader), [follower, leader, low], exclude_ids=("tx", "rx")
    )


# -- ordering and multiplicity ------------------------------------------------


def test_blockers_are_returned_nearest_to_the_transmitter_first() -> None:
    follower, leader = following_pair(40.0)
    near = FakeVehicle("near", x_m=4.25, y_m=8.0, heading_rad=EAST)
    far = FakeVehicle("far", x_m=4.25, y_m=28.0, heading_rad=EAST)

    result = occluding_vehicles(
        optical_link_path(follower, leader),
        [follower, leader, far, near],
        exclude_ids=("tx", "rx"),
    )
    assert result.blocker_ids == ("near", "far")
    assert result.nearest_blocker_id == "near"
    assert result.blocker_count == 2


# -- section 5.3: the module boundary ----------------------------------------


def test_occlusion_does_not_depend_on_any_path_loss_model() -> None:
    """Work plan section 5.3, enforced rather than merely stated.

    If blockage ever became a stochastic loss term it would stop being a
    function of where a blocker is, could not be predicted from tracked
    positions, and RQ3 would collapse.  Keeping this module free of channel
    imports is what makes that structural.
    """

    module = importlib.import_module("hybrid_v2x_rl.geometry.vehicle_occlusion")

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as stream:
        text = stream.read()

    for forbidden in ("pathloss", "path_loss", "hybrid_v2x_rl.channels", "shadowing", "fading"):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text
