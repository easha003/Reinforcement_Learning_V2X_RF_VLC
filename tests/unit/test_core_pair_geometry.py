from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import pytest

from hybrid_v2x_rl.core.pair_geometry import (
    DEFAULT_FOV_HALF_ANGLE_RAD,
    pair_geometry,
    wrap_to_pi,
)

EAST = 0.0
NORTH = 0.5 * math.pi
WEST = math.pi
SOUTH = 1.5 * math.pi

CAR_LENGTH_M = 4.5


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = CAR_LENGTH_M
    width_m: float = 1.8
    height_m: float = 1.5


# -- angle wrapping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (1.5 * math.pi, -0.5 * math.pi),
        (-1.5 * math.pi, 0.5 * math.pi),
        (3.0 * math.pi, math.pi),
    ],
)
def test_wrap_to_pi_is_half_open(given: float, expected: float) -> None:
    assert wrap_to_pi(given) == pytest.approx(expected)


# -- an aligned following pair -----------------------------------------------


def test_aligned_pair_is_boresight_to_boresight() -> None:
    """The nominal case: both angles vanish and the receiver is in view."""

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 20.0, NORTH)

    geometry = pair_geometry(follower, leader)

    assert geometry.separation_m == pytest.approx(20.0)
    assert geometry.optical_path_length_m == pytest.approx(20.0 - CAR_LENGTH_M)
    assert geometry.emission_angle_rad == pytest.approx(0.0)
    assert geometry.incidence_angle_rad == pytest.approx(0.0)
    assert geometry.relative_heading_rad == pytest.approx(0.0)
    assert geometry.within_field_of_view
    assert geometry.fov_margin_rad == pytest.approx(DEFAULT_FOV_HALF_ANGLE_RAD)


def test_separation_is_front_to_front_and_the_optical_path_is_shorter() -> None:
    """The two lengths differ by exactly the receiver's body."""

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 12.0, NORTH)

    geometry = pair_geometry(follower, leader)
    assert geometry.separation_m - geometry.optical_path_length_m == pytest.approx(CAR_LENGTH_M)


# -- the junction turn, which is why this module exists ----------------------


def test_a_leader_that_turned_at_a_junction_falls_out_of_view() -> None:
    """The outage no occlusion test can find.

    Tagged pairs are formed with matching headings but the continuation loop
    never re-checks, so a pair survives its leader turning.  The photodiode has
    then swung a quarter turn and receives nothing, however clear the path.
    """

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    turned_leader = FakeVehicle("rx", 0.0, 20.0, EAST)

    geometry = pair_geometry(follower, turned_leader)

    # Turning also carries the photodiode a body length to the west, so the
    # incidence angle exceeds the quarter turn the headings alone suggest.
    expected = math.atan2(20.0, -CAR_LENGTH_M)
    assert geometry.relative_heading_rad == pytest.approx(-NORTH)
    assert geometry.incidence_angle_rad == pytest.approx(expected)
    assert geometry.incidence_angle_rad > 0.5 * math.pi
    assert not geometry.within_field_of_view
    assert geometry.fov_margin_rad < 0.0


def test_a_leader_turning_the_other_way_also_falls_out_of_view() -> None:
    """Mirror image of the eastward turn; the same angle by symmetry."""

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    turned_leader = FakeVehicle("rx", 0.0, 20.0, WEST)

    geometry = pair_geometry(follower, turned_leader)
    assert geometry.incidence_angle_rad == pytest.approx(math.atan2(20.0, -CAR_LENGTH_M))
    assert not geometry.within_field_of_view


def test_a_leader_reversed_head_on_is_squarely_in_view() -> None:
    """A receiver facing the transmitter points its rear away; incidence is pi."""

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    oncoming = FakeVehicle("rx", 0.0, 20.0, SOUTH)

    geometry = pair_geometry(follower, oncoming)
    assert geometry.incidence_angle_rad == pytest.approx(math.pi)
    assert not geometry.within_field_of_view


# -- the acceptance cone -----------------------------------------------------


def test_incidence_exactly_at_the_half_angle_counts_as_in_view() -> None:
    """Boundary contact is inclusive, as everywhere else in the geometry layer."""

    half_angle = math.radians(60.0)
    # Keep the leader northbound so its photodiode sits directly below it, then
    # place the follower off to one side so the ray arrives exactly on the cone.
    leader = FakeVehicle("rx", 0.0, 20.0, NORTH)
    diode_y = 20.0 - CAR_LENGTH_M
    follower = FakeVehicle("tx", diode_y * math.tan(half_angle), 0.0, NORTH)

    geometry = pair_geometry(follower, leader, fov_half_angle_rad=half_angle)
    assert geometry.incidence_angle_rad == pytest.approx(half_angle)
    assert geometry.within_field_of_view
    assert geometry.fov_margin_rad == pytest.approx(0.0)


def test_a_narrower_cone_rejects_what_a_wider_one_accepts() -> None:
    """The semi-angle convention is applied by the caller, not assumed here."""

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 20.0, NORTH + math.radians(45.0))

    wide = pair_geometry(follower, leader, fov_half_angle_rad=math.radians(60.0))
    narrow = pair_geometry(follower, leader, fov_half_angle_rad=math.radians(30.0))

    assert wide.within_field_of_view
    assert not narrow.within_field_of_view


# -- lateral offset ----------------------------------------------------------


def test_lateral_offset_opens_both_angles_symmetrically() -> None:
    """A receiver off to one side is both off-boresight and off-axis."""

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    # Offset east by the same amount as the along-track optical run.
    leader = FakeVehicle("rx", 10.0, 10.0 + CAR_LENGTH_M, NORTH)

    geometry = pair_geometry(follower, leader)
    assert geometry.emission_angle_rad == pytest.approx(math.radians(45.0))
    assert geometry.incidence_angle_rad == pytest.approx(math.radians(45.0))
    assert geometry.relative_bearing_rad == pytest.approx(-math.radians(45.0))
    assert geometry.within_field_of_view


def test_heading_agnostic_pairs_give_the_same_angles_under_rotation() -> None:
    """Rotating the whole scene must not change any relative quantity."""

    base_tx = FakeVehicle("tx", 0.0, 0.0, NORTH)
    base_rx = FakeVehicle("rx", 3.0, 18.0, NORTH)
    base = pair_geometry(base_tx, base_rx)

    turn = math.radians(37.0)
    cos_t, sin_t = math.cos(turn), math.sin(turn)

    def rotate(v: FakeVehicle) -> FakeVehicle:
        return FakeVehicle(
            v.vehicle_id,
            v.x_m * cos_t - v.y_m * sin_t,
            v.x_m * sin_t + v.y_m * cos_t,
            v.heading_rad + turn,
        )

    rotated = pair_geometry(rotate(base_tx), rotate(base_rx))

    assert rotated.separation_m == pytest.approx(base.separation_m)
    assert rotated.optical_path_length_m == pytest.approx(base.optical_path_length_m)
    assert rotated.emission_angle_rad == pytest.approx(base.emission_angle_rad)
    assert rotated.incidence_angle_rad == pytest.approx(base.incidence_angle_rad)
    assert rotated.relative_heading_rad == pytest.approx(base.relative_heading_rad)


# -- section 5.2 / 5.3 boundary ----------------------------------------------


def test_pair_geometry_reports_angles_and_not_optical_power() -> None:
    """Geometry decides where things point; the channel decides what it costs.

    The measured non-Lambertian headlamp pattern maps an emission angle to a
    gain and belongs in ``channels/vlc``.  Keeping that out of this module is
    the same separation section 5.3 draws for blockage.
    """

    # ``importlib`` rather than ``import ... as``: a package attribute can
    # shadow a submodule of the same name, and this test must fail only on a
    # real boundary violation.
    module = importlib.import_module("hybrid_v2x_rl.core.pair_geometry")

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as stream:
        text = stream.read()

    for forbidden in ("hybrid_v2x_rl.channels", "pathloss", "lambertian_gain", "irradiance"):
        assert f"import {forbidden}" not in text
        assert f"from {forbidden}" not in text

    fields = set(pair_geometry.__annotations__)
    assert "gain" not in " ".join(fields)
