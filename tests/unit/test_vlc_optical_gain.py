"""Direct-path optical power, and the four consequences of this geometry.

The module cannot produce a real number today -- the measured headlamp pattern
does not exist and no Lambertian stand-in is permitted -- so these tests do two
things.  They check the interface behaves correctly against patterns supplied
*by name*, and they pin the structural findings that hold whatever the beam
turns out to be, because those follow from the endpoint convention and the
mobility model rather than from the optics.

``LambertianPattern`` appears below as an explicitly named comparison model and
a synthetic ``TabulatedPattern`` appears as a fixture.  Neither stands in for a
measured beam, and no test asserts an absolute received power.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.vlc.headlamp_pattern import (
    HeadlampPatternError,
    LambertianPattern,
    TabulatedPattern,
    load_pattern,
)
from hybrid_v2x_rl.channels.vlc.optical_gain import (
    COMPLETE_BLOCKAGE,
    MIN_VALID_PATH_LENGTH_M,
    BlockageModel,
    OpticalGainError,
    direct_received_power_w,
    horizontal_emission_angle_rad,
    received_power_from_geometry,
)
from hybrid_v2x_rl.channels.vlc.receiver import OpticalReceiver
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.link_endpoints import DEFAULT_HEADLAMP_HEIGHT_M, DEFAULT_PHOTODIODE_HEIGHT_M
from hybrid_v2x_rl.core.pair_geometry import pair_geometry
from hybrid_v2x_rl.mobility.car_following import IDMParameters, desired_gap_m

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NORTH = 0.5 * math.pi

CAR_LENGTH_M = 4.5
BUS_LENGTH_M = 11.0

#: A named comparison model, never a stand-in for a measurement.
COMPARISON_BEAM = LambertianPattern(
    peak_intensity_w_per_sr=100.0, half_power_semi_angle_rad=math.radians(20.0)
)


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    length_m: float = CAR_LENGTH_M
    width_m: float = 1.8
    height_m: float = 1.5


@dataclass(slots=True)
class RecordingPattern:
    """Records the bearings it was asked for, so the call can be inspected.

    Which angle a caller samples the beam at is the thing most easily got
    wrong here, and it is invisible in the returned power for any symmetric
    pattern.  Recording makes it observable.
    """

    pattern_id: str = "recording_stub"
    samples: list[tuple[float, float]] = field(default_factory=list)

    def radiant_intensity_w_per_sr(
        self, horizontal_angle_rad: float, vertical_angle_rad: float
    ) -> float:
        self.samples.append((horizontal_angle_rad, vertical_angle_rad))
        return 50.0


class RefusingPattern:
    """Fails if sampled at all.  Used to prove a blocked path never asks."""

    pattern_id = "refusing_stub"

    def radiant_intensity_w_per_sr(
        self, horizontal_angle_rad: float, vertical_angle_rad: float
    ) -> float:
        raise AssertionError("a blocked optical path must not sample the beam")


def asymmetric_beam() -> TabulatedPattern:
    """A synthetic fixture, brighter to the vehicle's right.  Not a measurement.

    Real low beams are asymmetric because they are aimed away from oncoming
    traffic; this fixture only needs to be asymmetric enough to detect a
    mirrored or absolute-valued bearing.
    """

    return TabulatedPattern(
        pattern_id="asymmetric_fixture",
        horizontal_angles_rad=(-0.6, 0.0, 0.6),
        vertical_angles_rad=(-0.2, 0.0, 0.2),
        intensity_w_per_sr=(
            (2.0, 10.0, 8.0),
            (2.0, 10.0, 8.0),
            (2.0, 10.0, 8.0),
        ),
        source="synthetic fixture, not a measurement",
    )


def power_of(pattern, geometry, **kwargs) -> float:
    return received_power_from_geometry(
        pattern=pattern, receiver=OpticalReceiver(), geometry=geometry, **kwargs
    )


# -- the rule that comes first ------------------------------------------------


def test_the_configured_beam_now_loads_and_is_a_declared_envelope() -> None:
    """This test used to assert the opposite, and that was its purpose.

    It asserted the configured artifact did *not* exist, so that when one
    arrived the test would fail and force every optical number computed without
    it to be revisited. The artifact arrived on 2026-08-10, built from the ECE
    R112 test-point geometry rather than measured off a lamp, and the revisiting
    it forced was substantial: the link had been sampling the beam along its
    cut-off, which is what led to the receiver being moved to 0.4 m.

    What the file still does not do is assert an absolute received power. A
    regulatory envelope constrains a compliant lamp, it does not describe one,
    so ratios and orderings are meaningful here and absolute watts are not.
    """

    config = load_headline_config(PROJECT_ROOT)
    assert config.vlc.headlamp_pattern == "measured_non_lambertian"
    pattern = load_pattern(config.vlc.pattern_artifact)
    # The artifact must declare itself as modelled rather than measured. The
    # wording is not the contract -- the disclosure is -- so this matches on
    # meaning rather than on a sentence that can be reworded.
    source = pattern.source.lower()
    assert "not a measured lamp" in source or "not measured" in source
    assert "r112" in source or "regulation no. 112" in source


def test_the_module_never_reaches_for_the_lambertian_comparison() -> None:
    """A fallback would be one import away, so the import is what is checked.

    Spec 12.1 permits Lambertian only as an explicitly named comparison model.
    A cosine lobe substituted for a missing measurement is indistinguishable
    from a real result afterwards, which is the failure mode W17 warns about.
    """

    import hybrid_v2x_rl.channels.vlc.optical_gain as module

    tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    assert "LambertianPattern" not in imported
    assert "load_pattern" not in imported


# -- finding 1: the distance is the bumper gap, not the separation ------------


def test_the_inverse_square_distance_is_the_gap_not_the_separation() -> None:
    """The photodiode sits one vehicle length back, so ``d`` is the gap.

    Stated as a test because ``separation_m`` is the field every other part of
    the project uses -- the observation vector, the tagged-pair window, the
    section 14.1 tables -- and it is the wrong one here by one whole vehicle.
    """

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 0.0, 20.0, NORTH)
    geometry = pair_geometry(follower, leader)

    assert geometry.separation_m == pytest.approx(20.0)
    assert geometry.optical_path_length_m == pytest.approx(20.0 - CAR_LENGTH_M)

    drop = DEFAULT_HEADLAMP_HEIGHT_M - DEFAULT_PHOTODIODE_HEIGHT_M
    at_gap = direct_received_power_w(
        pattern=COMPARISON_BEAM,
        receiver=OpticalReceiver(),
        path_length_m=geometry.optical_path_length_m,
        horizontal_emission_rad=0.0,
        incidence_angle_rad=0.0,
        vertical_emission_rad=-math.atan2(drop, geometry.optical_path_length_m),
    )
    assert power_of(COMPARISON_BEAM, geometry) == pytest.approx(at_gap)

    # And the separation would have been wrong by a whole vehicle: at 20 m
    # front-to-front the optical path is 15.5 m, which is 2.2 dB of budget.
    at_separation = direct_received_power_w(
        pattern=COMPARISON_BEAM,
        receiver=OpticalReceiver(),
        path_length_m=geometry.separation_m,
        horizontal_emission_rad=0.0,
        incidence_angle_rad=0.0,
        vertical_emission_rad=-math.atan2(drop, geometry.separation_m),
    )
    assert at_gap > at_separation


def test_the_optical_link_budget_improves_as_traffic_gets_denser() -> None:
    """The finding, and it points the opposite way to the RF mechanisms.

    For a same-lane pair the optical path *is* the car-following gap, and the
    IDM holds that gap at ``s0 + v T``.  Denser traffic is slower, so the gap
    shortens and ``1/d^2`` rises.  Between the measured 10 and 30 veh/lane-km
    speeds that is over 4 dB optical and roughly 9.5 dB electrical -- the same
    order as the whole 60-to-30-degree concentrator sensitivity, obtained for
    free at exactly the density where RF congestion is worst and where the
    measured geometric outage is also lowest.

    It also means the IDM time headway, flagged in work plan section 4.7 item 1
    as an undeclared *capacity* parameter, is the leading term in the V-VLC
    link budget.
    """

    idm = IDMParameters()
    # Speeds measured for the fading table, work plan section 7.1.
    light_gap_m = desired_gap_m(6.30, 0.0, idm)
    heavy_gap_m = desired_gap_m(2.77, 0.0, idm)
    assert heavy_gap_m < light_gap_m

    def power_at(gap_m: float) -> float:
        return direct_received_power_w(
            pattern=COMPARISON_BEAM,
            receiver=OpticalReceiver(),
            path_length_m=gap_m,
            horizontal_emission_rad=0.0,
            incidence_angle_rad=0.0,
        )

    ratio = power_at(heavy_gap_m) / power_at(light_gap_m)
    assert 10.0 * math.log10(ratio) > 4.0
    assert 20.0 * math.log10(ratio) == pytest.approx(9.5, abs=0.5)


def test_received_power_follows_the_inverse_square_of_the_path() -> None:
    """Halving the path quadruples the power; nothing else moves."""

    receiver = OpticalReceiver()

    def power_at(path_m: float) -> float:
        return direct_received_power_w(
            pattern=COMPARISON_BEAM,
            receiver=receiver,
            path_length_m=path_m,
            horizontal_emission_rad=0.0,
            incidence_angle_rad=0.0,
        )

    assert power_at(5.0) / power_at(10.0) == pytest.approx(4.0)


# -- finding 2: leader length is a lever the policy cannot see ----------------


def test_leader_length_moves_the_budget_as_much_as_the_concentrator_does() -> None:
    """Same separation, different leader, nine decibels of electrical SNR.

    At 20 m a passenger-car leader gives a 15.5 m optical path and an 11 m bus
    gives 9.0 m: 4.7 dB optical, 9.4 dB electrical.  Buses are 3% of the fleet
    and the section 14.1 frontier tables read a single PER off a single
    separation, so this spread is currently invisible in them.
    """

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    behind_car = pair_geometry(follower, FakeVehicle("rx", 0.0, 20.0, NORTH))
    behind_bus = pair_geometry(
        follower, FakeVehicle("rx", 0.0, 20.0, NORTH, length_m=BUS_LENGTH_M)
    )

    assert behind_car.separation_m == pytest.approx(behind_bus.separation_m)
    assert behind_car.optical_path_length_m == pytest.approx(15.5)
    assert behind_bus.optical_path_length_m == pytest.approx(9.0)

    ratio = power_of(COMPARISON_BEAM, behind_bus) / power_of(COMPARISON_BEAM, behind_car)
    assert 10.0 * math.log10(ratio) == pytest.approx(4.72, abs=0.05)
    assert 20.0 * math.log10(ratio) == pytest.approx(9.44, abs=0.05)


def test_the_policy_observes_separation_and_never_the_leader_that_sets_the_gap() -> None:
    """Which is what makes the previous test a finding rather than a detail.

    Lawful: a real vehicle does not know its leader's length.  But it means
    9 dB of the optical link budget is unmodelled spread from the policy's
    point of view, and any clear-path PER curve plotted against
    ``pair_distance`` alone is averaging over it.
    """

    features = load_headline_config(PROJECT_ROOT).observation.features
    assert "pair_distance" in features
    for forbidden in ("length", "vehicle_class", "leader_length", "optical_path"):
        assert not any(forbidden in feature for feature in features)


# -- finding 3: the pattern's vertical axis is never sampled ------------------


def test_the_beam_is_sampled_below_the_horizon_and_the_angle_depends_on_the_gap() -> None:
    """The photodiode sits 0.3 m below the headlamp, so the link looks downward.

    This test previously asserted the opposite -- that the vertical angle was
    *identically zero*, because both endpoints sat at 0.7 m -- and recorded as a
    finding that the pattern's vertical axis was therefore never sampled. That
    was true, and it was the symptom of a defect rather than a property of the
    world: co-height endpoints sample an ECE low beam exactly along its cut-off,
    the one elevation a headlamp is engineered to keep dark, because its purpose
    is not to dazzle oncoming drivers. The equal heights were a simulation
    convenience inherited from the planar-path simplification.

    With the receiver at 0.4 m the beam is read where a headlamp actually puts
    its light, and the mandated cut-off -- the feature most distinguishing a
    real lamp from a cosine lobe -- becomes live rather than inert.

    The angle shrinks with distance, which matters: a fixed height difference
    subtends less elevation as the gap grows, so the far end of the pair window
    drifts back toward the dark cut-off on its own.
    """

    assert DEFAULT_PHOTODIODE_HEIGHT_M < DEFAULT_HEADLAMP_HEIGHT_M

    angles = []
    for gap_m in (10.0, 20.0, 40.0):
        pattern = RecordingPattern()
        follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
        leader = FakeVehicle("rx", 0.0, gap_m + CAR_LENGTH_M, NORTH)
        power_of(pattern, pair_geometry(follower, leader))
        vertical = pattern.samples[0][1]
        assert vertical < 0.0, "the link must look downward, into the beam"
        angles.append(vertical)

    drop = DEFAULT_HEADLAMP_HEIGHT_M - DEFAULT_PHOTODIODE_HEIGHT_M
    assert angles[0] == pytest.approx(-math.atan2(drop, 10.0), abs=1e-9)
    # Shrinks toward the cut-off as the pair separates.
    assert angles[0] < angles[1] < angles[2] < 0.0


# -- finding 4: the obvious geometry field is the wrong one -------------------


def test_the_beam_is_sampled_at_a_signed_bearing_not_an_absolute_angle() -> None:
    """``emission_angle_rad`` is an absolute value and would fold the beam.

    A pair mirrored about the transmitter's axis produces the *same*
    ``emission_angle_rad`` and opposite ``relative_bearing_rad``.  Sampling an
    asymmetric beam at the former reports the same intensity for a leader on
    the left as for one on the right, which symmetrizes exactly the pattern
    whose asymmetry is the reason for insisting on a measured one.
    """

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    to_the_left = pair_geometry(follower, FakeVehicle("rx", -6.0, 20.0, NORTH))
    to_the_right = pair_geometry(follower, FakeVehicle("rx", 6.0, 20.0, NORTH))

    assert to_the_left.emission_angle_rad == pytest.approx(to_the_right.emission_angle_rad)
    assert to_the_left.relative_bearing_rad == pytest.approx(-to_the_right.relative_bearing_rad)

    beam = asymmetric_beam()
    left_power = power_of(beam, to_the_left)
    right_power = power_of(beam, to_the_right)
    assert left_power != pytest.approx(right_power)


def test_a_leader_to_the_left_is_sampled_on_the_pattern_s_left() -> None:
    """The sign flip, which is invisible on any symmetric pattern.

    Bearings are counter-clockwise-positive, so a positive relative bearing is
    a target to the vehicle's left; the pattern's horizontal axis is positive
    to the right.  A missing flip mirrors the beam and is undetectable until
    someone plots received power against junction turn direction.
    """

    assert horizontal_emission_angle_rad(0.4) == pytest.approx(-0.4)

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    to_the_left = pair_geometry(follower, FakeVehicle("rx", -6.0, 20.0, NORTH))
    assert to_the_left.relative_bearing_rad > 0.0

    pattern = RecordingPattern()
    power_of(pattern, to_the_left)
    horizontal, _ = pattern.samples[0]
    assert horizontal < 0.0


# -- the field of view, applied exactly once ----------------------------------


def test_no_acceptance_cutoff_is_applied_in_the_channel_layer() -> None:
    """Received power is continuous across psi_c.

    The configuration is explicit that a wide cone for availability combined
    with narrow-cone gain in the budget counts the same physics twice.  The
    same argument forbids counting the *cutoff* twice: geometry decides
    acceptance, and this module prices whatever geometry accepted.  A step at
    psi_c would be the second application.
    """

    receiver = OpticalReceiver()
    psi_c = receiver.fov_half_angle_rad

    def power_at(incidence_rad: float) -> float:
        return direct_received_power_w(
            pattern=COMPARISON_BEAM,
            receiver=receiver,
            path_length_m=15.0,
            horizontal_emission_rad=0.0,
            incidence_angle_rad=incidence_rad,
        )

    assert power_at(psi_c + 1e-6) == pytest.approx(power_at(psi_c - 1e-6), rel=1e-5)
    assert power_at(psi_c + 0.2) > 0.0


# -- blockage -----------------------------------------------------------------


def test_a_blocked_path_delivers_nothing_and_never_samples_the_beam() -> None:
    """``complete_blockage_main: true``, and the beam is not consulted.

    Not an optimization.  A severed 0.7 m path delivers the blockage model's
    power however brightly the lamp was pointing, and sampling the pattern
    first invites a later edit that lets a bright beam leak through a bus.
    """

    assert COMPLETE_BLOCKAGE.complete
    assert COMPLETE_BLOCKAGE.occluded_power_w == 0.0

    power = direct_received_power_w(
        pattern=RefusingPattern(),
        receiver=OpticalReceiver(),
        path_length_m=15.0,
        horizontal_emission_rad=0.0,
        incidence_angle_rad=0.0,
        occluded=True,
    )
    assert power == 0.0


def test_blockage_short_circuits_even_an_invalid_path_length() -> None:
    """A blocked link has no meaningful distance to validate.

    Occlusion is decided before the budget, so an occluded pair too close for
    the far-field form still answers zero rather than raising.  Otherwise
    every blocked short-range pair would become an exception.
    """

    assert (
        direct_received_power_w(
            pattern=RefusingPattern(),
            receiver=OpticalReceiver(),
            path_length_m=0.1,
            horizontal_emission_rad=0.0,
            incidence_angle_rad=0.0,
            occluded=True,
        )
        == 0.0
    )


def test_the_residual_floor_cannot_be_enabled_without_a_calibrated_value() -> None:
    """Work plan section 8.1 calls it a *calibrated* residual floor.

    An uncalibrated floor is a free parameter that converts every geometric
    outage into a survivable link -- the one direction the headline result must
    not be nudged by an invented number, since geometric outage is what sets
    the reliability floor the hybrid cannot cross.
    """

    with pytest.raises(OpticalGainError, match="calibrated"):
        BlockageModel(complete=False)


def test_complete_blockage_and_a_floor_are_refused_together() -> None:
    with pytest.raises(OpticalGainError, match="contradictory"):
        BlockageModel(complete=True, residual_floor_w=1e-9)


def test_a_calibrated_floor_is_what_an_occluded_path_then_delivers() -> None:
    """The sensitivity is implementable; it simply needs a number first."""

    floor = BlockageModel(complete=False, residual_floor_w=2.5e-9)
    assert floor.occluded_power_w == pytest.approx(2.5e-9)
    assert (
        direct_received_power_w(
            pattern=RefusingPattern(),
            receiver=OpticalReceiver(),
            path_length_m=15.0,
            horizontal_emission_rad=0.0,
            incidence_angle_rad=0.0,
            occluded=True,
            blockage=floor,
        )
        == pytest.approx(2.5e-9)
    )


def test_a_negative_floor_is_refused() -> None:
    with pytest.raises(OpticalGainError, match="non-negative"):
        BlockageModel(complete=False, residual_floor_w=-1e-9)


# -- near-field validity ------------------------------------------------------


def test_the_near_field_guard_fires_inside_the_configured_separation_window() -> None:
    """Unlike the RF module's equivalent, which never fires for a tagged link.

    The window admits separations from 5 m.  Behind a 4.5 m car that is a 0.5 m
    optical path, and behind a 5.2 m van the follower's bumper is *past* the
    leader's rear -- overlapping footprints.  ``1/d^2`` at 0.5 m returns a
    large fictional number, so the call is refused.
    """

    config = load_headline_config(PROJECT_ROOT)
    shortest_admitted_path_m = config.geometry.min_separation_m - CAR_LENGTH_M
    assert shortest_admitted_path_m < MIN_VALID_PATH_LENGTH_M

    with pytest.raises(OpticalGainError, match="far-field"):
        direct_received_power_w(
            pattern=COMPARISON_BEAM,
            receiver=OpticalReceiver(),
            path_length_m=shortest_admitted_path_m,
            horizontal_emission_rad=0.0,
            incidence_angle_rad=0.0,
        )


def test_the_car_following_minimum_gap_keeps_real_pairs_inside_validity() -> None:
    """With a factor of 2.5 to spare, which is why the guard is a guard.

    The IDM's 2.5 m minimum gap is what actually bounds the optical path from
    below, not the 5 m separation floor.  If either number moved -- a smaller
    minimum gap, or a larger assumed lens aperture -- the two would collide and
    the short end of the window would stop being evaluable.
    """

    minimum_gap_m = IDMParameters().minimum_gap_m
    assert minimum_gap_m > MIN_VALID_PATH_LENGTH_M

    power = direct_received_power_w(
        pattern=COMPARISON_BEAM,
        receiver=OpticalReceiver(),
        path_length_m=minimum_gap_m,
        horizontal_emission_rad=0.0,
        incidence_angle_rad=0.0,
    )
    assert power > 0.0


def test_a_non_finite_path_is_refused() -> None:
    with pytest.raises(OpticalGainError, match="far-field"):
        direct_received_power_w(
            pattern=COMPARISON_BEAM,
            receiver=OpticalReceiver(),
            path_length_m=float("nan"),
            horizontal_emission_rad=0.0,
            incidence_angle_rad=0.0,
        )


def test_a_bearing_outside_the_measured_envelope_still_refuses() -> None:
    """The pattern's own refusal must propagate rather than be swallowed.

    Clamping to the rim would report an intensity for a direction the lamp may
    not illuminate at all, which is precisely the junction geometry that
    decides the V-VLC contribution.
    """

    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    across = pair_geometry(follower, FakeVehicle("rx", 18.0, 6.0, NORTH))

    with pytest.raises(HeadlampPatternError, match="outside the measured envelope"):
        power_of(asymmetric_beam(), across)
