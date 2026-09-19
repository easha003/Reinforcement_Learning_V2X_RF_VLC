"""The optical front end, and the two things about it that constrain the design.

The load-bearing tests here are the ones that would fail if someone broke the
single-semi-angle rule -- a wide cone for availability with narrow-cone gain in
the budget counts the same physics twice -- and the ones that pin the
area/bandwidth/noise coupling, which is what stops a larger photodiode reading
as free signal gain.

Everything else is a guard on constants that carry no source ID yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.vlc.receiver import (
    ACTIVE_AREA_M2,
    DEFAULT_LOAD_RESISTANCE_OHM,
    OpticalReceiver,
    ReceiverError,
)
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.link_endpoints import (
    DEFAULT_HEADLAMP_HEIGHT_M,
    DEFAULT_PHOTODIODE_HEIGHT_M,
)
from hybrid_v2x_rl.core.pair_geometry import DEFAULT_FOV_HALF_ANGLE_RAD, pair_geometry, wrap_to_pi

#: ECE R112 Rev.4 para 6.2.4 specifies the passing beam only to 4 degrees
#: below the horizon; zone I is its lowest band.
R112_LOWEST_SPECIFIED_DEG = 4.0
#: IDM minimum bumper gap, from configs/mobility.
IDM_MINIMUM_GAP_M = 2.5

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


# -- one semi-angle, used twice ----------------------------------------------


def test_the_receiver_uses_the_same_semi_angle_as_the_acceptance_test() -> None:
    """Work plan section 20 item 3, stated as a test rather than a warning.

    Concentrator gain goes as ``n^2 / sin^2 psi_c`` and the acceptance cone is
    ``psi <= psi_c``.  If the two ever take different values, the link budget
    collects light the acceptance test admitted from a cone it did not price,
    or prices a cone it did not admit.  The default is imported from the
    geometry layer precisely so there is nothing to keep in step.
    """

    assert OpticalReceiver().fov_half_angle_rad == DEFAULT_FOV_HALF_ANGLE_RAD


def test_the_shared_semi_angle_is_the_configured_one() -> None:
    """And the shared value is what the frozen configuration actually declares.

    If the headline configuration moves to the 30-degree concentrated front
    end, this fails and every VLC number computed at 60 degrees is due a
    revisit -- which is the right moment to be interrupted.
    """

    config = load_headline_config(PROJECT_ROOT)
    assert OpticalReceiver().fov_half_angle_rad == pytest.approx(
        math.radians(config.vlc.receiver_fov_deg)
    )


def test_a_full_cone_opening_is_refused_where_a_semi_angle_is_expected() -> None:
    """The 120-degree misreading is the one that would silently pass.

    ``receiver_fov_deg: 60`` means a 120-degree total opening.  Someone
    converting "120 degree field of view" into this field is making the error
    the configuration comment exists to prevent, and it would inflate the
    acceptance cone while deflating the gain.
    """

    with pytest.raises(ReceiverError, match="semi-angle"):
        OpticalReceiver(fov_half_angle_rad=math.radians(120.0))


# -- what the cone costs ------------------------------------------------------


def test_widening_the_cone_loses_gain_rather_than_gaining_coverage_free() -> None:
    """The pessimism the configuration claims, measured.

    60 degrees against 30 degrees is a factor of three in collected power:
    4.8 dB optical.  Because IM/DD detection squares the optical power, that is
    9.5 dB electrical -- so the headline front end is nearly 10 dB weaker than
    the sensitivity configuration, and any V-VLC result it produces is
    correspondingly conservative.
    """

    wide = OpticalReceiver(fov_half_angle_rad=math.radians(60.0))
    narrow = OpticalReceiver(fov_half_angle_rad=math.radians(30.0))

    ratio = narrow.concentrator_gain / wide.concentrator_gain
    assert ratio == pytest.approx(3.0)
    assert 10.0 * math.log10(ratio) == pytest.approx(4.77, abs=0.01)
    assert 20.0 * math.log10(ratio) == pytest.approx(9.54, abs=0.01)


def test_the_angular_response_inside_the_cone_is_smaller_than_the_price_of_the_cone() -> None:
    """A consequence worth naming: the wide cone buys very little selectivity.

    Across the entire 60-degree acceptance cone the effective area falls by
    ``cos 60 = 0.5``, exactly 3 dB optical.  The cone itself costs 4.8 dB
    optical against the narrow alternative.  So the receiver gives up more to
    have the wide cone than the wide cone's whole interior dynamic range --
    which is why the availability-versus-budget trade cannot be settled by
    looking at received power alone.
    """

    receiver = OpticalReceiver(fov_half_angle_rad=math.radians(60.0))
    boresight = receiver.effective_area_m2(0.0)
    rim = receiver.effective_area_m2(math.radians(60.0))

    assert rim / boresight == pytest.approx(0.5)
    assert 10.0 * math.log10(boresight / rim) == pytest.approx(3.01, abs=0.01)


# -- the cutoff that must not be applied twice --------------------------------


def test_effective_area_is_continuous_across_the_acceptance_boundary() -> None:
    """No field-of-view cutoff lives in the channel layer.

    A discontinuity at psi_c would mean the acceptance test had been applied a
    second time here, after the geometry layer already applied it.  The
    configuration warns about exactly that double count.  Continuity is the
    observable form of "applied in exactly one place".
    """

    receiver = OpticalReceiver()
    psi_c = receiver.fov_half_angle_rad
    inside = receiver.effective_area_m2(psi_c - 1e-6)
    outside = receiver.effective_area_m2(psi_c + 1e-6)

    assert outside == pytest.approx(inside, rel=1e-5)
    assert outside > 0.0


def test_effective_area_keeps_falling_well_beyond_the_cone() -> None:
    """And the value beyond psi_c is a cosine, not a zero and not a cliff.

    It is also not physically deliverable -- an ideal concentrator passes
    nothing beyond psi_c -- which is why this module's output is conditional on
    the caller having tested acceptance first.  Recorded here so the
    conditionality is not mistaken for a modelling oversight.
    """

    receiver = OpticalReceiver(fov_half_angle_rad=math.radians(60.0))
    at_80 = receiver.effective_area_m2(math.radians(80.0))
    expected = (
        ACTIVE_AREA_M2
        * receiver.filter_transmission
        * receiver.concentrator_gain
        * math.cos(math.radians(80.0))
    )
    assert at_80 == pytest.approx(expected)


def test_a_ray_from_behind_the_detector_is_refused_rather_than_zeroed() -> None:
    """Returning zero would look like an ordinary blocked link.

    Since psi_c can never exceed 90 degrees, the geometry layer has already
    rejected any ray arriving at the back of the photodiode.  Reaching this
    function with one means the acceptance test was skipped, and a plausible
    zero would hide that.
    """

    receiver = OpticalReceiver()
    with pytest.raises(ReceiverError, match="back of the detector"):
        receiver.effective_area_m2(math.radians(90.0))
    with pytest.raises(ReceiverError, match="back of the detector"):
        receiver.effective_area_m2(math.radians(140.0))


def test_the_incidence_angle_carries_no_sign() -> None:
    """It is measured off the boresight, so a negative value is a caller error."""

    with pytest.raises(ReceiverError, match="non-negative"):
        OpticalReceiver().effective_area_m2(-0.1)


# -- area is not a free parameter ---------------------------------------------


def test_the_configured_front_end_is_realizable_at_its_declared_bandwidth() -> None:
    """The headline 5 MHz, a 1 cm^2 detector and 250 ohm have to coexist.

    This is a genuine cross-check between two configuration files and a module
    constant, not a restatement: the RC pole of 70 pF against 250 ohm must stay
    above the electrical bandwidth the timing feasibility check already assumed
    when it accepted 4.5 Mbit/s.

    Sourcing the detector to a real BPW34 *raised* this ceiling from 284 to
    455 ohm, because the datasheet part is 7.5 mm^2 against the 100 mm^2 that
    had been assumed. The front end got easier to build at the same time as the
    optical budget got 11.25 dB harder -- the two move oppositely with area,
    which is the trade the module exists to make explicit.
    """

    config = load_headline_config(PROJECT_ROOT)
    receiver = OpticalReceiver()

    receiver.check_front_end_feasible(config.vlc.electrical_bandwidth_hz)
    assert receiver.max_load_resistance_ohm(config.vlc.electrical_bandwidth_hz) == pytest.approx(
        454.7, abs=0.5
    )
    assert DEFAULT_LOAD_RESISTANCE_OHM < 454.7


def test_a_larger_detector_forces_the_load_resistance_down() -> None:
    """Nine times the area, and the same 250 ohm becomes infeasible.

    The ceiling falls to roughly 51 ohm.  This is the mechanism behind the
    module's central claim: capacitance scales with area, the RC pole must
    still pass the declared bandwidth, and thermal noise varies as ``1/R_L``.
    """

    big = OpticalReceiver(active_area_m2=9.0 * ACTIVE_AREA_M2)
    assert big.max_load_resistance_ohm(5e6) == pytest.approx(50.5, abs=0.3)

    with pytest.raises(ReceiverError, match="RC ceiling"):
        big.check_front_end_feasible(5e6)


def test_detector_area_buys_three_decibels_not_six() -> None:
    """The finding, in the form the optical design has to respect.

    Signal photocurrent goes as area, so electrical signal power goes as area
    squared.  Thermal noise goes as ``1/R_L``, and the bandwidth constraint
    forces ``R_L`` to fall as ``1/area``, so noise also goes as area.  The ratio
    is linear in area.  A designer reaching for a bigger photodiode to recover
    the 9.5 dB the wide cone gave up would need a nine-fold larger one -- and
    would then be holding a detector whose load resistance is capped at 32 ohm.

    Concentrator gain pays no such tax, because it raises collected power
    without touching junction capacitance.  That asymmetry is why the semi-angle
    is the lever and area is not.
    """

    bandwidth_hz = 5e6

    def thermal_limited_snr_scale(area_m2: float) -> float:
        receiver = OpticalReceiver(active_area_m2=area_m2)
        ceiling = receiver.max_load_resistance_ohm(bandwidth_hz)
        # (signal power) / (thermal noise power) up to constants:
        # (R * P * A)^2 * R_L, with R_L at its bandwidth-limited maximum.
        return area_m2 * area_m2 * ceiling

    single = thermal_limited_snr_scale(ACTIVE_AREA_M2)
    double = thermal_limited_snr_scale(2.0 * ACTIVE_AREA_M2)
    ninefold = thermal_limited_snr_scale(9.0 * ACTIVE_AREA_M2)

    assert double / single == pytest.approx(2.0)
    assert 10.0 * math.log10(double / single) == pytest.approx(3.01, abs=0.01)
    # Nine times the area to recover what one halving of the semi-angle gives.
    assert 10.0 * math.log10(ninefold / single) == pytest.approx(9.54, abs=0.01)


def test_a_capacitance_free_detector_has_no_ceiling() -> None:
    """The degenerate case answers honestly instead of dividing by zero."""

    ideal = OpticalReceiver(capacitance_per_area_f_per_m2=0.0)
    assert ideal.max_load_resistance_ohm(5e6) == math.inf
    ideal.check_front_end_feasible(5e6)


# -- orientation --------------------------------------------------------------


def test_the_photodiode_looks_backwards_and_geometry_agrees() -> None:
    """Two modules express the same mounting rule; this checks they agree.

    ``pair_geometry`` derives the incidence angle from the receiver's heading
    directly.  This module states the boresight explicitly.  Computing the
    incidence angle the long way -- the angle between the direction the ray
    arrives from and the declared boresight -- must reproduce it, or one of the
    two has quietly mounted the photodiode facing forwards.
    """

    receiver_optics = OpticalReceiver()
    follower = FakeVehicle("tx", 0.0, 0.0, NORTH)
    leader = FakeVehicle("rx", 6.0, 20.0, NORTH)

    geometry = pair_geometry(follower, leader)
    arrival_from = wrap_to_pi(geometry.bearing_rad + math.pi)
    the_long_way = abs(wrap_to_pi(arrival_from - receiver_optics.boresight_rad(leader.heading_rad)))

    assert the_long_way == pytest.approx(geometry.incidence_angle_rad)
    assert geometry.incidence_angle_rad > 0.0


def test_the_photodiode_sits_below_the_headlamp() -> None:
    """Below it, but only just -- and the margin is what the drop is chosen for.

    The endpoints were equal until it was measured that co-height endpoints
    sample an ECE low beam along its cut-off. Occlusion is unaffected either
    way, since the shortest vehicle class is 1.5 m and blocks a path at any of
    these heights.

    The magnitude is not a preference. The vertical emission angle is
    ``-atan(dh / d_o)``, so the drop decides which row of the beam a pair reads
    and therefore whether that row is one ECE R112 specifies at all: the
    regulation stops at 4D, and below it states neither a minimum nor a
    maximum. The drop must be small enough that the 4D crossing falls under the
    car-following model's minimum gap, so that essentially every pair is
    sampled inside the regulated domain rather than in extrapolation. A
    0.30 m drop crossed 4D at 4.29 m and put 56% of density-30 pair-instants
    outside it; this is the assertion that would have caught that.
    """

    assert OpticalReceiver().height_m == DEFAULT_PHOTODIODE_HEIGHT_M
    assert DEFAULT_PHOTODIODE_HEIGHT_M < DEFAULT_HEADLAMP_HEIGHT_M

    drop = DEFAULT_HEADLAMP_HEIGHT_M - DEFAULT_PHOTODIODE_HEIGHT_M
    crossing_m = drop / math.tan(math.radians(R112_LOWEST_SPECIFIED_DEG))
    assert crossing_m < IDM_MINIMUM_GAP_M, (
        f"a {drop:.2f} m drop reaches R112's 4D limit at {crossing_m:.2f} m, "
        f"above the {IDM_MINIMUM_GAP_M} m minimum gap, so close pairs would "
        f"sample directions the regulation does not describe"
    )


def test_photocurrent_is_linear_in_optical_power() -> None:
    """The one place responsivity is applied, and it is applied once."""

    receiver = OpticalReceiver()
    assert receiver.photocurrent_a(0.0) == 0.0
    assert receiver.photocurrent_a(2e-6) == pytest.approx(
        2.0 * receiver.photocurrent_a(1e-6)
    )


def test_negative_optical_power_is_refused() -> None:
    with pytest.raises(ReceiverError, match="non-negative"):
        OpticalReceiver().photocurrent_a(-1e-9)


# -- guards on unsourced constants --------------------------------------------


def test_a_filter_cannot_pass_more_light_than_reaches_it() -> None:
    with pytest.raises(ReceiverError, match="filter transmission"):
        OpticalReceiver(filter_transmission=1.2)


def test_a_sub_unity_refractive_index_is_refused() -> None:
    with pytest.raises(ReceiverError, match="refractive index"):
        OpticalReceiver(concentrator_refractive_index=0.9)


def test_an_excess_noise_factor_below_one_is_refused() -> None:
    """It multiplies the resistor's own thermal noise; below 1 is free energy."""

    with pytest.raises(ReceiverError, match="cannot be below 1"):
        OpticalReceiver(preamplifier_noise_factor=0.5)


@pytest.mark.parametrize(
    "field",
    ["active_area_m2", "responsivity_a_per_w", "load_resistance_ohm", "noise_temperature_k"],
)
def test_the_positive_quantities_are_checked_by_name(field: str) -> None:
    with pytest.raises(ReceiverError, match=field):
        OpticalReceiver(**{field: 0.0})


def test_the_receiver_computes_no_noise_power() -> None:
    """Spec 12.4 owns noise, and the split has to be visible.

    The receiver holds the *parameters* -- load resistance, temperature, excess
    noise factor, capacitance -- and forms no product of them, so the night and
    day ambient conditions stay separate configurations rather than a runtime
    boolean reaching in here for constants.  Boltzmann's constant appearing in
    this module would be the first sign that boundary had moved.
    """

    import hybrid_v2x_rl.channels.vlc.receiver as module

    source = Path(module.__file__ or "").read_text(encoding="utf-8")
    assert "1.380649e-23" not in source
    assert not hasattr(module, "BOLTZMANN_J_PER_K")
    for forbidden in ("def thermal_noise", "def shot_noise", "def total_noise", "def snr"):
        assert forbidden not in source


def test_the_acceptance_cone_reaches_the_receiver_as_well_as_the_geometry() -> None:
    """One semi-angle, and both uses must move together.

    The module docstring promises "there is one number in the codebase and the
    two uses cannot drift apart", but the promise lives in the *assembly*: the
    geometry test reads the configured semi-angle while the concentrator gain
    reads whatever the receiver was constructed with. Building the receiver
    without passing it left the gain pinned at the module default, so a profile
    narrowing the cone paid the availability cost of a narrow cone and received
    the gain of a wide one.

    Asserting it here rather than in the config tests is deliberate: the defect
    was in how the two are wired together, not in either alone.
    """

    import math
    from pathlib import Path

    from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
    from hybrid_v2x_rl.env.assembly import build_rollout, build_vlc_channel

    config = load_config(headline_config_layers(Path.cwd()), project_root=Path.cwd())
    for degrees in (60.0, 30.0):
        tuned = config.model_copy(
            update={"vlc": config.vlc.model_copy(
                update={"receiver_fov_deg": degrees})}, deep=True)
        channel = build_vlc_channel(tuned)
        rollout = build_rollout(tuned, buildings=())
        assert math.degrees(channel.receiver.fov_half_angle_rad) == pytest.approx(degrees)
        assert math.degrees(rollout.fov_half_angle_rad) == pytest.approx(degrees)
        # The two uses are the same number, so they agree exactly.
        assert channel.receiver.fov_half_angle_rad == pytest.approx(
            rollout.fov_half_angle_rad)
