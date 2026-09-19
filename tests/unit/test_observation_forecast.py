"""Work plan §6.2: a 200 ms constant-velocity horizon with propagated error."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.observation.forecast import ConstantVelocityForecaster
from hybrid_v2x_rl.observation.sensing import TrackSample
from hybrid_v2x_rl.observation.tracks import Track

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EAST = 0.0
NORTH = 0.5 * math.pi


def track(
    measured_at_s: float = 1.0,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    speed_mps: float = 10.0,
    heading_rad: float = EAST,
) -> Track:
    return Track(
        sample=TrackSample(
            vehicle_id="v1",
            measured_at_s=measured_at_s,
            observed_at_s=measured_at_s + 0.05,
            x_m=x_m,
            y_m=y_m,
            speed_mps=speed_mps,
            heading_rad=heading_rad,
        ),
        first_seen_s=measured_at_s,
        update_count=1,
    )


def forecaster(**overrides: float) -> ConstantVelocityForecaster:
    base = dict(horizon_s=0.2, position_noise_std_m=0.5, speed_noise_std_mps=0.2)
    base.update(overrides)
    return ConstantVelocityForecaster(**base)  # type: ignore[arg-type]


# -- extrapolation runs from the measurement, not from now --------------------


def test_a_stale_track_is_propagated_further_than_a_fresh_one() -> None:
    """Otherwise a stale track looks exactly as trustworthy as a fresh one,
    which is the easiest way to accidentally hand the policy a clean future."""

    fresh = forecaster().predict(track(measured_at_s=1.0), now_s=1.05)
    stale = forecaster().predict(track(measured_at_s=1.0), now_s=1.55)

    assert fresh.propagated_s == pytest.approx(0.25)
    assert stale.propagated_s == pytest.approx(0.75)
    assert stale.x_m > fresh.x_m
    assert stale.along_track_std_m > fresh.along_track_std_m
    assert stale.confidence < fresh.confidence


def test_propagation_is_age_plus_horizon() -> None:
    state = forecaster(horizon_s=0.2).predict(track(measured_at_s=2.0), now_s=2.10)
    assert state.propagated_s == pytest.approx(0.10 + 0.2)


def test_a_zero_horizon_still_extrapolates_because_the_sample_is_old() -> None:
    state = forecaster().predict(track(measured_at_s=1.0), now_s=1.05, horizon_s=0.0)

    assert state.propagated_s == pytest.approx(0.05)
    assert state.x_m == pytest.approx(0.5), "10 m/s for 50 ms"


# -- kinematics ---------------------------------------------------------------


@pytest.mark.parametrize(
    "heading,expected",
    [
        (EAST, (2.5, 0.0)),
        (NORTH, (0.0, 2.5)),
        (math.pi, (-2.5, 0.0)),
    ],
)
def test_constant_velocity_moves_along_the_measured_heading(
    heading: float, expected: tuple[float, float]
) -> None:
    state = forecaster().predict(track(heading_rad=heading), now_s=1.05)

    assert state.x_m == pytest.approx(expected[0], abs=1e-9)
    assert state.y_m == pytest.approx(expected[1], abs=1e-9)


def test_a_stopped_vehicle_is_predicted_to_stay() -> None:
    state = forecaster().predict(track(speed_mps=0.0, x_m=7.0, y_m=3.0), now_s=1.05)

    assert state.x_m == pytest.approx(7.0)
    assert state.y_m == pytest.approx(3.0)


def test_the_predicted_time_is_reported() -> None:
    state = forecaster(horizon_s=0.2).predict(track(), now_s=1.05)
    assert state.at_s == pytest.approx(1.25)


# -- uncertainty --------------------------------------------------------------


def test_without_heading_error_only_the_along_track_axis_grows() -> None:
    """Speed error displaces along the heading and nothing else.

    Isolated here with heading noise switched off, so the two contributions can
    be seen apart; the headline configuration declares both.
    """

    engine = forecaster(
        position_noise_std_m=0.5, speed_noise_std_mps=0.2, heading_noise_std_rad=0.0
    )
    short = engine.predict(track(), now_s=1.05)
    long = engine.predict(track(), now_s=3.05)

    assert long.along_track_std_m > short.along_track_std_m
    assert short.cross_track_std_m == pytest.approx(0.5)
    assert long.cross_track_std_m == pytest.approx(0.5)


def test_along_track_error_matches_the_declared_combination() -> None:
    engine = forecaster(position_noise_std_m=0.5, speed_noise_std_mps=0.2)
    state = engine.predict(track(measured_at_s=1.0), now_s=1.30, horizon_s=0.2)

    elapsed = 0.30 + 0.2
    assert state.propagated_s == pytest.approx(elapsed)
    assert state.along_track_std_m == pytest.approx(math.hypot(0.5, 0.2 * elapsed))


def test_confidence_is_one_when_nothing_has_been_extrapolated() -> None:
    engine = forecaster(horizon_s=0.0)
    state = engine.predict(track(measured_at_s=1.0), now_s=1.0)

    assert state.propagated_s == pytest.approx(0.0)
    assert state.confidence == pytest.approx(1.0)


def test_confidence_decays_monotonically_with_staleness() -> None:
    engine = forecaster()
    values = [engine.predict(track(measured_at_s=1.0), now_s=now).confidence
              for now in (1.0, 1.5, 2.0, 4.0, 8.0)]

    assert values == sorted(values, reverse=True)
    assert all(0.0 < value <= 1.0 for value in values)


def test_confidence_never_depends_on_whether_the_prediction_was_right() -> None:
    """It is a function of declared noise and elapsed time only.

    Anything else would be the simulator grading its own forecast, which the
    policy is not allowed to see.
    """

    engine = forecaster()
    straight = engine.predict(track(speed_mps=10.0), now_s=1.05)
    stopped = engine.predict(track(speed_mps=0.0), now_s=1.05)

    assert straight.confidence == pytest.approx(stopped.confidence)


# -- projecting the ellipse ---------------------------------------------------


def test_std_towards_returns_the_axes_it_was_built_from() -> None:
    state = forecaster().predict(track(heading_rad=EAST, measured_at_s=1.0), now_s=2.0)

    assert state.std_towards(EAST) == pytest.approx(state.along_track_std_m)
    assert state.std_towards(NORTH) == pytest.approx(state.cross_track_std_m)


def test_std_towards_is_symmetric_under_reversal() -> None:
    state = forecaster().predict(track(), now_s=2.0)
    for bearing in (0.3, 1.1, 2.7):
        assert state.std_towards(bearing) == pytest.approx(state.std_towards(bearing + math.pi))


def test_std_towards_lies_between_the_two_axes() -> None:
    state = forecaster().predict(track(), now_s=3.0)
    lower = min(state.along_track_std_m, state.cross_track_std_m)
    upper = max(state.along_track_std_m, state.cross_track_std_m)

    for bearing in (0.0, 0.4, 0.9, 1.4, 2.2, 3.0, 5.5):
        assert lower - 1e-9 <= state.std_towards(bearing) <= upper + 1e-9


# -- configuration and rejection ---------------------------------------------


def test_the_headline_configuration_builds_the_declared_forecaster() -> None:
    config = load_headline_config(PROJECT_ROOT)
    engine = ConstantVelocityForecaster.from_config(config.observation)

    assert engine.horizon_s == pytest.approx(0.2)
    assert engine.position_noise_std_m == pytest.approx(0.5)
    assert engine.speed_noise_std_mps == pytest.approx(0.2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("horizon_s", -0.1),
        ("horizon_s", math.nan),
        ("position_noise_std_m", -1.0),
        ("speed_noise_std_mps", -1.0),
    ],
)
def test_an_unusable_forecaster_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        forecaster(**{field: value})


def test_a_negative_horizon_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="horizon_s"):
        forecaster().predict(track(), now_s=1.05, horizon_s=-0.1)


def test_predict_all_preserves_order() -> None:
    engine = forecaster()
    many = [track(measured_at_s=1.0, x_m=float(i)) for i in range(5)]
    states = engine.predict_all(many, now_s=1.05)

    # age 0.05 s + horizon 0.2 s = 0.25 s of propagation, at 10 m/s = 2.5 m.
    assert [round(state.x_m, 6) for state in states] == [
        round(float(i) + 2.5, 6) for i in range(5)
    ]


# -- heading error widens the ellipse across the track ------------------------


def test_cross_track_error_now_grows_with_distance_travelled() -> None:
    """Pointing error displaces a prediction sideways, further the faster you go.

    Leaving it at zero would make the forecast perfectly certain about the
    direction a leader is turning, which §4.6.1 identified as the dominant
    cause of junction unavailability.
    """

    engine = forecaster(heading_noise_std_rad=math.radians(2.0))
    near = engine.predict(track(measured_at_s=1.0, speed_mps=10.0), now_s=1.05)
    far = engine.predict(track(measured_at_s=1.0, speed_mps=10.0), now_s=3.05)

    assert far.cross_track_std_m > near.cross_track_std_m
    assert near.cross_track_std_m > engine.position_noise_std_m


def test_a_stationary_vehicle_keeps_the_sensor_uncertainty_sideways() -> None:
    """Not knowing where a stopped car points cannot move it."""

    engine = forecaster(heading_noise_std_rad=math.radians(10.0))
    stopped = engine.predict(track(speed_mps=0.0, measured_at_s=1.0), now_s=5.0)

    assert stopped.cross_track_std_m == pytest.approx(engine.position_noise_std_m)


def test_cross_track_error_matches_the_declared_combination() -> None:
    heading_std = math.radians(2.0)
    engine = forecaster(heading_noise_std_rad=heading_std)
    state = engine.predict(track(measured_at_s=1.0, speed_mps=10.0), now_s=1.30, horizon_s=0.2)

    elapsed = 0.30 + 0.2
    assert state.cross_track_std_m == pytest.approx(
        math.hypot(0.5, 10.0 * elapsed * heading_std)
    )


def test_the_headline_forecaster_carries_the_declared_heading_error() -> None:
    config = load_headline_config(PROJECT_ROOT)
    engine = ConstantVelocityForecaster.from_config(config.observation)

    assert engine.heading_noise_std_rad == pytest.approx(math.radians(2.0))


def test_a_negative_heading_noise_is_rejected() -> None:
    with pytest.raises(ValueError, match="heading_noise_std_rad"):
        forecaster(heading_noise_std_rad=-0.1)
