"""Work plan §6.2: what the policy sees is sampled, late, and noisy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.observation.sensing import (
    SensorModel,
    sense_frame,
    sense_vehicle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_SEED = 20260728
TRACE = "synthetic-d20-train-000"


@dataclass(frozen=True, slots=True)
class FakeVehicle:
    vehicle_id: str
    x_m: float = 100.0
    y_m: float = 200.0
    heading_rad: float = 0.0
    speed_mps: float = 8.0


def model(**overrides: float) -> SensorModel:
    base = dict(
        update_s=0.05,
        latency_s=0.05,
        position_noise_std_m=0.5,
        speed_noise_std_mps=0.2,
    )
    base.update(overrides)
    return SensorModel(**base)  # type: ignore[arg-type]


def measure(vehicle: FakeVehicle, now_s: float, sensor: SensorModel | None = None):
    return sense_vehicle(
        vehicle,
        now_s=now_s,
        sensor=sensor or model(),
        root_seed=ROOT_SEED,
        trace_id=TRACE,
    )


# -- the property the whole noise model rests on ------------------------------


def test_the_same_vehicle_at_the_same_tick_reads_identically() -> None:
    """Otherwise a policy could average the error away by looking twice.

    No real receiver has that capability, and granting it would make the noise
    model decorative.
    """

    vehicle = FakeVehicle("v1")
    first = measure(vehicle, 1.0)
    second = measure(vehicle, 1.0)

    assert first == second


def test_reading_one_vehicle_does_not_disturb_another() -> None:
    """Noise is keyed by identity, not by call order.

    Order-dependent draws would make a replay differ from the run it replays
    whenever the vehicle set changed.
    """

    a, b = FakeVehicle("a"), FakeVehicle("b")
    alone = measure(b, 2.0)

    _ = measure(a, 2.0)
    after = measure(b, 2.0)

    assert alone == after


def test_different_vehicles_get_independent_noise() -> None:
    same_state = [FakeVehicle(f"v{i}") for i in range(8)]
    samples = sense_frame(
        same_state, now_s=3.0, sensor=model(), root_seed=ROOT_SEED, trace_id=TRACE
    )

    xs = {round(sample.x_m, 9) for sample in samples}
    assert len(xs) == len(same_state), "identical states must not produce identical noise"


def test_consecutive_ticks_differ() -> None:
    vehicle = FakeVehicle("v1")
    assert measure(vehicle, 1.00).x_m != measure(vehicle, 1.05).x_m


def test_a_different_trace_reseeds_the_sensor() -> None:
    vehicle = FakeVehicle("v1")
    other = sense_vehicle(
        vehicle, now_s=1.0, sensor=model(), root_seed=ROOT_SEED, trace_id="other-trace"
    )
    assert other.x_m != measure(vehicle, 1.0).x_m


# -- ageing -------------------------------------------------------------------


def test_a_measurement_describes_the_past_not_the_present() -> None:
    sample = measure(FakeVehicle("v1"), 1.0)

    assert sample.measured_at_s < 1.0
    assert sample.age_s == pytest.approx(0.05)


def test_between_ticks_the_freshest_sample_simply_gets_older() -> None:
    """Sampling is on a grid, so age is a sawtooth rather than a constant."""

    sensor = model(update_s=0.1, latency_s=0.1)
    at_tick = sensor.latest_measurement_time_s(1.0)
    mid_tick = sensor.latest_measurement_time_s(1.05)

    assert at_tick == mid_tick, "no new sample arrives between ticks"


def test_zero_latency_still_samples_on_the_grid() -> None:
    sensor = model(update_s=0.1, latency_s=0.0)
    assert sensor.latest_measurement_time_s(1.37) == pytest.approx(1.3)


def test_a_sample_is_never_used_before_it_arrives() -> None:
    sensor = model(update_s=0.05, latency_s=0.05)
    for now in (0.31, 0.55, 1.02, 2.999):
        when = sensor.latest_measurement_time_s(now)
        assert when + sensor.latency_s <= now + 1e-9


# -- noise shape --------------------------------------------------------------


def test_noise_is_zero_mean_and_has_the_declared_spread() -> None:
    sensor = model()
    truth = FakeVehicle("v", x_m=0.0, y_m=0.0, speed_mps=8.0)

    errors = []
    for tick in range(4000):
        sample = sense_vehicle(
            truth,
            now_s=0.0,
            sensor=sensor,
            root_seed=ROOT_SEED,
            trace_id=TRACE,
            measured_at_s=tick * sensor.update_s,
        )
        errors.append(sample.x_m)

    mean = sum(errors) / len(errors)
    variance = sum(value**2 for value in errors) / len(errors) - mean**2
    assert abs(mean) < 0.05
    assert math.sqrt(variance) == pytest.approx(sensor.position_noise_std_m, rel=0.08)


def test_speed_never_goes_negative() -> None:
    """A tracker reports a magnitude; a negative would flip predicted heading."""

    crawling = FakeVehicle("slow", speed_mps=0.0)
    sensor = model(speed_noise_std_mps=5.0)

    for tick in range(500):
        sample = sense_vehicle(
            crawling,
            now_s=0.0,
            sensor=sensor,
            root_seed=ROOT_SEED,
            trace_id=TRACE,
            measured_at_s=tick * sensor.update_s,
        )
        assert sample.speed_mps >= 0.0


def test_the_headline_sensor_does_not_hand_over_exact_heading() -> None:
    """Zero heading noise would decide the result.

    §4.6.1 measured a leader turning out of the acceptance cone as the dominant
    cause of junction unavailability.  A policy given exact heading would see
    every turn perfectly at exactly the moment that matters, so the model would
    be cleanest precisely where it is load-bearing.
    """

    config = load_headline_config(PROJECT_ROOT)
    sensor = SensorModel.from_config(config.observation)

    assert sensor.heading_noise_std_rad > 0.0
    vehicle = FakeVehicle("v1", heading_rad=1.234)
    observed = sense_vehicle(
        vehicle, now_s=1.0, sensor=sensor, root_seed=ROOT_SEED, trace_id=TRACE
    ).heading_rad
    assert observed != pytest.approx(1.234)


def test_heading_error_is_broadcast_scale_not_tracker_scale() -> None:
    """Cooperative awareness: heading is a broadcast field, not an inference.

    Its error is the sending vehicle's own attitude error, degrees rather than
    the tens of degrees a tracker would incur inferring heading from successive
    positions at the low speeds of a turn.
    """

    config = load_headline_config(PROJECT_ROOT)
    sensor = SensorModel.from_config(config.observation)

    assert sensor.heading_noise_std_rad == pytest.approx(math.radians(2.0))
    assert sensor.heading_noise_std_rad < math.radians(5.0)


def test_heading_noise_is_zero_mean_with_the_declared_spread() -> None:
    config = load_headline_config(PROJECT_ROOT)
    sensor = SensorModel.from_config(config.observation)
    truth = FakeVehicle("v", heading_rad=0.0)

    errors = [
        sense_vehicle(
            truth, now_s=0.0, sensor=sensor, root_seed=ROOT_SEED,
            trace_id=TRACE, measured_at_s=tick * sensor.update_s,
        ).heading_rad
        for tick in range(3000)
    ]
    mean = sum(errors) / len(errors)
    spread = math.sqrt(sum(e**2 for e in errors) / len(errors) - mean**2)

    assert abs(mean) < math.radians(0.2)
    assert spread == pytest.approx(sensor.heading_noise_std_rad, rel=0.1)


# -- configuration ------------------------------------------------------------


def test_the_headline_configuration_builds_the_declared_sensor() -> None:
    config = load_headline_config(PROJECT_ROOT)
    sensor = SensorModel.from_config(config.observation)

    assert sensor.update_s == pytest.approx(0.05), "20 Hz track updates"
    assert sensor.latency_s == pytest.approx(0.05)
    assert sensor.position_noise_std_m == pytest.approx(0.5)
    assert sensor.speed_noise_std_mps == pytest.approx(0.2)
    assert sensor.heading_noise_std_rad == pytest.approx(math.radians(2.0))


@pytest.mark.parametrize(
    "field,value",
    [
        ("update_s", 0.0),
        ("update_s", -0.1),
        ("update_s", math.nan),
        ("latency_s", -0.1),
        ("position_noise_std_m", -1.0),
        ("speed_noise_std_mps", -1.0),
        ("heading_noise_std_rad", -1.0),
    ],
)
def test_an_unusable_sensor_is_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        model(**{field: value})
