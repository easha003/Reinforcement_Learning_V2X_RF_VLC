"""Noisy, delayed measurements of other vehicles.

**This module is the leakage barrier.**  It is the one place in
:mod:`hybrid_v2x_rl.observation` that reads exact simulator state, and everything it
returns is degraded: sampled at a finite rate, delivered late, and corrupted by
zero-mean noise.  Work plan §6.2 declares the numbers as simulation
assumptions, not as claimed sensor performance.

Three properties are load-bearing and are asserted by the tests.

**Noise is a function of (vehicle, measurement tick), not of the call.**  A
caller that reads the same vehicle at the same tick twice gets the same answer.
Drawing fresh noise per read would let a policy average the error away by
observing repeatedly, which is not a capability any real receiver has, and it
would make the whole noise model decorative.

**Measurements are aged.**  A sample available at time *t* describes the world
at *t - latency*.  The policy therefore never sees the present, which is what
makes the 200 ms forecast of §6.1 necessary rather than ornamental.

**Sampling is on a fixed grid.**  Tracks update at ``1 / track_update_s``, so
between ticks the freshest available sample simply gets older.  That age is an
observable feature, and §6.3 makes it action-dependent.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from hybrid_v2x_rl.core.randomness import derive_seed

#: Stream name for sensor noise, kept distinct from mobility and channel draws
#: so that changing one does not perturb the others.
SENSOR_STREAM = "sensor_noise"


@runtime_checkable
class VehicleStateSource(Protocol):
    """The exact state a sensor observes.  Structural, so traces and test
    doubles both satisfy it without inheriting anything."""

    @property
    def vehicle_id(self) -> str: ...
    @property
    def x_m(self) -> float: ...
    @property
    def y_m(self) -> float: ...
    @property
    def heading_rad(self) -> float: ...
    @property
    def speed_mps(self) -> float: ...


@runtime_checkable
class SensorConfigSource(Protocol):
    """The subset of ``ObservationConfig`` this module needs.

    Structural rather than imported, so the observation package does not depend
    on the configuration models and cannot grow a cycle with them.
    """

    @property
    def track_update_s(self) -> float: ...
    @property
    def track_latency_s(self) -> float: ...
    @property
    def position_noise_std_m(self) -> float: ...
    @property
    def speed_noise_std_mps(self) -> float: ...
    @property
    def heading_noise_std_deg(self) -> float: ...


@dataclass(frozen=True, slots=True)
class TrackSample:
    """One noisy measurement of one vehicle.

    ``measured_at_s`` is when the state was true; ``observed_at_s`` is when it
    became available to the policy.  Keeping both is what lets age be computed
    honestly rather than assumed to be zero.
    """

    vehicle_id: str
    measured_at_s: float
    observed_at_s: float
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float

    @property
    def age_s(self) -> float:
        """How stale this sample is at the moment it arrives."""

        return self.observed_at_s - self.measured_at_s


@dataclass(frozen=True, slots=True)
class SensorModel:
    """Work plan §6.2's declared sensing assumptions.

    Sensing represents **cooperative awareness**: neighbours are known from the
    status messages they broadcast, not from onboard tracking of uncooperative
    vehicles.  Heading is therefore a *broadcast field* and its error is the
    sender's own GNSS/IMU attitude error, which is small -- rather than the far
    larger error a tracker would incur inferring heading from successive
    positions, worst exactly at the low speeds of a turn.

    It is deliberately not zero.  §4.6.1 measured a leader turning out of the
    acceptance cone as the dominant cause of junction unavailability, so a
    policy handed exact heading would see every turn perfectly at precisely the
    moment that decides the result.
    """

    update_s: float
    latency_s: float
    position_noise_std_m: float
    speed_noise_std_mps: float
    heading_noise_std_rad: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.update_s) or self.update_s <= 0.0:
            raise ValueError("update_s must be finite and positive")
        if not math.isfinite(self.latency_s) or self.latency_s < 0.0:
            raise ValueError("latency_s must be finite and non-negative")
        for name in ("position_noise_std_m", "speed_noise_std_mps", "heading_noise_std_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_config(cls, observation: SensorConfigSource) -> SensorModel:
        """Build from a loaded ``ObservationConfig``, structurally.

        Heading noise is declared in degrees, because that is the unit its
        source is quoted in, and converted here so the model stays in radians.
        """

        return cls(
            update_s=float(observation.track_update_s),
            latency_s=float(observation.track_latency_s),
            position_noise_std_m=float(observation.position_noise_std_m),
            speed_noise_std_mps=float(observation.speed_noise_std_mps),
            heading_noise_std_rad=math.radians(float(observation.heading_noise_std_deg)),
        )

    def tick_of(self, time_s: float) -> int:
        """Index of the most recent sampling tick at or before ``time_s``."""

        return int(math.floor(time_s / self.update_s + 1e-9))

    def latest_measurement_time_s(self, now_s: float) -> float:
        """When the freshest *available* measurement was taken.

        A measurement taken at ``t`` is only usable from ``t + latency``, so
        this walks back until it finds one that has had time to arrive.
        """

        tick = self.tick_of(now_s - self.latency_s)
        return tick * self.update_s


def _noise_for(
    vehicle_id: str,
    tick: int,
    *,
    root_seed: int,
    trace_id: str,
) -> tuple[float, float, float, float]:
    """Four standard normals, fixed by identity rather than by call order.

    Deriving from ``(trace, vehicle, tick)`` means the draw does not depend on
    how many vehicles were sensed first, or on whether this one was sensed at
    all on the previous step.  Order-dependent noise would make a replay differ
    from the run it replays.
    """

    seed = derive_seed(
        root_seed,
        SENSOR_STREAM,
        trace_id=f"{trace_id}:{vehicle_id}",
        packet_index=tick,
    )
    draws = np.random.default_rng(seed).standard_normal(4)
    return float(draws[0]), float(draws[1]), float(draws[2]), float(draws[3])


def sense_vehicle(
    vehicle: VehicleStateSource,
    *,
    now_s: float,
    sensor: SensorModel,
    root_seed: int,
    trace_id: str,
    measured_at_s: float | None = None,
) -> TrackSample:
    """Measure one vehicle as the policy would see it at ``now_s``.

    ``measured_at_s`` is normally derived from ``now_s`` and the latency; pass
    it only when the caller has already resolved which frame the exact state
    came from, so the two cannot disagree.
    """

    when = sensor.latest_measurement_time_s(now_s) if measured_at_s is None else measured_at_s
    tick = sensor.tick_of(when)
    dx, dy, dspeed, dheading = _noise_for(
        vehicle.vehicle_id, tick, root_seed=root_seed, trace_id=trace_id
    )

    return TrackSample(
        vehicle_id=vehicle.vehicle_id,
        measured_at_s=when,
        observed_at_s=when + sensor.latency_s,
        x_m=vehicle.x_m + dx * sensor.position_noise_std_m,
        y_m=vehicle.y_m + dy * sensor.position_noise_std_m,
        # Speed cannot be negative however unlucky the draw; a tracker reports
        # a magnitude, and letting it go negative would flip predicted heading
        # in the forecast downstream.
        speed_mps=max(0.0, vehicle.speed_mps + dspeed * sensor.speed_noise_std_mps),
        heading_rad=vehicle.heading_rad + dheading * sensor.heading_noise_std_rad,
    )


def sense_frame(
    vehicles: Iterable[VehicleStateSource],
    *,
    now_s: float,
    sensor: SensorModel,
    root_seed: int,
    trace_id: str,
    measured_at_s: float | None = None,
) -> tuple[TrackSample, ...]:
    """Measure every vehicle in one exact frame, in the order given."""

    present: Sequence[VehicleStateSource] = tuple(vehicles)
    return tuple(
        sense_vehicle(
            vehicle,
            now_s=now_s,
            sensor=sensor,
            root_seed=root_seed,
            trace_id=trace_id,
            measured_at_s=measured_at_s,
        )
        for vehicle in present
    )


__all__ = [
    "SENSOR_STREAM",
    "SensorConfigSource",
    "SensorModel",
    "TrackSample",
    "VehicleStateSource",
    "sense_frame",
    "sense_vehicle",
]
