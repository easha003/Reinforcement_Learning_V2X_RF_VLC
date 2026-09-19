"""Where a tracked vehicle will be, and how little that is worth knowing.

Work plan §6.2 specifies a 200 ms constant-velocity prediction horizon and
requires the predictor to *propagate track uncertainty* rather than emit a
perfect future state.  This module does the kinematics and the error growth;
turning that into a blockage probability is
:mod:`hybrid_v2x_rl.observation.blockage`.

**Extrapolation runs from the measurement, not from now.**  A track measured at
``t`` and used at ``t + age`` to predict ``t + age + horizon`` is propagated for
``age + horizon`` seconds, not ``horizon``.  Getting this wrong would make a
stale track look exactly as trustworthy as a fresh one, which is the single
easiest way to accidentally hand the policy a clean future.

**The error ellipse is anisotropic, and both axes grow.**  Speed error
displaces a prediction along the track's heading; heading error displaces it
across:

    along-track  sd = sqrt(position_sd^2 + (speed_sd * elapsed)^2)
    cross-track  sd = sqrt(position_sd^2 + (speed * elapsed * heading_sd)^2)

The cross-track term is *speed-dependent*, which is the physically right shape:
a stationary vehicle cannot be displaced sideways by not knowing where it
points, while a fast one can be displaced a long way.  Under cooperative
awareness heading is broadcast rather than inferred, so its error is small --
but it is not zero, and leaving it at zero would have made the forecast
perfectly certain about the direction a leader is turning, which §4.6.1
identified as the dominant cause of junction unavailability.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hybrid_v2x_rl.observation.tracks import Track


@runtime_checkable
class ForecastConfigSource(Protocol):
    """The subset of ``ObservationConfig`` this module needs."""

    @property
    def forecast_horizon_s(self) -> float: ...
    @property
    def position_noise_std_m(self) -> float: ...
    @property
    def speed_noise_std_mps(self) -> float: ...
    @property
    def heading_noise_std_deg(self) -> float: ...


@dataclass(frozen=True, slots=True)
class PredictedState:
    """One vehicle's extrapolated state, with the uncertainty that earned it."""

    vehicle_id: str
    at_s: float
    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float
    #: Seconds of extrapolation, i.e. track age plus forecast horizon.
    propagated_s: float
    along_track_std_m: float
    cross_track_std_m: float
    #: The sensor's position standard deviation, kept so confidence can be
    #: expressed relative to what a fresh measurement would have been worth.
    base_std_m: float

    @property
    def confidence(self) -> float:
        """Fraction of the sensor's original precision still retained, in (0, 1].

        One at zero propagation, falling as velocity error accumulates.  This
        is §6.1's ``predictor_confidence``: a scalar the policy can act on,
        derived only from declared noise and elapsed time, never from whether
        the prediction happens to be right.
        """

        if self.along_track_std_m <= 0.0:
            return 1.0
        return min(1.0, self.base_std_m / self.along_track_std_m)

    def std_towards(self, bearing_rad: float) -> float:
        """Positional standard deviation along an arbitrary direction.

        The uncertainty is an ellipse aligned with the vehicle's heading, so a
        consumer asking "how uncertain is this across *that* line" has to
        project.  Used by the blockage predictor, where the direction that
        matters is perpendicular to the optical path rather than along the
        blocker's own travel.
        """

        offset = bearing_rad - self.heading_rad
        along = self.along_track_std_m * math.cos(offset)
        across = self.cross_track_std_m * math.sin(offset)
        return math.hypot(along, across)


@dataclass(frozen=True, slots=True)
class ConstantVelocityForecaster:
    """§6.2's predictor: constant velocity, with declared error growth."""

    horizon_s: float
    position_noise_std_m: float
    speed_noise_std_mps: float
    heading_noise_std_rad: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.horizon_s) or self.horizon_s < 0.0:
            raise ValueError("horizon_s must be finite and non-negative")
        for name in ("position_noise_std_m", "speed_noise_std_mps", "heading_noise_std_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_config(cls, observation: ForecastConfigSource) -> ConstantVelocityForecaster:
        """Build from a loaded ``ObservationConfig``.

        Structural rather than typed against the config model, for the same
        reason as :meth:`SensorModel.from_config`: the observation package must
        not depend on the configuration models.
        """

        return cls(
            horizon_s=float(observation.forecast_horizon_s),
            position_noise_std_m=float(observation.position_noise_std_m),
            speed_noise_std_mps=float(observation.speed_noise_std_mps),
            heading_noise_std_rad=math.radians(float(observation.heading_noise_std_deg)),
        )

    def predict(
        self,
        track: Track,
        *,
        now_s: float,
        horizon_s: float | None = None,
    ) -> PredictedState:
        """Extrapolate one track to ``now_s + horizon``.

        ``horizon_s`` overrides the configured horizon, for the sensitivity
        runs §6.2 calls for and for asking "where is it *now*" with a horizon
        of zero, which is still an extrapolation because the measurement is
        already old.
        """

        reach = self.horizon_s if horizon_s is None else horizon_s
        if not math.isfinite(reach) or reach < 0.0:
            raise ValueError("horizon_s must be finite and non-negative")

        elapsed = track.age_s(now_s) + reach
        vx, vy = track.velocity_mps
        sample = track.sample

        along = math.hypot(self.position_noise_std_m, self.speed_noise_std_mps * elapsed)
        # Sideways displacement from pointing error grows with distance
        # travelled, so a stopped vehicle keeps the sensor's own uncertainty.
        cross = math.hypot(
            self.position_noise_std_m,
            sample.speed_mps * elapsed * self.heading_noise_std_rad,
        )
        return PredictedState(
            vehicle_id=track.vehicle_id,
            at_s=now_s + reach,
            x_m=sample.x_m + vx * elapsed,
            y_m=sample.y_m + vy * elapsed,
            heading_rad=sample.heading_rad,
            speed_mps=sample.speed_mps,
            propagated_s=elapsed,
            along_track_std_m=along,
            cross_track_std_m=cross,
            base_std_m=self.position_noise_std_m,
        )

    def predict_all(
        self,
        tracks: Iterable[Track],
        *,
        now_s: float,
        horizon_s: float | None = None,
    ) -> tuple[PredictedState, ...]:
        return tuple(
            self.predict(track, now_s=now_s, horizon_s=horizon_s) for track in tracks
        )


__all__ = [
    "ForecastConfigSource",
    "ConstantVelocityForecaster",
    "PredictedState",
]
