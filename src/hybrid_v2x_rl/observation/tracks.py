"""What the policy remembers about other vehicles, and how stale it is.

A track is the newest measurement of one vehicle plus the bookkeeping needed to
know how much to trust it.  Work plan §6.1 makes ``track_age`` an observed
feature and §6.3 makes freshness action-dependent, so age is not diagnostic
here -- it is part of the state the policy acts on.

**A vehicle that stops being measured does not disappear; its track ages.**
That distinction is the whole point.  Dropping the track on the first missed
update would hand the policy a clean "unknown" signal, when what a real
receiver has is a confident-looking estimate that is quietly going wrong.  The
forecast in :mod:`hybrid_v2x_rl.observation.forecast` widens its uncertainty with
that age rather than with anything the simulator knows.

The store never reads exact state: it consumes :class:`TrackSample` values,
which :mod:`hybrid_v2x_rl.observation.sensing` has already degraded.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from hybrid_v2x_rl.observation.sensing import TrackSample


@runtime_checkable
class TrackConfigSource(Protocol):
    """The subset of ``ObservationConfig`` this module needs."""

    @property
    def track_lifetime_s(self) -> float: ...


@dataclass(frozen=True, slots=True)
class Track:
    """The newest measurement of one vehicle, with its provenance."""

    sample: TrackSample
    first_seen_s: float
    update_count: int

    @property
    def vehicle_id(self) -> str:
        return self.sample.vehicle_id

    def age_s(self, now_s: float) -> float:
        """How old this estimate is at ``now_s``.

        Measured from when the state was true, not from when it arrived, so
        sensor latency is included.  A policy that treats a 50 ms-late sample
        as current is wrong by exactly that latency.
        """

        return max(0.0, now_s - self.sample.measured_at_s)

    @property
    def velocity_mps(self) -> tuple[float, float]:
        """Ground velocity implied by the measured speed and heading.

        Note the asymmetry this inherits: §6.2 declares speed noise but no
        heading noise, so the error here is along-track only.  A real tracker's
        cross-track error is not represented; see
        :class:`~hybrid_v2x_rl.observation.sensing.SensorModel`.
        """

        return (
            self.sample.speed_mps * math.cos(self.sample.heading_rad),
            self.sample.speed_mps * math.sin(self.sample.heading_rad),
        )

    def is_stale(self, now_s: float, *, limit_s: float) -> bool:
        return self.age_s(now_s) > limit_s


@dataclass(slots=True)
class TrackStore:
    """Newest track per vehicle, with optional forgetting.

    ``forget_after_s`` is the awareness-message timeout: a neighbour unheard
    for that long is dropped.  Under cooperative awareness that is what losing
    a neighbour actually means -- its broadcasts stopped -- rather than its
    leaving some sensor's field of view.

    It matters because ``neighbour_count`` is the RF-load proxy.  Unbounded, a
    vehicle that drives away is counted for the rest of the episode and the
    proxy only ever rises.  The default here is still ``None``, so a caller
    that has no configuration cannot silently inherit a number nobody chose;
    :meth:`from_config` supplies the declared one.
    """

    forget_after_s: float | None = None
    _tracks: dict[str, Track] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.forget_after_s is not None and (
            not math.isfinite(self.forget_after_s) or self.forget_after_s <= 0.0
        ):
            raise ValueError("forget_after_s must be finite and positive when set")

    @classmethod
    def from_config(cls, observation: TrackConfigSource) -> TrackStore:
        """Build with the declared awareness-message timeout."""

        return cls(forget_after_s=float(observation.track_lifetime_s))

    def __len__(self) -> int:
        return len(self._tracks)

    def __contains__(self, vehicle_id: object) -> bool:
        return vehicle_id in self._tracks

    def update(self, samples: Iterable[TrackSample], *, now_s: float) -> None:
        """Ingest one sensed frame.

        A sample that is not newer than the stored one is discarded rather than
        applied.  Measurements can arrive out of order once latency varies, and
        letting an older reading overwrite a newer one would make the track age
        jump backwards -- an observable feature moving the wrong way.
        """

        for sample in samples:
            existing = self._tracks.get(sample.vehicle_id)
            if existing is None:
                self._tracks[sample.vehicle_id] = Track(
                    sample=sample,
                    first_seen_s=sample.measured_at_s,
                    update_count=1,
                )
                continue
            if sample.measured_at_s <= existing.sample.measured_at_s:
                continue
            self._tracks[sample.vehicle_id] = replace(
                existing,
                sample=sample,
                update_count=existing.update_count + 1,
            )

        if self.forget_after_s is not None:
            self._forget(now_s)

    def _forget(self, now_s: float) -> None:
        limit = self.forget_after_s
        assert limit is not None
        for vehicle_id in [
            vehicle_id
            for vehicle_id, track in self._tracks.items()
            if track.is_stale(now_s, limit_s=limit)
        ]:
            del self._tracks[vehicle_id]

    def get(self, vehicle_id: str) -> Track | None:
        return self._tracks.get(vehicle_id)

    def require(self, vehicle_id: str) -> Track:
        track = self._tracks.get(vehicle_id)
        if track is None:
            raise KeyError(f"no track for {vehicle_id!r}")
        return track

    def tracks(self) -> Iterator[Track]:
        """Every held track, in insertion order so iteration is reproducible."""

        return iter(tuple(self._tracks.values()))

    def fresh_tracks(self, now_s: float, *, limit_s: float) -> tuple[Track, ...]:
        """Tracks no older than ``limit_s``, for consumers that need a cutoff."""

        return tuple(
            track for track in self._tracks.values() if not track.is_stale(now_s, limit_s=limit_s)
        )

    def neighbour_count(
        self,
        centre: Track,
        *,
        radius_m: float,
        now_s: float,
        limit_s: float | None = None,
    ) -> int:
        """Tracked vehicles within ``radius_m`` of ``centre``, excluding it.

        Counted from *tracks*, not from exact positions, so it inherits the
        sensor's error and its staleness.  This is the RF-load proxy of §6.1
        and it must degrade with the rest of the observation, not be handed
        over exactly.
        """

        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")

        found = 0
        for track in self._tracks.values():
            if track.vehicle_id == centre.vehicle_id:
                continue
            if limit_s is not None and track.is_stale(now_s, limit_s=limit_s):
                continue
            dx = track.sample.x_m - centre.sample.x_m
            dy = track.sample.y_m - centre.sample.y_m
            if dx * dx + dy * dy <= radius_m * radius_m:
                found += 1
        return found


__all__ = [
    "Track",
    "TrackConfigSource",
    "TrackStore",
]
