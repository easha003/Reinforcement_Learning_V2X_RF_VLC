"""Deterministic extraction of natural longitudinal V2V pair episodes.

The extractor consumes immutable vehicle observations (or row-like mappings)
and never changes a vehicle position, creates a vehicle, or extends an episode
beyond observed data.  A transmitter is the following vehicle and a receiver
is a vehicle ahead on the same lane, with the same planned route and travel
direction at the episode start.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

PairEndReason: TypeAlias = Literal[
    "trace_end",
    "vehicle_missing",
    "route_diverged",
    "outside_range_1s",
    "max_duration",
]

_PAIR_END_REASONS: frozenset[str] = frozenset(
    {
        "trace_end",
        "vehicle_missing",
        "route_diverged",
        "outside_range_1s",
        "max_duration",
    }
)
_TIME_TOLERANCE_S = 1e-9


@runtime_checkable
class VehicleObservation(Protocol):
    """Structural input accepted from trace readers without a hard dependency."""

    trace_id: str
    time_s: float
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    route_id: str
    lane_id: str


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _required_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _optional_route(row: Mapping[str, object]) -> tuple[str, ...] | None:
    value = row.get("planned_route")
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("planned_route must be a sequence of edge identifiers")
    route: list[str] = []
    for edge in value:
        if not isinstance(edge, str) or not edge.strip():
            raise ValueError("planned_route entries must be non-empty strings")
        route.append(edge.strip())
    if not route:
        raise ValueError("planned_route cannot be empty")
    return tuple(route)


@dataclass(frozen=True, slots=True)
class VehicleRecord:
    """One observed vehicle state from an immutable mobility trace."""

    trace_id: str
    time_s: float
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    route_id: str
    lane_id: str
    planned_route: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in ("trace_id", "vehicle_id", "route_id", "lane_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        for field_name in ("time_s", "x_m", "y_m", "heading_rad"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.time_s < 0.0:
            raise ValueError("time_s cannot be negative")
        if self.planned_route is not None:
            if not self.planned_route:
                raise ValueError("planned_route cannot be empty")
            if any(not edge.strip() for edge in self.planned_route):
                raise ValueError("planned_route entries must be non-empty strings")

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> VehicleRecord:
        """Construct a record from a Parquet/Pandas-style row mapping."""

        return cls(
            trace_id=_required_text(row, "trace_id"),
            time_s=_required_float(row, "time_s"),
            vehicle_id=_required_text(row, "vehicle_id"),
            x_m=_required_float(row, "x_m"),
            y_m=_required_float(row, "y_m"),
            heading_rad=_required_float(row, "heading_rad"),
            route_id=_required_text(row, "route_id"),
            lane_id=_required_text(row, "lane_id"),
            planned_route=_optional_route(row),
        )

    @classmethod
    def from_observation(cls, observation: VehicleObservation) -> VehicleRecord:
        """Adapt a structural trace-reader record without importing PyArrow."""

        return cls(
            trace_id=observation.trace_id,
            time_s=observation.time_s,
            vehicle_id=observation.vehicle_id,
            x_m=observation.x_m,
            y_m=observation.y_m,
            heading_rad=observation.heading_rad,
            route_id=observation.route_id,
            lane_id=observation.lane_id,
        )


@dataclass(frozen=True, slots=True)
class VehicleFrame:
    """All unique vehicle observations at one trace time."""

    trace_id: str
    time_s: float
    vehicles: tuple[VehicleRecord, ...]

    def __post_init__(self) -> None:
        if not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        vehicle_ids: set[str] = set()
        for vehicle in self.vehicles:
            if vehicle.trace_id != self.trace_id or vehicle.time_s != self.time_s:
                raise ValueError("every vehicle must match its frame trace_id and time_s")
            if vehicle.vehicle_id in vehicle_ids:
                raise ValueError(f"duplicate vehicle {vehicle.vehicle_id!r} in frame")
            vehicle_ids.add(vehicle.vehicle_id)


@dataclass(frozen=True, slots=True)
class TaggedPairSegment:
    """One observed follower-to-leader service episode.

    ``has_intervening_vehicle`` is true when at least one same-lane vehicle was
    observed longitudinally between the endpoints at any sampled time in the
    episode. ``eligibility_reason`` records why the eligible episode ended.
    """

    trace_id: str
    pair_id: str
    tx_id: str
    rx_id: str
    start_s: float
    end_s: float
    initial_distance_m: float
    route_id: str
    eligibility_reason: PairEndReason
    has_intervening_vehicle: bool

    def __post_init__(self) -> None:
        for field_name in ("trace_id", "pair_id", "tx_id", "rx_id", "route_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.tx_id == self.rx_id:
            raise ValueError("tx_id and rx_id must identify different vehicles")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("pair times must be finite")
        if self.start_s < 0.0 or self.end_s < self.start_s:
            raise ValueError("pair times must satisfy 0 <= start_s <= end_s")
        if not math.isfinite(self.initial_distance_m) or self.initial_distance_m <= 0.0:
            raise ValueError("initial_distance_m must be finite and positive")
        if self.eligibility_reason not in _PAIR_END_REASONS:
            raise ValueError(f"unknown eligibility_reason {self.eligibility_reason!r}")

    @property
    def duration_s(self) -> float:
        """Observed episode duration; no padding is included."""

        return self.end_s - self.start_s

    def to_record(self) -> dict[str, object]:
        """Return the publication schema as a column mapping."""

        return {
            "trace_id": self.trace_id,
            "pair_id": self.pair_id,
            "tx_id": self.tx_id,
            "rx_id": self.rx_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "initial_distance_m": self.initial_distance_m,
            "route_id": self.route_id,
            "eligibility_reason": self.eligibility_reason,
            "has_intervening_vehicle": self.has_intervening_vehicle,
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> TaggedPairSegment:
        """Construct a segment from a stored row."""

        reason = _required_text(row, "eligibility_reason")
        if reason not in _PAIR_END_REASONS:
            raise ValueError(f"unknown eligibility_reason {reason!r}")
        blocker = row.get("has_intervening_vehicle")
        if not isinstance(blocker, bool):
            raise TypeError("has_intervening_vehicle must be boolean")
        return cls(
            trace_id=_required_text(row, "trace_id"),
            pair_id=_required_text(row, "pair_id"),
            tx_id=_required_text(row, "tx_id"),
            rx_id=_required_text(row, "rx_id"),
            start_s=_required_float(row, "start_s"),
            end_s=_required_float(row, "end_s"),
            initial_distance_m=_required_float(row, "initial_distance_m"),
            route_id=_required_text(row, "route_id"),
            eligibility_reason=cast(PairEndReason, reason),
            has_intervening_vehicle=blocker,
        )


@dataclass(frozen=True, slots=True)
class TaggedPairConfig:
    """Frozen extraction thresholds."""

    min_separation_m: float = 10.0
    max_separation_m: float = 60.0
    outside_range_grace_s: float = 1.0
    max_duration_s: float = 60.0
    max_heading_difference_rad: float = math.radians(30.0)
    require_adjacent: bool = True
    """Whether the pair must be the immediate leader-follower.

    With this false, any two same-lane vehicles inside the separation window
    qualify, including pairs drawn out of a queue with several vehicles between
    them.  Measured on the generated traces, that yields a same-lane vehicle in
    the optical path 97-100% of the time at 40 and 60 veh/lane-km, because the
    mean spacing at those densities is smaller than the separation window --
    the pair cannot help but straddle other vehicles.  V-VLC then has no
    availability at all and the hybrid comparison has nothing to decide.

    Requiring adjacency leaves blockage to come from cross-traffic at
    junctions, which is intermittent, geometry-dependent, and predictable from
    tracked positions.
    """

    def __post_init__(self) -> None:
        values = (
            self.min_separation_m,
            self.max_separation_m,
            self.outside_range_grace_s,
            self.max_duration_s,
            self.max_heading_difference_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("all tagged-pair thresholds must be finite and positive")
        if self.max_separation_m <= self.min_separation_m:
            raise ValueError("max_separation_m must exceed min_separation_m")
        if self.max_heading_difference_rad > math.pi:
            raise ValueError("max_heading_difference_rad cannot exceed pi")


@dataclass(slots=True)
class _ActivePair:
    trace_id: str
    tx_id: str
    rx_id: str
    start_s: float
    last_observed_s: float
    initial_distance_m: float
    route_id: str
    outside_since_s: float | None
    has_intervening_vehicle: bool


def _coerce_records(
    records: Iterable[VehicleRecord | VehicleObservation | Mapping[str, object]],
) -> tuple[VehicleRecord, ...]:
    unique: dict[tuple[str, float, str], VehicleRecord] = {}
    for item in records:
        if isinstance(item, VehicleRecord):
            record = item
        elif isinstance(item, Mapping):
            record = VehicleRecord.from_mapping(item)
        elif isinstance(item, VehicleObservation):
            record = VehicleRecord.from_observation(item)
        else:
            raise TypeError("vehicle input must be a VehicleObservation or column mapping")
        key = (record.trace_id, record.time_s, record.vehicle_id)
        previous = unique.get(key)
        if previous is not None and previous != record:
            raise ValueError(
                "conflicting duplicate vehicle observation for "
                f"trace={record.trace_id!r}, time={record.time_s}, vehicle={record.vehicle_id!r}"
            )
        unique[key] = record
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (item.trace_id, item.time_s, item.vehicle_id),
        )
    )


def build_vehicle_frames(
    records: Iterable[VehicleRecord | VehicleObservation | Mapping[str, object]],
) -> tuple[VehicleFrame, ...]:
    """Group row records into deterministic immutable frames.

    Exact duplicate rows are de-duplicated. Conflicting rows with the same
    ``(trace_id, time_s, vehicle_id)`` identity are rejected.
    """

    grouped: dict[tuple[str, float], list[VehicleRecord]] = defaultdict(list)
    for record in _coerce_records(records):
        grouped[(record.trace_id, record.time_s)].append(record)
    return tuple(
        VehicleFrame(
            trace_id=trace_id,
            time_s=time_s,
            vehicles=tuple(sorted(vehicles, key=lambda item: item.vehicle_id)),
        )
        for (trace_id, time_s), vehicles in sorted(grouped.items())
    )


def _heading_difference(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def _distance(first: VehicleRecord, second: VehicleRecord) -> float:
    return math.hypot(second.x_m - first.x_m, second.y_m - first.y_m)


def _forward_projection(follower: VehicleRecord, other: VehicleRecord) -> float:
    dx = other.x_m - follower.x_m
    dy = other.y_m - follower.y_m
    return dx * math.cos(follower.heading_rad) + dy * math.sin(follower.heading_rad)


def _edge_of(lane_id: str) -> str:
    """Strip the lane index from a lane identifier (``B5B6_0`` -> ``B5B6``)."""

    return lane_id.rsplit("_", 1)[0]


def _same_planned_route(follower: VehicleRecord, leader: VehicleRecord) -> bool:
    """Whether the pair still shares an upcoming path.

    Implementation spec section 8.7 ends an episode when "the routes diverge",
    which for a following pair means the leader has left the path the follower
    is still going to take.  Requiring *identical itineraries* is far stricter
    than that: two vehicles following one another normally share a segment and
    part later.

    When both planned routes are known this tests the intended condition -- the
    leader is on the follower's own edge, or on an edge the follower has yet to
    reach.  A pair therefore survives the moment the leader crosses a junction
    first, which an equality test would incorrectly treat as divergence.

    Without planned routes it falls back to comparing route identifiers.  That
    fallback is a trap worth naming: under the SUMO backend every vehicle
    received a unique auto-generated route id, so the comparison never held and
    pair extraction silently returned zero episodes.
    """

    if follower.planned_route is None or leader.planned_route is None:
        return follower.route_id == leader.route_id

    follower_edge = _edge_of(follower.lane_id)
    leader_edge = _edge_of(leader.lane_id)
    if follower_edge == leader_edge:
        return True
    if follower_edge not in follower.planned_route:
        return False
    remaining = follower.planned_route[follower.planned_route.index(follower_edge) :]
    return leader_edge in remaining


def _vehicle_between(
    follower: VehicleRecord,
    leader: VehicleRecord,
    vehicles: tuple[VehicleRecord, ...],
    *,
    heading_tolerance_rad: float,
) -> bool:
    leader_projection = _forward_projection(follower, leader)
    for candidate in vehicles:
        if candidate.vehicle_id in {follower.vehicle_id, leader.vehicle_id}:
            continue
        if candidate.lane_id != follower.lane_id:
            continue
        if _heading_difference(candidate.heading_rad, follower.heading_rad) > heading_tolerance_rad:
            continue
        projection = _forward_projection(follower, candidate)
        if 0.0 < projection < leader_projection:
            return True
    return False


def _pair_id(trace_id: str, tx_id: str, rx_id: str, start_s: float) -> str:
    identity = "\x1f".join((trace_id, tx_id, rx_id, start_s.hex())).encode()
    return f"pair-{hashlib.sha256(identity).hexdigest()[:20]}"


class TaggedPairExtractor:
    """Extract natural same-lane follower-to-leader episodes."""

    def __init__(self, config: TaggedPairConfig | None = None) -> None:
        self.config = config or TaggedPairConfig()

    def extract(
        self,
        records: Iterable[VehicleRecord | VehicleObservation | Mapping[str, object]],
    ) -> tuple[TaggedPairSegment, ...]:
        """Extract episodes in stable trace/time/vehicle order.

        A temporary distance excursion is tolerated. It terminates the episode
        only after observations remain outside the inclusive configured range
        for at least ``outside_range_grace_s``. Missing observations are not
        padded or interpolated.
        """

        frames_by_trace: dict[str, list[VehicleFrame]] = defaultdict(list)
        for frame in build_vehicle_frames(records):
            frames_by_trace[frame.trace_id].append(frame)

        completed: list[TaggedPairSegment] = []
        for trace_id in sorted(frames_by_trace):
            completed.extend(self._extract_trace(trace_id, frames_by_trace[trace_id]))

        result = tuple(
            sorted(
                completed,
                key=lambda pair: (
                    pair.trace_id,
                    pair.start_s,
                    pair.tx_id,
                    pair.rx_id,
                    pair.pair_id,
                ),
            )
        )
        pair_ids = [pair.pair_id for pair in result]
        if len(pair_ids) != len(set(pair_ids)):
            raise RuntimeError("deterministic pair identifier collision")
        return result

    def _extract_trace(
        self,
        trace_id: str,
        frames: list[VehicleFrame],
    ) -> list[TaggedPairSegment]:
        active: dict[tuple[str, str], _ActivePair] = {}
        completed: list[TaggedPairSegment] = []

        for frame in frames:
            by_id = {vehicle.vehicle_id: vehicle for vehicle in frame.vehicles}
            for key in sorted(tuple(active)):
                state = active[key]
                elapsed = frame.time_s - state.start_s
                if elapsed > self.config.max_duration_s + _TIME_TOLERANCE_S:
                    completed.append(self._finish(state, state.last_observed_s, "max_duration"))
                    del active[key]
                    continue

                follower = by_id.get(state.tx_id)
                leader = by_id.get(state.rx_id)
                if follower is None or leader is None:
                    completed.append(self._finish(state, state.last_observed_s, "vehicle_missing"))
                    del active[key]
                    continue
                if not _same_planned_route(follower, leader):
                    completed.append(self._finish(state, frame.time_s, "route_diverged"))
                    del active[key]
                    continue

                distance_m = _distance(follower, leader)
                in_range = (
                    self.config.min_separation_m <= distance_m <= self.config.max_separation_m
                )
                if in_range:
                    state.outside_since_s = None
                elif state.outside_since_s is None:
                    state.outside_since_s = frame.time_s
                elif (
                    frame.time_s - state.outside_since_s
                    >= self.config.outside_range_grace_s - _TIME_TOLERANCE_S
                ):
                    completed.append(self._finish(state, frame.time_s, "outside_range_1s"))
                    del active[key]
                    continue

                state.has_intervening_vehicle = state.has_intervening_vehicle or _vehicle_between(
                    follower,
                    leader,
                    frame.vehicles,
                    heading_tolerance_rad=self.config.max_heading_difference_rad,
                )
                state.last_observed_s = frame.time_s
                if elapsed >= self.config.max_duration_s - _TIME_TOLERANCE_S:
                    completed.append(self._finish(state, frame.time_s, "max_duration"))
                    del active[key]

            for follower, leader, distance_m, blocker in self._initial_candidates(frame):
                key = (follower.vehicle_id, leader.vehicle_id)
                if key in active:
                    continue
                active[key] = _ActivePair(
                    trace_id=trace_id,
                    tx_id=follower.vehicle_id,
                    rx_id=leader.vehicle_id,
                    start_s=frame.time_s,
                    last_observed_s=frame.time_s,
                    initial_distance_m=distance_m,
                    route_id=follower.route_id,
                    outside_since_s=None,
                    has_intervening_vehicle=blocker,
                )

        for key in sorted(active):
            state = active[key]
            completed.append(self._finish(state, state.last_observed_s, "trace_end"))
        return completed

    def _initial_candidates(
        self,
        frame: VehicleFrame,
    ) -> tuple[tuple[VehicleRecord, VehicleRecord, float, bool], ...]:
        candidates: list[tuple[VehicleRecord, VehicleRecord, float, bool]] = []
        for follower in frame.vehicles:
            for leader in frame.vehicles:
                if follower.vehicle_id == leader.vehicle_id:
                    continue
                if follower.lane_id != leader.lane_id:
                    continue
                if not _same_planned_route(follower, leader):
                    continue
                if (
                    _heading_difference(follower.heading_rad, leader.heading_rad)
                    > self.config.max_heading_difference_rad
                ):
                    continue
                if _forward_projection(follower, leader) <= 0.0:
                    continue
                distance_m = _distance(follower, leader)
                if not (self.config.min_separation_m <= distance_m <= self.config.max_separation_m):
                    continue
                blocker = _vehicle_between(
                    follower,
                    leader,
                    frame.vehicles,
                    heading_tolerance_rad=self.config.max_heading_difference_rad,
                )
                if self.config.require_adjacent and blocker:
                    continue
                candidates.append((follower, leader, distance_m, blocker))
        return tuple(
            sorted(
                candidates,
                key=lambda item: (item[0].vehicle_id, item[1].vehicle_id),
            )
        )

    @staticmethod
    def _finish(
        state: _ActivePair,
        end_s: float,
        reason: PairEndReason,
    ) -> TaggedPairSegment:
        return TaggedPairSegment(
            trace_id=state.trace_id,
            pair_id=_pair_id(state.trace_id, state.tx_id, state.rx_id, state.start_s),
            tx_id=state.tx_id,
            rx_id=state.rx_id,
            start_s=state.start_s,
            end_s=end_s,
            initial_distance_m=state.initial_distance_m,
            route_id=state.route_id,
            eligibility_reason=reason,
            has_intervening_vehicle=state.has_intervening_vehicle,
        )


__all__ = [
    "PairEndReason",
    "TaggedPairConfig",
    "TaggedPairExtractor",
    "TaggedPairSegment",
    "VehicleFrame",
    "VehicleObservation",
    "VehicleRecord",
    "build_vehicle_frames",
]
