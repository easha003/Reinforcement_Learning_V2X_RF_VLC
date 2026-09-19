"""Mobility credibility metrics and explicit Gate-1 checks.

This module operates on immutable, already sampled mobility summaries.  Counts in a
``MobilityFrame`` are interval increments (departures, insertion failures, and
teleports), so aggregation does not depend on mutable simulator counters.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hybrid_v2x_rl.mobility.tagged_pairs import TaggedPairSegment


def _text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    if row.get(key) is None:
        return None
    return _float(row, key)


def _integer(row: Mapping[str, object], key: str, *, default: int = 0) -> int:
    value = row.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    if value < 0:
        raise ValueError(f"{key} cannot be negative")
    return value


def _text_sequence(row: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = row.get(key, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{key} must be a sequence of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} entries must be non-empty strings")
        items.append(item.strip())
    return tuple(items)


def _float_sequence(row: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = row.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{key} must be a sequence of real numbers")
    items: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{key} entries must be real numbers")
        converted = float(item)
        if not math.isfinite(converted) or converted < 0.0:
            raise ValueError(f"{key} entries must be finite and nonnegative")
        items.append(converted)
    return tuple(items)


def _signal_states(row: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    value = row.get("signal_states", ())
    if isinstance(value, Mapping):
        raw_items = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = tuple(value)
    else:
        raise TypeError("signal_states must be a mapping or sequence of two-item sequences")

    states: list[tuple[str, str]] = []
    for item in raw_items:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise TypeError("each signal_states item must contain signal ID and state")
        signal_id, state = item
        if not isinstance(signal_id, str) or not signal_id.strip():
            raise ValueError("signal ID must be a non-empty string")
        if not isinstance(state, str) or not state.strip():
            raise ValueError("signal state must be a non-empty string")
        states.append((signal_id.strip(), state.strip()))
    states.sort()
    if len({signal_id for signal_id, _ in states}) != len(states):
        raise ValueError("signal_states cannot repeat a signal ID")
    return tuple(states)


@dataclass(frozen=True, slots=True)
class MobilityFrame:
    """One interval's auditable network measurements."""

    trace_id: str
    time_s: float
    total_lane_length_km: float
    vehicle_speeds_mps: tuple[float, ...]
    queued_vehicle_count: int | None = None
    queue_length_m: float | None = None
    departed_vehicle_count: int = 0
    arrived_vehicle_count: int | None = None
    insertion_failures: int = 0
    teleport_reasons: tuple[str, ...] = ()
    signal_states: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        for name in ("time_s", "total_lane_length_km"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.time_s < 0.0:
            raise ValueError("time_s cannot be negative")
        if self.total_lane_length_km <= 0.0:
            raise ValueError("total_lane_length_km must be positive")
        for speed in self.vehicle_speeds_mps:
            if isinstance(speed, bool) or not isinstance(speed, (int, float)):
                raise TypeError("vehicle speeds must be real numbers")
            if not math.isfinite(float(speed)) or speed < 0.0:
                raise ValueError("vehicle speeds must be finite and nonnegative")
        if self.queued_vehicle_count is not None:
            if isinstance(self.queued_vehicle_count, bool) or not isinstance(
                self.queued_vehicle_count, int
            ):
                raise TypeError("queued_vehicle_count must be an integer when supplied")
            if not 0 <= self.queued_vehicle_count <= len(self.vehicle_speeds_mps):
                raise ValueError("queued_vehicle_count must be between zero and active vehicles")
        if self.queue_length_m is not None and (
            not math.isfinite(self.queue_length_m) or self.queue_length_m < 0.0
        ):
            raise ValueError("queue_length_m must be finite and nonnegative")
        for name in ("departed_vehicle_count", "insertion_failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.arrived_vehicle_count is not None:
            if isinstance(self.arrived_vehicle_count, bool) or not isinstance(
                self.arrived_vehicle_count, int
            ):
                raise TypeError("arrived_vehicle_count must be an integer when supplied")
            if self.arrived_vehicle_count < 0:
                raise ValueError("arrived_vehicle_count cannot be negative")
        if any(not reason.strip() for reason in self.teleport_reasons):
            raise ValueError("teleport reasons must be non-empty strings")
        signal_ids = [signal_id for signal_id, _ in self.signal_states]
        if len(signal_ids) != len(set(signal_ids)):
            raise ValueError("signal_states cannot repeat a signal ID")
        if any(
            not signal_id.strip() or not state.strip() for signal_id, state in self.signal_states
        ):
            raise ValueError("signal IDs and states must be non-empty strings")

    @property
    def active_vehicle_count(self) -> int:
        """Number of active vehicles represented by this frame."""

        return len(self.vehicle_speeds_mps)

    @property
    def density_veh_per_lane_km(self) -> float:
        """Realized lane-normalized density for this frame."""

        return self.active_vehicle_count / self.total_lane_length_km

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> MobilityFrame:
        """Construct a frame from a JSON/Parquet-friendly column mapping."""

        queued_raw = row.get("queued_vehicle_count")
        queued = None if queued_raw is None else _integer(row, "queued_vehicle_count")
        return cls(
            trace_id=_text(row, "trace_id"),
            time_s=_float(row, "time_s"),
            total_lane_length_km=_float(row, "total_lane_length_km"),
            vehicle_speeds_mps=_float_sequence(row, "vehicle_speeds_mps"),
            queued_vehicle_count=queued,
            queue_length_m=_optional_float(row, "queue_length_m"),
            departed_vehicle_count=_integer(row, "departed_vehicle_count"),
            arrived_vehicle_count=(
                None
                if row.get("arrived_vehicle_count") is None
                else _integer(row, "arrived_vehicle_count")
            ),
            insertion_failures=_integer(row, "insertion_failures"),
            teleport_reasons=_text_sequence(row, "teleport_reasons"),
            signal_states=_signal_states(row),
        )


@dataclass(frozen=True, slots=True)
class MobilityValidationCriteria:
    """Numerical and evidence requirements used for one Gate-1 report."""

    target_density_veh_per_lane_km: float
    speed_limit_mps: float
    density_tolerance_fraction: float = 0.05
    speed_limit_tolerance_fraction: float = 0.05
    stopped_speed_threshold_mps: float = 0.1
    queue_vehicle_length_proxy_m: float = 7.5
    max_insertion_failures: int = 0
    allowed_teleport_reasons: frozenset[str] = frozenset()
    require_positive_throughput: bool = True
    require_signal_variation: bool = True
    require_tagged_pairs: bool = True
    min_pair_separation_m: float = 10.0
    max_pair_separation_m: float = 60.0
    max_pair_duration_s: float = 60.0

    def __post_init__(self) -> None:
        positive = (
            self.target_density_veh_per_lane_km,
            self.speed_limit_mps,
            self.queue_vehicle_length_proxy_m,
            self.min_pair_separation_m,
            self.max_pair_separation_m,
            self.max_pair_duration_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("positive Gate-1 criteria must be finite and positive")
        for name in ("density_tolerance_fraction", "speed_limit_tolerance_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if (
            not math.isfinite(self.stopped_speed_threshold_mps)
            or self.stopped_speed_threshold_mps < 0.0
        ):
            raise ValueError("stopped_speed_threshold_mps must be finite and nonnegative")
        if self.max_insertion_failures < 0:
            raise ValueError("max_insertion_failures cannot be negative")
        if self.max_pair_separation_m <= self.min_pair_separation_m:
            raise ValueError("max_pair_separation_m must exceed min_pair_separation_m")
        if any(not reason.strip() for reason in self.allowed_teleport_reasons):
            raise ValueError("allowed teleport reasons must be non-empty strings")


@dataclass(frozen=True, slots=True)
class PairStatistics:
    """Summary of extracted tagged-pair episodes."""

    count: int
    mean_duration_s: float
    min_duration_s: float
    max_duration_s: float
    blocker_fraction: float
    distance_bin_counts: tuple[tuple[str, int], ...]
    end_reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Gate1Check:
    """One explicit, machine-readable Gate-1 decision."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class MobilityValidationReport:
    """Computed mobility statistics and Gate-1 decision evidence."""

    trace_id: str
    frame_count: int
    observation_duration_s: float
    active_vehicle_observations: int
    target_density_veh_per_lane_km: float
    realized_density_mean: float
    realized_density_std: float
    realized_density_min: float
    realized_density_max: float
    density_relative_error: float
    mean_speed_mps: float
    speed_std_mps: float
    stopped_fraction: float
    mean_queue_proxy_vehicles: float
    mean_queue_length_m: float
    throughput_vehicle_count: int
    throughput_vehicles_per_hour: float
    insertion_failures: int
    teleport_count: int
    teleport_reason_counts: tuple[tuple[str, int], ...]
    unexplained_teleport_count: int
    signal_count: int
    signals_with_state_variation: int
    signal_transition_count: int
    pair_statistics: PairStatistics
    gate1_checks: tuple[Gate1Check, ...]

    @property
    def gate1_passed(self) -> bool:
        """Whether every explicit Gate-1 check passed."""

        return bool(self.gate1_checks) and all(check.passed for check in self.gate1_checks)

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable nested report."""

        return {
            "trace_id": self.trace_id,
            "frame_count": self.frame_count,
            "observation_duration_s": self.observation_duration_s,
            "active_vehicle_observations": self.active_vehicle_observations,
            "target_density_veh_per_lane_km": self.target_density_veh_per_lane_km,
            "realized_density_mean": self.realized_density_mean,
            "realized_density_std": self.realized_density_std,
            "realized_density_min": self.realized_density_min,
            "realized_density_max": self.realized_density_max,
            "density_relative_error": self.density_relative_error,
            "mean_speed_mps": self.mean_speed_mps,
            "speed_std_mps": self.speed_std_mps,
            "stopped_fraction": self.stopped_fraction,
            "mean_queue_proxy_vehicles": self.mean_queue_proxy_vehicles,
            "mean_queue_length_m": self.mean_queue_length_m,
            "throughput_vehicle_count": self.throughput_vehicle_count,
            "throughput_vehicles_per_hour": self.throughput_vehicles_per_hour,
            "insertion_failures": self.insertion_failures,
            "teleport_count": self.teleport_count,
            "teleport_reason_counts": dict(self.teleport_reason_counts),
            "unexplained_teleport_count": self.unexplained_teleport_count,
            "signal_count": self.signal_count,
            "signals_with_state_variation": self.signals_with_state_variation,
            "signal_transition_count": self.signal_transition_count,
            "pair_statistics": {
                "count": self.pair_statistics.count,
                "mean_duration_s": self.pair_statistics.mean_duration_s,
                "min_duration_s": self.pair_statistics.min_duration_s,
                "max_duration_s": self.pair_statistics.max_duration_s,
                "blocker_fraction": self.pair_statistics.blocker_fraction,
                "distance_bin_counts": dict(self.pair_statistics.distance_bin_counts),
                "end_reason_counts": dict(self.pair_statistics.end_reason_counts),
            },
            "gate1_checks": [
                {"name": check.name, "passed": check.passed, "detail": check.detail}
                for check in self.gate1_checks
            ],
            "gate1_passed": self.gate1_passed,
        }


def _coerce_frames(
    frames: Iterable[MobilityFrame | Mapping[str, object]],
) -> tuple[MobilityFrame, ...]:
    unique: dict[tuple[str, float], MobilityFrame] = {}
    for item in frames:
        frame = item if isinstance(item, MobilityFrame) else MobilityFrame.from_mapping(item)
        key = (frame.trace_id, frame.time_s)
        previous = unique.get(key)
        if previous is not None and previous != frame:
            raise ValueError(
                f"conflicting mobility frame for trace={frame.trace_id!r}, time={frame.time_s}"
            )
        unique[key] = frame
    result = tuple(sorted(unique.values(), key=lambda frame: (frame.trace_id, frame.time_s)))
    if not result:
        raise ValueError("at least one mobility frame is required")
    trace_ids = {frame.trace_id for frame in result}
    if len(trace_ids) != 1:
        raise ValueError("one validation report cannot combine multiple trace IDs")
    lane_lengths = {frame.total_lane_length_km for frame in result}
    if len(lane_lengths) != 1:
        raise ValueError("total_lane_length_km must remain constant within a trace")
    return result


def _coerce_pairs(
    pairs: Iterable[TaggedPairSegment | Mapping[str, object]],
    trace_id: str,
) -> tuple[TaggedPairSegment, ...]:
    unique: dict[str, TaggedPairSegment] = {}
    for item in pairs:
        pair = item if isinstance(item, TaggedPairSegment) else TaggedPairSegment.from_mapping(item)
        if pair.trace_id != trace_id:
            raise ValueError("pair trace_id does not match the mobility report")
        previous = unique.get(pair.pair_id)
        if previous is not None and previous != pair:
            raise ValueError(f"conflicting tagged-pair rows for pair_id={pair.pair_id!r}")
        unique[pair.pair_id] = pair
    return tuple(sorted(unique.values(), key=lambda pair: (pair.start_s, pair.tx_id, pair.rx_id)))


def _separation_bin_edges(minimum_m: float, maximum_m: float) -> tuple[float, ...]:
    """Three equal-width strata spanning the configured separation window.

    Deriving the edges from the window rather than hard-coding them keeps the
    report meaningful when the window changes.  Fixed 10/20/40/60 edges put 45%
    of pairs in "outside" once the window moved to 5-40 m, which hid the
    stratification the evaluation depends on.
    """

    step = (maximum_m - minimum_m) / 3.0
    return (
        minimum_m,
        round(minimum_m + step, 1),
        round(minimum_m + 2.0 * step, 1),
        maximum_m,
    )


def _pair_statistics(
    pairs: tuple[TaggedPairSegment, ...],
    *,
    min_separation_m: float = 10.0,
    max_separation_m: float = 60.0,
) -> PairStatistics:
    durations = [pair.duration_s for pair in pairs]
    lo, a, b, hi = _separation_bin_edges(min_separation_m, max_separation_m)
    labels = (f"{lo:g}-{a:g}", f"{a:g}-{b:g}", f"{b:g}-{hi:g}")
    bins = Counter({labels[0]: 0, labels[1]: 0, labels[2]: 0, "outside": 0})
    for pair in pairs:
        distance = pair.initial_distance_m
        if lo <= distance < a:
            bins[labels[0]] += 1
        elif a <= distance < b:
            bins[labels[1]] += 1
        elif b <= distance <= hi:
            bins[labels[2]] += 1
        else:
            bins["outside"] += 1
    reasons = Counter(pair.eligibility_reason for pair in pairs)
    return PairStatistics(
        count=len(pairs),
        mean_duration_s=statistics.fmean(durations) if durations else 0.0,
        min_duration_s=min(durations, default=0.0),
        max_duration_s=max(durations, default=0.0),
        blocker_fraction=(
            sum(pair.has_intervening_vehicle for pair in pairs) / len(pairs) if pairs else 0.0
        ),
        distance_bin_counts=tuple(sorted(bins.items())),
        end_reason_counts=tuple(sorted(reasons.items())),
    )


class MobilityValidator:
    """Compute a complete, immutable validation report for one trace."""

    def __init__(self, criteria: MobilityValidationCriteria) -> None:
        self.criteria = criteria

    def validate(
        self,
        frames: Iterable[MobilityFrame | Mapping[str, object]],
        *,
        pair_segments: Iterable[TaggedPairSegment | Mapping[str, object]] = (),
        manifest_archived: bool = False,
    ) -> MobilityValidationReport:
        """Measure a trace and evaluate every declared Gate-1 condition.

        ``manifest_archived`` defaults to false deliberately: a numerical trace
        must not silently claim that its resolved configuration and provenance
        manifest were persisted.
        """

        ordered_frames = _coerce_frames(frames)
        trace_id = ordered_frames[0].trace_id
        pairs = _coerce_pairs(pair_segments, trace_id)

        densities = [frame.density_veh_per_lane_km for frame in ordered_frames]
        density_mean = statistics.fmean(densities)
        density_std = statistics.pstdev(densities)
        relative_error = (
            abs(density_mean - self.criteria.target_density_veh_per_lane_km)
            / self.criteria.target_density_veh_per_lane_km
        )

        speeds = [speed for frame in ordered_frames for speed in frame.vehicle_speeds_mps]
        speed_mean = statistics.fmean(speeds) if speeds else 0.0
        speed_std = statistics.pstdev(speeds) if speeds else 0.0
        stopped_count = sum(speed <= self.criteria.stopped_speed_threshold_mps for speed in speeds)
        stopped_fraction = stopped_count / len(speeds) if speeds else 0.0

        queue_counts: list[int] = []
        queue_lengths: list[float] = []
        for frame in ordered_frames:
            derived_queue = sum(
                speed <= self.criteria.stopped_speed_threshold_mps
                for speed in frame.vehicle_speeds_mps
            )
            queue_count = (
                frame.queued_vehicle_count
                if frame.queued_vehicle_count is not None
                else derived_queue
            )
            queue_counts.append(queue_count)
            queue_lengths.append(
                frame.queue_length_m
                if frame.queue_length_m is not None
                else queue_count * self.criteria.queue_vehicle_length_proxy_m
            )

        duration_s = ordered_frames[-1].time_s - ordered_frames[0].time_s
        arrivals_are_recorded = all(
            frame.arrived_vehicle_count is not None for frame in ordered_frames
        )
        throughput_count = (
            sum(frame.arrived_vehicle_count or 0 for frame in ordered_frames)
            if arrivals_are_recorded
            else sum(frame.departed_vehicle_count for frame in ordered_frames)
        )
        throughput_per_hour = throughput_count * 3600.0 / duration_s if duration_s > 0.0 else 0.0
        insertion_failures = sum(frame.insertion_failures for frame in ordered_frames)

        teleport_reasons = [reason for frame in ordered_frames for reason in frame.teleport_reasons]
        teleport_reason_counts = Counter(teleport_reasons)
        unexplained = [
            reason
            for reason in teleport_reasons
            if reason not in self.criteria.allowed_teleport_reasons
        ]

        signal_history: dict[str, list[str]] = defaultdict(list)
        for frame in ordered_frames:
            for signal_id, state in frame.signal_states:
                signal_history[signal_id].append(state)
        signal_transitions = sum(
            first != second
            for history in signal_history.values()
            for first, second in zip(history, history[1:], strict=False)
        )
        varying_signals = sum(len(set(history)) >= 2 for history in signal_history.values())

        pair_stats = _pair_statistics(
            pairs,
            min_separation_m=self.criteria.min_pair_separation_m,
            max_separation_m=self.criteria.max_pair_separation_m,
        )
        pairs_valid = all(
            self.criteria.min_pair_separation_m
            <= pair.initial_distance_m
            <= self.criteria.max_pair_separation_m
            and pair.duration_s <= self.criteria.max_pair_duration_s + 1e-9
            and pair.start_s >= ordered_frames[0].time_s - 1e-9
            and pair.end_s <= ordered_frames[-1].time_s + 1e-9
            for pair in pairs
        )
        if self.criteria.require_tagged_pairs:
            pairs_valid = pairs_valid and bool(pairs)

        density_passed = relative_error <= self.criteria.density_tolerance_fraction + 1e-12
        insertion_passed = insertion_failures <= self.criteria.max_insertion_failures
        throughput_passed = not self.criteria.require_positive_throughput or throughput_count > 0
        speed_passed = bool(speeds) and (
            0.0
            < speed_mean
            <= self.criteria.speed_limit_mps * (1.0 + self.criteria.speed_limit_tolerance_fraction)
        )
        queues_passed = all(
            0 <= queued <= frame.active_vehicle_count
            for queued, frame in zip(queue_counts, ordered_frames, strict=True)
        ) and all(
            length >= 0.0
            and math.isfinite(length)
            and length <= frame.total_lane_length_km * 1000.0 + 1e-9
            and ((queued == 0 and length == 0.0) or (queued > 0 and length > 0.0))
            for queued, length, frame in zip(
                queue_counts, queue_lengths, ordered_frames, strict=True
            )
        )
        signals_passed = not self.criteria.require_signal_variation or (
            bool(signal_history) and varying_signals == len(signal_history)
        )

        checks = (
            Gate1Check(
                "sufficient_time_samples",
                len(ordered_frames) >= 2 and duration_s > 0.0,
                f"{len(ordered_frames)} unique frames over {duration_s:.3f} s",
            ),
            Gate1Check(
                "stable_insertion_and_routing",
                insertion_passed and throughput_passed,
                f"{insertion_failures} insertion failures; {throughput_count} "
                f"{'arrivals' if arrivals_are_recorded else 'departures (legacy proxy)'}",
            ),
            Gate1Check(
                "density_within_tolerance",
                density_passed,
                f"mean={density_mean:.6g}, target="
                f"{self.criteria.target_density_veh_per_lane_km:.6g}, "
                f"relative error={relative_error:.3%}, "
                f"limit={self.criteria.density_tolerance_fraction:.3%}",
            ),
            Gate1Check(
                "plausible_speed",
                speed_passed,
                f"mean={speed_mean:.6g} m/s, std={speed_std:.6g} m/s, "
                f"speed limit={self.criteria.speed_limit_mps:.6g} m/s",
            ),
            Gate1Check(
                "plausible_queue_metrics",
                queues_passed,
                f"mean proxy={statistics.fmean(queue_counts):.6g} vehicles, "
                f"mean length={statistics.fmean(queue_lengths):.6g} m",
            ),
            Gate1Check(
                "signal_state_variation",
                signals_passed,
                f"{varying_signals}/{len(signal_history)} signals varied; "
                f"{signal_transitions} transitions",
            ),
            Gate1Check(
                "no_unexplained_teleportation",
                not unexplained,
                f"{len(teleport_reasons)} teleports; {len(unexplained)} unexplained",
            ),
            Gate1Check(
                "tagged_pair_separation_logic",
                pairs_valid,
                f"{len(pairs)} unique episodes; max duration={pair_stats.max_duration_s:.6g} s",
            ),
            Gate1Check(
                "manifest_archived",
                manifest_archived,
                "trace manifest archive explicitly confirmed"
                if manifest_archived
                else "trace manifest archive not confirmed",
            ),
        )

        return MobilityValidationReport(
            trace_id=trace_id,
            frame_count=len(ordered_frames),
            observation_duration_s=duration_s,
            active_vehicle_observations=len(speeds),
            target_density_veh_per_lane_km=self.criteria.target_density_veh_per_lane_km,
            realized_density_mean=density_mean,
            realized_density_std=density_std,
            realized_density_min=min(densities),
            realized_density_max=max(densities),
            density_relative_error=relative_error,
            mean_speed_mps=speed_mean,
            speed_std_mps=speed_std,
            stopped_fraction=stopped_fraction,
            mean_queue_proxy_vehicles=statistics.fmean(queue_counts),
            mean_queue_length_m=statistics.fmean(queue_lengths),
            throughput_vehicle_count=throughput_count,
            throughput_vehicles_per_hour=throughput_per_hour,
            insertion_failures=insertion_failures,
            teleport_count=len(teleport_reasons),
            teleport_reason_counts=tuple(sorted(teleport_reason_counts.items())),
            unexplained_teleport_count=len(unexplained),
            signal_count=len(signal_history),
            signals_with_state_variation=varying_signals,
            signal_transition_count=signal_transitions,
            pair_statistics=pair_stats,
            gate1_checks=checks,
        )


__all__ = [
    "Gate1Check",
    "MobilityFrame",
    "MobilityValidationCriteria",
    "MobilityValidationReport",
    "MobilityValidator",
    "PairStatistics",
]
