"""Deterministic chronological population frames from immutable traces.

The legacy replay API yields one pair at a time.  That is sufficient for the
fixed-policy feasibility study, but it loses the simultaneous population that
the mean-field environment must act on.  This module keeps one verified
mobility frame in memory and emits every active pair together on the global
packet-generation clock.

Only policy-independent trace facts live here.  In particular, source end
reasons are exposed only on the pair's final decision frame as simulator
lifecycle metadata.  They are not actor observations.  Action-dependent RF
load, collision risk, outcomes, rewards, and mean-field state belong to later
environment phases.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import yaml

from hybrid_v2x_rl.config.hashing import scope_hash
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex
from hybrid_v2x_rl.mobility.tagged_pairs import PairEndReason, TaggedPairSegment
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceReader, VehicleTraceRecord

TraceSplit = Literal["train", "validation", "test"]

_TRACE_ID = re.compile(
    r"^synthetic-d(?P<density>[0-9]+(?:\.[0-9]+)?)-"
    r"(?P<split>train|validation|test)-(?P<replicate>[0-9]{3,})$"
)
_SPLITS: tuple[TraceSplit, ...] = ("train", "validation", "test")
_NATURAL_END_REASONS: frozenset[PairEndReason] = frozenset(
    {"outside_range_1s", "route_diverged", "vehicle_missing"}
)
_TIME_TOLERANCE_S = 1e-9


class FrameReplayError(HybridV2XError):
    """A trace cannot be represented by the frozen population-frame contract."""


def _archived_config_scope_hash(path: Path, scope: str) -> str:
    """Hash one integrity-protected archived config under the current scope."""

    config_path = path / "resolved_config.yaml"
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FrameReplayError(
            "trace archived configuration cannot be read",
            artifact_path=config_path,
        ) from error
    if not isinstance(payload, Mapping):
        raise FrameReplayError(
            "trace archived configuration must be a mapping",
            artifact_path=config_path,
        )
    try:
        return scope_hash(payload, scope)
    except HybridV2XError as error:
        raise FrameReplayError(
            "trace archived configuration cannot satisfy the requested scope",
            artifact_path=config_path,
            context={"scope": scope},
        ) from error


@dataclass(frozen=True, slots=True)
class FrameTraceSource:
    """Immutable identity and split membership of one trace artifact."""

    path: Path
    trace_id: str
    split: TraceSplit
    density: float
    replicate: int

    @classmethod
    def discover(
        cls,
        path: str | Path,
        *,
        expected_split: TraceSplit | None = None,
    ) -> FrameTraceSource:
        """Parse canonical identity without allowing a caller to relabel a split."""

        artifact_path = Path(path)
        trace_id = artifact_path.name
        match = _TRACE_ID.fullmatch(trace_id)
        if match is None:
            raise FrameReplayError(
                "trace ID does not follow the frozen campaign naming scheme",
                context={"trace_id": trace_id},
            )
        split = cast(TraceSplit, match.group("split"))
        if expected_split is not None and split != expected_split:
            raise FrameReplayError(
                "trace ID split does not match configured membership",
                context={
                    "trace_id": trace_id,
                    "encoded_split": split,
                    "configured_split": expected_split,
                },
            )
        density = float(match.group("density"))
        if not math.isfinite(density) or density <= 0.0:
            raise FrameReplayError(
                "trace density must be finite and positive",
                context={"trace_id": trace_id, "density": density},
            )
        return cls(
            path=artifact_path,
            trace_id=trace_id,
            split=split,
            density=density,
            replicate=int(match.group("replicate")),
        )


@dataclass(frozen=True, slots=True)
class TraceCatalog:
    """Configuration-authoritative, immutable train/validation/test catalog."""

    sources: tuple[FrameTraceSource, ...]

    @classmethod
    def from_splits(
        cls,
        trace_root: str | Path,
        splits: TraceSplitConfig,
    ) -> TraceCatalog:
        """Bind configured IDs to paths while rejecting encoded split drift."""

        root = Path(trace_root)
        found: list[FrameTraceSource] = []
        seen: dict[str, TraceSplit] = {}
        for split in _SPLITS:
            for trace_id in getattr(splits, split):
                previous = seen.get(trace_id)
                if previous is not None:
                    raise FrameReplayError(
                        "trace ID belongs to more than one configured split",
                        context={
                            "trace_id": trace_id,
                            "first_split": previous,
                            "second_split": split,
                        },
                    )
                seen[trace_id] = split
                found.append(
                    FrameTraceSource.discover(root / trace_id, expected_split=split)
                )
        return cls(sources=tuple(found))

    def for_split(self, split: TraceSplit) -> tuple[FrameTraceSource, ...]:
        """Return configured sources in their declared order."""

        return tuple(source for source in self.sources if source.split == split)

    def source(self, trace_id: str) -> FrameTraceSource:
        """Resolve exactly one configured trace ID."""

        matches = tuple(source for source in self.sources if source.trace_id == trace_id)
        if len(matches) != 1:
            raise FrameReplayError(
                "trace ID is not a unique configured source",
                context={"trace_id": trace_id, "matches": len(matches)},
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class PairLifecycle:
    """Simulator-only lifecycle flags attached to one active pair row."""

    born: bool
    terminated: bool = False
    truncated: bool = False
    bootstrap_valid: bool = False
    end_reason: PairEndReason | None = None

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.born,
                self.terminated,
                self.truncated,
                self.bootstrap_valid,
            )
        ):
            raise ValueError("pair lifecycle flags must be booleans")
        if self.terminated and self.truncated:
            raise ValueError("a pair cannot be both terminated and truncated")
        if self.end_reason is None:
            if self.terminated or self.truncated or self.bootstrap_valid:
                raise ValueError("non-final lifecycle flags require an end reason")
            return
        if self.end_reason in _NATURAL_END_REASONS:
            if not self.terminated or self.truncated or self.bootstrap_valid:
                raise ValueError("natural pair endings must be terminal without bootstrap")
        elif self.end_reason == "max_duration":
            if self.terminated or not self.truncated:
                raise ValueError("max-duration endings must be truncations")
        elif self.end_reason == "trace_end":
            if self.terminated or not self.truncated or self.bootstrap_valid:
                raise ValueError("physical trace endings truncate without bootstrap")
        else:  # pragma: no cover - PairEndReason and source validation make this unreachable.
            raise ValueError(f"unsupported pair end reason {self.end_reason!r}")

    @property
    def continuing(self) -> bool:
        """Whether this row continues an already active episode."""

        return not self.born

    @property
    def final(self) -> bool:
        """Whether state must be released after the current frame."""

        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class PopulationPair:
    """One active pair in a simultaneous decision frame."""

    pair_id: str
    episode_step: int
    transmitter: VehicleTraceRecord
    receiver: VehicleTraceRecord
    lifecycle: PairLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise ValueError("pair_id must be a non-empty string")
        if (
            not isinstance(self.episode_step, int)
            or isinstance(self.episode_step, bool)
            or self.episode_step < 0
        ):
            raise ValueError("episode_step must be non-negative")
        if not isinstance(self.lifecycle, PairLifecycle):
            raise ValueError("pair lifecycle must be a PairLifecycle")
        if self.lifecycle.born != (self.episode_step == 0):
            raise ValueError(
                "born must be true exactly on pair episode step zero"
            )
        if self.transmitter.vehicle_id == self.receiver.vehicle_id:
            raise ValueError("pair endpoints must identify different vehicles")

    @property
    def episode_id(self) -> str:
        """Stable episode membership identifier."""

        return self.pair_id

    @property
    def endpoint_ids(self) -> tuple[str, str]:
        """Transmitter and receiver identities in resource-accounting order."""

        return self.transmitter.vehicle_id, self.receiver.vehicle_id


@dataclass(frozen=True, slots=True)
class PopulationFrame:
    """All active service agents and neighbours at one decision time."""

    source: FrameTraceSource
    index: int
    time_s: float
    vehicles: tuple[VehicleTraceRecord, ...]
    pairs: tuple[PopulationPair, ...]
    _spatial_index: SpatialIndex | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("frame index must be non-negative")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("frame time must be finite and non-negative")
        vehicle_ids = tuple(vehicle.vehicle_id for vehicle in self.vehicles)
        if vehicle_ids != tuple(sorted(vehicle_ids)):
            raise ValueError("vehicles must be ordered by stable vehicle ID")
        if len(vehicle_ids) != len(set(vehicle_ids)):
            raise ValueError("a population frame cannot contain duplicate vehicles")
        present = set(vehicle_ids)
        for vehicle in self.vehicles:
            if vehicle.trace_id != self.source.trace_id:
                raise ValueError("vehicle trace identity does not match its population frame")
            if not math.isclose(
                vehicle.time_s,
                self.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ):
                raise ValueError("vehicle time does not match its population frame")
        pair_ids = tuple(pair.pair_id for pair in self.pairs)
        if pair_ids != tuple(sorted(pair_ids)):
            raise ValueError("pairs must be ordered lexicographically by stable pair ID")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("a population frame cannot contain duplicate pair IDs")
        for pair in self.pairs:
            if not set(pair.endpoint_ids).issubset(present):
                raise ValueError("every active pair endpoint must be present in the frame")

    @property
    def trace_id(self) -> str:
        return self.source.trace_id

    @property
    def split(self) -> TraceSplit:
        return self.source.split

    @property
    def density(self) -> float:
        return self.source.density

    @property
    def active_pair_ids(self) -> tuple[str, ...]:
        return tuple(pair.pair_id for pair in self.pairs)

    @property
    def spatial_index(self) -> SpatialIndex:
        """Build the neighbour index only when downstream physics requests it."""

        index = self._spatial_index
        if index is None:
            index = SpatialIndex.build(self.vehicles)
            object.__setattr__(self, "_spatial_index", index)
        return index

    @property
    def endpoint_multiplicities(self) -> tuple[tuple[str, int], ...]:
        """Per-endpoint active-pair counts, including counts of one."""

        counts = Counter(endpoint for pair in self.pairs for endpoint in pair.endpoint_ids)
        return tuple(sorted(counts.items()))

    @property
    def shared_endpoint_ids(self) -> tuple[str, ...]:
        """Physical vehicles serving more than one active pair in this frame."""

        return tuple(
            endpoint for endpoint, count in self.endpoint_multiplicities if count > 1
        )

    @property
    def pairs_with_shared_endpoint(self) -> int:
        """Number of active agents affected by endpoint sharing."""

        shared = set(self.shared_endpoint_ids)
        return sum(bool(shared.intersection(pair.endpoint_ids)) for pair in self.pairs)

    @property
    def overlapping_endpoint_assignments(self) -> int:
        """Endpoint assignments participating in a multiplicity above one."""

        return sum(count for _, count in self.endpoint_multiplicities if count > 1)


@dataclass(slots=True)
class PopulationLifecycleTracker:
    """Fail-closed state machine for cross-frame pair lifecycle transitions.

    Per-row constructors validate a lifecycle in isolation.  This tracker adds
    the temporal facts that no single frame can prove: continuing pairs advance
    exactly one episode step, new later arrivals are births, final or vanished
    identities never reappear, and a live pair cannot silently disappear.

    The first observed frame is intentionally a reset boundary.  It may contain
    continuing rows when a rollout starts inside an existing trace episode.
    Call :meth:`reset` before binding the tracker to another trace.
    """

    _trace_id: str | None = field(default=None, init=False, repr=False)
    _previous_frame_index: int | None = field(default=None, init=False, repr=False)
    _previous_time_s: float | None = field(default=None, init=False, repr=False)
    _previous_pairs: dict[str, PopulationPair] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _seen_pair_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _completed_pair_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _observed_frames: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        """Forget one trace only at an explicit environment reset boundary."""

        self._trace_id = None
        self._previous_frame_index = None
        self._previous_time_s = None
        self._previous_pairs.clear()
        self._seen_pair_ids.clear()
        self._completed_pair_ids.clear()
        self._observed_frames = 0

    def observe(self, frame: PopulationFrame) -> None:
        """Validate one chronological frame and commit it as current state."""

        if not isinstance(frame, PopulationFrame):
            raise FrameReplayError("lifecycle tracking requires a PopulationFrame")
        if self._trace_id is None:
            self._accept(frame)
            return
        if frame.trace_id != self._trace_id:
            raise FrameReplayError(
                "population lifecycle cannot cross traces without reset",
                context={"active_trace_id": self._trace_id, "actual": frame.trace_id},
            )
        assert self._previous_frame_index is not None
        expected_index = self._previous_frame_index + 1
        if frame.index != expected_index:
            raise FrameReplayError(
                "population frames must advance by exactly one index",
                context={"actual": frame.index, "expected": expected_index},
            )
        assert self._previous_time_s is not None
        if frame.time_s <= self._previous_time_s:
            raise FrameReplayError(
                "population frame time must increase strictly",
                context={
                    "actual": frame.time_s,
                    "previous": self._previous_time_s,
                },
            )

        current = {pair.pair_id: pair for pair in frame.pairs}
        missing_live = tuple(
            pair_id
            for pair_id, pair in self._previous_pairs.items()
            if not pair.lifecycle.final and pair_id not in current
        )
        if missing_live:
            raise FrameReplayError(
                "a non-final pair disappeared from the next population frame",
                context={"pair_ids": missing_live, "frame_index": frame.index},
            )

        for pair in frame.pairs:
            previous = self._previous_pairs.get(pair.pair_id)
            if previous is None:
                if pair.pair_id in self._seen_pair_ids:
                    raise FrameReplayError(
                        "a completed or disappeared pair identity reappeared",
                        context={"pair_id": pair.pair_id, "frame_index": frame.index},
                    )
                if not pair.lifecycle.born:
                    raise FrameReplayError(
                        "a pair first seen after reset must be marked born",
                        context={"pair_id": pair.pair_id, "frame_index": frame.index},
                    )
                continue
            if previous.lifecycle.final:
                raise FrameReplayError(
                    "a final pair remained active in the next population frame",
                    context={"pair_id": pair.pair_id, "frame_index": frame.index},
                )
            if pair.lifecycle.born:
                raise FrameReplayError(
                    "a continuing pair cannot be marked born again",
                    context={"pair_id": pair.pair_id, "frame_index": frame.index},
                )
            expected_step = previous.episode_step + 1
            if pair.episode_step != expected_step:
                raise FrameReplayError(
                    "a continuing pair must advance by exactly one episode step",
                    context={
                        "pair_id": pair.pair_id,
                        "actual": pair.episode_step,
                        "expected": expected_step,
                    },
                )
            if pair.endpoint_ids != previous.endpoint_ids:
                raise FrameReplayError(
                    "pair endpoints cannot change within one episode",
                    context={
                        "pair_id": pair.pair_id,
                        "actual": pair.endpoint_ids,
                        "expected": previous.endpoint_ids,
                    },
                )

        self._accept(frame)

    def _accept(self, frame: PopulationFrame) -> None:
        current = {pair.pair_id: pair for pair in frame.pairs}
        self._trace_id = frame.trace_id
        self._previous_frame_index = frame.index
        self._previous_time_s = frame.time_s
        self._previous_pairs = current
        self._seen_pair_ids.update(current)
        self._completed_pair_ids.update(
            pair.pair_id for pair in frame.pairs if pair.lifecycle.final
        )
        self._observed_frames += 1

    @property
    def observed_frames(self) -> int:
        return self._observed_frames

    @property
    def live_pair_ids(self) -> tuple[str, ...]:
        """Pairs that must either continue or end in the next observed frame."""

        return tuple(
            pair_id
            for pair_id, pair in self._previous_pairs.items()
            if not pair.lifecycle.final
        )

    @property
    def completed_pair_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed_pair_ids))


@dataclass(frozen=True, slots=True)
class FrameReplayReport:
    """Source-to-frame reconciliation for one complete trace replay."""

    trace_id: str
    split: TraceSplit
    density: float
    config_hash: str
    artifact_manifest_sha256: str
    source_vehicle_rows: int
    source_signal_rows: int
    source_pair_rows: int
    positive_duration_pair_rows: int
    zero_duration_pair_rows: int
    no_decision_pair_rows: int
    decision_pair_episodes: int
    frames: int
    nonempty_frames: int
    pair_instances: int
    births: int
    continuing_instances: int
    natural_terminations: int
    internal_truncations: int
    trace_end_truncations: int
    frames_with_endpoint_overlap: int
    pair_instances_with_endpoint_overlap: int
    overlapping_endpoint_assignments: int
    max_endpoint_multiplicity: int
    source_end_reason_counts: tuple[tuple[str, int], ...]

    def as_dict(self) -> dict[str, object]:
        """JSON-ready mapping with stable field names."""

        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class FrameAggregate:
    """Policy-independent population and lifecycle counts for one frame."""

    frame_index: int
    time_s: float
    active_pairs: int
    births: int
    continuing_pairs: int
    natural_terminations: int
    internal_truncations: int
    trace_end_truncations: int
    shared_endpoints: int
    pairs_with_shared_endpoint: int
    overlapping_endpoint_assignments: int
    max_endpoint_multiplicity: int

    def as_dict(self) -> dict[str, object]:
        """Return a row compatible with the frozen cache schema."""

        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class PairEpisodeSchedule:
    """Policy-independent interval that reconstructs one pair's active frames."""

    segment: TaggedPairSegment
    first_frame: int
    last_frame: int


@dataclass(frozen=True, slots=True)
class _PairSourceStats:
    rows: int
    positive_duration_rows: int
    zero_duration_rows: int
    no_decision_rows: int
    end_reason_counts: tuple[tuple[str, int], ...]


def _frame_ceiling(time_s: float, origin_s: float, period_s: float) -> int:
    return max(0, math.ceil((time_s - origin_s) / period_s - _TIME_TOLERANCE_S))


def _frame_floor(time_s: float, origin_s: float, period_s: float) -> int:
    return math.floor((time_s - origin_s) / period_s + _TIME_TOLERANCE_S)


def _load_pair_schedule(
    reader: MobilityTraceReader,
    *,
    source: FrameTraceSource,
    period_s: float,
    last_frame: int,
) -> tuple[tuple[PairEpisodeSchedule, ...], _PairSourceStats]:
    columns = [
        "trace_id",
        "pair_id",
        "tx_id",
        "rx_id",
        "start_s",
        "end_s",
        "duration_s",
        "initial_distance_m",
        "route_id",
        "eligibility_reason",
        "has_intervening_vehicle",
    ]
    rows: list[Mapping[str, object]] = pq.read_table(
        reader.artifact.path / "pairs.parquet",
        columns=columns,
    ).to_pylist()
    scheduled: list[PairEpisodeSchedule] = []
    seen: set[str] = set()
    zero_duration = 0
    no_decision = 0
    reasons: Counter[str] = Counter()
    for row in rows:
        segment = TaggedPairSegment.from_mapping(row)
        if segment.trace_id != source.trace_id:
            raise FrameReplayError(
                "pair trace identity does not match its artifact",
                context={"pair_id": segment.pair_id, "trace_id": segment.trace_id},
            )
        if segment.pair_id in seen:
            raise FrameReplayError(
                "pair IDs must be unique within a trace",
                context={"trace_id": source.trace_id, "pair_id": segment.pair_id},
            )
        seen.add(segment.pair_id)
        declared_duration = row.get("duration_s")
        if isinstance(declared_duration, bool) or not isinstance(
            declared_duration, (int, float)
        ):
            raise FrameReplayError(
                "pair duration must be numeric",
                context={"pair_id": segment.pair_id},
            )
        if not math.isclose(
            float(declared_duration),
            segment.duration_s,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise FrameReplayError(
                "pair duration disagrees with its start and end times",
                context={
                    "pair_id": segment.pair_id,
                    "declared": float(declared_duration),
                    "derived": segment.duration_s,
                },
            )
        reasons[segment.eligibility_reason] += 1
        if (
            segment.start_s < reader.report.first_time_s - _TIME_TOLERANCE_S
            or segment.end_s > reader.report.last_time_s + _TIME_TOLERANCE_S
        ):
            raise FrameReplayError(
                "pair window lies outside the verified trace interval",
                context={
                    "pair_id": segment.pair_id,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "trace_first_s": reader.report.first_time_s,
                    "trace_last_s": reader.report.last_time_s,
                },
            )
        if segment.duration_s <= _TIME_TOLERANCE_S:
            zero_duration += 1
            continue
        first = _frame_ceiling(segment.start_s, reader.report.first_time_s, period_s)
        final = min(
            last_frame,
            _frame_floor(segment.end_s, reader.report.first_time_s, period_s),
        )
        if final < first:
            no_decision += 1
            continue
        scheduled.append(
            PairEpisodeSchedule(segment=segment, first_frame=first, last_frame=final)
        )
    scheduled.sort(key=lambda item: (item.first_frame, item.segment.pair_id))
    return (
        tuple(scheduled),
        _PairSourceStats(
            rows=len(rows),
            positive_duration_rows=len(rows) - zero_duration,
            zero_duration_rows=zero_duration,
            no_decision_rows=no_decision,
            end_reason_counts=tuple(sorted(reasons.items())),
        ),
    )


def _iter_vehicle_frames(
    reader: MobilityTraceReader,
    *,
    start_time_s: float | None = None,
    end_time_s: float | None = None,
) -> Iterator[tuple[float, tuple[VehicleTraceRecord, ...]]]:
    current_time: float | None = None
    current: list[VehicleTraceRecord] = []
    for vehicle in reader.iter_vehicles(
        start_time_s=start_time_s,
        end_time_s=end_time_s,
    ):
        if current_time is None:
            current_time = vehicle.time_s
        elif vehicle.time_s < current_time - _TIME_TOLERANCE_S:
            raise FrameReplayError(
                "vehicle records are not chronological",
                context={"previous_time_s": current_time, "time_s": vehicle.time_s},
            )
        elif not math.isclose(
            vehicle.time_s,
            current_time,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            yield current_time, tuple(sorted(current, key=lambda item: item.vehicle_id))
            current_time = vehicle.time_s
            current = []
        current.append(vehicle)
    if current_time is not None:
        yield current_time, tuple(sorted(current, key=lambda item: item.vehicle_id))


def _iter_selected_decision_vehicle_frames(
    reader: MobilityTraceReader,
    *,
    start_frame_index: int,
    end_frame_index: int,
    period_s: float,
) -> Iterator[tuple[int, float, tuple[VehicleTraceRecord, ...]]]:
    """Yield every required decision instant in one inclusive bounded range."""

    origin_s = reader.report.first_time_s
    next_index = start_frame_index
    for time_s, vehicles in _iter_vehicle_frames(
        reader,
        start_time_s=origin_s + start_frame_index * period_s,
        end_time_s=origin_s + end_frame_index * period_s,
    ):
        expected = origin_s + next_index * period_s
        if time_s < expected - _TIME_TOLERANCE_S:
            continue
        if time_s > expected + _TIME_TOLERANCE_S:
            raise FrameReplayError(
                "mobility trace is missing a required decision timestamp",
                context={
                    "trace_id": reader.trace_id,
                    "expected_time_s": expected,
                    "next_vehicle_time_s": time_s,
                },
            )
        yield next_index, expected, vehicles
        next_index += 1
        if next_index > end_frame_index:
            return
    if next_index <= end_frame_index:
        raise FrameReplayError(
            "trace ended before every selected decision frame was emitted",
            context={
                "trace_id": reader.trace_id,
                "next_frame_index": next_index,
                "end_frame_index": end_frame_index,
            },
        )


def _iter_decision_vehicle_ids(
    reader: MobilityTraceReader,
    *,
    period_s: float,
    last_frame: int,
) -> Iterator[tuple[int, frozenset[str]]]:
    """Scan only time and identity columns for fast campaign validation."""

    origin_s = reader.report.first_time_s
    current_frame: int | None = None
    current_ids: set[str] = set()
    previous_time: float | None = None
    for path in reader.vehicle_part_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=("time_s", "vehicle_id")):
            times = batch.column(batch.schema.get_field_index("time_s")).to_numpy(
                zero_copy_only=False
            )
            if len(times) == 0:
                continue
            if previous_time is not None and times[0] < previous_time - _TIME_TOLERANCE_S:
                raise FrameReplayError("vehicle records are not chronological")
            if len(times) > 1 and bool(np.any(np.diff(times) < -_TIME_TOLERANCE_S)):
                raise FrameReplayError("vehicle records are not chronological")
            previous_time = float(times[-1])
            coordinates = (times - origin_s) / period_s
            nearest = np.rint(coordinates).astype(np.int64)
            at_decision = (
                (nearest >= 0)
                & (nearest <= last_frame)
                & (np.abs(times - (origin_s + nearest * period_s)) <= _TIME_TOLERANCE_S)
            )
            if not bool(np.any(at_decision)):
                continue
            identifiers = pc.filter(
                batch.column(batch.schema.get_field_index("vehicle_id")),
                pa.array(at_decision),
            ).to_pylist()
            for raw_frame, raw_id in zip(nearest[at_decision], identifiers, strict=True):
                frame_index = int(raw_frame)
                vehicle_id = str(raw_id)
                if current_frame is None:
                    current_frame = frame_index
                elif frame_index != current_frame:
                    yield current_frame, frozenset(current_ids)
                    current_frame = frame_index
                    current_ids = set()
                if vehicle_id in current_ids:
                    raise FrameReplayError(
                        "a population frame cannot contain duplicate vehicles",
                        context={"frame_index": frame_index, "vehicle_id": vehicle_id},
                    )
                current_ids.add(vehicle_id)
    if current_frame is not None:
        yield current_frame, frozenset(current_ids)


class PopulationFrameReader:
    """Verified streaming reader for one complete population trace."""

    def __init__(
        self,
        source: FrameTraceSource,
        *,
        generation_period_s: float,
        expected_config_hash: str | None = None,
        expected_config_scope_hashes: Mapping[str, str] | None = None,
    ) -> None:
        if (
            isinstance(generation_period_s, bool)
            or not isinstance(generation_period_s, (int, float))
            or not math.isfinite(float(generation_period_s))
            or generation_period_s <= 0.0
        ):
            raise ValueError("generation_period_s must be finite and positive")
        self.source = source
        self.generation_period_s = float(generation_period_s)
        self.trace = MobilityTraceReader(source.path)
        manifest = self.trace.artifact.manifest
        if manifest.artifact_id != source.trace_id:
            raise FrameReplayError(
                "trace source identity does not match the artifact manifest",
                context={
                    "source": source.trace_id,
                    "artifact": manifest.artifact_id,
                },
            )
        exact_config_match = (
            expected_config_hash is not None
            and manifest.config_hash == expected_config_hash
        )
        if (
            expected_config_hash is not None
            and not exact_config_match
            and expected_config_scope_hashes is None
        ):
            raise FrameReplayError(
                "trace artifact was generated from a different configuration",
                context={
                    "trace_id": source.trace_id,
                    "artifact_config_hash": manifest.config_hash,
                    "expected_config_hash": expected_config_hash,
                },
            )
        if expected_config_scope_hashes is not None:
            if not isinstance(expected_config_scope_hashes, Mapping) or not (
                expected_config_scope_hashes
            ):
                raise ValueError("expected config scope hashes must be a nonempty mapping")
            for scope, expected in expected_config_scope_hashes.items():
                if (
                    not isinstance(scope, str)
                    or not scope.strip()
                    or not isinstance(expected, str)
                    or len(expected) != 64
                    or any(character not in "0123456789abcdef" for character in expected)
                ):
                    raise ValueError("expected config scope hashes must be named SHA-256 values")
                # An exact run-digest match is already the stronger contract.
                # Scope reconstruction is the explicit fallback for an artifact
                # created before optimizer-only values changed.
                if exact_config_match:
                    continue
                archived = _archived_config_scope_hash(source.path, scope)
                recorded = manifest.config_scope_hashes.get(scope)
                if recorded is not None and recorded != archived:
                    raise FrameReplayError(
                        "trace manifest scope hash differs from its archived configuration",
                        context={
                            "trace_id": source.trace_id,
                            "scope": scope,
                            "manifest_scope_hash": recorded,
                            "archived_scope_hash": archived,
                        },
                    )
                if archived != expected:
                    raise FrameReplayError(
                        "trace artifact is incompatible with the requested configuration scope",
                        context={
                            "trace_id": source.trace_id,
                            "scope": scope,
                            "artifact_scope_hash": archived,
                            "expected_scope_hash": expected,
                        },
                    )
        duration = self.trace.report.last_time_s - self.trace.report.first_time_s
        self.last_frame_index = _frame_floor(
            self.trace.report.last_time_s,
            self.trace.report.first_time_s,
            self.generation_period_s,
        )
        if duration < 0.0 or self.last_frame_index < 0:
            raise FrameReplayError("trace has no valid decision-frame interval")
        self._episodes, self._pair_stats = _load_pair_schedule(
            self.trace,
            source=source,
            period_s=self.generation_period_s,
            last_frame=self.last_frame_index,
        )
        starts: dict[int, list[PairEpisodeSchedule]] = {}
        for episode in self._episodes:
            starts.setdefault(episode.first_frame, []).append(episode)
        self._starts = {
            index: tuple(sorted(items, key=lambda item: item.segment.pair_id))
            for index, items in starts.items()
        }

    @property
    def decision_frame_count(self) -> int:
        return self.last_frame_index + 1

    @property
    def decision_pair_episodes(self) -> int:
        return len(self._episodes)

    @property
    def episode_schedule(self) -> tuple[PairEpisodeSchedule, ...]:
        """Return the frozen, policy-independent episode interval schedule."""

        return self._episodes

    def _lifecycle(
        self,
        episode: PairEpisodeSchedule,
        *,
        frame_index: int,
        next_vehicle_ids: frozenset[str] | None,
    ) -> PairLifecycle:
        born = frame_index == episode.first_frame
        if frame_index != episode.last_frame:
            return PairLifecycle(born=born)
        reason = episode.segment.eligibility_reason
        if reason in _NATURAL_END_REASONS:
            return PairLifecycle(born=born, terminated=True, end_reason=reason)
        if reason == "max_duration":
            endpoints_available = next_vehicle_ids is not None and {
                episode.segment.tx_id,
                episode.segment.rx_id,
            }.issubset(next_vehicle_ids)
            return PairLifecycle(
                born=born,
                truncated=True,
                bootstrap_valid=(
                    frame_index < self.last_frame_index and endpoints_available
                ),
                end_reason=reason,
            )
        return PairLifecycle(born=born, truncated=True, end_reason="trace_end")

    def _build_frame(
        self,
        *,
        frame_index: int,
        time_s: float,
        vehicles: tuple[VehicleTraceRecord, ...],
        next_vehicle_ids: frozenset[str] | None,
        active: dict[str, PairEpisodeSchedule],
    ) -> PopulationFrame:
        for episode in self._starts.get(frame_index, ()):
            pair_id = episode.segment.pair_id
            if pair_id in active:
                raise FrameReplayError(
                    "pair episode became active more than once",
                    context={"trace_id": self.source.trace_id, "pair_id": pair_id},
                )
            active[pair_id] = episode
        by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
        pairs: list[PopulationPair] = []
        completed: list[str] = []
        for pair_id in sorted(active):
            episode = active[pair_id]
            if not (episode.first_frame <= frame_index <= episode.last_frame):
                raise FrameReplayError(
                    "active pair lies outside its scheduled decision interval",
                    context={"pair_id": pair_id, "frame_index": frame_index},
                )
            tx = by_id.get(episode.segment.tx_id)
            rx = by_id.get(episode.segment.rx_id)
            if tx is None or rx is None:
                missing = [
                    endpoint
                    for endpoint in (episode.segment.tx_id, episode.segment.rx_id)
                    if endpoint not in by_id
                ]
                raise FrameReplayError(
                    "active pair endpoint is absent from a decision frame",
                    context={
                        "trace_id": self.source.trace_id,
                        "pair_id": pair_id,
                        "time_s": time_s,
                        "missing": missing,
                    },
                )
            lifecycle = self._lifecycle(
                episode,
                frame_index=frame_index,
                next_vehicle_ids=next_vehicle_ids,
            )
            pairs.append(
                PopulationPair(
                    pair_id=pair_id,
                    episode_step=frame_index - episode.first_frame,
                    transmitter=tx,
                    receiver=rx,
                    lifecycle=lifecycle,
                )
            )
            if lifecycle.final:
                completed.append(pair_id)
        for pair_id in completed:
            del active[pair_id]
        return PopulationFrame(
            source=self.source,
            index=frame_index,
            time_s=time_s,
            vehicles=vehicles,
            pairs=tuple(pairs),
        )

    def iter_frames(
        self,
        *,
        start_frame_index: int = 0,
        max_frames: int | None = None,
    ) -> Iterator[PopulationFrame]:
        """Yield a bounded physical-frame window, including empty populations.

        A nonzero start is an explicit sampled-episode reset. Pairs already
        active at that physical instant retain their physical episode step and
        ``born=False`` lifecycle, while downstream causal state owners reset
        their observation histories at the first sampled frame.
        """

        if (
            not isinstance(start_frame_index, int)
            or isinstance(start_frame_index, bool)
            or not 0 <= start_frame_index <= self.last_frame_index
        ):
            raise ValueError("start_frame_index must identify an available decision frame")
        if max_frames is not None and (
            not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0
        ):
            raise ValueError("max_frames must be a positive integer or None")
        available = self.decision_frame_count - start_frame_index
        selected_frames = available if max_frames is None else min(max_frames, available)
        final_frame_index = start_frame_index + selected_frames - 1
        active = {
            episode.segment.pair_id: episode
            for episode in self._episodes
            if episode.first_frame < start_frame_index <= episode.last_frame
        }
        lifecycle_tracker = PopulationLifecycleTracker()
        next_index = start_frame_index
        emitted = 0
        replay_end = min(final_frame_index + 1, self.last_frame_index)
        decision_frames = iter(
            _iter_selected_decision_vehicle_frames(
                self.trace,
                start_frame_index=start_frame_index,
                end_frame_index=replay_end,
                period_s=self.generation_period_s,
            )
        )
        try:
            current = next(decision_frames)
        except StopIteration as error:  # pragma: no cover - helper rejects this first.
            raise FrameReplayError("selected decision-frame range is empty") from error
        following = next(decision_frames, None)
        while emitted < selected_frames:
            frame_index, expected, vehicles = current
            if frame_index != next_index:
                raise FrameReplayError(
                    "selected decision frames are not contiguous",
                    context={"actual": frame_index, "expected": next_index},
                )
            next_vehicle_ids = (
                frozenset(vehicle.vehicle_id for vehicle in following[2])
                if following is not None
                else None
            )
            frame = self._build_frame(
                frame_index=frame_index,
                time_s=expected,
                vehicles=vehicles,
                next_vehicle_ids=next_vehicle_ids,
                active=active,
            )
            lifecycle_tracker.observe(frame)
            yield frame
            next_index += 1
            emitted += 1
            if emitted >= selected_frames:
                if next_index == self.decision_frame_count and active:
                    raise FrameReplayError(
                        "pair episodes remain active after the final decision frame",
                        context={
                            "trace_id": self.source.trace_id,
                            "pair_ids": sorted(active),
                        },
                    )
                return
            if following is None:
                raise FrameReplayError(
                    "trace ended before every selected decision frame was emitted",
                    context={
                        "trace_id": self.source.trace_id,
                        "next_frame_index": next_index,
                        "end_frame_index": final_frame_index,
                    },
                )
            current = following
            following = next(decision_frames, None)

    def validate(self) -> FrameReplayReport:
        """Replay the full trace and reconcile source, lifecycle, and overlap counts."""

        report, _ = self.validate_with_aggregates()
        return report

    def validate_with_aggregates(
        self,
    ) -> tuple[FrameReplayReport, tuple[FrameAggregate, ...]]:
        """Validate once and also return compact per-frame aggregate rows."""

        frames = self.decision_frame_count
        nonempty = 0
        pair_instances = 0
        births = 0
        continuing = 0
        natural = 0
        internal = 0
        trace_end = 0
        overlap_frames = 0
        overlap_pairs = 0
        overlap_assignments = 0
        max_multiplicity = 0
        aggregates: list[FrameAggregate] = []
        active: dict[str, PairEpisodeSchedule] = {}
        vehicle_frames = iter(
            _iter_decision_vehicle_ids(
                self.trace,
                period_s=self.generation_period_s,
                last_frame=self.last_frame_index,
            )
        )
        for frame_index in range(self.decision_frame_count):
            try:
                vehicle_frame_index, present = next(vehicle_frames)
            except StopIteration as error:
                raise FrameReplayError(
                    "trace ended before every decision frame was validated",
                    context={"expected_frame_index": frame_index},
                ) from error
            if vehicle_frame_index != frame_index:
                raise FrameReplayError(
                    "mobility trace is missing a required decision timestamp",
                    context={
                        "expected_frame_index": frame_index,
                        "next_vehicle_frame_index": vehicle_frame_index,
                    },
                )
            starting = self._starts.get(frame_index, ())
            for episode in starting:
                active[episode.segment.pair_id] = episode
            frame_births = len(starting)
            frame_continuing = len(active) - frame_births
            births += frame_births
            continuing += frame_continuing
            nonempty += bool(active)
            pair_instances += len(active)

            endpoint_counts: Counter[str] = Counter()
            completed: list[str] = []
            frame_natural = 0
            frame_internal = 0
            frame_trace_end = 0
            for pair_id, episode in active.items():
                segment = episode.segment
                missing = {segment.tx_id, segment.rx_id} - present
                if missing:
                    raise FrameReplayError(
                        "active pair endpoint is absent from a decision frame",
                        context={
                            "trace_id": self.source.trace_id,
                            "pair_id": pair_id,
                            "frame_index": frame_index,
                            "missing": sorted(missing),
                        },
                    )
                endpoint_counts.update((segment.tx_id, segment.rx_id))
                if episode.last_frame != frame_index:
                    continue
                completed.append(pair_id)
                reason = segment.eligibility_reason
                if reason in _NATURAL_END_REASONS:
                    frame_natural += 1
                elif reason == "max_duration":
                    frame_internal += 1
                else:
                    frame_trace_end += 1

            shared = {
                endpoint for endpoint, count in endpoint_counts.items() if count > 1
            }
            frame_overlap_pairs = sum(
                bool(shared.intersection((episode.segment.tx_id, episode.segment.rx_id)))
                for episode in active.values()
            )
            frame_overlap_assignments = sum(
                count for count in endpoint_counts.values() if count > 1
            )
            frame_max_multiplicity = max(endpoint_counts.values(), default=0)
            if shared:
                overlap_frames += 1
                overlap_pairs += frame_overlap_pairs
                overlap_assignments += frame_overlap_assignments
            max_multiplicity = max(max_multiplicity, frame_max_multiplicity)
            natural += frame_natural
            internal += frame_internal
            trace_end += frame_trace_end
            aggregates.append(
                FrameAggregate(
                    frame_index=frame_index,
                    time_s=(
                        self.trace.report.first_time_s
                        + frame_index * self.generation_period_s
                    ),
                    active_pairs=len(active),
                    births=frame_births,
                    continuing_pairs=frame_continuing,
                    natural_terminations=frame_natural,
                    internal_truncations=frame_internal,
                    trace_end_truncations=frame_trace_end,
                    shared_endpoints=len(shared),
                    pairs_with_shared_endpoint=frame_overlap_pairs,
                    overlapping_endpoint_assignments=frame_overlap_assignments,
                    max_endpoint_multiplicity=frame_max_multiplicity,
                )
            )
            for pair_id in completed:
                del active[pair_id]
        try:
            unexpected_frame = next(vehicle_frames)
        except StopIteration:
            unexpected_frame = None
        if unexpected_frame is not None:
            raise FrameReplayError(
                "mobility trace contains a decision frame beyond the trace interval"
            )
        if active or births != len(self._episodes) or natural + internal + trace_end != births:
            raise FrameReplayError("source pair episodes do not reconcile with lifecycle events")
        manifest = self.trace.artifact.manifest
        report = FrameReplayReport(
            trace_id=self.source.trace_id,
            split=self.source.split,
            density=self.source.density,
            config_hash=manifest.config_hash,
            artifact_manifest_sha256=self.trace.artifact.manifest_sha256,
            source_vehicle_rows=self.trace.report.vehicle_rows,
            source_signal_rows=self.trace.report.signal_rows,
            source_pair_rows=self._pair_stats.rows,
            positive_duration_pair_rows=self._pair_stats.positive_duration_rows,
            zero_duration_pair_rows=self._pair_stats.zero_duration_rows,
            no_decision_pair_rows=self._pair_stats.no_decision_rows,
            decision_pair_episodes=len(self._episodes),
            frames=frames,
            nonempty_frames=nonempty,
            pair_instances=pair_instances,
            births=births,
            continuing_instances=continuing,
            natural_terminations=natural,
            internal_truncations=internal,
            trace_end_truncations=trace_end,
            frames_with_endpoint_overlap=overlap_frames,
            pair_instances_with_endpoint_overlap=overlap_pairs,
            overlapping_endpoint_assignments=overlap_assignments,
            max_endpoint_multiplicity=max_multiplicity,
            source_end_reason_counts=self._pair_stats.end_reason_counts,
        )
        return report, tuple(aggregates)


__all__ = [
    "FrameReplayError",
    "FrameAggregate",
    "FrameReplayReport",
    "FrameTraceSource",
    "PairEpisodeSchedule",
    "PairLifecycle",
    "PopulationFrame",
    "PopulationFrameReader",
    "PopulationLifecycleTracker",
    "PopulationPair",
    "TraceCatalog",
    "TraceSplit",
]
