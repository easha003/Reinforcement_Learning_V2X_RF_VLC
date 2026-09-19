"""Versioned compact caches for deterministic population-frame replay.

The cache deliberately stores only policy-independent structure.  Vehicle
states remain in the immutable mobility trace; action-dependent RF load,
collisions, packet outcomes, rewards, and mean-field signals are recomputed by
the environment after actions are selected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from hybrid_v2x_rl.artifacts.store import ArtifactStore, StoredArtifact, verify_artifact
from hybrid_v2x_rl.core.errors import TraceIntegrityError
from hybrid_v2x_rl.mean_field.frames import (
    FrameAggregate,
    FrameReplayReport,
    PairEpisodeSchedule,
    PopulationFrameReader,
)
from hybrid_v2x_rl.mobility.trace_io import TRACE_ARTIFACT_TYPE

FRAME_CACHE_ARTIFACT_FAMILY = "frame_caches"
FRAME_CACHE_ARTIFACT_TYPE = "population_frame_cache"
FRAME_CACHE_FORMAT_VERSION = "1.0.0"
EPISODES_FILENAME = "episodes.parquet"
FRAMES_FILENAME = "frames.parquet"
SUMMARY_FILENAME = "summary.json"
_REQUIRED_FILES = (EPISODES_FILENAME, FRAMES_FILENAME, SUMMARY_FILENAME)
_SCHEMA_METADATA = {
    b"hybrid_v2x_rl.frame_cache_format_version": FRAME_CACHE_FORMAT_VERSION.encode()
}
_TERMINATION_KINDS = frozenset(
    {"natural", "internal_truncation", "trace_end_truncation"}
)
TerminationKind = Literal["natural", "internal_truncation", "trace_end_truncation"]

EPISODE_CACHE_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("density_veh_per_lane_km", pa.float64(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("tx_id", pa.string(), nullable=False),
        pa.field("rx_id", pa.string(), nullable=False),
        pa.field("source_start_s", pa.float64(), nullable=False),
        pa.field("source_end_s", pa.float64(), nullable=False),
        pa.field("first_frame", pa.int64(), nullable=False),
        pa.field("last_frame", pa.int64(), nullable=False),
        pa.field("first_decision_time_s", pa.float64(), nullable=False),
        pa.field("last_decision_time_s", pa.float64(), nullable=False),
        pa.field("source_end_reason", pa.string(), nullable=False),
        pa.field("termination_kind", pa.string(), nullable=False),
        pa.field("bootstrap_valid", pa.bool_(), nullable=False),
        pa.field("initial_distance_m", pa.float64(), nullable=False),
        pa.field("route_id", pa.string(), nullable=False),
        pa.field("has_intervening_vehicle", pa.bool_(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)

FRAME_AGGREGATE_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("density_veh_per_lane_km", pa.float64(), nullable=False),
        pa.field("frame_index", pa.int64(), nullable=False),
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("active_pairs", pa.int64(), nullable=False),
        pa.field("births", pa.int64(), nullable=False),
        pa.field("continuing_pairs", pa.int64(), nullable=False),
        pa.field("natural_terminations", pa.int64(), nullable=False),
        pa.field("internal_truncations", pa.int64(), nullable=False),
        pa.field("trace_end_truncations", pa.int64(), nullable=False),
        pa.field("shared_endpoints", pa.int64(), nullable=False),
        pa.field("pairs_with_shared_endpoint", pa.int64(), nullable=False),
        pa.field("overlapping_endpoint_assignments", pa.int64(), nullable=False),
        pa.field("max_endpoint_multiplicity", pa.int64(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)

CACHE_INCLUDES = (
    "episode and immutable split membership",
    "pair endpoints and source lifecycle metadata",
    "active-frame interval and decision timestamps",
    "per-frame population, lifecycle, and endpoint-overlap counts",
)
CACHE_EXCLUDES = (
    "policy actions",
    "RF demand, airtime contention, and collisions",
    "packet outcomes, rewards, and costs",
    "mean-field observations",
)


@dataclass(frozen=True, slots=True)
class CachedPairEpisode:
    """One verified pair interval loaded from a compact replay cache."""

    pair_id: str
    tx_id: str
    rx_id: str
    first_frame: int
    last_frame: int
    first_decision_time_s: float
    last_decision_time_s: float
    source_end_reason: str
    termination_kind: TerminationKind
    bootstrap_valid: bool


def _termination_kind(
    episode: PairEpisodeSchedule,
    *,
    final_trace_frame: int,
) -> tuple[TerminationKind, bool]:
    reason = episode.segment.eligibility_reason
    if reason in {"outside_range_1s", "route_diverged", "vehicle_missing"}:
        return "natural", False
    if reason == "max_duration":
        return "internal_truncation", episode.last_frame < final_trace_frame
    return "trace_end_truncation", False


def _episode_row(
    reader: PopulationFrameReader,
    episode: PairEpisodeSchedule,
) -> dict[str, object]:
    segment = episode.segment
    origin_s = reader.trace.report.first_time_s
    kind, bootstrap_valid = _termination_kind(
        episode,
        final_trace_frame=reader.last_frame_index,
    )
    return {
        "trace_id": reader.source.trace_id,
        "split": reader.source.split,
        "density_veh_per_lane_km": reader.source.density,
        "pair_id": segment.pair_id,
        "tx_id": segment.tx_id,
        "rx_id": segment.rx_id,
        "source_start_s": segment.start_s,
        "source_end_s": segment.end_s,
        "first_frame": episode.first_frame,
        "last_frame": episode.last_frame,
        "first_decision_time_s": origin_s + episode.first_frame * reader.generation_period_s,
        "last_decision_time_s": origin_s + episode.last_frame * reader.generation_period_s,
        "source_end_reason": segment.eligibility_reason,
        "termination_kind": kind,
        "bootstrap_valid": bootstrap_valid,
        "initial_distance_m": segment.initial_distance_m,
        "route_id": segment.route_id,
        "has_intervening_vehicle": segment.has_intervening_vehicle,
    }


def _frame_row(
    reader: PopulationFrameReader,
    aggregate: FrameAggregate,
) -> dict[str, object]:
    return {
        "trace_id": reader.source.trace_id,
        "split": reader.source.split,
        "density_veh_per_lane_km": reader.source.density,
        **aggregate.as_dict(),
    }


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_population_frame_cache(
    reader: PopulationFrameReader,
    store: ArtifactStore,
    *,
    code_version: str,
) -> tuple[StoredArtifact, FrameReplayReport]:
    """Validate a trace once and atomically publish its compact replay cache."""

    report, aggregates = reader.validate_with_aggregates()
    episode_rows = sorted(
        (_episode_row(reader, episode) for episode in reader.episode_schedule),
        key=lambda row: cast(str, row["pair_id"]),
    )
    frame_rows = [_frame_row(reader, aggregate) for aggregate in aggregates]
    if sum(cast(int, row["active_pairs"]) for row in frame_rows) != report.pair_instances:
        raise TraceIntegrityError(
            "frame-cache aggregates do not reconcile with replay report",
            artifact_path=reader.source.path,
        )

    summary: dict[str, object] = {
        "cache_format_version": FRAME_CACHE_FORMAT_VERSION,
        "trace_id": reader.source.trace_id,
        "split": reader.source.split,
        "density_veh_per_lane_km": reader.source.density,
        "generation_period_s": reader.generation_period_s,
        "first_time_s": reader.trace.report.first_time_s,
        "last_time_s": reader.trace.report.last_time_s,
        "policy_independent": True,
        "includes": list(CACHE_INCLUDES),
        "excludes": list(CACHE_EXCLUDES),
        "report": report.as_dict(),
    }

    def produce(root: Path) -> None:
        pq.write_table(
            pa.Table.from_pylist(episode_rows, schema=EPISODE_CACHE_SCHEMA),
            root / EPISODES_FILENAME,
            compression="zstd",
            version="2.6",
            write_statistics=True,
        )
        pq.write_table(
            pa.Table.from_pylist(frame_rows, schema=FRAME_AGGREGATE_SCHEMA),
            root / FRAMES_FILENAME,
            compression="zstd",
            version="2.6",
            write_statistics=True,
        )
        _write_json(root / SUMMARY_FILENAME, summary)

    artifact = store.create(
        family=FRAME_CACHE_ARTIFACT_FAMILY,
        artifact_id=reader.source.trace_id,
        artifact_type=FRAME_CACHE_ARTIFACT_TYPE,
        config_hash=report.config_hash,
        code_version=code_version,
        producer=produce,
        required_files=_REQUIRED_FILES,
        input_artifacts=(reader.trace.artifact.reference,),
        software_versions={"pyarrow": pa.__version__},
        notes=(
            f"frame_cache_format_version={FRAME_CACHE_FORMAT_VERSION}",
            "contains policy-independent replay structure only",
        ),
    )
    PopulationFrameCacheReader(artifact.path, expected_trace=reader).verify()
    return artifact, report


class PopulationFrameCacheReader:
    """Verified reader and interval lookup for one compact frame cache."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_trace: PopulationFrameReader | None = None,
    ) -> None:
        self.artifact = verify_artifact(
            path,
            expected_artifact_type=FRAME_CACHE_ARTIFACT_TYPE,
            expected_artifact_id=Path(path).name,
        )
        self.expected_trace = expected_trace
        self._summary: dict[str, Any] | None = None
        self._episodes: tuple[CachedPairEpisode, ...] | None = None

    @property
    def summary(self) -> Mapping[str, Any]:
        if self._summary is None:
            self.verify()
        assert self._summary is not None
        return self._summary

    @property
    def episodes(self) -> tuple[CachedPairEpisode, ...]:
        if self._episodes is None:
            self.verify()
        assert self._episodes is not None
        return self._episodes

    def active_pair_ids(self, frame_index: int) -> tuple[str, ...]:
        """Reconstruct stable active membership from cached episode intervals."""

        if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        return tuple(
            episode.pair_id
            for episode in self.episodes
            if episode.first_frame <= frame_index <= episode.last_frame
        )

    def iter_frame_rows(self) -> Iterator[Mapping[str, object]]:
        """Yield the compact aggregate table in chronological order."""

        self.verify()
        rows: list[dict[str, object]] = pq.read_table(
            self.artifact.path / FRAMES_FILENAME
        ).to_pylist()
        yield from rows

    def verify(self) -> None:
        """Verify version, schemas, provenance, identities, and aggregate counts."""

        episode_path = self.artifact.path / EPISODES_FILENAME
        frame_path = self.artifact.path / FRAMES_FILENAME
        for path, schema in (
            (episode_path, EPISODE_CACHE_SCHEMA),
            (frame_path, FRAME_AGGREGATE_SCHEMA),
        ):
            actual = pq.ParquetFile(path).schema_arrow
            if not actual.equals(schema, check_metadata=True):
                raise TraceIntegrityError(
                    "frame-cache Parquet schema does not match the frozen format",
                    artifact_path=self.artifact.path,
                    context={"file": path.name},
                )
        try:
            summary = json.loads(
                (self.artifact.path / SUMMARY_FILENAME).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TraceIntegrityError(
                "frame-cache summary is invalid",
                artifact_path=self.artifact.path,
            ) from error
        if not isinstance(summary, dict):
            raise TraceIntegrityError(
                "frame-cache summary must be a JSON object",
                artifact_path=self.artifact.path,
            )
        if summary.get("cache_format_version") != FRAME_CACHE_FORMAT_VERSION:
            raise TraceIntegrityError(
                "unsupported frame-cache format version",
                artifact_path=self.artifact.path,
            )
        trace_id = self.artifact.manifest.artifact_id
        report = summary.get("report")
        if (
            summary.get("trace_id") != trace_id
            or summary.get("policy_independent") is not True
            or summary.get("includes") != list(CACHE_INCLUDES)
            or summary.get("excludes") != list(CACHE_EXCLUDES)
            or not isinstance(report, dict)
        ):
            raise TraceIntegrityError(
                "frame-cache identity, scope declaration, or report is invalid",
                artifact_path=self.artifact.path,
            )
        references = self.artifact.manifest.input_artifacts
        if (
            len(references) != 1
            or references[0].artifact_type != TRACE_ARTIFACT_TYPE
            or references[0].artifact_id != trace_id
            or report.get("config_hash") != self.artifact.manifest.config_hash
            or report.get("artifact_manifest_sha256")
            != references[0].manifest_sha256
        ):
            raise TraceIntegrityError(
                "frame-cache manifest and source provenance do not reconcile",
                artifact_path=self.artifact.path,
            )
        episode_rows: list[dict[str, object]] = pq.read_table(episode_path).to_pylist()
        frame_rows: list[dict[str, object]] = pq.read_table(frame_path).to_pylist()
        _verify_rows(
            trace_id=trace_id,
            summary=summary,
            report=report,
            episode_rows=episode_rows,
            frame_rows=frame_rows,
            artifact_path=self.artifact.path,
        )
        if self.expected_trace is not None:
            trace = self.expected_trace
            expected_reference = trace.trace.artifact.reference
            if (
                references != [expected_reference]
                or self.artifact.manifest.config_hash
                != trace.trace.artifact.manifest.config_hash
            ):
                raise TraceIntegrityError(
                    "frame cache does not reference the expected mobility trace",
                    artifact_path=self.artifact.path,
                )
            period = summary.get("generation_period_s")
            if (
                summary.get("trace_id") != trace.source.trace_id
                or summary.get("split") != trace.source.split
                or summary.get("density_veh_per_lane_km") != trace.source.density
                or isinstance(period, bool)
                or not isinstance(period, (int, float))
                or not math.isclose(
                    float(period),
                    trace.generation_period_s,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise TraceIntegrityError(
                    "frame cache does not match the expected replay source",
                    artifact_path=self.artifact.path,
                )
        self._summary = summary
        self._episodes = tuple(_cached_episode(row) for row in episode_rows)


def _cached_episode(row: Mapping[str, object]) -> CachedPairEpisode:
    kind = row["termination_kind"]
    if not isinstance(kind, str) or kind not in _TERMINATION_KINDS:
        raise ValueError(f"invalid cached termination kind {kind!r}")
    return CachedPairEpisode(
        pair_id=cast(str, row["pair_id"]),
        tx_id=cast(str, row["tx_id"]),
        rx_id=cast(str, row["rx_id"]),
        first_frame=cast(int, row["first_frame"]),
        last_frame=cast(int, row["last_frame"]),
        first_decision_time_s=cast(float, row["first_decision_time_s"]),
        last_decision_time_s=cast(float, row["last_decision_time_s"]),
        source_end_reason=cast(str, row["source_end_reason"]),
        termination_kind=cast(TerminationKind, kind),
        bootstrap_valid=cast(bool, row["bootstrap_valid"]),
    )


def _verify_rows(
    *,
    trace_id: str,
    summary: Mapping[str, object],
    report: Mapping[str, object],
    episode_rows: list[dict[str, object]],
    frame_rows: list[dict[str, object]],
    artifact_path: Path,
) -> None:
    pair_ids = [row["pair_id"] for row in episode_rows]
    frame_indices = [row["frame_index"] for row in frame_rows]
    expected_frames = int(cast(int, report.get("frames", -1)))
    checks = {
        "episode row count": len(episode_rows)
        == int(cast(int, report.get("decision_pair_episodes", -1))),
        "unique pair IDs": len(pair_ids) == len(set(pair_ids)),
        "pair ordering": pair_ids == sorted(pair_ids, key=str),
        "frame row count": len(frame_rows) == expected_frames,
        "contiguous frame indices": frame_indices == list(range(expected_frames)),
        "pair instances": sum(cast(int, row["active_pairs"]) for row in frame_rows)
        == int(cast(int, report.get("pair_instances", -1))),
        "births": sum(cast(int, row["births"]) for row in frame_rows)
        == int(cast(int, report.get("births", -1))),
        "continuing instances": sum(
            cast(int, row["continuing_pairs"]) for row in frame_rows
        )
        == int(cast(int, report.get("continuing_instances", -1))),
        "nonempty frames": sum(cast(int, row["active_pairs"]) > 0 for row in frame_rows)
        == int(cast(int, report.get("nonempty_frames", -1))),
        "natural terminations": sum(
            cast(int, row["natural_terminations"]) for row in frame_rows
        )
        == int(cast(int, report.get("natural_terminations", -1))),
        "internal truncations": sum(
            cast(int, row["internal_truncations"]) for row in frame_rows
        )
        == int(cast(int, report.get("internal_truncations", -1))),
        "trace-end truncations": sum(
            cast(int, row["trace_end_truncations"]) for row in frame_rows
        )
        == int(cast(int, report.get("trace_end_truncations", -1))),
        "frames with endpoint overlap": sum(
            cast(int, row["shared_endpoints"]) > 0 for row in frame_rows
        )
        == int(cast(int, report.get("frames_with_endpoint_overlap", -1))),
        "pairs with endpoint overlap": sum(
            cast(int, row["pairs_with_shared_endpoint"]) for row in frame_rows
        )
        == int(cast(int, report.get("pair_instances_with_endpoint_overlap", -1))),
        "overlapping endpoint assignments": sum(
            cast(int, row["overlapping_endpoint_assignments"])
            for row in frame_rows
        )
        == int(cast(int, report.get("overlapping_endpoint_assignments", -1))),
        "maximum endpoint multiplicity": max(
            (cast(int, row["max_endpoint_multiplicity"]) for row in frame_rows),
            default=0,
        )
        == int(cast(int, report.get("max_endpoint_multiplicity", -1))),
        "per-frame population partition": all(
            cast(int, row["births"]) + cast(int, row["continuing_pairs"])
            == cast(int, row["active_pairs"])
            for row in frame_rows
        ),
        "valid episode intervals": all(
            0 <= cast(int, row["first_frame"]) <= cast(int, row["last_frame"])
            < expected_frames
            for row in episode_rows
        ),
    }
    identity_ok = all(
        row["trace_id"] == trace_id
        and row["split"] == summary.get("split")
        and row["density_veh_per_lane_km"] == summary.get("density_veh_per_lane_km")
        for row in (*episode_rows, *frame_rows)
    )
    checks["row identities"] = identity_ok
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise TraceIntegrityError(
            "frame-cache rows failed reconciliation",
            artifact_path=artifact_path,
            context={"failed": failed},
        )


__all__ = [
    "CACHE_EXCLUDES",
    "CACHE_INCLUDES",
    "EPISODE_CACHE_SCHEMA",
    "FRAME_AGGREGATE_SCHEMA",
    "FRAME_CACHE_ARTIFACT_FAMILY",
    "FRAME_CACHE_ARTIFACT_TYPE",
    "FRAME_CACHE_FORMAT_VERSION",
    "CachedPairEpisode",
    "PopulationFrameCacheReader",
    "write_population_frame_cache",
]
