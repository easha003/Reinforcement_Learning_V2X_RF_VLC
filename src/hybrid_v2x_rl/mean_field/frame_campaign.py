"""Campaign-wide population-frame validation and cache publication."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.mean_field.frame_cache import (
    FRAME_CACHE_ARTIFACT_FAMILY,
    FRAME_CACHE_FORMAT_VERSION,
    PopulationFrameCacheReader,
    write_population_frame_cache,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameReplayReport,
    PopulationFrameReader,
    TraceCatalog,
)

FRAME_CAMPAIGN_REPORT_VERSION = "1.0.0"
_COUNT_FIELDS = (
    "source_vehicle_rows",
    "source_signal_rows",
    "source_pair_rows",
    "positive_duration_pair_rows",
    "zero_duration_pair_rows",
    "no_decision_pair_rows",
    "decision_pair_episodes",
    "frames",
    "nonempty_frames",
    "pair_instances",
    "births",
    "continuing_instances",
    "natural_terminations",
    "internal_truncations",
    "trace_end_truncations",
    "frames_with_endpoint_overlap",
    "pair_instances_with_endpoint_overlap",
    "overlapping_endpoint_assignments",
)


@dataclass(frozen=True, slots=True)
class FrameCacheResult:
    """Validation result and immutable cache reference for one trace."""

    report: FrameReplayReport
    cache_manifest_sha256: str
    cache_path: Path
    reused: bool
    cache_size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            **self.report.as_dict(),
            "cache_manifest_sha256": self.cache_manifest_sha256,
            "cache_path": str(self.cache_path),
            "cache_reused": self.reused,
            "cache_size_bytes": self.cache_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FrameCampaignReport:
    """Complete source-to-frame reconciliation for an immutable trace catalog."""

    config_hash: str
    generation_period_s: float
    traces: tuple[FrameCacheResult, ...]
    generated_at_utc: datetime

    @property
    def passed(self) -> bool:
        return bool(self.traces)

    def as_dict(self) -> dict[str, object]:
        splits = Counter(item.report.split for item in self.traces)
        densities = Counter(f"{item.report.density:g}" for item in self.traces)
        totals = {
            field: sum(cast(int, getattr(item.report, field)) for item in self.traces)
            for field in _COUNT_FIELDS
        }
        totals["max_endpoint_multiplicity"] = max(
            item.report.max_endpoint_multiplicity for item in self.traces
        )
        totals["cache_size_bytes"] = sum(item.cache_size_bytes for item in self.traces)
        return {
            "report_version": FRAME_CAMPAIGN_REPORT_VERSION,
            "frame_cache_format_version": FRAME_CACHE_FORMAT_VERSION,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "passed": self.passed,
            "config_hash": self.config_hash,
            "generation_period_s": self.generation_period_s,
            "trace_count": len(self.traces),
            "split_counts": dict(sorted(splits.items())),
            "density_counts": dict(sorted(densities.items(), key=lambda item: float(item[0]))),
            "totals": totals,
            "traces": [item.as_dict() for item in self.traces],
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the machine-readable campaign report."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target


def validate_frame_campaign(
    catalog: TraceCatalog,
    *,
    generation_period_s: float,
    expected_config_hash: str,
    artifact_root: str | Path,
    code_version: str,
) -> FrameCampaignReport:
    """Validate or reuse every configured trace cache, failing closed on drift."""

    store = ArtifactStore(artifact_root)
    results: list[FrameCacheResult] = []
    for source in catalog.sources:
        reader = PopulationFrameReader(
            source,
            generation_period_s=generation_period_s,
            expected_config_hash=expected_config_hash,
        )
        cache_path = store.artifact_path(FRAME_CACHE_ARTIFACT_FAMILY, source.trace_id)
        reused = cache_path.exists()
        if reused:
            cached = PopulationFrameCacheReader(cache_path, expected_trace=reader)
            cached.verify()
            artifact = cached.artifact
            report = _report_from_mapping(cast(Mapping[str, object], cached.summary["report"]))
        else:
            artifact, report = write_population_frame_cache(
                reader,
                store,
                code_version=code_version,
            )
        results.append(
            FrameCacheResult(
                report=report,
                cache_manifest_sha256=artifact.manifest_sha256,
                cache_path=artifact.path,
                reused=reused,
                cache_size_bytes=sum(
                    entry.size_bytes for entry in artifact.manifest.files
                ),
            )
        )
    return FrameCampaignReport(
        config_hash=expected_config_hash,
        generation_period_s=generation_period_s,
        traces=tuple(results),
        generated_at_utc=datetime.now(UTC),
    )


def _report_from_mapping(value: Mapping[str, object]) -> FrameReplayReport:
    payload: dict[str, Any] = dict(value)
    reason_counts = payload.get("source_end_reason_counts")
    if not isinstance(reason_counts, list):
        raise ValueError("cached source_end_reason_counts must be a list")
    payload["source_end_reason_counts"] = tuple(
        (str(item[0]), int(item[1]))
        for item in reason_counts
        if isinstance(item, list) and len(item) == 2
    )
    if len(payload["source_end_reason_counts"]) != len(reason_counts):
        raise ValueError("cached source_end_reason_counts entries are invalid")
    return FrameReplayReport(**payload)


__all__ = [
    "FRAME_CAMPAIGN_REPORT_VERSION",
    "FrameCacheResult",
    "FrameCampaignReport",
    "validate_frame_campaign",
]
