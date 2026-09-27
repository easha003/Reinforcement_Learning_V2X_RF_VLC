"""Audit the global mean-field RF pool against the declared local RF domain."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from hybrid_v2x_rl.agents.regime_evaluation import (
    EvaluationWindow,
    load_frozen_regime_audit,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.env.perception import CONTENTION_RADIUS_M as ACTOR_RADIUS_M
from hybrid_v2x_rl.env.rollout import CONTENTION_RADIUS_M as LEGACY_RADIUS_M
from hybrid_v2x_rl.mean_field.frames import (
    PopulationFrame,
    PopulationFrameReader,
    TraceCatalog,
)

CONTENTION_DOMAIN_AUDIT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.rf-contention-domain-audit.v1"
)


class ContentionDomainAuditError(HybridV2XError):
    """The RF contention-domain audit input or accounting is invalid."""


@dataclass(frozen=True, slots=True)
class FrameContentionDomains:
    """Exact geometric domain sizes for every active pair in one frame."""

    trace_id: str
    frame_index: int
    active_pairs: int
    vehicles: int
    local_pair_flows: tuple[int, ...]
    local_vehicle_neighbours: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.trace_id or self.frame_index < 0:
            raise ContentionDomainAuditError("frame contention identity is invalid")
        if self.active_pairs < 0 or self.vehicles < 0:
            raise ContentionDomainAuditError("frame contention counts cannot be negative")
        if len(self.local_pair_flows) != self.active_pairs or len(
            self.local_vehicle_neighbours
        ) != self.active_pairs:
            raise ContentionDomainAuditError(
                "local contention rows must align with active pairs"
            )
        if any(not 1 <= count <= self.active_pairs for count in self.local_pair_flows):
            raise ContentionDomainAuditError(
                "a nonempty pair-local domain must include its focal flow"
            )
        if any(not 0 <= count < self.vehicles for count in self.local_vehicle_neighbours):
            raise ContentionDomainAuditError("local vehicle-neighbour count is invalid")


def measure_frame_contention_domains(
    frame: PopulationFrame,
    *,
    radius_m: float,
) -> FrameContentionDomains:
    """Measure exact 2-D transmitter-centred domains without choosing actions."""

    if not isinstance(frame, PopulationFrame):
        raise ContentionDomainAuditError(
            "contention-domain measurement requires a PopulationFrame"
        )
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ContentionDomainAuditError("contention radius must be finite and positive")
    if not frame.pairs:
        return FrameContentionDomains(
            trace_id=frame.trace_id,
            frame_index=frame.index,
            active_pairs=0,
            vehicles=len(frame.vehicles),
            local_pair_flows=(),
            local_vehicle_neighbours=(),
        )

    pair_transmitter_ids = tuple(pair.transmitter.vehicle_id for pair in frame.pairs)
    unique_transmitter_ids = tuple(dict.fromkeys(pair_transmitter_ids))
    transmitter_by_id = {
        pair.transmitter.vehicle_id: pair.transmitter for pair in frame.pairs
    }
    centres = np.asarray(
        [
            (transmitter_by_id[vehicle_id].x_m, transmitter_by_id[vehicle_id].y_m)
            for vehicle_id in unique_transmitter_ids
        ],
        dtype=np.float64,
    )
    pair_positions = np.asarray(
        [(pair.transmitter.x_m, pair.transmitter.y_m) for pair in frame.pairs],
        dtype=np.float64,
    )
    vehicle_positions = np.asarray(
        [(vehicle.x_m, vehicle.y_m) for vehicle in frame.vehicles],
        dtype=np.float64,
    )
    radius_squared = radius_m * radius_m
    pair_squared = np.sum(
        (centres[:, np.newaxis, :] - pair_positions[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    vehicle_squared = np.sum(
        (centres[:, np.newaxis, :] - vehicle_positions[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    pair_counts = np.count_nonzero(pair_squared <= radius_squared, axis=1)
    vehicle_counts = np.count_nonzero(vehicle_squared <= radius_squared, axis=1) - 1
    row_by_transmitter = {
        vehicle_id: row for row, vehicle_id in enumerate(unique_transmitter_ids)
    }
    return FrameContentionDomains(
        trace_id=frame.trace_id,
        frame_index=frame.index,
        active_pairs=len(frame.pairs),
        vehicles=len(frame.vehicles),
        local_pair_flows=tuple(
            int(pair_counts[row_by_transmitter[vehicle_id]])
            for vehicle_id in pair_transmitter_ids
        ),
        local_vehicle_neighbours=tuple(
            int(vehicle_counts[row_by_transmitter[vehicle_id]])
            for vehicle_id in pair_transmitter_ids
        ),
    )


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ContentionDomainAuditError("cannot summarize an empty distribution")
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.all(np.isfinite(array))):
        raise ContentionDomainAuditError("contention distribution is non-finite")
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


@dataclass(slots=True)
class _DomainTally:
    sampled_frames: int = 0
    empty_frames: int = 0
    active_pairs_per_frame: list[float] = field(default_factory=list)
    vehicles_per_frame: list[float] = field(default_factory=list)
    global_pairs_by_row: list[float] = field(default_factory=list)
    local_pair_flows: list[float] = field(default_factory=list)
    local_vehicle_neighbours: list[float] = field(default_factory=list)

    def observe(self, measurement: FrameContentionDomains) -> None:
        self.sampled_frames += 1
        if measurement.active_pairs == 0:
            self.empty_frames += 1
            return
        self.active_pairs_per_frame.append(float(measurement.active_pairs))
        self.vehicles_per_frame.append(float(measurement.vehicles))
        self.global_pairs_by_row.extend(
            [float(measurement.active_pairs)] * measurement.active_pairs
        )
        self.local_pair_flows.extend(float(value) for value in measurement.local_pair_flows)
        self.local_vehicle_neighbours.extend(
            float(value) for value in measurement.local_vehicle_neighbours
        )

    def merge(self, other: _DomainTally) -> None:
        self.sampled_frames += other.sampled_frames
        self.empty_frames += other.empty_frames
        self.active_pairs_per_frame.extend(other.active_pairs_per_frame)
        self.vehicles_per_frame.extend(other.vehicles_per_frame)
        self.global_pairs_by_row.extend(other.global_pairs_by_row)
        self.local_pair_flows.extend(other.local_pair_flows)
        self.local_vehicle_neighbours.extend(other.local_vehicle_neighbours)

    def as_dict(self, *, density: float | None) -> dict[str, object]:
        rows = len(self.local_pair_flows)
        if (
            self.sampled_frames < 1
            or not self.active_pairs_per_frame
            or rows < 1
            or len(self.global_pairs_by_row) != rows
            or len(self.local_vehicle_neighbours) != rows
        ):
            raise ContentionDomainAuditError("contention-domain tally is incomplete")
        ratios = [
            global_count / local_count
            for global_count, local_count in zip(
                self.global_pairs_by_row, self.local_pair_flows, strict=True
            )
        ]
        outside_fractions = [
            1.0 - local_count / global_count
            for global_count, local_count in zip(
                self.global_pairs_by_row, self.local_pair_flows, strict=True
            )
        ]
        globally_scoped_rows = sum(
            global_count == local_count
            for global_count, local_count in zip(
                self.global_pairs_by_row, self.local_pair_flows, strict=True
            )
        )
        return {
            "density_vehicles_per_lane_km": density,
            "sampled_frames": self.sampled_frames,
            "nonempty_frames": len(self.active_pairs_per_frame),
            "empty_frames": self.empty_frames,
            "pair_rows": rows,
            "active_pairs_per_nonempty_frame": _distribution(
                self.active_pairs_per_frame
            ),
            "vehicles_per_nonempty_frame": _distribution(self.vehicles_per_frame),
            "local_active_pair_flows_per_pair": _distribution(self.local_pair_flows),
            "local_vehicle_neighbours_per_pair": _distribution(
                self.local_vehicle_neighbours
            ),
            "global_to_local_pair_domain_ratio": _distribution(ratios),
            "pair_weighted_fraction_of_global_flows_outside_local_domain": float(
                math.fsum(outside_fractions) / rows
            ),
            "rows_whose_local_domain_equals_global_frame": globally_scoped_rows,
            "fraction_rows_whose_local_domain_equals_global_frame": (
                globally_scoped_rows / rows
            ),
        }


@dataclass(frozen=True, slots=True)
class ContentionDomainAuditReport:
    config_hash: str
    policy_environment_scope_hash: str
    audit_path: Path
    audit_sha256: str
    radius_m: float
    windows: tuple[EvaluationWindow, ...]
    densities: tuple[dict[str, object], ...]
    campaign: dict[str, object]
    generated_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        campaign_equal_fraction = self.campaign[
            "fraction_rows_whose_local_domain_equals_global_frame"
        ]
        if not isinstance(campaign_equal_fraction, float):
            raise ContentionDomainAuditError("campaign domain fraction is invalid")
        domains_match = campaign_equal_fraction == 1.0
        return {
            "schema": CONTENTION_DOMAIN_AUDIT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "scope": (
                "read-only geometry audit of global frame pooling against the "
                "declared transmitter-centred contention radius"
            ),
            "test_split_opened": False,
            "config_hash": self.config_hash,
            "policy_environment_scope_hash": self.policy_environment_scope_hash,
            "contention_radius_m": self.radius_m,
            "window_source": {
                "path": str(self.audit_path),
                "sha256": self.audit_sha256,
            },
            "sampled_windows": [window.as_dict() for window in self.windows],
            "implementation_findings": {
                "actor_congestion_scope": (
                    "tracked vehicles within 200 m of the pair transmitter"
                ),
                "legacy_physical_scope": (
                    "true vehicles within 200 m of the pair transmitter"
                ),
                "mean_field_demand_scope": "RF attempts from every active pair in the frame",
                "mean_field_collision_scope": (
                    "every other offered RF attempt in the frame"
                ),
                "mean_field_sensed_fraction": 1.0,
                "pair_local_measurement": (
                    "active pair flows whose transmitters lie within 200 m of the "
                    "focal transmitter; includes the focal flow"
                ),
            },
            "densities": list(self.densities),
            "campaign": self.campaign,
            "decision": {
                "global_frame_pool_matches_declared_local_domain": domains_match,
                "rf_contention_model_repair_required": not domains_match,
                "ppo_training_authorized": False,
                "system_feasibility_frontier_authorized": False,
                "next_action": (
                    "retain the global frame pool and document why every frame is one "
                    "physical collision domain"
                    if domains_match
                    else "specify and implement pair-local RF contention with spatial reuse, "
                    "then rerun the exact population-joint feasibility oracle"
                ),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
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


def build_contention_domain_audit(
    config: ProjectConfig,
    *,
    state_regime_audit_path: str | Path,
) -> ContentionDomainAuditReport:
    """Replay the frozen validation windows and compare global/local domains."""

    if not isinstance(config, ProjectConfig):
        raise ContentionDomainAuditError("domain audit requires ProjectConfig")
    if not math.isclose(ACTOR_RADIUS_M, LEGACY_RADIUS_M, rel_tol=0.0, abs_tol=0.0):
        raise ContentionDomainAuditError(
            "actor and legacy physical contention radii have diverged"
        )
    audit = Path(state_regime_audit_path).expanduser().resolve(strict=True)
    digest = config_hash(config)
    environment_digest = scope_hash(config, "policy_environment")
    _, windows, _, audit_sha256 = load_frozen_regime_audit(
        audit,
        expected_policy_environment_scope_hash=environment_digest,
    )
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation_sources = {
        source.trace_id: source for source in catalog.for_split("validation")
    }
    tallies: dict[float, _DomainTally] = {}
    for window in windows:
        try:
            source = validation_sources[window.trace_id]
        except KeyError as error:
            raise ContentionDomainAuditError(
                "audit validation window is absent from the configured catalog"
            ) from error
        if source.density != window.density:
            raise ContentionDomainAuditError(
                "audit window density differs from the trace catalog"
            )
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=digest,
            expected_config_scope_hashes={
                "mobility_trace": scope_hash(config, "mobility_trace")
            },
        )
        tally = tallies.setdefault(window.density, _DomainTally())
        for frame in reader.iter_frames(
            start_frame_index=window.start_frame_index,
            max_frames=window.frames,
        ):
            tally.observe(
                measure_frame_contention_domains(frame, radius_m=LEGACY_RADIUS_M)
            )
    expected_densities = tuple(sorted(set(window.density for window in windows)))
    if set(tallies) != set(expected_densities):
        raise ContentionDomainAuditError("contention audit density coverage is incomplete")
    density_rows = tuple(
        tallies[density].as_dict(density=density) for density in expected_densities
    )
    campaign_tally = _DomainTally()
    for density in expected_densities:
        campaign_tally.merge(tallies[density])
    return ContentionDomainAuditReport(
        config_hash=digest,
        policy_environment_scope_hash=environment_digest,
        audit_path=audit,
        audit_sha256=audit_sha256,
        radius_m=LEGACY_RADIUS_M,
        windows=windows,
        densities=density_rows,
        campaign=campaign_tally.as_dict(density=None),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "CONTENTION_DOMAIN_AUDIT_SCHEMA",
    "ContentionDomainAuditError",
    "ContentionDomainAuditReport",
    "FrameContentionDomains",
    "build_contention_domain_audit",
    "measure_frame_contention_domains",
]
