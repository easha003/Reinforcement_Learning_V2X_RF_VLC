"""Executor and result contract for the predeclared feasibility frontier."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from hybrid_v2x_rl.agents.joint_oracle_evaluation import (
    JointOracleWindowEvaluation,
    evaluate_pair_local_joint_windows,
)
from hybrid_v2x_rl.agents.regime_evaluation import (
    EvaluationWindow,
    load_frozen_regime_audit,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    FrontierEvaluationCell,
    FrontierPhysicalPoint,
    SystemFeasibilityFrontierDeclaration,
)
from hybrid_v2x_rl.channels.rf.diversity import RFReceiveDiversity
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_config
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.frames import (
    PopulationFrameReader,
    TraceCatalog,
)
from hybrid_v2x_rl.mean_field.joint_risk_oracle import JOINT_RISK_ORACLE_METHOD
from hybrid_v2x_rl.mean_field.local_rf_pipeline import LocalRFPhysicsModel
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationState,
    ObservationNormalizer,
)

SYSTEM_FEASIBILITY_FRONTIER_DRY_RUN_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-dry-run.v1"
)
SYSTEM_FEASIBILITY_FRONTIER_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-result.v1"
)
SYSTEM_FEASIBILITY_FRONTIER_PROGRESS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-progress.v1"
)

DensityVerdict = Literal["feasible", "infeasible", "inconclusive"]
CellVerdict = DensityVerdict
DesignVerdict = DensityVerdict
ProgressCallback = Callable[[int, int, str], None]


class SystemFeasibilityExecutionError(HybridV2XError):
    """A frontier execution input, result, or invariant is invalid."""


def _boolean(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise SystemFeasibilityExecutionError(
            "joint-search row has an invalid boolean field",
            context={"field": key, "value": value},
        )
    return value


def _number(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise SystemFeasibilityExecutionError(
            "joint-search row has an invalid numeric field",
            context={"field": key, "value": value},
        )
    return float(value)


def density_verdict(
    row: Mapping[str, object],
    *,
    miss_budget: float,
) -> DensityVerdict:
    """Apply the frozen asymmetric certificate rule to one density row."""

    risk = _number(row, "mean_pair_local_candidate_conditional_miss_risk")
    candidate_passes = _boolean(
        row,
        "pair_local_candidate_mean_meets_budget",
    )
    lower_bound_fails = _boolean(
        row,
        "certified_lower_bound_exceeds_budget",
    )
    all_exact = _boolean(row, "all_frames_optimality_proven")
    if candidate_passes != (risk <= miss_budget):
        raise SystemFeasibilityExecutionError(
            "candidate budget flag disagrees with conditional miss risk"
        )
    if candidate_passes:
        return "feasible"
    if lower_bound_fails or all_exact:
        return "infeasible"
    return "inconclusive"


def cell_verdict(verdicts: tuple[DensityVerdict, ...]) -> CellVerdict:
    """Aggregate every density without turning an open gap into a failure."""

    if not verdicts:
        raise SystemFeasibilityExecutionError("frontier cell has no density verdicts")
    if all(verdict == "feasible" for verdict in verdicts):
        return "feasible"
    if any(verdict == "infeasible" for verdict in verdicts):
        return "infeasible"
    return "inconclusive"


def _design_verdict(verdicts: tuple[CellVerdict, ...]) -> DesignVerdict:
    if not verdicts:
        raise SystemFeasibilityExecutionError("frontier design has no sensing cells")
    if all(verdict == "feasible" for verdict in verdicts):
        return "feasible"
    if any(verdict == "infeasible" for verdict in verdicts):
        return "infeasible"
    return "inconclusive"


def _design_cost(row: Mapping[str, object]) -> tuple[int, int]:
    subchannels = row.get("subchannels")
    optical_changed = row.get("optical_profile_changed")
    if (
        not isinstance(subchannels, int)
        or isinstance(subchannels, bool)
        or subchannels < 1
        or type(optical_changed) is not bool
    ):
        raise SystemFeasibilityExecutionError(
            "frontier design has an invalid Pareto cost"
        )
    return subchannels, int(optical_changed)


def _validate_windows(
    declaration: SystemFeasibilityFrontierDeclaration,
    windows: tuple[EvaluationWindow, ...],
) -> None:
    if not windows or any(not isinstance(window, EvaluationWindow) for window in windows):
        raise SystemFeasibilityExecutionError(
            "frontier result requires frozen evaluation windows"
        )
    identities = tuple(
        (window.trace_id, window.start_frame_index, window.frames)
        for window in windows
    )
    if len(set(identities)) != len(identities):
        raise SystemFeasibilityExecutionError("frontier windows are duplicated")
    if Counter(window.density for window in windows) != Counter(
        {
            density: declaration.validation_windows_per_density
            for density in declaration.densities
        }
    ):
        raise SystemFeasibilityExecutionError(
            "frontier windows do not provide the declared samples per density"
        )
    if any(window.frames != declaration.frames_per_window for window in windows):
        raise SystemFeasibilityExecutionError(
            "frontier window length differs from the declaration"
        )


@dataclass(frozen=True, slots=True)
class FrontierCellResult:
    """One complete physical/sensing/fallback evaluation cell."""

    cell: FrontierEvaluationCell
    config_hash: str
    policy_environment_scope_hash: str
    density_rows: tuple[dict[str, object], ...]
    campaign: dict[str, object]
    density_verdicts: tuple[DensityVerdict, ...]
    verdict: CellVerdict

    def __post_init__(self) -> None:
        if not isinstance(self.cell, FrontierEvaluationCell):
            raise SystemFeasibilityExecutionError(
                "frontier cell result requires a declared cell"
            )
        for name, value in (
            ("config_hash", self.config_hash),
            ("policy_environment_scope_hash", self.policy_environment_scope_hash),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise SystemFeasibilityExecutionError(f"{name} must be a SHA-256")
        if (
            not self.density_rows
            or len(self.density_rows) != len(self.density_verdicts)
            or self.verdict != cell_verdict(self.density_verdicts)
        ):
            raise SystemFeasibilityExecutionError(
                "frontier cell density results do not reconcile"
            )

    @property
    def physical_point(self) -> FrontierPhysicalPoint:
        return self.cell.physical_point

    def as_dict(self) -> dict[str, object]:
        point = self.physical_point
        density_rows = []
        for row, verdict in zip(
            self.density_rows,
            self.density_verdicts,
            strict=True,
        ):
            density_rows.append({**row, "verdict": verdict})
        return {
            "cell_id": self.cell.cell_id,
            "physical_point_id": point.point_id,
            "rf_capacity": {
                "name": point.rf_capacity.name,
                "subchannels": point.rf_capacity.subchannels,
                "selection_window_slots": (
                    point.rf_capacity.candidate_resources
                    // point.rf_capacity.subchannels
                ),
                "candidate_resources": point.rf_capacity.candidate_resources,
                "equivalent_system_bandwidth_mhz": (
                    point.rf_capacity.equivalent_system_bandwidth_mhz
                ),
            },
            "sensing_band": {
                "name": point.sensing_band.name,
                "sensing_reliability": point.sensing_band.sensing_reliability,
            },
            "optical_configuration": {
                "name": point.optical_configuration.name,
                "receiver_fov_deg": point.optical_configuration.receiver_fov_deg,
            },
            "fallback_view": {
                "name": self.cell.fallback_view.name,
                "mode": self.cell.fallback_view.mode,
                "diagnostic_only": self.cell.fallback_view.diagnostic_only,
                "authorizes_training": self.cell.fallback_view.authorizes_training,
            },
            "config_hash": self.config_hash,
            "policy_environment_scope_hash": self.policy_environment_scope_hash,
            "densities": density_rows,
            "campaign": self.campaign,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        cell: FrontierEvaluationCell,
        miss_budget: float,
    ) -> FrontierCellResult:
        expected = {
            "cell_id",
            "physical_point_id",
            "rf_capacity",
            "sensing_band",
            "optical_configuration",
            "fallback_view",
            "config_hash",
            "policy_environment_scope_hash",
            "densities",
            "campaign",
            "verdict",
        }
        if set(payload) != expected or payload.get("cell_id") != cell.cell_id:
            raise SystemFeasibilityExecutionError(
                "progress cell identity or fields differ from the declaration"
            )
        raw_rows = payload.get("densities")
        campaign = payload.get("campaign")
        if not isinstance(raw_rows, list) or not isinstance(campaign, Mapping):
            raise SystemFeasibilityExecutionError(
                "progress cell results are malformed"
            )
        density_rows: list[dict[str, object]] = []
        verdicts: list[DensityVerdict] = []
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                raise SystemFeasibilityExecutionError(
                    "progress density row must be an object"
                )
            row = dict(raw)
            stored_verdict = row.pop("verdict", None)
            computed = density_verdict(row, miss_budget=miss_budget)
            if stored_verdict != computed:
                raise SystemFeasibilityExecutionError(
                    "progress density verdict differs from its certificate"
                )
            density_rows.append(row)
            verdicts.append(computed)
        stored_cell_verdict = payload.get("verdict")
        computed_cell_verdict = cell_verdict(tuple(verdicts))
        if stored_cell_verdict != computed_cell_verdict:
            raise SystemFeasibilityExecutionError(
                "progress cell verdict differs from its density verdicts"
            )
        config_digest = payload.get("config_hash")
        scope_digest = payload.get("policy_environment_scope_hash")
        if not isinstance(config_digest, str) or not isinstance(scope_digest, str):
            raise SystemFeasibilityExecutionError(
                "progress cell configuration identity is malformed"
            )
        result = cls(
            cell=cell,
            config_hash=config_digest,
            policy_environment_scope_hash=scope_digest,
            density_rows=tuple(density_rows),
            campaign=dict(campaign),
            density_verdicts=tuple(verdicts),
            verdict=computed_cell_verdict,
        )
        if result.as_dict() != dict(payload):
            raise SystemFeasibilityExecutionError(
                "progress cell physical parameters differ from the declaration"
            )
        return result


CheckpointCallback = Callable[[tuple[FrontierCellResult, ...]], None]


@dataclass(frozen=True, slots=True)
class FrontierDryRunReport:
    """Structural proof that every declared cell can be instantiated."""

    declaration: SystemFeasibilityFrontierDeclaration
    windows: tuple[EvaluationWindow, ...]
    cell_plans: tuple[dict[str, object], ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_windows(self.declaration, self.windows)
        expected = tuple(cell.cell_id for cell in self.declaration.evaluation_cells)
        actual = tuple(str(row.get("cell_id")) for row in self.cell_plans)
        if actual != expected:
            raise SystemFeasibilityExecutionError(
                "dry-run cells differ from the frozen declaration"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SYSTEM_FEASIBILITY_FRONTIER_DRY_RUN_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "window_source": {
                "path": str(self.declaration.window_source),
                "sha256": self.declaration.window_source_sha256,
                "environment_seed": self.declaration.environment_seed,
                "windows": [window.as_dict() for window in self.windows],
            },
            "actor_used": False,
            "checkpoint_used": False,
            "normalization": "fresh frozen zero-statistic identity state",
            "test_split_opened": False,
            "frontier_executed": False,
            "physical_points": len(self.declaration.physical_points),
            "evaluation_cells": len(self.cell_plans),
            "cells": list(self.cell_plans),
        }


@dataclass(frozen=True, slots=True)
class SystemFeasibilityFrontierResult:
    """Complete versioned result and frozen training-authorization decision."""

    declaration: SystemFeasibilityFrontierDeclaration
    windows: tuple[EvaluationWindow, ...]
    cells: tuple[FrontierCellResult, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_windows(self.declaration, self.windows)
        expected = tuple(cell.cell_id for cell in self.declaration.evaluation_cells)
        actual = tuple(result.cell.cell_id for result in self.cells)
        if actual != expected or len(set(actual)) != len(actual):
            raise SystemFeasibilityExecutionError(
                "frontier result cells differ from the complete frozen grid"
            )
        expected_densities = self.declaration.densities
        for result in self.cells:
            actual_densities = tuple(
                _number(row, "density_vehicles_per_lane_km")
                for row in result.density_rows
            )
            if actual_densities != expected_densities:
                raise SystemFeasibilityExecutionError(
                    "frontier result density coverage differs from the declaration",
                    context={"cell_id": result.cell.cell_id},
                )
            expected_verdicts = tuple(
                density_verdict(row, miss_budget=self.declaration.miss_budget)
                for row in result.density_rows
            )
            if result.density_verdicts != expected_verdicts:
                raise SystemFeasibilityExecutionError(
                    "frontier density verdict differs from its certificate",
                    context={"cell_id": result.cell.cell_id},
                )

    def _design_rows(self) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for capacity in self.declaration.rf_capacities:
            for optical in self.declaration.optical_configurations:
                contract_cells = tuple(
                    result
                    for result in self.cells
                    if result.physical_point.rf_capacity.name == capacity.name
                    and result.physical_point.optical_configuration.name == optical.name
                    and result.cell.fallback_view.mode == "contract"
                )
                if len(contract_cells) != len(self.declaration.sensing_bands):
                    raise SystemFeasibilityExecutionError(
                        "robust design does not contain every sensing band"
                    )
                by_band = {
                    result.physical_point.sensing_band.name: result.verdict
                    for result in contract_cells
                }
                rows.append(
                    {
                        "design_id": f"{capacity.name}__{optical.name}",
                        "rf_capacity_name": capacity.name,
                        "subchannels": capacity.subchannels,
                        "optical_configuration_name": optical.name,
                        "optical_profile_changed": not optical.headline,
                        "sensing_band_verdicts": by_band,
                        "verdict": _design_verdict(
                            tuple(by_band[band.name] for band in self.declaration.sensing_bands)
                        ),
                    }
                )
        return tuple(rows)

    def _pareto_minimal_design_ids(
        self,
        designs: tuple[dict[str, object], ...],
    ) -> tuple[str, ...]:
        feasible = tuple(row for row in designs if row["verdict"] == "feasible")
        minimal: list[str] = []
        for candidate in feasible:
            candidate_cost = _design_cost(candidate)
            dominated = False
            for other in feasible:
                if other is candidate:
                    continue
                other_cost = _design_cost(other)
                if (
                    other_cost[0] <= candidate_cost[0]
                    and other_cost[1] <= candidate_cost[1]
                    and other_cost != candidate_cost
                ):
                    dominated = True
                    break
            if not dominated:
                minimal.append(str(candidate["design_id"]))
        return tuple(minimal)

    def decision(self) -> dict[str, object]:
        designs = self._design_rows()
        headline = self.declaration.headline_point
        current_nominal = next(
            result
            for result in self.cells
            if result.physical_point.point_id == headline.point_id
            and result.cell.fallback_view.mode == "contract"
        )
        current_design_id = (
            f"{headline.rf_capacity.name}__{headline.optical_configuration.name}"
        )
        current_design = next(
            row for row in designs if row["design_id"] == current_design_id
        )
        feasible_designs = tuple(
            str(row["design_id"])
            for row in designs
            if row["verdict"] == "feasible"
        )
        pareto = self._pareto_minimal_design_ids(designs)
        return {
            "current_system_nominal_cell_id": current_nominal.cell.cell_id,
            "current_system_nominal_verdict": current_nominal.verdict,
            "current_system_robust_verdict": current_design["verdict"],
            "training_authorized": bool(feasible_designs),
            "robust_feasible_design_ids": list(feasible_designs),
            "pareto_minimal_robust_feasible_design_ids": list(pareto),
            "designs": list(designs),
            "fallback_diagnostic_authorized_training": False,
            "test_split_opened": False,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SYSTEM_FEASIBILITY_FRONTIER_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "scope": (
                "bounded certificate-aware pair-local system-feasibility "
                "screening on frozen validation windows"
            ),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "window_source": {
                "path": str(self.declaration.window_source),
                "sha256": self.declaration.window_source_sha256,
                "environment_seed": self.declaration.environment_seed,
                "windows": [window.as_dict() for window in self.windows],
            },
            "reliability_miss_budget": self.declaration.miss_budget,
            "densities_vehicles_per_lane_km": list(self.declaration.densities),
            "optimization_method": JOINT_RISK_ORACLE_METHOD,
            "exact_assignment_cap": self.declaration.exact_assignment_cap,
            "max_search_iterations": self.declaration.max_search_iterations,
            "actor_used": False,
            "checkpoint_used": False,
            "normalization": "fresh frozen zero-statistic identity state",
            "test_split_opened": False,
            "frontier_complete": True,
            "cells": [cell.as_dict() for cell in self.cells],
            "decision": self.decision(),
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True)
            + "\n"
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


def _load_frozen_windows(
    declaration: SystemFeasibilityFrontierDeclaration,
) -> tuple[EvaluationWindow, ...]:
    _, windows, environment_seed, artifact_sha256 = load_frozen_regime_audit(
        declaration.window_source,
        expected_policy_environment_scope_hash=(
            declaration.baseline_policy_environment_scope_hash
        ),
    )
    if (
        environment_seed != declaration.environment_seed
        or artifact_sha256 != declaration.window_source_sha256
    ):
        raise SystemFeasibilityExecutionError(
            "frontier window provenance differs from the declaration"
        )
    _validate_windows(declaration, windows)
    return windows


def _config_for_point(
    declaration: SystemFeasibilityFrontierDeclaration,
    point: FrontierPhysicalPoint,
    *,
    project_root: Path,
) -> ProjectConfig:
    return load_config(
        (*declaration.base_config_layers, *point.optical_configuration.additional_config_layers),
        project_root=project_root,
    )


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if (
        not state.frozen
        or any(state.count)
        or any(state.mean)
        or any(state.second_moment)
    ):
        raise SystemFeasibilityExecutionError(
            "frontier normalization is not a fresh frozen identity state"
        )
    return state


def _validate_trace_catalog(
    config: ProjectConfig,
    windows: tuple[EvaluationWindow, ...],
) -> None:
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {source.trace_id: source for source in catalog.for_split("validation")}
    for window in windows:
        try:
            source = validation[window.trace_id]
        except KeyError as error:
            raise SystemFeasibilityExecutionError(
                "frozen frontier window is absent from the trace catalog"
            ) from error
        if source.density != window.density:
            raise SystemFeasibilityExecutionError(
                "frozen frontier window density differs from the trace catalog"
            )
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
            expected_config_scope_hashes={
                "mobility_trace": scope_hash(config, "mobility_trace")
            },
        )
        if window.start_frame_index + window.frames > reader.decision_frame_count:
            raise SystemFeasibilityExecutionError(
                "frozen frontier window exceeds its trace"
            )


def structural_dry_run(
    declaration: SystemFeasibilityFrontierDeclaration,
    *,
    project_root: str | Path,
) -> FrontierDryRunReport:
    """Instantiate every dependency without evaluating a physical frame."""

    if not isinstance(declaration, SystemFeasibilityFrontierDeclaration):
        raise SystemFeasibilityExecutionError(
            "frontier dry run requires a validated declaration"
        )
    root = Path(project_root).expanduser().resolve(strict=False)
    windows = _load_frozen_windows(declaration)
    cell_plans: list[dict[str, object]] = []
    validated_configs: set[str] = set()
    for point in declaration.physical_points:
        config = _config_for_point(declaration, point, project_root=root)
        digest = config_hash(config)
        local_model = LocalRFPhysicsModel.from_config(
            config,
            sensitivity_band=point.sensing_band.band,
            collision_subchannels=point.rf_capacity.subchannels,
        )
        physical = build_rollout(
            config,
            buildings=local_model.buildings,
            root_seed=declaration.environment_seed,
            band=point.sensing_band.band,
            collision_subchannels=point.rf_capacity.subchannels,
        )
        parameters = local_model.response_model.parameters
        if (
            parameters.subchannels != point.rf_capacity.subchannels
            or parameters.selection_window_slots
            != declaration.selection_window_slots
            or not math.isclose(
                parameters.sensing_reliability,
                point.sensing_band.sensing_reliability,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or physical.lifecycle.rf.collision != parameters
        ):
            raise SystemFeasibilityExecutionError(
                "effective frontier RF parameters differ from the declaration",
                context={"physical_point_id": point.point_id},
            )
        action_space = MaskedActionSpace.from_config(
            config.environment,
            config.rf,
            config.vlc,
        )
        if action_space.fallback_action.label != "DUP-4":
            raise SystemFeasibilityExecutionError(
                "frontier contract fallback is not DUP-4"
            )
        _identity_normalization(config)
        if digest not in validated_configs:
            _validate_trace_catalog(config, windows)
            validated_configs.add(digest)
        for fallback in declaration.fallback_views:
            cell = FrontierEvaluationCell(point, fallback)
            cell_plans.append(
                {
                    "cell_id": cell.cell_id,
                    "physical_point_id": point.point_id,
                    "config_hash": digest,
                    "policy_environment_scope_hash": scope_hash(
                        config,
                        "policy_environment",
                    ),
                    "rf_subchannels": parameters.subchannels,
                    "candidate_resources": parameters.candidate_resources,
                    "sensing_band": point.sensing_band.name,
                    "sensing_reliability": parameters.sensing_reliability,
                    "receiver_fov_deg": config.vlc.receiver_fov_deg,
                    "fallback_mode": fallback.mode,
                    "diagnostic_only": fallback.diagnostic_only,
                    "authorizes_training": fallback.authorizes_training,
                }
            )
    return FrontierDryRunReport(
        declaration=declaration,
        windows=windows,
        cell_plans=tuple(cell_plans),
        generated_at_utc=datetime.now(UTC),
    )


def _cell_result(
    cell: FrontierEvaluationCell,
    config: ProjectConfig,
    evaluation: JointOracleWindowEvaluation,
    *,
    miss_budget: float,
) -> FrontierCellResult:
    verdicts = tuple(
        density_verdict(row, miss_budget=miss_budget)
        for row in evaluation.densities
    )
    return FrontierCellResult(
        cell=cell,
        config_hash=config_hash(config),
        policy_environment_scope_hash=scope_hash(config, "policy_environment"),
        density_rows=evaluation.densities,
        campaign=evaluation.campaign,
        density_verdicts=verdicts,
        verdict=cell_verdict(verdicts),
    )


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_frontier_progress(
    path: str | Path,
    *,
    declaration: SystemFeasibilityFrontierDeclaration,
    cells: tuple[FrontierCellResult, ...],
) -> Path:
    """Atomically persist a validated ordered prefix of completed cells."""

    expected = tuple(cell.cell_id for cell in declaration.evaluation_cells)
    actual = tuple(result.cell.cell_id for result in cells)
    if actual != expected[: len(actual)]:
        raise SystemFeasibilityExecutionError(
            "progress cells must be an ordered prefix of the frozen grid"
        )
    return _atomic_json_write(
        Path(path),
        {
            "schema": SYSTEM_FEASIBILITY_FRONTIER_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "window_source_sha256": declaration.window_source_sha256,
            "test_split_opened": False,
            "completed_cells": [cell.as_dict() for cell in cells],
        },
    )


def load_frontier_progress(
    path: str | Path,
    *,
    declaration: SystemFeasibilityFrontierDeclaration,
) -> tuple[FrontierCellResult, ...]:
    """Load a resumable prefix only when every frozen identity still matches."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemFeasibilityExecutionError(
            "frontier progress artifact is unreadable",
            artifact_path=source,
        ) from error
    expected_fields = {
        "schema",
        "declaration_sha256",
        "window_source_sha256",
        "test_split_opened",
        "completed_cells",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise SystemFeasibilityExecutionError(
            "frontier progress fields do not match the schema"
        )
    if (
        payload["schema"] != SYSTEM_FEASIBILITY_FRONTIER_PROGRESS_SCHEMA
        or payload["declaration_sha256"] != declaration.sha256
        or payload["window_source_sha256"] != declaration.window_source_sha256
        or payload["test_split_opened"] is not False
    ):
        raise SystemFeasibilityExecutionError(
            "frontier progress provenance differs from the frozen declaration"
        )
    raw_cells = payload["completed_cells"]
    if not isinstance(raw_cells, list):
        raise SystemFeasibilityExecutionError(
            "frontier progress completed_cells must be an array"
        )
    declared = declaration.evaluation_cells
    if len(raw_cells) > len(declared):
        raise SystemFeasibilityExecutionError(
            "frontier progress contains more cells than the declaration"
        )
    results: list[FrontierCellResult] = []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping):
            raise SystemFeasibilityExecutionError(
                "frontier progress cell must be an object"
            )
        results.append(
            FrontierCellResult.from_dict(
                raw,
                cell=declared[index],
                miss_budget=declaration.miss_budget,
            )
        )
    return tuple(results)


def execute_system_feasibility_frontier(
    declaration: SystemFeasibilityFrontierDeclaration,
    *,
    project_root: str | Path,
    progress: ProgressCallback | None = None,
    completed_cells: tuple[FrontierCellResult, ...] = (),
    checkpoint: CheckpointCallback | None = None,
    receive_diversity: RFReceiveDiversity | None = None,
) -> SystemFeasibilityFrontierResult:
    """Execute all predeclared cells; partial grids never produce a result."""

    dry_run = structural_dry_run(declaration, project_root=project_root)
    if receive_diversity is not None and not isinstance(
        receive_diversity,
        RFReceiveDiversity,
    ):
        raise SystemFeasibilityExecutionError(
            "receive_diversity must be an RFReceiveDiversity profile or None"
        )
    root = Path(project_root).expanduser().resolve(strict=False)
    total = len(declaration.evaluation_cells)
    expected_prefix = declaration.evaluation_cells[: len(completed_cells)]
    if tuple(result.cell for result in completed_cells) != expected_prefix:
        raise SystemFeasibilityExecutionError(
            "completed cells must be an ordered prefix of the frozen grid"
        )
    results = list(completed_cells)
    index = 0
    for point in declaration.physical_points:
        config = _config_for_point(declaration, point, project_root=root)
        normalization = _identity_normalization(config)
        for fallback in declaration.fallback_views:
            cell = FrontierEvaluationCell(point, fallback)
            index += 1
            if index <= len(completed_cells):
                restored = completed_cells[index - 1]
                if (
                    restored.config_hash != config_hash(config)
                    or restored.policy_environment_scope_hash
                    != scope_hash(config, "policy_environment")
                ):
                    raise SystemFeasibilityExecutionError(
                        "completed cell configuration has drifted",
                        context={"cell_id": cell.cell_id},
                    )
                continue
            if progress is not None:
                progress(index, total, cell.cell_id)
            evaluation = evaluate_pair_local_joint_windows(
                config,
                windows=dry_run.windows,
                environment_seed=declaration.environment_seed,
                normalization_state=normalization,
                sensitivity_band=point.sensing_band.band,
                collision_subchannels=point.rf_capacity.subchannels,
                oracle_controls_unusable_rows=fallback.mode == "all_usable",
                exact_assignment_cap=declaration.exact_assignment_cap,
                max_search_iterations=declaration.max_search_iterations,
                receive_diversity=receive_diversity,
            )
            results.append(
                _cell_result(
                    cell,
                    config,
                    evaluation,
                    miss_budget=declaration.miss_budget,
                )
            )
            if checkpoint is not None:
                checkpoint(tuple(results))
    return SystemFeasibilityFrontierResult(
        declaration=declaration,
        windows=dry_run.windows,
        cells=tuple(results),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "SYSTEM_FEASIBILITY_FRONTIER_DRY_RUN_SCHEMA",
    "SYSTEM_FEASIBILITY_FRONTIER_PROGRESS_SCHEMA",
    "SYSTEM_FEASIBILITY_FRONTIER_RESULT_SCHEMA",
    "CellVerdict",
    "DensityVerdict",
    "FrontierCellResult",
    "FrontierDryRunReport",
    "SystemFeasibilityExecutionError",
    "SystemFeasibilityFrontierResult",
    "cell_verdict",
    "density_verdict",
    "execute_system_feasibility_frontier",
    "load_frontier_progress",
    "structural_dry_run",
    "write_frontier_progress",
]
