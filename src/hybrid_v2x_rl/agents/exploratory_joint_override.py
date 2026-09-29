"""Exploratory joint characterization for the user-authorized best pair."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from hybrid_v2x_rl.agents.combined_receiver_block_frontier import (
    COMBINED_FRONTIER_RESULT_SCHEMA,
    CombinedReceiveProfile,
    CombinedReceiverBlockDeclaration,
    load_combined_receiver_block_declaration,
)
from hybrid_v2x_rl.agents.joint_oracle_evaluation import (
    evaluate_pair_local_joint_windows,
)
from hybrid_v2x_rl.agents.longer_block_frontier import config_for_candidate
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    FrontierCellResult,
    cell_verdict,
    density_verdict,
)
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    FrontierEvaluationCell,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.frames import PopulationFrameReader, TraceCatalog
from hybrid_v2x_rl.mean_field.local_rf_pipeline import LocalRFPhysicsModel
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationState,
    ObservationNormalizer,
)

EXPLORATORY_JOINT_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.exploratory-joint-override-declaration.v1"
)
EXPLORATORY_JOINT_RESULT_SCHEMA: Final = "hybrid-rf-vlc-rl.exploratory-joint-override-result.v1"
EXPLORATORY_JOINT_PROGRESS_SCHEMA: Final = "hybrid-rf-vlc-rl.exploratory-joint-override-progress.v1"
ProgressCallback = Callable[[int, int, str], None]
CheckpointCallback = Callable[[tuple[FrontierCellResult, ...]], None]


class ExploratoryJointOverrideError(HybridV2XError):
    """The override declaration, execution, or result has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ExploratoryJointOverrideError(f"{name} fields do not match the frozen schema")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ExploratoryJointOverrideError(f"{name} must be a nonempty array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExploratoryJointOverrideError(f"{name} must be nonempty text")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ExploratoryJointOverrideError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExploratoryJointOverrideError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExploratoryJointOverrideError(f"{name} must be a positive integer")
    return value


def _false(value: object, *, name: str) -> None:
    if value is not False:
        raise ExploratoryJointOverrideError(f"{name} must be false")


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ExploratoryJointOverrideError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name)).expanduser()
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


def _json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExploratoryJointOverrideError(f"{name} is unreadable", artifact_path=path) from error
    if not isinstance(payload, Mapping):
        raise ExploratoryJointOverrideError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], payload)


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    path: Path
    sha256: str
    schema: str | None = None


@dataclass(frozen=True, slots=True)
class ExploratoryJointOverrideDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    payload_bytes: int
    deadline_s: float
    miss_budget: float
    combined_declaration_artifact: EvidenceArtifact
    combined_result_artifact: EvidenceArtifact
    source_frontier_artifact: EvidenceArtifact
    combined_declaration: CombinedReceiverBlockDeclaration
    selected_profile: CombinedReceiveProfile
    selected_optical_name: str
    propagation_mean: float
    propagation_budget_multiple: float
    selected_capacity_name: str
    selected_sensing_band_names: tuple[str, ...]
    selected_fallback_name: str
    densities: tuple[float, ...]
    expected_windows: int
    expected_cells: int
    exact_assignment_cap: int
    max_search_iterations: int
    selected_training_sensing_band: str
    output_path: Path

    @property
    def selected_cells(self) -> tuple[FrontierEvaluationCell, ...]:
        source = self.combined_declaration.longer_declaration.receive_declaration.source_frontier
        return tuple(
            cell
            for cell in source.evaluation_cells
            if cell.physical_point.rf_capacity.name == self.selected_capacity_name
            and cell.physical_point.optical_configuration.name == self.selected_optical_name
            and cell.physical_point.sensing_band.name in self.selected_sensing_band_names
            and cell.fallback_view.name == self.selected_fallback_name
        )


def _artifact(
    value: object,
    *,
    root: Path,
    name: str,
    expected_schema: str | None = None,
) -> EvidenceArtifact:
    keys = {"path", "sha256", "schema"} if expected_schema else {"path", "sha256"}
    row = _mapping(value, name=name, keys=keys)
    schema = _text(row["schema"], name=f"{name} schema") if expected_schema else None
    if schema != expected_schema:
        raise ExploratoryJointOverrideError(f"{name} schema is unsupported")
    return EvidenceArtifact(
        path=_resolve(root, row["path"], name=f"{name} path"),
        sha256=_digest(row["sha256"], name=f"{name} SHA-256"),
        schema=schema,
    )


def load_exploratory_joint_override_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> ExploratoryJointOverrideDeclaration:
    """Load and fail-closed validate the explicit exploratory override."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="override declaration")
    top = _mapping(
        load_yaml_file(declaration_path),
        name="override declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "selected_pair",
            "joint_characterization",
            "execution",
            "decision",
        },
    )
    if top["schema"] != EXPLORATORY_JOINT_DECLARATION_SCHEMA:
        raise ExploratoryJointOverrideError("override declaration schema is unsupported")
    objective = _mapping(
        top["objective"],
        name="objective",
        keys={
            "payload_bytes",
            "deadline_s",
            "exact_miss_budget",
            "required_split",
            "interpretation",
        },
    )
    payload_bytes = _integer(objective["payload_bytes"], name="payload bytes")
    deadline_s = _number(objective["deadline_s"], name="deadline")
    miss_budget = _number(objective["exact_miss_budget"], name="miss budget")
    if (
        payload_bytes != 300
        or not math.isclose(deadline_s, 0.010, abs_tol=1e-15)
        or not math.isclose(miss_budget, 1e-4, abs_tol=1e-15)
        or objective["required_split"] != "validation"
    ):
        raise ExploratoryJointOverrideError("override objective has drifted")

    evidence = _mapping(
        top["evidence"],
        name="evidence",
        keys={
            "combined_declaration",
            "combined_result",
            "source_joint_frontier",
            "user_directed_exploratory_override",
            "actor_used",
            "checkpoint_used",
        },
    )
    if evidence["user_directed_exploratory_override"] is not True:
        raise ExploratoryJointOverrideError("user-directed override must be explicit")
    _false(evidence["actor_used"], name="actor_used")
    _false(evidence["checkpoint_used"], name="checkpoint_used")
    combined_declaration_artifact = _artifact(
        evidence["combined_declaration"], root=root, name="combined declaration"
    )
    combined_result_artifact = _artifact(
        evidence["combined_result"],
        root=root,
        name="combined result",
        expected_schema=COMBINED_FRONTIER_RESULT_SCHEMA,
    )
    source_artifact = _artifact(
        evidence["source_joint_frontier"], root=root, name="source joint frontier"
    )
    if verify_evidence:
        for artifact, name in (
            (combined_declaration_artifact, "combined declaration"),
            (combined_result_artifact, "combined result"),
            (source_artifact, "source joint frontier"),
        ):
            if not artifact.path.is_file() or _sha256(artifact.path) != artifact.sha256:
                raise ExploratoryJointOverrideError(
                    f"{name} evidence is absent or has drifted", artifact_path=artifact.path
                )
        combined_payload = _json_mapping(combined_result_artifact.path, name="combined result")
        if (
            combined_payload.get("schema") != combined_result_artifact.schema
            or combined_payload.get("propagation_screen_complete") is not True
            or combined_payload.get("training_run_performed") is not False
            or combined_payload.get("test_split_opened") is not False
        ):
            raise ExploratoryJointOverrideError("combined result safety boundary has drifted")

    combined = load_combined_receiver_block_declaration(
        combined_declaration_artifact.path,
        project_root=root,
        verify_evidence=verify_evidence,
    )
    if combined.sha256 != combined_declaration_artifact.sha256:
        raise ExploratoryJointOverrideError("combined declaration hash has drifted")
    source = combined.longer_declaration.receive_declaration.source_frontier
    if source.sha256 != source_artifact.sha256:
        raise ExploratoryJointOverrideError("source joint frontier hash has drifted")

    selected = _mapping(
        top["selected_pair"],
        name="selected pair",
        keys={
            "rf_candidate",
            "receive_profile",
            "receive_profile_role",
            "optical_configuration",
            "propagation_only_worst_density_mean",
            "propagation_only_exact_budget_multiple",
            "exact_target_met",
            "selection_basis",
        },
    )
    profile_name = _text(selected["receive_profile"], name="receive profile")
    profiles = tuple(
        profile for profile in combined.receive_profiles if profile.profile.name == profile_name
    )
    propagation_mean = _number(
        selected["propagation_only_worst_density_mean"], name="propagation mean"
    )
    propagation_multiple = _number(
        selected["propagation_only_exact_budget_multiple"], name="propagation multiple"
    )
    if (
        selected["rf_candidate"] != combined.candidate.name
        or len(profiles) != 1
        or profile_name != "rx2-mrc__independent-ideal__integrated-zero-loss"
        or selected["receive_profile_role"] != profiles[0].role
        or selected["optical_configuration"] != "wide-60deg"
        or not math.isclose(propagation_mean, 0.0001182919226271053, abs_tol=1e-18)
        or not math.isclose(propagation_multiple, propagation_mean / miss_budget, abs_tol=1e-12)
        or selected["exact_target_met"] is not False
    ):
        raise ExploratoryJointOverrideError("selected exploratory pair has drifted")

    joint = _mapping(
        top["joint_characterization"],
        name="joint characterization",
        keys={
            "rf_capacity",
            "sensing_bands",
            "fallback_view",
            "densities_vehicles_per_lane_km",
            "validation_windows",
            "expected_cells",
            "exact_assignment_cap",
            "max_search_iterations",
            "capacity_selection_reason",
        },
    )
    capacity_name = _text(joint["rf_capacity"], name="RF capacity")
    sensing_names = tuple(
        _text(raw, name="sensing band")
        for raw in _sequence(joint["sensing_bands"], name="sensing bands")
    )
    fallback_name = _text(joint["fallback_view"], name="fallback view")
    densities = tuple(
        _number(raw, name="density")
        for raw in _sequence(joint["densities_vehicles_per_lane_km"], name="densities")
    )
    expected_windows = _integer(joint["validation_windows"], name="validation windows")
    expected_cells = _integer(joint["expected_cells"], name="expected cells")
    exact_cap = _integer(joint["exact_assignment_cap"], name="exact assignment cap")
    max_iterations = _integer(joint["max_search_iterations"], name="max search iterations")
    if (
        capacity_name != "rf-capacity-4x"
        or sensing_names != ("nominal", "pessimistic", "optimistic")
        or fallback_name != "contract-dup4"
        or densities != (10.0, 20.0, 30.0)
        or expected_windows != 9
        or expected_cells != 3
        or exact_cap != source.exact_assignment_cap
        or max_iterations != source.max_search_iterations
    ):
        raise ExploratoryJointOverrideError("joint characterization grid has drifted")

    execution = _mapping(
        top["execution"],
        name="execution",
        keys={
            "no_adaptive_axis_expansion",
            "no_training_during_characterization",
            "no_test_split",
            "output_path",
        },
    )
    if (
        execution["no_adaptive_axis_expansion"] is not True
        or execution["no_training_during_characterization"] is not True
        or execution["no_test_split"] is not True
    ):
        raise ExploratoryJointOverrideError("override execution boundary has drifted")
    decision = _mapping(
        top["decision"],
        name="decision",
        keys={
            "exact_feasibility_claim_allowed",
            "exploratory_training_after_joint_characterization",
            "selected_training_sensing_band",
            "claim_boundary",
        },
    )
    _false(decision["exact_feasibility_claim_allowed"], name="exact feasibility claim")
    if decision["exploratory_training_after_joint_characterization"] is not True:
        raise ExploratoryJointOverrideError("exploratory training authorization has drifted")
    training_band = _text(decision["selected_training_sensing_band"], name="training band")
    if training_band != "nominal":
        raise ExploratoryJointOverrideError("selected training sensing band has drifted")

    declaration = ExploratoryJointOverrideDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        payload_bytes=payload_bytes,
        deadline_s=deadline_s,
        miss_budget=miss_budget,
        combined_declaration_artifact=combined_declaration_artifact,
        combined_result_artifact=combined_result_artifact,
        source_frontier_artifact=source_artifact,
        combined_declaration=combined,
        selected_profile=profiles[0],
        selected_optical_name="wide-60deg",
        propagation_mean=propagation_mean,
        propagation_budget_multiple=propagation_multiple,
        selected_capacity_name=capacity_name,
        selected_sensing_band_names=sensing_names,
        selected_fallback_name=fallback_name,
        densities=densities,
        expected_windows=expected_windows,
        expected_cells=expected_cells,
        exact_assignment_cap=exact_cap,
        max_search_iterations=max_iterations,
        selected_training_sensing_band=training_band,
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )
    if (
        tuple(cell.physical_point.sensing_band.name for cell in declaration.selected_cells)
        != sensing_names
    ):
        raise ExploratoryJointOverrideError("selected source cells have drifted")
    return declaration


def _selected_config(
    declaration: ExploratoryJointOverrideDeclaration,
    *,
    project_root: Path,
) -> ProjectConfig:
    optical = tuple(
        optical
        for optical in declaration.combined_declaration.longer_declaration.optical_configurations
        if optical.name == declaration.selected_optical_name
    )
    if len(optical) != 1:
        raise ExploratoryJointOverrideError("selected optical configuration is absent")
    return config_for_candidate(
        declaration.combined_declaration.longer_declaration,
        declaration.combined_declaration.candidate,
        optical[0],
        project_root=project_root,
    )


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if not state.frozen or any(state.count) or any(state.mean) or any(state.second_moment):
        raise ExploratoryJointOverrideError("override requires identity normalization")
    return state


def _validate_traces(
    config: ProjectConfig,
    windows: tuple[EvaluationWindow, ...],
) -> None:
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {trace.trace_id: trace for trace in catalog.for_split("validation")}
    for window in windows:
        try:
            trace = validation[window.trace_id]
        except KeyError as error:
            raise ExploratoryJointOverrideError("override validation trace is absent") from error
        reader = PopulationFrameReader(
            trace,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
            expected_config_scope_hashes={"mobility_trace": scope_hash(config, "mobility_trace")},
        )
        if window.start_frame_index + window.frames > reader.decision_frame_count:
            raise ExploratoryJointOverrideError("override window exceeds its trace")


def structural_exploratory_joint_dry_run(
    declaration: ExploratoryJointOverrideDeclaration,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Validate selected cells, custom physics, and traces without evaluation."""

    root = Path(project_root).expanduser().resolve(strict=False)
    source = declaration.combined_declaration.longer_declaration.receive_declaration.source_frontier
    source_report = structural_system_dry_run(source, project_root=root)
    if len(source_report.windows) != declaration.expected_windows:
        raise ExploratoryJointOverrideError("override window count has drifted")
    config = _selected_config(declaration, project_root=root)
    if (
        config.rf.modulation != "qpsk"
        or not math.isclose(config.rf.timing.airtime_s, 0.002, abs_tol=1e-15)
        or config.rf.slots_per_transmission != 4
        or config.service.payload_bytes != declaration.payload_bytes
        or not math.isclose(config.service.deadline_s, declaration.deadline_s, abs_tol=1e-15)
    ):
        raise ExploratoryJointOverrideError("override RF/service configuration has drifted")
    action_space = MaskedActionSpace.from_config(config.environment, config.rf, config.vlc)
    if action_space.fallback_action.label != "DUP-4":
        raise ExploratoryJointOverrideError("override fallback action has drifted")
    _identity_normalization(config)
    _validate_traces(config, source_report.windows)
    cell_plans: list[dict[str, object]] = []
    for cell in declaration.selected_cells:
        point = cell.physical_point
        local_model = LocalRFPhysicsModel.from_config(
            config,
            sensitivity_band=point.sensing_band.band,
            collision_subchannels=point.rf_capacity.subchannels,
        )
        rollout = build_rollout(
            config,
            buildings=local_model.buildings,
            root_seed=source.environment_seed,
            band=point.sensing_band.band,
            collision_subchannels=point.rf_capacity.subchannels,
            receive_diversity=declaration.selected_profile.profile.physical_profile(),
        )
        parameters = local_model.response_model.parameters
        if (
            parameters.subchannels != point.rf_capacity.subchannels
            or parameters.candidate_resources != point.rf_capacity.candidate_resources
            or rollout.lifecycle.rf.collision != parameters
            or rollout.lifecycle.rf.receive_diversity
            != declaration.selected_profile.profile.physical_profile()
        ):
            raise ExploratoryJointOverrideError("override effective RF parameters have drifted")
        cell_plans.append(
            {
                "cell_id": cell.cell_id,
                "rf_capacity_name": point.rf_capacity.name,
                "subchannels": point.rf_capacity.subchannels,
                "candidate_resources": point.rf_capacity.candidate_resources,
                "equivalent_system_bandwidth_mhz": point.rf_capacity.equivalent_system_bandwidth_mhz,
                "sensing_band": point.sensing_band.name,
                "sensing_reliability": point.sensing_band.sensing_reliability,
                "fallback_view": cell.fallback_view.name,
            }
        )
    if len(cell_plans) != declaration.expected_cells:
        raise ExploratoryJointOverrideError("override cell count has drifted")
    return {
        "schema": EXPLORATORY_JOINT_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "selected_rf_candidate": declaration.combined_declaration.candidate.name,
        "selected_receive_profile": declaration.selected_profile.profile.name,
        "selected_optical_configuration": declaration.selected_optical_name,
        "propagation_only_worst_density_mean": declaration.propagation_mean,
        "propagation_only_exact_budget_multiple": declaration.propagation_budget_multiple,
        "validation_windows": len(source_report.windows),
        "cells": cell_plans,
        "config_hash": config_hash(config),
        "channel_frames_evaluated": 0,
        "training_performed": False,
        "test_split_opened": False,
    }


def _cell_result(
    declaration: ExploratoryJointOverrideDeclaration,
    cell: FrontierEvaluationCell,
    config: ProjectConfig,
    densities: tuple[dict[str, object], ...],
    campaign: dict[str, object],
) -> FrontierCellResult:
    verdicts = tuple(density_verdict(row, miss_budget=declaration.miss_budget) for row in densities)
    return FrontierCellResult(
        cell=cell,
        config_hash=config_hash(config),
        policy_environment_scope_hash=scope_hash(config, "policy_environment"),
        density_rows=densities,
        campaign=campaign,
        density_verdicts=verdicts,
        verdict=cell_verdict(verdicts),
    )


def _validate_cells(
    declaration: ExploratoryJointOverrideDeclaration,
    cells: tuple[FrontierCellResult, ...],
    *,
    require_complete: bool,
) -> None:
    expected = declaration.selected_cells
    if tuple(result.cell for result in cells) != expected[: len(cells)]:
        raise ExploratoryJointOverrideError("override results are not an ordered cell prefix")
    if require_complete and len(cells) != len(expected):
        raise ExploratoryJointOverrideError("override joint characterization is incomplete")
    for result in cells:
        densities = tuple(
            float(cast(float, row["density_vehicles_per_lane_km"])) for row in result.density_rows
        )
        if densities != declaration.densities:
            raise ExploratoryJointOverrideError("override result densities have drifted")


@dataclass(frozen=True, slots=True)
class ExploratoryJointOverrideResult:
    declaration: ExploratoryJointOverrideDeclaration
    cells: tuple[FrontierCellResult, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_cells(self.declaration, self.cells, require_complete=True)

    def decision(self) -> dict[str, object]:
        def worst(result: FrontierCellResult) -> float:
            return max(
                float(
                    cast(
                        float,
                        row["mean_pair_local_candidate_conditional_miss_risk"],
                    )
                )
                for row in result.density_rows
            )

        best = min(
            self.cells,
            key=lambda result: (
                worst(result),
                self.declaration.selected_cells.index(result.cell),
            ),
        )
        nominal = next(
            result
            for result in self.cells
            if result.physical_point.sensing_band.name
            == self.declaration.selected_training_sensing_band
        )
        exact_met = any(result.verdict == "feasible" for result in self.cells)
        return {
            "exact_system_target_met": exact_met,
            "exact_feasibility_claim_allowed": False,
            "best_joint_cell_id": best.cell.cell_id,
            "best_joint_worst_density_mean": worst(best),
            "best_joint_exact_budget_multiple": worst(best) / self.declaration.miss_budget,
            "nominal_training_cell_id": nominal.cell.cell_id,
            "nominal_training_worst_density_mean": worst(nominal),
            "nominal_training_exact_budget_multiple": (
                worst(nominal) / self.declaration.miss_budget
            ),
            "exploratory_training_authorized": True,
            "training_authorization_basis": "explicit user-directed exploratory override after complete joint characterization",
            "training_performed": False,
            "test_split_opened": False,
            "next_action": "freeze the selected exploratory physical/training configuration, then run PPO without claiming exact 1e-4 feasibility",
            "claim_boundary": (
                "the selected joint configuration is exploratory and may violate 1e-4; "
                "training studies policy behavior rather than proving reliability feasibility"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": EXPLORATORY_JOINT_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {"path": str(self.declaration.path), "sha256": self.declaration.sha256},
            "sources": {
                "combined_declaration_sha256": self.declaration.combined_declaration_artifact.sha256,
                "combined_result_sha256": self.declaration.combined_result_artifact.sha256,
                "source_joint_frontier_sha256": self.declaration.source_frontier_artifact.sha256,
            },
            "exact_miss_budget": self.declaration.miss_budget,
            "user_directed_exploratory_override": True,
            "selected_pair": {
                "rf_candidate": self.declaration.combined_declaration.candidate.name,
                "receive_profile": self.declaration.selected_profile.as_dict(),
                "optical_configuration": self.declaration.selected_optical_name,
                "propagation_only_worst_density_mean": self.declaration.propagation_mean,
                "propagation_only_exact_budget_multiple": self.declaration.propagation_budget_multiple,
            },
            "actor_used": False,
            "checkpoint_used": False,
            "training_run_performed": False,
            "test_split_opened": False,
            "joint_characterization_complete": True,
            "cells": [cell.as_dict() for cell in self.cells],
            "decision": self.decision(),
        }

    def write_json(self, path: str | Path) -> Path:
        return _atomic_json_write(Path(path), self.as_dict())


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_exploratory_joint_progress(
    path: str | Path,
    *,
    declaration: ExploratoryJointOverrideDeclaration,
    cells: tuple[FrontierCellResult, ...],
) -> Path:
    _validate_cells(declaration, cells, require_complete=False)
    return _atomic_json_write(
        Path(path),
        {
            "schema": EXPLORATORY_JOINT_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "combined_result_sha256": declaration.combined_result_artifact.sha256,
            "training_run_performed": False,
            "test_split_opened": False,
            "completed_cells": [cell.as_dict() for cell in cells],
        },
    )


def load_exploratory_joint_progress(
    path: str | Path,
    *,
    declaration: ExploratoryJointOverrideDeclaration,
) -> tuple[FrontierCellResult, ...]:
    payload = _json_mapping(Path(path), name="override progress")
    expected = {
        "schema",
        "declaration_sha256",
        "combined_result_sha256",
        "training_run_performed",
        "test_split_opened",
        "completed_cells",
    }
    if set(payload) != expected:
        raise ExploratoryJointOverrideError("override progress fields have drifted")
    if (
        payload["schema"] != EXPLORATORY_JOINT_PROGRESS_SCHEMA
        or payload["declaration_sha256"] != declaration.sha256
        or payload["combined_result_sha256"] != declaration.combined_result_artifact.sha256
        or payload["training_run_performed"] is not False
        or payload["test_split_opened"] is not False
    ):
        raise ExploratoryJointOverrideError("override progress provenance has drifted")
    raw_cells = payload["completed_cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) > len(declaration.selected_cells):
        raise ExploratoryJointOverrideError("override progress cell prefix is malformed")
    cells: list[FrontierCellResult] = []
    for raw, cell in zip(raw_cells, declaration.selected_cells, strict=False):
        if not isinstance(raw, Mapping):
            raise ExploratoryJointOverrideError("override progress cell is malformed")
        cells.append(
            FrontierCellResult.from_dict(
                raw,
                cell=cell,
                miss_budget=declaration.miss_budget,
            )
        )
    result = tuple(cells)
    _validate_cells(declaration, result, require_complete=False)
    return result


def execute_exploratory_joint_override(
    declaration: ExploratoryJointOverrideDeclaration,
    *,
    project_root: str | Path,
    completed_cells: tuple[FrontierCellResult, ...] = (),
    progress: ProgressCallback | None = None,
    checkpoint: CheckpointCallback | None = None,
) -> ExploratoryJointOverrideResult:
    """Execute the three frozen sensing cells under the selected pair."""

    root = Path(project_root).expanduser().resolve(strict=False)
    dry = structural_exploratory_joint_dry_run(declaration, project_root=root)
    del dry
    _validate_cells(declaration, completed_cells, require_complete=False)
    source = declaration.combined_declaration.longer_declaration.receive_declaration.source_frontier
    source_report = structural_system_dry_run(source, project_root=root)
    config = _selected_config(declaration, project_root=root)
    normalization = _identity_normalization(config)
    results = list(completed_cells)
    total = len(declaration.selected_cells)
    for index, cell in enumerate(declaration.selected_cells, start=1):
        if index <= len(results):
            restored = results[index - 1]
            if restored.config_hash != config_hash(
                config
            ) or restored.policy_environment_scope_hash != scope_hash(config, "policy_environment"):
                raise ExploratoryJointOverrideError("override checkpoint configuration drifted")
            continue
        if progress is not None:
            progress(index, total, cell.cell_id)
        point = cell.physical_point
        evaluation = evaluate_pair_local_joint_windows(
            config,
            windows=source_report.windows,
            environment_seed=source.environment_seed,
            normalization_state=normalization,
            sensitivity_band=point.sensing_band.band,
            collision_subchannels=point.rf_capacity.subchannels,
            oracle_controls_unusable_rows=False,
            exact_assignment_cap=declaration.exact_assignment_cap,
            max_search_iterations=declaration.max_search_iterations,
            receive_diversity=declaration.selected_profile.profile.physical_profile(),
        )
        results.append(
            _cell_result(
                declaration,
                cell,
                config,
                evaluation.densities,
                evaluation.campaign,
            )
        )
        if checkpoint is not None:
            checkpoint(tuple(results))
    return ExploratoryJointOverrideResult(
        declaration=declaration,
        cells=tuple(results),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "EXPLORATORY_JOINT_DECLARATION_SCHEMA",
    "EXPLORATORY_JOINT_PROGRESS_SCHEMA",
    "EXPLORATORY_JOINT_RESULT_SCHEMA",
    "ExploratoryJointOverrideDeclaration",
    "ExploratoryJointOverrideError",
    "ExploratoryJointOverrideResult",
    "execute_exploratory_joint_override",
    "load_exploratory_joint_override_declaration",
    "load_exploratory_joint_progress",
    "structural_exploratory_joint_dry_run",
    "write_exploratory_joint_progress",
]
