"""Two-stage executor for the frozen receive-diversity feasibility frontier."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from hybrid_v2x_rl.agents.receive_diversity_frontier import (
    ReceiveDiversityFrontierDeclaration,
    ReceiveDiversityProfile,
    structural_receive_diversity_dry_run,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    FrontierCellResult,
    SystemFeasibilityFrontierResult,
    execute_system_feasibility_frontier,
)
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import FrontierEvaluationCell
from hybrid_v2x_rl.config.loader import load_config
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    run_policy_rollout_with_state,
)
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationState,
    ObservationNormalizer,
)
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PolicyProposal,
    PopulationPolicyFrame,
)

RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.receive-diversity-frontier-result.v1"
)
RECEIVE_DIVERSITY_FRONTIER_PROGRESS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.receive-diversity-frontier-progress.v1"
)

ProgressCallback = Callable[[str, int, int, str], None]


class ReceiveDiversityExecutionError(HybridV2XError):
    """A receive-diversity screen, result, or progress artifact is invalid."""


def propagation_only_action_risk(
    action: PolicyAction,
    *,
    rf_decoding_failure_probability: float,
    vlc_failure_probability: float,
) -> float:
    """Return the action risk after optimistically removing RF access losses."""

    for name, value in (
        ("RF decoding failure", rf_decoding_failure_probability),
        ("VLC failure", vlc_failure_probability),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ReceiveDiversityExecutionError(
                f"{name} probability must lie in [0, 1]"
            )
    resources = action_resources(action)
    risk = 1.0
    if resources.uses_rf:
        risk *= rf_decoding_failure_probability ** resources.reserved_rf_attempts
    if resources.uses_vlc:
        risk *= vlc_failure_probability
    return float(risk)


def _profile_dict(profile: ReceiveDiversityProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "antenna_count": profile.antenna_count,
        "combining_rule": profile.combining_rule,
        "branch_correlation": profile.branch_correlation,
        "implementation_loss_db": profile.implementation_loss_db,
        "headline": profile.headline,
        "authorizes_training": profile.authorizes_training,
    }


@dataclass(slots=True)
class _PropagationTally:
    frames: int = 0
    transitions: int = 0
    risk_sum: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * len(PolicyAction))

    def observe(
        self,
        actions: tuple[PolicyAction, ...],
        risks: tuple[float, ...],
    ) -> None:
        if len(actions) != len(risks) or not actions:
            raise ReceiveDiversityExecutionError(
                "propagation screen requires aligned nonempty rows"
            )
        self.frames += 1
        self.transitions += len(actions)
        self.risk_sum += math.fsum(risks)
        for action in actions:
            self.action_counts[int(action)] += 1


@dataclass(slots=True)
class _PropagationScreenPolicy:
    tallies: dict[float, _PropagationTally] = field(default_factory=dict)
    name: str = "receive-diversity-propagation-only-oracle"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise ReceiveDiversityExecutionError(
                "propagation screen requires oracle channel truth"
            )
        if decision.frame.source.split != "validation":
            raise ReceiveDiversityExecutionError(
                "propagation screen accepts validation data only"
            )
        if set(channel_truth) != set(decision.frame.active_pair_ids):
            raise ReceiveDiversityExecutionError(
                "propagation screen truth is not pair aligned"
            )
        if decision.population_size == 0:
            return ()
        selected: list[PolicyAction] = []
        risks: list[float] = []
        allowed = decision.action_space.mask.allowed_actions
        fallback = decision.action_space.fallback_action
        for actor_row in decision.actor_frame.rows:
            truth = channel_truth[actor_row.pair_id]
            choices = allowed if actor_row.usable else (fallback,)

            def rank(
                action: PolicyAction,
                *,
                pair_truth: object = truth,
            ) -> tuple[float, float, int, int]:
                if not hasattr(pair_truth, "rf_propagation") or not hasattr(
                    pair_truth,
                    "vlc_result",
                ):
                    raise ReceiveDiversityExecutionError(
                        "propagation-screen channel truth is malformed"
                    )
                risk = propagation_only_action_risk(
                    action,
                    rf_decoding_failure_probability=(
                        pair_truth.rf_propagation.decoding_failure_probability
                    ),
                    vlc_failure_probability=(
                        pair_truth.vlc_result.total_failure_probability
                    ),
                )
                resources = action_resources(action)
                return (
                    risk,
                    decision.resource_map.activation_cost(action),
                    resources.reserved_rf_attempts,
                    int(action),
                )

            action = min(choices, key=rank)
            selected.append(action)
            risks.append(rank(action)[0])
        self.tallies.setdefault(
            decision.frame.source.density,
            _PropagationTally(),
        ).observe(tuple(selected), tuple(risks))
        return tuple(fallback if row.usable else None for row in decision.actor_frame.rows)


@dataclass(frozen=True, slots=True)
class PropagationScreenProfileResult:
    """Necessary-condition rows for one frozen receive profile."""

    profile: ReceiveDiversityProfile
    rows: tuple[dict[str, object], ...]
    passing_optical_configuration_names: tuple[str, ...]

    @property
    def survives(self) -> bool:
        return bool(self.passing_optical_configuration_names)

    def as_dict(self) -> dict[str, object]:
        return {
            "receive_profile": _profile_dict(self.profile),
            "criterion": (
                "survives only when at least one frozen optical configuration has "
                "mean propagation-only conditional miss risk <= miss budget at "
                "every required density"
            ),
            "rows": list(self.rows),
            "passing_optical_configuration_names": list(
                self.passing_optical_configuration_names
            ),
            "survives": self.survives,
        }


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if (
        not state.frozen
        or any(state.count)
        or any(state.mean)
        or any(state.second_moment)
    ):
        raise ReceiveDiversityExecutionError(
            "receive-diversity execution requires identity normalization"
        )
    return state


def _screen_profile_optical_configuration(
    declaration: ReceiveDiversityFrontierDeclaration,
    profile: ReceiveDiversityProfile,
    optical_name: str,
    *,
    project_root: Path,
    windows: tuple[EvaluationWindow, ...],
) -> tuple[dict[str, object], ...]:
    source = declaration.source_frontier
    matches = tuple(
        optical
        for optical in source.optical_configurations
        if optical.name == optical_name
    )
    if len(matches) != 1:
        raise ReceiveDiversityExecutionError(
            "propagation screen optical configuration is not frozen"
        )
    optical = matches[0]
    config = load_config(
        (*source.base_config_layers, *optical.additional_config_layers),
        project_root=project_root,
    )
    normalization = _identity_normalization(config)
    catalog = TraceCatalog.from_splits(
        config.paths.trace_root,
        config.environment.splits,
    )
    validation = {
        trace.trace_id: trace for trace in catalog.for_split("validation")
    }
    headline = source.headline_point
    policy = _PropagationScreenPolicy()
    for window in windows:
        try:
            trace = validation[window.trace_id]
        except KeyError as error:
            raise ReceiveDiversityExecutionError(
                "propagation-screen window is absent from validation"
            ) from error
        result = run_policy_rollout_with_state(
            config,
            trace,
            policy=policy,
            environment_seed=source.environment_seed,
            policy_seed=0,
            start_frame_index=window.start_frame_index,
            max_frames=window.frames,
            normalization_state=normalization,
            sensitivity_band=headline.sensing_band.band,
            collision_subchannels=headline.rf_capacity.subchannels,
            receive_diversity=profile.physical_profile(),
            oracle_controls_unusable_rows=False,
        )
        if result.normalization_state != normalization:
            raise ReceiveDiversityExecutionError(
                "frozen propagation-screen normalization changed"
            )
    rows: list[dict[str, object]] = []
    for density in declaration.densities:
        try:
            tally = policy.tallies[density]
        except KeyError as error:
            raise ReceiveDiversityExecutionError(
                "propagation screen is missing a declared density"
            ) from error
        mean = tally.risk_sum / tally.transitions
        rows.append(
            {
                "optical_configuration_name": optical.name,
                "receiver_fov_deg": optical.receiver_fov_deg,
                "density_vehicles_per_lane_km": density,
                "frames": tally.frames,
                "transitions": tally.transitions,
                "mean_optimistic_propagation_only_conditional_miss_lower_bound": mean,
                "budget_multiple": mean / declaration.miss_budget,
                "meets_budget": mean <= declaration.miss_budget,
                "lower_bound_action_counts": {
                    label: tally.action_counts[index]
                    for index, label in enumerate(POLICY_ACTION_ORDER)
                },
            }
        )
    return tuple(rows)


def _validate_screen_results(
    declaration: ReceiveDiversityFrontierDeclaration,
    results: tuple[PropagationScreenProfileResult, ...],
) -> None:
    if tuple(result.profile for result in results) != declaration.receive_profiles:
        raise ReceiveDiversityExecutionError(
            "screen results differ from the frozen receive-profile order"
        )
    optical_names = tuple(
        optical.name
        for optical in declaration.source_frontier.optical_configurations
    )
    for result in results:
        expected_rows = tuple(
            (optical, density)
            for optical in optical_names
            for density in declaration.densities
        )
        actual_rows: list[tuple[str, float]] = []
        passing: list[str] = []
        for row in result.rows:
            optical = row.get("optical_configuration_name")
            density = row.get("density_vehicles_per_lane_km")
            mean = row.get(
                "mean_optimistic_propagation_only_conditional_miss_lower_bound"
            )
            meets = row.get("meets_budget")
            if (
                not isinstance(optical, str)
                or not isinstance(density, int | float)
                or isinstance(density, bool)
                or not isinstance(mean, int | float)
                or isinstance(mean, bool)
                or not math.isfinite(float(mean))
                or not 0.0 <= float(mean) <= 1.0
                or type(meets) is not bool
                or meets != (float(mean) <= declaration.miss_budget)
            ):
                raise ReceiveDiversityExecutionError(
                    "propagation-screen row is malformed"
                )
            actual_rows.append((optical, float(density)))
        if tuple(actual_rows) != expected_rows:
            raise ReceiveDiversityExecutionError(
                "propagation-screen rows do not cover the frozen grid"
            )
        for optical in optical_names:
            rows = tuple(
                row
                for row in result.rows
                if row["optical_configuration_name"] == optical
            )
            if all(cast(bool, row["meets_budget"]) for row in rows):
                passing.append(optical)
        if tuple(passing) != result.passing_optical_configuration_names:
            raise ReceiveDiversityExecutionError(
                "propagation-screen survival decision does not reconcile"
            )


def execute_propagation_screen(
    declaration: ReceiveDiversityFrontierDeclaration,
    *,
    project_root: str | Path,
    progress: ProgressCallback | None = None,
) -> tuple[PropagationScreenProfileResult, ...]:
    """Evaluate the frozen profile-level necessary condition on validation."""

    root = Path(project_root).expanduser().resolve(strict=False)
    source_report = structural_system_dry_run(
        declaration.source_frontier,
        project_root=root,
    )
    optical_names = tuple(
        optical.name
        for optical in declaration.source_frontier.optical_configurations
    )
    total = len(declaration.receive_profiles) * len(optical_names)
    index = 0
    results: list[PropagationScreenProfileResult] = []
    for profile in declaration.receive_profiles:
        rows: list[dict[str, object]] = []
        for optical_name in optical_names:
            index += 1
            if progress is not None:
                progress(
                    "propagation-screen",
                    index,
                    total,
                    f"{profile.name}__{optical_name}",
                )
            rows.extend(
                _screen_profile_optical_configuration(
                    declaration,
                    profile,
                    optical_name,
                    project_root=root,
                    windows=source_report.windows,
                )
            )
        passing = tuple(
            optical_name
            for optical_name in optical_names
            if all(
                cast(bool, row["meets_budget"])
                for row in rows
                if row["optical_configuration_name"] == optical_name
            )
        )
        results.append(
            PropagationScreenProfileResult(
                profile=profile,
                rows=tuple(rows),
                passing_optical_configuration_names=passing,
            )
        )
    frozen = tuple(results)
    _validate_screen_results(declaration, frozen)
    return frozen


@dataclass(frozen=True, slots=True)
class ReceiveDiversityCellResult:
    profile: ReceiveDiversityProfile
    source_cell: FrontierCellResult

    @property
    def cell_id(self) -> str:
        return f"{self.profile.name}__{self.source_cell.cell.cell_id}"

    def as_dict(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "receive_profile": _profile_dict(self.profile),
            "source_cell": self.source_cell.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReceiveProfileFrontierResult:
    profile: ReceiveDiversityProfile
    system_result: SystemFeasibilityFrontierResult

    def as_dict(self) -> dict[str, object]:
        return {
            "receive_profile": _profile_dict(self.profile),
            "source_frontier_result": self.system_result.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReceiveDiversityFrontierResult:
    declaration: ReceiveDiversityFrontierDeclaration
    screen_results: tuple[PropagationScreenProfileResult, ...]
    profile_frontiers: tuple[ReceiveProfileFrontierResult, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_screen_results(self.declaration, self.screen_results)
        survivors = tuple(
            result.profile for result in self.screen_results if result.survives
        )
        if tuple(result.profile for result in self.profile_frontiers) != survivors:
            raise ReceiveDiversityExecutionError(
                "joint frontiers do not cover every surviving profile in order"
            )
        for result in self.profile_frontiers:
            if result.system_result.declaration != self.declaration.source_frontier:
                raise ReceiveDiversityExecutionError(
                    "joint profile result uses a different source frontier"
                )

    def decision(self) -> dict[str, object]:
        by_name = {
            result.profile.name: result for result in self.profile_frontiers
        }
        profile_rows: list[dict[str, object]] = []
        for screen in self.screen_results:
            frontier = by_name.get(screen.profile.name)
            source_decision = (
                frontier.system_result.decision() if frontier is not None else None
            )
            system_feasible = bool(
                source_decision is not None
                and source_decision["training_authorized"] is True
            )
            profile_rows.append(
                {
                    "receive_profile_name": screen.profile.name,
                    "survived_propagation_screen": screen.survives,
                    "joint_frontier_executed": frontier is not None,
                    "system_feasible": system_feasible,
                    "profile_authorizes_training": bool(
                        system_feasible and screen.profile.authorizes_training
                    ),
                    "robust_feasible_design_ids": (
                        source_decision["robust_feasible_design_ids"]
                        if source_decision is not None
                        else []
                    ),
                }
            )
        headline = self.declaration.headline_receive_profile
        headline_row = next(
            row
            for row in profile_rows
            if row["receive_profile_name"] == headline.name
        )
        return {
            "headline_receive_profile": headline.name,
            "headline_survived_propagation_screen": headline_row[
                "survived_propagation_screen"
            ],
            "training_authorized": headline_row["profile_authorizes_training"],
            "profiles": profile_rows,
            "sensitivity_profiles_authorized_training": False,
            "test_split_opened": False,
            "claim_boundary": (
                "conditional synthetic feasibility evidence, not measured on-road "
                "reliability"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        survivors = tuple(
            result for result in self.screen_results if result.survives
        )
        return {
            "schema": RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "source_system_frontier": {
                "path": str(self.declaration.source_frontier.path),
                "sha256": self.declaration.source_frontier.sha256,
            },
            "reliability_miss_budget": self.declaration.miss_budget,
            "densities_vehicles_per_lane_km": list(self.declaration.densities),
            "actor_used": False,
            "checkpoint_used": False,
            "training_run_performed": False,
            "test_split_opened": False,
            "frontier_complete": True,
            "profiles_screened": len(self.screen_results),
            "profiles_surviving": len(survivors),
            "evaluation_cells_before_screening": (
                self.declaration.evaluation_cells_before_screening
            ),
            "evaluation_cells_executed": sum(
                len(result.system_result.cells)
                for result in self.profile_frontiers
            ),
            "propagation_screen": [
                result.as_dict() for result in self.screen_results
            ],
            "joint_frontiers": [
                result.as_dict() for result in self.profile_frontiers
            ],
            "decision": self.decision(),
        }

    def write_json(self, path: str | Path) -> Path:
        return _atomic_json_write(Path(path), self.as_dict())


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


def _expected_joint_plan(
    declaration: ReceiveDiversityFrontierDeclaration,
    screen_results: tuple[PropagationScreenProfileResult, ...],
) -> tuple[tuple[ReceiveDiversityProfile, FrontierEvaluationCell], ...]:
    return tuple(
        (screen.profile, cell)
        for screen in screen_results
        if screen.survives
        for cell in declaration.source_frontier.evaluation_cells
    )


def write_receive_diversity_progress(
    path: str | Path,
    *,
    declaration: ReceiveDiversityFrontierDeclaration,
    screen_results: tuple[PropagationScreenProfileResult, ...],
    cells: tuple[ReceiveDiversityCellResult, ...],
) -> Path:
    """Atomically persist the complete screen and an ordered joint-cell prefix."""

    _validate_screen_results(declaration, screen_results)
    expected = _expected_joint_plan(declaration, screen_results)
    actual = tuple((result.profile, result.source_cell.cell) for result in cells)
    if actual != expected[: len(actual)]:
        raise ReceiveDiversityExecutionError(
            "receive-diversity progress is not an ordered frozen-grid prefix"
        )
    return _atomic_json_write(
        Path(path),
        {
            "schema": RECEIVE_DIVERSITY_FRONTIER_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "source_frontier_sha256": declaration.source_frontier.sha256,
            "window_source_sha256": (
                declaration.source_frontier.window_source_sha256
            ),
            "test_split_opened": False,
            "screen_results": [result.as_dict() for result in screen_results],
            "completed_cells": [result.as_dict() for result in cells],
        },
    )


def _screen_result_from_dict(
    payload: object,
    *,
    profile: ReceiveDiversityProfile,
) -> PropagationScreenProfileResult:
    if not isinstance(payload, Mapping):
        raise ReceiveDiversityExecutionError("progress screen result must be an object")
    expected = {
        "receive_profile",
        "criterion",
        "rows",
        "passing_optical_configuration_names",
        "survives",
    }
    if set(payload) != expected or payload.get("receive_profile") != _profile_dict(profile):
        raise ReceiveDiversityExecutionError(
            "progress screen profile differs from the declaration"
        )
    raw_rows = payload.get("rows")
    raw_passing = payload.get("passing_optical_configuration_names")
    if (
        not isinstance(raw_rows, list)
        or any(not isinstance(row, Mapping) for row in raw_rows)
        or not isinstance(raw_passing, list)
        or any(not isinstance(name, str) for name in raw_passing)
    ):
        raise ReceiveDiversityExecutionError("progress screen rows are malformed")
    result = PropagationScreenProfileResult(
        profile=profile,
        rows=tuple(dict(row) for row in raw_rows),
        passing_optical_configuration_names=tuple(raw_passing),
    )
    if payload.get("survives") is not result.survives:
        raise ReceiveDiversityExecutionError(
            "progress screen survival flag does not reconcile"
        )
    if result.as_dict() != dict(payload):
        raise ReceiveDiversityExecutionError(
            "progress screen result differs from the frozen result contract"
        )
    return result


def load_receive_diversity_progress(
    path: str | Path,
    *,
    declaration: ReceiveDiversityFrontierDeclaration,
) -> tuple[
    tuple[PropagationScreenProfileResult, ...],
    tuple[ReceiveDiversityCellResult, ...],
]:
    """Load progress only when screen, declaration, and ordered cells reconcile."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiveDiversityExecutionError(
            "receive-diversity progress artifact is unreadable",
            artifact_path=source,
        ) from error
    expected_fields = {
        "schema",
        "declaration_sha256",
        "source_frontier_sha256",
        "window_source_sha256",
        "test_split_opened",
        "screen_results",
        "completed_cells",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ReceiveDiversityExecutionError(
            "receive-diversity progress fields do not match the schema"
        )
    if (
        payload["schema"] != RECEIVE_DIVERSITY_FRONTIER_PROGRESS_SCHEMA
        or payload["declaration_sha256"] != declaration.sha256
        or payload["source_frontier_sha256"]
        != declaration.source_frontier.sha256
        or payload["window_source_sha256"]
        != declaration.source_frontier.window_source_sha256
        or payload["test_split_opened"] is not False
    ):
        raise ReceiveDiversityExecutionError(
            "receive-diversity progress provenance has drifted"
        )
    raw_screens = payload["screen_results"]
    if not isinstance(raw_screens, list) or len(raw_screens) != len(
        declaration.receive_profiles
    ):
        raise ReceiveDiversityExecutionError(
            "progress does not contain every propagation screen result"
        )
    screens = tuple(
        _screen_result_from_dict(raw, profile=profile)
        for raw, profile in zip(
            raw_screens,
            declaration.receive_profiles,
            strict=True,
        )
    )
    _validate_screen_results(declaration, screens)
    expected_plan = _expected_joint_plan(declaration, screens)
    raw_cells = payload["completed_cells"]
    if not isinstance(raw_cells, list) or len(raw_cells) > len(expected_plan):
        raise ReceiveDiversityExecutionError(
            "progress completed cells exceed the screened frontier"
        )
    cells: list[ReceiveDiversityCellResult] = []
    for index, raw in enumerate(raw_cells):
        if not isinstance(raw, Mapping) or set(raw) != {
            "cell_id",
            "receive_profile",
            "source_cell",
        }:
            raise ReceiveDiversityExecutionError(
                "progress joint cell is malformed"
            )
        profile, cell = expected_plan[index]
        if raw["receive_profile"] != _profile_dict(profile):
            raise ReceiveDiversityExecutionError(
                "progress joint-cell profile differs from the declaration"
            )
        source_cell = raw["source_cell"]
        if not isinstance(source_cell, Mapping):
            raise ReceiveDiversityExecutionError(
                "progress source cell must be an object"
            )
        restored = ReceiveDiversityCellResult(
            profile=profile,
            source_cell=FrontierCellResult.from_dict(
                source_cell,
                cell=cell,
                miss_budget=declaration.miss_budget,
            ),
        )
        if raw["cell_id"] != restored.cell_id:
            raise ReceiveDiversityExecutionError(
                "progress joint-cell identity does not reconcile"
            )
        cells.append(restored)
    frozen_cells = tuple(cells)
    actual = tuple((row.profile, row.source_cell.cell) for row in frozen_cells)
    if actual != expected_plan[: len(actual)]:
        raise ReceiveDiversityExecutionError(
            "progress joint cells are not an ordered prefix"
        )
    return screens, frozen_cells


CheckpointCallback = Callable[
    [
        tuple[PropagationScreenProfileResult, ...],
        tuple[ReceiveDiversityCellResult, ...],
    ],
    None,
]


def execute_receive_diversity_frontier(
    declaration: ReceiveDiversityFrontierDeclaration,
    *,
    project_root: str | Path,
    progress: ProgressCallback | None = None,
    screen_results: tuple[PropagationScreenProfileResult, ...] | None = None,
    completed_cells: tuple[ReceiveDiversityCellResult, ...] = (),
    checkpoint: CheckpointCallback | None = None,
) -> ReceiveDiversityFrontierResult:
    """Run the frozen screen and complete every surviving joint frontier."""

    root = Path(project_root).expanduser().resolve(strict=False)
    structural_receive_diversity_dry_run(declaration, project_root=root)
    screens = (
        execute_propagation_screen(
            declaration,
            project_root=root,
            progress=progress,
        )
        if screen_results is None
        else screen_results
    )
    _validate_screen_results(declaration, screens)
    if screen_results is None and checkpoint is not None:
        checkpoint(screens, ())
    plan = _expected_joint_plan(declaration, screens)
    actual_prefix = tuple(
        (result.profile, result.source_cell.cell) for result in completed_cells
    )
    if actual_prefix != plan[: len(actual_prefix)]:
        raise ReceiveDiversityExecutionError(
            "completed receive-diversity cells are not a frozen-grid prefix"
        )
    source_cells_per_profile = len(declaration.source_frontier.evaluation_cells)
    completed = list(completed_cells)
    profile_frontiers: list[ReceiveProfileFrontierResult] = []
    survivors = tuple(screen.profile for screen in screens if screen.survives)
    for profile_index, profile in enumerate(survivors):
        offset = profile_index * source_cells_per_profile
        local_completed = tuple(
            result.source_cell
            for result in completed_cells[
                offset : min(len(completed_cells), offset + source_cells_per_profile)
            ]
            if result.profile == profile
        )

        def local_progress(
            index: int,
            _total: int,
            cell_id: str,
            *,
            profile_name: str = profile.name,
            profile_offset: int = offset,
        ) -> None:
            if progress is not None:
                progress(
                    "joint-frontier",
                    profile_offset + index,
                    len(plan),
                    f"{profile_name}__{cell_id}",
                )

        def local_checkpoint(
            source_cells: tuple[FrontierCellResult, ...],
            *,
            active_profile: ReceiveDiversityProfile = profile,
            profile_offset: int = offset,
        ) -> None:
            del completed[profile_offset:]
            completed.extend(
                ReceiveDiversityCellResult(active_profile, source_cell)
                for source_cell in source_cells
            )
            if checkpoint is not None:
                checkpoint(screens, tuple(completed))

        system_result = execute_system_feasibility_frontier(
            declaration.source_frontier,
            project_root=root,
            progress=local_progress,
            completed_cells=local_completed,
            checkpoint=local_checkpoint,
            receive_diversity=profile.physical_profile(),
        )
        completed[:] = list(completed[:offset]) + [
            ReceiveDiversityCellResult(profile, source_cell)
            for source_cell in system_result.cells
        ]
        profile_frontiers.append(
            ReceiveProfileFrontierResult(
                profile=profile,
                system_result=system_result,
            )
        )
    if len(completed) != len(plan):
        raise ReceiveDiversityExecutionError(
            "receive-diversity execution did not complete the screened grid"
        )
    return ReceiveDiversityFrontierResult(
        declaration=declaration,
        screen_results=screens,
        profile_frontiers=tuple(profile_frontiers),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "RECEIVE_DIVERSITY_FRONTIER_PROGRESS_SCHEMA",
    "RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA",
    "PropagationScreenProfileResult",
    "ReceiveDiversityCellResult",
    "ReceiveDiversityExecutionError",
    "ReceiveDiversityFrontierResult",
    "execute_propagation_screen",
    "execute_receive_diversity_frontier",
    "load_receive_diversity_progress",
    "propagation_only_action_risk",
    "write_receive_diversity_progress",
]
