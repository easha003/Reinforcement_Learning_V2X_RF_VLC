"""Validation-only decomposition of the failed propagation lower bound.

The receive-diversity frontier deliberately stopped before its joint search
because its optimistic propagation-only screen already exceeded the headline
miss budget.  This module replays only the frozen validation windows for the
frozen headline receive profile and partitions that same lower bound by the
mechanisms needed to select the next physical intervention.  It never loads a
checkpoint, trains a policy, or opens the test split.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from hybrid_v2x_rl.agents.receive_diversity_execution import (
    RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA,
    propagation_only_action_risk,
)
from hybrid_v2x_rl.agents.receive_diversity_frontier import (
    ReceiveDiversityFrontierDeclaration,
    ReceiveDiversityProfile,
    load_receive_diversity_frontier_declaration,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    OpticalConfigurationLevel,
)
from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult
from hybrid_v2x_rl.config.loader import load_config, load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources
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

PROPAGATION_TAIL_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.propagation-tail-decomposition-declaration.v1"
)
PROPAGATION_TAIL_RESULT_SCHEMA: Final = "hybrid-rf-vlc-rl.propagation-tail-decomposition-result.v1"
EXPECTED_DIMENSIONS: Final = (
    "lower_bound_action",
    "actor_control_status",
    "rf_propagation_state",
    "vlc_geometric_availability",
)


class PropagationTailDiagnosticError(HybridV2XError):
    """A frozen declaration, source result, or decomposition is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PropagationTailDiagnosticError(f"{name} must be a non-empty path")
    supplied = Path(value).expanduser()
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


def _mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PropagationTailDiagnosticError(f"{name} fields do not match the frozen schema")
    return cast(Mapping[str, object], value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PropagationTailDiagnosticError(f"{name} must be non-empty text")
    return value


def _false(value: object, *, name: str) -> None:
    if value is not False:
        raise PropagationTailDiagnosticError(f"{name} must be false")


def _float_sequence(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise PropagationTailDiagnosticError(f"{name} must be a nonempty list")
    converted: list[float] = []
    for raw in value:
        if (
            not isinstance(raw, int | float)
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        ):
            raise PropagationTailDiagnosticError(f"{name} must contain finite numbers")
        converted.append(float(raw))
    return tuple(converted)


def _text_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise PropagationTailDiagnosticError(f"{name} must be a nonempty list")
    converted = tuple(_text(raw, name=name) for raw in value)
    if len(converted) != len(set(converted)):
        raise PropagationTailDiagnosticError(f"{name} must not contain duplicates")
    return converted


@dataclass(frozen=True, slots=True)
class PropagationTailDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    miss_budget: float
    densities: tuple[float, ...]
    receive_declaration_path: Path
    receive_declaration_sha256: str
    receive_result_path: Path
    receive_result_sha256: str
    receive_result_schema: str
    receive_profile_name: str
    optical_configuration_names: tuple[str, ...]
    dimensions: tuple[str, ...]
    risk_tail_thresholds: tuple[float, ...]
    output_path: Path


def load_propagation_tail_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> PropagationTailDeclaration:
    """Load and fail-closed validate the bounded diagnostic declaration."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="diagnostic declaration")
    top = _mapping(
        load_yaml_file(declaration_path),
        name="diagnostic declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "selection",
            "execution",
            "decision",
        },
    )
    if top["schema"] != PROPAGATION_TAIL_DECLARATION_SCHEMA:
        raise PropagationTailDiagnosticError("diagnostic declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="diagnostic objective",
        keys={
            "miss_budget",
            "densities_vehicles_per_lane_km",
            "required_split",
            "interpretation",
        },
    )
    miss_budget_values = _float_sequence([objective["miss_budget"]], name="miss budget")
    miss_budget = miss_budget_values[0]
    densities = _float_sequence(
        objective["densities_vehicles_per_lane_km"], name="diagnostic densities"
    )
    if (
        not 0.0 < miss_budget < 1.0
        or densities != (20.0, 30.0)
        or objective["required_split"] != "validation"
    ):
        raise PropagationTailDiagnosticError(
            "diagnostic objective must remain frozen to validation densities 20 and 30"
        )

    evidence = _mapping(
        top["evidence"],
        name="diagnostic evidence",
        keys={
            "receive_diversity_declaration",
            "receive_diversity_result",
            "actor_used",
            "checkpoint_used",
        },
    )
    _false(evidence["actor_used"], name="actor_used")
    _false(evidence["checkpoint_used"], name="checkpoint_used")
    declaration_evidence = _mapping(
        evidence["receive_diversity_declaration"],
        name="receive-diversity declaration evidence",
        keys={"path", "sha256"},
    )
    result_evidence = _mapping(
        evidence["receive_diversity_result"],
        name="receive-diversity result evidence",
        keys={"path", "sha256", "schema"},
    )
    receive_declaration_path = _resolve(
        root,
        declaration_evidence["path"],
        name="receive-diversity declaration path",
    )
    receive_result_path = _resolve(
        root,
        result_evidence["path"],
        name="receive-diversity result path",
    )
    receive_declaration_sha256 = _text(
        declaration_evidence["sha256"], name="receive-diversity declaration SHA-256"
    )
    receive_result_sha256 = _text(
        result_evidence["sha256"], name="receive-diversity result SHA-256"
    )
    receive_result_schema = _text(result_evidence["schema"], name="receive-diversity result schema")
    if receive_result_schema != RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA:
        raise PropagationTailDiagnosticError(
            "source receive-diversity result schema is unsupported"
        )
    if verify_evidence:
        for source, expected, name in (
            (
                receive_declaration_path,
                receive_declaration_sha256,
                "receive-diversity declaration",
            ),
            (receive_result_path, receive_result_sha256, "receive-diversity result"),
        ):
            if not source.is_file() or _sha256(source) != expected:
                raise PropagationTailDiagnosticError(
                    f"{name} evidence is absent or has drifted",
                    artifact_path=source,
                )

    selection = _mapping(
        top["selection"],
        name="diagnostic selection",
        keys={
            "receive_profile",
            "optical_configurations",
            "fallback_view",
            "dimensions",
            "risk_tail_thresholds",
        },
    )
    profile_name = _text(selection["receive_profile"], name="receive profile")
    optical_names = _text_sequence(
        selection["optical_configurations"], name="optical configurations"
    )
    dimensions = _text_sequence(selection["dimensions"], name="dimensions")
    thresholds = _float_sequence(selection["risk_tail_thresholds"], name="risk tail thresholds")
    if selection["fallback_view"] != "contract":
        raise PropagationTailDiagnosticError("diagnostic must preserve contract fallback")
    if dimensions != EXPECTED_DIMENSIONS:
        raise PropagationTailDiagnosticError("diagnostic dimensions have drifted")
    if (
        tuple(sorted(thresholds)) != thresholds
        or len(thresholds) != len(set(thresholds))
        or thresholds[0] <= 0.0
        or thresholds[-1] >= 1.0
    ):
        raise PropagationTailDiagnosticError(
            "risk tail thresholds must be unique, increasing, and lie in (0, 1)"
        )

    execution = _mapping(
        top["execution"],
        name="diagnostic execution",
        keys={
            "no_adaptive_axis_expansion",
            "no_training",
            "no_test_split",
            "output_path",
        },
    )
    if (
        execution["no_adaptive_axis_expansion"] is not True
        or execution["no_training"] is not True
        or execution["no_test_split"] is not True
    ):
        raise PropagationTailDiagnosticError("diagnostic safety gates must remain true")
    decision = _mapping(
        top["decision"],
        name="diagnostic decision",
        keys={"purpose", "training_authorization", "claim_boundary"},
    )
    _false(decision["training_authorization"], name="training authorization")

    return PropagationTailDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        miss_budget=miss_budget,
        densities=densities,
        receive_declaration_path=receive_declaration_path,
        receive_declaration_sha256=receive_declaration_sha256,
        receive_result_path=receive_result_path,
        receive_result_sha256=receive_result_sha256,
        receive_result_schema=receive_result_schema,
        receive_profile_name=profile_name,
        optical_configuration_names=optical_names,
        dimensions=dimensions,
        risk_tail_thresholds=thresholds,
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )


@dataclass(frozen=True, slots=True, order=True)
class PropagationTailKey:
    lower_bound_action: str
    actor_control_status: str
    rf_propagation_state: str
    vlc_geometric_availability: str

    def as_dict(self) -> dict[str, object]:
        return {
            "lower_bound_action": self.lower_bound_action,
            "actor_control_status": self.actor_control_status,
            "rf_propagation_state": self.rf_propagation_state,
            "vlc_geometric_availability": self.vlc_geometric_availability,
        }


@dataclass(slots=True)
class _GroupTally:
    transitions: int = 0
    selected_risk_sum: float = 0.0
    rf_decoding_failure_sum: float = 0.0
    vlc_failure_sum: float = 0.0

    def observe(self, *, selected_risk: float, rf_failure: float, vlc_failure: float) -> None:
        self.transitions += 1
        self.selected_risk_sum += selected_risk
        self.rf_decoding_failure_sum += rf_failure
        self.vlc_failure_sum += vlc_failure


@dataclass(slots=True)
class PropagationTailTally:
    frames: int = 0
    transitions: int = 0
    selected_risk_sum: float = 0.0
    selected_risks: list[float] = field(default_factory=list)
    groups: dict[PropagationTailKey, _GroupTally] = field(default_factory=dict)

    def begin_frame(self) -> None:
        self.frames += 1

    def observe(
        self,
        key: PropagationTailKey,
        *,
        selected_risk: float,
        rf_failure: float,
        vlc_failure: float,
    ) -> None:
        for name, value in (
            ("selected risk", selected_risk),
            ("RF failure", rf_failure),
            ("VLC failure", vlc_failure),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise PropagationTailDiagnosticError(f"{name} must lie in [0, 1]")
        self.transitions += 1
        self.selected_risk_sum += selected_risk
        self.selected_risks.append(selected_risk)
        self.groups.setdefault(key, _GroupTally()).observe(
            selected_risk=selected_risk,
            rf_failure=rf_failure,
            vlc_failure=vlc_failure,
        )

    def as_dict(self, *, thresholds: tuple[float, ...]) -> dict[str, object]:
        if self.frames <= 0 or self.transitions <= 0:
            raise PropagationTailDiagnosticError("diagnostic tally is empty")
        group_count = sum(group.transitions for group in self.groups.values())
        group_risk = math.fsum(group.selected_risk_sum for group in self.groups.values())
        if group_count != self.transitions or not math.isclose(
            group_risk,
            self.selected_risk_sum,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise PropagationTailDiagnosticError("diagnostic group decomposition does not close")

        total = self.selected_risk_sum
        rows: list[dict[str, object]] = []
        for key, group in sorted(self.groups.items()):
            row: dict[str, object] = key.as_dict()
            row.update(
                {
                    "transitions": group.transitions,
                    "transition_fraction": group.transitions / self.transitions,
                    "selected_risk_sum": group.selected_risk_sum,
                    "mean_selected_risk": (group.selected_risk_sum / group.transitions),
                    "contribution_to_density_mean": (group.selected_risk_sum / self.transitions),
                    "fraction_of_total_risk_sum": (
                        group.selected_risk_sum / total if total else 0.0
                    ),
                    "mean_rf_decoding_failure_probability": (
                        group.rf_decoding_failure_sum / group.transitions
                    ),
                    "mean_vlc_failure_probability": (group.vlc_failure_sum / group.transitions),
                }
            )
            rows.append(row)
        rows.sort(
            key=lambda row: (
                -cast(float, row["fraction_of_total_risk_sum"]),
                cast(str, row["lower_bound_action"]),
                cast(str, row["actor_control_status"]),
                cast(str, row["rf_propagation_state"]),
                cast(str, row["vlc_geometric_availability"]),
            )
        )

        ordered = sorted(self.selected_risks)

        def nearest_rank(probability: float) -> float:
            index = max(0, math.ceil(probability * len(ordered)) - 1)
            return ordered[index]

        tails: list[dict[str, object]] = []
        for threshold in thresholds:
            selected = tuple(value for value in ordered if value > threshold)
            risk_sum = math.fsum(selected)
            tails.append(
                {
                    "threshold": threshold,
                    "transitions_above": len(selected),
                    "transition_fraction_above": len(selected) / self.transitions,
                    "risk_sum_above": risk_sum,
                    "fraction_of_total_risk_sum_above": (risk_sum / total if total else 0.0),
                }
            )
        return {
            "frames": self.frames,
            "transitions": self.transitions,
            "selected_risk_sum": total,
            "mean_selected_risk": total / self.transitions,
            "empirical_nearest_rank_quantiles": {
                "p50": nearest_rank(0.50),
                "p90": nearest_rank(0.90),
                "p95": nearest_rank(0.95),
                "p99": nearest_rank(0.99),
                "p999": nearest_rank(0.999),
                "maximum": ordered[-1],
            },
            "risk_tails": tails,
            "groups_by_risk_contribution": rows,
        }


def _vlc_availability(result: VLCChannelResult) -> str:
    if not isinstance(result, VLCChannelResult):
        raise PropagationTailDiagnosticError("VLC truth is malformed")
    if not result.is_geometric_failure:
        return "available"
    causes: list[str] = []
    if result.occluded:
        causes.append("occluded")
    if not result.within_field_of_view:
        causes.append("outside-fov")
    if not result.beam_aimed:
        causes.append("beam-not-aimed")
    if not causes:  # pragma: no cover - is_geometric_failure defines the flags.
        raise PropagationTailDiagnosticError("VLC geometric failure has no cause")
    return "unavailable:" + "+".join(causes)


@dataclass(slots=True)
class _PropagationTailPolicy:
    densities: tuple[float, ...]
    tallies: dict[float, PropagationTailTally] = field(default_factory=dict)
    name: str = "propagation-tail-lower-bound-oracle"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise PropagationTailDiagnosticError("diagnostic requires oracle channel truth")
        if decision.frame.source.split != "validation":
            raise PropagationTailDiagnosticError("diagnostic accepts validation data only")
        density = float(decision.frame.source.density)
        if density not in self.densities:
            raise PropagationTailDiagnosticError("diagnostic received an undeclared density")
        if set(channel_truth) != set(decision.frame.active_pair_ids):
            raise PropagationTailDiagnosticError("diagnostic truth is not pair aligned")
        tally = self.tallies.setdefault(density, PropagationTailTally())
        tally.begin_frame()
        if decision.population_size == 0:
            return ()

        allowed = decision.action_space.mask.allowed_actions
        fallback = decision.action_space.fallback_action
        for actor_row in decision.actor_frame.rows:
            truth = channel_truth[actor_row.pair_id]
            choices = allowed if actor_row.usable else (fallback,)
            rf_failure = truth.rf_propagation.decoding_failure_probability
            vlc_failure = truth.vlc_result.total_failure_probability

            def rank(
                action: PolicyAction,
                *,
                pair_rf_failure: float = rf_failure,
                pair_vlc_failure: float = vlc_failure,
            ) -> tuple[float, float, int, int]:
                risk = propagation_only_action_risk(
                    action,
                    rf_decoding_failure_probability=pair_rf_failure,
                    vlc_failure_probability=pair_vlc_failure,
                )
                resources = action_resources(action)
                return (
                    risk,
                    decision.resource_map.activation_cost(action),
                    resources.reserved_rf_attempts,
                    int(action),
                )

            action = min(choices, key=rank)
            tally.observe(
                PropagationTailKey(
                    lower_bound_action=action.label,
                    actor_control_status=(
                        "actor-usable" if actor_row.usable else "contract-fallback"
                    ),
                    rf_propagation_state=truth.rf_propagation.propagation_state.value,
                    vlc_geometric_availability=_vlc_availability(truth.vlc_result),
                ),
                selected_risk=rank(action)[0],
                rf_failure=rf_failure,
                vlc_failure=vlc_failure,
            )
        # Preserve the source screen's non-learning rollout behavior.  The
        # selected lower-bound actions above are diagnostic only.
        return tuple(fallback if row.usable else None for row in decision.actor_frame.rows)


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if not state.frozen or any(state.count) or any(state.mean) or any(state.second_moment):
        raise PropagationTailDiagnosticError("diagnostic requires identity normalization")
    return state


def _source_result(path: Path, *, expected_schema: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PropagationTailDiagnosticError(
            "source receive-diversity result is unreadable", artifact_path=path
        ) from error
    if not isinstance(payload, Mapping) or payload.get("schema") != expected_schema:
        raise PropagationTailDiagnosticError("source receive-diversity result is invalid")
    for flag in ("actor_used", "checkpoint_used", "training_run_performed", "test_split_opened"):
        if payload.get(flag) is not False:
            raise PropagationTailDiagnosticError(f"source result {flag} must be false")
    if payload.get("frontier_complete") is not True:
        raise PropagationTailDiagnosticError("source frontier is incomplete")
    return cast(Mapping[str, object], payload)


def _source_means(
    payload: Mapping[str, object],
    *,
    profile_name: str,
    optical_names: tuple[str, ...],
    densities: tuple[float, ...],
) -> dict[tuple[str, float], float]:
    raw_screens = payload.get("propagation_screen")
    if not isinstance(raw_screens, list):
        raise PropagationTailDiagnosticError("source propagation screen is absent")
    profiles = [
        row
        for row in raw_screens
        if isinstance(row, Mapping)
        and isinstance(row.get("receive_profile"), Mapping)
        and row["receive_profile"].get("name") == profile_name
    ]
    if len(profiles) != 1 or not isinstance(profiles[0].get("rows"), list):
        raise PropagationTailDiagnosticError("source headline receive profile is absent")
    means: dict[tuple[str, float], float] = {}
    for raw in cast(list[object], profiles[0]["rows"]):
        if not isinstance(raw, Mapping):
            continue
        optical = raw.get("optical_configuration_name")
        density = raw.get("density_vehicles_per_lane_km")
        mean = raw.get("mean_optimistic_propagation_only_conditional_miss_lower_bound")
        if (
            isinstance(optical, str)
            and optical in optical_names
            and isinstance(density, int | float)
            and not isinstance(density, bool)
            and float(density) in densities
            and isinstance(mean, int | float)
            and not isinstance(mean, bool)
        ):
            means[(optical, float(density))] = float(mean)
    expected = {(optical, density) for optical in optical_names for density in densities}
    if set(means) != expected:
        raise PropagationTailDiagnosticError("source result does not cover the diagnostic grid")
    return means


def _selected_dependencies(
    declaration: PropagationTailDeclaration,
    *,
    project_root: Path,
) -> tuple[
    ReceiveDiversityFrontierDeclaration,
    ReceiveDiversityProfile,
    tuple[OpticalConfigurationLevel, ...],
    tuple[EvaluationWindow, ...],
    dict[tuple[str, float], float],
]:
    receive = load_receive_diversity_frontier_declaration(
        declaration.receive_declaration_path,
        project_root=project_root,
        verify_evidence=True,
    )
    if receive.sha256 != declaration.receive_declaration_sha256:
        raise PropagationTailDiagnosticError("receive declaration hash does not reconcile")
    profiles = tuple(
        profile
        for profile in receive.receive_profiles
        if profile.name == declaration.receive_profile_name
    )
    if len(profiles) != 1 or profiles[0] != receive.headline_receive_profile:
        raise PropagationTailDiagnosticError("diagnostic profile is not the frozen headline")
    optical = tuple(
        row
        for row in receive.source_frontier.optical_configurations
        if row.name in declaration.optical_configuration_names
    )
    if tuple(row.name for row in optical) != declaration.optical_configuration_names:
        raise PropagationTailDiagnosticError("diagnostic optical configurations have drifted")
    source_report = structural_system_dry_run(
        receive.source_frontier,
        project_root=project_root,
    )
    windows = tuple(
        window for window in source_report.windows if window.density in declaration.densities
    )
    expected_windows = receive.source_frontier.validation_windows_per_density * len(
        declaration.densities
    )
    if len(windows) != expected_windows:
        raise PropagationTailDiagnosticError("diagnostic windows do not cover the frozen grid")
    source = _source_result(
        declaration.receive_result_path,
        expected_schema=declaration.receive_result_schema,
    )
    means = _source_means(
        source,
        profile_name=declaration.receive_profile_name,
        optical_names=declaration.optical_configuration_names,
        densities=declaration.densities,
    )
    return receive, profiles[0], optical, windows, means


def structural_propagation_tail_dry_run(
    declaration: PropagationTailDeclaration,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Validate every dependency without evaluating a channel frame."""

    root = Path(project_root).expanduser().resolve(strict=False)
    receive, profile, optical, windows, means = _selected_dependencies(
        declaration,
        project_root=root,
    )
    return {
        "receive_profile": profile.name,
        "optical_configurations": [row.name for row in optical],
        "densities_vehicles_per_lane_km": list(declaration.densities),
        "validation_windows": len(windows),
        "frames": sum(window.frames for window in windows) * len(optical),
        "source_rows": len(means),
        "actor_used": False,
        "checkpoint_used": False,
        "training_run_performed": False,
        "test_split_opened": False,
        "source_frontier_sha256": receive.source_frontier.sha256,
    }


@dataclass(frozen=True, slots=True)
class PropagationTailResult:
    declaration: PropagationTailDeclaration
    receive_declaration: ReceiveDiversityFrontierDeclaration
    receive_profile: ReceiveDiversityProfile
    rows: tuple[dict[str, object], ...]
    generated_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PROPAGATION_TAIL_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "sources": {
                "receive_diversity_declaration": {
                    "path": str(self.declaration.receive_declaration_path),
                    "sha256": self.declaration.receive_declaration_sha256,
                },
                "receive_diversity_result": {
                    "path": str(self.declaration.receive_result_path),
                    "sha256": self.declaration.receive_result_sha256,
                    "schema": self.declaration.receive_result_schema,
                },
                "source_system_frontier_sha256": (self.receive_declaration.source_frontier.sha256),
            },
            "receive_profile": {
                "name": self.receive_profile.name,
                "antenna_count": self.receive_profile.antenna_count,
                "combining_rule": self.receive_profile.combining_rule,
                "branch_correlation": self.receive_profile.branch_correlation,
                "implementation_loss_db": self.receive_profile.implementation_loss_db,
            },
            "reliability_miss_budget": self.declaration.miss_budget,
            "densities_vehicles_per_lane_km": list(self.declaration.densities),
            "dimensions": list(self.declaration.dimensions),
            "actor_used": False,
            "checkpoint_used": False,
            "training_run_performed": False,
            "test_split_opened": False,
            "rows": list(self.rows),
            "decision": {
                "training_authorized": False,
                "test_split_opened": False,
                "next_intervention_must_be_chosen_from_measured_tail_mechanisms": True,
                "claim_boundary": (
                    "validation-only conditional synthetic diagnosis; not a "
                    "feasibility verdict or measured on-road reliability"
                ),
            },
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination


def execute_propagation_tail_diagnostic(
    declaration: PropagationTailDeclaration,
    *,
    project_root: str | Path,
    progress: Callable[[int, int, str], None] | None = None,
) -> PropagationTailResult:
    """Replay the frozen bounded validation windows and decompose their risk."""

    root = Path(project_root).expanduser().resolve(strict=False)
    receive, profile, optical_rows, windows, source_means = _selected_dependencies(
        declaration,
        project_root=root,
    )
    output_rows: list[dict[str, object]] = []
    for optical_index, optical in enumerate(optical_rows, start=1):
        if progress is not None:
            progress(optical_index, len(optical_rows), optical.name)
        config = load_config(
            (*receive.source_frontier.base_config_layers, *optical.additional_config_layers),
            project_root=root,
        )
        normalization = _identity_normalization(config)
        catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
        validation = {trace.trace_id: trace for trace in catalog.for_split("validation")}
        policy = _PropagationTailPolicy(declaration.densities)
        for window in windows:
            try:
                trace = validation[window.trace_id]
            except KeyError as error:
                raise PropagationTailDiagnosticError(
                    "diagnostic window is absent from validation"
                ) from error
            result = run_policy_rollout_with_state(
                config,
                trace,
                policy=policy,
                environment_seed=receive.source_frontier.environment_seed,
                policy_seed=0,
                start_frame_index=window.start_frame_index,
                max_frames=window.frames,
                normalization_state=normalization,
                sensitivity_band=receive.source_frontier.headline_point.sensing_band.band,
                collision_subchannels=(
                    receive.source_frontier.headline_point.rf_capacity.subchannels
                ),
                receive_diversity=profile.physical_profile(),
                oracle_controls_unusable_rows=False,
            )
            if result.normalization_state != normalization:
                raise PropagationTailDiagnosticError("frozen diagnostic normalization changed")
        for density in declaration.densities:
            try:
                tally = policy.tallies[density]
            except KeyError as error:
                raise PropagationTailDiagnosticError(
                    "diagnostic is missing a declared density"
                ) from error
            row = tally.as_dict(thresholds=declaration.risk_tail_thresholds)
            source_mean = source_means[(optical.name, density)]
            measured = cast(float, row["mean_selected_risk"])
            difference = measured - source_mean
            if not math.isclose(measured, source_mean, rel_tol=1e-12, abs_tol=1e-15):
                raise PropagationTailDiagnosticError(
                    "diagnostic does not reproduce the frozen propagation screen",
                    context={
                        "optical_configuration": optical.name,
                        "density": density,
                        "measured": measured,
                        "source": source_mean,
                    },
                )
            row.update(
                {
                    "optical_configuration_name": optical.name,
                    "receiver_fov_deg": optical.receiver_fov_deg,
                    "density_vehicles_per_lane_km": density,
                    "source_mean_selected_risk": source_mean,
                    "source_reproduction_difference": difference,
                    "budget_multiple": measured / declaration.miss_budget,
                    "meets_budget": measured <= declaration.miss_budget,
                }
            )
            output_rows.append(row)
    return PropagationTailResult(
        declaration=declaration,
        receive_declaration=receive,
        receive_profile=profile,
        rows=tuple(output_rows),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "EXPECTED_DIMENSIONS",
    "PROPAGATION_TAIL_DECLARATION_SCHEMA",
    "PROPAGATION_TAIL_RESULT_SCHEMA",
    "PropagationTailDeclaration",
    "PropagationTailDiagnosticError",
    "PropagationTailKey",
    "PropagationTailResult",
    "PropagationTailTally",
    "execute_propagation_tail_diagnostic",
    "load_propagation_tail_declaration",
    "structural_propagation_tail_dry_run",
]
