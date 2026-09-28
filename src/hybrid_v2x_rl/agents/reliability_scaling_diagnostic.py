"""Predeclared RF-decoding reliability-scaling lower-bound diagnostic."""

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
from typing import Final, Literal, cast

from hybrid_v2x_rl.agents.regime_evaluation import (
    EvaluationWindow,
    load_frozen_regime_audit,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    FrontierPhysicalPoint,
    SystemFeasibilityFrontierDeclaration,
    load_system_feasibility_frontier_declaration,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_config, load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout_with_state
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

RF_RELIABILITY_SCALING_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.rf-decoding-reliability-scaling-declaration.v1"
)
RF_RELIABILITY_SCALING_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.rf-decoding-reliability-scaling-result.v1"
)
_FRONTIER_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-result.v1"
)

ScalingMode = Literal["contract", "all_usable"]
ProgressCallback = Callable[[int, int, EvaluationWindow], None]


class ReliabilityScalingError(HybridV2XError):
    """A reliability-scaling declaration or result is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ReliabilityScalingError(f"{name} fields do not match the schema")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise ReliabilityScalingError(f"{name} must be an array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReliabilityScalingError(f"{name} must be a nonempty string")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ReliabilityScalingError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ReliabilityScalingError(f"{name} must be finite and positive")
    return result


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ReliabilityScalingError(f"{name} must be a positive integer")
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ReliabilityScalingError(f"{name} must be boolean")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ReliabilityScalingError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name))
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class ScalingView:
    name: str
    mode: ScalingMode


@dataclass(frozen=True, slots=True)
class ReliabilityScalingDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    miss_budget: float
    densities: tuple[float, ...]
    frontier: SystemFeasibilityFrontierDeclaration
    frontier_result_path: Path
    frontier_result_sha256: str
    anchor_point: FrontierPhysicalPoint
    anchor_diagnostic_cell_id: str
    baseline_lower_bounds: tuple[float, ...]
    factors: tuple[float, ...]
    views: tuple[ScalingView, ...]
    output_path: Path


def _frontier_result(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
    declaration: SystemFeasibilityFrontierDeclaration,
    anchor_cell_id: str,
) -> tuple[float, ...]:
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise ReliabilityScalingError(
            "system-frontier result is absent or its digest has drifted",
            artifact_path=path,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReliabilityScalingError(
            "system-frontier result is unreadable",
            artifact_path=path,
        ) from error
    if not isinstance(payload, Mapping):
        raise ReliabilityScalingError("system-frontier result must be an object")
    decision = payload.get("decision")
    declared = payload.get("declaration")
    if (
        payload.get("schema") != expected_schema
        or expected_schema != _FRONTIER_RESULT_SCHEMA
        or payload.get("frontier_complete") is not True
        or payload.get("test_split_opened") is not False
        or not isinstance(decision, Mapping)
        or decision.get("training_authorized") is not False
        or not isinstance(declared, Mapping)
        or declared.get("sha256") != declaration.sha256
    ):
        raise ReliabilityScalingError(
            "system-frontier result does not preserve the required failed gate"
        )
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ReliabilityScalingError("system-frontier result cells must be an array")
    matches = [cell for cell in cells if isinstance(cell, Mapping) and cell.get("cell_id") == anchor_cell_id]
    if len(matches) != 1:
        raise ReliabilityScalingError("anchor diagnostic cell is absent or duplicated")
    cell = matches[0]
    fallback = cell.get("fallback_view")
    rows = cell.get("densities")
    if (
        not isinstance(fallback, Mapping)
        or fallback.get("mode") != "all_usable"
        or not isinstance(rows, list)
    ):
        raise ReliabilityScalingError("anchor cell is not the all-usable diagnostic")
    by_density: dict[float, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReliabilityScalingError("anchor density row must be an object")
        density = _number(row.get("density_vehicles_per_lane_km"), name="anchor density")
        lower = _number(
            row.get("mean_certified_conditional_miss_lower_bound"),
            name="anchor lower bound",
        )
        by_density[density] = lower
    if tuple(sorted(by_density)) != declaration.densities:
        raise ReliabilityScalingError("anchor density coverage differs from the declaration")
    return tuple(by_density[density] for density in declaration.densities)


def load_reliability_scaling_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> ReliabilityScalingDeclaration:
    """Load the frozen diagnostic without evaluating a channel or opening test."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="scaling declaration")
    payload = load_yaml_file(declaration_path)
    top = _mapping(
        payload,
        name="scaling declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "intervention",
            "views",
            "execution",
            "decision",
        },
    )
    if top["schema"] != RF_RELIABILITY_SCALING_DECLARATION_SCHEMA:
        raise ReliabilityScalingError("scaling declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="scaling objective",
        keys={
            "miss_budget",
            "densities_vehicles_per_lane_km",
            "required_split",
            "test_split_opened",
            "interpretation",
        },
    )
    miss_budget = _number(objective["miss_budget"], name="miss budget")
    densities = tuple(
        _number(value, name="density")
        for value in _sequence(
            objective["densities_vehicles_per_lane_km"],
            name="densities",
        )
    )
    if (
        miss_budget >= 1.0
        or densities != tuple(sorted(set(densities)))
        or objective["required_split"] != "validation"
        or _boolean(objective["test_split_opened"], name="test-split flag")
    ):
        raise ReliabilityScalingError("objective must be ordered validation-only data")
    _text(objective["interpretation"], name="objective interpretation")

    evidence = _mapping(
        top["evidence"],
        name="scaling evidence",
        keys={
            "system_frontier_declaration",
            "system_frontier_result",
            "anchor_physical_point_id",
            "anchor_diagnostic_cell_id",
            "actor_used",
            "checkpoint_used",
        },
    )
    if (
        _boolean(evidence["actor_used"], name="actor-used flag")
        or _boolean(evidence["checkpoint_used"], name="checkpoint-used flag")
    ):
        raise ReliabilityScalingError("diagnostic cannot use an actor or checkpoint")
    source_declaration = _mapping(
        evidence["system_frontier_declaration"],
        name="source frontier declaration",
        keys={"path", "sha256"},
    )
    frontier_path = _resolve(root, source_declaration["path"], name="frontier declaration")
    expected_frontier_sha = _digest(
        source_declaration["sha256"],
        name="frontier declaration SHA-256",
    )
    frontier = load_system_feasibility_frontier_declaration(
        frontier_path,
        project_root=root,
        verify_evidence=verify_evidence,
        # This diagnostic is frozen to the completed, now-superseded
        # two-12-RB interpretation. It remains evidence for the abstract
        # zero-contention propagation divisor, but cannot be executed as the
        # current physical frontier.
        enforce_current_headline=False,
    )
    if frontier.sha256 != expected_frontier_sha:
        raise ReliabilityScalingError("source frontier declaration digest has drifted")
    if frontier.miss_budget != miss_budget or frontier.densities != densities:
        raise ReliabilityScalingError("scaling objective differs from the source frontier")

    anchor_point_id = _text(
        evidence["anchor_physical_point_id"],
        name="anchor physical point",
    )
    points = [point for point in frontier.physical_points if point.point_id == anchor_point_id]
    if len(points) != 1:
        raise ReliabilityScalingError("anchor physical point is absent or duplicated")
    anchor_point = points[0]
    anchor_cell_id = _text(
        evidence["anchor_diagnostic_cell_id"],
        name="anchor diagnostic cell",
    )
    expected_cell_id = f"{anchor_point.point_id}__all-rows-oracle-controlled"
    if anchor_cell_id != expected_cell_id:
        raise ReliabilityScalingError("anchor diagnostic cell does not match the physical point")

    result_source = _mapping(
        evidence["system_frontier_result"],
        name="source frontier result",
        keys={"path", "sha256", "schema"},
    )
    result_path = _resolve(root, result_source["path"], name="frontier result")
    result_sha = _digest(result_source["sha256"], name="frontier result SHA-256")
    result_schema = _text(result_source["schema"], name="frontier result schema")
    baseline = (
        _frontier_result(
            result_path,
            expected_sha256=result_sha,
            expected_schema=result_schema,
            declaration=frontier,
            anchor_cell_id=anchor_cell_id,
        )
        if verify_evidence
        else tuple(0.0 for _ in densities)
    )

    intervention = _mapping(
        top["intervention"],
        name="scaling intervention",
        keys={"name", "factors", "semantics", "unchanged", "nonphysical_diagnostic"},
    )
    if (
        intervention["name"] != "rf-decoding-failure-divisor"
        or not _boolean(
            intervention["nonphysical_diagnostic"],
            name="nonphysical diagnostic flag",
        )
    ):
        raise ReliabilityScalingError("only the declared diagnostic intervention is allowed")
    _text(intervention["semantics"], name="intervention semantics")
    _text(intervention["unchanged"], name="unchanged mechanisms")
    factors = tuple(
        _number(value, name="improvement factor")
        for value in _sequence(intervention["factors"], name="improvement factors")
    )
    if not factors or factors[0] != 1.0 or factors != tuple(sorted(set(factors))):
        raise ReliabilityScalingError("factors must be unique, increasing, and start at one")

    raw_views = _sequence(top["views"], name="scaling views")
    views: list[ScalingView] = []
    for raw_view in raw_views:
        view = _mapping(raw_view, name="scaling view", keys={"name", "mode"})
        name = _text(view["name"], name="view name")
        mode = _text(view["mode"], name="view mode")
        if mode not in {"contract", "all_usable"}:
            raise ReliabilityScalingError("scaling view mode is unsupported")
        views.append(ScalingView(name=name, mode=cast(ScalingMode, mode)))
    if tuple((view.name, view.mode) for view in views) != (
        ("contract-dup4", "contract"),
        ("all-rows-oracle-controlled", "all_usable"),
    ):
        raise ReliabilityScalingError("scaling views must preserve the frozen order")

    execution = _mapping(
        top["execution"],
        name="scaling execution",
        keys={
            "expected_factors",
            "expected_views",
            "expected_density_rows",
            "no_joint_action_search",
            "no_training",
            "no_test_split",
            "output_path",
        },
    )
    for flag in ("no_joint_action_search", "no_training", "no_test_split"):
        if not _boolean(execution[flag], name=flag):
            raise ReliabilityScalingError("scaling execution safety flags must be true")
    if (
        _integer(execution["expected_factors"], name="expected factors") != len(factors)
        or _integer(execution["expected_views"], name="expected views") != len(views)
        or _integer(execution["expected_density_rows"], name="expected rows")
        != len(factors) * len(views) * len(densities)
    ):
        raise ReliabilityScalingError("declared scaling grid size does not reconcile")

    decision = _mapping(
        top["decision"],
        name="scaling decision",
        keys={"threshold_rule", "interpretation", "training_authorization"},
    )
    _text(decision["threshold_rule"], name="threshold rule")
    _text(decision["interpretation"], name="decision interpretation")
    if _boolean(decision["training_authorization"], name="training authorization"):
        raise ReliabilityScalingError("a diagnostic cannot authorize training")

    return ReliabilityScalingDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        miss_budget=miss_budget,
        densities=densities,
        frontier=frontier,
        frontier_result_path=result_path,
        frontier_result_sha256=result_sha,
        anchor_point=anchor_point,
        anchor_diagnostic_cell_id=anchor_cell_id,
        baseline_lower_bounds=baseline,
        factors=factors,
        views=tuple(views),
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )


def scaled_action_risk(
    action: PolicyAction,
    *,
    rf_decoding_failure_probability: float,
    vlc_failure_probability: float,
    improvement_factor: float,
) -> float:
    """Return an optimistic action risk after scaling RF decoding only."""

    for name, value in (
        ("RF decoding failure", rf_decoding_failure_probability),
        ("VLC failure", vlc_failure_probability),
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ReliabilityScalingError(f"{name} probability must lie in [0, 1]")
    factor = _number(improvement_factor, name="improvement factor")
    spec = action_resources(action)
    risk = 1.0
    if spec.uses_rf:
        risk *= (rf_decoding_failure_probability / factor) ** spec.reserved_rf_attempts
    if spec.uses_vlc:
        risk *= vlc_failure_probability
    return float(risk)


@dataclass(slots=True)
class _ScalingTally:
    frames: int = 0
    transitions: int = 0
    risk_sum: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * len(PolicyAction))

    def observe(self, actions: tuple[PolicyAction, ...], risks: tuple[float, ...]) -> None:
        if len(actions) != len(risks) or not actions:
            raise ReliabilityScalingError("scaling tally requires aligned nonempty rows")
        self.frames += 1
        self.transitions += len(actions)
        self.risk_sum += math.fsum(risks)
        for action in actions:
            self.action_counts[int(action)] += 1


@dataclass(slots=True)
class _ScalingPolicy:
    factors: tuple[float, ...]
    views: tuple[ScalingView, ...]
    tallies: dict[tuple[str, float, float], _ScalingTally] = field(default_factory=dict)
    name: str = "rf-decoding-reliability-scaling-oracle"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise ReliabilityScalingError("scaling diagnostic requires channel truth")
        if decision.frame.source.split != "validation":
            raise ReliabilityScalingError("scaling diagnostic accepts validation only")
        if set(channel_truth) != set(decision.frame.active_pair_ids):
            raise ReliabilityScalingError("scaling channel truth is not pair aligned")
        if decision.population_size == 0:
            return ()
        density = decision.frame.source.density
        allowed = decision.action_space.mask.allowed_actions
        fallback = decision.action_space.fallback_action
        for view in self.views:
            for factor in self.factors:
                selected: list[PolicyAction] = []
                risks: list[float] = []
                for actor_row in decision.actor_frame.rows:
                    truth = channel_truth[actor_row.pair_id]
                    choices = (
                        allowed
                        if view.mode == "all_usable" or actor_row.usable
                        else (fallback,)
                    )

                    def rank(
                        action: PolicyAction,
                        *,
                        pair_truth: object = truth,
                        divisor: float = factor,
                    ) -> tuple[float, float, int, int]:
                        if not hasattr(pair_truth, "rf_propagation") or not hasattr(
                            pair_truth,
                            "vlc_result",
                        ):
                            raise ReliabilityScalingError(
                                "scaling channel truth has an invalid pair result"
                            )
                        risk = scaled_action_risk(
                            action,
                            rf_decoding_failure_probability=(
                                pair_truth.rf_propagation.decoding_failure_probability
                            ),
                            vlc_failure_probability=(
                                pair_truth.vlc_result.total_failure_probability
                            ),
                            improvement_factor=divisor,
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
                    (view.name, factor, density),
                    _ScalingTally(),
                ).observe(tuple(selected), tuple(risks))
        return tuple(
            fallback if row.usable else None for row in decision.actor_frame.rows
        )


@dataclass(frozen=True, slots=True)
class ReliabilityScalingResult:
    declaration: ReliabilityScalingDeclaration
    config: ProjectConfig
    windows: tuple[EvaluationWindow, ...]
    rows: tuple[dict[str, object], ...]
    baseline_reconciled: bool
    generated_at_utc: datetime

    def decision(self) -> dict[str, object]:
        minimum: dict[str, float | None] = {}
        brackets: dict[str, dict[str, float | None]] = {}
        for view in self.declaration.views:
            passing = [
                factor
                for factor in self.declaration.factors
                if all(
                    cast(float, row["mean_optimistic_conditional_miss_lower_bound"])
                    <= self.declaration.miss_budget
                    for row in self.rows
                    if row["view"] == view.name and row["improvement_factor"] == factor
                )
            ]
            selected = min(passing) if passing else None
            minimum[view.name] = selected
            previous = None
            if selected is not None:
                selected_index = self.declaration.factors.index(selected)
                if selected_index:
                    previous = self.declaration.factors[selected_index - 1]
            brackets[view.name] = {
                "last_failing_factor": previous,
                "first_passing_factor": selected,
            }
        return {
            "minimum_declared_factor_by_view": minimum,
            "factor_bracket_by_view": brackets,
            "training_authorized": False,
            "test_split_opened": False,
            "interpretation": (
                "necessary RF-decoding improvement under an optimistic lower bound; "
                "not sufficient for a realizable policy"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        point = self.declaration.anchor_point
        return {
            "schema": RF_RELIABILITY_SCALING_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "source_system_frontier": {
                "path": str(self.declaration.frontier_result_path),
                "sha256": self.declaration.frontier_result_sha256,
                "anchor_diagnostic_cell_id": self.declaration.anchor_diagnostic_cell_id,
                "baseline_reconciled": self.baseline_reconciled,
            },
            "scope": (
                "RF decoding failure divisor with zero RF contention and zero "
                "receiver half-duplex risk"
            ),
            "reliability_miss_budget": self.declaration.miss_budget,
            "test_split_opened": False,
            "actor_used": False,
            "checkpoint_used": False,
            "joint_action_search_used": False,
            "training_authorized": False,
            "config_hash": config_hash(self.config),
            "policy_environment_scope_hash": scope_hash(
                self.config,
                "policy_environment",
            ),
            "anchor_physical_point": {
                "point_id": point.point_id,
                "rf_subchannels": point.rf_capacity.subchannels,
                "sensing_band": point.sensing_band.name,
                "optical_configuration": point.optical_configuration.name,
                "receiver_fov_deg": point.optical_configuration.receiver_fov_deg,
            },
            "improvement_factors": list(self.declaration.factors),
            "views": [
                {"name": view.name, "mode": view.mode}
                for view in self.declaration.views
            ],
            "windows": [window.as_dict() for window in self.windows],
            "rows": list(self.rows),
            "decision": self.decision(),
        }

    def write_json(self, path: str | Path) -> Path:
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


def _config(declaration: ReliabilityScalingDeclaration) -> ProjectConfig:
    point = declaration.anchor_point
    return load_config(
        (
            *declaration.frontier.base_config_layers,
            *point.optical_configuration.additional_config_layers,
        ),
        project_root=declaration.path.parents[2],
    )


def _windows(
    declaration: ReliabilityScalingDeclaration,
) -> tuple[EvaluationWindow, ...]:
    _, windows, environment_seed, source_sha = load_frozen_regime_audit(
        declaration.frontier.window_source,
        expected_policy_environment_scope_hash=(
            declaration.frontier.baseline_policy_environment_scope_hash
        ),
    )
    if (
        environment_seed != declaration.frontier.environment_seed
        or source_sha != declaration.frontier.window_source_sha256
        or tuple(sorted({window.density for window in windows}))
        != declaration.densities
        or len(windows)
        != len(declaration.densities)
        * declaration.frontier.validation_windows_per_density
        or any(window.frames != declaration.frontier.frames_per_window for window in windows)
    ):
        raise ReliabilityScalingError("frozen scaling windows have drifted")
    return windows


def structural_scaling_dry_run(
    declaration: ReliabilityScalingDeclaration,
) -> dict[str, object]:
    """Validate every dependency without evaluating a channel frame."""

    config = _config(declaration)
    windows = _windows(declaration)
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {source.trace_id: source for source in catalog.for_split("validation")}
    if any(window.trace_id not in validation for window in windows):
        raise ReliabilityScalingError("a frozen scaling window is absent from validation")
    normalizer = ObservationNormalizer.from_config(config).freeze()
    if not normalizer.frozen or any(normalizer.count):
        raise ReliabilityScalingError("scaling diagnostic requires identity normalization")
    return {
        "schema": RF_RELIABILITY_SCALING_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "validation_windows": len(windows),
        "improvement_factors": list(declaration.factors),
        "views": len(declaration.views),
        "density_rows": (
            len(declaration.factors)
            * len(declaration.views)
            * len(declaration.densities)
        ),
        "channel_frames_evaluated": 0,
        "joint_action_search_used": False,
        "training_authorized": False,
        "test_split_opened": False,
    }


def execute_reliability_scaling_diagnostic(
    declaration: ReliabilityScalingDeclaration,
    *,
    progress: ProgressCallback | None = None,
) -> ReliabilityScalingResult:
    """Replay frozen validation truth once and evaluate every declared divisor."""

    _ = structural_scaling_dry_run(declaration)
    config = _config(declaration)
    windows = _windows(declaration)
    normalization: ObservationNormalizationState = ObservationNormalizer.from_config(
        config
    ).freeze()
    point = declaration.anchor_point
    policy = _ScalingPolicy(
        factors=declaration.factors,
        views=declaration.views,
    )
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {source.trace_id: source for source in catalog.for_split("validation")}
    for index, window in enumerate(windows, start=1):
        if progress is not None:
            progress(index, len(windows), window)
        result = run_policy_rollout_with_state(
            config,
            validation[window.trace_id],
            policy=policy,
            environment_seed=declaration.frontier.environment_seed,
            policy_seed=0,
            start_frame_index=window.start_frame_index,
            max_frames=window.frames,
            normalization_state=normalization,
            sensitivity_band=point.sensing_band.band,
            collision_subchannels=point.rf_capacity.subchannels,
            oracle_controls_unusable_rows=False,
        )
        if result.normalization_state != normalization:
            raise ReliabilityScalingError("frozen scaling normalization changed")

    rows: list[dict[str, object]] = []
    for view in declaration.views:
        for factor in declaration.factors:
            for density in declaration.densities:
                try:
                    tally = policy.tallies[(view.name, factor, density)]
                except KeyError as error:
                    raise ReliabilityScalingError(
                        "scaling result is missing a declared row"
                    ) from error
                mean = tally.risk_sum / tally.transitions
                rows.append(
                    {
                        "view": view.name,
                        "mode": view.mode,
                        "improvement_factor": factor,
                        "density_vehicles_per_lane_km": density,
                        "frames": tally.frames,
                        "transitions": tally.transitions,
                        "mean_optimistic_conditional_miss_lower_bound": mean,
                        "budget_multiple": mean / declaration.miss_budget,
                        "meets_budget": mean <= declaration.miss_budget,
                        "lower_bound_action_counts": {
                            name: tally.action_counts[index]
                            for index, name in enumerate(POLICY_ACTION_ORDER)
                        },
                    }
                )

    baseline_rows = [
        row
        for row in rows
        if row["view"] == "all-rows-oracle-controlled"
        and row["improvement_factor"] == 1.0
    ]
    baseline_reconciled = len(baseline_rows) == len(declaration.densities) and all(
        math.isclose(
            cast(float, row["mean_optimistic_conditional_miss_lower_bound"]),
            declaration.baseline_lower_bounds[index],
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for index, row in enumerate(baseline_rows)
    )
    if not baseline_reconciled:
        raise ReliabilityScalingError(
            "factor-one diagnostic does not reproduce the frozen frontier baseline"
        )
    return ReliabilityScalingResult(
        declaration=declaration,
        config=config,
        windows=windows,
        rows=tuple(rows),
        baseline_reconciled=True,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "RF_RELIABILITY_SCALING_DECLARATION_SCHEMA",
    "RF_RELIABILITY_SCALING_RESULT_SCHEMA",
    "ReliabilityScalingDeclaration",
    "ReliabilityScalingError",
    "ReliabilityScalingResult",
    "ScalingView",
    "execute_reliability_scaling_diagnostic",
    "load_reliability_scaling_declaration",
    "scaled_action_risk",
    "structural_scaling_dry_run",
]
