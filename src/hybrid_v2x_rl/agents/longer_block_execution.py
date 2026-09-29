"""Resumable propagation-only execution for the longer-block RF frontier."""

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

from hybrid_v2x_rl.agents.longer_block_frontier import (
    LongerBlockCandidate,
    LongerBlockFrontierDeclaration,
    config_for_candidate,
    structural_longer_block_dry_run,
)
from hybrid_v2x_rl.agents.receive_diversity_execution import (
    propagation_only_action_risk,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
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

LONGER_BLOCK_FRONTIER_RESULT_SCHEMA: Final = "hybrid-rf-vlc-rl.longer-block-rf-frontier-result.v1"
LONGER_BLOCK_FRONTIER_PROGRESS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.longer-block-rf-frontier-progress.v1"
)
ProgressCallback = Callable[[int, int, str], None]
CheckpointCallback = Callable[[tuple["LongerBlockCandidateResult", ...]], None]


class LongerBlockExecutionError(HybridV2XError):
    """Longer-block execution or persisted progress is inconsistent."""


def _candidate_dict(candidate: LongerBlockCandidate) -> dict[str, object]:
    return {
        "name": candidate.name,
        "role": candidate.role,
        "modulation": candidate.modulation,
        "airtime_s": candidate.airtime_s,
        "gross_bit_rate_bps": candidate.gross_bit_rate_bps,
        "code_rate": candidate.code_rate,
        "expected_slots": candidate.expected_slots,
        "expected_available_coded_bits": candidate.expected_available_coded_bits,
        "expected_channel_uses": candidate.expected_channel_uses,
        "expected_rf4_airtime_s": candidate.expected_rf4_airtime_s,
    }


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


def _load_json(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LongerBlockExecutionError(f"{name} is unreadable", artifact_path=path) from error
    if not isinstance(payload, Mapping):
        raise LongerBlockExecutionError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], payload)


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
            raise LongerBlockExecutionError("propagation screen requires aligned nonempty rows")
        self.frames += 1
        self.transitions += len(actions)
        self.risk_sum += math.fsum(risks)
        for action in actions:
            self.action_counts[int(action)] += 1


@dataclass(slots=True)
class _LongerBlockOracle:
    tallies: dict[float, _PropagationTally] = field(default_factory=dict)
    name: str = "longer-block-propagation-only-oracle"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise LongerBlockExecutionError("propagation screen requires oracle truth")
        if decision.frame.source.split != "validation":
            raise LongerBlockExecutionError("propagation screen accepts validation only")
        if set(channel_truth) != set(decision.frame.active_pair_ids):
            raise LongerBlockExecutionError("oracle truth is not pair aligned")
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
                    pair_truth, "vlc_result"
                ):
                    raise LongerBlockExecutionError("oracle truth is malformed")
                risk = propagation_only_action_risk(
                    action,
                    rf_decoding_failure_probability=(
                        pair_truth.rf_propagation.decoding_failure_probability
                    ),
                    vlc_failure_probability=(pair_truth.vlc_result.total_failure_probability),
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
        self.tallies.setdefault(decision.frame.source.density, _PropagationTally()).observe(
            tuple(selected), tuple(risks)
        )
        # Propagation truth is action independent.  Use the fixed contract
        # fallback in the actual rollout so every candidate has the same
        # causal-history path while the oracle choices remain tally-only.
        return tuple(fallback if row.usable else None for row in decision.actor_frame.rows)


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if not state.frozen or any(state.count) or any(state.mean) or any(state.second_moment):
        raise LongerBlockExecutionError("screen requires identity normalization")
    return state


@dataclass(frozen=True, slots=True)
class LongerBlockCandidateResult:
    candidate: LongerBlockCandidate
    execution_mode: str
    rows: tuple[dict[str, object], ...]
    passing_optical_configuration_names: tuple[str, ...]

    @property
    def survives(self) -> bool:
        return bool(self.passing_optical_configuration_names)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate": _candidate_dict(self.candidate),
            "execution_mode": self.execution_mode,
            "criterion": (
                "survives only when at least one frozen optical configuration "
                "has mean propagation-only conditional miss risk <= miss budget "
                "at every required density"
            ),
            "rows": list(self.rows),
            "passing_optical_configuration_names": list(self.passing_optical_configuration_names),
            "survives": self.survives,
        }


def _validate_candidate_results(
    declaration: LongerBlockFrontierDeclaration,
    results: tuple[LongerBlockCandidateResult, ...],
    *,
    require_complete: bool,
) -> None:
    expected_candidates = declaration.candidates
    if tuple(result.candidate for result in results) != expected_candidates[: len(results)]:
        raise LongerBlockExecutionError("results are not an ordered candidate prefix")
    if require_complete and len(results) != len(expected_candidates):
        raise LongerBlockExecutionError("result does not cover the complete candidate grid")
    expected_rows = tuple(
        (optical_name, density)
        for optical_name in declaration.optical_configuration_names
        for density in declaration.densities
    )
    for result in results:
        actual_rows: list[tuple[str, float]] = []
        passing: list[str] = []
        for row in result.rows:
            optical = row.get("optical_configuration_name")
            density = row.get("density_vehicles_per_lane_km")
            mean = row.get("mean_optimistic_propagation_only_conditional_miss_lower_bound")
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
                raise LongerBlockExecutionError("candidate result row is malformed")
            actual_rows.append((optical, float(density)))
        if tuple(actual_rows) != expected_rows:
            raise LongerBlockExecutionError("candidate rows do not cover the frozen grid")
        for optical_name in declaration.optical_configuration_names:
            optical_rows = tuple(
                row for row in result.rows if row["optical_configuration_name"] == optical_name
            )
            if all(cast(bool, row["meets_budget"]) for row in optical_rows):
                passing.append(optical_name)
        if tuple(passing) != result.passing_optical_configuration_names:
            raise LongerBlockExecutionError("candidate survival decision does not reconcile")


def _control_result(
    declaration: LongerBlockFrontierDeclaration,
) -> LongerBlockCandidateResult:
    payload = _load_json(
        declaration.receive_result.path,
        name="receive-diversity source result",
    )
    screens = payload.get("propagation_screen")
    if not isinstance(screens, list):
        raise LongerBlockExecutionError("source result has no propagation screen")
    matches = tuple(
        row
        for row in screens
        if isinstance(row, Mapping)
        and isinstance(row.get("receive_profile"), Mapping)
        and row["receive_profile"].get("name") == declaration.receive_profile_name
    )
    if len(matches) != 1:
        raise LongerBlockExecutionError("source result lacks the frozen headline profile")
    source = matches[0]
    raw_rows = source.get("rows")
    raw_passing = source.get("passing_optical_configuration_names")
    if (
        not isinstance(raw_rows, list)
        or any(not isinstance(row, Mapping) for row in raw_rows)
        or not isinstance(raw_passing, list)
        or any(not isinstance(name, str) for name in raw_passing)
    ):
        raise LongerBlockExecutionError("source control rows are malformed")
    result = LongerBlockCandidateResult(
        candidate=declaration.candidates[0],
        execution_mode="reused-hash-verified-receive-diversity-control",
        rows=tuple(dict(row) for row in raw_rows),
        passing_optical_configuration_names=tuple(raw_passing),
    )
    _validate_candidate_results(declaration, (result,), require_complete=False)
    return result


def _screen_candidate_optical(
    declaration: LongerBlockFrontierDeclaration,
    candidate: LongerBlockCandidate,
    optical_name: str,
    *,
    project_root: Path,
    windows: tuple[EvaluationWindow, ...],
) -> tuple[dict[str, object], ...]:
    matches = tuple(
        optical for optical in declaration.optical_configurations if optical.name == optical_name
    )
    if len(matches) != 1:
        raise LongerBlockExecutionError("optical configuration is not frozen")
    optical = matches[0]
    config = config_for_candidate(
        declaration,
        candidate,
        optical,
        project_root=project_root,
    )
    normalization = _identity_normalization(config)
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {trace.trace_id: trace for trace in catalog.for_split("validation")}
    source = declaration.receive_declaration.source_frontier
    headline = source.headline_point
    policy = _LongerBlockOracle()
    for window in windows:
        try:
            trace = validation[window.trace_id]
        except KeyError as error:
            raise LongerBlockExecutionError(
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
            receive_diversity=declaration.receive_profile.physical_profile(),
            oracle_controls_unusable_rows=False,
        )
        if result.normalization_state != normalization:
            raise LongerBlockExecutionError("screen normalization changed")
    rows: list[dict[str, object]] = []
    for density in declaration.densities:
        try:
            tally = policy.tallies[density]
        except KeyError as error:
            raise LongerBlockExecutionError("screen is missing a declared density") from error
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


def _screen_candidate(
    declaration: LongerBlockFrontierDeclaration,
    candidate: LongerBlockCandidate,
    *,
    project_root: Path,
    windows: tuple[EvaluationWindow, ...],
) -> LongerBlockCandidateResult:
    rows: list[dict[str, object]] = []
    for optical_name in declaration.optical_configuration_names:
        rows.extend(
            _screen_candidate_optical(
                declaration,
                candidate,
                optical_name,
                project_root=project_root,
                windows=windows,
            )
        )
    passing = tuple(
        optical_name
        for optical_name in declaration.optical_configuration_names
        if all(
            cast(bool, row["meets_budget"])
            for row in rows
            if row["optical_configuration_name"] == optical_name
        )
    )
    return LongerBlockCandidateResult(
        candidate=candidate,
        execution_mode="validation-replay",
        rows=tuple(rows),
        passing_optical_configuration_names=passing,
    )


@dataclass(frozen=True, slots=True)
class LongerBlockFrontierResult:
    declaration: LongerBlockFrontierDeclaration
    candidate_results: tuple[LongerBlockCandidateResult, ...]
    resource_grids: tuple[dict[str, object], ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_candidate_results(self.declaration, self.candidate_results, require_complete=True)
        if len(self.resource_grids) != len(self.declaration.candidates):
            raise LongerBlockExecutionError("resource-grid evidence is incomplete")

    def decision(self) -> dict[str, object]:
        survivors = tuple(result for result in self.candidate_results if result.survives)
        selected = survivors[0] if survivors else None
        return {
            "surviving_candidate_names": [result.candidate.name for result in survivors],
            "shortest_passing_candidate_name": (
                selected.candidate.name if selected is not None else None
            ),
            "shortest_passing_airtime_s": (
                selected.candidate.airtime_s if selected is not None else None
            ),
            "propagation_necessary_condition_met": selected is not None,
            "joint_contention_frontier_authorized": selected is not None,
            "training_authorized": False,
            "test_split_opened": False,
            "next_action": (
                self.declaration.next_stage_rule
                if selected is not None
                else "longer blocklength alone is insufficient; select a new physical intervention"
            ),
            "claim_boundary": (
                "validation-only conditional synthetic propagation evidence; "
                "passing is necessary but not sufficient for system feasibility"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": LONGER_BLOCK_FRONTIER_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "sources": {
                "propagation_tail_declaration_sha256": (self.declaration.tail_declaration.sha256),
                "propagation_tail_result_sha256": self.declaration.tail_result.sha256,
                "receive_diversity_declaration_sha256": (
                    self.declaration.receive_declaration_artifact.sha256
                ),
                "receive_diversity_result_sha256": (self.declaration.receive_result.sha256),
            },
            "payload_bytes": self.declaration.payload_bytes,
            "deadline_s": self.declaration.deadline_s,
            "predecision_lead_s": self.declaration.predecision_lead_s,
            "reliability_miss_budget": self.declaration.miss_budget,
            "densities_vehicles_per_lane_km": list(self.declaration.densities),
            "receive_profile": self.declaration.receive_profile_name,
            "actor_used": False,
            "checkpoint_used": False,
            "training_run_performed": False,
            "test_split_opened": False,
            "propagation_screen_complete": True,
            "control_reuse_rule": self.declaration.control_reuse_rule,
            "selection_rule": self.declaration.selection_rule,
            "resource_grids": list(self.resource_grids),
            "candidate_results": [result.as_dict() for result in self.candidate_results],
            "decision": self.decision(),
        }

    def write_json(self, path: str | Path) -> Path:
        return _atomic_json_write(Path(path), self.as_dict())


def write_longer_block_progress(
    path: str | Path,
    *,
    declaration: LongerBlockFrontierDeclaration,
    results: tuple[LongerBlockCandidateResult, ...],
) -> Path:
    """Persist a declaration-bound ordered candidate prefix atomically."""

    _validate_candidate_results(declaration, results, require_complete=False)
    return _atomic_json_write(
        Path(path),
        {
            "schema": LONGER_BLOCK_FRONTIER_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "propagation_tail_result_sha256": declaration.tail_result.sha256,
            "receive_diversity_result_sha256": declaration.receive_result.sha256,
            "training_run_performed": False,
            "test_split_opened": False,
            "candidate_results": [result.as_dict() for result in results],
        },
    )


def _result_from_dict(
    payload: object,
    *,
    candidate: LongerBlockCandidate,
) -> LongerBlockCandidateResult:
    if not isinstance(payload, Mapping):
        raise LongerBlockExecutionError("progress candidate must be an object")
    expected = {
        "candidate",
        "execution_mode",
        "criterion",
        "rows",
        "passing_optical_configuration_names",
        "survives",
    }
    raw_rows = payload.get("rows")
    raw_passing = payload.get("passing_optical_configuration_names")
    if (
        set(payload) != expected
        or payload.get("candidate") != _candidate_dict(candidate)
        or not isinstance(payload.get("execution_mode"), str)
        or not isinstance(raw_rows, list)
        or any(not isinstance(row, Mapping) for row in raw_rows)
        or not isinstance(raw_passing, list)
        or any(not isinstance(name, str) for name in raw_passing)
    ):
        raise LongerBlockExecutionError("progress candidate differs from declaration")
    result = LongerBlockCandidateResult(
        candidate=candidate,
        execution_mode=cast(str, payload["execution_mode"]),
        rows=tuple(dict(row) for row in raw_rows),
        passing_optical_configuration_names=tuple(raw_passing),
    )
    if result.as_dict() != dict(payload):
        raise LongerBlockExecutionError("progress candidate result has drifted")
    return result


def load_longer_block_progress(
    path: str | Path,
    *,
    declaration: LongerBlockFrontierDeclaration,
) -> tuple[LongerBlockCandidateResult, ...]:
    """Restore only an exact, declaration-bound ordered candidate prefix."""

    payload = _load_json(Path(path), name="longer-block progress")
    expected = {
        "schema",
        "declaration_sha256",
        "propagation_tail_result_sha256",
        "receive_diversity_result_sha256",
        "training_run_performed",
        "test_split_opened",
        "candidate_results",
    }
    if set(payload) != expected:
        raise LongerBlockExecutionError("progress fields do not match the schema")
    if (
        payload["schema"] != LONGER_BLOCK_FRONTIER_PROGRESS_SCHEMA
        or payload["declaration_sha256"] != declaration.sha256
        or payload["propagation_tail_result_sha256"] != declaration.tail_result.sha256
        or payload["receive_diversity_result_sha256"] != declaration.receive_result.sha256
        or payload["training_run_performed"] is not False
        or payload["test_split_opened"] is not False
    ):
        raise LongerBlockExecutionError("progress provenance has drifted")
    raw_results = payload["candidate_results"]
    if not isinstance(raw_results, list) or len(raw_results) > len(declaration.candidates):
        raise LongerBlockExecutionError("progress candidate prefix is malformed")
    results = tuple(
        _result_from_dict(raw, candidate=candidate)
        for raw, candidate in zip(
            raw_results,
            declaration.candidates,
            strict=False,
        )
    )
    _validate_candidate_results(declaration, results, require_complete=False)
    return results


def execute_longer_block_frontier(
    declaration: LongerBlockFrontierDeclaration,
    *,
    project_root: str | Path,
    completed_results: tuple[LongerBlockCandidateResult, ...] = (),
    progress: ProgressCallback | None = None,
    checkpoint: CheckpointCallback | None = None,
) -> LongerBlockFrontierResult:
    """Run every undeclared-result candidate and retain an ordered checkpoint."""

    root = Path(project_root).expanduser().resolve(strict=False)
    dry_run = structural_longer_block_dry_run(declaration, project_root=root)
    _validate_candidate_results(declaration, completed_results, require_complete=False)
    results = list(completed_results)
    if not results:
        results.append(_control_result(declaration))
        if checkpoint is not None:
            checkpoint(tuple(results))
    source_report = structural_system_dry_run(
        declaration.receive_declaration.source_frontier,
        project_root=root,
    )
    total = len(declaration.candidates)
    for index, candidate in enumerate(declaration.candidates, start=1):
        if index <= len(results):
            continue
        if progress is not None:
            progress(index, total, candidate.name)
        results.append(
            _screen_candidate(
                declaration,
                candidate,
                project_root=root,
                windows=source_report.windows,
            )
        )
        if checkpoint is not None:
            checkpoint(tuple(results))
    grids = dry_run.get("resource_grids")
    if not isinstance(grids, list) or any(not isinstance(row, Mapping) for row in grids):
        raise LongerBlockExecutionError("dry-run resource-grid evidence is malformed")
    return LongerBlockFrontierResult(
        declaration=declaration,
        candidate_results=tuple(results),
        resource_grids=tuple(dict(row) for row in grids),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "LONGER_BLOCK_FRONTIER_PROGRESS_SCHEMA",
    "LONGER_BLOCK_FRONTIER_RESULT_SCHEMA",
    "LongerBlockCandidateResult",
    "LongerBlockExecutionError",
    "LongerBlockFrontierResult",
    "execute_longer_block_frontier",
    "load_longer_block_progress",
    "write_longer_block_progress",
]
