"""Combined 2 ms QPSK and integrated zero-loss MRC propagation frontier."""

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

from hybrid_v2x_rl.agents.deadline_edge_threshold import DEADLINE_EDGE_RESULT_SCHEMA
from hybrid_v2x_rl.agents.longer_block_execution import (
    LONGER_BLOCK_FRONTIER_RESULT_SCHEMA,
)
from hybrid_v2x_rl.agents.longer_block_frontier import (
    LongerBlockCandidate,
    LongerBlockFrontierDeclaration,
    candidate_grid_row,
    config_for_candidate,
    load_longer_block_frontier_declaration,
)
from hybrid_v2x_rl.agents.receive_diversity_execution import (
    RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA,
    propagation_only_action_risk,
)
from hybrid_v2x_rl.agents.receive_diversity_frontier import ReceiveDiversityProfile
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.env.assembly import build_lifecycle
from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout_with_state
from hybrid_v2x_rl.mean_field.frames import PopulationFrameReader, TraceCatalog
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationState,
    ObservationNormalizer,
)
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PolicyProposal,
    PopulationPolicyFrame,
)

COMBINED_FRONTIER_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.combined-receiver-block-frontier-declaration.v1"
)
COMBINED_FRONTIER_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.combined-receiver-block-frontier-result.v1"
)
COMBINED_FRONTIER_PROGRESS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.combined-receiver-block-frontier-progress.v1"
)
ProfileRole = Literal[
    "optimistic-sensitivity",
    "hardware-primary",
    "correlated-sensitivity",
]
ProgressCallback = Callable[[int, int, str], None]
CheckpointCallback = Callable[[tuple["CombinedProfileResult", ...]], None]


class CombinedReceiverBlockError(HybridV2XError):
    """The combined declaration, replay, or result has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise CombinedReceiverBlockError(f"{name} fields do not match the frozen schema")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise CombinedReceiverBlockError(f"{name} must be a nonempty array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CombinedReceiverBlockError(f"{name} must be nonempty text")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise CombinedReceiverBlockError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CombinedReceiverBlockError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CombinedReceiverBlockError(f"{name} must be a positive integer")
    return value


def _false(value: object, *, name: str) -> None:
    if value is not False:
        raise CombinedReceiverBlockError(f"{name} must be false")


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise CombinedReceiverBlockError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name)).expanduser()
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


def _json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CombinedReceiverBlockError(f"{name} is unreadable", artifact_path=path) from error
    if not isinstance(payload, Mapping):
        raise CombinedReceiverBlockError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], payload)


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    path: Path
    sha256: str
    schema: str | None


@dataclass(frozen=True, slots=True)
class CombinedReceiveProfile:
    profile: ReceiveDiversityProfile
    role: ProfileRole

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.profile.name,
            "role": self.role,
            "antenna_count": self.profile.antenna_count,
            "combining_rule": self.profile.combining_rule,
            "branch_correlation": self.profile.branch_correlation,
            "implementation_loss_db": self.profile.implementation_loss_db,
        }


@dataclass(frozen=True, slots=True)
class CombinedReceiverBlockDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    payload_bytes: int
    deadline_s: float
    miss_budget: float
    near_budget: float
    maximum_relative_excess: float
    densities: tuple[float, ...]
    receive_declaration_artifact: EvidenceArtifact
    receive_result_artifact: EvidenceArtifact
    longer_declaration_artifact: EvidenceArtifact
    longer_result_artifact: EvidenceArtifact
    deadline_declaration_artifact: EvidenceArtifact
    deadline_result_artifact: EvidenceArtifact
    longer_declaration: LongerBlockFrontierDeclaration
    candidate: LongerBlockCandidate
    receive_profiles: tuple[CombinedReceiveProfile, ...]
    optical_configuration_names: tuple[str, ...]
    selection_rule: str
    stage_rule: str
    exact_rule: str
    near_rule: str
    next_stage_rule: str
    output_path: Path


def _artifact(
    raw: object,
    *,
    root: Path,
    name: str,
    expected_schema: str | None,
) -> EvidenceArtifact:
    keys = {"path", "sha256", "schema"} if expected_schema is not None else {"path", "sha256"}
    row = _mapping(raw, name=name, keys=keys)
    schema = _text(row["schema"], name=f"{name} schema") if expected_schema else None
    if schema != expected_schema:
        raise CombinedReceiverBlockError(f"{name} schema is unsupported")
    return EvidenceArtifact(
        path=_resolve(root, row["path"], name=f"{name} path"),
        sha256=_digest(row["sha256"], name=f"{name} SHA-256"),
        schema=schema,
    )


def _validate_result_safety(artifact: EvidenceArtifact, *, name: str) -> None:
    payload = _json_mapping(artifact.path, name=name)
    if payload.get("schema") != artifact.schema:
        raise CombinedReceiverBlockError(f"{name} schema has drifted")
    if (
        payload.get("training_run_performed") is not False
        or payload.get("test_split_opened") is not False
    ):
        raise CombinedReceiverBlockError(f"{name} safety flags have drifted")


def load_combined_receiver_block_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> CombinedReceiverBlockDeclaration:
    """Load and fail-closed validate the combined physical frontier."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="combined declaration")
    top = _mapping(
        load_yaml_file(declaration_path),
        name="combined declaration",
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
    if top["schema"] != COMBINED_FRONTIER_DECLARATION_SCHEMA:
        raise CombinedReceiverBlockError("combined declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="objective",
        keys={
            "payload_bytes",
            "deadline_s",
            "miss_budget",
            "exploratory_near_budget",
            "maximum_relative_excess",
            "densities_vehicles_per_lane_km",
            "required_split",
            "interpretation",
        },
    )
    payload_bytes = _integer(objective["payload_bytes"], name="payload bytes")
    deadline_s = _number(objective["deadline_s"], name="deadline")
    miss_budget = _number(objective["miss_budget"], name="miss budget")
    near_budget = _number(objective["exploratory_near_budget"], name="near budget")
    relative_excess = _number(objective["maximum_relative_excess"], name="relative excess")
    densities = tuple(
        _number(raw, name="density")
        for raw in _sequence(objective["densities_vehicles_per_lane_km"], name="densities")
    )
    if (
        payload_bytes != 300
        or not math.isclose(deadline_s, 0.010, abs_tol=1e-15)
        or not math.isclose(miss_budget, 1e-4, abs_tol=1e-15)
        or not math.isclose(near_budget, 1.1e-4, abs_tol=1e-15)
        or not math.isclose(relative_excess, 0.10, abs_tol=1e-15)
        or not math.isclose(near_budget, miss_budget * (1.0 + relative_excess), abs_tol=1e-15)
        or densities != (10.0, 20.0, 30.0)
        or objective["required_split"] != "validation"
    ):
        raise CombinedReceiverBlockError("combined objective or near-feasible boundary has drifted")

    evidence = _mapping(
        top["evidence"],
        name="evidence",
        keys={
            "receive_diversity_declaration",
            "receive_diversity_result",
            "longer_block_declaration",
            "longer_block_result",
            "deadline_edge_declaration",
            "deadline_edge_result",
            "actor_used",
            "checkpoint_used",
        },
    )
    _false(evidence["actor_used"], name="actor_used")
    _false(evidence["checkpoint_used"], name="checkpoint_used")
    receive_declaration_artifact = _artifact(
        evidence["receive_diversity_declaration"],
        root=root,
        name="receive-diversity declaration",
        expected_schema=None,
    )
    receive_result_artifact = _artifact(
        evidence["receive_diversity_result"],
        root=root,
        name="receive-diversity result",
        expected_schema=RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA,
    )
    longer_declaration_artifact = _artifact(
        evidence["longer_block_declaration"],
        root=root,
        name="longer-block declaration",
        expected_schema=None,
    )
    longer_result_artifact = _artifact(
        evidence["longer_block_result"],
        root=root,
        name="longer-block result",
        expected_schema=LONGER_BLOCK_FRONTIER_RESULT_SCHEMA,
    )
    deadline_declaration_artifact = _artifact(
        evidence["deadline_edge_declaration"],
        root=root,
        name="deadline-edge declaration",
        expected_schema=None,
    )
    deadline_result_artifact = _artifact(
        evidence["deadline_edge_result"],
        root=root,
        name="deadline-edge result",
        expected_schema=DEADLINE_EDGE_RESULT_SCHEMA,
    )
    artifacts = (
        (receive_declaration_artifact, "receive-diversity declaration"),
        (receive_result_artifact, "receive-diversity result"),
        (longer_declaration_artifact, "longer-block declaration"),
        (longer_result_artifact, "longer-block result"),
        (deadline_declaration_artifact, "deadline-edge declaration"),
        (deadline_result_artifact, "deadline-edge result"),
    )
    if verify_evidence:
        for artifact, name in artifacts:
            if not artifact.path.is_file() or _sha256(artifact.path) != artifact.sha256:
                raise CombinedReceiverBlockError(
                    f"{name} evidence is absent or has drifted", artifact_path=artifact.path
                )
            if artifact.schema is not None:
                _validate_result_safety(artifact, name=name)
        deadline_payload = _json_mapping(deadline_result_artifact.path, name="deadline-edge result")
        decision = deadline_payload.get("decision")
        if (
            deadline_payload.get("threshold_search_complete") is not True
            or not isinstance(decision, Mapping)
            or decision.get("theoretical_deadline_boundary_passes") is not False
            or decision.get("joint_contention_frontier_authorized") is not False
        ):
            raise CombinedReceiverBlockError("deadline-edge source decision has drifted")

    longer = load_longer_block_frontier_declaration(
        longer_declaration_artifact.path,
        project_root=root,
        verify_evidence=verify_evidence,
    )
    if longer.sha256 != longer_declaration_artifact.sha256:
        raise CombinedReceiverBlockError("longer-block declaration hash has drifted")
    if longer.receive_declaration.sha256 != receive_declaration_artifact.sha256:
        raise CombinedReceiverBlockError("receive-diversity declaration hash has drifted")

    selection = _mapping(
        top["selection"],
        name="selection",
        keys={"rf_candidate", "receive_profiles", "optical_configurations", "selection_rule"},
    )
    candidate_name = _text(selection["rf_candidate"], name="RF candidate")
    candidates = tuple(
        candidate for candidate in longer.candidates if candidate.name == candidate_name
    )
    if len(candidates) != 1 or candidate_name != "qpsk-2p0ms-sensitivity":
        raise CombinedReceiverBlockError("combined RF candidate has drifted")
    expected_profile_rows = (
        ("rx2-mrc__independent-ideal__integrated-zero-loss", "optimistic-sensitivity"),
        ("rx2-mrc__low-correlation-hardware-bound__integrated-zero-loss", "hardware-primary"),
        ("rx2-mrc__correlated-stress__integrated-zero-loss", "correlated-sensitivity"),
    )
    raw_profiles = _sequence(selection["receive_profiles"], name="receive profiles")
    parsed_rows: list[tuple[str, str]] = []
    for index, raw in enumerate(raw_profiles):
        row = _mapping(raw, name=f"receive profile {index}", keys={"name", "role"})
        parsed_rows.append(
            (_text(row["name"], name="profile name"), _text(row["role"], name="profile role"))
        )
    if tuple(parsed_rows) != expected_profile_rows:
        raise CombinedReceiverBlockError("combined receive-profile grid has drifted")
    by_name = {profile.name: profile for profile in longer.receive_declaration.receive_profiles}
    combined_profiles: list[CombinedReceiveProfile] = []
    for name, role in parsed_rows:
        if name not in by_name or role not in {
            "optimistic-sensitivity",
            "hardware-primary",
            "correlated-sensitivity",
        }:
            raise CombinedReceiverBlockError("combined receive profile is unsupported")
        profile = by_name[name]
        if (
            profile.implementation_loss_db != 0.0
            or profile.combining_rule != "maximum-ratio-combining"
        ):
            raise CombinedReceiverBlockError(
                "combined receive profile is not integrated zero-loss MRC"
            )
        combined_profiles.append(
            CombinedReceiveProfile(profile=profile, role=cast(ProfileRole, role))
        )
    optical_names = tuple(
        _text(raw, name="optical configuration")
        for raw in _sequence(selection["optical_configurations"], name="optical configurations")
    )
    if optical_names != longer.optical_configuration_names:
        raise CombinedReceiverBlockError("combined optical grid has drifted")
    expected_selection_rule = (
        "minimize worst-density mean risk, then mean risk, receive-profile order, and optical order"
    )
    if selection["selection_rule"] != expected_selection_rule:
        raise CombinedReceiverBlockError("combined selection rule has drifted")

    execution = _mapping(
        top["execution"],
        name="execution",
        keys={
            "stage",
            "exact_rule",
            "exploratory_near_rule",
            "next_stage_rule",
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
        or execution["stage"]
        != "evaluate every profile, optical configuration, density, and frozen validation window before selection"
    ):
        raise CombinedReceiverBlockError("combined execution contract has drifted")
    decision_section = _mapping(
        top["decision"],
        name="decision",
        keys={
            "exact_target_remains_primary",
            "exploratory_near_path_is_not_feasibility",
            "training_authorization",
            "sensitivity_profiles_may_be_selected_only_under_the_frozen_rule",
            "claim_boundary",
        },
    )
    if (
        decision_section["exact_target_remains_primary"] is not True
        or decision_section["exploratory_near_path_is_not_feasibility"] is not True
        or decision_section["sensitivity_profiles_may_be_selected_only_under_the_frozen_rule"]
        is not True
    ):
        raise CombinedReceiverBlockError("combined decision boundary has drifted")
    _false(decision_section["training_authorization"], name="training authorization")

    return CombinedReceiverBlockDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        payload_bytes=payload_bytes,
        deadline_s=deadline_s,
        miss_budget=miss_budget,
        near_budget=near_budget,
        maximum_relative_excess=relative_excess,
        densities=densities,
        receive_declaration_artifact=receive_declaration_artifact,
        receive_result_artifact=receive_result_artifact,
        longer_declaration_artifact=longer_declaration_artifact,
        longer_result_artifact=longer_result_artifact,
        deadline_declaration_artifact=deadline_declaration_artifact,
        deadline_result_artifact=deadline_result_artifact,
        longer_declaration=longer,
        candidate=candidates[0],
        receive_profiles=tuple(combined_profiles),
        optical_configuration_names=optical_names,
        selection_rule=expected_selection_rule,
        stage_rule=_text(execution["stage"], name="stage rule"),
        exact_rule=_text(execution["exact_rule"], name="exact rule"),
        near_rule=_text(execution["exploratory_near_rule"], name="near rule"),
        next_stage_rule=_text(execution["next_stage_rule"], name="next-stage rule"),
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )


def _validate_trace_windows(
    config: ProjectConfig,
    *,
    windows: tuple[EvaluationWindow, ...],
) -> None:
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {source.trace_id: source for source in catalog.for_split("validation")}
    for window in windows:
        try:
            source = validation[window.trace_id]
        except KeyError as error:
            raise CombinedReceiverBlockError("combined validation window is absent") from error
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
            expected_config_scope_hashes={"mobility_trace": scope_hash(config, "mobility_trace")},
        )
        if window.start_frame_index + window.frames > reader.decision_frame_count:
            raise CombinedReceiverBlockError("combined validation window exceeds its trace")


def structural_combined_receiver_block_dry_run(
    declaration: CombinedReceiverBlockDeclaration,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Validate the complete combined grid without evaluating channel frames."""

    root = Path(project_root).expanduser().resolve(strict=False)
    source = structural_system_dry_run(
        declaration.longer_declaration.receive_declaration.source_frontier,
        project_root=root,
    )
    grid = candidate_grid_row(
        declaration.longer_declaration,
        declaration.candidate,
        project_root=root,
    )
    if (
        grid["modulation"] != "qpsk"
        or grid["slots_per_attempt"] != 4
        or grid["finite_blocklength_channel_uses"] != 9676
        or grid["rf4_fits_deadline"] is not True
    ):
        raise CombinedReceiverBlockError("combined RF grid has drifted")
    configs: list[dict[str, object]] = []
    for optical in declaration.longer_declaration.optical_configurations:
        config = config_for_candidate(
            declaration.longer_declaration,
            declaration.candidate,
            optical,
            project_root=root,
        )
        _validate_trace_windows(config, windows=source.windows)
        for combined_profile in declaration.receive_profiles:
            build_lifecycle(
                config,
                receive_diversity=combined_profile.profile.physical_profile(),
            )
            configs.append(
                {
                    "receive_profile_name": combined_profile.profile.name,
                    "role": combined_profile.role,
                    "optical_configuration_name": optical.name,
                    "config_hash": config_hash(config),
                }
            )
    return {
        "schema": COMBINED_FRONTIER_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "payload_bytes": declaration.payload_bytes,
        "deadline_s": declaration.deadline_s,
        "exact_miss_budget": declaration.miss_budget,
        "exploratory_near_budget": declaration.near_budget,
        "receive_profiles": len(declaration.receive_profiles),
        "optical_configurations": len(declaration.optical_configuration_names),
        "densities": len(declaration.densities),
        "validation_windows": len(source.windows),
        "physical_profile_instances": len(configs),
        "evaluation_rows": len(configs) * len(declaration.densities),
        "rf_resource_grid": grid,
        "configurations": configs,
        "channel_frames_evaluated": 0,
        "training_authorized": False,
        "test_split_opened": False,
    }


@dataclass(slots=True)
class _PropagationTally:
    frames: int = 0
    transitions: int = 0
    risk_sum: float = 0.0
    action_counts: list[int] = field(default_factory=lambda: [0] * len(PolicyAction))

    def observe(self, actions: tuple[PolicyAction, ...], risks: tuple[float, ...]) -> None:
        if len(actions) != len(risks) or not actions:
            raise CombinedReceiverBlockError("combined tally requires aligned nonempty rows")
        self.frames += 1
        self.transitions += len(actions)
        self.risk_sum += math.fsum(risks)
        for action in actions:
            self.action_counts[int(action)] += 1


@dataclass(slots=True)
class _CombinedOracle:
    tallies: dict[float, _PropagationTally] = field(default_factory=dict)
    name: str = "combined-receiver-block-propagation-oracle"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise CombinedReceiverBlockError("combined screen requires oracle truth")
        if decision.frame.source.split != "validation":
            raise CombinedReceiverBlockError("combined screen accepts validation only")
        if set(channel_truth) != set(decision.frame.active_pair_ids):
            raise CombinedReceiverBlockError("combined truth is not pair aligned")
        if decision.population_size == 0:
            return ()
        selected: list[PolicyAction] = []
        risks: list[float] = []
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
            selected.append(action)
            risks.append(rank(action)[0])
        self.tallies.setdefault(decision.frame.source.density, _PropagationTally()).observe(
            tuple(selected), tuple(risks)
        )
        return tuple(fallback if row.usable else None for row in decision.actor_frame.rows)


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if not state.frozen or any(state.count) or any(state.mean) or any(state.second_moment):
        raise CombinedReceiverBlockError("combined screen requires identity normalization")
    return state


def _screen_profile_optical(
    declaration: CombinedReceiverBlockDeclaration,
    combined_profile: CombinedReceiveProfile,
    optical_name: str,
    *,
    project_root: Path,
    windows: tuple[EvaluationWindow, ...],
) -> tuple[dict[str, object], ...]:
    matches = tuple(
        optical
        for optical in declaration.longer_declaration.optical_configurations
        if optical.name == optical_name
    )
    if len(matches) != 1:
        raise CombinedReceiverBlockError("combined optical configuration is absent")
    optical = matches[0]
    config = config_for_candidate(
        declaration.longer_declaration,
        declaration.candidate,
        optical,
        project_root=project_root,
    )
    normalization = _identity_normalization(config)
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {trace.trace_id: trace for trace in catalog.for_split("validation")}
    source = declaration.longer_declaration.receive_declaration.source_frontier
    headline = source.headline_point
    policy = _CombinedOracle()
    for window in windows:
        try:
            trace = validation[window.trace_id]
        except KeyError as error:
            raise CombinedReceiverBlockError("combined screen window is absent") from error
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
            receive_diversity=combined_profile.profile.physical_profile(),
            oracle_controls_unusable_rows=False,
        )
        if result.normalization_state != normalization:
            raise CombinedReceiverBlockError("combined normalization changed")
    rows: list[dict[str, object]] = []
    for density in declaration.densities:
        try:
            tally = policy.tallies[density]
        except KeyError as error:
            raise CombinedReceiverBlockError("combined screen is missing a density") from error
        mean = tally.risk_sum / tally.transitions
        rows.append(
            {
                "optical_configuration_name": optical.name,
                "receiver_fov_deg": optical.receiver_fov_deg,
                "density_vehicles_per_lane_km": density,
                "frames": tally.frames,
                "transitions": tally.transitions,
                "mean_optimistic_propagation_only_conditional_miss_lower_bound": mean,
                "exact_budget_multiple": mean / declaration.miss_budget,
                "near_budget_multiple": mean / declaration.near_budget,
                "meets_exact_budget": mean <= declaration.miss_budget,
                "meets_exploratory_near_budget": mean <= declaration.near_budget,
                "lower_bound_action_counts": {
                    label: tally.action_counts[index]
                    for index, label in enumerate(POLICY_ACTION_ORDER)
                },
            }
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CombinedProfileResult:
    combined_profile: CombinedReceiveProfile
    rows: tuple[dict[str, object], ...]
    exact_optical_configuration_names: tuple[str, ...]
    near_optical_configuration_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "receive_profile": self.combined_profile.as_dict(),
            "rows": list(self.rows),
            "exact_optical_configuration_names": list(self.exact_optical_configuration_names),
            "near_optical_configuration_names": list(self.near_optical_configuration_names),
        }


def _passing_optical_names(
    declaration: CombinedReceiverBlockDeclaration,
    rows: tuple[dict[str, object], ...],
    *,
    key: str,
) -> tuple[str, ...]:
    return tuple(
        optical_name
        for optical_name in declaration.optical_configuration_names
        if all(
            cast(bool, row[key])
            for row in rows
            if row["optical_configuration_name"] == optical_name
        )
    )


def _screen_profile(
    declaration: CombinedReceiverBlockDeclaration,
    combined_profile: CombinedReceiveProfile,
    *,
    project_root: Path,
    windows: tuple[EvaluationWindow, ...],
) -> CombinedProfileResult:
    rows: list[dict[str, object]] = []
    for optical_name in declaration.optical_configuration_names:
        rows.extend(
            _screen_profile_optical(
                declaration,
                combined_profile,
                optical_name,
                project_root=project_root,
                windows=windows,
            )
        )
    result_rows = tuple(rows)
    return CombinedProfileResult(
        combined_profile=combined_profile,
        rows=result_rows,
        exact_optical_configuration_names=_passing_optical_names(
            declaration, result_rows, key="meets_exact_budget"
        ),
        near_optical_configuration_names=_passing_optical_names(
            declaration, result_rows, key="meets_exploratory_near_budget"
        ),
    )


def _validate_profile_results(
    declaration: CombinedReceiverBlockDeclaration,
    results: tuple[CombinedProfileResult, ...],
    *,
    require_complete: bool,
) -> None:
    if (
        tuple(result.combined_profile for result in results)
        != declaration.receive_profiles[: len(results)]
    ):
        raise CombinedReceiverBlockError("combined results are not an ordered profile prefix")
    if require_complete and len(results) != len(declaration.receive_profiles):
        raise CombinedReceiverBlockError("combined result is incomplete")
    expected_rows = tuple(
        (optical, density)
        for optical in declaration.optical_configuration_names
        for density in declaration.densities
    )
    for result in results:
        actual_rows: list[tuple[str, float]] = []
        for row in result.rows:
            optical = row.get("optical_configuration_name")
            density = row.get("density_vehicles_per_lane_km")
            mean = row.get("mean_optimistic_propagation_only_conditional_miss_lower_bound")
            exact = row.get("meets_exact_budget")
            near = row.get("meets_exploratory_near_budget")
            if (
                not isinstance(optical, str)
                or not isinstance(density, int | float)
                or isinstance(density, bool)
                or not isinstance(mean, int | float)
                or isinstance(mean, bool)
                or not math.isfinite(float(mean))
                or not 0.0 <= float(mean) <= 1.0
                or type(exact) is not bool
                or type(near) is not bool
                or exact != (float(mean) <= declaration.miss_budget)
                or near != (float(mean) <= declaration.near_budget)
            ):
                raise CombinedReceiverBlockError("combined result row is malformed")
            actual_rows.append((optical, float(density)))
        if tuple(actual_rows) != expected_rows:
            raise CombinedReceiverBlockError("combined rows do not cover the frozen grid")
        if result.exact_optical_configuration_names != _passing_optical_names(
            declaration, result.rows, key="meets_exact_budget"
        ) or result.near_optical_configuration_names != _passing_optical_names(
            declaration, result.rows, key="meets_exploratory_near_budget"
        ):
            raise CombinedReceiverBlockError("combined profile decision does not reconcile")


def _candidate_pairs(
    declaration: CombinedReceiverBlockDeclaration,
    results: tuple[CombinedProfileResult, ...],
    *,
    exact: bool,
) -> tuple[tuple[CombinedProfileResult, str], ...]:
    pairs: list[tuple[CombinedProfileResult, str]] = []
    for result in results:
        names = (
            result.exact_optical_configuration_names
            if exact
            else result.near_optical_configuration_names
        )
        pairs.extend((result, optical) for optical in names)
    return tuple(pairs)


def _pair_rank(
    declaration: CombinedReceiverBlockDeclaration,
    pair: tuple[CombinedProfileResult, str],
) -> tuple[float, float, int, int]:
    result, optical = pair
    means = tuple(
        cast(float, row["mean_optimistic_propagation_only_conditional_miss_lower_bound"])
        for row in result.rows
        if row["optical_configuration_name"] == optical
    )
    return (
        max(means),
        math.fsum(means) / len(means),
        declaration.receive_profiles.index(result.combined_profile),
        declaration.optical_configuration_names.index(optical),
    )


@dataclass(frozen=True, slots=True)
class CombinedReceiverBlockResult:
    declaration: CombinedReceiverBlockDeclaration
    profile_results: tuple[CombinedProfileResult, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_profile_results(self.declaration, self.profile_results, require_complete=True)

    def decision(self) -> dict[str, object]:
        exact_pairs = _candidate_pairs(self.declaration, self.profile_results, exact=True)
        near_pairs = _candidate_pairs(self.declaration, self.profile_results, exact=False)
        all_pairs = tuple(
            (result, optical)
            for result in self.profile_results
            for optical in self.declaration.optical_configuration_names
        )
        best_observed = min(
            all_pairs,
            key=lambda pair: _pair_rank(self.declaration, pair),
        )
        best_observed_rank = _pair_rank(self.declaration, best_observed)
        selection_basis: str | None
        selected: tuple[CombinedProfileResult, str] | None
        if exact_pairs:
            selection_basis = "exact-feasible"
            selected = min(exact_pairs, key=lambda pair: _pair_rank(self.declaration, pair))
        elif near_pairs:
            selection_basis = "exploratory-near-feasible"
            selected = min(near_pairs, key=lambda pair: _pair_rank(self.declaration, pair))
        else:
            selection_basis = None
            selected = None
        rank = _pair_rank(self.declaration, selected) if selected is not None else None
        return {
            "exact_propagation_target_met": bool(exact_pairs),
            "exact_candidate_pairs": [
                {
                    "receive_profile_name": result.combined_profile.profile.name,
                    "optical_configuration_name": optical,
                }
                for result, optical in exact_pairs
            ],
            "exploratory_near_candidate_exists": bool(near_pairs),
            "exploratory_near_budget": self.declaration.near_budget,
            "selected_receive_profile_name": (
                selected[0].combined_profile.profile.name if selected is not None else None
            ),
            "selected_receive_profile_role": (
                selected[0].combined_profile.role if selected is not None else None
            ),
            "selected_optical_configuration_name": (selected[1] if selected is not None else None),
            "selection_basis": selection_basis,
            "selected_worst_density_mean": rank[0] if rank is not None else None,
            "selected_average_density_mean": rank[1] if rank is not None else None,
            "best_observed_receive_profile_name": (best_observed[0].combined_profile.profile.name),
            "best_observed_receive_profile_role": best_observed[0].combined_profile.role,
            "best_observed_optical_configuration_name": best_observed[1],
            "best_observed_worst_density_mean": best_observed_rank[0],
            "best_observed_average_density_mean": best_observed_rank[1],
            "best_observed_exact_budget_multiple": (
                best_observed_rank[0] / self.declaration.miss_budget
            ),
            "best_observed_near_budget_multiple": (
                best_observed_rank[0] / self.declaration.near_budget
            ),
            "joint_contention_frontier_authorized": selected is not None,
            "training_authorized": False,
            "test_split_opened": False,
            "next_action": (
                self.declaration.next_stage_rule
                if selected is not None
                else (
                    "combined receiver/block intervention exceeds the frozen 10% near margin; "
                    "stop before PPO unless the user explicitly authorizes a separately labeled "
                    "exploratory override using the best-observed pair"
                )
            ),
            "claim_boundary": (
                "an exploratory-near selection is not exact 1e-4 feasibility; "
                "all propagation results are validation-only synthetic necessary conditions"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": COMBINED_FRONTIER_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {"path": str(self.declaration.path), "sha256": self.declaration.sha256},
            "sources": {
                "receive_diversity_declaration_sha256": self.declaration.receive_declaration_artifact.sha256,
                "receive_diversity_result_sha256": self.declaration.receive_result_artifact.sha256,
                "longer_block_declaration_sha256": self.declaration.longer_declaration_artifact.sha256,
                "longer_block_result_sha256": self.declaration.longer_result_artifact.sha256,
                "deadline_edge_declaration_sha256": self.declaration.deadline_declaration_artifact.sha256,
                "deadline_edge_result_sha256": self.declaration.deadline_result_artifact.sha256,
            },
            "payload_bytes": self.declaration.payload_bytes,
            "deadline_s": self.declaration.deadline_s,
            "exact_miss_budget": self.declaration.miss_budget,
            "exploratory_near_budget": self.declaration.near_budget,
            "maximum_relative_excess": self.declaration.maximum_relative_excess,
            "densities_vehicles_per_lane_km": list(self.declaration.densities),
            "rf_candidate": self.declaration.candidate.name,
            "selection_rule": self.declaration.selection_rule,
            "actor_used": False,
            "checkpoint_used": False,
            "training_run_performed": False,
            "test_split_opened": False,
            "propagation_screen_complete": True,
            "profile_results": [result.as_dict() for result in self.profile_results],
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


def write_combined_receiver_block_progress(
    path: str | Path,
    *,
    declaration: CombinedReceiverBlockDeclaration,
    results: tuple[CombinedProfileResult, ...],
) -> Path:
    _validate_profile_results(declaration, results, require_complete=False)
    return _atomic_json_write(
        Path(path),
        {
            "schema": COMBINED_FRONTIER_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "deadline_edge_result_sha256": declaration.deadline_result_artifact.sha256,
            "training_run_performed": False,
            "test_split_opened": False,
            "profile_results": [result.as_dict() for result in results],
        },
    )


def _profile_result_from_dict(
    payload: object,
    *,
    combined_profile: CombinedReceiveProfile,
) -> CombinedProfileResult:
    if not isinstance(payload, Mapping):
        raise CombinedReceiverBlockError("progress profile result must be an object")
    expected = {
        "receive_profile",
        "rows",
        "exact_optical_configuration_names",
        "near_optical_configuration_names",
    }
    rows = payload.get("rows")
    exact_names = payload.get("exact_optical_configuration_names")
    near_names = payload.get("near_optical_configuration_names")
    if (
        set(payload) != expected
        or payload.get("receive_profile") != combined_profile.as_dict()
        or not isinstance(rows, list)
        or any(not isinstance(row, Mapping) for row in rows)
        or not isinstance(exact_names, list)
        or any(not isinstance(name, str) for name in exact_names)
        or not isinstance(near_names, list)
        or any(not isinstance(name, str) for name in near_names)
    ):
        raise CombinedReceiverBlockError("progress profile result has drifted")
    result = CombinedProfileResult(
        combined_profile=combined_profile,
        rows=tuple(dict(row) for row in rows),
        exact_optical_configuration_names=tuple(exact_names),
        near_optical_configuration_names=tuple(near_names),
    )
    if result.as_dict() != dict(payload):
        raise CombinedReceiverBlockError("progress profile result does not round trip")
    return result


def load_combined_receiver_block_progress(
    path: str | Path,
    *,
    declaration: CombinedReceiverBlockDeclaration,
) -> tuple[CombinedProfileResult, ...]:
    payload = _json_mapping(Path(path), name="combined progress")
    expected = {
        "schema",
        "declaration_sha256",
        "deadline_edge_result_sha256",
        "training_run_performed",
        "test_split_opened",
        "profile_results",
    }
    if set(payload) != expected:
        raise CombinedReceiverBlockError("combined progress fields have drifted")
    if (
        payload["schema"] != COMBINED_FRONTIER_PROGRESS_SCHEMA
        or payload["declaration_sha256"] != declaration.sha256
        or payload["deadline_edge_result_sha256"] != declaration.deadline_result_artifact.sha256
        or payload["training_run_performed"] is not False
        or payload["test_split_opened"] is not False
    ):
        raise CombinedReceiverBlockError("combined progress provenance has drifted")
    raw_results = payload["profile_results"]
    if not isinstance(raw_results, list) or len(raw_results) > len(declaration.receive_profiles):
        raise CombinedReceiverBlockError("combined progress profile prefix is malformed")
    results = tuple(
        _profile_result_from_dict(raw, combined_profile=combined_profile)
        for raw, combined_profile in zip(
            raw_results,
            declaration.receive_profiles,
            strict=False,
        )
    )
    _validate_profile_results(declaration, results, require_complete=False)
    return results


def execute_combined_receiver_block_frontier(
    declaration: CombinedReceiverBlockDeclaration,
    *,
    project_root: str | Path,
    completed_results: tuple[CombinedProfileResult, ...] = (),
    progress: ProgressCallback | None = None,
    checkpoint: CheckpointCallback | None = None,
) -> CombinedReceiverBlockResult:
    """Execute every frozen profile and return the exact/near decision."""

    root = Path(project_root).expanduser().resolve(strict=False)
    structural_combined_receiver_block_dry_run(declaration, project_root=root)
    _validate_profile_results(declaration, completed_results, require_complete=False)
    source = structural_system_dry_run(
        declaration.longer_declaration.receive_declaration.source_frontier,
        project_root=root,
    )
    results = list(completed_results)
    total = len(declaration.receive_profiles)
    for index, combined_profile in enumerate(declaration.receive_profiles, start=1):
        if index <= len(results):
            continue
        if progress is not None:
            progress(index, total, combined_profile.profile.name)
        results.append(
            _screen_profile(
                declaration,
                combined_profile,
                project_root=root,
                windows=source.windows,
            )
        )
        if checkpoint is not None:
            checkpoint(tuple(results))
    return CombinedReceiverBlockResult(
        declaration=declaration,
        profile_results=tuple(results),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "COMBINED_FRONTIER_DECLARATION_SCHEMA",
    "COMBINED_FRONTIER_PROGRESS_SCHEMA",
    "COMBINED_FRONTIER_RESULT_SCHEMA",
    "CombinedProfileResult",
    "CombinedReceiveProfile",
    "CombinedReceiverBlockDeclaration",
    "CombinedReceiverBlockError",
    "CombinedReceiverBlockResult",
    "execute_combined_receiver_block_frontier",
    "load_combined_receiver_block_declaration",
    "load_combined_receiver_block_progress",
    "structural_combined_receiver_block_dry_run",
    "write_combined_receiver_block_progress",
]
