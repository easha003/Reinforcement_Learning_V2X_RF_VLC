"""Matched legacy-global versus current pair-local RF-accounting A/B.

The experiment replays one authoritative pair-local rollout and reconstructs
the superseded frame-global collision and population-mean half-duplex risk
from the same committed action ledger and the same RF/VLC channel truth.  It
therefore changes accounting only; it is neither a policy comparison nor a
3 ms versus 10 ms comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from hybrid_v2x_rl.agents.exploratory_joint_override import (
    ExploratoryJointOverrideDeclaration,
    load_exploratory_joint_override_declaration,
)
from hybrid_v2x_rl.agents.longer_block_frontier import config_for_candidate
from hybrid_v2x_rl.agents.regime_evaluation import (
    EvaluationWindow,
    load_frozen_regime_audit,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    load_system_feasibility_frontier_declaration,
)
from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.channels.rf.diversity import RFReceiveDiversity
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_config, load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    run_policy_rollout_with_state,
)
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizer
from hybrid_v2x_rl.mean_field.packet_outcomes import FramePacketOutcomes
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PopulationPolicyFrame,
)
from hybrid_v2x_rl.mean_field.return_boundaries import FrameReturnBoundary
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel

GLOBAL_LOCAL_AB_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.global-local-accounting-ab-declaration.v1"
)
GLOBAL_LOCAL_AB_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.global-local-accounting-ab-result.v1"
)
GLOBAL_LOCAL_AB_PROGRESS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.global-local-accounting-ab-progress.v1"
)
_COMPARISON_TOLERANCE: Final = 1e-15

ProgressCallback = Callable[[int, int, str], None]
CheckpointCallback = Callable[[tuple[dict[str, object], ...]], None]


class GlobalLocalAccountingABError(HybridV2XError):
    """The declaration, matched replay, or persisted result has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise GlobalLocalAccountingABError(
            f"{name} fields do not match the frozen schema"
        )
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise GlobalLocalAccountingABError(f"{name} must be a nonempty array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GlobalLocalAccountingABError(f"{name} must be nonempty text")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise GlobalLocalAccountingABError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GlobalLocalAccountingABError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise GlobalLocalAccountingABError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise GlobalLocalAccountingABError(f"{name} must be boolean")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise GlobalLocalAccountingABError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name)).expanduser()
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ABProfileDeclaration:
    key: str
    profile_id: str
    deadline_s: float
    rf_capacity_name: str
    subchannels: int
    candidate_resources: int
    optical_configuration: str
    receive_profile: str
    historical_capacity_caveat: str | None = None


@dataclass(frozen=True, slots=True)
class GlobalLocalAccountingABDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    payload_bytes: int
    deadlines_s: tuple[float, ...]
    miss_budget: float
    three_ms_source: EvidenceArtifact
    ten_ms_source: EvidenceArtifact
    window_source: EvidenceArtifact
    environment_seed: int
    actions: tuple[str, ...]
    fallback_action: str
    sensing_band: SensitivityBand
    legacy_global_sensed_fraction: float
    pair_local_contention_radius_m: float
    profiles: tuple[ABProfileDeclaration, ...]
    expected_windows: int
    expected_profiles: int
    expected_cells: int
    output_path: Path

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{profile.profile_id}__{action.label}"
            for profile in self.profiles
            for action in PolicyAction
        )


def _artifact(value: object, *, root: Path, name: str) -> EvidenceArtifact:
    row = _mapping(value, name=name, keys={"path", "sha256"})
    return EvidenceArtifact(
        path=_resolve(root, row["path"], name=f"{name} path"),
        sha256=_digest(row["sha256"], name=f"{name} SHA-256"),
    )


def _profile(value: object, *, key: str) -> ABProfileDeclaration:
    required = {
        "profile_id",
        "deadline_s",
        "rf_capacity_name",
        "subchannels",
        "candidate_resources",
        "optical_configuration",
        "receive_profile",
    }
    if key == "three_ms":
        required.add("historical_capacity_caveat")
    row = _mapping(value, name=key, keys=required)
    return ABProfileDeclaration(
        key=key,
        profile_id=_text(row["profile_id"], name=f"{key} profile ID"),
        deadline_s=_number(row["deadline_s"], name=f"{key} deadline"),
        rf_capacity_name=_text(row["rf_capacity_name"], name=f"{key} RF capacity"),
        subchannels=_integer(row["subchannels"], name=f"{key} subchannels"),
        candidate_resources=_integer(
            row["candidate_resources"], name=f"{key} candidate resources"
        ),
        optical_configuration=_text(
            row["optical_configuration"], name=f"{key} optical configuration"
        ),
        receive_profile=_text(row["receive_profile"], name=f"{key} receive profile"),
        historical_capacity_caveat=(
            _text(row["historical_capacity_caveat"], name="historical capacity caveat")
            if key == "three_ms"
            else None
        ),
    )


def load_global_local_accounting_ab_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> GlobalLocalAccountingABDeclaration:
    """Load the fail-closed matched-accounting declaration."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="A/B declaration")
    top = _mapping(
        load_yaml_file(declaration_path),
        name="A/B declaration",
        keys={"schema", "frozen_date", "objective", "evidence", "design", "profiles", "execution"},
    )
    if top["schema"] != GLOBAL_LOCAL_AB_DECLARATION_SCHEMA:
        raise GlobalLocalAccountingABError("A/B declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="objective",
        keys={"payload_bytes", "deadlines_s", "miss_budget", "required_split", "interpretation"},
    )
    deadlines = tuple(
        _number(value, name="deadline")
        for value in _sequence(objective["deadlines_s"], name="deadlines")
    )
    if (
        _integer(objective["payload_bytes"], name="payload bytes") != 300
        or deadlines != (0.003, 0.010)
        or not math.isclose(_number(objective["miss_budget"], name="miss budget"), 1e-4, abs_tol=1e-15)
        or objective["required_split"] != "validation"
    ):
        raise GlobalLocalAccountingABError("A/B objective has drifted")

    evidence = _mapping(
        top["evidence"],
        name="evidence",
        keys={
            "three_ms_source",
            "ten_ms_source",
            "window_source",
            "environment_seed",
            "actor_used",
            "checkpoint_used",
        },
    )
    if _boolean(evidence["actor_used"], name="actor_used") or _boolean(
        evidence["checkpoint_used"], name="checkpoint_used"
    ):
        raise GlobalLocalAccountingABError("A/B analysis cannot use an actor or checkpoint")
    three_source = _artifact(evidence["three_ms_source"], root=root, name="3 ms source")
    ten_source = _artifact(evidence["ten_ms_source"], root=root, name="10 ms source")
    window_source = _artifact(evidence["window_source"], root=root, name="window source")

    design = _mapping(
        top["design"],
        name="design",
        keys={
            "actions",
            "fallback_action",
            "sensing_band",
            "legacy_global_sensed_fraction",
            "pair_local_contention_radius_m",
            "identical_joint_actions",
            "identical_channel_truth",
            "conditional_risk_only",
            "usable_and_all_row_views",
        },
    )
    actions = tuple(_text(value, name="action") for value in _sequence(design["actions"], name="actions"))
    flags = (
        design["identical_joint_actions"],
        design["identical_channel_truth"],
        design["conditional_risk_only"],
        design["usable_and_all_row_views"],
    )
    if actions != POLICY_ACTION_ORDER or any(flag is not True for flag in flags):
        raise GlobalLocalAccountingABError("matched A/B design has drifted")
    try:
        sensing_band = SensitivityBand(_text(design["sensing_band"], name="sensing band"))
    except ValueError as error:
        raise GlobalLocalAccountingABError("sensing band is unsupported") from error
    sensed_fraction = _number(
        design["legacy_global_sensed_fraction"], name="legacy sensed fraction"
    )
    radius_m = _number(
        design["pair_local_contention_radius_m"], name="contention radius"
    )
    if (
        design["fallback_action"] != "DUP-4"
        or sensing_band is not SensitivityBand.NOMINAL
        or not math.isclose(sensed_fraction, 1.0, abs_tol=0.0)
        or not math.isclose(radius_m, 200.0, abs_tol=1e-12)
    ):
        raise GlobalLocalAccountingABError("accounting parameters have drifted")

    profiles_raw = _mapping(top["profiles"], name="profiles", keys={"three_ms", "ten_ms"})
    profiles = tuple(_profile(profiles_raw[key], key=key) for key in ("three_ms", "ten_ms"))
    if tuple(profile.deadline_s for profile in profiles) != deadlines:
        raise GlobalLocalAccountingABError("profile deadlines differ from the objective")

    execution = _mapping(
        top["execution"],
        name="execution",
        keys={
            "validation_windows",
            "expected_profiles",
            "expected_cells",
            "no_training",
            "no_test_split",
            "output_path",
        },
    )
    expected_profiles = _integer(execution["expected_profiles"], name="expected profiles")
    expected_cells = _integer(execution["expected_cells"], name="expected cells")
    if (
        execution["no_training"] is not True
        or execution["no_test_split"] is not True
        or expected_profiles != len(profiles)
        or expected_cells != len(profiles) * len(PolicyAction)
    ):
        raise GlobalLocalAccountingABError("execution boundary has drifted")

    declaration = GlobalLocalAccountingABDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        payload_bytes=300,
        deadlines_s=deadlines,
        miss_budget=1e-4,
        three_ms_source=three_source,
        ten_ms_source=ten_source,
        window_source=window_source,
        environment_seed=_integer(evidence["environment_seed"], name="environment seed"),
        actions=actions,
        fallback_action="DUP-4",
        sensing_band=sensing_band,
        legacy_global_sensed_fraction=sensed_fraction,
        pair_local_contention_radius_m=radius_m,
        profiles=profiles,
        expected_windows=_integer(execution["validation_windows"], name="validation windows"),
        expected_profiles=expected_profiles,
        expected_cells=expected_cells,
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )
    if verify_evidence:
        for artifact in (three_source, ten_source, window_source):
            if not artifact.path.is_file() or _sha256(artifact.path) != artifact.sha256:
                raise GlobalLocalAccountingABError(
                    "A/B evidence is absent or has drifted", artifact_path=artifact.path
                )
    return declaration


@dataclass(frozen=True, slots=True)
class _ExecutionProfile:
    declaration: ABProfileDeclaration
    config: ProjectConfig
    subchannels: int
    receive_diversity: RFReceiveDiversity | None


def _ten_ms_config(
    declaration: ExploratoryJointOverrideDeclaration,
    *,
    project_root: Path,
) -> ProjectConfig:
    optical = tuple(
        row
        for row in declaration.combined_declaration.longer_declaration.optical_configurations
        if row.name == declaration.selected_optical_name
    )
    if len(optical) != 1:
        raise GlobalLocalAccountingABError("10 ms optical configuration is absent")
    return config_for_candidate(
        declaration.combined_declaration.longer_declaration,
        declaration.combined_declaration.candidate,
        optical[0],
        project_root=project_root,
    )


def _execution_context(
    declaration: GlobalLocalAccountingABDeclaration,
    *,
    project_root: Path,
) -> tuple[tuple[_ExecutionProfile, ...], tuple[EvaluationWindow, ...]]:
    three = load_system_feasibility_frontier_declaration(
        declaration.three_ms_source.path,
        project_root=project_root,
        verify_evidence=True,
        enforce_current_headline=False,
    )
    ten = load_exploratory_joint_override_declaration(
        declaration.ten_ms_source.path,
        project_root=project_root,
        verify_evidence=True,
    )
    _, windows, environment_seed, window_sha = load_frozen_regime_audit(
        declaration.window_source.path,
        expected_policy_environment_scope_hash=three.baseline_policy_environment_scope_hash,
    )
    if (
        environment_seed != declaration.environment_seed
        or environment_seed != three.environment_seed
        or window_sha != declaration.window_source.sha256
        or len(windows) != declaration.expected_windows
    ):
        raise GlobalLocalAccountingABError("validation-window provenance has drifted")

    three_point = three.headline_point
    three_config = load_config(
        (*three.base_config_layers, *three_point.optical_configuration.additional_config_layers),
        project_root=project_root,
    )
    ten_config = _ten_ms_config(ten, project_root=project_root)
    ten_cells = tuple(
        cell
        for cell in ten.selected_cells
        if cell.physical_point.sensing_band.name == ten.selected_training_sensing_band
    )
    if len(ten_cells) != 1:
        raise GlobalLocalAccountingABError("10 ms nominal cell is not unique")
    ten_point = ten_cells[0].physical_point
    receive_diversity = ten.selected_profile.profile.physical_profile()
    profiles = (
        _ExecutionProfile(declaration.profiles[0], three_config, three_point.rf_capacity.subchannels, None),
        _ExecutionProfile(declaration.profiles[1], ten_config, ten_point.rf_capacity.subchannels, receive_diversity),
    )
    source_seed = ten.combined_declaration.longer_declaration.receive_declaration.source_frontier.environment_seed
    if source_seed != declaration.environment_seed:
        raise GlobalLocalAccountingABError("3 ms and 10 ms environment seeds differ")
    for profile in profiles:
        declared = profile.declaration
        parameters = profile.subchannels * 200
        action_space = MaskedActionSpace.from_config(
            profile.config.environment, profile.config.rf, profile.config.vlc
        )
        if (
            profile.config.service.payload_bytes != declaration.payload_bytes
            or not math.isclose(profile.config.service.deadline_s, declared.deadline_s, abs_tol=1e-15)
            or not math.isclose(profile.config.service.miss_budget, declaration.miss_budget, abs_tol=1e-15)
            or profile.subchannels != declared.subchannels
            or parameters != declared.candidate_resources
            or action_space.fallback_action.label != declaration.fallback_action
            or action_space.mask.allowed_names != declaration.actions
        ):
            raise GlobalLocalAccountingABError(
                f"effective profile {declared.profile_id} differs from its declaration"
            )
    if three_point.rf_capacity.name != declaration.profiles[0].rf_capacity_name:
        raise GlobalLocalAccountingABError("3 ms capacity identity has drifted")
    if ten_point.rf_capacity.name != declaration.profiles[1].rf_capacity_name:
        raise GlobalLocalAccountingABError("10 ms capacity identity has drifted")
    if receive_diversity is None or ten.selected_profile.profile.name != declaration.profiles[1].receive_profile:
        raise GlobalLocalAccountingABError("10 ms receive profile has drifted")
    return profiles, windows


def structural_global_local_accounting_ab_dry_run(
    declaration: GlobalLocalAccountingABDeclaration,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Validate profiles, frozen windows, and trace replay without evaluation."""

    root = Path(project_root).expanduser().resolve(strict=False)
    profiles, windows = _execution_context(declaration, project_root=root)
    plans: list[dict[str, object]] = []
    for profile in profiles:
        catalog = TraceCatalog.from_splits(
            profile.config.paths.trace_root, profile.config.environment.splits
        )
        validation_ids = {source.trace_id for source in catalog.for_split("validation")}
        missing = tuple(window.trace_id for window in windows if window.trace_id not in validation_ids)
        if missing:
            raise GlobalLocalAccountingABError(
                "a frozen validation trace is absent", context={"trace_ids": missing}
            )
        frozen = ObservationNormalizer.from_config(profile.config).freeze()
        if not frozen.frozen or any(frozen.count):
            raise GlobalLocalAccountingABError("A/B analysis requires identity normalization")
        plans.append(
            {
                "profile_id": profile.declaration.profile_id,
                "deadline_s": profile.declaration.deadline_s,
                "config_hash": config_hash(profile.config),
                "policy_environment_scope_hash": scope_hash(profile.config, "policy_environment"),
                "subchannels": profile.subchannels,
                "candidate_resources": profile.subchannels * 200,
                "receive_diversity": (
                    profile.receive_diversity.as_dict()
                    if profile.receive_diversity is not None
                    else {"profile": "siso"}
                ),
                "cells": [f"{profile.declaration.profile_id}__{action.label}" for action in PolicyAction],
            }
        )
    return {
        "schema": GLOBAL_LOCAL_AB_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "validation_windows": len(windows),
        "profiles": plans,
        "expected_cells": declaration.expected_cells,
        "channel_frames_evaluated": 0,
        "training_performed": False,
        "test_split_opened": False,
    }


@dataclass(frozen=True, slots=True)
class _FixedActionPolicy:
    action: PolicyAction

    @property
    def name(self) -> str:
        return f"fixed-{self.action.label}"

    @property
    def requires_oracle_truth(self) -> bool:
        return False

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyAction | None, ...]:
        if channel_truth is not None:
            raise GlobalLocalAccountingABError("fixed deployable policy received oracle truth")
        return tuple(
            self.action if row.usable else None for row in decision.actor_frame.rows
        )


def packet_conditional_risk(
    action: PolicyAction,
    *,
    rf_attempt_failure_probability: float | None,
    vlc_miss_probability: float | None,
) -> float:
    """Compose packet risk from a per-attempt RF risk and the unchanged VLC leg."""

    spec = action_resources(action)
    risk = 1.0
    if spec.uses_rf:
        if (
            rf_attempt_failure_probability is None
            or not math.isfinite(rf_attempt_failure_probability)
            or not 0.0 <= rf_attempt_failure_probability <= 1.0
        ):
            raise GlobalLocalAccountingABError("RF action requires a valid attempt risk")
        risk *= rf_attempt_failure_probability ** spec.reserved_rf_attempts
    elif rf_attempt_failure_probability is not None:
        raise GlobalLocalAccountingABError("VLC-only action cannot carry RF risk")
    if spec.uses_vlc:
        if (
            vlc_miss_probability is None
            or not math.isfinite(vlc_miss_probability)
            or not 0.0 <= vlc_miss_probability <= 1.0
        ):
            raise GlobalLocalAccountingABError("VLC action requires a valid VLC risk")
        risk *= vlc_miss_probability
    elif vlc_miss_probability is not None:
        raise GlobalLocalAccountingABError("RF-only action cannot carry VLC risk")
    return risk


def _mean(total: float, count: int) -> float | None:
    return total / count if count else None


@dataclass(slots=True)
class _MatchedTally:
    rows: int = 0
    usable_rows: int = 0
    fallback_rows: int = 0
    rf_rows: int = 0
    usable_rf_rows: int = 0
    global_risk_sum: float = 0.0
    local_risk_sum: float = 0.0
    usable_global_risk_sum: float = 0.0
    usable_local_risk_sum: float = 0.0
    global_collision_sum: float = 0.0
    local_collision_sum: float = 0.0
    global_half_duplex_sum: float = 0.0
    local_half_duplex_sum: float = 0.0
    global_access_sum: float = 0.0
    local_access_sum: float = 0.0
    global_utilization_sum: float = 0.0
    local_utilization_sum: float = 0.0
    usable_global_collision_sum: float = 0.0
    usable_local_collision_sum: float = 0.0
    usable_global_half_duplex_sum: float = 0.0
    usable_local_half_duplex_sum: float = 0.0
    usable_global_access_sum: float = 0.0
    usable_local_access_sum: float = 0.0
    usable_global_utilization_sum: float = 0.0
    usable_local_utilization_sum: float = 0.0
    local_better_rows: int = 0
    global_better_rows: int = 0
    equal_rows: int = 0
    usable_local_better_rows: int = 0
    usable_global_better_rows: int = 0
    usable_equal_rows: int = 0
    global_target_rows: int = 0
    local_target_rows: int = 0
    usable_global_target_rows: int = 0
    usable_local_target_rows: int = 0
    reserved_rf_attempts: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)

    def observe(
        self,
        *,
        usable: bool,
        action: PolicyAction,
        global_risk: float,
        local_risk: float,
        miss_budget: float,
        global_mechanisms: tuple[float, float, float, float] | None,
        local_mechanisms: tuple[float, float, float, float] | None,
    ) -> None:
        self.rows += 1
        self.usable_rows += int(usable)
        self.fallback_rows += int(not usable)
        self.global_risk_sum += global_risk
        self.local_risk_sum += local_risk
        self.global_target_rows += int(global_risk <= miss_budget)
        self.local_target_rows += int(local_risk <= miss_budget)
        self.action_counts[action.label] += 1
        self.reserved_rf_attempts += action_resources(action).reserved_rf_attempts
        if local_risk < global_risk - _COMPARISON_TOLERANCE:
            self.local_better_rows += 1
            if usable:
                self.usable_local_better_rows += 1
        elif global_risk < local_risk - _COMPARISON_TOLERANCE:
            self.global_better_rows += 1
            if usable:
                self.usable_global_better_rows += 1
        else:
            self.equal_rows += 1
            if usable:
                self.usable_equal_rows += 1
        if usable:
            self.usable_global_risk_sum += global_risk
            self.usable_local_risk_sum += local_risk
            self.usable_global_target_rows += int(global_risk <= miss_budget)
            self.usable_local_target_rows += int(local_risk <= miss_budget)
        if global_mechanisms is None or local_mechanisms is None:
            if global_mechanisms is not None or local_mechanisms is not None:
                raise GlobalLocalAccountingABError("RF mechanism rows are not paired")
            return
        self.rf_rows += 1
        if usable:
            self.usable_rf_rows += 1
        for prefix, values in (("global", global_mechanisms), ("local", local_mechanisms)):
            for suffix, value in zip(
                ("collision", "half_duplex", "access", "utilization"), values, strict=True
            ):
                setattr(self, f"{prefix}_{suffix}_sum", getattr(self, f"{prefix}_{suffix}_sum") + value)
                if usable:
                    name = f"usable_{prefix}_{suffix}_sum"
                    setattr(self, name, getattr(self, name) + value)

    def merge(self, other: _MatchedTally) -> None:
        for name in self.__dataclass_fields__:
            if name == "action_counts":
                self.action_counts.update(other.action_counts)
            else:
                setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self, *, density: float | None, miss_budget: float) -> dict[str, object]:
        global_mean = _mean(self.global_risk_sum, self.rows)
        local_mean = _mean(self.local_risk_sum, self.rows)
        usable_global = _mean(self.usable_global_risk_sum, self.usable_rows)
        usable_local = _mean(self.usable_local_risk_sum, self.usable_rows)

        def delta(local: float | None, global_: float | None) -> float | None:
            return None if local is None or global_ is None else local - global_

        def reduction(local: float | None, global_: float | None) -> float | None:
            if local is None or global_ is None or global_ == 0.0:
                return None
            return (global_ - local) / global_

        row: dict[str, object] = {
            "rows": self.rows,
            "usable_rows": self.usable_rows,
            "fallback_rows": self.fallback_rows,
            "rf_rows": self.rf_rows,
            "usable_rf_rows": self.usable_rf_rows,
            "action_counts": dict(sorted(self.action_counts.items())),
            "reserved_rf_attempts": self.reserved_rf_attempts,
            "all_rows": {
                "legacy_global_mean_conditional_miss_risk": global_mean,
                "pair_local_mean_conditional_miss_risk": local_mean,
                "pair_local_minus_global": delta(local_mean, global_mean),
                "relative_risk_reduction": reduction(local_mean, global_mean),
                "legacy_global_target_fraction": _mean(float(self.global_target_rows), self.rows),
                "pair_local_target_fraction": _mean(float(self.local_target_rows), self.rows),
                "pair_local_better_fraction": _mean(float(self.local_better_rows), self.rows),
                "legacy_global_better_fraction": _mean(float(self.global_better_rows), self.rows),
                "equal_fraction": _mean(float(self.equal_rows), self.rows),
            },
            "usable_rows_only": {
                "legacy_global_mean_conditional_miss_risk": usable_global,
                "pair_local_mean_conditional_miss_risk": usable_local,
                "pair_local_minus_global": delta(usable_local, usable_global),
                "relative_risk_reduction": reduction(usable_local, usable_global),
                "legacy_global_target_fraction": _mean(float(self.usable_global_target_rows), self.usable_rows),
                "pair_local_target_fraction": _mean(float(self.usable_local_target_rows), self.usable_rows),
                "pair_local_better_fraction": _mean(float(self.usable_local_better_rows), self.usable_rows),
                "legacy_global_better_fraction": _mean(float(self.usable_global_better_rows), self.usable_rows),
                "equal_fraction": _mean(float(self.usable_equal_rows), self.usable_rows),
            },
            "rf_mechanisms_all_rf_rows": {
                f"{side}_mean_{mechanism}": _mean(
                    getattr(self, f"{side}_{mechanism}_sum"), self.rf_rows
                )
                for side in ("global", "local")
                for mechanism in ("collision", "half_duplex", "access", "utilization")
            },
            "rf_mechanisms_usable_rf_rows": {
                f"{side}_mean_{mechanism}": _mean(
                    getattr(self, f"usable_{side}_{mechanism}_sum"), self.usable_rf_rows
                )
                for side in ("global", "local")
                for mechanism in ("collision", "half_duplex", "access", "utilization")
            },
            "miss_budget": miss_budget,
        }
        if density is not None:
            row["density_vehicles_per_lane_km"] = density
        return row


@dataclass(slots=True)
class _MatchedObserver:
    selected_action: PolicyAction
    miss_budget: float
    legacy_sensed_fraction: float
    tally: _MatchedTally = field(default_factory=_MatchedTally)

    def observe_frame(
        self,
        *,
        decision: PopulationPolicyFrame,
        actions: tuple[PolicyAction, ...],
        outcomes: FramePacketOutcomes,
        boundary: FrameReturnBoundary,
        final_observation: Mapping[str, FrameObservation],
    ) -> None:
        del boundary, final_observation
        if outcomes.pair_ids != decision.frame.active_pair_ids or len(actions) != len(outcomes.pair_ids):
            raise GlobalLocalAccountingABError("completed A/B frame rows do not align")
        physics = outcomes.local_rf_physics
        demand = RFPoolDemand.from_ledger(physics.ledger)
        global_model = RFPoolModel(
            parameters=decision.local_rf_model.response_model.parameters,
            sensitivity_band=decision.local_rf_model.response_model.sensitivity_band,
            attempt_airtime_s=decision.local_rf_model.attempt_airtime_s,
        )
        global_response = (
            global_model.evaluate(demand, sensed_fraction=self.legacy_sensed_fraction)
            if demand.offered_rf_attempts
            else None
        )
        for actor_row, action, outcome in zip(
            decision.actor_frame.rows, actions, outcomes.pair_outcomes, strict=True
        ):
            usable = actor_row.usable
            if usable and action is not self.selected_action:
                raise GlobalLocalAccountingABError("usable row differs from fixed action")
            if not usable and action.label != "DUP-4":
                raise GlobalLocalAccountingABError("unusable row differs from fallback action")
            spec = action_resources(action)
            global_mechanisms: tuple[float, float, float, float] | None = None
            local_mechanisms: tuple[float, float, float, float] | None = None
            global_attempt_probability: float | None = None
            if spec.uses_rf:
                if global_response is None:
                    raise GlobalLocalAccountingABError("RF action has no global pool response")
                local_attempt = physics.attempt_risks.risk_for(outcome.pair_id)
                global_attempt = global_model.combine_attempt_risk(
                    global_response,
                    pair_id=outcome.pair_id,
                    propagation=local_attempt.propagation,
                )
                global_attempt_probability = global_attempt.total_failure_probability
                global_mechanisms = (
                    global_attempt.collision_probability,
                    global_attempt.half_duplex_probability,
                    global_attempt.access_failure_probability,
                    global_response.pool_utilization,
                )
                local_mechanisms = (
                    local_attempt.collision_probability,
                    local_attempt.half_duplex_probability,
                    local_attempt.access_failure_probability,
                    local_attempt.local_response.pool_utilization,
                )
            vlc_probability = outcome.vlc_miss_probability if spec.uses_vlc else None
            global_risk = packet_conditional_risk(
                action,
                rf_attempt_failure_probability=global_attempt_probability,
                vlc_miss_probability=vlc_probability,
            )
            local_risk = outcome.conditional_miss_probability
            recomposed_local = packet_conditional_risk(
                action,
                rf_attempt_failure_probability=(
                    physics.attempt_risks.risk_for(outcome.pair_id).total_failure_probability
                    if spec.uses_rf
                    else None
                ),
                vlc_miss_probability=vlc_probability,
            )
            if not math.isclose(local_risk, recomposed_local, rel_tol=1e-12, abs_tol=1e-15):
                raise GlobalLocalAccountingABError("local packet risk failed recomposition")
            self.tally.observe(
                usable=usable,
                action=action,
                global_risk=global_risk,
                local_risk=local_risk,
                miss_budget=self.miss_budget,
                global_mechanisms=global_mechanisms,
                local_mechanisms=local_mechanisms,
            )


def _cell_result(
    declaration: GlobalLocalAccountingABDeclaration,
    profile: _ExecutionProfile,
    action: PolicyAction,
    windows: tuple[EvaluationWindow, ...],
) -> dict[str, object]:
    catalog = TraceCatalog.from_splits(
        profile.config.paths.trace_root, profile.config.environment.splits
    )
    by_density: dict[float, _MatchedTally] = {
        density: _MatchedTally() for density in sorted({window.density for window in windows})
    }
    campaign = _MatchedTally()
    for window in windows:
        observer = _MatchedObserver(
            selected_action=action,
            miss_budget=declaration.miss_budget,
            legacy_sensed_fraction=declaration.legacy_global_sensed_fraction,
        )
        frozen = ObservationNormalizer.from_config(profile.config).freeze()
        run_policy_rollout_with_state(
            profile.config,
            catalog.source(window.trace_id),
            policy=_FixedActionPolicy(action),
            environment_seed=declaration.environment_seed,
            start_frame_index=window.start_frame_index,
            max_frames=window.frames,
            normalization_state=frozen,
            frame_observer=observer,
            sensitivity_band=declaration.sensing_band,
            collision_subchannels=profile.subchannels,
            receive_diversity=profile.receive_diversity,
        )
        by_density[window.density].merge(observer.tally)
        campaign.merge(observer.tally)
    return {
        "cell_id": f"{profile.declaration.profile_id}__{action.label}",
        "profile_id": profile.declaration.profile_id,
        "deadline_s": profile.declaration.deadline_s,
        "action_index": int(action),
        "action_name": action.label,
        "config_hash": config_hash(profile.config),
        "policy_environment_scope_hash": scope_hash(profile.config, "policy_environment"),
        "subchannels": profile.subchannels,
        "candidate_resources": profile.subchannels * 200,
        "densities": [
            by_density[density].as_dict(density=density, miss_budget=declaration.miss_budget)
            for density in sorted(by_density)
        ],
        "campaign": campaign.as_dict(density=None, miss_budget=declaration.miss_budget),
    }


def _validate_completed_cells(
    declaration: GlobalLocalAccountingABDeclaration,
    cells: tuple[dict[str, object], ...],
    *,
    require_complete: bool,
) -> None:
    if len(cells) > declaration.expected_cells:
        raise GlobalLocalAccountingABError("A/B cell prefix is too long")
    actual = tuple(cell.get("cell_id") for cell in cells)
    if actual != declaration.cell_ids[: len(cells)]:
        raise GlobalLocalAccountingABError("A/B cells are not an ordered prefix")
    if require_complete and len(cells) != declaration.expected_cells:
        raise GlobalLocalAccountingABError("A/B execution is incomplete")


def execute_global_local_accounting_ab(
    declaration: GlobalLocalAccountingABDeclaration,
    *,
    project_root: str | Path,
    completed_cells: tuple[dict[str, object], ...] = (),
    progress: ProgressCallback | None = None,
    checkpoint: CheckpointCallback | None = None,
) -> dict[str, object]:
    """Execute or resume the 18 matched validation cells."""

    _validate_completed_cells(declaration, completed_cells, require_complete=False)
    root = Path(project_root).expanduser().resolve(strict=False)
    profiles, windows = _execution_context(declaration, project_root=root)
    cells = list(completed_cells)
    ordered = tuple((profile, action) for profile in profiles for action in PolicyAction)
    for index, (profile, action) in enumerate(ordered[len(cells) :], start=len(cells) + 1):
        cell_id = f"{profile.declaration.profile_id}__{action.label}"
        if progress is not None:
            progress(index, len(ordered), cell_id)
        cells.append(_cell_result(declaration, profile, action, windows))
        if checkpoint is not None:
            checkpoint(tuple(cells))
    completed = tuple(cells)
    _validate_completed_cells(declaration, completed, require_complete=True)
    return {
        "schema": GLOBAL_LOCAL_AB_RESULT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "declaration": {"path": str(declaration.path), "sha256": declaration.sha256},
        "scope": "matched legacy-global versus pair-local contention and half-duplex accounting",
        "interpretation": "3 ms and 10 ms are separate within-deadline A/B analyses, not a cross-deadline comparison",
        "legacy_model": "frame-global RF pool with population-mean half-duplex",
        "current_model": "200 m pair-local RF pool with endpoint-specific half-duplex",
        "matched_controls": {
            "joint_actions": True,
            "propagation_truth": True,
            "vlc_truth": True,
            "environment_seed": declaration.environment_seed,
            "windows": [window.as_dict() for window in windows],
        },
        "actor_used": False,
        "checkpoint_used": False,
        "training_run_performed": False,
        "test_split_opened": False,
        "cells": list(completed),
    }


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


def write_global_local_accounting_ab_progress(
    path: str | Path,
    *,
    declaration: GlobalLocalAccountingABDeclaration,
    cells: tuple[dict[str, object], ...],
) -> Path:
    _validate_completed_cells(declaration, cells, require_complete=False)
    return _atomic_json_write(
        Path(path),
        {
            "schema": GLOBAL_LOCAL_AB_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "training_run_performed": False,
            "test_split_opened": False,
            "completed_cells": list(cells),
        },
    )


def load_global_local_accounting_ab_progress(
    path: str | Path,
    *,
    declaration: GlobalLocalAccountingABDeclaration,
) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GlobalLocalAccountingABError("A/B progress is unreadable") from error
    row = _mapping(
        payload,
        name="A/B progress",
        keys={
            "schema",
            "declaration_sha256",
            "training_run_performed",
            "test_split_opened",
            "completed_cells",
        },
    )
    if (
        row["schema"] != GLOBAL_LOCAL_AB_PROGRESS_SCHEMA
        or row["declaration_sha256"] != declaration.sha256
        or row["training_run_performed"] is not False
        or row["test_split_opened"] is not False
    ):
        raise GlobalLocalAccountingABError("A/B progress provenance has drifted")
    raw_cells = row["completed_cells"]
    if not isinstance(raw_cells, list) or any(not isinstance(cell, dict) for cell in raw_cells):
        raise GlobalLocalAccountingABError("A/B progress cells are malformed")
    cells = tuple(cast(dict[str, object], cell) for cell in raw_cells)
    _validate_completed_cells(declaration, cells, require_complete=False)
    return cells


def write_global_local_accounting_ab_result(
    path: str | Path,
    result: Mapping[str, object],
) -> Path:
    if result.get("schema") != GLOBAL_LOCAL_AB_RESULT_SCHEMA:
        raise GlobalLocalAccountingABError("refusing to write an unsupported A/B result")
    return _atomic_json_write(Path(path), result)


__all__ = [
    "GLOBAL_LOCAL_AB_DECLARATION_SCHEMA",
    "GLOBAL_LOCAL_AB_PROGRESS_SCHEMA",
    "GLOBAL_LOCAL_AB_RESULT_SCHEMA",
    "GlobalLocalAccountingABDeclaration",
    "GlobalLocalAccountingABError",
    "execute_global_local_accounting_ab",
    "load_global_local_accounting_ab_declaration",
    "load_global_local_accounting_ab_progress",
    "packet_conditional_risk",
    "structural_global_local_accounting_ab_dry_run",
    "write_global_local_accounting_ab_progress",
    "write_global_local_accounting_ab_result",
]
