"""Residual-tail and deadline-edge blocklength threshold diagnostic.

One replay per frozen optical profile captures the density-20 instantaneous
MRC SNR, VLC result, actor-control status, and propagation class.  The same
rows are then rescored for a predeclared finite set of anchors and by an exact
integer-channel-use bisection.  This keeps mobility, fading, geometry, action
availability, and fallback behavior matched across the complete threshold
screen.
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

from hybrid_v2x_rl.agents.longer_block_execution import (
    LONGER_BLOCK_FRONTIER_RESULT_SCHEMA,
)
from hybrid_v2x_rl.agents.longer_block_frontier import (
    LongerBlockCandidate,
    LongerBlockFrontierDeclaration,
    config_for_candidate,
    load_longer_block_frontier_declaration,
)
from hybrid_v2x_rl.agents.propagation_tail_diagnostic import (
    EXPECTED_DIMENSIONS,
    PropagationTailKey,
    PropagationTailTally,
)
from hybrid_v2x_rl.agents.receive_diversity_execution import (
    propagation_only_action_risk,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
from hybrid_v2x_rl.channels.rf.bler import block_error_probability
from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    run_policy_rollout_with_state,
)
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

DEADLINE_EDGE_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.deadline-edge-blocklength-threshold-declaration.v1"
)
DEADLINE_EDGE_RESULT_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.deadline-edge-blocklength-threshold-result.v1"
)
DEADLINE_EDGE_PROGRESS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.deadline-edge-blocklength-threshold-progress.v1"
)
ProgressCallback = Callable[[int, int, str], None]
CheckpointCallback = Callable[[tuple["OpticalThresholdResult", ...]], None]


class DeadlineEdgeThresholdError(HybridV2XError):
    """The threshold declaration, replay, or result is inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DeadlineEdgeThresholdError(f"{name} fields do not match the frozen schema")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise DeadlineEdgeThresholdError(f"{name} must be a nonempty array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeadlineEdgeThresholdError(f"{name} must be nonempty text")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise DeadlineEdgeThresholdError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DeadlineEdgeThresholdError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise DeadlineEdgeThresholdError(f"{name} must be a positive integer")
    return value


def _false(value: object, *, name: str) -> None:
    if value is not False:
        raise DeadlineEdgeThresholdError(f"{name} must be false")


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise DeadlineEdgeThresholdError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name)).expanduser()
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class DeadlineEdgeThresholdDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    payload_bytes: int
    deadline_s: float
    predecision_lead_s: float
    miss_budget: float
    density: float
    longer_declaration_path: Path
    longer_declaration_sha256: str
    longer_result_path: Path
    longer_result_sha256: str
    longer_result_schema: str
    longer_declaration: LongerBlockFrontierDeclaration
    source_candidate_name: str
    receive_profile_name: str
    optical_configuration_names: tuple[str, ...]
    decomposition_dimensions: tuple[str, ...]
    risk_tail_thresholds: tuple[float, ...]
    lower_airtime_s: float
    upper_airtime_s: float
    lower_channel_uses: int
    upper_channel_uses: int
    data_channel_uses_per_second: float
    anchor_airtimes_s: tuple[float, ...]
    anchor_channel_uses: tuple[int, ...]
    search_rule: str
    slot_s: float
    current_full_slots_per_attempt: int
    next_full_slots_per_attempt: int
    next_full_slot_airtime_s: float
    next_full_slot_rf4_airtime_s: float
    maximum_rf4_airtime_s: float
    output_path: Path

    @property
    def source_candidate(self) -> LongerBlockCandidate:
        matches = tuple(
            candidate
            for candidate in self.longer_declaration.candidates
            if candidate.name == self.source_candidate_name
        )
        if len(matches) != 1:  # pragma: no cover - loader proves this.
            raise DeadlineEdgeThresholdError("source candidate is absent")
        return matches[0]


def _float_sequence(value: object, *, name: str) -> tuple[float, ...]:
    return tuple(_number(raw, name=name) for raw in _sequence(value, name=name))


def _integer_sequence(value: object, *, name: str) -> tuple[int, ...]:
    return tuple(_integer(raw, name=name) for raw in _sequence(value, name=name))


def _json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeadlineEdgeThresholdError(f"{name} is unreadable", artifact_path=path) from error
    if not isinstance(payload, Mapping):
        raise DeadlineEdgeThresholdError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def load_deadline_edge_threshold_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> DeadlineEdgeThresholdDeclaration:
    """Load and fail-closed validate the residual threshold experiment."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="threshold declaration")
    top = _mapping(
        load_yaml_file(declaration_path),
        name="threshold declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "selection",
            "threshold_screen",
            "grid_realizability",
            "execution",
            "decision",
        },
    )
    if top["schema"] != DEADLINE_EDGE_DECLARATION_SCHEMA:
        raise DeadlineEdgeThresholdError("threshold declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="objective",
        keys={
            "payload_bytes",
            "deadline_s",
            "predecision_lead_s",
            "miss_budget",
            "density_vehicles_per_lane_km",
            "required_split",
            "interpretation",
        },
    )
    payload_bytes = _integer(objective["payload_bytes"], name="payload bytes")
    deadline_s = _number(objective["deadline_s"], name="deadline")
    lead_s = _number(objective["predecision_lead_s"], name="predecision lead")
    miss_budget = _number(objective["miss_budget"], name="miss budget")
    density = _number(objective["density_vehicles_per_lane_km"], name="density")
    if (
        payload_bytes != 300
        or not math.isclose(deadline_s, 0.010, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(lead_s, 0.0001, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(miss_budget, 1e-4, rel_tol=0.0, abs_tol=1e-15)
        or density != 20.0
        or objective["required_split"] != "validation"
    ):
        raise DeadlineEdgeThresholdError(
            "objective must remain frozen to density-20 300 B / 10 ms / 1e-4 validation"
        )

    evidence = _mapping(
        top["evidence"],
        name="evidence",
        keys={
            "longer_block_declaration",
            "longer_block_result",
            "actor_used",
            "checkpoint_used",
        },
    )
    _false(evidence["actor_used"], name="actor_used")
    _false(evidence["checkpoint_used"], name="checkpoint_used")
    declaration_evidence = _mapping(
        evidence["longer_block_declaration"],
        name="longer-block declaration evidence",
        keys={"path", "sha256"},
    )
    result_evidence = _mapping(
        evidence["longer_block_result"],
        name="longer-block result evidence",
        keys={"path", "sha256", "schema"},
    )
    longer_declaration_path = _resolve(
        root, declaration_evidence["path"], name="longer-block declaration path"
    )
    longer_declaration_sha256 = _digest(
        declaration_evidence["sha256"], name="longer-block declaration SHA-256"
    )
    longer_result_path = _resolve(root, result_evidence["path"], name="longer-block result path")
    longer_result_sha256 = _digest(result_evidence["sha256"], name="longer-block result SHA-256")
    longer_result_schema = _text(result_evidence["schema"], name="longer-block result schema")
    if longer_result_schema != LONGER_BLOCK_FRONTIER_RESULT_SCHEMA:
        raise DeadlineEdgeThresholdError("longer-block result schema is unsupported")
    if verify_evidence:
        for source, expected, name in (
            (
                longer_declaration_path,
                longer_declaration_sha256,
                "longer-block declaration",
            ),
            (longer_result_path, longer_result_sha256, "longer-block result"),
        ):
            if not source.is_file() or _sha256(source) != expected:
                raise DeadlineEdgeThresholdError(
                    f"{name} evidence is absent or has drifted", artifact_path=source
                )
        source_payload = _json_mapping(longer_result_path, name="longer-block result")
        if source_payload.get("schema") != longer_result_schema:
            raise DeadlineEdgeThresholdError("longer-block result schema has drifted")
        decision = source_payload.get("decision")
        if not isinstance(decision, Mapping) or (
            decision.get("joint_contention_frontier_authorized") is not False
            or decision.get("training_authorized") is not False
            or decision.get("test_split_opened") is not False
        ):
            raise DeadlineEdgeThresholdError("source safety decision has drifted")

    longer = load_longer_block_frontier_declaration(
        longer_declaration_path,
        project_root=root,
        verify_evidence=verify_evidence,
    )
    if longer.sha256 != longer_declaration_sha256:
        raise DeadlineEdgeThresholdError("longer-block declaration hash has drifted")

    selection = _mapping(
        top["selection"],
        name="selection",
        keys={
            "source_candidate",
            "receive_profile",
            "optical_configurations",
            "decomposition_dimensions",
            "risk_tail_thresholds",
        },
    )
    source_candidate_name = _text(selection["source_candidate"], name="source candidate")
    receive_profile_name = _text(selection["receive_profile"], name="receive profile")
    optical_names = tuple(
        _text(raw, name="optical configuration")
        for raw in _sequence(selection["optical_configurations"], name="optical configurations")
    )
    dimensions = tuple(
        _text(raw, name="decomposition dimension")
        for raw in _sequence(selection["decomposition_dimensions"], name="decomposition dimensions")
    )
    thresholds = _float_sequence(selection["risk_tail_thresholds"], name="risk-tail thresholds")
    if (
        source_candidate_name != "qpsk-2p0ms-sensitivity"
        or source_candidate_name not in {candidate.name for candidate in longer.candidates}
        or receive_profile_name != longer.receive_profile_name
        or optical_names != longer.optical_configuration_names
        or dimensions != EXPECTED_DIMENSIONS
        or thresholds != tuple(sorted(set(thresholds)))
        or thresholds[0] <= 0.0
        or thresholds[-1] >= 1.0
    ):
        raise DeadlineEdgeThresholdError("threshold selection has drifted")

    screen = _mapping(
        top["threshold_screen"],
        name="threshold screen",
        keys={
            "lower_airtime_s",
            "upper_airtime_s",
            "lower_channel_uses",
            "upper_channel_uses",
            "data_channel_uses_per_second",
            "anchor_airtimes_s",
            "anchor_channel_uses",
            "search_rule",
        },
    )
    lower_airtime_s = _number(screen["lower_airtime_s"], name="lower airtime")
    upper_airtime_s = _number(screen["upper_airtime_s"], name="upper airtime")
    lower_uses = _integer(screen["lower_channel_uses"], name="lower channel uses")
    upper_uses = _integer(screen["upper_channel_uses"], name="upper channel uses")
    uses_per_second = _number(
        screen["data_channel_uses_per_second"], name="channel uses per second"
    )
    anchor_airtimes = _float_sequence(screen["anchor_airtimes_s"], name="anchor airtimes")
    anchor_uses = _integer_sequence(screen["anchor_channel_uses"], name="anchor channel uses")
    expected_airtimes = (0.002, 0.002125, 0.00225, 0.002375, 0.002475)
    expected_uses = tuple(math.floor(uses_per_second * value) for value in anchor_airtimes)
    if (
        not math.isclose(lower_airtime_s, 0.002, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(upper_airtime_s, 0.002475, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(uses_per_second, 4_838_400.0, rel_tol=0.0, abs_tol=1e-9)
        or anchor_airtimes != expected_airtimes
        or anchor_uses != expected_uses
        or lower_uses != anchor_uses[0]
        or upper_uses != anchor_uses[-1]
        or screen["search_rule"]
        != "exact monotone integer-channel-use bisection over every frozen row"
    ):
        raise DeadlineEdgeThresholdError("threshold interval or anchors have drifted")

    grid = _mapping(
        top["grid_realizability"],
        name="grid realizability",
        keys={
            "slot_s",
            "current_full_slots_per_attempt",
            "next_full_slots_per_attempt",
            "next_full_slot_airtime_s",
            "next_full_slot_rf4_airtime_s",
            "maximum_rf4_airtime_s",
            "interpretation",
        },
    )
    slot_s = _number(grid["slot_s"], name="slot duration")
    current_slots = _integer(grid["current_full_slots_per_attempt"], name="current full slots")
    next_slots = _integer(grid["next_full_slots_per_attempt"], name="next full slots")
    next_airtime = _number(grid["next_full_slot_airtime_s"], name="next full-slot airtime")
    next_rf4 = _number(grid["next_full_slot_rf4_airtime_s"], name="next full-slot RF-4 airtime")
    maximum_rf4 = _number(grid["maximum_rf4_airtime_s"], name="maximum RF-4 airtime")
    if (
        not math.isclose(slot_s, 0.0005, rel_tol=0.0, abs_tol=1e-15)
        or current_slots != 4
        or next_slots != 5
        or not math.isclose(next_airtime, next_slots * slot_s, abs_tol=1e-15)
        or not math.isclose(next_rf4, 4 * next_airtime, abs_tol=1e-15)
        or not math.isclose(maximum_rf4, deadline_s - lead_s, abs_tol=1e-15)
        or next_rf4 <= maximum_rf4
        or not math.isclose(upper_airtime_s, maximum_rf4 / 4, abs_tol=1e-15)
    ):
        raise DeadlineEdgeThresholdError("slot-grid boundary has drifted")

    execution = _mapping(
        top["execution"],
        name="execution",
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
        raise DeadlineEdgeThresholdError("execution safety contract has drifted")
    decision = _mapping(
        top["decision"],
        name="decision",
        keys={
            "joint_frontier_authorization",
            "training_authorization",
            "next_rule",
            "claim_boundary",
        },
    )
    _false(decision["joint_frontier_authorization"], name="joint authorization")
    _false(decision["training_authorization"], name="training authorization")

    return DeadlineEdgeThresholdDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        payload_bytes=payload_bytes,
        deadline_s=deadline_s,
        predecision_lead_s=lead_s,
        miss_budget=miss_budget,
        density=density,
        longer_declaration_path=longer_declaration_path,
        longer_declaration_sha256=longer_declaration_sha256,
        longer_result_path=longer_result_path,
        longer_result_sha256=longer_result_sha256,
        longer_result_schema=longer_result_schema,
        longer_declaration=longer,
        source_candidate_name=source_candidate_name,
        receive_profile_name=receive_profile_name,
        optical_configuration_names=optical_names,
        decomposition_dimensions=dimensions,
        risk_tail_thresholds=thresholds,
        lower_airtime_s=lower_airtime_s,
        upper_airtime_s=upper_airtime_s,
        lower_channel_uses=lower_uses,
        upper_channel_uses=upper_uses,
        data_channel_uses_per_second=uses_per_second,
        anchor_airtimes_s=anchor_airtimes,
        anchor_channel_uses=anchor_uses,
        search_rule=_text(screen["search_rule"], name="search rule"),
        slot_s=slot_s,
        current_full_slots_per_attempt=current_slots,
        next_full_slots_per_attempt=next_slots,
        next_full_slot_airtime_s=next_airtime,
        next_full_slot_rf4_airtime_s=next_rf4,
        maximum_rf4_airtime_s=maximum_rf4,
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )


def _source_means(
    declaration: DeadlineEdgeThresholdDeclaration,
) -> dict[str, float]:
    payload = _json_mapping(declaration.longer_result_path, name="longer-block result")
    raw_candidates = payload.get("candidate_results")
    if not isinstance(raw_candidates, list):
        raise DeadlineEdgeThresholdError("longer-block result has no candidate rows")
    matches = tuple(
        row
        for row in raw_candidates
        if isinstance(row, Mapping)
        and isinstance(row.get("candidate"), Mapping)
        and row["candidate"].get("name") == declaration.source_candidate_name
    )
    if len(matches) != 1 or not isinstance(matches[0].get("rows"), list):
        raise DeadlineEdgeThresholdError("source candidate rows are absent")
    means: dict[str, float] = {}
    for raw in cast(list[object], matches[0]["rows"]):
        if not isinstance(raw, Mapping):
            continue
        optical = raw.get("optical_configuration_name")
        density = raw.get("density_vehicles_per_lane_km")
        mean = raw.get("mean_optimistic_propagation_only_conditional_miss_lower_bound")
        if (
            isinstance(optical, str)
            and optical in declaration.optical_configuration_names
            and density == declaration.density
            and isinstance(mean, int | float)
            and not isinstance(mean, bool)
        ):
            means[optical] = float(mean)
    if tuple(means) != declaration.optical_configuration_names:
        raise DeadlineEdgeThresholdError("source result does not cover the optical grid")
    return means


def _selected_dependencies(
    declaration: DeadlineEdgeThresholdDeclaration,
    *,
    project_root: Path,
) -> tuple[LongerBlockCandidate, tuple[EvaluationWindow, ...], dict[str, float]]:
    candidate = declaration.source_candidate
    source_report = structural_system_dry_run(
        declaration.longer_declaration.receive_declaration.source_frontier,
        project_root=project_root,
    )
    windows = tuple(
        window for window in source_report.windows if window.density == declaration.density
    )
    expected = declaration.longer_declaration.receive_declaration.source_frontier.validation_windows_per_density
    if len(windows) != expected:
        raise DeadlineEdgeThresholdError("threshold windows do not cover density 20")
    return candidate, windows, _source_means(declaration)


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
            raise DeadlineEdgeThresholdError("threshold window is absent") from error
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
            expected_config_scope_hashes={"mobility_trace": scope_hash(config, "mobility_trace")},
        )
        if window.start_frame_index + window.frames > reader.decision_frame_count:
            raise DeadlineEdgeThresholdError("threshold window exceeds its trace")


def structural_deadline_edge_dry_run(
    declaration: DeadlineEdgeThresholdDeclaration,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Validate the frozen interval, slot boundary, and traces without replay."""

    root = Path(project_root).expanduser().resolve(strict=False)
    candidate, windows, means = _selected_dependencies(declaration, project_root=root)
    configs: list[dict[str, object]] = []
    for optical in declaration.longer_declaration.optical_configurations:
        config = config_for_candidate(
            declaration.longer_declaration,
            candidate,
            optical,
            project_root=root,
        )
        if (
            config.rf.modulation != "qpsk"
            or not math.isclose(
                config.rf.timing.airtime_s,
                declaration.lower_airtime_s,
                abs_tol=1e-15,
            )
            or config.rf.slot_s != declaration.slot_s
        ):
            raise DeadlineEdgeThresholdError("source candidate physics has drifted")
        actual_rate = config.rf.available_coded_bits() / 2.0 / config.rf.timing.airtime_s
        if not math.isclose(
            actual_rate,
            declaration.data_channel_uses_per_second,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise DeadlineEdgeThresholdError("channel-use rate has drifted")
        _validate_trace_windows(config, windows=windows)
        configs.append(
            {
                "optical_configuration_name": optical.name,
                "config_hash": config_hash(config),
                "source_channel_uses": declaration.lower_channel_uses,
                "upper_channel_uses": declaration.upper_channel_uses,
            }
        )
    return {
        "schema": DEADLINE_EDGE_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "source_candidate": candidate.name,
        "density_vehicles_per_lane_km": declaration.density,
        "validation_windows": len(windows),
        "physical_profile_instances": len(configs),
        "source_means": means,
        "anchor_airtimes_s": list(declaration.anchor_airtimes_s),
        "anchor_channel_uses": list(declaration.anchor_channel_uses),
        "upper_airtime_s": declaration.upper_airtime_s,
        "next_full_slot_airtime_s": declaration.next_full_slot_airtime_s,
        "next_full_slot_fits": (
            declaration.next_full_slot_rf4_airtime_s <= declaration.maximum_rf4_airtime_s
        ),
        "configurations": configs,
        "channel_frames_evaluated": 0,
        "joint_frontier_authorized": False,
        "training_authorized": False,
        "test_split_opened": False,
    }


def _vlc_availability(result: VLCChannelResult) -> str:
    if not isinstance(result, VLCChannelResult):
        raise DeadlineEdgeThresholdError("VLC truth is malformed")
    if not result.is_geometric_failure:
        return "available"
    causes: list[str] = []
    if result.occluded:
        causes.append("occluded")
    if not result.within_field_of_view:
        causes.append("outside-fov")
    if not result.beam_aimed:
        causes.append("beam-not-aimed")
    if not causes:
        raise DeadlineEdgeThresholdError("VLC geometric failure has no cause")
    return "unavailable:" + "+".join(causes)


@dataclass(frozen=True, slots=True)
class _ThresholdRecord:
    snr_linear: float
    control_rf_failure: float
    vlc_failure: float
    choices: tuple[PolicyAction, ...]
    costs: tuple[float, ...]
    actor_control_status: str
    rf_propagation_state: str
    vlc_geometric_availability: str


def _select_record(
    record: _ThresholdRecord,
    *,
    channel_uses: int,
    control_channel_uses: int,
    information_bits: int,
) -> tuple[PolicyAction, float, float]:
    rf_failure = (
        record.control_rf_failure
        if channel_uses == control_channel_uses
        else block_error_probability(
            record.snr_linear,
            channel_uses,
            information_bits,
        )
    )

    def rank(index: int) -> tuple[float, float, int, int]:
        action = record.choices[index]
        risk = propagation_only_action_risk(
            action,
            rf_decoding_failure_probability=rf_failure,
            vlc_failure_probability=record.vlc_failure,
        )
        resources = action_resources(action)
        return risk, record.costs[index], resources.reserved_rf_attempts, int(action)

    winner = min(range(len(record.choices)), key=rank)
    action = record.choices[winner]
    return action, rank(winner)[0], rf_failure


@dataclass(slots=True)
class _ThresholdPolicy:
    declaration: DeadlineEdgeThresholdDeclaration
    information_bits: int
    records: list[_ThresholdRecord] = field(default_factory=list)
    frames: int = 0
    name: str = "deadline-edge-blocklength-oracle"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise DeadlineEdgeThresholdError("threshold replay requires oracle truth")
        if (
            decision.frame.source.split != "validation"
            or decision.frame.source.density != self.declaration.density
        ):
            raise DeadlineEdgeThresholdError("threshold replay accepts density-20 validation only")
        if set(channel_truth) != set(decision.frame.active_pair_ids):
            raise DeadlineEdgeThresholdError("threshold truth is not pair aligned")
        self.frames += 1
        if decision.population_size == 0:
            return ()
        allowed = decision.action_space.mask.allowed_actions
        fallback = decision.action_space.fallback_action
        for actor_row in decision.actor_frame.rows:
            truth = channel_truth[actor_row.pair_id]
            choices = allowed if actor_row.usable else (fallback,)
            self.records.append(
                _ThresholdRecord(
                    snr_linear=10.0 ** (truth.rf_propagation.sinr_db / 10.0),
                    control_rf_failure=(truth.rf_propagation.decoding_failure_probability),
                    vlc_failure=truth.vlc_result.total_failure_probability,
                    choices=choices,
                    costs=tuple(
                        decision.resource_map.activation_cost(action) for action in choices
                    ),
                    actor_control_status=(
                        "actor-usable" if actor_row.usable else "contract-fallback"
                    ),
                    rf_propagation_state=(truth.rf_propagation.propagation_state.value),
                    vlc_geometric_availability=_vlc_availability(truth.vlc_result),
                )
            )
        return tuple(fallback if row.usable else None for row in decision.actor_frame.rows)


def _score_records(
    records: tuple[_ThresholdRecord, ...],
    *,
    channel_uses: int,
    control_channel_uses: int,
    information_bits: int,
) -> tuple[float, dict[str, int]]:
    risk_terms: list[float] = []
    counts: dict[str, int] = {action.label: 0 for action in PolicyAction}
    for record in records:
        action, risk, _ = _select_record(
            record,
            channel_uses=channel_uses,
            control_channel_uses=control_channel_uses,
            information_bits=information_bits,
        )
        risk_terms.append(risk)
        counts[action.label] += 1
    return math.fsum(risk_terms) / len(records), counts


def _decompose_control(
    declaration: DeadlineEdgeThresholdDeclaration,
    records: tuple[_ThresholdRecord, ...],
    *,
    frames: int,
    information_bits: int,
) -> dict[str, object]:
    tally = PropagationTailTally()
    tally.frames = frames
    for record in records:
        action, risk, rf_failure = _select_record(
            record,
            channel_uses=declaration.lower_channel_uses,
            control_channel_uses=declaration.lower_channel_uses,
            information_bits=information_bits,
        )
        tally.observe(
            PropagationTailKey(
                lower_bound_action=action.label,
                actor_control_status=record.actor_control_status,
                rf_propagation_state=record.rf_propagation_state,
                vlc_geometric_availability=record.vlc_geometric_availability,
            ),
            selected_risk=risk,
            rf_failure=rf_failure,
            vlc_failure=record.vlc_failure,
        )
    return tally.as_dict(thresholds=declaration.risk_tail_thresholds)


def _threshold_search(
    declaration: DeadlineEdgeThresholdDeclaration,
    records: tuple[_ThresholdRecord, ...],
    *,
    information_bits: int,
) -> dict[str, object]:
    cache: dict[int, tuple[float, dict[str, int]]] = {}

    def score(channel_uses: int) -> tuple[float, dict[str, int]]:
        if channel_uses not in cache:
            cache[channel_uses] = _score_records(
                records,
                channel_uses=channel_uses,
                control_channel_uses=declaration.lower_channel_uses,
                information_bits=information_bits,
            )
        return cache[channel_uses]

    lower_mean, _ = score(declaration.lower_channel_uses)
    upper_mean, upper_counts = score(declaration.upper_channel_uses)
    threshold: int | None = None
    predecessor_mean: float | None = None
    if upper_mean <= declaration.miss_budget:
        low = declaration.lower_channel_uses
        high = declaration.upper_channel_uses
        while low < high:
            middle = (low + high) // 2
            if score(middle)[0] <= declaration.miss_budget:
                high = middle
            else:
                low = middle + 1
        threshold = low
        predecessor_mean = score(threshold - 1)[0] if threshold > 1 else None
    threshold_airtime = (
        threshold / declaration.data_channel_uses_per_second if threshold is not None else None
    )
    required_full_slots = (
        math.ceil(threshold_airtime / declaration.slot_s) if threshold_airtime is not None else None
    )
    deployable_airtime = (
        required_full_slots * declaration.slot_s if required_full_slots is not None else None
    )
    deployable_rf4 = 4 * deployable_airtime if deployable_airtime is not None else None
    deployable_fits = bool(
        deployable_rf4 is not None and deployable_rf4 <= declaration.maximum_rf4_airtime_s + 1e-15
    )
    return {
        "lower_channel_uses": declaration.lower_channel_uses,
        "lower_mean_selected_risk": lower_mean,
        "upper_channel_uses": declaration.upper_channel_uses,
        "upper_mean_selected_risk": upper_mean,
        "upper_meets_budget": upper_mean <= declaration.miss_budget,
        "upper_action_counts": upper_counts,
        "minimum_passing_channel_uses": threshold,
        "minimum_passing_airtime_s": threshold_airtime,
        "predecessor_channel_uses": threshold - 1 if threshold is not None else None,
        "predecessor_mean_selected_risk": predecessor_mean,
        "required_full_slots_per_attempt": required_full_slots,
        "rounded_full_slot_airtime_s": deployable_airtime,
        "rounded_full_slot_rf4_airtime_s": deployable_rf4,
        "rounded_full_slot_candidate_fits_deadline": deployable_fits,
    }


@dataclass(frozen=True, slots=True)
class OpticalThresholdResult:
    optical_configuration_name: str
    receiver_fov_deg: float
    frames: int
    transitions: int
    source_mean_selected_risk: float
    control_decomposition: dict[str, object]
    anchors: tuple[dict[str, object], ...]
    threshold: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "optical_configuration_name": self.optical_configuration_name,
            "receiver_fov_deg": self.receiver_fov_deg,
            "frames": self.frames,
            "transitions": self.transitions,
            "source_mean_selected_risk": self.source_mean_selected_risk,
            "control_reproduction_difference": (
                cast(float, self.control_decomposition["mean_selected_risk"])
                - self.source_mean_selected_risk
            ),
            "control_decomposition": self.control_decomposition,
            "anchors": list(self.anchors),
            "threshold": self.threshold,
        }


def _identity_normalization(config: ProjectConfig) -> ObservationNormalizationState:
    state = ObservationNormalizer.from_config(config).freeze()
    if not state.frozen or any(state.count) or any(state.mean) or any(state.second_moment):
        raise DeadlineEdgeThresholdError("threshold replay requires identity normalization")
    return state


def _execute_optical(
    declaration: DeadlineEdgeThresholdDeclaration,
    optical_name: str,
    *,
    project_root: Path,
    candidate: LongerBlockCandidate,
    windows: tuple[EvaluationWindow, ...],
    source_mean: float,
) -> OpticalThresholdResult:
    matches = tuple(
        optical
        for optical in declaration.longer_declaration.optical_configurations
        if optical.name == optical_name
    )
    if len(matches) != 1:
        raise DeadlineEdgeThresholdError("optical configuration is not frozen")
    optical = matches[0]
    config = config_for_candidate(
        declaration.longer_declaration,
        candidate,
        optical,
        project_root=project_root,
    )
    normalization = _identity_normalization(config)
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {trace.trace_id: trace for trace in catalog.for_split("validation")}
    policy = _ThresholdPolicy(
        declaration=declaration,
        information_bits=(config.service.payload_bytes + config.rf.timing.framing_overhead_bytes)
        * 8,
    )
    source = declaration.longer_declaration.receive_declaration.source_frontier
    headline = source.headline_point
    for window in windows:
        try:
            trace = validation[window.trace_id]
        except KeyError as error:
            raise DeadlineEdgeThresholdError("threshold window is absent") from error
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
            receive_diversity=(declaration.longer_declaration.receive_profile.physical_profile()),
            oracle_controls_unusable_rows=False,
        )
        if result.normalization_state != normalization:
            raise DeadlineEdgeThresholdError("threshold normalization changed")
    records = tuple(policy.records)
    if not records:
        raise DeadlineEdgeThresholdError("threshold replay produced no records")
    decomposition = _decompose_control(
        declaration,
        records,
        frames=policy.frames,
        information_bits=policy.information_bits,
    )
    measured = cast(float, decomposition["mean_selected_risk"])
    if not math.isclose(measured, source_mean, rel_tol=1e-12, abs_tol=1e-15):
        raise DeadlineEdgeThresholdError(
            "threshold replay does not reproduce the 2.0 ms source",
            context={"optical": optical.name, "measured": measured, "source": source_mean},
        )
    anchors: list[dict[str, object]] = []
    for airtime, channel_uses in zip(
        declaration.anchor_airtimes_s,
        declaration.anchor_channel_uses,
        strict=True,
    ):
        mean, counts = _score_records(
            records,
            channel_uses=channel_uses,
            control_channel_uses=declaration.lower_channel_uses,
            information_bits=policy.information_bits,
        )
        anchors.append(
            {
                "airtime_s": airtime,
                "channel_uses": channel_uses,
                "mean_selected_risk": mean,
                "budget_multiple": mean / declaration.miss_budget,
                "meets_budget": mean <= declaration.miss_budget,
                "lower_bound_action_counts": counts,
            }
        )
    return OpticalThresholdResult(
        optical_configuration_name=optical.name,
        receiver_fov_deg=optical.receiver_fov_deg,
        frames=policy.frames,
        transitions=len(records),
        source_mean_selected_risk=source_mean,
        control_decomposition=decomposition,
        anchors=tuple(anchors),
        threshold=_threshold_search(
            declaration,
            records,
            information_bits=policy.information_bits,
        ),
    )


def _validate_results(
    declaration: DeadlineEdgeThresholdDeclaration,
    results: tuple[OpticalThresholdResult, ...],
    *,
    require_complete: bool,
) -> None:
    names = tuple(result.optical_configuration_name for result in results)
    if names != declaration.optical_configuration_names[: len(results)]:
        raise DeadlineEdgeThresholdError("results are not an ordered optical prefix")
    if require_complete and len(results) != len(declaration.optical_configuration_names):
        raise DeadlineEdgeThresholdError("threshold result is incomplete")
    for result in results:
        payload = result.as_dict()
        difference = cast(float, payload["control_reproduction_difference"])
        if not math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=1e-15):
            raise DeadlineEdgeThresholdError("control reproduction does not close")
        if tuple(
            (cast(float, row["airtime_s"]), cast(int, row["channel_uses"]))
            for row in result.anchors
        ) != tuple(
            zip(
                declaration.anchor_airtimes_s,
                declaration.anchor_channel_uses,
                strict=True,
            )
        ):
            raise DeadlineEdgeThresholdError("anchor result grid has drifted")


@dataclass(frozen=True, slots=True)
class DeadlineEdgeThresholdResult:
    declaration: DeadlineEdgeThresholdDeclaration
    optical_results: tuple[OpticalThresholdResult, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        _validate_results(self.declaration, self.optical_results, require_complete=True)

    def decision(self) -> dict[str, object]:
        passing = tuple(
            result
            for result in self.optical_results
            if result.threshold["upper_meets_budget"] is True
        )
        best = min(
            (
                result
                for result in passing
                if isinstance(result.threshold["minimum_passing_channel_uses"], int)
            ),
            key=lambda result: cast(int, result.threshold["minimum_passing_channel_uses"]),
            default=None,
        )
        theoretical_pass = best is not None
        grid_pass = bool(
            best is not None and best.threshold["rounded_full_slot_candidate_fits_deadline"] is True
        )
        if theoretical_pass and not grid_pass:
            next_action = (
                "predeclare a timing/numerology intervention; the theoretical "
                "threshold fits the continuous deadline but not the current full-slot grid"
            )
        elif theoretical_pass:
            next_action = "predeclare the grid-realizable candidate for the joint frontier"
        else:
            next_action = (
                "the continuous deadline boundary also fails; select a different "
                "physical reliability intervention"
            )
        return {
            "theoretical_deadline_boundary_passes": theoretical_pass,
            "passing_optical_configuration_names": [
                result.optical_configuration_name for result in passing
            ],
            "minimum_passing_channel_uses": (
                best.threshold["minimum_passing_channel_uses"] if best is not None else None
            ),
            "minimum_passing_airtime_s": (
                best.threshold["minimum_passing_airtime_s"] if best is not None else None
            ),
            "current_full_slot_grid_has_passing_candidate": grid_pass,
            "joint_contention_frontier_authorized": grid_pass,
            "training_authorized": False,
            "test_split_opened": False,
            "next_action": next_action,
            "claim_boundary": (
                "validation-only synthetic threshold under the optimistic finite-blocklength model"
            ),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": DEADLINE_EDGE_RESULT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "declaration": {
                "path": str(self.declaration.path),
                "sha256": self.declaration.sha256,
            },
            "sources": {
                "longer_block_declaration": {
                    "path": str(self.declaration.longer_declaration_path),
                    "sha256": self.declaration.longer_declaration_sha256,
                },
                "longer_block_result": {
                    "path": str(self.declaration.longer_result_path),
                    "sha256": self.declaration.longer_result_sha256,
                    "schema": self.declaration.longer_result_schema,
                },
            },
            "payload_bytes": self.declaration.payload_bytes,
            "deadline_s": self.declaration.deadline_s,
            "predecision_lead_s": self.declaration.predecision_lead_s,
            "reliability_miss_budget": self.declaration.miss_budget,
            "density_vehicles_per_lane_km": self.declaration.density,
            "source_candidate": self.declaration.source_candidate_name,
            "receive_profile": self.declaration.receive_profile_name,
            "actor_used": False,
            "checkpoint_used": False,
            "training_run_performed": False,
            "test_split_opened": False,
            "threshold_search_complete": True,
            "threshold_interval": {
                "lower_airtime_s": self.declaration.lower_airtime_s,
                "upper_airtime_s": self.declaration.upper_airtime_s,
                "lower_channel_uses": self.declaration.lower_channel_uses,
                "upper_channel_uses": self.declaration.upper_channel_uses,
                "search_rule": self.declaration.search_rule,
            },
            "grid_realizability": {
                "slot_s": self.declaration.slot_s,
                "current_full_slots_per_attempt": (self.declaration.current_full_slots_per_attempt),
                "next_full_slots_per_attempt": (self.declaration.next_full_slots_per_attempt),
                "next_full_slot_airtime_s": (self.declaration.next_full_slot_airtime_s),
                "next_full_slot_rf4_airtime_s": (self.declaration.next_full_slot_rf4_airtime_s),
                "maximum_rf4_airtime_s": self.declaration.maximum_rf4_airtime_s,
                "next_full_slot_fits": False,
            },
            "optical_results": [result.as_dict() for result in self.optical_results],
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


def write_deadline_edge_progress(
    path: str | Path,
    *,
    declaration: DeadlineEdgeThresholdDeclaration,
    results: tuple[OpticalThresholdResult, ...],
) -> Path:
    _validate_results(declaration, results, require_complete=False)
    return _atomic_json_write(
        Path(path),
        {
            "schema": DEADLINE_EDGE_PROGRESS_SCHEMA,
            "declaration_sha256": declaration.sha256,
            "longer_block_result_sha256": declaration.longer_result_sha256,
            "training_run_performed": False,
            "test_split_opened": False,
            "optical_results": [result.as_dict() for result in results],
        },
    )


def _optical_result_from_dict(payload: object) -> OpticalThresholdResult:
    if not isinstance(payload, Mapping):
        raise DeadlineEdgeThresholdError("progress optical result must be an object")
    expected = {
        "optical_configuration_name",
        "receiver_fov_deg",
        "frames",
        "transitions",
        "source_mean_selected_risk",
        "control_reproduction_difference",
        "control_decomposition",
        "anchors",
        "threshold",
    }
    if set(payload) != expected:
        raise DeadlineEdgeThresholdError("progress optical fields have drifted")
    anchors = payload["anchors"]
    decomposition = payload["control_decomposition"]
    threshold = payload["threshold"]
    if (
        not isinstance(anchors, list)
        or any(not isinstance(row, Mapping) for row in anchors)
        or not isinstance(decomposition, Mapping)
        or not isinstance(threshold, Mapping)
    ):
        raise DeadlineEdgeThresholdError("progress optical result is malformed")
    result = OpticalThresholdResult(
        optical_configuration_name=cast(str, payload["optical_configuration_name"]),
        receiver_fov_deg=float(cast(float, payload["receiver_fov_deg"])),
        frames=cast(int, payload["frames"]),
        transitions=cast(int, payload["transitions"]),
        source_mean_selected_risk=float(cast(float, payload["source_mean_selected_risk"])),
        control_decomposition=dict(decomposition),
        anchors=tuple(dict(row) for row in anchors),
        threshold=dict(threshold),
    )
    if result.as_dict() != dict(payload):
        raise DeadlineEdgeThresholdError("progress optical result has drifted")
    return result


def load_deadline_edge_progress(
    path: str | Path,
    *,
    declaration: DeadlineEdgeThresholdDeclaration,
) -> tuple[OpticalThresholdResult, ...]:
    payload = _json_mapping(Path(path), name="deadline-edge progress")
    expected = {
        "schema",
        "declaration_sha256",
        "longer_block_result_sha256",
        "training_run_performed",
        "test_split_opened",
        "optical_results",
    }
    if set(payload) != expected:
        raise DeadlineEdgeThresholdError("progress fields do not match the schema")
    if (
        payload["schema"] != DEADLINE_EDGE_PROGRESS_SCHEMA
        or payload["declaration_sha256"] != declaration.sha256
        or payload["longer_block_result_sha256"] != declaration.longer_result_sha256
        or payload["training_run_performed"] is not False
        or payload["test_split_opened"] is not False
    ):
        raise DeadlineEdgeThresholdError("progress provenance has drifted")
    raw_results = payload["optical_results"]
    if not isinstance(raw_results, list) or len(raw_results) > len(
        declaration.optical_configuration_names
    ):
        raise DeadlineEdgeThresholdError("progress optical prefix is malformed")
    results = tuple(_optical_result_from_dict(raw) for raw in raw_results)
    _validate_results(declaration, results, require_complete=False)
    return results


def execute_deadline_edge_threshold(
    declaration: DeadlineEdgeThresholdDeclaration,
    *,
    project_root: str | Path,
    completed_results: tuple[OpticalThresholdResult, ...] = (),
    progress: ProgressCallback | None = None,
    checkpoint: CheckpointCallback | None = None,
) -> DeadlineEdgeThresholdResult:
    """Replay each optical profile once and solve the frozen threshold."""

    root = Path(project_root).expanduser().resolve(strict=False)
    structural_deadline_edge_dry_run(declaration, project_root=root)
    _validate_results(declaration, completed_results, require_complete=False)
    candidate, windows, source_means = _selected_dependencies(declaration, project_root=root)
    results = list(completed_results)
    total = len(declaration.optical_configuration_names)
    for index, optical_name in enumerate(declaration.optical_configuration_names, start=1):
        if index <= len(results):
            continue
        if progress is not None:
            progress(index, total, optical_name)
        results.append(
            _execute_optical(
                declaration,
                optical_name,
                project_root=root,
                candidate=candidate,
                windows=windows,
                source_mean=source_means[optical_name],
            )
        )
        if checkpoint is not None:
            checkpoint(tuple(results))
    return DeadlineEdgeThresholdResult(
        declaration=declaration,
        optical_results=tuple(results),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "DEADLINE_EDGE_DECLARATION_SCHEMA",
    "DEADLINE_EDGE_PROGRESS_SCHEMA",
    "DEADLINE_EDGE_RESULT_SCHEMA",
    "DeadlineEdgeThresholdDeclaration",
    "DeadlineEdgeThresholdError",
    "DeadlineEdgeThresholdResult",
    "OpticalThresholdResult",
    "execute_deadline_edge_threshold",
    "load_deadline_edge_progress",
    "load_deadline_edge_threshold_declaration",
    "structural_deadline_edge_dry_run",
    "write_deadline_edge_progress",
]
