"""Frozen 300 B / 10 ms longer-block RF frontier declaration.

The declaration changes only the service deadline and the RF modulation/timing
row needed to test longer finite blocklengths.  It retains the mobility,
receiver-diversity, optical, action, and split contracts.  Loading or
structurally validating it evaluates no channel frame and cannot authorize
training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from hybrid_v2x_rl.agents.propagation_tail_diagnostic import (
    PROPAGATION_TAIL_RESULT_SCHEMA,
)
from hybrid_v2x_rl.agents.receive_diversity_execution import (
    RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA,
)
from hybrid_v2x_rl.agents.receive_diversity_frontier import (
    ReceiveDiversityFrontierDeclaration,
    ReceiveDiversityProfile,
    load_receive_diversity_frontier_declaration,
)
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    structural_dry_run as structural_system_dry_run,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    OpticalConfigurationLevel,
)
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.loader import load_config, load_yaml_file
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.config.validation import validate_project_config
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER
from hybrid_v2x_rl.env.assembly import build_lifecycle, build_rf_channel
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.frames import PopulationFrameReader, TraceCatalog

LONGER_BLOCK_FRONTIER_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.longer-block-rf-frontier-declaration.v1"
)
CandidateRole = Literal["control", "primary", "sensitivity"]
Modulation = Literal["qpsk", "16qam"]


class LongerBlockFrontierError(HybridV2XError):
    """The longer-block declaration or one of its frozen inputs has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise LongerBlockFrontierError(f"{name} fields do not match the frozen schema")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise LongerBlockFrontierError(f"{name} must be a nonempty array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongerBlockFrontierError(f"{name} must be nonempty text")
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise LongerBlockFrontierError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LongerBlockFrontierError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LongerBlockFrontierError(f"{name} must be a positive integer")
    return value


def _false(value: object, *, name: str) -> None:
    if value is not False:
        raise LongerBlockFrontierError(f"{name} must be false")


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise LongerBlockFrontierError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name)).expanduser()
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    path: Path
    sha256: str
    schema: str | None = None


@dataclass(frozen=True, slots=True)
class LongerBlockCandidate:
    name: str
    role: CandidateRole
    modulation: Modulation
    airtime_s: float
    gross_bit_rate_bps: float
    code_rate: float
    expected_slots: int
    expected_available_coded_bits: float
    expected_channel_uses: int
    expected_rf4_airtime_s: float


@dataclass(frozen=True, slots=True)
class LongerBlockFrontierDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    payload_bytes: int
    deadline_s: float
    predecision_lead_s: float
    miss_budget: float
    densities: tuple[float, ...]
    tail_declaration: EvidenceArtifact
    tail_result: EvidenceArtifact
    receive_declaration_artifact: EvidenceArtifact
    receive_result: EvidenceArtifact
    receive_declaration: ReceiveDiversityFrontierDeclaration
    base_config_layers: tuple[Path, ...]
    receive_profile_name: str
    optical_configuration_names: tuple[str, ...]
    candidates: tuple[LongerBlockCandidate, ...]
    control_reuse_rule: str
    selection_rule: str
    next_stage_rule: str
    output_path: Path

    @property
    def receive_profile(self) -> ReceiveDiversityProfile:
        matches = tuple(
            profile
            for profile in self.receive_declaration.receive_profiles
            if profile.name == self.receive_profile_name
        )
        if len(matches) != 1:  # pragma: no cover - loader proves this.
            raise LongerBlockFrontierError("frozen receive profile is absent")
        return matches[0]

    @property
    def optical_configurations(self) -> tuple[OpticalConfigurationLevel, ...]:
        by_name = {
            optical.name: optical
            for optical in self.receive_declaration.source_frontier.optical_configurations
        }
        return tuple(by_name[name] for name in self.optical_configuration_names)


def _artifact(
    value: object,
    *,
    root: Path,
    name: str,
    schema_required: bool,
) -> EvidenceArtifact:
    keys = {"path", "sha256", "schema"} if schema_required else {"path", "sha256"}
    row = _mapping(value, name=name, keys=keys)
    return EvidenceArtifact(
        path=_resolve(root, row["path"], name=f"{name} path"),
        sha256=_digest(row["sha256"], name=f"{name} SHA-256"),
        schema=_text(row["schema"], name=f"{name} schema") if schema_required else None,
    )


def _candidate(value: object, *, index: int) -> LongerBlockCandidate:
    row = _mapping(
        value,
        name=f"candidate {index}",
        keys={
            "name",
            "role",
            "modulation",
            "airtime_s",
            "gross_bit_rate_bps",
            "code_rate",
            "expected_slots",
            "expected_available_coded_bits",
            "expected_channel_uses",
            "expected_rf4_airtime_s",
        },
    )
    role = _text(row["role"], name="candidate role")
    modulation = _text(row["modulation"], name="candidate modulation")
    if role not in ("control", "primary", "sensitivity"):
        raise LongerBlockFrontierError("candidate role is unsupported")
    if modulation not in ("qpsk", "16qam"):
        raise LongerBlockFrontierError("candidate modulation is unsupported")
    return LongerBlockCandidate(
        name=_text(row["name"], name="candidate name"),
        role=cast(CandidateRole, role),
        modulation=cast(Modulation, modulation),
        airtime_s=_number(row["airtime_s"], name="candidate airtime"),
        gross_bit_rate_bps=_number(row["gross_bit_rate_bps"], name="candidate gross bit rate"),
        code_rate=_number(row["code_rate"], name="candidate code rate"),
        expected_slots=_integer(row["expected_slots"], name="candidate slots"),
        expected_available_coded_bits=_number(
            row["expected_available_coded_bits"], name="candidate available coded bits"
        ),
        expected_channel_uses=_integer(row["expected_channel_uses"], name="candidate channel uses"),
        expected_rf4_airtime_s=_number(
            row["expected_rf4_airtime_s"], name="candidate RF-4 airtime"
        ),
    )


def _load_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LongerBlockFrontierError(f"{name} is unreadable", artifact_path=path) from error
    if not isinstance(payload, Mapping):
        raise LongerBlockFrontierError(f"{name} must contain a JSON object")
    return cast(Mapping[str, object], payload)


def load_longer_block_frontier_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> LongerBlockFrontierDeclaration:
    """Load and fail-closed validate the frozen longer-block experiment."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="longer-block declaration")
    top = _mapping(
        load_yaml_file(declaration_path),
        name="longer-block declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "base_config_layers",
            "selection",
            "execution",
            "decision",
        },
    )
    if top["schema"] != LONGER_BLOCK_FRONTIER_DECLARATION_SCHEMA:
        raise LongerBlockFrontierError("longer-block declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="objective",
        keys={
            "payload_bytes",
            "deadline_s",
            "predecision_lead_s",
            "miss_budget",
            "densities_vehicles_per_lane_km",
            "required_split",
            "interpretation",
        },
    )
    payload_bytes = _integer(objective["payload_bytes"], name="payload bytes")
    deadline_s = _number(objective["deadline_s"], name="deadline")
    lead_s = _number(objective["predecision_lead_s"], name="predecision lead")
    miss_budget = _number(objective["miss_budget"], name="miss budget")
    densities = tuple(
        _number(raw, name="density")
        for raw in _sequence(objective["densities_vehicles_per_lane_km"], name="densities")
    )
    if (
        payload_bytes != 300
        or not math.isclose(deadline_s, 0.010, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(lead_s, 0.0001, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(miss_budget, 1e-4, rel_tol=0.0, abs_tol=1e-15)
        or densities != (10.0, 20.0, 30.0)
        or objective["required_split"] != "validation"
    ):
        raise LongerBlockFrontierError(
            "objective must remain frozen to 300 B / 10 ms / 1e-4 on validation"
        )

    evidence = _mapping(
        top["evidence"],
        name="evidence",
        keys={
            "propagation_tail_declaration",
            "propagation_tail_result",
            "receive_diversity_declaration",
            "receive_diversity_result",
            "actor_used",
            "checkpoint_used",
        },
    )
    _false(evidence["actor_used"], name="actor_used")
    _false(evidence["checkpoint_used"], name="checkpoint_used")
    tail_declaration = _artifact(
        evidence["propagation_tail_declaration"],
        root=root,
        name="propagation-tail declaration",
        schema_required=False,
    )
    tail_result = _artifact(
        evidence["propagation_tail_result"],
        root=root,
        name="propagation-tail result",
        schema_required=True,
    )
    receive_artifact = _artifact(
        evidence["receive_diversity_declaration"],
        root=root,
        name="receive-diversity declaration",
        schema_required=False,
    )
    receive_result = _artifact(
        evidence["receive_diversity_result"],
        root=root,
        name="receive-diversity result",
        schema_required=True,
    )
    if tail_result.schema != PROPAGATION_TAIL_RESULT_SCHEMA:
        raise LongerBlockFrontierError("propagation-tail result schema is unsupported")
    if receive_result.schema != RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA:
        raise LongerBlockFrontierError("receive-diversity result schema is unsupported")
    if verify_evidence:
        for artifact, name in (
            (tail_declaration, "propagation-tail declaration"),
            (tail_result, "propagation-tail result"),
            (receive_artifact, "receive-diversity declaration"),
            (receive_result, "receive-diversity result"),
        ):
            if not artifact.path.is_file() or _sha256(artifact.path) != artifact.sha256:
                raise LongerBlockFrontierError(
                    f"{name} evidence is absent or has drifted",
                    artifact_path=artifact.path,
                )
        if (
            _load_json_mapping(tail_result.path, name="propagation-tail result").get("schema")
            != tail_result.schema
        ):
            raise LongerBlockFrontierError("propagation-tail result schema has drifted")
        if (
            _load_json_mapping(receive_result.path, name="receive-diversity result").get("schema")
            != receive_result.schema
        ):
            raise LongerBlockFrontierError("receive-diversity result schema has drifted")

    receive_declaration = load_receive_diversity_frontier_declaration(
        receive_artifact.path,
        project_root=root,
        verify_evidence=verify_evidence,
    )
    if receive_declaration.sha256 != receive_artifact.sha256:
        raise LongerBlockFrontierError("receive-diversity declaration digest has drifted")

    layer_values = _sequence(top["base_config_layers"], name="base config layers")
    base_layers = tuple(_resolve(root, value, name="base config layer") for value in layer_values)
    expected_layers = (
        "configs/project/default.yaml",
        "configs/mobility/synthetic_manhattan.yaml",
        "configs/service/ev2x_300B_10ms_1e-4.yaml",
        "configs/channel/rf_nr_v2x.yaml",
        "configs/channel/vlc_vehicle.yaml",
        "configs/observation/causal_200ms_forecast.yaml",
        "configs/training/primal_dual_ppo.yaml",
        "configs/evaluation/primary.yaml",
    )
    if tuple(path.relative_to(root).as_posix() for path in base_layers) != expected_layers:
        raise LongerBlockFrontierError("base config layer order has drifted")

    selection = _mapping(
        top["selection"],
        name="selection",
        keys={"receive_profile", "optical_configurations", "candidates"},
    )
    profile_name = _text(selection["receive_profile"], name="receive profile")
    optical_names = tuple(
        _text(value, name="optical configuration")
        for value in _sequence(selection["optical_configurations"], name="optical configurations")
    )
    candidates = tuple(
        _candidate(value, index=index)
        for index, value in enumerate(_sequence(selection["candidates"], name="candidates"))
    )
    expected_candidate_identity = (
        ("16qam-0p5ms-control", "control", "16qam", 0.0005),
        ("qpsk-1p0ms-primary", "primary", "qpsk", 0.001),
        ("qpsk-1p5ms-sensitivity", "sensitivity", "qpsk", 0.0015),
        ("qpsk-2p0ms-sensitivity", "sensitivity", "qpsk", 0.002),
    )
    if (
        tuple(
            (candidate.name, candidate.role, candidate.modulation, candidate.airtime_s)
            for candidate in candidates
        )
        != expected_candidate_identity
    ):
        raise LongerBlockFrontierError("candidate identity or order has drifted")
    if tuple(candidate.expected_slots for candidate in candidates) != (1, 2, 3, 4):
        raise LongerBlockFrontierError("candidate slot counts must be 1, 2, 3, and 4")
    if profile_name != receive_declaration.headline_receive_profile.name:
        raise LongerBlockFrontierError("receive profile must remain the headline MRC profile")
    source_optical_names = tuple(
        optical.name for optical in receive_declaration.source_frontier.optical_configurations
    )
    if optical_names != source_optical_names:
        raise LongerBlockFrontierError("optical configuration axis has drifted")

    execution = _mapping(
        top["execution"],
        name="execution",
        keys={
            "control_reuse_rule",
            "candidate_order",
            "selection_rule",
            "next_stage_rule",
            "no_adaptive_axis_expansion",
            "no_training",
            "no_test_split",
            "output_path",
        },
    )
    if (
        execution["candidate_order"] != "increasing per-attempt airtime"
        or execution["no_adaptive_axis_expansion"] is not True
        or execution["no_training"] is not True
        or execution["no_test_split"] is not True
    ):
        raise LongerBlockFrontierError("execution safety contract has drifted")
    decision = _mapping(
        top["decision"],
        name="decision",
        keys={"action_contract", "training_authorization", "claim_boundary"},
    )
    _false(decision["training_authorization"], name="training authorization")
    if decision["action_contract"] != (
        "retain the canonical nine actions and maximum four RF attempts"
    ):
        raise LongerBlockFrontierError("action contract has drifted")

    return LongerBlockFrontierDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        payload_bytes=payload_bytes,
        deadline_s=deadline_s,
        predecision_lead_s=lead_s,
        miss_budget=miss_budget,
        densities=densities,
        tail_declaration=tail_declaration,
        tail_result=tail_result,
        receive_declaration_artifact=receive_artifact,
        receive_result=receive_result,
        receive_declaration=receive_declaration,
        base_config_layers=base_layers,
        receive_profile_name=profile_name,
        optical_configuration_names=optical_names,
        candidates=candidates,
        control_reuse_rule=_text(execution["control_reuse_rule"], name="control reuse rule"),
        selection_rule=_text(execution["selection_rule"], name="selection rule"),
        next_stage_rule=_text(execution["next_stage_rule"], name="next-stage rule"),
        output_path=_resolve(root, execution["output_path"], name="output path"),
    )


def config_for_candidate(
    declaration: LongerBlockFrontierDeclaration,
    candidate: LongerBlockCandidate,
    optical: OpticalConfigurationLevel,
    *,
    project_root: str | Path,
) -> ProjectConfig:
    """Build and validate one exact candidate/optical configuration."""

    root = Path(project_root).expanduser().resolve(strict=False)
    base = load_config(
        (*declaration.base_config_layers, *optical.additional_config_layers),
        project_root=root,
    )
    timing = base.rf.timing.model_copy(
        update={
            "airtime_s": candidate.airtime_s,
            "gross_bit_rate_bps": candidate.gross_bit_rate_bps,
            "code_rate": candidate.code_rate,
        }
    )
    rf = base.rf.model_copy(update={"modulation": candidate.modulation, "timing": timing})
    return validate_project_config(base.model_copy(update={"rf": rf}))


def candidate_grid_row(
    declaration: LongerBlockFrontierDeclaration,
    candidate: LongerBlockCandidate,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Return reconciled RF resource-grid and deadline arithmetic."""

    optical = declaration.optical_configurations[0]
    config = config_for_candidate(declaration, candidate, optical, project_root=project_root)
    rf_channel = build_rf_channel(
        config,
        receive_diversity=declaration.receive_profile.physical_profile(),
    )
    available_coded_bits = config.rf.available_coded_bits()
    slots = config.rf.slots_per_transmission
    channel_uses = rf_channel.blocklength
    rf4_airtime_s = 4 * config.rf.timing.airtime_s
    available_deadline_s = config.service.deadline_s - config.service.predecision_lead_s
    expected = (
        math.isclose(slots, candidate.expected_slots, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(
            available_coded_bits,
            candidate.expected_available_coded_bits,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and channel_uses == candidate.expected_channel_uses
        and math.isclose(
            rf4_airtime_s,
            candidate.expected_rf4_airtime_s,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )
    if not expected:
        raise LongerBlockFrontierError(
            "candidate resource-grid arithmetic differs from the declaration",
            context={"candidate": candidate.name},
        )
    return {
        "candidate_name": candidate.name,
        "role": candidate.role,
        "modulation": config.rf.modulation,
        "airtime_s": config.rf.timing.airtime_s,
        "slots_per_attempt": slots,
        "resource_blocks": config.rf.resource_blocks,
        "resource_element_overhead": config.rf.resource_element_overhead,
        "available_coded_bits": available_coded_bits,
        "configured_capacity_bits": config.rf.timing.capacity_bits,
        "required_coded_bits": config.rf.timing.coded_block_bits(config.service.payload_bytes),
        "finite_blocklength_channel_uses": channel_uses,
        "information_bits": rf_channel.information_bits,
        "information_rate_bits_per_channel_use": (rf_channel.information_bits / channel_uses),
        "rf4_total_airtime_s": rf4_airtime_s,
        "available_deadline_s": available_deadline_s,
        "rf4_fits_deadline": rf4_airtime_s <= available_deadline_s,
        "config_hash": config_hash(config),
        "policy_environment_scope_hash": scope_hash(config, "policy_environment"),
    }


def _validate_trace_windows(
    config: ProjectConfig,
    *,
    windows: tuple[object, ...],
) -> None:
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    validation = {source.trace_id: source for source in catalog.for_split("validation")}
    for raw in windows:
        trace_id = getattr(raw, "trace_id", None)
        density = getattr(raw, "density", None)
        start = getattr(raw, "start_frame_index", None)
        frames = getattr(raw, "frames", None)
        if not isinstance(trace_id, str) or trace_id not in validation:
            raise LongerBlockFrontierError("validation window is absent from the trace catalog")
        source = validation[trace_id]
        if density != source.density or not isinstance(start, int) or not isinstance(frames, int):
            raise LongerBlockFrontierError("validation window metadata is malformed")
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
            expected_config_scope_hashes={"mobility_trace": scope_hash(config, "mobility_trace")},
        )
        if start + frames > reader.decision_frame_count:
            raise LongerBlockFrontierError("validation window exceeds its trace")


def structural_longer_block_dry_run(
    declaration: LongerBlockFrontierDeclaration,
    *,
    project_root: str | Path,
) -> dict[str, object]:
    """Validate every candidate grid and dependency without evaluating frames."""

    if not isinstance(declaration, LongerBlockFrontierDeclaration):
        raise LongerBlockFrontierError("dry run requires a validated declaration")
    root = Path(project_root).expanduser().resolve(strict=False)
    source_report = structural_system_dry_run(
        declaration.receive_declaration.source_frontier,
        project_root=root,
    )
    grid_rows = tuple(
        candidate_grid_row(declaration, candidate, project_root=root)
        for candidate in declaration.candidates
    )
    validated_hashes: set[str] = set()
    physical_instances = 0
    for candidate in declaration.candidates:
        for optical in declaration.optical_configurations:
            config = config_for_candidate(
                declaration,
                candidate,
                optical,
                project_root=root,
            )
            if config.service.payload_bytes != declaration.payload_bytes:
                raise LongerBlockFrontierError("candidate payload differs from objective")
            if not math.isclose(
                config.service.deadline_s,
                declaration.deadline_s,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise LongerBlockFrontierError("candidate deadline differs from objective")
            action_space = MaskedActionSpace.from_config(
                config.environment,
                config.rf,
                config.vlc,
            )
            if (
                tuple(action.label for action in action_space.mask.allowed_actions)
                != POLICY_ACTION_ORDER
                or config.environment.max_rf_attempts != 4
            ):
                raise LongerBlockFrontierError("candidate action contract has drifted")
            build_lifecycle(
                config,
                receive_diversity=declaration.receive_profile.physical_profile(),
            )
            digest = config_hash(config)
            if digest not in validated_hashes:
                _validate_trace_windows(config, windows=source_report.windows)
                validated_hashes.add(digest)
            physical_instances += 1
    return {
        "schema": LONGER_BLOCK_FRONTIER_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "payload_bytes": declaration.payload_bytes,
        "deadline_s": declaration.deadline_s,
        "miss_budget": declaration.miss_budget,
        "receive_profile": declaration.receive_profile_name,
        "candidates": len(declaration.candidates),
        "optical_configurations": len(declaration.optical_configurations),
        "validation_windows": len(source_report.windows),
        "physical_profile_instances": physical_instances,
        "resource_grids": list(grid_rows),
        "channel_frames_evaluated": 0,
        "training_authorized": False,
        "test_split_opened": False,
    }


__all__ = [
    "LONGER_BLOCK_FRONTIER_DECLARATION_SCHEMA",
    "LongerBlockCandidate",
    "LongerBlockFrontierDeclaration",
    "LongerBlockFrontierError",
    "candidate_grid_row",
    "config_for_candidate",
    "load_longer_block_frontier_declaration",
    "structural_longer_block_dry_run",
]
