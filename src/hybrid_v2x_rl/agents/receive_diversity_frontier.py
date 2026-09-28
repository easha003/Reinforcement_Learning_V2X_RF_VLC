"""Frozen declaration contract for the receive-diversity feasibility frontier.

Loading this module validates the experiment design only. It never evaluates a
channel frame, samples a fading branch, opens test data, or authorizes training.
The later physical implementation and executor must consume the profiles emitted
here instead of constructing configurations after seeing results.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final, Literal, cast

from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    SystemFeasibilityFrontierDeclaration,
    load_system_feasibility_frontier_declaration,
)
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.core.errors import HybridV2XError

RECEIVE_DIVERSITY_FRONTIER_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.receive-diversity-frontier-declaration.v1"
)
CombiningRule = Literal["none", "maximum-ratio-combining"]


class ReceiveDiversityFrontierError(HybridV2XError):
    """The frozen receive-diversity declaration is malformed or has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys or any(
        not isinstance(key, str) for key in value
    ):
        raise ReceiveDiversityFrontierError(
            f"{name} fields do not match the frozen declaration schema"
        )
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise ReceiveDiversityFrontierError(f"{name} must be an array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiveDiversityFrontierError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ReceiveDiversityFrontierError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _number(value: object, *, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ReceiveDiversityFrontierError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReceiveDiversityFrontierError(f"{name} must be finite")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ReceiveDiversityFrontierError(f"{name} must be boolean")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ReceiveDiversityFrontierError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve(root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name))
    return (supplied if supplied.is_absolute() else root / supplied).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    source_id: str
    url: str
    role: str


@dataclass(frozen=True, slots=True)
class BranchCorrelationLevel:
    name: str
    coefficient: float
    headline: bool
    evidence_role: str

    @property
    def equivalent_ecc(self) -> float:
        return self.coefficient**2


@dataclass(frozen=True, slots=True)
class ImplementationLossLevel:
    name: str
    loss_db: float
    headline: bool
    evidence_role: str

    @property
    def secondary_branch_power_factor(self) -> float:
        return math.pow(10.0, -self.loss_db / 10.0)


@dataclass(frozen=True, slots=True)
class ReceiveDiversityProfile:
    name: str
    antenna_count: int
    combining_rule: CombiningRule
    branch_correlation: float | None
    implementation_loss_db: float | None
    headline: bool
    authorizes_training: bool


@dataclass(frozen=True, slots=True)
class ReceiveDiversityFrontierDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    miss_budget: float
    densities: tuple[float, ...]
    source_frontier: SystemFeasibilityFrontierDeclaration
    calibration_sources: tuple[CalibrationSource, ...]
    control_profile: ReceiveDiversityProfile
    antenna_count: int
    combining_rule: CombiningRule
    channel_state_information: str
    combined_snr_rule: str
    branch_generation_rule: str
    branch_power_normalization: str
    receiver_noise: str
    correlation_parameter: str
    ecc_relation: str
    shared_mechanisms: tuple[str, ...]
    independently_innovated_mechanism: str
    correlation_levels: tuple[BranchCorrelationLevel, ...]
    implementation_loss_application: str
    loss_levels: tuple[ImplementationLossLevel, ...]
    output_path: Path
    expected_evaluation_cells_before_screening: int

    @property
    def receive_profiles(self) -> tuple[ReceiveDiversityProfile, ...]:
        intervention_profiles = tuple(
            ReceiveDiversityProfile(
                name=f"rx2-mrc__{correlation.name}__{loss.name}",
                antenna_count=self.antenna_count,
                combining_rule=self.combining_rule,
                branch_correlation=correlation.coefficient,
                implementation_loss_db=loss.loss_db,
                headline=correlation.headline and loss.headline,
                authorizes_training=correlation.headline and loss.headline,
            )
            for correlation, loss in product(self.correlation_levels, self.loss_levels)
        )
        return (self.control_profile, *intervention_profiles)

    @property
    def headline_receive_profile(self) -> ReceiveDiversityProfile:
        matches = tuple(profile for profile in self.receive_profiles if profile.headline)
        if len(matches) != 1:  # pragma: no cover - the loader proves this.
            raise ReceiveDiversityFrontierError(
                "declaration does not have exactly one headline receive profile"
            )
        return matches[0]

    @property
    def evaluation_cells_before_screening(self) -> int:
        return (
            len(self.receive_profiles)
            * len(self.source_frontier.physical_points)
            * len(self.source_frontier.fallback_views)
        )


def _parse_sources(value: object) -> tuple[CalibrationSource, ...]:
    sources: list[CalibrationSource] = []
    for index, raw in enumerate(_sequence(value, name="calibration sources")):
        row = _mapping(
            raw,
            name=f"calibration source {index}",
            keys={"id", "url", "role"},
        )
        url = _text(row["url"], name="calibration source URL")
        if not url.startswith("https://doi.org/"):
            raise ReceiveDiversityFrontierError(
                "calibration sources must use stable DOI URLs"
            )
        sources.append(
            CalibrationSource(
                source_id=_text(row["id"], name="calibration source ID"),
                url=url,
                role=_text(row["role"], name="calibration source role"),
            )
        )
    result = tuple(sources)
    if tuple(source.source_id for source in result) != (
        "abbas-karedal-tufvesson-2013",
        "sathyanarayanan-et-al-2026",
    ):
        raise ReceiveDiversityFrontierError(
            "receive diversity must preserve both ordered calibration sources"
        )
    return result


def _parse_correlations(value: object) -> tuple[BranchCorrelationLevel, ...]:
    section = _mapping(value, name="branch-correlation axis", keys={"levels"})
    levels: list[BranchCorrelationLevel] = []
    for index, raw in enumerate(_sequence(section["levels"], name="correlation levels")):
        row = _mapping(
            raw,
            name=f"correlation level {index}",
            keys={"name", "coefficient", "headline", "evidence_role"},
        )
        coefficient = _number(row["coefficient"], name="branch correlation")
        if not 0.0 <= coefficient < 1.0:
            raise ReceiveDiversityFrontierError(
                "branch correlation must lie in [0, 1)"
            )
        levels.append(
            BranchCorrelationLevel(
                name=_text(row["name"], name="correlation name"),
                coefficient=coefficient,
                headline=_boolean(row["headline"], name="correlation headline flag"),
                evidence_role=_text(row["evidence_role"], name="correlation evidence role"),
            )
        )
    result = tuple(levels)
    if (
        len(result) < 3
        or tuple(level.coefficient for level in result)
        != tuple(sorted(level.coefficient for level in result))
        or len({level.name for level in result}) != len(result)
        or sum(level.headline for level in result) != 1
        or not math.isclose(result[0].coefficient, 0.0, abs_tol=0.0)
    ):
        raise ReceiveDiversityFrontierError(
            "correlation levels must be unique, increasing from independence, and have one headline"
        )
    headline = next(level for level in result if level.headline)
    if (
        tuple(level.name for level in result)
        != (
            "independent-ideal",
            "low-correlation-hardware-bound",
            "correlated-stress",
        )
        or not math.isclose(headline.equivalent_ecc, 0.03, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(result[-1].coefficient, 0.7, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ReceiveDiversityFrontierError(
            "correlation axis must preserve the independent, ECC, and stress boundaries"
        )
    return result


def _parse_losses(value: object) -> tuple[str, tuple[ImplementationLossLevel, ...]]:
    section = _mapping(
        value,
        name="implementation-loss axis",
        keys={"application", "levels"},
    )
    application = _text(section["application"], name="implementation-loss application")
    levels: list[ImplementationLossLevel] = []
    for index, raw in enumerate(_sequence(section["levels"], name="loss levels")):
        row = _mapping(
            raw,
            name=f"loss level {index}",
            keys={"name", "loss_db", "headline", "evidence_role"},
        )
        loss = _number(row["loss_db"], name="implementation loss")
        if loss < 0.0:
            raise ReceiveDiversityFrontierError(
                "implementation loss must be non-negative"
            )
        levels.append(
            ImplementationLossLevel(
                name=_text(row["name"], name="loss name"),
                loss_db=loss,
                headline=_boolean(row["headline"], name="loss headline flag"),
                evidence_role=_text(row["evidence_role"], name="loss evidence role"),
            )
        )
    result = tuple(levels)
    if (
        len(result) < 3
        or tuple(level.loss_db for level in result)
        != tuple(sorted(level.loss_db for level in result))
        or len({level.name for level in result}) != len(result)
        or sum(level.headline for level in result) != 1
        or not math.isclose(result[0].loss_db, 0.0, abs_tol=0.0)
    ):
        raise ReceiveDiversityFrontierError(
            "loss levels must be unique, increasing from zero, and have one headline"
        )
    headline = next(level for level in result if level.headline)
    if (
        tuple(level.name for level in result)
        != ("integrated-zero-loss", "short-cable-loss", "long-cable-loss")
        or not math.isclose(headline.loss_db, 3.5, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(result[-1].loss_db, 7.0, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise ReceiveDiversityFrontierError(
            "loss axis must preserve the zero, short-cable, and long-cable boundaries"
        )
    return application, result


def load_receive_diversity_frontier_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> ReceiveDiversityFrontierDeclaration:
    """Load and fail-closed validate the pre-result diversity declaration."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = _resolve(root, str(path), name="diversity declaration")
    payload = load_yaml_file(declaration_path)
    top = _mapping(
        payload,
        name="diversity declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "receive_diversity",
            "execution",
            "decision",
        },
    )
    if top["schema"] != RECEIVE_DIVERSITY_FRONTIER_DECLARATION_SCHEMA:
        raise ReceiveDiversityFrontierError("diversity declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="diversity objective",
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
            objective["densities_vehicles_per_lane_km"], name="densities"
        )
    )
    if (
        not 0.0 < miss_budget < 1.0
        or densities != tuple(sorted(set(densities)))
        or any(density <= 0.0 for density in densities)
        or objective["required_split"] != "validation"
        or _boolean(objective["test_split_opened"], name="test-split flag")
    ):
        raise ReceiveDiversityFrontierError(
            "objective must use ordered validation-only data with test closed"
        )
    _text(objective["interpretation"], name="objective interpretation")

    evidence = _mapping(
        top["evidence"],
        name="diversity evidence",
        keys={
            "source_system_frontier",
            "actor_used",
            "checkpoint_used",
            "calibration_sources",
        },
    )
    if (
        _boolean(evidence["actor_used"], name="actor-used flag")
        or _boolean(evidence["checkpoint_used"], name="checkpoint-used flag")
    ):
        raise ReceiveDiversityFrontierError(
            "frontier declaration cannot use an actor or checkpoint"
        )
    source = _mapping(
        evidence["source_system_frontier"],
        name="source system frontier",
        keys={"path", "sha256"},
    )
    source_path = _resolve(root, source["path"], name="source system frontier path")
    source_sha256 = _digest(source["sha256"], name="source system frontier SHA-256")
    source_frontier = load_system_feasibility_frontier_declaration(
        source_path,
        project_root=root,
        verify_evidence=verify_evidence,
    )
    if source_frontier.sha256 != source_sha256:
        raise ReceiveDiversityFrontierError(
            "source system-frontier declaration digest has drifted"
        )
    if (
        not math.isclose(source_frontier.miss_budget, miss_budget, abs_tol=0.0)
        or source_frontier.densities != densities
    ):
        raise ReceiveDiversityFrontierError(
            "diversity objective differs from the source system frontier"
        )
    calibration_sources = _parse_sources(evidence["calibration_sources"])

    diversity = _mapping(
        top["receive_diversity"],
        name="receive-diversity model",
        keys={
            "control",
            "intervention",
            "branch_correlation",
            "implementation_loss",
        },
    )
    control = _mapping(
        diversity["control"],
        name="SISO control",
        keys={
            "name",
            "antenna_count",
            "combining_rule",
            "headline",
            "authorizes_training",
        },
    )
    if (
        _integer(control["antenna_count"], name="control antenna count") != 1
        or control["name"] != "rx1-siso-control"
        or control["combining_rule"] != "none"
        or _boolean(control["headline"], name="control headline flag")
        or _boolean(control["authorizes_training"], name="control authorization flag")
    ):
        raise ReceiveDiversityFrontierError(
            "control must be non-authorizing one-antenna SISO"
        )
    control_profile = ReceiveDiversityProfile(
        name=_text(control["name"], name="control name"),
        antenna_count=1,
        combining_rule="none",
        branch_correlation=None,
        implementation_loss_db=None,
        headline=False,
        authorizes_training=False,
    )

    intervention = _mapping(
        diversity["intervention"],
        name="receive-diversity intervention",
        keys={
            "antenna_count",
            "combining_rule",
            "channel_state_information",
            "combined_snr_rule",
            "branch_generation_rule",
            "branch_power_normalization",
            "receiver_noise",
            "correlation_parameter",
            "ecc_relation",
            "shared_mechanisms",
            "independently_innovated_mechanism",
        },
    )
    antenna_count = _integer(
        intervention["antenna_count"], name="intervention antenna count"
    )
    combining_rule = _text(intervention["combining_rule"], name="combining rule")
    if antenna_count != 2 or combining_rule != "maximum-ratio-combining":
        raise ReceiveDiversityFrontierError(
            "v1 intervention must use two receive antennas with MRC"
        )
    csi = _text(
        intervention["channel_state_information"], name="channel-state information"
    )
    if csi != "perfect-per-attempt":
        raise ReceiveDiversityFrontierError(
            "v1 MRC must state its perfect per-attempt CSI assumption"
        )
    shared = tuple(
        _text(value, name="shared mechanism")
        for value in _sequence(
            intervention["shared_mechanisms"], name="shared mechanisms"
        )
    )
    expected_shared = {
        "large-scale path loss",
        "propagation state and vehicle blockage",
        "lognormal shadowing",
        "configured Rician specular component",
        "RF contention",
        "receiver half-duplex exposure",
    }
    if set(shared) != expected_shared or len(shared) != len(expected_shared):
        raise ReceiveDiversityFrontierError(
            "shared mechanisms must preserve every non-small-scale failure source"
        )
    independently_innovated = _text(
        intervention["independently_innovated_mechanism"],
        name="independently innovated mechanism",
    )
    if independently_innovated != "small-scale fading on the second receive branch":
        raise ReceiveDiversityFrontierError(
            "only second-branch small-scale fading may receive a new innovation"
        )
    snr_rule = _text(intervention["combined_snr_rule"], name="combined SNR rule")
    branch_generation_rule = _text(
        intervention["branch_generation_rule"], name="branch generation rule"
    )
    branch_power_normalization = _text(
        intervention["branch_power_normalization"], name="branch power normalization"
    )
    receiver_noise = _text(intervention["receiver_noise"], name="receiver noise")
    if (
        snr_rule
        != "gamma_mrc = gamma_1 + 10^(-implementation_loss_db/10) * gamma_2"
        or branch_generation_rule
        != "h_2 = rho*h_1 + sqrt(1-rho^2)*z_2 for diffuse unit-power complex innovations, preserving configured Rician power normalization"
        or branch_power_normalization
        != "each pre-loss branch has the same unit mean small-scale power as the existing SISO branch; no antenna-gain bonus is added"
        or receiver_noise != "independent equal-variance noise across receive chains"
    ):
        raise ReceiveDiversityFrontierError(
            "v1 MRC equations and receiver assumptions have drifted"
        )
    correlation_parameter = _text(
        intervention["correlation_parameter"], name="correlation parameter"
    )
    ecc_relation = _text(intervention["ecc_relation"], name="ECC relation")
    correlations = _parse_correlations(diversity["branch_correlation"])
    loss_application, losses = _parse_losses(diversity["implementation_loss"])

    execution = _mapping(
        top["execution"],
        name="diversity execution",
        keys={
            "expansion",
            "stage_order",
            "screening_rule",
            "expected_receive_profiles",
            "expected_source_physical_points",
            "expected_source_fallback_views",
            "expected_evaluation_cells_before_screening",
            "no_adaptive_axis_expansion",
            "no_training",
            "no_test_split",
            "output_path",
        },
    )
    _text(execution["expansion"], name="expansion rule")
    stages = tuple(
        _text(value, name="execution stage")
        for value in _sequence(execution["stage_order"], name="stage order")
    )
    if stages != (
        "propagation-only necessary-condition screen",
        "certificate-aware pair-local joint feasibility for every surviving predeclared profile",
    ):
        raise ReceiveDiversityFrontierError(
            "execution stages must preserve screening before joint search"
        )
    _text(execution["screening_rule"], name="screening rule")
    for flag in ("no_adaptive_axis_expansion", "no_training", "no_test_split"):
        if not _boolean(execution[flag], name=flag):
            raise ReceiveDiversityFrontierError(
                "diversity execution safety flags must be true"
            )

    expected_profiles = 1 + len(correlations) * len(losses)
    expected_points = len(source_frontier.physical_points)
    expected_views = len(source_frontier.fallback_views)
    expected_cells = expected_profiles * expected_points * expected_views
    if (
        _integer(execution["expected_receive_profiles"], name="expected profiles")
        != expected_profiles
        or _integer(
            execution["expected_source_physical_points"], name="expected source points"
        )
        != expected_points
        or _integer(
            execution["expected_source_fallback_views"], name="expected fallback views"
        )
        != expected_views
        or _integer(
            execution["expected_evaluation_cells_before_screening"],
            name="expected evaluation cells",
        )
        != expected_cells
    ):
        raise ReceiveDiversityFrontierError(
            "declared diversity grid size does not reconcile"
        )

    decision = _mapping(
        top["decision"],
        name="diversity decision",
        keys={
            "headline_receive_profile",
            "training_authorization",
            "sensitivity_interpretation",
            "claim_boundary",
        },
    )
    expected_headline = (
        "rx2-mrc__"
        f"{next(level.name for level in correlations if level.headline)}__"
        f"{next(level.name for level in losses if level.headline)}"
    )
    if _text(decision["headline_receive_profile"], name="headline profile") != expected_headline:
        raise ReceiveDiversityFrontierError(
            "decision headline does not match the declared headline axes"
        )
    _text(decision["training_authorization"], name="training-authorization rule")
    _text(decision["sensitivity_interpretation"], name="sensitivity interpretation")
    _text(decision["claim_boundary"], name="claim boundary")

    declaration = ReceiveDiversityFrontierDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        miss_budget=miss_budget,
        densities=densities,
        source_frontier=source_frontier,
        calibration_sources=calibration_sources,
        control_profile=control_profile,
        antenna_count=antenna_count,
        combining_rule=cast(CombiningRule, combining_rule),
        channel_state_information=csi,
        combined_snr_rule=snr_rule,
        branch_generation_rule=branch_generation_rule,
        branch_power_normalization=branch_power_normalization,
        receiver_noise=receiver_noise,
        correlation_parameter=correlation_parameter,
        ecc_relation=ecc_relation,
        shared_mechanisms=shared,
        independently_innovated_mechanism=independently_innovated,
        correlation_levels=correlations,
        implementation_loss_application=loss_application,
        loss_levels=losses,
        output_path=_resolve(root, execution["output_path"], name="output path"),
        expected_evaluation_cells_before_screening=expected_cells,
    )
    if declaration.evaluation_cells_before_screening != expected_cells:
        raise ReceiveDiversityFrontierError(
            "expanded diversity grid differs from the frozen declaration"
        )
    return declaration


def structural_receive_diversity_dry_run(
    declaration: ReceiveDiversityFrontierDeclaration,
) -> dict[str, object]:
    """Summarize the frozen grid without reading frames or evaluating channels."""

    return {
        "schema": RECEIVE_DIVERSITY_FRONTIER_DECLARATION_SCHEMA,
        "declaration_sha256": declaration.sha256,
        "source_frontier_sha256": declaration.source_frontier.sha256,
        "receive_profiles": len(declaration.receive_profiles),
        "source_physical_points": len(declaration.source_frontier.physical_points),
        "source_fallback_views": len(declaration.source_frontier.fallback_views),
        "evaluation_cells_before_screening": declaration.evaluation_cells_before_screening,
        "headline_receive_profile": declaration.headline_receive_profile.name,
        "channel_frames_evaluated": 0,
        "training_authorized": False,
        "test_split_opened": False,
    }


__all__ = [
    "BranchCorrelationLevel",
    "CalibrationSource",
    "ImplementationLossLevel",
    "RECEIVE_DIVERSITY_FRONTIER_DECLARATION_SCHEMA",
    "ReceiveDiversityFrontierDeclaration",
    "ReceiveDiversityFrontierError",
    "ReceiveDiversityProfile",
    "load_receive_diversity_frontier_declaration",
    "structural_receive_diversity_dry_run",
]
