"""Frozen declaration contract for the pair-local feasibility frontier.

This module validates the experiment before any frontier result exists.  It
does not evaluate a channel, search a joint action, write an artifact, or open
the test split.  The later runner must consume the expanded cells emitted here
rather than constructing an adaptive grid from observed results.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final, Literal, cast

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand, headline_parameters
from hybrid_v2x_rl.config.loader import load_config, load_yaml_file
from hybrid_v2x_rl.core.errors import HybridV2XError

SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-declaration.v1"
)
FallbackMode = Literal["contract", "all_usable"]


class SystemFeasibilityFrontierError(HybridV2XError):
    """The frozen system-feasibility declaration is malformed or has drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(
    value: object,
    *,
    name: str,
    keys: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys or any(
        not isinstance(key, str) for key in value
    ):
        raise SystemFeasibilityFrontierError(
            f"{name} fields do not match the frozen declaration schema"
        )
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, name: str) -> tuple[object, ...]:
    if not isinstance(value, list | tuple):
        raise SystemFeasibilityFrontierError(f"{name} must be an array")
    return tuple(value)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SystemFeasibilityFrontierError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 1) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise SystemFeasibilityFrontierError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _number(value: object, *, name: str, positive: bool = False) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SystemFeasibilityFrontierError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise SystemFeasibilityFrontierError(f"{name} must be {qualifier}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise SystemFeasibilityFrontierError(f"{name} must be boolean")
    return value


def _digest(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SystemFeasibilityFrontierError(f"{name} must be a lowercase SHA-256")
    return result


def _resolve_existing(project_root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name))
    resolved = supplied if supplied.is_absolute() else project_root / supplied
    resolved = resolved.resolve(strict=False)
    if not resolved.is_file():
        raise SystemFeasibilityFrontierError(
            f"{name} does not identify an existing file",
            artifact_path=resolved,
        )
    return resolved


def _resolve_path(project_root: Path, value: object, *, name: str) -> Path:
    supplied = Path(_text(value, name=name))
    resolved = supplied if supplied.is_absolute() else project_root / supplied
    return resolved.resolve(strict=False)


@dataclass(frozen=True, slots=True)
class RFCapacityLevel:
    name: str
    subchannels: int
    candidate_resources: int
    equivalent_system_bandwidth_mhz: float
    headline: bool


@dataclass(frozen=True, slots=True)
class SensingBandLevel:
    band: SensitivityBand
    sensing_reliability: float
    headline: bool

    @property
    def name(self) -> str:
        return self.band.value


@dataclass(frozen=True, slots=True)
class OpticalConfigurationLevel:
    name: str
    additional_config_layers: tuple[Path, ...]
    receiver_fov_deg: float
    headline: bool


@dataclass(frozen=True, slots=True)
class FallbackView:
    name: str
    mode: FallbackMode
    diagnostic_only: bool
    authorizes_training: bool


@dataclass(frozen=True, slots=True)
class FrontierPhysicalPoint:
    rf_capacity: RFCapacityLevel
    sensing_band: SensingBandLevel
    optical_configuration: OpticalConfigurationLevel

    @property
    def point_id(self) -> str:
        return "__".join(
            (
                self.rf_capacity.name,
                self.optical_configuration.name,
                self.sensing_band.name,
            )
        )


@dataclass(frozen=True, slots=True)
class FrontierEvaluationCell:
    physical_point: FrontierPhysicalPoint
    fallback_view: FallbackView

    @property
    def cell_id(self) -> str:
        return f"{self.physical_point.point_id}__{self.fallback_view.name}"


@dataclass(frozen=True, slots=True)
class SystemFeasibilityFrontierDeclaration:
    path: Path
    sha256: str
    frozen_date: str
    miss_budget: float
    densities: tuple[float, ...]
    environment_seed: int
    window_source: Path
    window_source_sha256: str
    baseline_policy_environment_scope_hash: str
    base_config_layers: tuple[Path, ...]
    selection_window_slots: int
    rf_capacities: tuple[RFCapacityLevel, ...]
    sensing_bands: tuple[SensingBandLevel, ...]
    optical_configurations: tuple[OpticalConfigurationLevel, ...]
    fallback_views: tuple[FallbackView, ...]
    exact_assignment_cap: int
    max_search_iterations: int
    output_path: Path

    @property
    def physical_points(self) -> tuple[FrontierPhysicalPoint, ...]:
        return tuple(
            FrontierPhysicalPoint(capacity, sensing, optical)
            for capacity, optical, sensing in product(
                self.rf_capacities,
                self.optical_configurations,
                self.sensing_bands,
            )
        )

    @property
    def evaluation_cells(self) -> tuple[FrontierEvaluationCell, ...]:
        return tuple(
            FrontierEvaluationCell(point, fallback)
            for point, fallback in product(
                self.physical_points,
                self.fallback_views,
            )
        )

    @property
    def headline_point(self) -> FrontierPhysicalPoint:
        matches = tuple(
            point
            for point in self.physical_points
            if point.rf_capacity.headline
            and point.sensing_band.headline
            and point.optical_configuration.headline
        )
        if len(matches) != 1:  # pragma: no cover - loader proves this.
            raise SystemFeasibilityFrontierError(
                "declaration does not have exactly one headline point"
            )
        return matches[0]


def _parse_capacity_levels(
    payload: Mapping[str, object],
) -> tuple[int, tuple[RFCapacityLevel, ...]]:
    section = _mapping(
        payload,
        name="RF-capacity axis",
        keys={"selection_window_slots", "levels"},
    )
    slots = _integer(
        section["selection_window_slots"],
        name="selection_window_slots",
    )
    levels: list[RFCapacityLevel] = []
    for index, raw in enumerate(_sequence(section["levels"], name="RF-capacity levels")):
        row = _mapping(
            raw,
            name=f"RF-capacity level {index}",
            keys={
                "name",
                "subchannels",
                "candidate_resources",
                "equivalent_system_bandwidth_mhz",
                "headline",
            },
        )
        subchannels = _integer(row["subchannels"], name="RF subchannels")
        resources = _integer(
            row["candidate_resources"],
            name="candidate resources",
        )
        if resources != subchannels * slots:
            raise SystemFeasibilityFrontierError(
                "candidate resources must equal subchannels times selection-window slots"
            )
        levels.append(
            RFCapacityLevel(
                name=_text(row["name"], name="RF-capacity name"),
                subchannels=subchannels,
                candidate_resources=resources,
                equivalent_system_bandwidth_mhz=_number(
                    row["equivalent_system_bandwidth_mhz"],
                    name="equivalent system bandwidth",
                    positive=True,
                ),
                headline=_boolean(row["headline"], name="RF headline flag"),
            )
        )
    result = tuple(levels)
    if (
        len(result) < 2
        or tuple(level.subchannels for level in result)
        != tuple(sorted(level.subchannels for level in result))
        or len({level.name for level in result}) != len(result)
        or sum(level.headline for level in result) != 1
    ):
        raise SystemFeasibilityFrontierError(
            "RF-capacity levels must be unique, increasing, and have one headline"
        )
    return slots, result


def _parse_sensing_levels(payload: object) -> tuple[SensingBandLevel, ...]:
    section = _mapping(
        payload,
        name="sensing-band axis",
        keys={"levels"},
    )
    levels: list[SensingBandLevel] = []
    for index, raw in enumerate(_sequence(section["levels"], name="sensing levels")):
        row = _mapping(
            raw,
            name=f"sensing level {index}",
            keys={"name", "sensing_reliability", "headline"},
        )
        try:
            band = SensitivityBand(_text(row["name"], name="sensing-band name"))
        except ValueError as error:
            raise SystemFeasibilityFrontierError(
                "sensing-band name is not declared by the RF model"
            ) from error
        reliability = _number(
            row["sensing_reliability"],
            name="sensing reliability",
        )
        expected = headline_parameters(band).sensing_reliability
        if not math.isclose(reliability, expected, rel_tol=0.0, abs_tol=1e-15):
            raise SystemFeasibilityFrontierError(
                "sensing reliability differs from the declared band"
            )
        levels.append(
            SensingBandLevel(
                band=band,
                sensing_reliability=reliability,
                headline=_boolean(row["headline"], name="sensing headline flag"),
            )
        )
    result = tuple(levels)
    if (
        set(level.band for level in result) != set(SensitivityBand)
        or sum(level.headline for level in result) != 1
    ):
        raise SystemFeasibilityFrontierError(
            "sensing axis must contain every declared band and one headline"
        )
    return result


def _parse_optical_levels(
    payload: object,
    *,
    project_root: Path,
    base_layers: tuple[Path, ...],
    miss_budget: float,
) -> tuple[OpticalConfigurationLevel, ...]:
    section = _mapping(
        payload,
        name="optical axis",
        keys={"levels"},
    )
    levels: list[OpticalConfigurationLevel] = []
    for index, raw in enumerate(_sequence(section["levels"], name="optical levels")):
        row = _mapping(
            raw,
            name=f"optical level {index}",
            keys={
                "name",
                "additional_config_layers",
                "receiver_fov_deg",
                "headline",
            },
        )
        additional = tuple(
            _resolve_existing(
                project_root,
                value,
                name=f"optical level {index} config layer",
            )
            for value in _sequence(
                row["additional_config_layers"],
                name="additional optical config layers",
            )
        )
        fov = _number(row["receiver_fov_deg"], name="receiver FOV", positive=True)
        config = load_config((*base_layers, *additional), project_root=project_root)
        if not math.isclose(config.vlc.receiver_fov_deg, fov, rel_tol=0.0, abs_tol=1e-12):
            raise SystemFeasibilityFrontierError(
                "named optical configuration does not produce its declared FOV"
            )
        if not math.isclose(
            config.service.miss_budget,
            miss_budget,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise SystemFeasibilityFrontierError(
                "frontier config changes the frozen miss budget"
            )
        levels.append(
            OpticalConfigurationLevel(
                name=_text(row["name"], name="optical configuration name"),
                additional_config_layers=additional,
                receiver_fov_deg=fov,
                headline=_boolean(row["headline"], name="optical headline flag"),
            )
        )
    result = tuple(levels)
    if (
        len(result) < 2
        or len({level.name for level in result}) != len(result)
        or sum(level.headline for level in result) != 1
    ):
        raise SystemFeasibilityFrontierError(
            "optical levels must be unique and have exactly one headline"
        )
    return result


def _parse_fallback_views(payload: object) -> tuple[FallbackView, ...]:
    section = _mapping(
        payload,
        name="fallback axis",
        keys={"levels"},
    )
    levels: list[FallbackView] = []
    for index, raw in enumerate(_sequence(section["levels"], name="fallback views")):
        row = _mapping(
            raw,
            name=f"fallback view {index}",
            keys={"name", "mode", "diagnostic_only", "authorizes_training"},
        )
        mode_text = _text(row["mode"], name="fallback mode")
        if mode_text not in {"contract", "all_usable"}:
            raise SystemFeasibilityFrontierError("fallback mode is unsupported")
        levels.append(
            FallbackView(
                name=_text(row["name"], name="fallback view name"),
                mode=cast(FallbackMode, mode_text),
                diagnostic_only=_boolean(
                    row["diagnostic_only"],
                    name="fallback diagnostic flag",
                ),
                authorizes_training=_boolean(
                    row["authorizes_training"],
                    name="fallback authorization flag",
                ),
            )
        )
    result = tuple(levels)
    by_mode = {level.mode: level for level in result}
    if (
        set(by_mode) != {"contract", "all_usable"}
        or len(result) != 2
        or by_mode["contract"].diagnostic_only
        or not by_mode["contract"].authorizes_training
        or not by_mode["all_usable"].diagnostic_only
        or by_mode["all_usable"].authorizes_training
    ):
        raise SystemFeasibilityFrontierError(
            "fallback views must separate the contract gate from all-usable diagnosis"
        )
    return result


def _verify_window_source(
    path: Path,
    *,
    expected_sha256: str,
    expected_schema: str,
    environment_seed: int,
    policy_environment_scope_hash: str,
    densities: tuple[float, ...],
    windows_per_density: int,
    frames_per_window: int,
) -> None:
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise SystemFeasibilityFrontierError(
            "frontier window source SHA-256 has drifted",
            artifact_path=path,
            context={"actual": actual_sha256, "expected": expected_sha256},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemFeasibilityFrontierError(
            "frontier window source is unreadable",
            artifact_path=path,
        ) from error
    if not isinstance(payload, dict):
        raise SystemFeasibilityFrontierError("frontier window source must be an object")
    if (
        payload.get("schema") != expected_schema
        or payload.get("environment_seed") != environment_seed
        or payload.get("policy_environment_scope_hash")
        != policy_environment_scope_hash
    ):
        raise SystemFeasibilityFrontierError(
            "frontier window source provenance differs from the declaration"
        )
    windows = payload.get("sampled_windows")
    if not isinstance(windows, list):
        raise SystemFeasibilityFrontierError(
            "frontier window source has no sampled-window array"
        )
    validation = tuple(
        row for row in windows if isinstance(row, dict) and row.get("split") == "validation"
    )
    density_counts: Counter[float] = Counter()
    for row in validation:
        try:
            density = float(row["density_vehicles_per_lane_km"])
        except (KeyError, TypeError, ValueError) as error:
            raise SystemFeasibilityFrontierError(
                "validation window density is invalid"
            ) from error
        if (
            density not in densities
            or row.get("frames") != frames_per_window
            or not isinstance(row.get("trace_id"), str)
            or "-validation-" not in cast(str, row["trace_id"])
            or not isinstance(row.get("start_frame_index"), int)
        ):
            raise SystemFeasibilityFrontierError(
                "validation window differs from the frozen frontier boundary"
            )
        density_counts[density] += 1
    if density_counts != Counter({density: windows_per_density for density in densities}):
        raise SystemFeasibilityFrontierError(
            "validation windows do not cover every density exactly as declared"
        )


def load_system_feasibility_frontier_declaration(
    path: str | Path,
    *,
    project_root: str | Path,
    verify_evidence: bool = True,
) -> SystemFeasibilityFrontierDeclaration:
    """Load and fail-closed validate the pre-result frontier declaration."""

    root = Path(project_root).expanduser().resolve(strict=False)
    declaration_path = Path(path).expanduser()
    if not declaration_path.is_absolute():
        declaration_path = root / declaration_path
    declaration_path = declaration_path.resolve(strict=False)
    payload = load_yaml_file(declaration_path)
    top = _mapping(
        payload,
        name="frontier declaration",
        keys={
            "schema",
            "frozen_date",
            "objective",
            "evidence",
            "base_config_layers",
            "axes",
            "execution",
            "verdicts",
        },
    )
    if top["schema"] != SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA:
        raise SystemFeasibilityFrontierError("frontier declaration schema is unsupported")

    objective = _mapping(
        top["objective"],
        name="frontier objective",
        keys={
            "miss_budget",
            "densities_vehicles_per_lane_km",
            "required_split",
            "test_split_opened",
            "interpretation",
        },
    )
    miss_budget = _number(objective["miss_budget"], name="miss budget", positive=True)
    if miss_budget >= 1.0:
        raise SystemFeasibilityFrontierError("miss budget must lie below one")
    densities = tuple(
        _number(value, name="density", positive=True)
        for value in _sequence(
            objective["densities_vehicles_per_lane_km"],
            name="densities",
        )
    )
    if (
        densities != tuple(sorted(set(densities)))
        or objective["required_split"] != "validation"
        or _boolean(objective["test_split_opened"], name="test-split flag")
    ):
        raise SystemFeasibilityFrontierError(
            "frontier objective must use unique ordered validation densities with test closed"
        )
    _text(objective["interpretation"], name="frontier interpretation")

    base_layers = tuple(
        _resolve_existing(root, value, name="base config layer")
        for value in _sequence(top["base_config_layers"], name="base config layers")
    )
    base_config = load_config(base_layers, project_root=root)
    if not math.isclose(
        base_config.service.miss_budget,
        miss_budget,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SystemFeasibilityFrontierError(
            "base configuration and frontier objective have different budgets"
        )

    axes = _mapping(
        top["axes"],
        name="frontier axes",
        keys={
            "rf_capacity",
            "sensing_band",
            "optical_configuration",
            "fallback_view",
        },
    )
    rf_axis = cast(Mapping[str, object], axes["rf_capacity"])
    slots, capacities = _parse_capacity_levels(rf_axis)
    baseline_collision = headline_parameters()
    headline_capacity = next(level for level in capacities if level.headline)
    if (
        slots != baseline_collision.selection_window_slots
        or headline_capacity.subchannels != baseline_collision.subchannels
        or headline_capacity.candidate_resources
        != baseline_collision.candidate_resources
    ):
        raise SystemFeasibilityFrontierError(
            "headline RF capacity differs from the implemented collision model"
        )
    base_bandwidth_mhz = base_config.rf.bandwidth_hz / 1e6
    for capacity in capacities:
        expected_bandwidth = (
            base_bandwidth_mhz
            * capacity.subchannels
            / baseline_collision.subchannels
        )
        if not math.isclose(
            capacity.equivalent_system_bandwidth_mhz,
            expected_bandwidth,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise SystemFeasibilityFrontierError(
                "equivalent RF bandwidth must scale with the declared resource pool"
            )
    sensing = _parse_sensing_levels(axes["sensing_band"])
    optical = _parse_optical_levels(
        axes["optical_configuration"],
        project_root=root,
        base_layers=base_layers,
        miss_budget=miss_budget,
    )
    fallback = _parse_fallback_views(axes["fallback_view"])

    evidence = _mapping(
        top["evidence"],
        name="frontier evidence",
        keys={
            "environment_seed",
            "window_source",
            "baseline_policy_environment_scope_hash",
            "actor_used",
            "checkpoint_used",
            "normalization",
        },
    )
    environment_seed = _integer(
        evidence["environment_seed"],
        name="environment seed",
        minimum=0,
    )
    scope_hash = _digest(
        evidence["baseline_policy_environment_scope_hash"],
        name="baseline policy-environment scope hash",
    )
    if (
        _boolean(evidence["actor_used"], name="actor-used flag")
        or _boolean(evidence["checkpoint_used"], name="checkpoint-used flag")
    ):
        raise SystemFeasibilityFrontierError(
            "frontier declaration must not use an actor or pre-migration checkpoint"
        )
    _text(evidence["normalization"], name="normalization declaration")
    window = _mapping(
        evidence["window_source"],
        name="window source",
        keys={
            "path",
            "sha256",
            "schema",
            "validation_windows_per_density",
            "frames_per_window",
        },
    )
    window_path = _resolve_path(root, window["path"], name="window source")
    window_sha256 = _digest(window["sha256"], name="window source SHA-256")
    window_schema = _text(window["schema"], name="window source schema")
    windows_per_density = _integer(
        window["validation_windows_per_density"],
        name="validation windows per density",
    )
    frames_per_window = _integer(
        window["frames_per_window"],
        name="frames per window",
    )
    if verify_evidence:
        if not window_path.is_file():
            raise SystemFeasibilityFrontierError(
                "window source does not identify an existing file",
                artifact_path=window_path,
            )
        _verify_window_source(
            window_path,
            expected_sha256=window_sha256,
            expected_schema=window_schema,
            environment_seed=environment_seed,
            policy_environment_scope_hash=scope_hash,
            densities=densities,
            windows_per_density=windows_per_density,
            frames_per_window=frames_per_window,
        )

    execution = _mapping(
        top["execution"],
        name="frontier execution",
        keys={
            "cross_product",
            "expected_physical_points",
            "expected_evaluation_cells",
            "exact_assignment_cap",
            "max_search_iterations",
            "no_adaptive_axis_expansion",
            "no_training",
            "no_test_split",
            "output_path",
        },
    )
    required_true = (
        "cross_product",
        "no_adaptive_axis_expansion",
        "no_training",
        "no_test_split",
    )
    if any(not _boolean(execution[name], name=name) for name in required_true):
        raise SystemFeasibilityFrontierError(
            "frontier execution safety flags must all be true"
        )
    exact_cap = _integer(
        execution["exact_assignment_cap"],
        name="exact assignment cap",
    )
    search_iterations = _integer(
        execution["max_search_iterations"],
        name="search iteration limit",
    )
    output_supplied = Path(_text(execution["output_path"], name="output path"))
    output_path = (
        output_supplied
        if output_supplied.is_absolute()
        else root / output_supplied
    ).resolve(strict=False)
    verdicts = _mapping(
        top["verdicts"],
        name="frontier verdicts",
        keys={
            "density_feasible",
            "density_infeasible",
            "density_inconclusive",
            "cell_feasible",
            "cell_infeasible",
            "cell_inconclusive",
            "training_authorization",
            "fallback_diagnostic",
            "minimal_change_rule",
        },
    )
    for name, value in verdicts.items():
        _text(value, name=f"verdict {name}")

    declaration = SystemFeasibilityFrontierDeclaration(
        path=declaration_path,
        sha256=_sha256(declaration_path),
        frozen_date=_text(top["frozen_date"], name="frozen date"),
        miss_budget=miss_budget,
        densities=densities,
        environment_seed=environment_seed,
        window_source=window_path,
        window_source_sha256=window_sha256,
        baseline_policy_environment_scope_hash=scope_hash,
        base_config_layers=base_layers,
        selection_window_slots=slots,
        rf_capacities=capacities,
        sensing_bands=sensing,
        optical_configurations=optical,
        fallback_views=fallback,
        exact_assignment_cap=exact_cap,
        max_search_iterations=search_iterations,
        output_path=output_path,
    )
    expected_points = _integer(
        execution["expected_physical_points"],
        name="expected physical points",
    )
    expected_cells = _integer(
        execution["expected_evaluation_cells"],
        name="expected evaluation cells",
    )
    if (
        len(declaration.physical_points) != expected_points
        or len(declaration.evaluation_cells) != expected_cells
        or len({point.point_id for point in declaration.physical_points})
        != expected_points
        or len({cell.cell_id for cell in declaration.evaluation_cells})
        != expected_cells
    ):
        raise SystemFeasibilityFrontierError(
            "expanded frontier size or identity differs from the declaration"
        )
    _ = declaration.headline_point
    return declaration


__all__ = [
    "SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA",
    "FallbackView",
    "FrontierEvaluationCell",
    "FrontierPhysicalPoint",
    "OpticalConfigurationLevel",
    "RFCapacityLevel",
    "SensingBandLevel",
    "SystemFeasibilityFrontierDeclaration",
    "SystemFeasibilityFrontierError",
    "load_system_feasibility_frontier_declaration",
]
