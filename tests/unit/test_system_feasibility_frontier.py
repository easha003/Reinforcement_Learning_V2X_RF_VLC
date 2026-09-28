"""Frozen declaration contract for the pair-local feasibility frontier."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA,
    SystemFeasibilityFrontierError,
    load_system_feasibility_frontier_declaration,
)
from hybrid_v2x_rl.config.loader import load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path(
    "configs/evaluation/pair_local_system_feasibility_frontier.yaml"
)


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write_declaration(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "frontier.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_declaration_expands_the_frozen_grid() -> None:
    declaration = load_system_feasibility_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )

    assert len(declaration.sha256) == 64
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.densities == (10.0, 20.0, 30.0)
    assert declaration.environment_seed == 20260728
    assert declaration.selection_window_slots == 200
    assert [level.candidate_resources for level in declaration.rf_capacities] == [
        400,
        800,
        1600,
    ]
    assert [level.name for level in declaration.sensing_bands] == [
        "nominal",
        "pessimistic",
        "optimistic",
    ]
    assert [level.receiver_fov_deg for level in declaration.optical_configurations] == [
        60.0,
        30.0,
    ]
    assert len(declaration.physical_points) == 18
    assert len(declaration.evaluation_cells) == 36
    assert len({point.point_id for point in declaration.physical_points}) == 18
    assert len({cell.cell_id for cell in declaration.evaluation_cells}) == 36
    assert declaration.headline_point.point_id == (
        "rf-capacity-1x__wide-60deg__nominal"
    )


def test_only_contract_fallback_can_authorize_training() -> None:
    declaration = load_system_feasibility_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )

    fallback = {view.mode: view for view in declaration.fallback_views}
    assert fallback["contract"].authorizes_training is True
    assert fallback["contract"].diagnostic_only is False
    assert fallback["all_usable"].authorizes_training is False
    assert fallback["all_usable"].diagnostic_only is True


def test_canonical_evidence_is_verified_when_artifact_is_available() -> None:
    payload = _payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    window = evidence["window_source"]
    assert isinstance(window, dict)
    artifact = PROJECT_ROOT / str(window["path"])
    if not artifact.is_file():
        pytest.skip("frozen evaluation artifact is not present in this checkout")

    declaration = load_system_feasibility_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=True,
    )

    assert declaration.window_source == artifact.resolve()


def test_candidate_resource_arithmetic_is_fail_closed(tmp_path: Path) -> None:
    payload = _payload()
    axes = payload["axes"]
    assert isinstance(axes, dict)
    rf_capacity = axes["rf_capacity"]
    assert isinstance(rf_capacity, dict)
    levels = rf_capacity["levels"]
    assert isinstance(levels, list)
    assert isinstance(levels[1], dict)
    levels[1]["candidate_resources"] = 801

    with pytest.raises(SystemFeasibilityFrontierError, match="candidate resources"):
        load_system_feasibility_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_sensing_band_values_cannot_drift(tmp_path: Path) -> None:
    payload = _payload()
    axes = payload["axes"]
    assert isinstance(axes, dict)
    sensing = axes["sensing_band"]
    assert isinstance(sensing, dict)
    levels = sensing["levels"]
    assert isinstance(levels, list)
    assert isinstance(levels[2], dict)
    levels[2]["sensing_reliability"] = 0.96

    with pytest.raises(SystemFeasibilityFrontierError, match="declared band"):
        load_system_feasibility_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_baseline_policy_environment_scope_cannot_drift(tmp_path: Path) -> None:
    payload = _payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    evidence["baseline_policy_environment_scope_hash"] = "c" * 64

    with pytest.raises(SystemFeasibilityFrontierError, match="scope has drifted"):
        load_system_feasibility_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_diagnostic_fallback_cannot_authorize_training(tmp_path: Path) -> None:
    payload = _payload()
    axes = payload["axes"]
    assert isinstance(axes, dict)
    fallback = axes["fallback_view"]
    assert isinstance(fallback, dict)
    levels = fallback["levels"]
    assert isinstance(levels, list)
    assert isinstance(levels[1], dict)
    levels[1]["authorizes_training"] = True

    with pytest.raises(SystemFeasibilityFrontierError, match="all-usable diagnosis"):
        load_system_feasibility_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_declared_grid_size_must_match_expansion(tmp_path: Path) -> None:
    payload = _payload()
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["expected_evaluation_cells"] = 35

    with pytest.raises(SystemFeasibilityFrontierError, match="expanded frontier"):
        load_system_feasibility_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_schema_drift_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    payload["schema"] = f"{SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA}.drift"

    with pytest.raises(SystemFeasibilityFrontierError, match="schema is unsupported"):
        load_system_feasibility_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )
