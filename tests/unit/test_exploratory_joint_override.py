"""Declaration, result, and resume tests for the exploratory joint override."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
import yaml

from hybrid_v2x_rl.agents.exploratory_joint_override import (
    EXPLORATORY_JOINT_PROGRESS_SCHEMA,
    EXPLORATORY_JOINT_RESULT_SCHEMA,
    ExploratoryJointOverrideDeclaration,
    ExploratoryJointOverrideError,
    ExploratoryJointOverrideResult,
    load_exploratory_joint_override_declaration,
    load_exploratory_joint_progress,
    structural_exploratory_joint_dry_run,
    write_exploratory_joint_progress,
)
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    FrontierCellResult,
    cell_verdict,
    density_verdict,
)
from hybrid_v2x_rl.config.loader import load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path("configs/evaluation/exploratory_joint_override.yaml")


def _declaration() -> ExploratoryJointOverrideDeclaration:
    return load_exploratory_joint_override_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _cell_result(
    declaration: ExploratoryJointOverrideDeclaration,
    index: int,
    *,
    worst_mean: float,
) -> FrontierCellResult:
    rows: list[dict[str, object]] = []
    for density in declaration.densities:
        mean = worst_mean if density == 20.0 else worst_mean / 2.0
        rows.append(
            {
                "density_vehicles_per_lane_km": density,
                "mean_pair_local_candidate_conditional_miss_risk": mean,
                "pair_local_candidate_mean_meets_budget": mean <= declaration.miss_budget,
                "certified_lower_bound_exceeds_budget": mean > declaration.miss_budget,
                "all_frames_optimality_proven": False,
            }
        )
    verdicts = tuple(density_verdict(row, miss_budget=declaration.miss_budget) for row in rows)
    return FrontierCellResult(
        cell=declaration.selected_cells[index],
        config_hash="a" * 64,
        policy_environment_scope_hash="b" * 64,
        density_rows=tuple(rows),
        campaign={},
        density_verdicts=verdicts,
        verdict=cell_verdict(verdicts),
    )


def _complete_cells(
    declaration: ExploratoryJointOverrideDeclaration,
) -> tuple[FrontierCellResult, ...]:
    return (
        _cell_result(declaration, 0, worst_mean=2.0e-3),
        _cell_result(declaration, 1, worst_mean=2.2e-3),
        _cell_result(declaration, 2, worst_mean=1.8e-3),
    )


def test_canonical_declaration_records_explicit_override_and_selected_pair() -> None:
    declaration = _declaration()

    assert declaration.payload_bytes == 300
    assert declaration.deadline_s == pytest.approx(0.010)
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.propagation_mean == pytest.approx(1.182919226271053e-4)
    assert declaration.selected_profile.role == "optimistic-sensitivity"
    assert declaration.selected_optical_name == "wide-60deg"
    assert declaration.selected_capacity_name == "rf-capacity-4x"
    assert declaration.selected_sensing_band_names == (
        "nominal",
        "pessimistic",
        "optimistic",
    )
    assert len(declaration.selected_cells) == 3


def test_override_flag_cannot_be_removed_after_freeze(tmp_path: Path) -> None:
    payload = _payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    evidence["user_directed_exploratory_override"] = False

    with pytest.raises(ExploratoryJointOverrideError, match="must be explicit"):
        load_exploratory_joint_override_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_complete_joint_result_authorizes_only_exploratory_training() -> None:
    declaration = _declaration()
    result = ExploratoryJointOverrideResult(
        declaration=declaration,
        cells=_complete_cells(declaration),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert result.as_dict()["schema"] == EXPLORATORY_JOINT_RESULT_SCHEMA
    assert decision["exact_system_target_met"] is False
    assert decision["exact_feasibility_claim_allowed"] is False
    assert decision["best_joint_cell_id"] == declaration.selected_cells[2].cell_id
    assert decision["nominal_training_cell_id"] == declaration.selected_cells[0].cell_id
    assert decision["exploratory_training_authorized"] is True
    assert decision["training_performed"] is False
    assert decision["test_split_opened"] is False


def test_progress_round_trips_an_ordered_cell_prefix(tmp_path: Path) -> None:
    declaration = _declaration()
    prefix = _complete_cells(declaration)[:2]
    path = write_exploratory_joint_progress(
        tmp_path / "override.progress.json",
        declaration=declaration,
        cells=prefix,
    )

    assert load_exploratory_joint_progress(path, declaration=declaration) == prefix
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == EXPLORATORY_JOINT_PROGRESS_SCHEMA
    assert payload["training_run_performed"] is False
    assert payload["test_split_opened"] is False


def test_progress_rejects_provenance_drift(tmp_path: Path) -> None:
    declaration = _declaration()
    path = write_exploratory_joint_progress(
        tmp_path / "override.progress.json",
        declaration=declaration,
        cells=_complete_cells(declaration)[:1],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["combined_result_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExploratoryJointOverrideError, match="provenance has drifted"):
        load_exploratory_joint_progress(path, declaration=declaration)


def test_structural_dry_run_opens_no_frames_or_test_data() -> None:
    declaration = _declaration()
    if not declaration.combined_result_artifact.path.is_file():
        pytest.skip("frozen evidence and validation artifacts are not present in this checkout")
    report = structural_exploratory_joint_dry_run(
        declaration,
        project_root=PROJECT_ROOT,
    )

    assert report["validation_windows"] == 9
    assert len(cast(list[object], report["cells"])) == 3
    assert report["channel_frames_evaluated"] == 0
    assert report["training_performed"] is False
    assert report["test_split_opened"] is False
