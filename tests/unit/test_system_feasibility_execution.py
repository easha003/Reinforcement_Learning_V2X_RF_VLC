"""Executor, certificate verdict, and training-gate contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    SYSTEM_FEASIBILITY_FRONTIER_DRY_RUN_SCHEMA,
    SYSTEM_FEASIBILITY_FRONTIER_RESULT_SCHEMA,
    FrontierCellResult,
    SystemFeasibilityExecutionError,
    SystemFeasibilityFrontierResult,
    cell_verdict,
    density_verdict,
    load_frontier_progress,
    structural_dry_run,
    write_frontier_progress,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    FrontierEvaluationCell,
    SystemFeasibilityFrontierDeclaration,
    load_system_feasibility_frontier_declaration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path(
    "configs/evaluation/full_carrier_system_feasibility_frontier.yaml"
)
VerdictChooser = Callable[[FrontierEvaluationCell], str]


def _declaration() -> SystemFeasibilityFrontierDeclaration:
    return load_system_feasibility_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _row(density: float, verdict: str) -> dict[str, object]:
    candidate_passes = verdict == "feasible"
    return {
        "density_vehicles_per_lane_km": density,
        "mean_pair_local_candidate_conditional_miss_risk": (
            5e-5 if candidate_passes else 2e-4
        ),
        "pair_local_candidate_mean_meets_budget": candidate_passes,
        "certified_lower_bound_exceeds_budget": False,
        "all_frames_optimality_proven": verdict == "infeasible",
    }


def _windows(
    declaration: SystemFeasibilityFrontierDeclaration,
) -> tuple[EvaluationWindow, ...]:
    return tuple(
        EvaluationWindow(
            trace_id=f"synthetic-d{density:g}-validation-{replicate:03d}",
            density=density,
            start_frame_index=replicate,
            frames=declaration.frames_per_window,
        )
        for density in declaration.densities
        for replicate in range(declaration.validation_windows_per_density)
    )


def _result(chooser: VerdictChooser) -> SystemFeasibilityFrontierResult:
    declaration = _declaration()
    cells: list[FrontierCellResult] = []
    for cell in declaration.evaluation_cells:
        selected = chooser(cell)
        assert selected in {"feasible", "infeasible", "inconclusive"}
        rows = tuple(_row(density, selected) for density in declaration.densities)
        verdicts = tuple(
            density_verdict(row, miss_budget=declaration.miss_budget)
            for row in rows
        )
        cells.append(
            FrontierCellResult(
                cell=cell,
                config_hash="a" * 64,
                policy_environment_scope_hash="b" * 64,
                density_rows=rows,
                campaign={},
                density_verdicts=verdicts,
                verdict=cell_verdict(verdicts),
            )
        )
    return SystemFeasibilityFrontierResult(
        declaration=declaration,
        windows=_windows(declaration),
        cells=tuple(cells),
        generated_at_utc=datetime(2026, 9, 27, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("feasible", "feasible"),
        ("infeasible", "infeasible"),
        ("inconclusive", "inconclusive"),
    ],
)
def test_density_verdict_preserves_the_certificate_gap(
    label: str,
    expected: str,
) -> None:
    assert density_verdict(_row(10.0, label), miss_budget=1e-4) == expected


def test_all_feasible_grid_authorizes_only_the_minimal_current_design() -> None:
    report = _result(lambda _cell: "feasible")

    payload = report.as_dict()
    decision = report.decision()
    assert payload["schema"] == SYSTEM_FEASIBILITY_FRONTIER_RESULT_SCHEMA
    assert len(payload["cells"]) == 36
    assert decision["current_system_nominal_verdict"] == "feasible"
    assert decision["current_system_robust_verdict"] == "feasible"
    assert decision["training_authorized"] is True
    assert decision["pareto_minimal_robust_feasible_design_ids"] == [
        "rf-capacity-1x__wide-60deg"
    ]
    assert decision["fallback_diagnostic_authorized_training"] is False


def test_diagnostic_success_never_authorizes_training() -> None:
    report = _result(
        lambda cell: (
            "feasible" if cell.fallback_view.mode == "all_usable" else "inconclusive"
        )
    )

    decision = report.decision()
    assert decision["training_authorized"] is False
    assert decision["robust_feasible_design_ids"] == []
    assert decision["fallback_diagnostic_authorized_training"] is False


def test_current_nominal_success_is_not_robust_sensing_success() -> None:
    def choose(cell: FrontierEvaluationCell) -> str:
        point = cell.physical_point
        if cell.fallback_view.mode == "all_usable":
            return "feasible"
        if (
            point.rf_capacity.headline
            and point.optical_configuration.headline
        ):
            return "feasible" if point.sensing_band.name != "pessimistic" else "inconclusive"
        return "infeasible"

    decision = _result(choose).decision()

    assert decision["current_system_nominal_verdict"] == "feasible"
    assert decision["current_system_robust_verdict"] == "inconclusive"
    assert decision["training_authorized"] is False


def test_partial_grid_cannot_construct_a_final_result() -> None:
    complete = _result(lambda _cell: "feasible")

    with pytest.raises(SystemFeasibilityExecutionError, match="complete frozen grid"):
        SystemFeasibilityFrontierResult(
            declaration=complete.declaration,
            windows=complete.windows,
            cells=complete.cells[:-1],
            generated_at_utc=complete.generated_at_utc,
        )


def test_progress_artifact_round_trips_an_ordered_cell_prefix(
    tmp_path: Path,
) -> None:
    complete = _result(lambda _cell: "feasible")
    progress_path = tmp_path / "frontier.progress.json"
    prefix = complete.cells[:3]

    written = write_frontier_progress(
        progress_path,
        declaration=complete.declaration,
        cells=prefix,
    )
    restored = load_frontier_progress(
        written,
        declaration=complete.declaration,
    )

    assert restored == prefix
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["test_split_opened"] is False
    assert len(payload["completed_cells"]) == 3


def test_progress_artifact_rejects_provenance_or_physics_drift(
    tmp_path: Path,
) -> None:
    complete = _result(lambda _cell: "feasible")
    progress_path = write_frontier_progress(
        tmp_path / "frontier.progress.json",
        declaration=complete.declaration,
        cells=complete.cells[:1],
    )
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["completed_cells"][0]["rf_capacity"]["subchannels"] = 99
    progress_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        SystemFeasibilityExecutionError,
        match="physical parameters",
    ):
        load_frontier_progress(
            progress_path,
            declaration=complete.declaration,
        )


def test_structural_dry_run_instantiates_every_declared_cell() -> None:
    declaration = _declaration()
    if not declaration.window_source.is_file():
        pytest.skip("frozen evaluation artifact is not present in this checkout")

    report = structural_dry_run(declaration, project_root=PROJECT_ROOT)
    payload = report.as_dict()

    assert payload["schema"] == SYSTEM_FEASIBILITY_FRONTIER_DRY_RUN_SCHEMA
    assert payload["frontier_executed"] is False
    assert payload["test_split_opened"] is False
    assert payload["physical_points"] == 18
    assert payload["evaluation_cells"] == 36
    assert len(payload["cells"]) == 36
    assert {row["rf_subchannels"] for row in report.cell_plans} == {1, 2, 4}
    assert {row["sensing_reliability"] for row in report.cell_plans} == {
        0.70,
        0.85,
        0.95,
    }
