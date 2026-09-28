"""Receive-diversity screening, progress, and authorization contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.receive_diversity_execution import (
    RECEIVE_DIVERSITY_FRONTIER_PROGRESS_SCHEMA,
    RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA,
    PropagationScreenProfileResult,
    ReceiveDiversityCellResult,
    ReceiveDiversityExecutionError,
    ReceiveDiversityFrontierResult,
    ReceiveProfileFrontierResult,
    load_receive_diversity_progress,
    propagation_only_action_risk,
    write_receive_diversity_progress,
)
from hybrid_v2x_rl.agents.receive_diversity_frontier import (
    ReceiveDiversityFrontierDeclaration,
    load_receive_diversity_frontier_declaration,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.agents.system_feasibility_execution import (
    FrontierCellResult,
    SystemFeasibilityFrontierResult,
    cell_verdict,
    density_verdict,
)
from hybrid_v2x_rl.core.policy_actions import PolicyAction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path(
    "configs/evaluation/receive_diversity_system_feasibility_frontier.yaml"
)


def _declaration() -> ReceiveDiversityFrontierDeclaration:
    return load_receive_diversity_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _screen(
    declaration: ReceiveDiversityFrontierDeclaration,
    *,
    survivors: set[str],
) -> tuple[PropagationScreenProfileResult, ...]:
    results: list[PropagationScreenProfileResult] = []
    for profile in declaration.receive_profiles:
        rows: list[dict[str, object]] = []
        passing = profile.name in survivors
        for optical in declaration.source_frontier.optical_configurations:
            for density in declaration.densities:
                mean = 5e-5 if passing else 2e-4
                rows.append(
                    {
                        "optical_configuration_name": optical.name,
                        "receiver_fov_deg": optical.receiver_fov_deg,
                        "density_vehicles_per_lane_km": density,
                        "frames": 48,
                        "transitions": 100,
                        "mean_optimistic_propagation_only_conditional_miss_lower_bound": mean,
                        "budget_multiple": mean / declaration.miss_budget,
                        "meets_budget": passing,
                        "lower_bound_action_counts": {},
                    }
                )
        results.append(
            PropagationScreenProfileResult(
                profile=profile,
                rows=tuple(rows),
                passing_optical_configuration_names=(
                    tuple(
                        optical.name
                        for optical in declaration.source_frontier.optical_configurations
                    )
                    if passing
                    else ()
                ),
            )
        )
    return tuple(results)


def _density_row(density: float, *, feasible: bool) -> dict[str, object]:
    return {
        "density_vehicles_per_lane_km": density,
        "mean_pair_local_candidate_conditional_miss_risk": (
            5e-5 if feasible else 2e-4
        ),
        "pair_local_candidate_mean_meets_budget": feasible,
        "certified_lower_bound_exceeds_budget": not feasible,
        "all_frames_optimality_proven": False,
    }


def _windows(
    declaration: ReceiveDiversityFrontierDeclaration,
) -> tuple[EvaluationWindow, ...]:
    source = declaration.source_frontier
    return tuple(
        EvaluationWindow(
            trace_id=f"synthetic-d{density:g}-validation-{replicate:03d}",
            density=density,
            start_frame_index=replicate,
            frames=source.frames_per_window,
        )
        for density in declaration.densities
        for replicate in range(source.validation_windows_per_density)
    )


def _system_result(
    declaration: ReceiveDiversityFrontierDeclaration,
    *,
    feasible: bool,
) -> SystemFeasibilityFrontierResult:
    source = declaration.source_frontier
    cells: list[FrontierCellResult] = []
    for cell in source.evaluation_cells:
        rows = tuple(
            _density_row(density, feasible=feasible)
            for density in declaration.densities
        )
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
        declaration=source,
        windows=_windows(declaration),
        cells=tuple(cells),
        generated_at_utc=datetime(2026, 9, 28, tzinfo=UTC),
    )


def test_propagation_only_risk_matches_action_resource_semantics() -> None:
    rf = 0.1
    vlc = 0.2

    assert propagation_only_action_risk(
        PolicyAction.RF_1,
        rf_decoding_failure_probability=rf,
        vlc_failure_probability=vlc,
    ) == pytest.approx(0.1)
    assert propagation_only_action_risk(
        PolicyAction.RF_4,
        rf_decoding_failure_probability=rf,
        vlc_failure_probability=vlc,
    ) == pytest.approx(1e-4)
    assert propagation_only_action_risk(
        PolicyAction.VLC,
        rf_decoding_failure_probability=rf,
        vlc_failure_probability=vlc,
    ) == pytest.approx(0.2)
    assert propagation_only_action_risk(
        PolicyAction.DUP_4,
        rf_decoding_failure_probability=rf,
        vlc_failure_probability=vlc,
    ) == pytest.approx(2e-5)


def test_only_feasible_headline_profile_authorizes_training() -> None:
    declaration = _declaration()
    headline = declaration.headline_receive_profile
    screens = _screen(declaration, survivors={headline.name})
    result = ReceiveDiversityFrontierResult(
        declaration=declaration,
        screen_results=screens,
        profile_frontiers=(
            ReceiveProfileFrontierResult(
                profile=headline,
                system_result=_system_result(declaration, feasible=True),
            ),
        ),
        generated_at_utc=datetime(2026, 9, 28, tzinfo=UTC),
    )

    payload = result.as_dict()
    assert payload["schema"] == RECEIVE_DIVERSITY_FRONTIER_RESULT_SCHEMA
    assert payload["profiles_screened"] == 10
    assert payload["profiles_surviving"] == 1
    assert payload["evaluation_cells_executed"] == 36
    assert result.decision()["training_authorized"] is True
    assert result.decision()["test_split_opened"] is False


def test_ideal_sensitivity_profile_cannot_authorize_training() -> None:
    declaration = _declaration()
    sensitivity = next(
        profile
        for profile in declaration.receive_profiles
        if not profile.authorizes_training and not profile.physical_profile().is_siso
    )
    screens = _screen(declaration, survivors={sensitivity.name})
    result = ReceiveDiversityFrontierResult(
        declaration=declaration,
        screen_results=screens,
        profile_frontiers=(
            ReceiveProfileFrontierResult(
                profile=sensitivity,
                system_result=_system_result(declaration, feasible=True),
            ),
        ),
        generated_at_utc=datetime(2026, 9, 28, tzinfo=UTC),
    )

    assert result.decision()["training_authorized"] is False
    assert result.decision()["sensitivity_profiles_authorized_training"] is False


def test_progress_round_trips_screen_and_ordered_joint_prefix(
    tmp_path: Path,
) -> None:
    declaration = _declaration()
    headline = declaration.headline_receive_profile
    screens = _screen(declaration, survivors={headline.name})
    source_result = _system_result(declaration, feasible=True)
    cells = tuple(
        ReceiveDiversityCellResult(headline, cell)
        for cell in source_result.cells[:2]
    )
    progress_path = write_receive_diversity_progress(
        tmp_path / "receive.progress.json",
        declaration=declaration,
        screen_results=screens,
        cells=cells,
    )

    restored_screens, restored_cells = load_receive_diversity_progress(
        progress_path,
        declaration=declaration,
    )

    assert restored_screens == screens
    assert restored_cells == cells
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["schema"] == RECEIVE_DIVERSITY_FRONTIER_PROGRESS_SCHEMA
    assert payload["test_split_opened"] is False


def test_progress_rejects_a_mutated_receive_profile(tmp_path: Path) -> None:
    declaration = _declaration()
    headline = declaration.headline_receive_profile
    path = write_receive_diversity_progress(
        tmp_path / "receive.progress.json",
        declaration=declaration,
        screen_results=_screen(declaration, survivors={headline.name}),
        cells=(),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["screen_results"][0]["receive_profile"]["antenna_count"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ReceiveDiversityExecutionError,
        match="profile differs",
    ):
        load_receive_diversity_progress(path, declaration=declaration)
