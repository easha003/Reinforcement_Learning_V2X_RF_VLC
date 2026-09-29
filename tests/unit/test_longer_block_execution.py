"""Result, selection, and resume contracts for the longer-block screen."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.longer_block_execution import (
    LONGER_BLOCK_FRONTIER_PROGRESS_SCHEMA,
    LONGER_BLOCK_FRONTIER_RESULT_SCHEMA,
    LongerBlockCandidateResult,
    LongerBlockExecutionError,
    LongerBlockFrontierResult,
    load_longer_block_progress,
    write_longer_block_progress,
)
from hybrid_v2x_rl.agents.longer_block_frontier import (
    LongerBlockFrontierDeclaration,
    candidate_grid_row,
    load_longer_block_frontier_declaration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path("configs/evaluation/longer_block_rf_frontier.yaml")


def _declaration() -> LongerBlockFrontierDeclaration:
    return load_longer_block_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _candidate_result(
    declaration: LongerBlockFrontierDeclaration,
    index: int,
    *,
    passing: bool,
) -> LongerBlockCandidateResult:
    candidate = declaration.candidates[index]
    rows: list[dict[str, object]] = []
    for optical in declaration.optical_configurations:
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
    return LongerBlockCandidateResult(
        candidate=candidate,
        execution_mode=(
            "reused-hash-verified-receive-diversity-control" if index == 0 else "validation-replay"
        ),
        rows=tuple(rows),
        passing_optical_configuration_names=(
            declaration.optical_configuration_names if passing else ()
        ),
    )


def _complete_results(
    declaration: LongerBlockFrontierDeclaration,
    *,
    passing_indices: set[int],
) -> tuple[LongerBlockCandidateResult, ...]:
    return tuple(
        _candidate_result(declaration, index, passing=index in passing_indices)
        for index in range(len(declaration.candidates))
    )


def test_result_selects_the_shortest_passing_candidate() -> None:
    declaration = _declaration()
    result = LongerBlockFrontierResult(
        declaration=declaration,
        candidate_results=_complete_results(
            declaration,
            passing_indices={2, 3},
        ),
        resource_grids=tuple(
            candidate_grid_row(declaration, candidate, project_root=PROJECT_ROOT)
            for candidate in declaration.candidates
        ),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert result.as_dict()["schema"] == LONGER_BLOCK_FRONTIER_RESULT_SCHEMA
    assert decision["shortest_passing_candidate_name"] == ("qpsk-1p5ms-sensitivity")
    assert decision["shortest_passing_airtime_s"] == pytest.approx(0.0015)
    assert decision["joint_contention_frontier_authorized"] is True
    assert decision["training_authorized"] is False


def test_no_propagation_survivor_blocks_the_joint_frontier() -> None:
    declaration = _declaration()
    result = LongerBlockFrontierResult(
        declaration=declaration,
        candidate_results=_complete_results(declaration, passing_indices=set()),
        resource_grids=tuple({"candidate": candidate.name} for candidate in declaration.candidates),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert decision["propagation_necessary_condition_met"] is False
    assert decision["joint_contention_frontier_authorized"] is False
    assert decision["shortest_passing_candidate_name"] is None


def test_progress_round_trips_an_ordered_prefix(tmp_path: Path) -> None:
    declaration = _declaration()
    prefix = _complete_results(declaration, passing_indices={1})[:2]
    path = write_longer_block_progress(
        tmp_path / "frontier.progress.json",
        declaration=declaration,
        results=prefix,
    )

    assert load_longer_block_progress(path, declaration=declaration) == prefix
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == LONGER_BLOCK_FRONTIER_PROGRESS_SCHEMA
    assert payload["training_run_performed"] is False
    assert payload["test_split_opened"] is False


def test_progress_rejects_declaration_drift(tmp_path: Path) -> None:
    declaration = _declaration()
    path = write_longer_block_progress(
        tmp_path / "frontier.progress.json",
        declaration=declaration,
        results=_complete_results(declaration, passing_indices=set())[:1],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["declaration_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LongerBlockExecutionError, match="provenance has drifted"):
        load_longer_block_progress(path, declaration=declaration)
