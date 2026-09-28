"""Pair-local joint-search evaluation artifact and verdict contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from hybrid_v2x_rl.agents.joint_oracle_evaluation import (
    JOINT_ORACLE_EVALUATION_SCHEMA,
    JointOracleEvaluationReport,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow


def _density(
    density: float,
    *,
    candidate_passes: bool,
    lower_bound_fails: bool = False,
    exact: bool = False,
) -> dict[str, object]:
    return {
        "density_vehicles_per_lane_km": density,
        "pair_local_candidate_mean_meets_budget": candidate_passes,
        "certified_lower_bound_exceeds_budget": lower_bound_fails,
        "all_frames_optimality_proven": exact,
    }


def _report(
    *,
    candidate_passes: tuple[bool, ...],
    lower_bound_fails: tuple[bool, ...] = (False, False, False),
    exact: tuple[bool, ...] = (False, False, False),
) -> JointOracleEvaluationReport:
    densities = tuple(
        _density(
            density,
            candidate_passes=passes,
            lower_bound_fails=lower_fails,
            exact=is_exact,
        )
        for density, passes, lower_fails, is_exact in zip(
            (10.0, 20.0, 30.0),
            candidate_passes,
            lower_bound_fails,
            exact,
            strict=True,
        )
    )
    return JointOracleEvaluationReport(
        config_hash="a" * 64,
        policy_environment_scope_hash="b" * 64,
        checkpoint_path=Path("checkpoint.pt"),
        checkpoint_sha256="c" * 64,
        checkpoint_policy_seed=1001,
        checkpoint_completed_iterations=265,
        normalization_training_rows=10_000,
        miss_budget=0.0001,
        audit_path=Path("audit.json"),
        audit_sha256="d" * 64,
        windows=(
            EvaluationWindow(
                trace_id="synthetic-d10-validation-000",
                density=10.0,
                start_frame_index=0,
                frames=16,
            ),
        ),
        densities=densities,
        campaign={"transitions": 100},
        generated_at_utc=datetime(2026, 9, 27, tzinfo=UTC),
    )


def test_realisable_candidate_below_budget_proves_feasibility(
    tmp_path: Path,
) -> None:
    report = _report(candidate_passes=(True, True, True))

    payload = report.as_dict()

    assert payload["schema"] == JOINT_ORACLE_EVALUATION_SCHEMA
    assert payload["test_split_opened"] is False
    assert payload["non_deployable_oracle_truth"] is True
    decision = payload["decision"]
    assert decision["verdict"] == "feasible-candidate-found"
    assert decision["all_densities_have_feasible_realizable_candidate"] is True
    assert decision["coordination_aware_recovery_authorized"] is True
    assert decision["standard_independent_ppo_recovery_authorized"] is False
    output = report.write_json(tmp_path / "joint.json")
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_failed_candidate_without_closed_gap_is_inconclusive() -> None:
    payload = _report(candidate_passes=(True, False, True)).as_dict()

    decision = payload["decision"]
    assert decision["verdict"] == "inconclusive-optimality-gap"
    assert decision["coordination_aware_recovery_authorized"] is False
    assert "certificate" in decision["next_action"]


def test_lower_bound_above_budget_proves_infeasibility() -> None:
    payload = _report(
        candidate_passes=(True, False, True),
        lower_bound_fails=(False, True, False),
    ).as_dict()

    decision = payload["decision"]
    assert decision["verdict"] == "infeasible-by-certified-lower-bound"
    assert decision["any_density_infeasible_by_certified_lower_bound"] is True
    assert decision["coordination_aware_recovery_authorized"] is False


def test_exhaustive_failed_search_proves_infeasibility() -> None:
    payload = _report(
        candidate_passes=(True, False, True),
        exact=(True, True, True),
    ).as_dict()

    decision = payload["decision"]
    assert decision["verdict"] == "infeasible-by-exact-search"
    assert decision["all_frames_optimality_proven"] is True
    assert decision["coordination_aware_recovery_authorized"] is False
