"""Population-joint oracle evaluation artifact contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from hybrid_v2x_rl.agents.joint_oracle_evaluation import (
    JOINT_ORACLE_EVALUATION_SCHEMA,
    JointOracleEvaluationReport,
)
from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow


def _density(density: float, *, passes: bool) -> dict[str, object]:
    return {
        "density_vehicles_per_lane_km": density,
        "joint_oracle_mean_meets_budget": passes,
    }


def _report(*, passes: tuple[bool, ...]) -> JointOracleEvaluationReport:
    densities = tuple(
        _density(density, passes=value)
        for density, value in zip((10.0, 20.0, 30.0), passes, strict=True)
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


def test_report_authorizes_only_coordination_aware_recovery_when_all_pass(
    tmp_path: Path,
) -> None:
    report = _report(passes=(True, True, True))

    payload = report.as_dict()

    assert payload["schema"] == JOINT_ORACLE_EVALUATION_SCHEMA
    assert payload["test_split_opened"] is False
    assert payload["non_deployable_oracle_truth"] is True
    decision = payload["decision"]
    assert decision["all_densities_meet_joint_oracle_floor"] is True
    assert decision["coordination_aware_recovery_authorized"] is True
    assert decision["standard_independent_ppo_recovery_authorized"] is False
    output = report.write_json(tmp_path / "joint.json")
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_one_failing_density_blocks_recovery() -> None:
    payload = _report(passes=(True, False, True)).as_dict()

    decision = payload["decision"]
    assert decision["all_densities_meet_joint_oracle_floor"] is False
    assert decision["coordination_aware_recovery_authorized"] is False
    assert "revise" in decision["next_action"]
