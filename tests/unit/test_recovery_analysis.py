"""Predeclared bounded constraint-recovery comparison gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.recovery_analysis import (
    RECOVERY_ARMS,
    ConstraintRecoveryAnalysisError,
    build_constraint_recovery_analysis,
)
from hybrid_v2x_rl.agents.regime_evaluation import (
    POLICY_INDUCED_LOAD,
    PPO_REGIME_EVALUATION_SCHEMA,
)
from hybrid_v2x_rl.mean_field.state_regime_audit import REGIME_NAMES


def _evaluation(
    path: Path,
    *,
    feasible: float,
    risk: float,
    overrides: dict[str, float] | None = None,
) -> Path:
    regimes = []
    for index, regime in enumerate(REGIME_NAMES, start=1):
        regime_feasible = (overrides or {}).get(regime, feasible)
        regimes.append(
            {
                "regime": regime,
                "rows": index * 10,
                "counterfactuals": {
                    POLICY_INDUCED_LOAD: {
                        "mean_feasible_action_probability_mass": regime_feasible,
                        "mean_policy_expected_conditional_miss_risk": risk,
                    }
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": PPO_REGIME_EVALUATION_SCHEMA,
                "config_hash": path.stem.ljust(64, "0")[:64],
                "test_split_opened": False,
                "campaign_regimes": regimes,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_analysis_selects_the_largest_passing_feasible_mass_gain(tmp_path: Path) -> None:
    paths = {
        "control": _evaluation(tmp_path / "control.json", feasible=0.20, risk=0.10),
        "dual_lr_5": _evaluation(tmp_path / "lr.json", feasible=0.35, risk=0.09),
        "dual_init_10": _evaluation(tmp_path / "init.json", feasible=0.31, risk=0.10),
        "entropy_005": _evaluation(tmp_path / "entropy.json", feasible=0.25, risk=0.09),
    }

    report = build_constraint_recovery_analysis(paths)

    assert report.selected_arm == "dual_lr_5"
    assert report.comparisons["dual_lr_5"]["passes_primary_gate"] is True
    assert report.comparisons["dual_init_10"]["passes_primary_gate"] is True
    assert report.comparisons["entropy_005"]["passes_primary_gate"] is False
    assert report.as_dict()["full_seed_1001_authorized"] is True


def test_analysis_rejects_a_material_single_regime_loss(tmp_path: Path) -> None:
    paths = {
        "control": _evaluation(tmp_path / "control.json", feasible=0.20, risk=0.10),
        "dual_lr_5": _evaluation(
            tmp_path / "lr.json",
            feasible=0.35,
            risk=0.09,
            overrides={REGIME_NAMES[0]: 0.17},
        ),
        "dual_init_10": _evaluation(tmp_path / "init.json", feasible=0.20, risk=0.10),
        "entropy_005": _evaluation(tmp_path / "entropy.json", feasible=0.20, risk=0.10),
    }

    report = build_constraint_recovery_analysis(paths)

    assert report.selected_arm is None
    assert report.comparisons["dual_lr_5"]["passes_primary_gate"] is False


def test_analysis_requires_exactly_the_predeclared_arms(tmp_path: Path) -> None:
    path = _evaluation(tmp_path / "control.json", feasible=0.2, risk=0.1)

    with pytest.raises(ConstraintRecoveryAnalysisError, match="exactly the four"):
        build_constraint_recovery_analysis({RECOVERY_ARMS[0]: path})
