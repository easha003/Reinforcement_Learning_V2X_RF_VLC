"""Residual oracle-floor and policy-regret diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.regime_evaluation import (
    ALL_USABLE_ROWS,
    POLICY_INDUCED_LOAD,
    PPO_REGIME_EVALUATION_SCHEMA,
)
from hybrid_v2x_rl.agents.residual_feasibility import (
    ResidualFeasibilityError,
    build_residual_feasibility_report,
)
from hybrid_v2x_rl.mean_field.state_regime_audit import LOAD_PROFILES


def _profile(floor: float, *, expected_regret: float = 0.001) -> dict[str, object]:
    selected_regret = 0.002
    return {
        "rows": 20,
        "any_feasible_fraction": 0.5,
        "mean_minimum_action_conditional_miss_risk": floor,
        "mean_policy_expected_conditional_miss_risk": floor + expected_regret,
        "mean_deterministic_selected_conditional_miss_risk": floor + selected_regret,
        "mean_policy_expected_action_risk_regret": expected_regret,
        "mean_deterministic_selected_action_risk_regret": selected_regret,
        "minimum_risk_action_counts": {"VLC": 10, "DUP-4": 10},
    }


def _scope(
    density: float | None,
    *,
    induced_floor: float,
    offload_floor: float,
) -> dict[str, object]:
    return {
        "regime": ALL_USABLE_ROWS,
        "density_vehicles_per_lane_km": density,
        "rows": 20,
        "counterfactuals": {
            POLICY_INDUCED_LOAD: _profile(induced_floor),
            "vlc_offload": _profile(offload_floor),
            "rf1_pressure": _profile(max(induced_floor, offload_floor)),
            "rf4_pressure": _profile(max(induced_floor, offload_floor) + 0.001),
        },
    }


def _evaluation(path: Path) -> Path:
    payload = {
        "schema": PPO_REGIME_EVALUATION_SCHEMA,
        "test_split_opened": False,
        "config_hash": "a" * 64,
        "checkpoint": {"sha256": "b" * 64, "policy_seed": 1001},
        "reliability_miss_budget": 0.0001,
        "all_usable_campaign": _scope(
            None,
            induced_floor=0.00015,
            offload_floor=0.00005,
        ),
        "all_usable_by_density": [
            _scope(10.0, induced_floor=0.0002, offload_floor=0.00005),
            _scope(20.0, induced_floor=0.00005, offload_floor=0.00004),
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_diagnosis_separates_load_floor_from_policy_action_regret(tmp_path: Path) -> None:
    report = build_residual_feasibility_report(_evaluation(tmp_path / "evaluation.json"))

    assert report.standard_ppo_recovery_arm_authorized is False
    assert "population-coupled" in report.next_action
    assert report.densities[0]["diagnosis"] == (
        "policy_load_or_population_coordination_floor_exceeds_target"
    )
    assert report.densities[1]["diagnosis"] == (
        "fixed_policy_load_floor_meets_target_action_selection_regret_remains"
    )
    induced = report.densities[0]["counterfactuals"][POLICY_INDUCED_LOAD]
    assert induced["mean_deterministic_selected_action_risk_regret"] == pytest.approx(
        0.002
    )
    assert induced["deterministic_selected_decomposition_residual"] == pytest.approx(0.0)


def test_diagnosis_identifies_action_or_physical_floor(tmp_path: Path) -> None:
    source = _evaluation(tmp_path / "evaluation.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["all_usable_by_density"][0] = _scope(
        10.0,
        induced_floor=0.0003,
        offload_floor=0.0002,
    )
    source.write_text(json.dumps(payload), encoding="utf-8")

    report = build_residual_feasibility_report(source)

    assert report.densities[0]["diagnosis"] == (
        "physical_or_action_floor_exceeds_target_even_with_vlc_offload"
    )
    assert "physical/action model" in report.next_action


def test_diagnosis_rejects_a_nonclosing_decomposition(tmp_path: Path) -> None:
    source = _evaluation(tmp_path / "evaluation.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    profiles = payload["all_usable_campaign"]["counterfactuals"]
    profiles[POLICY_INDUCED_LOAD]["mean_policy_expected_action_risk_regret"] = 0.0
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResidualFeasibilityError, match="does not close"):
        build_residual_feasibility_report(source)


def test_fixture_tracks_all_declared_profiles() -> None:
    assert set(LOAD_PROFILES) == {"vlc_offload", "rf1_pressure", "rf4_pressure"}
