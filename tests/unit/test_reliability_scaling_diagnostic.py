"""RF-decoding reliability-scaling declaration and analytical contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hybrid_v2x_rl.agents.reliability_scaling_diagnostic import (
    ReliabilityScalingError,
    load_reliability_scaling_declaration,
    scaled_action_risk,
    structural_scaling_dry_run,
)
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.core.policy_actions import PolicyAction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path(
    "configs/evaluation/rf_decoding_reliability_scaling.yaml"
)


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    target = tmp_path / "scaling.yaml"
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


def test_canonical_scaling_declaration_is_frozen() -> None:
    declaration = load_reliability_scaling_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )

    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.densities == (10.0, 20.0, 30.0)
    assert declaration.factors == (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
    assert [view.mode for view in declaration.views] == ["contract", "all_usable"]
    assert declaration.anchor_point.point_id == (
        "rf-capacity-4x__concentrated-30deg__nominal"
    )


def test_scaling_action_risk_changes_only_rf_decoding() -> None:
    assert scaled_action_risk(
        PolicyAction.RF_4,
        rf_decoding_failure_probability=0.2,
        vlc_failure_probability=0.5,
        improvement_factor=2.0,
    ) == pytest.approx(0.1**4)
    assert scaled_action_risk(
        PolicyAction.DUP_4,
        rf_decoding_failure_probability=0.2,
        vlc_failure_probability=0.5,
        improvement_factor=2.0,
    ) == pytest.approx(0.5 * 0.1**4)
    assert scaled_action_risk(
        PolicyAction.VLC,
        rf_decoding_failure_probability=0.2,
        vlc_failure_probability=0.5,
        improvement_factor=32.0,
    ) == pytest.approx(0.5)


def test_factor_grid_must_start_at_one_and_increase(tmp_path: Path) -> None:
    payload = _payload()
    intervention = payload["intervention"]
    assert isinstance(intervention, dict)
    intervention["factors"] = [2.0, 1.0]
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["expected_factors"] = 2
    execution["expected_density_rows"] = 12

    with pytest.raises(ReliabilityScalingError, match="start at one"):
        load_reliability_scaling_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_diagnostic_cannot_authorize_training(tmp_path: Path) -> None:
    payload = _payload()
    decision = payload["decision"]
    assert isinstance(decision, dict)
    decision["training_authorization"] = True

    with pytest.raises(ReliabilityScalingError, match="cannot authorize training"):
        load_reliability_scaling_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_canonical_evidence_and_structural_dry_run() -> None:
    result_path = (
        PROJECT_ROOT
        / "artifacts/evaluations/phase8_pair_local_system_feasibility_frontier.json"
    )
    if not result_path.is_file():
        pytest.skip("local frozen frontier result is not present")
    declaration = load_reliability_scaling_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=True,
    )

    report = structural_scaling_dry_run(declaration)

    assert report["validation_windows"] == 9
    assert report["density_rows"] == 36
    assert report["channel_frames_evaluated"] == 0
    assert report["joint_action_search_used"] is False
    assert report["training_authorized"] is False
    assert report["test_split_opened"] is False
