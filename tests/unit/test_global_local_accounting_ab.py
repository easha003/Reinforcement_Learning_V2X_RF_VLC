"""Frozen declaration and pure accounting tests for the global/local A/B."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import yaml

from hybrid_v2x_rl.agents.global_local_accounting_ab import (
    GLOBAL_LOCAL_AB_PROGRESS_SCHEMA,
    GlobalLocalAccountingABDeclaration,
    GlobalLocalAccountingABError,
    load_global_local_accounting_ab_declaration,
    load_global_local_accounting_ab_progress,
    packet_conditional_risk,
    structural_global_local_accounting_ab_dry_run,
    write_global_local_accounting_ab_progress,
)
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER, PolicyAction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path("configs/evaluation/global_local_accounting_ab.yaml")


def _declaration() -> GlobalLocalAccountingABDeclaration:
    return load_global_local_accounting_ab_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "global-local-ab.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_declaration_freezes_two_within_deadline_comparisons() -> None:
    declaration = _declaration()

    assert declaration.payload_bytes == 300
    assert declaration.deadlines_s == pytest.approx((0.003, 0.010))
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.actions == POLICY_ACTION_ORDER
    assert declaration.environment_seed == 20260728
    assert declaration.expected_windows == 9
    assert declaration.expected_profiles == 2
    assert declaration.expected_cells == 18
    assert len(declaration.cell_ids) == 18
    assert declaration.cell_ids[0].endswith("__VLC")
    assert declaration.cell_ids[-1].endswith("__DUP-4")


def test_matched_controls_cannot_be_relaxed(tmp_path: Path) -> None:
    payload = _payload()
    design = cast(dict[str, object], payload["design"])
    design["identical_channel_truth"] = False

    with pytest.raises(GlobalLocalAccountingABError, match="design has drifted"):
        load_global_local_accounting_ab_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_packet_conditional_risk_respects_action_resources() -> None:
    assert packet_conditional_risk(
        PolicyAction.VLC,
        rf_attempt_failure_probability=None,
        vlc_miss_probability=0.2,
    ) == pytest.approx(0.2)
    assert packet_conditional_risk(
        PolicyAction.RF_3,
        rf_attempt_failure_probability=0.1,
        vlc_miss_probability=None,
    ) == pytest.approx(0.001)
    assert packet_conditional_risk(
        PolicyAction.DUP_2,
        rf_attempt_failure_probability=0.1,
        vlc_miss_probability=0.2,
    ) == pytest.approx(0.002)


def test_packet_conditional_risk_rejects_unselected_leg() -> None:
    with pytest.raises(GlobalLocalAccountingABError, match="cannot carry RF"):
        packet_conditional_risk(
            PolicyAction.VLC,
            rf_attempt_failure_probability=0.1,
            vlc_miss_probability=0.2,
        )


def test_progress_round_trips_only_an_ordered_prefix(tmp_path: Path) -> None:
    declaration = _declaration()
    prefix = (
        {"cell_id": declaration.cell_ids[0], "value": 1},
        {"cell_id": declaration.cell_ids[1], "value": 2},
    )
    path = write_global_local_accounting_ab_progress(
        tmp_path / "ab.progress.json",
        declaration=declaration,
        cells=prefix,
    )

    assert load_global_local_accounting_ab_progress(path, declaration=declaration) == prefix
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == GLOBAL_LOCAL_AB_PROGRESS_SCHEMA
    assert payload["training_run_performed"] is False
    assert payload["test_split_opened"] is False

    payload["completed_cells"][0]["cell_id"] = declaration.cell_ids[1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GlobalLocalAccountingABError, match="ordered prefix"):
        load_global_local_accounting_ab_progress(path, declaration=declaration)


def test_structural_dry_run_opens_no_channel_frames_or_test_data() -> None:
    declaration = _declaration()
    required = (
        declaration.three_ms_source.path,
        declaration.ten_ms_source.path,
        declaration.window_source.path,
        PROJECT_ROOT
        / "artifacts"
        / "evaluations"
        / "phase8_combined_receiver_block_frontier.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("frozen evidence is not present in this checkout")
    trace_root = PROJECT_ROOT / "artifacts" / "traces"
    if not trace_root.is_dir():
        pytest.skip("validation traces are not present in this checkout")

    report = structural_global_local_accounting_ab_dry_run(
        declaration,
        project_root=PROJECT_ROOT,
    )

    assert report["validation_windows"] == 9
    assert report["expected_cells"] == 18
    assert report["channel_frames_evaluated"] == 0
    assert report["training_performed"] is False
    assert report["test_split_opened"] is False
