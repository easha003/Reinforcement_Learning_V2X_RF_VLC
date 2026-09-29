"""Frozen declaration, decision, and resume tests for the deadline-edge screen."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from hybrid_v2x_rl.agents.deadline_edge_threshold import (
    DEADLINE_EDGE_PROGRESS_SCHEMA,
    DEADLINE_EDGE_RESULT_SCHEMA,
    DeadlineEdgeThresholdDeclaration,
    DeadlineEdgeThresholdError,
    DeadlineEdgeThresholdResult,
    OpticalThresholdResult,
    load_deadline_edge_progress,
    load_deadline_edge_threshold_declaration,
    structural_deadline_edge_dry_run,
    write_deadline_edge_progress,
)
from hybrid_v2x_rl.config.loader import load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path("configs/evaluation/deadline_edge_blocklength_threshold.yaml")


def _declaration() -> DeadlineEdgeThresholdDeclaration:
    return load_deadline_edge_threshold_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "deadline-edge.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _optical_result(
    declaration: DeadlineEdgeThresholdDeclaration,
    index: int,
    *,
    theoretical_pass: bool,
) -> OpticalThresholdResult:
    optical = declaration.longer_declaration.optical_configurations[index]
    source_mean = 1.723841e-4
    anchors: tuple[dict[str, object], ...] = tuple(
        {
            "airtime_s": airtime,
            "channel_uses": channel_uses,
            "mean_selected_risk": source_mean,
        }
        for airtime, channel_uses in zip(
            declaration.anchor_airtimes_s,
            declaration.anchor_channel_uses,
            strict=True,
        )
    )
    threshold: dict[str, object] = {
        "upper_meets_budget": theoretical_pass,
        "minimum_passing_channel_uses": 10_500 if theoretical_pass else None,
        "minimum_passing_airtime_s": (
            10_500 / declaration.data_channel_uses_per_second if theoretical_pass else None
        ),
        "rounded_full_slot_candidate_fits_deadline": False,
    }
    return OpticalThresholdResult(
        optical_configuration_name=optical.name,
        receiver_fov_deg=optical.receiver_fov_deg,
        frames=192,
        transitions=23_204,
        source_mean_selected_risk=source_mean,
        control_decomposition={"mean_selected_risk": source_mean},
        anchors=anchors,
        threshold=threshold,
    )


def test_canonical_declaration_freezes_deadline_edge_and_grid_boundaries() -> None:
    declaration = _declaration()

    assert declaration.payload_bytes == 300
    assert declaration.deadline_s == pytest.approx(0.010)
    assert declaration.predecision_lead_s == pytest.approx(0.0001)
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.density == 20.0
    assert declaration.source_candidate_name == "qpsk-2p0ms-sensitivity"
    assert declaration.anchor_channel_uses == (9676, 10281, 10886, 11491, 11975)
    assert declaration.upper_airtime_s == pytest.approx(0.002475)
    assert declaration.next_full_slot_airtime_s == pytest.approx(0.0025)
    assert declaration.next_full_slot_rf4_airtime_s > declaration.maximum_rf4_airtime_s


def test_threshold_or_grid_cannot_expand_after_freeze(tmp_path: Path) -> None:
    payload = _payload()
    screen = payload["threshold_screen"]
    assert isinstance(screen, dict)
    screen["upper_airtime_s"] = 0.0025

    with pytest.raises(DeadlineEdgeThresholdError, match="interval or anchors"):
        load_deadline_edge_threshold_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_theoretical_pass_does_not_authorize_current_full_slot_grid() -> None:
    declaration = _declaration()
    result = DeadlineEdgeThresholdResult(
        declaration=declaration,
        optical_results=(
            _optical_result(declaration, 0, theoretical_pass=True),
            _optical_result(declaration, 1, theoretical_pass=True),
        ),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert result.as_dict()["schema"] == DEADLINE_EDGE_RESULT_SCHEMA
    assert decision["theoretical_deadline_boundary_passes"] is True
    assert decision["current_full_slot_grid_has_passing_candidate"] is False
    assert decision["joint_contention_frontier_authorized"] is False
    assert decision["training_authorized"] is False
    assert decision["test_split_opened"] is False


def test_progress_round_trips_an_ordered_optical_prefix(tmp_path: Path) -> None:
    declaration = _declaration()
    prefix = (_optical_result(declaration, 0, theoretical_pass=True),)
    path = write_deadline_edge_progress(
        tmp_path / "deadline-edge.progress.json",
        declaration=declaration,
        results=prefix,
    )

    assert load_deadline_edge_progress(path, declaration=declaration) == prefix
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == DEADLINE_EDGE_PROGRESS_SCHEMA
    assert payload["training_run_performed"] is False
    assert payload["test_split_opened"] is False


def test_progress_rejects_declaration_drift(tmp_path: Path) -> None:
    declaration = _declaration()
    path = write_deadline_edge_progress(
        tmp_path / "deadline-edge.progress.json",
        declaration=declaration,
        results=(_optical_result(declaration, 0, theoretical_pass=False),),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["declaration_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DeadlineEdgeThresholdError, match="provenance has drifted"):
        load_deadline_edge_progress(path, declaration=declaration)


def test_structural_dry_run_opens_no_frames_or_test_data() -> None:
    declaration = _declaration()
    if not declaration.longer_result_path.is_file():
        pytest.skip("frozen result and validation artifacts are not present in this checkout")
    report = structural_deadline_edge_dry_run(
        declaration,
        project_root=PROJECT_ROOT,
    )

    assert report["validation_windows"] == 3
    assert report["physical_profile_instances"] == 2
    assert report["next_full_slot_fits"] is False
    assert report["channel_frames_evaluated"] == 0
    assert report["joint_frontier_authorized"] is False
    assert report["training_authorized"] is False
    assert report["test_split_opened"] is False
