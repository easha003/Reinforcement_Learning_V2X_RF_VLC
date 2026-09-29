"""Declaration, selection, and resume tests for the combined frontier."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from hybrid_v2x_rl.agents.combined_receiver_block_frontier import (
    COMBINED_FRONTIER_PROGRESS_SCHEMA,
    COMBINED_FRONTIER_RESULT_SCHEMA,
    CombinedProfileResult,
    CombinedReceiverBlockDeclaration,
    CombinedReceiverBlockError,
    CombinedReceiverBlockResult,
    load_combined_receiver_block_declaration,
    load_combined_receiver_block_progress,
    structural_combined_receiver_block_dry_run,
    write_combined_receiver_block_progress,
)
from hybrid_v2x_rl.config.loader import load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path("configs/evaluation/combined_receiver_block_frontier.yaml")


def _declaration() -> CombinedReceiverBlockDeclaration:
    return load_combined_receiver_block_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "combined.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _profile_result(
    declaration: CombinedReceiverBlockDeclaration,
    index: int,
    *,
    wide_mean: float,
    concentrated_mean: float,
) -> CombinedProfileResult:
    rows: list[dict[str, object]] = []
    means = {
        declaration.optical_configuration_names[0]: wide_mean,
        declaration.optical_configuration_names[1]: concentrated_mean,
    }
    for optical in declaration.optical_configuration_names:
        for density in declaration.densities:
            mean = means[optical]
            rows.append(
                {
                    "optical_configuration_name": optical,
                    "receiver_fov_deg": 60.0 if optical == "wide-60deg" else 30.0,
                    "density_vehicles_per_lane_km": density,
                    "frames": 48,
                    "transitions": 23_204,
                    "mean_optimistic_propagation_only_conditional_miss_lower_bound": mean,
                    "exact_budget_multiple": mean / declaration.miss_budget,
                    "near_budget_multiple": mean / declaration.near_budget,
                    "meets_exact_budget": mean <= declaration.miss_budget,
                    "meets_exploratory_near_budget": mean <= declaration.near_budget,
                    "lower_bound_action_counts": {},
                }
            )
    exact_names = tuple(
        optical for optical in declaration.optical_configuration_names if means[optical] <= 1e-4
    )
    near_names = tuple(
        optical for optical in declaration.optical_configuration_names if means[optical] <= 1.1e-4
    )
    return CombinedProfileResult(
        combined_profile=declaration.receive_profiles[index],
        rows=tuple(rows),
        exact_optical_configuration_names=exact_names,
        near_optical_configuration_names=near_names,
    )


def _complete_results(
    declaration: CombinedReceiverBlockDeclaration,
    means: tuple[tuple[float, float], ...],
) -> tuple[CombinedProfileResult, ...]:
    return tuple(
        _profile_result(
            declaration,
            index,
            wide_mean=profile_means[0],
            concentrated_mean=profile_means[1],
        )
        for index, profile_means in enumerate(means)
    )


def test_canonical_declaration_freezes_grid_and_near_margin() -> None:
    declaration = _declaration()

    assert declaration.payload_bytes == 300
    assert declaration.deadline_s == pytest.approx(0.010)
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.near_budget == pytest.approx(1.1e-4)
    assert declaration.maximum_relative_excess == pytest.approx(0.10)
    assert declaration.candidate.name == "qpsk-2p0ms-sensitivity"
    assert declaration.densities == (10.0, 20.0, 30.0)
    assert [profile.profile.implementation_loss_db for profile in declaration.receive_profiles] == [
        0.0,
        0.0,
        0.0,
    ]
    assert [profile.role for profile in declaration.receive_profiles] == [
        "optimistic-sensitivity",
        "hardware-primary",
        "correlated-sensitivity",
    ]


def test_near_margin_cannot_change_after_freeze(tmp_path: Path) -> None:
    payload = _payload()
    objective = payload["objective"]
    assert isinstance(objective, dict)
    objective["exploratory_near_budget"] = 0.00012

    with pytest.raises(CombinedReceiverBlockError, match="boundary has drifted"):
        load_combined_receiver_block_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_exact_candidate_is_selected_before_near_candidates() -> None:
    declaration = _declaration()
    result = CombinedReceiverBlockResult(
        declaration=declaration,
        profile_results=_complete_results(
            declaration,
            (
                (1.05e-4, 1.06e-4),
                (9.8e-5, 9.9e-5),
                (9.5e-5, 9.6e-5),
            ),
        ),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert result.as_dict()["schema"] == COMBINED_FRONTIER_RESULT_SCHEMA
    assert decision["exact_propagation_target_met"] is True
    assert decision["selection_basis"] == "exact-feasible"
    assert decision["selected_receive_profile_role"] == "correlated-sensitivity"
    assert decision["joint_contention_frontier_authorized"] is True
    assert decision["training_authorized"] is False


def test_near_candidate_authorizes_joint_characterization_not_training() -> None:
    declaration = _declaration()
    result = CombinedReceiverBlockResult(
        declaration=declaration,
        profile_results=_complete_results(
            declaration,
            (
                (1.08e-4, 1.07e-4),
                (1.05e-4, 1.06e-4),
                (1.20e-4, 1.21e-4),
            ),
        ),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert decision["exact_propagation_target_met"] is False
    assert decision["exploratory_near_candidate_exists"] is True
    assert decision["selection_basis"] == "exploratory-near-feasible"
    assert decision["selected_receive_profile_role"] == "hardware-primary"
    assert decision["selected_worst_density_mean"] == pytest.approx(1.05e-4)
    assert decision["joint_contention_frontier_authorized"] is True
    assert decision["training_authorized"] is False
    assert decision["test_split_opened"] is False


def test_above_near_margin_stops_before_joint_frontier() -> None:
    declaration = _declaration()
    result = CombinedReceiverBlockResult(
        declaration=declaration,
        profile_results=_complete_results(
            declaration,
            (
                (1.11e-4, 1.12e-4),
                (1.13e-4, 1.14e-4),
                (1.15e-4, 1.16e-4),
            ),
        ),
        generated_at_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )

    decision = result.decision()
    assert decision["selection_basis"] is None
    assert decision["best_observed_receive_profile_role"] == "optimistic-sensitivity"
    assert decision["best_observed_worst_density_mean"] == pytest.approx(1.11e-4)
    assert decision["joint_contention_frontier_authorized"] is False
    assert decision["training_authorized"] is False


def test_progress_round_trips_an_ordered_profile_prefix(tmp_path: Path) -> None:
    declaration = _declaration()
    prefix = _complete_results(
        declaration,
        ((1.05e-4, 1.06e-4), (1.07e-4, 1.08e-4), (1.09e-4, 1.10e-4)),
    )[:1]
    path = write_combined_receiver_block_progress(
        tmp_path / "combined.progress.json",
        declaration=declaration,
        results=prefix,
    )

    assert load_combined_receiver_block_progress(path, declaration=declaration) == prefix
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == COMBINED_FRONTIER_PROGRESS_SCHEMA
    assert payload["training_run_performed"] is False
    assert payload["test_split_opened"] is False


def test_progress_rejects_declaration_drift(tmp_path: Path) -> None:
    declaration = _declaration()
    prefix = _complete_results(
        declaration,
        ((1.05e-4, 1.06e-4), (1.07e-4, 1.08e-4), (1.09e-4, 1.10e-4)),
    )[:1]
    path = write_combined_receiver_block_progress(
        tmp_path / "combined.progress.json",
        declaration=declaration,
        results=prefix,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["declaration_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CombinedReceiverBlockError, match="provenance has drifted"):
        load_combined_receiver_block_progress(path, declaration=declaration)


def test_structural_dry_run_opens_no_frames_or_test_data() -> None:
    declaration = _declaration()
    if not declaration.deadline_result_artifact.path.is_file():
        pytest.skip("frozen evidence and validation artifacts are not present in this checkout")
    report = structural_combined_receiver_block_dry_run(
        declaration,
        project_root=PROJECT_ROOT,
    )

    assert report["receive_profiles"] == 3
    assert report["optical_configurations"] == 2
    assert report["densities"] == 3
    assert report["validation_windows"] == 9
    assert report["physical_profile_instances"] == 6
    assert report["evaluation_rows"] == 18
    assert report["channel_frames_evaluated"] == 0
    assert report["training_authorized"] is False
    assert report["test_split_opened"] is False
