"""Frozen declaration contract for the receive-diversity frontier."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hybrid_v2x_rl.agents.receive_diversity_frontier import (
    RECEIVE_DIVERSITY_FRONTIER_DECLARATION_SCHEMA,
    ReceiveDiversityFrontierError,
    load_receive_diversity_frontier_declaration,
    structural_receive_diversity_dry_run,
)
from hybrid_v2x_rl.config.loader import load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path(
    "configs/evaluation/receive_diversity_system_feasibility_frontier.yaml"
)


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write_declaration(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "receive-diversity.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_declaration_expands_the_frozen_grid() -> None:
    declaration = load_receive_diversity_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
    )

    assert len(declaration.sha256) == 64
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.densities == (10.0, 20.0, 30.0)
    assert declaration.antenna_count == 2
    assert declaration.combining_rule == "maximum-ratio-combining"
    assert declaration.channel_state_information == "perfect-per-attempt"
    assert [level.coefficient for level in declaration.correlation_levels] == [
        0.0,
        pytest.approx(0.03**0.5),
        0.7,
    ]
    assert [level.loss_db for level in declaration.loss_levels] == [0.0, 3.5, 7.0]
    assert len(declaration.receive_profiles) == 10
    assert declaration.evaluation_cells_before_screening == 360
    assert declaration.headline_receive_profile.name == (
        "rx2-mrc__low-correlation-hardware-bound__short-cable-loss"
    )


def test_only_the_literature_bounded_headline_profile_can_authorize() -> None:
    declaration = load_receive_diversity_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )

    authorizing = tuple(
        profile for profile in declaration.receive_profiles if profile.authorizes_training
    )
    assert authorizing == (declaration.headline_receive_profile,)
    assert declaration.control_profile.authorizes_training is False
    independent = tuple(
        profile
        for profile in declaration.receive_profiles
        if profile.branch_correlation == 0.0
    )
    assert independent
    assert all(not profile.authorizes_training for profile in independent)


def test_dry_run_evaluates_no_frames_and_opens_no_test_data() -> None:
    declaration = load_receive_diversity_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )

    report = structural_receive_diversity_dry_run(declaration)

    assert report["schema"] == RECEIVE_DIVERSITY_FRONTIER_DECLARATION_SCHEMA
    assert report["receive_profiles"] == 10
    assert report["evaluation_cells_before_screening"] == 360
    assert report["channel_frames_evaluated"] == 0
    assert report["training_authorized"] is False
    assert report["test_split_opened"] is False


def test_source_frontier_digest_drift_fails_closed(tmp_path: Path) -> None:
    payload = _payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    source = evidence["source_system_frontier"]
    assert isinstance(source, dict)
    source["sha256"] = "0" * 64

    with pytest.raises(
        ReceiveDiversityFrontierError,
        match="source system-frontier declaration digest has drifted",
    ):
        load_receive_diversity_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_headline_correlation_must_preserve_the_ecc_boundary(tmp_path: Path) -> None:
    payload = _payload()
    receive = payload["receive_diversity"]
    assert isinstance(receive, dict)
    correlation = receive["branch_correlation"]
    assert isinstance(correlation, dict)
    levels = correlation["levels"]
    assert isinstance(levels, list)
    headline = levels[1]
    assert isinstance(headline, dict)
    headline["coefficient"] = 0.3

    with pytest.raises(
        ReceiveDiversityFrontierError,
        match="preserve the independent, ECC, and stress boundaries",
    ):
        load_receive_diversity_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_grid_size_must_reconcile_before_results_exist(tmp_path: Path) -> None:
    payload = _payload()
    execution = payload["execution"]
    assert isinstance(execution, dict)
    execution["expected_evaluation_cells_before_screening"] = 359

    with pytest.raises(
        ReceiveDiversityFrontierError,
        match="grid size does not reconcile",
    ):
        load_receive_diversity_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_non_fading_failure_sources_cannot_be_independently_redrawn(
    tmp_path: Path,
) -> None:
    payload = _payload()
    receive = payload["receive_diversity"]
    assert isinstance(receive, dict)
    intervention = receive["intervention"]
    assert isinstance(intervention, dict)
    shared = intervention["shared_mechanisms"]
    assert isinstance(shared, list)
    shared.remove("RF contention")

    with pytest.raises(
        ReceiveDiversityFrontierError,
        match="shared mechanisms must preserve",
    ):
        load_receive_diversity_frontier_declaration(
            _write_declaration(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )
