"""Frozen declaration and resource-grid tests for the longer-block frontier."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import yaml

from hybrid_v2x_rl.agents.longer_block_frontier import (
    LONGER_BLOCK_FRONTIER_DECLARATION_SCHEMA,
    LongerBlockFrontierDeclaration,
    LongerBlockFrontierError,
    candidate_grid_row,
    config_for_candidate,
    load_longer_block_frontier_declaration,
    structural_longer_block_dry_run,
)
from hybrid_v2x_rl.config.loader import load_yaml_file
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = Path("configs/evaluation/longer_block_rf_frontier.yaml")


def _declaration() -> LongerBlockFrontierDeclaration:
    return load_longer_block_frontier_declaration(
        DECLARATION_PATH,
        project_root=PROJECT_ROOT,
        verify_evidence=False,
    )


def _payload() -> dict[str, object]:
    loaded = load_yaml_file(PROJECT_ROOT / DECLARATION_PATH)
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "longer-block.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_canonical_declaration_freezes_the_minimal_frontier() -> None:
    declaration = _declaration()

    assert declaration.payload_bytes == 300
    assert declaration.deadline_s == pytest.approx(0.010)
    assert declaration.miss_budget == pytest.approx(1e-4)
    assert declaration.densities == (10.0, 20.0, 30.0)
    assert declaration.receive_profile_name == (
        "rx2-mrc__low-correlation-hardware-bound__short-cable-loss"
    )
    assert [candidate.name for candidate in declaration.candidates] == [
        "16qam-0p5ms-control",
        "qpsk-1p0ms-primary",
        "qpsk-1p5ms-sensitivity",
        "qpsk-2p0ms-sensitivity",
    ]
    assert [candidate.expected_slots for candidate in declaration.candidates] == [
        1,
        2,
        3,
        4,
    ]


def test_every_resource_grid_and_deadline_reconciles() -> None:
    declaration = _declaration()
    rows = [
        candidate_grid_row(declaration, candidate, project_root=PROJECT_ROOT)
        for candidate in declaration.candidates
    ]

    assert [row["finite_blocklength_channel_uses"] for row in rows] == [
        2419,
        4838,
        7257,
        9676,
    ]
    assert [row["rf4_total_airtime_s"] for row in rows] == pytest.approx(
        [0.002, 0.004, 0.006, 0.008]
    )
    assert all(row["rf4_fits_deadline"] is True for row in rows)
    rates = [cast(float, row["information_rate_bits_per_channel_use"]) for row in rows]
    assert rates == sorted(rates, reverse=True)


def test_candidate_configs_preserve_action_and_receiver_contracts() -> None:
    declaration = _declaration()
    optical = declaration.optical_configurations[0]
    for candidate in declaration.candidates:
        config = config_for_candidate(
            declaration,
            candidate,
            optical,
            project_root=PROJECT_ROOT,
        )
        assert config.service.payload_bytes == 300
        assert config.service.deadline_s == pytest.approx(0.010)
        assert config.environment.actions == POLICY_ACTION_ORDER
        assert config.environment.max_rf_attempts == 4
        assert config.service.rf_attempts_per_packet == 3


def test_candidate_order_cannot_be_changed_after_freeze(tmp_path: Path) -> None:
    payload = _payload()
    selection = payload["selection"]
    assert isinstance(selection, dict)
    candidates = selection["candidates"]
    assert isinstance(candidates, list)
    candidates[1], candidates[2] = candidates[2], candidates[1]

    with pytest.raises(LongerBlockFrontierError, match="identity or order"):
        load_longer_block_frontier_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=False,
        )


def test_source_digest_drift_fails_closed(tmp_path: Path) -> None:
    payload = _payload()
    evidence = payload["evidence"]
    assert isinstance(evidence, dict)
    source = evidence["receive_diversity_result"]
    assert isinstance(source, dict)
    source["sha256"] = "0" * 64

    with pytest.raises(LongerBlockFrontierError, match="absent or has drifted"):
        load_longer_block_frontier_declaration(
            _write(tmp_path, payload),
            project_root=PROJECT_ROOT,
            verify_evidence=True,
        )


def test_structural_dry_run_opens_no_frames_or_test_data() -> None:
    declaration = _declaration()
    base = config_for_candidate(
        declaration,
        declaration.candidates[0],
        declaration.optical_configurations[0],
        project_root=PROJECT_ROOT,
    )
    first_validation_trace = base.paths.trace_root / base.environment.splits.validation[0]
    if (
        not declaration.receive_declaration.source_frontier.window_source.is_file()
        or not first_validation_trace.is_dir()
    ):
        pytest.skip("frozen validation artifacts are not present in this checkout")
    report = structural_longer_block_dry_run(
        declaration,
        project_root=PROJECT_ROOT,
    )

    assert report["schema"] == LONGER_BLOCK_FRONTIER_DECLARATION_SCHEMA
    assert report["candidates"] == 4
    assert report["optical_configurations"] == 2
    assert report["validation_windows"] == 9
    assert report["physical_profile_instances"] == 8
    assert report["channel_frames_evaluated"] == 0
    assert report["training_authorized"] is False
    assert report["test_split_opened"] is False
