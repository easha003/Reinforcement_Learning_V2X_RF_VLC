"""Fail-closed audit of the headline RF link and collision-pool semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.profile_audit import (
    RF_PHYSICAL_PROFILE_AUDIT_SCHEMA,
    build_rf_physical_profile_audit,
    write_rf_physical_profile_audit,
)
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def audit() -> dict[str, object]:
    config = load_config(
        headline_config_layers(PROJECT_ROOT), project_root=PROJECT_ROOT
    )
    return build_rf_physical_profile_audit(config)


def test_audit_pins_the_live_finite_blocklength_inputs(
    audit: dict[str, object],
) -> None:
    active = audit["active_profile"]
    assert isinstance(active, dict)
    assert active["active_finite_blocklength_channel_uses"] == 2419
    assert active["information_bits"] == 2784
    assert active["active_information_rate_bits_per_channel_use"] == pytest.approx(
        1.150888797
    )
    required = active["required_snr_db"]
    assert isinstance(required, dict)
    assert required["bler_1e-5"] == pytest.approx(1.457377, abs=1e-6)


def test_audit_detects_the_resource_granularity_mismatch(
    audit: dict[str, object],
) -> None:
    pool = audit["collision_pool_interpretation"]
    checks = audit["checks"]
    decision = audit["decision"]
    assert isinstance(pool, dict)
    assert isinstance(checks, dict)
    assert isinstance(decision, dict)
    assert pool["resource_blocks_per_subchannel"] == 12
    assert pool["active_link_allocation_spans_subchannels"] == 2.0
    assert pool["one_subchannel_available_coded_bits"] == pytest.approx(4838.4)
    assert pool["minimum_code_rate_to_fit_current_block_in_one_subchannel"] == (
        pytest.approx(0.5753968254)
    )
    assert checks["configured_block_fits_full_carrier"] is True
    assert checks["configured_block_fits_one_pool_subchannel"] is False
    assert checks["link_allocation_equals_one_pool_subchannel"] is False
    assert decision["physical_profile_freeze_ready"] is False
    assert decision["training_authorization"] is False


def test_audit_records_missing_calibration_and_never_opens_test_split(
    audit: dict[str, object], tmp_path: Path
) -> None:
    assert audit["schema"] == RF_PHYSICAL_PROFILE_AUDIT_SCHEMA
    checks = audit["checks"]
    assert isinstance(checks, dict)
    assert checks["declared_calibration_artifact_exists"] is False
    assert audit["test_split_opened"] is False
    assert audit["training_run_started"] is False
    output = write_rf_physical_profile_audit(audit, tmp_path / "audit.json")
    assert json.loads(output.read_text(encoding="utf-8")) == audit
