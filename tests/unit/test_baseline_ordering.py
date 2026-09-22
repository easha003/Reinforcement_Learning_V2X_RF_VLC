"""Guaranteed baseline ordering in analytical limits and matched rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.mean_field.baseline_ordering import (
    BASELINE_ORDERING_SCHEMA,
    REQUIRED_FIXED_POLICIES,
    BaselineOrderingError,
    verify_baseline_ordering,
)
from hybrid_v2x_rl.mean_field.baselines import baseline_policy
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.matched_campaign import run_matched_policy_campaign
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

TRACE_IDS = (
    "synthetic-d10-train-991",
    "synthetic-d10-validation-991",
    "synthetic-d10-test-991",
)


def _vehicle(trace_id: str, time_s: float, index: int) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=trace_id,
        time_s=time_s,
        vehicle_id=f"veh-{index}",
        x_m=8.0 * index + 4.0 * time_s,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=4.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


def _write_trace(root: Path, config, trace_id: str) -> None:
    times = tuple(0.05 * index for index in range(7))
    vehicles = [
        _vehicle(trace_id, time_s, vehicle_index)
        for time_s in times
        for vehicle_index in range(1, 4)
    ]
    pair = {
        "trace_id": trace_id,
        "pair_id": "pair-a",
        "tx_id": "veh-1",
        "rx_id": "veh-2",
        "start_s": 0.0,
        "end_s": 0.3,
        "duration_s": 0.3,
        "initial_distance_m": 8.0,
        "route_id": "route-0",
        "eligibility_reason": "trace_end",
        "has_intervening_vehicle": False,
    }
    MobilityTraceWriter(
        ArtifactStore(root / "artifacts"),
        rows_per_part=7,
    ).write(
        trace_id=trace_id,
        vehicles=vehicles,
        signals=(),
        pairs=(pair,),
        network_definition="{}",
        route_definition="{}",
        resolved_config_yaml="test: baseline-ordering\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 13},
    )


@pytest.fixture()
def ordering_inputs(tmp_path: Path):
    base = load_headline_config(Path.cwd())
    splits = TraceSplitConfig(
        train=(TRACE_IDS[0],),
        validation=(TRACE_IDS[1],),
        test=(TRACE_IDS[2],),
    )
    environment = base.environment.model_copy(update={"splits": splits})
    paths = base.paths.model_copy(
        update={"trace_root": tmp_path / "artifacts" / "traces"}
    )
    config = base.model_copy(update={"environment": environment, "paths": paths})
    for trace_id in TRACE_IDS:
        _write_trace(tmp_path, config, trace_id)
    return config, TraceCatalog.from_splits(config.paths.trace_root, splits)


def test_guaranteed_ordering_passes_and_persists_evidence(
    ordering_inputs,
    tmp_path: Path,
) -> None:
    config, catalog = ordering_inputs
    campaign = run_matched_policy_campaign(
        config,
        catalog,
        policies=tuple(baseline_policy(name) for name in REQUIRED_FIXED_POLICIES),
        environment_seed=81,
        max_frames=2,
    )

    ordering = verify_baseline_ordering(config, campaign)

    assert ordering.passed
    assert ordering.failed_checks == ()
    assert ordering.trace_count == 3
    assert ordering.usable_transitions > 0
    assert {check.scenario for check in ordering.checks} >= {
        "fixed-load-interior-risk",
        "perfect-vlc",
        "certain-vlc-failure",
        "perfect-rf-attempt",
        "certain-rf-attempt-failure",
        f"matched-trace:{TRACE_IDS[0]}",
    }
    duplicate_checks = tuple(
        check
        for check in ordering.checks
        if check.name.startswith("duplicate-")
    )
    assert duplicate_checks
    assert all(check.passed for check in duplicate_checks)

    output = ordering.write_json(tmp_path / "ordering.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == BASELINE_ORDERING_SCHEMA
    assert payload["passed"] is True
    assert payload["failed_check_count"] == 0
    assert payload["check_count"] == len(ordering.checks)


def test_ordering_rejects_a_campaign_missing_fixed_comparators(
    ordering_inputs,
) -> None:
    config, catalog = ordering_inputs
    campaign = run_matched_policy_campaign(
        config,
        catalog,
        policies=(baseline_policy("always-vlc"), baseline_policy("always-rf-1")),
        environment_seed=81,
        max_frames=2,
    )

    with pytest.raises(BaselineOrderingError, match="missing fixed baselines"):
        verify_baseline_ordering(config, campaign)
