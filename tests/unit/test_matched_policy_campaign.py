"""Matched trace membership, packet tapes, and split-frozen normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.mean_field.baselines import baseline_policy
from hybrid_v2x_rl.mean_field.deterministic_rollout import run_policy_rollout
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.matched_campaign import (
    MATCHED_CAMPAIGN_SCHEMA,
    MatchedCampaignError,
    run_matched_policy_campaign,
)
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

TRAIN_TRACE = "synthetic-d10-train-990"
SECOND_TRAIN_TRACE = "synthetic-d20-train-990"
VALIDATION_TRACE = "synthetic-d10-validation-990"
TEST_TRACE = "synthetic-d10-test-990"


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


def _pair(trace_id: str, pair_id: str, start_s: float, end_s: float) -> dict[str, object]:
    return {
        "trace_id": trace_id,
        "pair_id": pair_id,
        "tx_id": "veh-1",
        "rx_id": "veh-2",
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "initial_distance_m": 8.0,
        "route_id": "route-0",
        "eligibility_reason": "trace_end",
        "has_intervening_vehicle": False,
    }


def _write_trace(root: Path, config, trace_id: str) -> None:
    times = tuple(0.05 * index for index in range(7))
    vehicles = [
        _vehicle(trace_id, time_s, vehicle_index)
        for time_s in times
        for vehicle_index in range(1, 4)
    ]
    MobilityTraceWriter(
        ArtifactStore(root / "artifacts"),
        rows_per_part=7,
    ).write(
        trace_id=trace_id,
        vehicles=vehicles,
        signals=(),
        pairs=(_pair(trace_id, "pair-a", 0.0, 0.3),),
        network_definition="{}",
        route_definition="{}",
        resolved_config_yaml="test: matched-policy-campaign\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 13},
    )


@pytest.fixture()
def campaign_inputs(tmp_path: Path):
    base = load_headline_config(Path.cwd())
    splits = TraceSplitConfig(
        train=(TRAIN_TRACE, SECOND_TRAIN_TRACE),
        validation=(VALIDATION_TRACE,),
        test=(TEST_TRACE,),
    )
    environment = base.environment.model_copy(update={"splits": splits})
    paths = base.paths.model_copy(
        update={"trace_root": tmp_path / "artifacts" / "traces"}
    )
    config = base.model_copy(update={"environment": environment, "paths": paths})
    for trace_id in (TRAIN_TRACE, SECOND_TRAIN_TRACE, VALIDATION_TRACE, TEST_TRACE):
        _write_trace(tmp_path, config, trace_id)
    catalog = TraceCatalog.from_splits(config.paths.trace_root, splits)
    return config, catalog


def test_campaign_matches_splits_tapes_and_frozen_normalization(
    campaign_inputs,
    tmp_path: Path,
) -> None:
    config, catalog = campaign_inputs
    policies = (baseline_policy("always-vlc"), baseline_policy("always-rf-1"))

    campaign = run_matched_policy_campaign(
        config,
        catalog,
        policies=policies,
        environment_seed=81,
        max_frames=2,
    )

    assert campaign.passed
    assert campaign.policies == ("always-vlc", "always-rf-1")
    assert [item.source.trace_id for item in campaign.comparisons] == [
        TRAIN_TRACE,
        SECOND_TRAIN_TRACE,
        VALIDATION_TRACE,
        TEST_TRACE,
    ]
    for comparison in campaign.comparisons:
        assert len({report.matched_tape_fingerprint for report in comparison.reports}) == 1
        assert len({report.transitions for report in comparison.reports}) == 1
        assert len({report.usable_transitions for report in comparison.reports}) == 1

    first_training = campaign.comparisons[0].reports
    final_training = campaign.comparisons[1].reports
    assert all(report.normalization_updates_enabled for report in first_training)
    assert all(not report.normalization_frozen for report in first_training)
    assert all(report.normalization_updates_enabled for report in final_training)
    assert all(report.normalization_frozen for report in final_training)
    assert all(
        report.normalization_training_rows == report.usable_transitions
        for report in (*first_training, *final_training)
    )
    training_totals = {
        report.policy: report.normalization_total_training_rows
        for report in final_training
    }
    for first, final in zip(first_training, final_training, strict=True):
        assert final.normalization_total_training_rows == (
            first.normalization_total_training_rows
            + final.normalization_training_rows
        )
    for comparison in campaign.comparisons[2:]:
        for report in comparison.reports:
            assert not report.normalization_updates_enabled
            assert report.normalization_training_rows == 0
            assert report.normalization_total_training_rows == training_totals[report.policy]
            assert report.normalization_frozen

    # Different actions change accounting while trace structure and tapes stay matched.
    vlc, rf = final_training
    assert vlc.reserved_rf_attempts < rf.reserved_rf_attempts
    assert vlc.vlc_activations > rf.vlc_activations

    output = campaign.write_json(tmp_path / "matched.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == MATCHED_CAMPAIGN_SCHEMA
    assert payload["trace_membership"] == {
        "train": [TRAIN_TRACE, SECOND_TRAIN_TRACE],
        "validation": [VALIDATION_TRACE],
        "test": [TEST_TRACE],
    }
    assert len(payload["normalization_checkpoints"]) == 2


def test_campaign_is_replayable_and_seed_changes_tape_identity(campaign_inputs) -> None:
    config, catalog = campaign_inputs
    policies = (baseline_policy("always-vlc"), baseline_policy("always-rf-1"))
    first = run_matched_policy_campaign(
        config,
        catalog,
        policies=policies,
        environment_seed=81,
        max_frames=1,
    )
    repeated = run_matched_policy_campaign(
        config,
        catalog,
        policies=policies,
        environment_seed=81,
        max_frames=1,
    )

    assert repeated.comparisons == first.comparisons
    assert repeated.normalization_checkpoints == first.normalization_checkpoints

    changed = run_policy_rollout(
        config,
        catalog.for_split("train")[0],
        policy=policies[0],
        environment_seed=82,
        max_frames=1,
    )
    assert (
        changed.matched_tape_fingerprint
        != first.comparisons[0].matched_tape_fingerprint
    )


def test_campaign_rejects_catalog_that_drops_a_configured_trace(campaign_inputs) -> None:
    config, catalog = campaign_inputs
    incomplete = TraceCatalog(sources=catalog.sources[:-1])

    with pytest.raises(MatchedCampaignError, match="split membership"):
        run_matched_policy_campaign(
            config,
            incomplete,
            policies=(baseline_policy("always-vlc"), baseline_policy("always-rf-1")),
            max_frames=1,
        )
