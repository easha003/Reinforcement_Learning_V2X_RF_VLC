"""Per-density metrics from pair-episode clustered matched campaigns."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.baselines import baseline_policy
from hybrid_v2x_rl.mean_field.density_metrics import (
    DENSITY_METRICS_SCHEMA,
    build_density_metrics_report,
)
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.matched_campaign import (
    MatchedCampaignError,
    MatchedTraceComparison,
    run_matched_policy_campaign,
)
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

TRAIN_TRACE = "synthetic-d10-train-992"
VALIDATION_TRACE = "synthetic-d10-validation-992"
TEST_TRACES = (
    "synthetic-d10-test-992",
    "synthetic-d20-test-992",
    "synthetic-d30-test-992",
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
        resolved_config_yaml="test: density-metrics\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 13},
    )


@pytest.fixture()
def density_inputs(tmp_path: Path):
    base = load_headline_config(Path.cwd())
    splits = TraceSplitConfig(
        train=(TRAIN_TRACE,),
        validation=(VALIDATION_TRACE,),
        test=TEST_TRACES,
    )
    environment = base.environment.model_copy(update={"splits": splits})
    evaluation = base.evaluation.model_copy(
        update={
            "min_packets_per_policy_density": 1,
            "min_trajectory_pair_clusters": 1,
            "bootstrap_replicates": 1_000,
        }
    )
    paths = base.paths.model_copy(
        update={"trace_root": tmp_path / "artifacts" / "traces"}
    )
    config = base.model_copy(
        update={
            "environment": environment,
            "evaluation": evaluation,
            "paths": paths,
        }
    )
    for trace_id in (TRAIN_TRACE, VALIDATION_TRACE, *TEST_TRACES):
        _write_trace(tmp_path, config, trace_id)
    return config, TraceCatalog.from_splits(config.paths.trace_root, splits)


def _campaign(config, catalog, *, max_frames: int | None):
    return run_matched_policy_campaign(
        config,
        catalog,
        policies=(baseline_policy("always-vlc"), baseline_policy("always-rf-1")),
        environment_seed=81,
        max_frames=max_frames,
    )


def test_density_report_groups_test_metrics_and_labels_a_cutoff_diagnostic(
    density_inputs,
    tmp_path: Path,
) -> None:
    config, catalog = density_inputs
    campaign = _campaign(config, catalog, max_frames=2)

    report = build_density_metrics_report(
        config,
        campaign,
        bootstrap_replicates=1_000,
        bootstrap_seed=17,
    )

    assert report.split == "test"
    assert report.policies == ("always-vlc", "always-rf-1")
    assert tuple(block.density for block in report.densities) == (10.0, 20.0, 30.0)
    assert not report.evaluation_ready
    for density, block in zip((10, 20, 30), report.densities, strict=True):
        assert tuple(metric.policy for metric in block.policies) == report.policies
        for metric in block.policies:
            assert metric.trace_ids == (f"synthetic-d{density}-test-992",)
            assert metric.frames == 2
            assert metric.packets == 2
            assert metric.usable_packets == 1
            assert metric.fallback_packets == 1
            assert metric.pair_episode_clusters == 1
            assert not metric.source_exhausted
            assert metric.meets_miss_budget is None
            assert sum(metric.action_counts) == metric.packets
            assert 0.0 <= metric.conditional_miss_rate <= 1.0
            assert 0.0 <= metric.conditional_bootstrap_upper <= 1.0

        vlc, rf = block.policies
        assert vlc.mean_activation_cost == pytest.approx(3.0)
        assert vlc.mean_reserved_rf_attempts == pytest.approx(2.0)
        assert vlc.rf_use_fraction == pytest.approx(0.5)
        assert vlc.vlc_use_fraction == pytest.approx(1.0)
        assert vlc.duplication_fraction == pytest.approx(0.5)
        assert vlc.action_counts[int(PolicyAction.VLC)] == 1
        assert vlc.action_counts[int(PolicyAction.DUP_4)] == 1

        assert rf.mean_activation_cost == pytest.approx(3.0)
        assert rf.mean_reserved_rf_attempts == pytest.approx(2.5)
        assert rf.rf_use_fraction == pytest.approx(1.0)
        assert rf.vlc_use_fraction == pytest.approx(0.5)
        assert rf.duplication_fraction == pytest.approx(0.5)
        assert rf.action_counts[int(PolicyAction.RF_1)] == 1
        assert rf.action_counts[int(PolicyAction.DUP_4)] == 1

    output = report.write_json(tmp_path / "density.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == DENSITY_METRICS_SCHEMA
    assert payload["interval_method"] == "trajectory_pair_cluster_bootstrap"
    assert payload["matched_bootstrap_across_policies"] is True
    assert payload["evaluation_ready"] is False


def test_full_sources_enable_budget_verdict_when_configured_evidence_is_met(
    density_inputs,
) -> None:
    config, catalog = density_inputs
    campaign = _campaign(config, catalog, max_frames=None)

    report = build_density_metrics_report(
        config,
        campaign,
        bootstrap_replicates=1_000,
        bootstrap_seed=17,
    )

    assert report.evaluation_ready
    for block in report.densities:
        for metric in block.policies:
            assert metric.source_exhausted
            assert metric.sufficient_packets
            assert metric.sufficient_clusters
            assert metric.evaluation_ready
            assert type(metric.meets_miss_budget) is bool


def test_matched_comparison_rejects_different_episode_cluster_identity(
    density_inputs,
) -> None:
    config, catalog = density_inputs
    campaign = _campaign(config, catalog, max_frames=2)
    comparison = campaign.comparisons[-1]
    reference, candidate = comparison.reports
    first_cluster, *remaining = candidate.episode_clusters
    changed_cluster = replace(first_cluster, pair_id="different-pair")
    changed_report = replace(
        candidate,
        episode_clusters=(changed_cluster, *remaining),
    )

    with pytest.raises(MatchedCampaignError, match="pair-episode clusters"):
        MatchedTraceComparison(
            source=comparison.source,
            reports=(reference, changed_report),
        )
