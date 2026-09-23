"""Reliability-first, matched deployable-to-oracle gap measurement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.mean_field.baselines import (
    BASELINE_ORACLE,
    SupervisedOpticalRiskEstimator,
    baseline_policy,
)
from hybrid_v2x_rl.mean_field.congestion_feedback import ActorObservationSchema
from hybrid_v2x_rl.mean_field.density_metrics import build_density_metrics_report
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.matched_campaign import run_matched_policy_campaign
from hybrid_v2x_rl.mean_field.oracle_gap import (
    ORACLE_GAP_SCHEMA,
    REQUIRED_DEPLOYABLE_BASELINES,
    REQUIRED_GAP_POLICIES,
    OracleGapError,
    build_oracle_gap_report,
)
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord
from hybrid_v2x_rl.observation.builder import ObservationBuilder

TRAIN_TRACE = "synthetic-d10-train-993"
VALIDATION_TRACE = "synthetic-d10-validation-993"
TEST_TRACES = (
    "synthetic-d10-test-993",
    "synthetic-d20-test-993",
    "synthetic-d30-test-993",
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
        resolved_config_yaml="test: oracle-gap\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 13},
    )


def _estimator(config) -> SupervisedOpticalRiskEstimator:
    columns = ActorObservationSchema(
        local=ObservationBuilder.from_config(config.observation).schema
    ).columns
    rows = np.zeros((64, len(columns)), dtype=np.float64)
    distance = np.linspace(3.0, 35.0, len(rows))
    rows[:, columns.index("neighbor_count")] = np.linspace(20.0, 180.0, len(rows))
    rows[:, columns.index("pair_distance")] = distance
    rows[:, columns.index("pair_bearing")] = np.linspace(-0.2, 0.2, len(rows))
    rows[:, columns.index("relative_speed")] = np.sin(distance)
    rows[:, columns.index("heading_difference")] = np.linspace(0.0, 0.5, len(rows))
    rows[:, columns.index("optical_fov_margin")] = np.linspace(1.1, -0.1, len(rows))
    rows[:, columns.index("distance_to_junction")] = np.linspace(50.0, 0.0, len(rows))
    rows[:, columns.index("path_spans_junction")] = (distance > 28.0).astype(float)
    rows[:, columns.index("predicted_blockage_probability")] = np.linspace(
        0.001, 0.9, len(rows)
    )
    rows[:, columns.index("predictor_confidence")] = np.linspace(0.95, 0.6, len(rows))
    rows[:, columns.index("track_age")] = np.linspace(0.0, 0.1, len(rows))
    risks = np.clip(1.0 / (1.0 + np.exp(-(distance - 14.0))), 1e-6, 1.0 - 1e-6)
    return SupervisedOpticalRiskEstimator.fit(rows, risks, columns=columns)


@pytest.fixture()
def gap_inputs(tmp_path: Path):
    base = load_headline_config(Path.cwd())
    splits = TraceSplitConfig(
        train=(TRAIN_TRACE,),
        validation=(VALIDATION_TRACE,),
        test=TEST_TRACES,
    )
    environment = base.environment.model_copy(update={"splits": splits})
    service = base.service.model_copy(update={"miss_budget": 0.99})
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
            "service": service,
            "evaluation": evaluation,
            "paths": paths,
        }
    )
    for trace_id in (TRAIN_TRACE, VALIDATION_TRACE, *TEST_TRACES):
        _write_trace(tmp_path, config, trace_id)
    return config, TraceCatalog.from_splits(config.paths.trace_root, splits)


def _complete_policies(config):
    estimator = _estimator(config)
    return tuple(
        baseline_policy(name, supervised_estimator=estimator)
        for name in REQUIRED_GAP_POLICIES
    )


def _reports(config, catalog, *, max_frames: int | None):
    campaign = run_matched_policy_campaign(
        config,
        catalog,
        policies=_complete_policies(config),
        environment_seed=81,
        max_frames=max_frames,
    )
    density = build_density_metrics_report(
        config,
        campaign,
        bootstrap_replicates=1_000,
        bootstrap_seed=17,
    )
    return campaign, density


def test_cutoff_campaign_reports_diagnostic_without_gap_claim(
    gap_inputs,
    tmp_path: Path,
) -> None:
    config, catalog = gap_inputs
    campaign, density = _reports(config, catalog, max_frames=2)

    gap = build_oracle_gap_report(config, campaign, density)

    assert not gap.comparison_ready
    assert not gap.all_densities_comparable
    assert tuple(row.density for row in gap.densities) == (10.0, 20.0, 30.0)
    for row in gap.densities:
        assert row.status == "diagnostic"
        assert row.closest_deployable_policy in REQUIRED_DEPLOYABLE_BASELINES
        assert row.feasible_deployable_policies == ()
        assert row.best_deployable_policy is None
        assert row.oracle_meets_budget is None
        assert row.estimates == ()

    output = gap.write_json(tmp_path / "gap.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == ORACLE_GAP_SCHEMA
    assert payload["interval_method"] == "paired_trajectory_pair_cluster_bootstrap"
    assert payload["comparison_ready"] is False


def test_complete_evidence_selects_feasible_deployable_and_pairs_gap(
    gap_inputs,
) -> None:
    config, catalog = gap_inputs
    campaign, density = _reports(config, catalog, max_frames=None)

    gap = build_oracle_gap_report(config, campaign, density)

    assert gap.comparison_ready
    assert gap.all_densities_comparable
    expected_metrics = {
        "mean_activation_cost",
        "mean_reserved_rf_attempts",
        "rf_use_fraction",
        "vlc_use_fraction",
        "duplication_fraction",
        "conditional_miss_rate",
        "sampled_miss_rate",
    }
    for row in gap.densities:
        assert row.status == "comparable"
        assert row.oracle_meets_budget is True
        assert row.best_deployable_policy in row.feasible_deployable_policies
        assert {estimate.metric for estimate in row.estimates} == expected_metrics
        for estimate in row.estimates:
            assert estimate.gap == pytest.approx(
                estimate.deployable_value - estimate.oracle_value
            )
            # One pair episode per test density makes this paired bootstrap
            # deliberately degenerate and therefore exactly reproducible.
            assert estimate.confidence_lower == pytest.approx(estimate.gap)
            assert estimate.confidence_upper == pytest.approx(estimate.gap)


def test_gap_refuses_to_name_best_from_an_incomplete_baseline_suite(
    gap_inputs,
) -> None:
    config, catalog = gap_inputs
    policies = (
        baseline_policy("always-vlc"),
        baseline_policy("duplicate-all"),
        baseline_policy(BASELINE_ORACLE),
    )
    campaign = run_matched_policy_campaign(
        config,
        catalog,
        policies=policies,
        environment_seed=81,
        max_frames=2,
    )
    density = build_density_metrics_report(
        config,
        campaign,
        bootstrap_replicates=1_000,
        bootstrap_seed=17,
    )

    with pytest.raises(OracleGapError, match="complete baseline suite"):
        build_oracle_gap_report(config, campaign, density)
