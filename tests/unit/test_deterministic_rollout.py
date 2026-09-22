"""Integrated deterministic-rollout checks over a tiny immutable trace."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.baselines import (
    BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_CONTEXTUAL,
    BASELINE_DUPLICATE_ALL,
    BASELINE_GEOMETRY_THRESHOLD,
    BASELINE_ORACLE,
    CONTEXTUAL_FEATURES,
    OPTICAL_ESTIMATOR_FEATURES,
    BaselinePolicyError,
    SupervisedOpticalRiskEstimator,
    baseline_policy,
    run_baseline_rollout,
)
from hybrid_v2x_rl.mean_field.congestion_feedback import ActorObservationSchema
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    DeterministicRolloutError,
    canonical_policy_name,
    run_deterministic_rollout,
)
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord
from hybrid_v2x_rl.observation.builder import ObservationBuilder

TRACE_ID = "synthetic-d10-train-900"


def _vehicle(time_s: float, index: int) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
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


def _pair(
    pair_id: str,
    tx_id: str,
    rx_id: str,
    start_s: float,
    end_s: float,
    reason: str,
) -> dict[str, object]:
    return {
        "trace_id": TRACE_ID,
        "pair_id": pair_id,
        "tx_id": tx_id,
        "rx_id": rx_id,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": end_s - start_s,
        "initial_distance_m": 8.0,
        "route_id": "route-0",
        "eligibility_reason": reason,
        "has_intervening_vehicle": False,
    }


@pytest.fixture(scope="module")
def config():
    return load_headline_config(Path.cwd())


@pytest.fixture()
def source(tmp_path: Path, config) -> FrameTraceSource:
    times = tuple(0.05 * index for index in range(7))
    vehicles = [
        _vehicle(time_s, vehicle_index) for time_s in times for vehicle_index in range(1, 5)
    ]
    pairs = (
        _pair("pair-a", "veh-1", "veh-2", 0.0, 0.3, "route_diverged"),
        _pair("pair-b", "veh-2", "veh-3", 0.1, 0.3, "trace_end"),
        _pair("pair-c", "veh-3", "veh-4", 0.0, 0.2, "max_duration"),
    )
    artifact = MobilityTraceWriter(
        ArtifactStore(tmp_path / "artifacts"),
        rows_per_part=7,
    ).write(
        trace_id=TRACE_ID,
        vehicles=vehicles,
        signals=(),
        pairs=pairs,
        network_definition="{}",
        route_definition="{}",
        resolved_config_yaml="test: deterministic-rollout\n",
        config_hash=config_hash(config),
        code_version="test",
        random_seeds={"mobility": 13},
    )
    return FrameTraceSource.discover(artifact.path, expected_split="train")


def test_random_rollout_is_bit_replayable_and_seed_addressed(config, source) -> None:
    first = run_deterministic_rollout(
        config,
        source,
        policy="random",
        environment_seed=81,
        policy_seed=92,
    )
    repeated = run_deterministic_rollout(
        config,
        source,
        policy="random",
        environment_seed=81,
        policy_seed=92,
    )
    changed = run_deterministic_rollout(
        config,
        source,
        policy="random",
        environment_seed=81,
        policy_seed=93,
    )

    assert repeated == first
    assert first.source_exhausted
    assert first.frames == first.available_frames == 4
    assert first.transitions == 10
    assert first.usable_transitions + first.fallback_transitions == first.transitions
    assert first.usable_transitions > 0
    assert first.normalization_training_rows == first.usable_transitions
    assert first.internal_truncations == 1
    assert first.natural_terminations == 1
    assert first.trace_end_truncations == 1
    assert changed.fingerprint != first.fingerprint


def test_scripted_cycle_composes_shared_pool_and_cutoff(config, source) -> None:
    report = run_deterministic_rollout(
        config,
        source,
        policy="cycle",
        environment_seed=81,
        policy_seed=92,
        max_frames=3,
    )

    assert report.frames == 3
    assert not report.source_exhausted
    assert report.reserved_rf_attempts > 0
    assert report.vlc_activations > 0
    assert report.max_population == 3
    assert report.max_pool_utilization >= 0.0


def test_policy_names_accept_fixed_actions_and_reject_unknown_names() -> None:
    assert canonical_policy_name("rf_2") == "RF-2"
    assert canonical_policy_name("DUP-4") == "DUP-4"
    with pytest.raises(DeterministicRolloutError, match="unknown validation policy"):
        canonical_policy_name("oracle-truth")


def _fitted_estimator(config) -> SupervisedOpticalRiskEstimator:
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


@pytest.mark.parametrize(
    ("name", "rf_attempts", "uses_vlc"),
    [
        (BASELINE_ALWAYS_VLC, 0, True),
        *((name, attempts, False) for attempts, name in enumerate(BASELINE_ALWAYS_RF, 1)),
        (BASELINE_DUPLICATE_ALL, 4, True),
    ],
)
def test_fixed_baselines_share_fallback_and_resource_accounting(
    config,
    source,
    name: str,
    rf_attempts: int,
    uses_vlc: bool,
) -> None:
    report = run_baseline_rollout(
        config,
        source,
        policy=baseline_policy(name),
        environment_seed=81,
        max_frames=2,
    )

    # Every unusable row takes the contract fallback DUP-4; all other rows
    # carry the baseline action through the common action ledger.
    assert report.policy == name
    assert report.reserved_rf_attempts == (
        report.usable_transitions * rf_attempts + report.fallback_transitions * 4
    )
    assert report.vlc_activations == (
        report.usable_transitions * int(uses_vlc) + report.fallback_transitions
    )


@pytest.mark.parametrize(
    "name",
    [BASELINE_GEOMETRY_THRESHOLD, BASELINE_CONTEXTUAL, BASELINE_ORACLE],
)
def test_rule_and_oracle_baselines_run_through_complete_population_path(
    config,
    source,
    name: str,
) -> None:
    report = run_baseline_rollout(
        config,
        source,
        policy=baseline_policy(name),
        environment_seed=81,
        max_frames=2,
    )

    assert report.policy == name
    assert report.frames == 2
    assert report.transitions > 0
    assert report.normalization_training_rows == report.usable_transitions


def test_supervised_baseline_requires_training_fit_and_runs_common_path(config, source) -> None:
    with pytest.raises(BaselinePolicyError, match="training-fitted estimator"):
        baseline_policy("supervised-risk-allocation")

    estimator = _fitted_estimator(config)
    report = run_baseline_rollout(
        config,
        source,
        policy=baseline_policy(
            "supervised-risk-allocation",
            supervised_estimator=estimator,
        ),
        environment_seed=81,
        max_frames=2,
    )

    assert report.policy == "supervised-risk-allocation"
    assert report.transitions > 0
    assert estimator.training_rows == 64


def test_shared_runner_never_passes_truth_to_a_causal_policy(config, source) -> None:
    class CausalProbe:
        name = "causal-truth-boundary-probe"
        requires_oracle_truth = False

        def __init__(self) -> None:
            self.calls = 0

        def select_actions(self, decision, *, channel_truth):
            assert channel_truth is None
            self.calls += 1
            return tuple(
                PolicyAction.VLC if row.usable else None
                for row in decision.actor_frame.rows
            )

    probe = CausalProbe()
    run_baseline_rollout(
        config,
        source,
        policy=probe,
        environment_seed=81,
        max_frames=2,
    )

    assert probe.calls == 2


def test_contextual_and_supervised_features_exclude_temporal_link_history() -> None:
    forbidden = {
        "rf_quality",
        "rf_quality_age",
        "vlc_quality",
        "vlc_quality_age",
        "previous_action",
        "last_delivery_outcome",
        "consecutive_miss_count",
        "delayed_mean_rf_attempt_fraction",
        "mean_field_valid",
    }
    for features in (CONTEXTUAL_FEATURES, OPTICAL_ESTIMATOR_FEATURES):
        assert forbidden.isdisjoint(features)
        assert not any("_history[" in name for name in features)
