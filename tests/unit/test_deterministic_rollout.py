"""Integrated deterministic-rollout checks over a tiny immutable trace."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    DeterministicRolloutError,
    canonical_policy_name,
    run_deterministic_rollout,
)
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource
from hybrid_v2x_rl.mobility.trace_io import MobilityTraceWriter, VehicleTraceRecord

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
