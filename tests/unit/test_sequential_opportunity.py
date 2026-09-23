"""Phase 6 necessary-condition tests for sequential-control opportunity."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.config.models import TraceSplitConfig
from hybrid_v2x_rl.env.cache import ColumnPlan, TransitionCache
from hybrid_v2x_rl.mean_field.sequential_opportunity import (
    measure_history_opportunity,
    measure_population_opportunity,
)
from hybrid_v2x_rl.observation.builder import ObservationSchema


def _config(tmp_path: Path):
    base = load_headline_config(Path.cwd())
    splits = TraceSplitConfig(
        train=("opportunity-train",),
        validation=("opportunity-validation",),
        test=("opportunity-test",),
    )
    environment = base.environment.model_copy(update={"splits": splits})
    paths = base.paths.model_copy(update={"project_root": tmp_path})
    return base.model_copy(update={"environment": environment, "paths": paths})


def _write_history_cache(
    root: Path,
    config,
    *,
    split: str,
    trace_id: str,
    episodes: int = 60,
    packets_per_episode: int = 12,
) -> Path:
    schema = ObservationSchema(
        features=tuple(config.observation.features),
        history_packets=config.observation.history_packets,
    )
    plan = ColumnPlan.from_schema(schema)
    packets = episodes * packets_per_episode
    trace = np.zeros((packets, len(plan.trace_columns)), dtype=np.float32)
    risks = np.empty((packets, 2), dtype=np.float32)
    delivered = np.empty((packets, 2), dtype=np.uint8)
    quality = np.empty((packets, 2), dtype=np.float32)
    times = np.tile(
        np.arange(packets_per_episode, dtype=np.float64) * 0.1,
        episodes,
    )
    episode = np.repeat(np.arange(episodes, dtype=np.int32), packets_per_episode)
    final = np.zeros(packets, dtype=np.uint8)
    final[packets_per_episode - 1 :: packets_per_episode] = 1

    for episode_id in range(episodes):
        start = episode_id * packets_per_episode
        stop = start + packets_per_episode
        vlc_risk = 0.01 if episode_id % 2 == 0 else 0.85
        risks[start:stop, 0] = 0.02
        risks[start:stop, 1] = vlc_risk
        delivered[start:stop, 0] = 1
        delivered[start:stop, 1] = int(vlc_risk < 0.5)
        quality[start:stop, 0] = 0.75
        quality[start:stop, 1] = 1.0 - vlc_risk

    path = root / f"{split}-d20-000"
    TransitionCache.write(
        path,
        trace=trace,
        risk=risks,
        delivered=delivered,
        quality=quality,
        time_s=times,
        episode=episode,
        final=final,
        manifest={
            "density": 20,
            "split": split,
            "replicate": 0,
            "trace": trace_id,
            "config_hash": config_hash(config),
            "trace_features": list(plan.trace_features),
            "history_packets": schema.history_packets,
        },
    )
    return path


def test_causal_history_improves_held_out_optical_risk_prediction(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    train = _write_history_cache(
        tmp_path,
        config,
        split="train",
        trace_id="opportunity-train",
    )
    test = _write_history_cache(
        tmp_path,
        config,
        split="test",
        trace_id="opportunity-test",
    )

    result = measure_history_opportunity(
        config,
        training_cache_paths=(train,),
        evaluation_cache_paths=(test,),
        confidence_level=0.95,
        bootstrap_replicates=1_000,
        bootstrap_seed=17,
        minimum_evaluation_rows=1,
        minimum_evaluation_clusters=1,
    )

    assert result.status == "opportunity"
    assert result.evidence_ready
    assert result.opportunity_detected is True
    optical = next(item for item in result.estimates if item.link == "vlc")
    assert optical.history_brier < optical.contextual_brier
    assert optical.confidence_lower > 0.0
    assert dict(result.behavior_action_counts).keys() == {"DUP", "RF", "VLC"}


def test_history_probe_does_not_claim_from_undersized_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    train = _write_history_cache(
        tmp_path,
        config,
        split="train",
        trace_id="opportunity-train",
    )
    test = _write_history_cache(
        tmp_path,
        config,
        split="test",
        trace_id="opportunity-test",
    )

    result = measure_history_opportunity(
        config,
        training_cache_paths=(train,),
        evaluation_cache_paths=(test,),
        confidence_level=0.95,
        bootstrap_replicates=1_000,
        bootstrap_seed=17,
        minimum_evaluation_rows=100_000,
        minimum_evaluation_clusters=1,
    )

    assert result.status == "diagnostic"
    assert not result.evidence_ready
    assert result.opportunity_detected is None
    assert result.estimates


def test_population_actions_change_focal_rf_feasibility(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = measure_population_opportunity(
        config,
        populations_by_density={20.0: (300, 320, 340)},
        evaluation_sources=("opportunity-test",),
        missing_evaluation_sources=(),
    )

    assert result.status == "opportunity"
    assert result.evidence_ready
    assert result.opportunity_detected is True
    density = result.densities[0]
    assert density.opportunity_detected
    rf_three = density.attempts[2]
    assert rf_three.decision_relevant
    assert rf_three.low_demand_risk_at_median <= config.service.miss_budget
    assert rf_three.all_rf_risk_at_median > config.service.miss_budget


def test_incomplete_population_sources_remain_diagnostic(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = measure_population_opportunity(
        config,
        populations_by_density={20.0: (300,)},
        evaluation_sources=(),
        missing_evaluation_sources=("opportunity-test",),
    )

    assert result.status == "diagnostic"
    assert not result.evidence_ready
    assert result.opportunity_detected is None
