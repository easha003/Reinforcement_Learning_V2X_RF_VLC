"""Analytic mobility campaign tests.

Exercise the full §24.1 flow end to end on a deliberately short run: build the
network, simulate, archive an immutable trace, extract tagged pairs, and
produce a Gate-1 report.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import headline_config_layers, load_config
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import ConfigurationError
from hybrid_v2x_rl.mobility.pipeline import (
    GridTracePipeline,
    SplitCounts,
    TraceCampaignPlan,
    trace_id_for,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name="config")
def _config() -> ProjectConfig:
    """Headline configuration shortened so a full campaign fits a unit test."""

    resolved = load_config(headline_config_layers(PROJECT_ROOT), project_root=PROJECT_ROOT)
    return resolved.model_copy(
        update={
            "mobility": resolved.mobility.model_copy(
                update={"warmup_s": 20.0, "trace_duration_s": 10.0}
            )
        }
    )


def _pipeline(config: ProjectConfig, tmp_path: Path) -> GridTracePipeline:
    return GridTracePipeline(config, TraceCampaignPlan(output_root=tmp_path, code_version="test"))


def test_campaign_produces_a_verifiable_trace_artifact(
    config: ProjectConfig, tmp_path: Path
) -> None:
    result = _pipeline(config, tmp_path).run_density(20.0, index=0)

    written = {path.name for path in result.artifact_path.iterdir()}
    assert {"manifest.json", "network.json", "routes.json", "resolved_config.yaml"} <= written
    assert (result.artifact_path / "vehicles").is_dir()

    # The archived network definition replaces SUMO's network.net.xml and must
    # be enough to rebuild the topology.
    network = json.loads((result.artifact_path / "network.json").read_text(encoding="utf-8"))
    assert network["avenues"] == config.mobility.grid.avenues
    assert network["cross_street_spacing_m"] == pytest.approx(61.0)


def test_archived_routes_are_edge_sequences(config: ProjectConfig, tmp_path: Path) -> None:
    """Persisting routes is what lets pair extraction test a shared path.

    SUMO wrote anonymous embedded routes, so every vehicle received a unique
    auto-generated identifier and pair extraction returned zero episodes.
    """

    result = _pipeline(config, tmp_path).run_density(20.0, index=0)
    routes = json.loads((result.artifact_path / "routes.json").read_text(encoding="utf-8"))

    assert routes
    sample = next(iter(routes.values()))
    assert isinstance(sample, list) and sample
    assert all(isinstance(edge_id, str) and edge_id for edge_id in sample)


def test_campaign_extracts_tagged_pairs(config: ProjectConfig, tmp_path: Path) -> None:
    """The SUMO backend produced zero episodes; this must produce many."""

    result = _pipeline(config, tmp_path).run_density(40.0, index=1)

    assert len(result.pair_segments) > 0
    for segment in result.pair_segments:
        assert segment.tx_id != segment.rx_id
        assert (
            config.geometry.min_separation_m
            <= segment.initial_distance_m
            <= config.geometry.max_separation_m
        )


def test_realized_density_is_reported_not_assumed(config: ProjectConfig, tmp_path: Path) -> None:
    """Work plan §4.2: density is measured, never inferred from the request."""

    result = _pipeline(config, tmp_path).run_density(40.0, index=1)

    assert result.realized_density_veh_per_lane_km == pytest.approx(40.0, rel=0.05)
    assert result.target_density_veh_per_lane_km == 40.0


def test_campaign_manifest_records_provenance(config: ProjectConfig, tmp_path: Path) -> None:
    result = _pipeline(config, tmp_path).run(
        target_densities=[20.0], splits=SplitCounts(train=1, validation=1, test=1)
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["mobility_model"] == "analytic-manhattan-grid"
    assert manifest["config_hash"] == result.config_hash
    assert manifest["network"]["total_lane_length_m"] == pytest.approx(37_332.0)
    # one density x three splits
    assert len(manifest["densities"]) == 3
    assert set(manifest["splits"]) == {"train", "validation", "test"}
    assert all(len(ids) == 1 for ids in manifest["splits"].values())


def test_non_grid_mobility_is_rejected(config: ProjectConfig, tmp_path: Path) -> None:
    broken = config.model_copy(
        update={"mobility": config.mobility.model_copy(update={"grid": None})}
    )
    with pytest.raises(ConfigurationError, match="requires mobility.grid"):
        _pipeline(broken, tmp_path)


def test_plan_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="code_version"):
        TraceCampaignPlan(output_root=tmp_path, code_version="  ")
    with pytest.raises(ValueError, match="validation_sample_period_s"):
        TraceCampaignPlan(output_root=tmp_path, code_version="t", validation_sample_period_s=0.0)


def test_tagged_pairs_are_persisted_with_the_trace(config: ProjectConfig, tmp_path: Path) -> None:
    """Episodes must survive the run that produced them.

    Re-deriving them means re-reading every vehicle row in the trace, which at
    full length is tens of millions of Parquet rows per density.
    """

    import pyarrow.parquet as pq

    result = _pipeline(config, tmp_path).run_density(40.0, index=1)
    table = pq.read_table(result.artifact_path / "pairs.parquet")

    assert table.num_rows == len(result.pair_segments)
    assert set(table.column_names) >= {
        "trace_id",
        "pair_id",
        "tx_id",
        "rx_id",
        "start_s",
        "end_s",
        "duration_s",
        "initial_distance_m",
        "route_id",
        "eligibility_reason",
    }
    durations = table.column("duration_s").to_pylist()
    assert all(value >= 0.0 for value in durations)


def test_degenerate_episodes_are_counted_not_hidden(config: ProjectConfig, tmp_path: Path) -> None:
    """A zero-length episode carries no packet and must stay visible."""

    result = _pipeline(config, tmp_path).run_density(40.0, index=1)
    record = result.to_record()

    assert record["degenerate_pair_count"] == sum(
        1 for segment in result.pair_segments if segment.duration_s <= 0.0
    )
    assert record["usable_pair_count"] + record["degenerate_pair_count"] == record["pair_count"]


def test_trace_ids_encode_density_split_and_replicate() -> None:
    """Work plan §13.1 splits by trajectory and seed, so density alone is not
    a sufficient identity — each split needs its own realizations."""

    assert trace_id_for(10.0, "train", 0) == "synthetic-d10-train-000"
    assert trace_id_for(30.0, "test", 2) == "synthetic-d30-test-002"
    with pytest.raises(ValueError, match="unknown split"):
        trace_id_for(10.0, "holdout", 0)


def test_configured_splits_match_what_the_generator_produces() -> None:
    """The declared split IDs must be the ones a campaign actually writes.

    They previously named nine traces that were never generated, so a
    "held-out" evaluation would have silently had nothing to hold out.
    """

    resolved = load_config(headline_config_layers(PROJECT_ROOT), project_root=PROJECT_ROOT)
    counts = SplitCounts()
    densities = resolved.mobility.target_densities_veh_per_lane_km

    expected = {
        name: [
            trace_id_for(density, name, replicate)
            for replicate in range(number)
            for density in densities
        ]
        for name, number in counts.as_pairs()
    }
    splits = resolved.environment.splits

    assert list(splits.train) == expected["train"]
    assert list(splits.validation) == expected["validation"]
    assert list(splits.test) == expected["test"]


def test_splits_are_disjoint_and_cover_every_density() -> None:
    resolved = load_config(headline_config_layers(PROJECT_ROOT), project_root=PROJECT_ROOT)
    splits = resolved.environment.splits
    everything = list(splits.train) + list(splits.validation) + list(splits.test)

    assert len(everything) == len(set(everything)), "a trace may not appear in two splits"
    for density in resolved.mobility.target_densities_veh_per_lane_km:
        for name in ("train", "validation", "test"):
            assert any(
                f"-d{density:g}-{name}-" in trace_id for trace_id in getattr(splits, name)
            ), f"{name} split has no trace at density {density}"
