"""Phase 7 optimizer, critic, constraint, and dual metric logging."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.ppo import PPOError, PPOUpdateMetrics
from hybrid_v2x_rl.agents.training_metrics import (
    TRAINING_METRICS_SCHEMA,
    TrainingMetricsError,
    TrainingMetricsJSONL,
    aggregate_ppo_updates,
    build_training_iteration_metrics,
    explained_variance,
)
from hybrid_v2x_rl.config.models import DensityMultiplierConfig


def _ppo_metrics(*, minibatch_size: int, scale: float) -> PPOUpdateMetrics:
    return PPOUpdateMetrics(
        minibatch_size=minibatch_size,
        actor_loss=-scale,
        policy_loss=-0.5 * scale,
        reward_value_loss=scale,
        cost_value_loss=2.0 * scale,
        entropy=0.25 * scale,
        approximate_kl=0.01 * scale,
        clip_fraction=0.1 * scale,
        ratio_mean=1.0 + 0.05 * scale,
    )


def _record():
    dual = PerDensityDualAscent(
        (
            DensityMultiplierConfig(
                density_veh_per_lane_km=20.0,
                initial_value=0.2,
                learning_rate=1.0,
                maximum=2.0,
            ),
            DensityMultiplierConfig(
                density_veh_per_lane_km=10.0,
                initial_value=0.1,
                learning_rate=1.0,
                maximum=2.0,
            ),
        )
    )
    report = dual.update(
        densities_veh_per_lane_km=torch.tensor([10.0, 10.0]),
        costs=torch.tensor([0.4, 0.2]),
        miss_budget=0.1,
    )
    return build_training_iteration_metrics(
        config_hash="a" * 64,
        policy_seed=1001,
        iteration=0,
        environment_transitions=100,
        rollout_transitions=2,
        ppo_updates=(
            _ppo_metrics(minibatch_size=2, scale=1.0),
            _ppo_metrics(minibatch_size=1, scale=2.0),
        ),
        reward_predictions=torch.tensor([1.0, 2.0]),
        reward_targets=torch.tensor([1.0, 2.0]),
        cost_predictions=torch.tensor([0.0, 0.0]),
        cost_targets=torch.tensor([0.0, 1.0]),
        dual_report=report,
        dual_snapshot=dual.snapshot(),
    )


def test_explained_variance_has_known_and_constant_target_behavior() -> None:
    assert explained_variance(
        predictions=torch.tensor([1.0, 2.0, 3.0]),
        targets=torch.tensor([1.0, 2.0, 3.0]),
    ) == pytest.approx(1.0)
    assert explained_variance(
        predictions=torch.tensor([1.0, 1.0, 1.0]),
        targets=torch.tensor([1.0, 2.0, 3.0]),
    ) == pytest.approx(0.0)
    assert (
        explained_variance(
            predictions=torch.tensor([0.0, 2.0]),
            targets=torch.tensor([1.0, 1.0]),
        )
        == 0.0
    )


def test_ppo_metrics_are_weighted_by_minibatch_rows() -> None:
    combined = aggregate_ppo_updates(
        (
            _ppo_metrics(minibatch_size=2, scale=1.0),
            _ppo_metrics(minibatch_size=1, scale=2.0),
        )
    )

    expected_scale = 4.0 / 3.0
    assert combined.minibatch_updates == 2
    assert combined.optimizer_rows == 3
    assert combined.actor_loss == pytest.approx(-expected_scale)
    assert combined.policy_loss == pytest.approx(-0.5 * expected_scale)
    assert combined.reward_value_loss == pytest.approx(expected_scale)
    assert combined.cost_value_loss == pytest.approx(2.0 * expected_scale)
    assert combined.entropy == pytest.approx(0.25 * expected_scale)
    assert combined.approximate_kl == pytest.approx(0.01 * expected_scale)
    assert combined.clip_fraction == pytest.approx(0.1 * expected_scale)
    assert combined.ratio_mean == pytest.approx(1.0 + 0.05 * expected_scale)


def test_complete_record_contains_critic_and_per_density_dual_diagnostics() -> None:
    record = _record()
    payload = record.as_dict()

    assert payload["schema"] == TRAINING_METRICS_SCHEMA
    assert payload["config_hash"] == "a" * 64
    assert payload["reliability_cost_signal"] == "conditional_miss_probability"
    assert payload["critics"] == {
        "reward_explained_variance": 1.0,
        "cost_explained_variance": 0.0,
    }
    assert payload["counters"] == {
        "environment_transitions": 100,
        "rollout_transitions": 2,
        "learning_rows": 2,
    }
    densities = payload["densities"]
    assert isinstance(densities, list)
    assert [row["density_veh_per_lane_km"] for row in densities] == [10.0, 20.0]
    assert densities[0]["conditional_miss_estimate"] == pytest.approx(0.3)
    assert densities[0]["violation"] == pytest.approx(0.2)
    assert densities[0]["dual_before"] == pytest.approx(0.1)
    assert densities[0]["dual_after"] == pytest.approx(0.3)
    assert densities[0]["dual_update_count"] == 1
    assert densities[1]["sample_count"] == 0
    assert densities[1]["conditional_miss_estimate"] is None
    assert densities[1]["dual_before"] == densities[1]["dual_after"] == 0.2
    assert densities[1]["dual_update_count"] == 0
    json.dumps(payload, allow_nan=False)


def test_jsonl_logger_appends_and_recovers_monotonic_state(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "training.jsonl"
    first = _record()
    second = replace(
        first,
        iteration=1,
        environment_transitions=200,
    )
    logger = TrainingMetricsJSONL(path)

    assert logger.append(first) == path
    assert logger.append(second) == path

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["iteration"] for line in lines] == [0, 1]
    resumed = TrainingMetricsJSONL(path)
    with pytest.raises(TrainingMetricsError, match="iterations"):
        resumed.append(second)
    wrong_seed = replace(
        first,
        policy_seed=1002,
        iteration=2,
        environment_transitions=300,
    )
    with pytest.raises(TrainingMetricsError, match="different configuration"):
        resumed.append(wrong_seed)


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        (torch.tensor([]), torch.tensor([]), "nonempty"),
        (torch.tensor([0]), torch.tensor([0]), "floating"),
        (torch.tensor([0.0]), torch.tensor([0.0, 1.0]), "shape"),
        (torch.tensor([float("nan")]), torch.tensor([0.0]), "non-finite"),
        (torch.tensor([0.0], requires_grad=True), torch.tensor([0.0]), "detached"),
    ],
)
def test_explained_variance_rejects_invalid_vectors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(TrainingMetricsError, match=message):
        explained_variance(predictions=predictions, targets=targets)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"minibatch_size": 0}, "minibatch_size"),
        ({"reward_value_loss": -1.0}, "value-loss"),
        ({"approximate_kl": -0.1}, "KL"),
        ({"clip_fraction": 1.1}, "clip-fraction"),
        ({"ratio_mean": 0.0}, "ratio mean"),
        ({"entropy": float("nan")}, "finite"),
    ],
)
def test_ppo_update_metrics_fail_closed(
    override: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "minibatch_size": 2,
        "actor_loss": -1.0,
        "policy_loss": -1.0,
        "reward_value_loss": 1.0,
        "cost_value_loss": 1.0,
        "entropy": 0.5,
        "approximate_kl": 0.01,
        "clip_fraction": 0.1,
        "ratio_mean": 1.0,
    }
    values.update(override)

    with pytest.raises(PPOError, match=message):
        PPOUpdateMetrics(**values)  # type: ignore[arg-type]


def test_logger_rejects_partial_or_wrong_schema_existing_logs(tmp_path: Path) -> None:
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"schema":"unfinished"}', encoding="utf-8")
    with pytest.raises(TrainingMetricsError, match="partial"):
        TrainingMetricsJSONL(partial)

    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text('{"schema":"other","iteration":0}\n', encoding="utf-8")
    with pytest.raises(TrainingMetricsError, match="schema"):
        TrainingMetricsJSONL(wrong)
