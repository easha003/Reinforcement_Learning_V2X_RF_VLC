"""Phase 7 projected per-density reliability multipliers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hybrid_v2x_rl.agents.dual_ascent import (
    DensityDualSnapshot,
    DualAscentError,
    PerDensityDualAscent,
    projected_dual_ascent,
)
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.config.models import DensityMultiplierConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _definition(
    density: float,
    *,
    initial: float = 0.5,
    learning_rate: float = 1.0,
    maximum: float = 2.0,
) -> DensityMultiplierConfig:
    return DensityMultiplierConfig(
        density_veh_per_lane_km=density,
        initial_value=initial,
        learning_rate=learning_rate,
        maximum=maximum,
    )


def _controller() -> PerDensityDualAscent:
    return PerDensityDualAscent(
        (
            _definition(30.0, initial=0.3),
            _definition(10.0, initial=0.1),
            _definition(20.0, initial=0.2),
        )
    )


def test_projected_update_has_correct_violation_slack_floor_and_cap_directions() -> None:
    assert projected_dual_ascent(
        current=0.25,
        learning_rate=0.5,
        estimated_cost=0.4,
        miss_budget=0.1,
        maximum=1.0,
    ) == pytest.approx(0.4)
    assert projected_dual_ascent(
        current=0.25,
        learning_rate=0.5,
        estimated_cost=0.0,
        miss_budget=0.1,
        maximum=1.0,
    ) == pytest.approx(0.2)
    assert (
        projected_dual_ascent(
            current=0.01,
            learning_rate=1.0,
            estimated_cost=0.0,
            miss_budget=0.1,
            maximum=1.0,
        )
        == 0.0
    )
    assert (
        projected_dual_ascent(
            current=0.9,
            learning_rate=1.0,
            estimated_cost=1.0,
            miss_budget=0.1,
            maximum=1.0,
        )
        == 1.0
    )


def test_controller_updates_each_density_from_only_its_own_mean_cost() -> None:
    controller = _controller()

    report = controller.update(
        densities_veh_per_lane_km=torch.tensor([10.0, 10.0, 20.0]),
        costs=torch.tensor([0.0, 1.0, 0.1]),
        miss_budget=0.2,
    )

    assert controller.snapshot().densities_veh_per_lane_km == (10.0, 20.0, 30.0)
    assert controller.multiplier_for_density(10.0) == pytest.approx(0.4)
    assert controller.multiplier_for_density(20.0) == pytest.approx(0.1)
    assert controller.multiplier_for_density(30.0) == pytest.approx(0.3)
    assert controller.snapshot().update_counts == (1, 1, 0)
    assert report.for_density(10.0).sample_count == 2
    assert report.for_density(10.0).estimated_cost == pytest.approx(0.5)
    assert report.for_density(10.0).violation == pytest.approx(0.3)
    assert report.for_density(20.0).estimated_cost == pytest.approx(0.1)
    assert report.for_density(30.0).estimated_cost is None
    assert report.for_density(30.0).violation is None


def test_zero_multiplier_stays_projected_at_zero_under_slack() -> None:
    controller = PerDensityDualAscent((_definition(10.0, initial=0.0),))

    report = controller.update(
        densities_veh_per_lane_km=torch.tensor([10.0, 10.0]),
        costs=torch.tensor([0.0, 0.0]),
        miss_budget=0.1,
    )

    assert report.for_density(10.0).multiplier_before == 0.0
    assert report.for_density(10.0).multiplier_after == 0.0
    assert controller.multiplier_for_density(10.0) == 0.0


def test_restore_replaces_complete_dual_state_and_fails_atomically() -> None:
    source = _controller()
    source.update(
        densities_veh_per_lane_km=torch.tensor([10.0, 20.0]),
        costs=torch.tensor([0.8, 0.0]),
        miss_budget=0.2,
    )
    restored = _controller()

    restored.restore(source.snapshot())

    assert restored.snapshot() == source.snapshot()
    before = restored.snapshot()
    with pytest.raises(DualAscentError, match="exceeds"):
        restored.restore(
            DensityDualSnapshot(
                densities_veh_per_lane_km=before.densities_veh_per_lane_km,
                multipliers=(3.0, *before.multipliers[1:]),
                update_counts=before.update_counts,
            )
        )
    assert restored.snapshot() == before


def test_penalty_weights_follow_row_density_not_batch_order() -> None:
    controller = _controller()
    densities = torch.tensor([30.0, 10.0, 20.0, 10.0], dtype=torch.float64)

    weights = controller.penalty_weights(densities)

    torch.testing.assert_close(
        weights,
        torch.tensor([0.3, 0.1, 0.2, 0.1], dtype=torch.float64),
    )
    assert weights.dtype == densities.dtype
    assert weights.device == densities.device
    assert not weights.requires_grad


def test_config_constructor_covers_all_headline_densities() -> None:
    config = load_headline_config(PROJECT_ROOT)

    controller = PerDensityDualAscent.from_config(config.training)

    assert controller.densities_veh_per_lane_km == (10.0, 20.0, 30.0)
    assert controller.snapshot().multipliers == (0.0, 0.0, 0.0)
    torch.testing.assert_close(
        controller.penalty_weights(torch.tensor([20.0, 30.0, 10.0])),
        torch.zeros(3),
    )


def test_unknown_density_fails_exactly_and_does_not_partially_update_state() -> None:
    controller = _controller()
    before = controller.snapshot()

    with pytest.raises(DualAscentError, match="exactly match configured"):
        controller.update(
            densities_veh_per_lane_km=torch.tensor([10.0, 20.0001]),
            costs=torch.tensor([1.0, 1.0]),
            miss_budget=0.1,
        )

    assert controller.snapshot() == before


@pytest.mark.parametrize(
    ("densities", "costs", "message"),
    [
        (torch.tensor([[10.0]]), torch.tensor([0.1]), "one-dimensional"),
        (torch.tensor([10]), torch.tensor([0.1]), "floating-point"),
        (torch.tensor([10.0]), torch.tensor([0.1, 0.2]), "same shape"),
        (torch.tensor([10.0]), torch.tensor([float("nan")]), "non-finite"),
        (torch.tensor([10.0]), torch.tensor([1.1]), "closed interval"),
    ],
)
def test_update_rejects_invalid_rollout_vectors_without_mutation(
    densities: torch.Tensor,
    costs: torch.Tensor,
    message: str,
) -> None:
    controller = _controller()
    before = controller.snapshot()

    with pytest.raises(DualAscentError, match=message):
        controller.update(
            densities_veh_per_lane_km=densities,
            costs=costs,
            miss_budget=0.1,
        )

    assert controller.snapshot() == before


def test_gradient_bearing_costs_and_labels_are_rejected() -> None:
    controller = _controller()

    with pytest.raises(DualAscentError, match="density labels must be detached"):
        controller.penalty_weights(torch.tensor([10.0], requires_grad=True))
    with pytest.raises(DualAscentError, match="dual costs must be detached"):
        controller.update(
            densities_veh_per_lane_km=torch.tensor([10.0]),
            costs=torch.tensor([0.1], requires_grad=True),
            miss_budget=0.1,
        )


def test_initial_multiplier_cannot_exceed_its_projection_cap() -> None:
    definition = _definition(10.0, initial=2.0, maximum=1.0)

    with pytest.raises(DualAscentError, match="initial dual multiplier exceeds"):
        PerDensityDualAscent((definition,))


@pytest.mark.parametrize("budget", [0.0, 1.0, float("nan")])
def test_update_rejects_invalid_curriculum_budget_without_mutation(budget: float) -> None:
    controller = _controller()
    before = controller.snapshot()

    with pytest.raises(DualAscentError, match="miss_budget"):
        controller.update(
            densities_veh_per_lane_km=torch.tensor([10.0]),
            costs=torch.tensor([0.1]),
            miss_budget=budget,
        )

    assert controller.snapshot() == before
