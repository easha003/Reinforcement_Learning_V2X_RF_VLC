"""Phase 7 reliability training-signal selection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hybrid_v2x_rl.agents.advantages import generalized_advantage_estimate
from hybrid_v2x_rl.agents.cost_signal import (
    CONDITIONAL_MISS_PROBABILITY,
    ReliabilityCostBatch,
    ReliabilityCostSignalError,
    reliability_costs_from_config,
    select_reliability_costs,
)
from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.config.models import DensityMultiplierConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _select(
    sampled: torch.Tensor,
    conditional: torch.Tensor,
) -> ReliabilityCostBatch:
    return select_reliability_costs(
        sampled_miss_costs=sampled,
        conditional_miss_probabilities=conditional,
        cost_signal=CONDITIONAL_MISS_PROBABILITY,
    )


def test_headline_config_selects_conditional_risk_and_retains_sampled_misses() -> None:
    config = load_headline_config(PROJECT_ROOT)
    sampled = torch.tensor([0.0, 1.0, 0.0])
    conditional = torch.tensor([0.25, 0.10, 0.80])

    selected = reliability_costs_from_config(
        sampled_miss_costs=sampled,
        conditional_miss_probabilities=conditional,
        training=config.training,
    )

    assert selected.signal_name == "conditional_miss_probability"
    torch.testing.assert_close(selected.training_costs, conditional)
    torch.testing.assert_close(selected.sampled_miss_costs, sampled)
    assert not torch.equal(selected.training_costs, selected.sampled_miss_costs)
    assert not selected.training_costs.requires_grad


def test_selected_conditional_risk_drives_cost_gae_and_dual_estimate() -> None:
    selected = _select(
        sampled=torch.tensor([0.0, 0.0]),
        conditional=torch.tensor([0.4, 0.1]),
    )
    inactive_recursion = torch.zeros(2, dtype=torch.bool)
    active = torch.ones(2, dtype=torch.bool)

    estimate = generalized_advantage_estimate(
        signals=selected.training_costs,
        values=torch.zeros(2),
        next_values=torch.zeros(2),
        value_bootstrap_mask=inactive_recursion,
        gae_continuation_mask=inactive_recursion,
        active_mask=active,
        gamma=0.99,
        gae_lambda=0.95,
        time_dimension=0,
    )
    dual = PerDensityDualAscent(
        (
            DensityMultiplierConfig(
                density_veh_per_lane_km=10.0,
                initial_value=0.0,
                learning_rate=1.0,
                maximum=1.0,
            ),
        )
    )
    report = dual.update(
        densities_veh_per_lane_km=torch.tensor([10.0, 10.0]),
        costs=selected.training_costs,
        miss_budget=0.2,
    )

    torch.testing.assert_close(estimate.advantages, torch.tensor([0.4, 0.1]))
    assert report.for_density(10.0).estimated_cost == pytest.approx(0.25)
    assert dual.multiplier_for_density(10.0) == pytest.approx(0.05)


def test_selection_copies_sources_and_preserves_shape_dtype_and_device() -> None:
    sampled = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    conditional = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64)

    selected = _select(sampled, conditional)
    sampled.fill_(0.0)
    conditional.fill_(1.0)

    torch.testing.assert_close(
        selected.sampled_miss_costs,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        selected.training_costs,
        torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64),
    )
    assert selected.training_costs.shape == (2, 2)
    assert selected.training_costs.dtype == torch.float64
    assert selected.training_costs.device == conditional.device
    assert selected.training_costs.data_ptr() != selected.conditional_miss_probabilities.data_ptr()


@pytest.mark.parametrize(
    "cost_signal",
    ["sampled_miss_cost", "staged", None],
)
def test_unconfigured_sampled_or_staged_training_modes_are_rejected(
    cost_signal: object,
) -> None:
    with pytest.raises(ReliabilityCostSignalError, match="unsupported"):
        select_reliability_costs(
            sampled_miss_costs=torch.tensor([0.0]),
            conditional_miss_probabilities=torch.tensor([0.2]),
            cost_signal=cost_signal,
        )


@pytest.mark.parametrize(
    ("sampled", "conditional", "message"),
    [
        (torch.tensor([0.2]), torch.tensor([0.1]), "binary"),
        (torch.tensor([0.0]), torch.tensor([1.1]), "lie in"),
        (torch.tensor([0.0]), torch.tensor([float("nan")]), "non-finite"),
        (torch.tensor([0.0, 1.0]), torch.tensor([0.1]), "shape"),
        (torch.tensor([0.0], dtype=torch.float64), torch.tensor([0.1]), "dtype"),
        (torch.tensor(0.0), torch.tensor(0.1), "rollout dimension"),
    ],
)
def test_invalid_reliability_cost_tensors_fail_closed(
    sampled: torch.Tensor,
    conditional: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ReliabilityCostSignalError, match=message):
        _select(sampled, conditional)


def test_gradient_bearing_rollout_costs_are_rejected() -> None:
    with pytest.raises(ReliabilityCostSignalError, match="detached"):
        _select(
            torch.tensor([0.0]),
            torch.tensor([0.1], requires_grad=True),
        )
    with pytest.raises(ReliabilityCostSignalError, match="detached"):
        _select(
            torch.tensor([0.0], requires_grad=True),
            torch.tensor([0.1]),
        )
