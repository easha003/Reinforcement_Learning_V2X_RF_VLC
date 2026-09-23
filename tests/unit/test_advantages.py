"""Phase 7 reward/cost generalized advantage estimation contract."""

from __future__ import annotations

import pytest
import torch

from hybrid_v2x_rl.agents.advantages import (
    AdvantageEstimationError,
    generalized_advantage_estimate,
    reward_cost_generalized_advantage_estimate,
)


def _all_true(values: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(values, dtype=torch.bool)


def test_gae_matches_hand_calculated_three_step_trajectory() -> None:
    signals = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    next_values = torch.tensor([1.0, 1.5, 0.0], dtype=torch.float64)
    bootstrap = torch.tensor([True, True, False])
    continuation = torch.tensor([True, True, False])

    estimate = generalized_advantage_estimate(
        signals=signals,
        values=values,
        next_values=next_values,
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=_all_true(signals),
        gamma=0.9,
        gae_lambda=0.8,
        time_dimension=0,
    )

    assert estimate.advantages == pytest.approx(torch.tensor([3.8696, 3.43, 1.5]))
    assert estimate.value_targets == pytest.approx(torch.tensor([4.3696, 4.43, 3.0]))
    assert estimate.advantages.dtype == torch.float64


def test_value_bootstrap_does_not_imply_recursive_continuation() -> None:
    signals = torch.tensor([1.0, 2.0])
    values = torch.zeros(2)
    next_values = torch.tensor([10.0, 100.0])

    estimate = generalized_advantage_estimate(
        signals=signals,
        values=values,
        next_values=next_values,
        value_bootstrap_mask=torch.tensor([True, False]),
        gae_continuation_mask=torch.tensor([False, False]),
        active_mask=_all_true(signals),
        gamma=0.9,
        gae_lambda=0.95,
        time_dimension=0,
    )

    assert estimate.advantages == pytest.approx(torch.tensor([10.0, 2.0]))
    assert estimate.value_targets == pytest.approx(estimate.advantages)


def test_reward_and_cost_advantages_use_independent_signals_and_critics() -> None:
    rewards = torch.tensor([[[-1.0], [-2.0], [-3.0]]])
    costs = torch.tensor([[[0.1], [0.2], [0.3]]])
    reward_values = torch.tensor([[[0.5], [0.4], [0.3]]])
    cost_values = torch.tensor([[[0.01], [0.02], [0.03]]])
    reward_next_values = torch.tensor([[[0.4], [0.3], [0.0]]])
    cost_next_values = torch.tensor([[[0.02], [0.03], [0.0]]])
    bootstrap = torch.tensor([[[True], [True], [False]]])
    continuation = torch.tensor([[[True], [True], [False]]])
    active = torch.ones_like(bootstrap)

    combined = reward_cost_generalized_advantage_estimate(
        rewards=rewards,
        costs=costs,
        reward_values=reward_values,
        reward_next_values=reward_next_values,
        cost_values=cost_values,
        cost_next_values=cost_next_values,
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=active,
        gamma=0.9,
        gae_lambda=0.8,
        time_dimension=1,
    )
    reward_only = generalized_advantage_estimate(
        signals=rewards,
        values=reward_values,
        next_values=reward_next_values,
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=active,
        gamma=0.9,
        gae_lambda=0.8,
        time_dimension=1,
    )
    cost_only = generalized_advantage_estimate(
        signals=costs,
        values=cost_values,
        next_values=cost_next_values,
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=active,
        gamma=0.9,
        gae_lambda=0.8,
        time_dimension=1,
    )

    assert torch.equal(combined.reward.advantages, reward_only.advantages)
    assert torch.equal(combined.reward.value_targets, reward_only.value_targets)
    assert torch.equal(combined.cost.advantages, cost_only.advantages)
    assert torch.equal(combined.cost.value_targets, cost_only.value_targets)
    assert not torch.equal(combined.reward.advantages, combined.cost.advantages)


def test_combined_estimates_require_matching_reward_and_cost_dtypes() -> None:
    values = torch.zeros(2)
    mask = torch.tensor([True, False])

    with pytest.raises(AdvantageEstimationError, match="same dtype"):
        reward_cost_generalized_advantage_estimate(
            rewards=values,
            costs=values.to(torch.float64),
            reward_values=values,
            reward_next_values=values,
            cost_values=values.to(torch.float64),
            cost_next_values=values.to(torch.float64),
            value_bootstrap_mask=mask,
            gae_continuation_mask=mask,
            active_mask=torch.ones(2, dtype=torch.bool),
            gamma=0.9,
            gae_lambda=0.8,
            time_dimension=0,
        )


def test_padded_rows_are_zero_and_cannot_influence_active_rows() -> None:
    signals = torch.tensor(
        [
            [[1.0, 4.0], [2.0, 9_999.0], [3.0, -9_999.0]],
            [[0.5, 0.0], [0.25, 0.0], [0.125, 0.0]],
        ]
    )
    values = torch.zeros_like(signals)
    active = torch.tensor(
        [
            [[True, True], [True, False], [True, False]],
            [[True, False], [True, False], [True, False]],
        ]
    )
    bootstrap = active.clone()
    continuation = torch.tensor(
        [
            [[True, False], [True, False], [False, False]],
            [[True, False], [True, False], [False, False]],
        ]
    )

    estimate = generalized_advantage_estimate(
        signals=signals,
        values=values,
        next_values=torch.zeros_like(signals),
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=active,
        gamma=1.0,
        gae_lambda=1.0,
        time_dimension=1,
    )

    assert estimate.advantages[0, :, 0] == pytest.approx(torch.tensor([6.0, 5.0, 3.0]))
    assert estimate.advantages[1, :, 0] == pytest.approx(torch.tensor([0.875, 0.375, 0.125]))
    assert estimate.advantages[0, 0, 1].item() == pytest.approx(4.0)
    assert estimate.advantages[:, 1:, 1].count_nonzero().item() == 0
    assert estimate.value_targets[~active].count_nonzero().item() == 0


def test_estimates_are_detached_from_rollout_value_graphs() -> None:
    signals = torch.tensor([1.0, 2.0], requires_grad=True)
    values = torch.tensor([0.5, 0.25], requires_grad=True)
    next_values = torch.tensor([0.25, 0.0], requires_grad=True)
    bootstrap = torch.tensor([True, False])
    continuation = torch.tensor([True, False])

    estimate = generalized_advantage_estimate(
        signals=signals,
        values=values,
        next_values=next_values,
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=_all_true(signals),
        gamma=0.9,
        gae_lambda=0.95,
        time_dimension=0,
    )

    assert not estimate.advantages.requires_grad
    assert not estimate.value_targets.requires_grad


def test_zero_length_time_dimension_returns_shape_preserving_empty_estimates() -> None:
    empty = torch.empty((2, 0, 3))
    empty_mask = torch.empty((2, 0, 3), dtype=torch.bool)

    estimate = generalized_advantage_estimate(
        signals=empty,
        values=empty.clone(),
        next_values=empty.clone(),
        value_bootstrap_mask=empty_mask,
        gae_continuation_mask=empty_mask.clone(),
        active_mask=empty_mask.clone(),
        gamma=0.99,
        gae_lambda=0.95,
        time_dimension=1,
    )

    assert estimate.advantages.shape == (2, 0, 3)
    assert estimate.value_targets.shape == (2, 0, 3)


@pytest.mark.parametrize("name", ["value_bootstrap_mask", "gae_continuation_mask"])
def test_estimation_masks_cannot_enable_padded_rows(name: str) -> None:
    values = torch.zeros(2)
    masks = {
        "value_bootstrap_mask": torch.tensor([True, False]),
        "gae_continuation_mask": torch.tensor([False, False]),
    }
    masks[name] = torch.tensor([True, True])

    with pytest.raises(AdvantageEstimationError, match="padded"):
        generalized_advantage_estimate(
            signals=values,
            values=values,
            next_values=values,
            value_bootstrap_mask=masks["value_bootstrap_mask"],
            gae_continuation_mask=masks["gae_continuation_mask"],
            active_mask=torch.tensor([True, False]),
            gamma=0.99,
            gae_lambda=0.95,
            time_dimension=0,
        )


def test_recursive_continuation_requires_bootstrap_and_an_active_next_row() -> None:
    values = torch.zeros(2)

    with pytest.raises(AdvantageEstimationError, match="requires"):
        generalized_advantage_estimate(
            signals=values,
            values=values,
            next_values=values,
            value_bootstrap_mask=torch.tensor([False, False]),
            gae_continuation_mask=torch.tensor([True, False]),
            active_mask=torch.tensor([True, True]),
            gamma=0.99,
            gae_lambda=0.95,
            time_dimension=0,
        )

    with pytest.raises(AdvantageEstimationError, match="inactive next-time"):
        generalized_advantage_estimate(
            signals=values,
            values=values,
            next_values=values,
            value_bootstrap_mask=torch.tensor([True, False]),
            gae_continuation_mask=torch.tensor([True, False]),
            active_mask=torch.tensor([True, False]),
            gamma=0.99,
            gae_lambda=0.95,
            time_dimension=0,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"values": torch.zeros(3)}, "signal shape"),
        ({"next_values": torch.zeros(2, dtype=torch.float64)}, "signal dtype"),
        ({"signals": torch.zeros(2, dtype=torch.long)}, "floating"),
        ({"signals": torch.tensor([0.0, float("nan")])}, "non-finite"),
        ({"active_mask": torch.ones(2)}, "torch.bool"),
        ({"time_dimension": 1}, "outside"),
        ({"time_dimension": True}, "integer"),
        ({"gamma": 0.0}, "gamma"),
        ({"gamma": 1.01}, "gamma"),
        ({"gae_lambda": float("nan")}, "gae_lambda"),
    ],
)
def test_invalid_rollout_inputs_fail_closed(
    override: dict[str, object],
    message: str,
) -> None:
    values = torch.zeros(2)
    kwargs: dict[str, object] = {
        "signals": values,
        "values": values,
        "next_values": values,
        "value_bootstrap_mask": torch.tensor([True, False]),
        "gae_continuation_mask": torch.tensor([True, False]),
        "active_mask": torch.tensor([True, True]),
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "time_dimension": 0,
    }
    kwargs.update(override)

    with pytest.raises(AdvantageEstimationError, match=message):
        generalized_advantage_estimate(**kwargs)  # type: ignore[arg-type]
