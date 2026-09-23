"""Phase 7 masked categorical actor and sampling contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hybrid_v2x_rl.agents.masked_categorical import (
    ACTION_COUNT,
    MaskedCategorical,
    MaskedCategoricalError,
    SharedCategoricalActor,
)
from hybrid_v2x_rl.config.models import NetworkArchitectureConfig
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.action_masks import ActionMask


def _all_actions_mask() -> torch.Tensor:
    return torch.ones(ACTION_COUNT, dtype=torch.bool)


def test_stochastic_sampling_never_selects_masked_actions() -> None:
    logits = torch.tensor([[50.0, 2.0, 1.0, 0.0, -1.0, 40.0, 3.0, 2.0, 1.0]])
    mask = torch.tensor([False, True, True, False, False, False, True, False, False])
    distribution = MaskedCategorical(logits.expand(4_000, -1), mask)

    selected = distribution.select(generator=torch.Generator().manual_seed(41))

    assert set(selected.actions.tolist()) <= {1, 2, 6}
    assert distribution.probabilities[:, ~mask].count_nonzero().item() == 0
    assert torch.isfinite(selected.log_probabilities).all()
    assert torch.isfinite(selected.entropy).all()


def test_deterministic_selection_ignores_masked_maximum_and_breaks_ties_by_index() -> None:
    logits = torch.tensor(
        [
            [100.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 9.0, 9.0, 8.0],
        ]
    )
    masks = torch.tensor(
        [
            [False, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True],
        ]
    )

    selected = MaskedCategorical(logits, masks).select(deterministic=True)

    assert selected.actions.tolist() == [1, 6]


def test_repository_action_mask_preserves_persistent_action_indices() -> None:
    mask = ActionMask.from_availability(
        rf_hardware_available=False,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=0,
    )
    logits = torch.arange(ACTION_COUNT, dtype=torch.float32).unsqueeze(0)

    selected = MaskedCategorical(logits, mask).select(deterministic=True)

    assert selected.actions.item() == int(PolicyAction.VLC)
    assert selected.log_probabilities.item() == pytest.approx(0.0)
    assert selected.entropy.item() == pytest.approx(0.0)


def test_probabilities_and_policy_statistics_renormalize_allowed_actions() -> None:
    logits = torch.tensor([[0.0, 1.0, 20.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    mask = torch.tensor([True, True, False, False, False, False, False, False, False])
    distribution = MaskedCategorical(logits, mask)

    evaluated = distribution.evaluate_actions(torch.tensor([1]))
    expected_probabilities = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)
    expected_entropy = -(expected_probabilities * expected_probabilities.log()).sum()

    assert distribution.probabilities[0, :2] == pytest.approx(expected_probabilities)
    assert distribution.probabilities[0, 2:].count_nonzero().item() == 0
    assert evaluated.log_probabilities.item() == pytest.approx(
        expected_probabilities[1].log().item()
    )
    assert evaluated.entropy.item() == pytest.approx(expected_entropy.item())


def test_explicit_generator_makes_sampling_independent_of_global_rng() -> None:
    logits = torch.zeros((128, ACTION_COUNT))
    first_generator = torch.Generator().manual_seed(912)
    second_generator = torch.Generator().manual_seed(912)

    torch.manual_seed(1)
    first = MaskedCategorical(logits, _all_actions_mask()).select(generator=first_generator)
    torch.manual_seed(999)
    second = MaskedCategorical(logits, _all_actions_mask()).select(generator=second_generator)

    assert torch.equal(first.actions, second.actions)


def test_masked_logits_receive_no_policy_gradient() -> None:
    logits = torch.zeros((1, ACTION_COUNT), requires_grad=True)
    mask = torch.tensor([True, True, False, False, False, False, False, False, False])
    evaluated = MaskedCategorical(logits, mask).evaluate_actions(torch.tensor([1]))

    (-evaluated.log_probabilities.mean()).backward()

    assert logits.grad is not None
    assert logits.grad[0, 2:].count_nonzero().item() == 0
    assert logits.grad[0, :2].abs().sum().item() > 0


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        (torch.tensor([0]), "masked"),
        (torch.tensor([ACTION_COUNT]), "outside"),
        (torch.tensor([1.0]), "torch.long"),
        (torch.tensor([[1]]), "one index"),
    ],
)
def test_action_evaluation_rejects_invalid_indices(
    actions: torch.Tensor,
    message: str,
) -> None:
    logits = torch.zeros((1, ACTION_COUNT))
    mask = torch.tensor([False, True, True, True, True, True, True, True, True])

    with pytest.raises(MaskedCategoricalError, match=message):
        MaskedCategorical(logits, mask).evaluate_actions(actions)


@pytest.mark.parametrize(
    ("logits", "mask", "message"),
    [
        (torch.zeros(ACTION_COUNT), _all_actions_mask(), "one column"),
        (
            torch.zeros((1, ACTION_COUNT - 1)),
            torch.ones(ACTION_COUNT - 1, dtype=torch.bool),
            "one column",
        ),
        (torch.zeros((1, ACTION_COUNT), dtype=torch.long), _all_actions_mask(), "floating"),
        (torch.full((1, ACTION_COUNT), float("nan")), _all_actions_mask(), "non-finite"),
        (torch.zeros((1, ACTION_COUNT)), torch.ones(ACTION_COUNT), "torch.bool"),
        (
            torch.zeros((1, ACTION_COUNT)),
            torch.zeros(ACTION_COUNT, dtype=torch.bool),
            "at least one",
        ),
        (
            torch.zeros((2, ACTION_COUNT)),
            torch.ones((1, ACTION_COUNT), dtype=torch.bool),
            "match logits",
        ),
    ],
)
def test_malformed_logits_and_masks_are_rejected(
    logits: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(MaskedCategoricalError, match=message):
        MaskedCategorical(logits, mask)


def test_empty_population_is_a_valid_batch() -> None:
    distribution = MaskedCategorical(
        torch.empty((0, ACTION_COUNT)),
        _all_actions_mask(),
    )

    selected = distribution.select(generator=torch.Generator().manual_seed(7))
    evaluated = distribution.evaluate_actions(torch.empty(0, dtype=torch.long))

    assert distribution.probabilities.shape == (0, ACTION_COUNT)
    assert selected.actions.shape == (0,)
    assert selected.log_probabilities.shape == (0,)
    assert evaluated.entropy.shape == (0,)


def test_actor_uses_configured_two_by_64_tanh_network_and_nine_outputs() -> None:
    architecture = NetworkArchitectureConfig()
    actor = SharedCategoricalActor.from_config(
        observation_width=37,
        architecture=architecture,
    )
    modules = list(actor.network)

    assert [module.in_features for module in modules if isinstance(module, nn.Linear)] == [
        37,
        64,
        64,
    ]
    assert [module.out_features for module in modules if isinstance(module, nn.Linear)] == [
        64,
        64,
        ACTION_COUNT,
    ]
    assert sum(isinstance(module, nn.Tanh) for module in modules) == 2
    assert actor(torch.zeros((3, 37))).shape == (3, ACTION_COUNT)


def test_actor_deterministic_evaluation_is_stable_and_masked() -> None:
    actor = SharedCategoricalActor(observation_width=2, hidden_units=(4, 4))
    for parameter in actor.parameters():
        nn.init.constant_(parameter, 0.0)
    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([False, False, True, True, True, True, True, True, True])

    first = actor.select(observations, mask, deterministic=True)
    second = actor.select(observations, mask, deterministic=True)
    reevaluated = actor.evaluate_actions(observations, mask, first.actions)

    assert first.actions.tolist() == [2, 2]
    assert torch.equal(first.actions, second.actions)
    assert torch.equal(first.log_probabilities, reevaluated.log_probabilities)
    assert torch.equal(first.entropy, reevaluated.entropy)


def test_actor_rejects_noncausal_input_shapes_and_values() -> None:
    actor = SharedCategoricalActor(observation_width=2)

    with pytest.raises(MaskedCategoricalError, match="wrong shape"):
        actor(torch.zeros((2, 3)))
    with pytest.raises(MaskedCategoricalError, match="floating"):
        actor(torch.zeros((2, 2), dtype=torch.long))
    with pytest.raises(MaskedCategoricalError, match="non-finite"):
        actor(torch.tensor([[0.0, float("inf")]]))
