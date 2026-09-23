"""Phase 7 termination and truncation bootstrap integration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
import torch

from hybrid_v2x_rl.agents.advantages import generalized_advantage_estimate
from hybrid_v2x_rl.agents.lifecycle_bootstrap import (
    LifecycleBootstrapError,
    PairedCriticValues,
    assemble_lifecycle_bootstrap,
)
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation, FrameStepOutput

PAIR_IDS = (
    "pair-a-continuing",
    "pair-b-internal",
    "pair-c-natural",
    "pair-d-trace-end",
)


def _observation(pair_ids: tuple[str, ...], *, frame_index: int = 11) -> FrameObservation:
    return FrameObservation(
        trace_id="synthetic-d10-train-000",
        frame_index=frame_index,
        time_s=float(frame_index) / 10.0,
        pair_ids=pair_ids,
        actor_observations=np.zeros((len(pair_ids), 37), dtype=np.float32),
        action_masks=np.ones((len(pair_ids), 9), dtype=np.bool_),
    )


def _step(
    *,
    next_pair_ids: tuple[str, ...] = ("pair-a-continuing", "pair-z-newborn"),
    final_observation: Mapping[str, object] | None = None,
) -> FrameStepOutput:
    final = (
        {"pair-b-internal": ("final-physical-row",)}
        if final_observation is None
        else dict(final_observation)
    )
    return FrameStepOutput(
        next_observation=_observation(next_pair_ids),
        transition_pair_ids=PAIR_IDS,
        rewards=np.ones(4, dtype=np.float32),
        terminated=np.array([False, False, True, False], dtype=np.bool_),
        truncated=np.array([False, True, False, True], dtype=np.bool_),
        bootstrap_valid=np.array([False, True, False, False], dtype=np.bool_),
        learn_mask=np.array([True, True, True, True], dtype=np.bool_),
        info={"final_observation": final},
    )


def _values(
    pair_ids: tuple[str, ...],
    reward: tuple[float, ...],
    cost: tuple[float, ...],
    *,
    dtype: torch.dtype = torch.float32,
) -> PairedCriticValues:
    return PairedCriticValues(
        pair_ids=pair_ids,
        reward=torch.tensor(reward, dtype=dtype),
        cost=torch.tensor(cost, dtype=dtype),
    )


def _assembled():
    return assemble_lifecycle_bootstrap(
        step_output=_step(),
        current_values=_values(PAIR_IDS, (1.0, 2.0, 3.0, 4.0), (0.1, 0.2, 0.3, 0.4)),
        next_population_values=_values(
            ("pair-a-continuing", "pair-z-newborn"),
            (11.0, 99.0),
            (1.1, 9.9),
        ),
        final_observation_values=_values(
            ("pair-b-internal",),
            (22.0,),
            (2.2,),
        ),
    )


def test_each_lifecycle_uses_its_authoritative_bootstrap_source() -> None:
    batch = _assembled()

    assert batch.transition_pair_ids == PAIR_IDS
    assert batch.reward_next_values == pytest.approx(torch.tensor([11.0, 22.0, 0.0, 0.0]))
    assert batch.cost_next_values == pytest.approx(torch.tensor([1.1, 2.2, 0.0, 0.0]))
    assert batch.terminated.tolist() == [False, False, True, False]
    assert batch.truncated.tolist() == [False, True, False, True]
    assert batch.bootstrap_valid.tolist() == [False, True, False, False]
    assert batch.value_bootstrap_mask.tolist() == [True, True, False, False]
    assert batch.gae_continuation_mask.tolist() == [True, False, False, False]
    assert batch.learn_mask.tolist() == [True, True, True, True]
    assert not batch.reward_values.requires_grad
    assert not batch.reward_next_values.requires_grad


def test_newborn_next_row_cannot_be_used_by_a_final_transition() -> None:
    batch = _assembled()

    assert batch.reward_next_values[1].item() == pytest.approx(22.0)
    assert 99.0 not in batch.reward_next_values.tolist()


def test_lifecycle_bootstraps_drive_gae_without_crossing_reset() -> None:
    batch = _assembled()
    signals = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 100.0, 200.0, 300.0],
        ]
    )
    values = torch.zeros_like(signals)
    next_values = torch.stack(
        (batch.reward_next_values, torch.zeros_like(batch.reward_next_values))
    )
    bootstrap = torch.stack(
        (batch.value_bootstrap_mask, torch.zeros_like(batch.value_bootstrap_mask))
    )
    continuation = torch.stack(
        (batch.gae_continuation_mask, torch.zeros_like(batch.gae_continuation_mask))
    )

    estimate = generalized_advantage_estimate(
        signals=signals,
        values=values,
        next_values=next_values,
        value_bootstrap_mask=bootstrap,
        gae_continuation_mask=continuation,
        active_mask=torch.ones_like(bootstrap),
        gamma=0.9,
        gae_lambda=1.0,
        time_dimension=0,
    )

    assert estimate.advantages[0] == pytest.approx(torch.tensor([12.7, 20.8, 1.0, 1.0]))
    assert estimate.advantages[1] == pytest.approx(torch.tensor([2.0, 100.0, 200.0, 300.0]))


def test_gradient_carrying_rollout_values_are_detached_and_copied() -> None:
    current_reward = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    current_cost = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    current = PairedCriticValues(PAIR_IDS, current_reward, current_cost)

    batch = assemble_lifecycle_bootstrap(
        step_output=_step(),
        current_values=current,
        next_population_values=_values(
            ("pair-a-continuing", "pair-z-newborn"),
            (11.0, 99.0),
            (1.1, 9.9),
        ),
        final_observation_values=_values(
            ("pair-b-internal",),
            (22.0,),
            (2.2,),
        ),
    )
    current_reward.detach()[0] = 999.0

    assert not batch.reward_values.requires_grad
    assert batch.reward_values[0].item() == pytest.approx(1.0)


def test_missing_or_extra_final_values_are_rejected() -> None:
    current = _values(PAIR_IDS, (1.0, 2.0, 3.0, 4.0), (0.1, 0.2, 0.3, 0.4))
    next_values = _values(
        ("pair-a-continuing", "pair-z-newborn"),
        (11.0, 99.0),
        (1.1, 9.9),
    )

    with pytest.raises(LifecycleBootstrapError, match="exactly"):
        assemble_lifecycle_bootstrap(
            step_output=_step(),
            current_values=current,
            next_population_values=next_values,
        )
    with pytest.raises(LifecycleBootstrapError, match="exactly"):
        assemble_lifecycle_bootstrap(
            step_output=_step(),
            current_values=current,
            next_population_values=next_values,
            final_observation_values=_values(
                ("pair-b-internal", "pair-c-natural"),
                (22.0, 33.0),
                (2.2, 3.3),
            ),
        )


def test_continuing_pair_must_exist_in_next_population() -> None:
    with pytest.raises(LifecycleBootstrapError, match="stable pair lifecycle") as error:
        assemble_lifecycle_bootstrap(
            step_output=_step(next_pair_ids=("pair-z-newborn",)),
            current_values=_values(
                PAIR_IDS,
                (1.0, 2.0, 3.0, 4.0),
                (0.1, 0.2, 0.3, 0.4),
            ),
            next_population_values=_values(
                ("pair-z-newborn",),
                (99.0,),
                (9.9,),
            ),
            final_observation_values=_values(
                ("pair-b-internal",),
                (22.0,),
                (2.2,),
            ),
        )

    assert error.value.context["missing_continuing_pair_ids"] == ("pair-a-continuing",)


def test_finalized_pair_cannot_reappear_after_reset() -> None:
    next_ids = ("pair-a-continuing", "pair-b-internal", "pair-z-newborn")

    with pytest.raises(LifecycleBootstrapError, match="stable pair lifecycle") as error:
        assemble_lifecycle_bootstrap(
            step_output=_step(next_pair_ids=next_ids),
            current_values=_values(
                PAIR_IDS,
                (1.0, 2.0, 3.0, 4.0),
                (0.1, 0.2, 0.3, 0.4),
            ),
            next_population_values=_values(
                next_ids,
                (11.0, 22.0, 99.0),
                (1.1, 2.2, 9.9),
            ),
            final_observation_values=_values(
                ("pair-b-internal",),
                (22.0,),
                (2.2,),
            ),
        )

    assert error.value.context["repeated_final_pair_ids"] == ("pair-b-internal",)


def test_value_batches_must_align_with_current_and_next_identities() -> None:
    with pytest.raises(LifecycleBootstrapError, match="transition pair IDs"):
        assemble_lifecycle_bootstrap(
            step_output=_step(),
            current_values=_values(
                PAIR_IDS[:-1],
                (1.0, 2.0, 3.0),
                (0.1, 0.2, 0.3),
            ),
            next_population_values=_values(
                ("pair-a-continuing", "pair-z-newborn"),
                (11.0, 99.0),
                (1.1, 9.9),
            ),
            final_observation_values=_values(
                ("pair-b-internal",),
                (22.0,),
                (2.2,),
            ),
        )

    with pytest.raises(LifecycleBootstrapError, match="next observation"):
        assemble_lifecycle_bootstrap(
            step_output=_step(),
            current_values=_values(
                PAIR_IDS,
                (1.0, 2.0, 3.0, 4.0),
                (0.1, 0.2, 0.3, 0.4),
            ),
            next_population_values=_values(
                ("pair-a-continuing",),
                (11.0,),
                (1.1,),
            ),
            final_observation_values=_values(
                ("pair-b-internal",),
                (22.0,),
                (2.2,),
            ),
        )


def test_critic_value_batches_reject_malformed_tensors_and_ids() -> None:
    with pytest.raises(LifecycleBootstrapError, match="canonical"):
        _values(("pair-b", "pair-a"), (1.0, 2.0), (0.1, 0.2))
    with pytest.raises(LifecycleBootstrapError, match="one entry"):
        _values(("pair-a",), (1.0, 2.0), (0.1, 0.2))
    with pytest.raises(LifecycleBootstrapError, match="same dtype"):
        PairedCriticValues(
            pair_ids=("pair-a",),
            reward=torch.tensor([1.0]),
            cost=torch.tensor([0.1], dtype=torch.float64),
        )
    with pytest.raises(LifecycleBootstrapError, match="finite"):
        _values(("pair-a",), (float("nan"),), (0.1,))


def test_empty_terminal_population_needs_no_final_value_batch() -> None:
    step = FrameStepOutput(
        next_observation=_observation((), frame_index=1),
        transition_pair_ids=(),
        rewards=np.empty(0, dtype=np.float32),
        terminated=np.empty(0, dtype=np.bool_),
        truncated=np.empty(0, dtype=np.bool_),
        bootstrap_valid=np.empty(0, dtype=np.bool_),
        learn_mask=np.empty(0, dtype=np.bool_),
    )
    empty = _values((), (), ())

    batch = assemble_lifecycle_bootstrap(
        step_output=step,
        current_values=empty,
        next_population_values=empty,
    )

    assert batch.reward_values.shape == (0,)
    assert batch.value_bootstrap_mask.shape == (0,)
