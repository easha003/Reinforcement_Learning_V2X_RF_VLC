"""Phase 5 multi-agent frame API contract and intentional Gymnasium deviation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.mean_field.environment_api import (
    CONTRACT_ACTION_COUNT,
    CONTRACT_ACTOR_WIDTH,
    ActionArray,
    FrameAPIError,
    FrameAPISchema,
    FrameObservation,
    FrameStepOutput,
    MultiAgentFrameEnv,
    ResetReturn,
    StepReturn,
)

TRACE_ID = "synthetic-d20-train-000"


def _schema() -> FrameAPISchema:
    from pathlib import Path

    return FrameAPISchema.from_config(
        load_headline_config(Path(__file__).resolve().parents[2])
    )


def _observation(
    pair_ids: tuple[str, ...] = ("pair-a", "pair-b"),
    *,
    frame_index: int = 4,
) -> FrameObservation:
    population = len(pair_ids)
    return FrameObservation(
        trace_id=TRACE_ID,
        frame_index=frame_index,
        time_s=0.1 * frame_index,
        pair_ids=pair_ids,
        actor_observations=np.zeros(
            (population, CONTRACT_ACTOR_WIDTH),
            dtype=np.float32,
        ),
        action_masks=np.ones(
            (population, CONTRACT_ACTION_COUNT),
            dtype=np.bool_,
        ),
    )


def test_headline_schema_is_derived_as_37_actor_columns_and_nine_actions() -> None:
    schema = _schema()
    observation = _observation()

    assert schema.contract_version == "1.0.0"
    assert schema.actor_width == 37
    assert schema.action_count == 9
    schema.validate_observation(observation)


def test_selected_actions_are_mask_checked_and_frozen_before_step() -> None:
    schema = _schema()
    masks = np.ones((2, CONTRACT_ACTION_COUNT), dtype=np.bool_)
    masks[1, 0] = False
    observation = replace(_observation(), action_masks=masks)
    actions = np.array([0, 8], dtype=np.int64)

    validated = schema.validate_actions(observation, actions)

    actions[0] = 7
    assert validated.tolist() == [0, 8]
    assert not validated.flags.writeable


@pytest.mark.parametrize(
    "actions, message",
    [
        pytest.param(
            np.array([0, 1], dtype=np.int32),
            "int64",
            id="wrong-dtype",
        ),
        pytest.param(
            np.array([[0, 1]], dtype=np.int64),
            "align one-to-one",
            id="wrong-shape",
        ),
        pytest.param(
            np.array([-1, 1], dtype=np.int64),
            "outside",
            id="negative-index",
        ),
        pytest.param(
            np.array([0, CONTRACT_ACTION_COUNT], dtype=np.int64),
            "outside",
            id="too-large",
        ),
    ],
)
def test_selected_action_batch_fails_closed_on_invalid_policy_output(
    actions: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(FrameAPIError, match=message):
        _schema().validate_actions(_observation(), actions)  # type: ignore[arg-type]


def test_selected_action_must_be_enabled_by_its_own_row_mask() -> None:
    masks = np.ones((2, CONTRACT_ACTION_COUNT), dtype=np.bool_)
    masks[1, 8] = False
    observation = replace(_observation(), action_masks=masks)

    with pytest.raises(FrameAPIError, match="enabled"):
        _schema().validate_actions(
            observation,
            np.array([8, 8], dtype=np.int64),
        )


def test_frame_observation_preserves_canonical_ids_and_freezes_arrays() -> None:
    actor = np.zeros((2, CONTRACT_ACTOR_WIDTH), dtype=np.float32)
    masks = np.ones((2, CONTRACT_ACTION_COUNT), dtype=np.bool_)
    observation = FrameObservation(
        trace_id=TRACE_ID,
        frame_index=0,
        time_s=0.0,
        pair_ids=("pair-a", "pair-b"),
        actor_observations=actor,
        action_masks=masks,
    )

    actor[0, 0] = 1.0
    masks[0, 0] = False
    assert observation.population_size == 2
    assert observation.actor_observations[0, 0] == 0.0
    assert bool(observation.action_masks[0, 0])
    assert not observation.actor_observations.flags.writeable
    assert not observation.action_masks.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        observation.actor_observations[0, 0] = 2.0


def test_empty_frame_retains_the_contract_widths() -> None:
    observation = _observation(())

    assert observation.population_size == 0
    assert observation.actor_observations.shape == (0, 37)
    assert observation.action_masks.shape == (0, 9)
    _schema().validate_observation(observation)


@pytest.mark.parametrize(
    "observation",
    [
        pytest.param(
            lambda: replace(_observation(), pair_ids=("pair-b", "pair-a")),
            id="noncanonical-ids",
        ),
        pytest.param(
            lambda: replace(_observation(), pair_ids=("pair-a", "pair-a")),
            id="duplicate-ids",
        ),
        pytest.param(
            lambda: replace(
                _observation(),
                actor_observations=np.zeros((2, 37), dtype=np.float64),
            ),
            id="actor-dtype",
        ),
        pytest.param(
            lambda: replace(
                _observation(),
                action_masks=np.ones((2, 9), dtype=np.int64),
            ),
            id="mask-dtype",
        ),
        pytest.param(
            lambda: replace(
                _observation(),
                actor_observations=np.zeros((1, 37), dtype=np.float32),
            ),
            id="population-shape",
        ),
        pytest.param(
            lambda: replace(
                _observation(),
                actor_observations=np.full((2, 37), np.nan, dtype=np.float32),
            ),
            id="nonfinite-actor",
        ),
        pytest.param(
            lambda: replace(
                _observation(),
                action_masks=np.zeros((2, 9), dtype=np.bool_),
            ),
            id="no-legal-action",
        ),
    ],
)
def test_frame_observation_fails_closed_on_contract_drift(
    observation: object,
) -> None:
    with pytest.raises(FrameAPIError):
        observation()  # type: ignore[operator]


def test_schema_rejects_wrong_actor_or_action_width() -> None:
    with pytest.raises(FrameAPIError, match="actor width"):
        FrameAPISchema("1.0.0", actor_width=36, action_count=9)
    with pytest.raises(FrameAPIError, match="action count"):
        FrameAPISchema("1.0.0", actor_width=37, action_count=8)


def test_step_outputs_align_to_current_ids_while_next_population_may_change() -> None:
    next_observation = _observation(("pair-b", "pair-c", "pair-d"), frame_index=5)
    rewards = np.array([0.5, -0.25], dtype=np.float32)
    terminated = np.array([True, False], dtype=np.bool_)
    truncated = np.array([False, True], dtype=np.bool_)
    bootstrap_valid = np.array([False, True], dtype=np.bool_)
    learn_mask = np.array([True, False], dtype=np.bool_)
    output = FrameStepOutput(
        next_observation=next_observation,
        transition_pair_ids=("pair-a", "pair-b"),
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        bootstrap_valid=bootstrap_valid,
        learn_mask=learn_mask,
        info={
            "diagnostic": "kept-out-of-observation",
            "final_observation": {"pair-b": ("next-trace-row",)},
        },
    )

    step = output.as_tuple()
    assert len(step) == 5
    assert step[0].pair_ids == ("pair-b", "pair-c", "pair-d")
    assert step[1].shape == step[2].shape == step[3].shape == (2,)
    assert step[4]["transition_pair_ids"] == ("pair-a", "pair-b")
    assert not output.rewards.flags.writeable
    assert not output.terminated.flags.writeable
    assert not output.truncated.flags.writeable
    assert not output.bootstrap_valid.flags.writeable
    assert not output.learn_mask.flags.writeable
    assert output.info["final_observation"]["pair-b"] == ("next-trace-row",)
    assert np.array_equal(
        output.info["value_bootstrap_mask"],
        np.array([False, True], dtype=np.bool_),
    )
    assert np.array_equal(
        output.info["gae_continuation_mask"],
        np.array([False, False], dtype=np.bool_),
    )

    rewards[0] = 9.0
    assert output.rewards[0] == pytest.approx(0.5)
    with pytest.raises(TypeError):
        output.info["new"] = "forbidden"  # type: ignore[index]


def test_standard_cost_and_probability_info_is_validated_and_frozen() -> None:
    sampled = np.array([0.0, 1.0], dtype=np.float32)
    conditional = np.array([0.25, 1.0], dtype=np.float32)

    output = FrameStepOutput(
        next_observation=_observation(frame_index=5),
        transition_pair_ids=("pair-a", "pair-b"),
        rewards=np.zeros(2, dtype=np.float32),
        terminated=np.zeros(2, dtype=np.bool_),
        truncated=np.zeros(2, dtype=np.bool_),
        bootstrap_valid=np.zeros(2, dtype=np.bool_),
        learn_mask=np.ones(2, dtype=np.bool_),
        info={
            "sampled_miss_cost": sampled,
            "conditional_miss_probability": conditional,
        },
    )

    sampled[0] = 1.0
    conditional[0] = 0.75
    assert output.info["sampled_miss_cost"].tolist() == [0.0, 1.0]
    assert output.info["conditional_miss_probability"].tolist() == [0.25, 1.0]
    assert not output.info["sampled_miss_cost"].flags.writeable
    assert not output.info["conditional_miss_probability"].flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"rewards": np.zeros(1, dtype=np.float32)},
            id="reward-shape",
        ),
        pytest.param(
            {"rewards": np.zeros(2, dtype=np.float64)},
            id="reward-dtype",
        ),
        pytest.param(
            {"rewards": np.full(2, np.inf, dtype=np.float32)},
            id="nonfinite-reward",
        ),
        pytest.param(
            {
                "terminated": np.array([True, False], dtype=np.bool_),
                "truncated": np.array([True, False], dtype=np.bool_),
            },
            id="both-done",
        ),
        pytest.param(
            {"bootstrap_valid": np.ones(2, dtype=np.bool_)},
            id="bootstrap-without-truncation",
        ),
        pytest.param(
            {
                "truncated": np.array([False, True], dtype=np.bool_),
                "bootstrap_valid": np.array([False, True], dtype=np.bool_),
            },
            id="missing-final-observation",
        ),
        pytest.param(
            {"info": {"final_observation": {"pair-a": object()}}},
            id="unexpected-final-observation",
        ),
        pytest.param(
            {"info": {"learn_mask": np.zeros(2, dtype=np.bool_)}},
            id="info-learn-mask-drift",
        ),
        pytest.param(
            {"info": {"transition_pair_ids": ("pair-x", "pair-y")}},
            id="info-id-drift",
        ),
        pytest.param(
            {"info": {"sampled_miss_cost": np.array([0.0, 0.5], dtype=np.float32)}},
            id="nonbinary-sampled-cost",
        ),
        pytest.param(
            {
                "info": {
                    "conditional_miss_probability": np.array(
                        [0.0, np.nan], dtype=np.float32
                    )
                }
            },
            id="nonfinite-probability",
        ),
        pytest.param(
            {
                "info": {
                    "conditional_miss_probability": np.array(
                        [0.0, 1.01], dtype=np.float32
                    )
                }
            },
            id="out-of-range-probability",
        ),
    ],
)
def test_step_output_fails_closed_on_alignment_drift(
    kwargs: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "next_observation": _observation(frame_index=5),
        "transition_pair_ids": ("pair-a", "pair-b"),
        "rewards": np.zeros(2, dtype=np.float32),
        "terminated": np.zeros(2, dtype=np.bool_),
        "truncated": np.zeros(2, dtype=np.bool_),
        "bootstrap_valid": np.zeros(2, dtype=np.bool_),
        "learn_mask": np.ones(2, dtype=np.bool_),
    }
    values.update(kwargs)
    with pytest.raises(FrameAPIError):
        FrameStepOutput(**values)  # type: ignore[arg-type]


class _StructuralFrameEnvironment:
    def __init__(self) -> None:
        self._schema = _schema()
        self._current = _observation(("pair-a",), frame_index=0)

    @property
    def api_schema(self) -> FrameAPISchema:
        return self._schema

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> ResetReturn:
        return self._current, {"seed": seed, "options": options}

    def step(self, actions: ActionArray) -> StepReturn:
        assert actions.shape == (1,)
        next_observation = _observation((), frame_index=1)
        return FrameStepOutput(
            next_observation=next_observation,
            transition_pair_ids=self._current.pair_ids,
            rewards=np.zeros(1, dtype=np.float32),
            terminated=np.ones(1, dtype=np.bool_),
            truncated=np.zeros(1, dtype=np.bool_),
            bootstrap_valid=np.zeros(1, dtype=np.bool_),
            learn_mask=np.ones(1, dtype=np.bool_),
        ).as_tuple()

    def close(self) -> None:
        return None


def test_protocol_preserves_gymnasium_shape_without_claiming_scalar_semantics() -> None:
    environment = _StructuralFrameEnvironment()

    assert isinstance(environment, MultiAgentFrameEnv)
    reset_observation, reset_info = environment.reset(seed=17)
    result = environment.step(np.array([0], dtype=np.int64))

    assert reset_observation.population_size == 1
    assert reset_info["seed"] == 17
    assert len(result) == 5
    assert result[0].population_size == 0
    assert result[1].shape == result[2].shape == result[3].shape == (1,)
