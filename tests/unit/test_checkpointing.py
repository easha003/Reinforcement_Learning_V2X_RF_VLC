"""Phase 7 complete and atomic training-checkpoint snapshots."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch

from hybrid_v2x_rl.agents.checkpointing import (
    DUAL_STATE_SCHEMA,
    TRAINING_CHECKPOINT_SCHEMA,
    TRAINING_COUNTERS_SCHEMA,
    TRAINING_RANDOM_STATE_SCHEMA,
    TrainingCheckpointError,
    TrainingCounters,
    save_training_checkpoint,
)
from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.masked_categorical import ACTION_COUNT
from hybrid_v2x_rl.agents.ppo import PPOBatch, PPOUpdater
from hybrid_v2x_rl.config import config_hash, load_headline_config
from hybrid_v2x_rl.core.randomness import RandomStreams
from hybrid_v2x_rl.mean_field.normalization import NORMALIZATION_STATE_SCHEMA, ObservationNormalizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTOR_WIDTH = 37
CRITIC_WIDTH = 78


def _components():
    config = load_headline_config(PROJECT_ROOT)
    updater = PPOUpdater.from_config(
        actor_observation_width=ACTOR_WIDTH,
        critic_observation_width=CRITIC_WIDTH,
        training=config.training,
    )
    dual = PerDensityDualAscent.from_config(config.training)
    normalizer = ObservationNormalizer.from_config(config)
    return config, updater, dual, normalizer


def _populate_optimizer_state(updater: PPOUpdater, dual: PerDensityDualAscent) -> None:
    batch_size = 4
    actor_observations = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * ACTOR_WIDTH,
    ).reshape(batch_size, ACTOR_WIDTH)
    critic_observations = torch.linspace(
        1.0,
        -1.0,
        steps=batch_size * CRITIC_WIDTH,
    ).reshape(batch_size, CRITIC_WIDTH)
    action_masks = torch.ones((batch_size, ACTION_COUNT), dtype=torch.bool)
    with torch.no_grad():
        selected = updater.actor.select(
            actor_observations,
            action_masks,
            deterministic=True,
        )
    densities = torch.full((batch_size,), dual.densities_veh_per_lane_km[0])
    updater.update(
        PPOBatch(
            actor_observations=actor_observations,
            critic_observations=critic_observations,
            action_masks=action_masks,
            actions=selected.actions,
            old_log_probabilities=selected.log_probabilities,
            reward_advantages=torch.linspace(0.1, 0.4, steps=batch_size),
            cost_advantages=torch.linspace(0.4, 0.1, steps=batch_size),
            reward_value_targets=torch.linspace(-0.4, -0.1, steps=batch_size),
            cost_value_targets=torch.linspace(0.1, 0.4, steps=batch_size),
            cost_penalty_weights=dual.penalty_weights(densities),
        )
    )
    dual.update(
        densities_veh_per_lane_km=densities,
        costs=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        miss_budget=0.15,
    )


def _random_generators():
    numpy_generators = RandomStreams.from_root_seed(781, episode_id=4).as_dict()
    torch_generators = {
        "minibatch": torch.Generator().manual_seed(782),
        "policy": torch.Generator().manual_seed(783),
    }
    numpy_generators["policy"].random(5)
    torch.rand(5, generator=torch_generators["policy"])
    return numpy_generators, torch_generators


def _counters() -> TrainingCounters:
    return TrainingCounters(
        completed_iterations=1,
        environment_transitions=1_024,
        learning_transitions=1_000,
        episodes_completed=6,
        optimizer_steps=1,
    )


def _save(path: Path):
    config, updater, dual, normalizer = _components()
    _populate_optimizer_state(updater, dual)
    numpy_generators, torch_generators = _random_generators()
    summary = save_training_checkpoint(
        path,
        config=config,
        policy_seed=config.training.policy_seeds[0],
        updater=updater,
        dual_ascent=dual,
        normalizer=normalizer,
        counters=_counters(),
        numpy_generators=numpy_generators,
        torch_generators=torch_generators,
    )
    return summary, config, updater, dual, normalizer, numpy_generators, torch_generators


def test_checkpoint_contains_complete_portable_training_state(tmp_path: Path) -> None:
    path = tmp_path / "seed-0" / "iteration-000003.pt"
    summary, config, updater, dual, normalizer, numpy_generators, torch_generators = _save(path)

    assert summary.path == path
    assert summary.config_hash == config_hash(config)
    assert summary.policy_seed == config.training.policy_seeds[0]
    assert summary.counters == _counters()
    assert summary.size_bytes == path.stat().st_size
    assert summary.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {
        "schema",
        "config_hash",
        "configuration",
        "policy_seed",
        "counters",
        "network_contract",
        "models",
        "optimizers",
        "duals",
        "normalization",
        "random_state",
        "software",
    }
    assert payload["schema"] == TRAINING_CHECKPOINT_SCHEMA
    assert payload["config_hash"] == config_hash(config)
    assert payload["configuration"]["training"]["root_seed"] == config.training.root_seed
    assert payload["counters"] == _counters().as_dict()
    assert payload["counters"]["schema"] == TRAINING_COUNTERS_SCHEMA
    assert payload["network_contract"] == {
        "actor_observation_width": ACTOR_WIDTH,
        "critic_observation_width": CRITIC_WIDTH,
        "actor_hidden_units": list(config.training.architecture.actor_hidden_units),
        "reward_critic_hidden_units": list(config.training.architecture.reward_critic_hidden_units),
        "cost_critic_hidden_units": list(config.training.architecture.cost_critic_hidden_units),
    }

    assert set(payload["models"]) == {"actor", "reward_critic", "cost_critic"}
    assert set(payload["optimizers"]) == {"actor", "reward_critic", "cost_critic"}
    for state in payload["optimizers"].values():
        assert state["state"]
        assert all(
            tensor.device.type == "cpu"
            for slot in state["state"].values()
            for tensor in slot.values()
            if isinstance(tensor, torch.Tensor)
        )
    for name, state in payload["models"].items():
        source = {
            "actor": updater.actor,
            "reward_critic": updater.reward_critic,
            "cost_critic": updater.cost_critic,
        }[name]
        for key, tensor in state.items():
            assert tensor.device.type == "cpu"
            torch.testing.assert_close(tensor, source.state_dict()[key].detach().cpu())

    snapshot = dual.snapshot()
    assert payload["duals"] == {
        "schema": DUAL_STATE_SCHEMA,
        "densities_veh_per_lane_km": list(snapshot.densities_veh_per_lane_km),
        "multipliers": list(snapshot.multipliers),
        "update_counts": list(snapshot.update_counts),
    }
    assert payload["normalization"] == dict(normalizer.state_dict())
    assert payload["normalization"]["schema"] == NORMALIZATION_STATE_SCHEMA

    random_state = payload["random_state"]
    assert random_state["schema"] == TRAINING_RANDOM_STATE_SCHEMA
    assert set(random_state["numpy_generators"]) == set(numpy_generators)
    assert set(random_state["torch_generators"]) == set(torch_generators)
    assert random_state["numpy_generators"]["policy"]["state"] == (
        numpy_generators["policy"].bit_generator.state
    )
    torch.testing.assert_close(
        random_state["torch_generators"]["policy"]["state"],
        torch_generators["policy"].get_state(),
        rtol=0,
        atol=0,
    )
    assert random_state["python_global"]["state"]
    assert random_state["numpy_global"]["keys"].dtype == torch.uint32
    assert random_state["torch_global"]["cpu"].dtype == torch.uint8


def test_checkpoint_destination_is_immutable(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    summary, config, updater, dual, normalizer, numpy_generators, torch_generators = _save(path)
    before = path.read_bytes()

    with pytest.raises(TrainingCheckpointError, match="already exists"):
        save_training_checkpoint(
            path,
            config=config,
            policy_seed=config.training.policy_seeds[0],
            updater=updater,
            dual_ascent=dual,
            normalizer=normalizer,
            counters=summary.counters,
            numpy_generators=numpy_generators,
            torch_generators=torch_generators,
        )

    assert path.read_bytes() == before
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_checkpoint_rejects_component_or_seed_drift_before_writing(tmp_path: Path) -> None:
    config, updater, dual, normalizer = _components()
    numpy_generators, torch_generators = _random_generators()
    path = tmp_path / "invalid.pt"

    with pytest.raises(TrainingCheckpointError, match="not declared"):
        save_training_checkpoint(
            path,
            config=config,
            policy_seed=max(config.training.policy_seeds) + 1,
            updater=updater,
            dual_ascent=dual,
            normalizer=normalizer,
            counters=_counters(),
            numpy_generators=numpy_generators,
            torch_generators=torch_generators,
        )
    assert not path.exists()

    wrong_width = PPOUpdater.from_config(
        actor_observation_width=ACTOR_WIDTH - 1,
        critic_observation_width=CRITIC_WIDTH,
        training=config.training,
    )
    with pytest.raises(TrainingCheckpointError, match="do not match"):
        save_training_checkpoint(
            path,
            config=config,
            policy_seed=config.training.policy_seeds[0],
            updater=wrong_width,
            dual_ascent=dual,
            normalizer=normalizer,
            counters=_counters(),
            numpy_generators=numpy_generators,
            torch_generators=torch_generators,
        )
    assert not path.exists()


def test_checkpoint_rejects_missing_explicit_rng_state(tmp_path: Path) -> None:
    config, updater, dual, normalizer = _components()
    _populate_optimizer_state(updater, dual)
    numpy_generators, torch_generators = _random_generators()

    with pytest.raises(TrainingCheckpointError, match="numpy_generators"):
        save_training_checkpoint(
            tmp_path / "no-numpy.pt",
            config=config,
            policy_seed=config.training.policy_seeds[0],
            updater=updater,
            dual_ascent=dual,
            normalizer=normalizer,
            counters=_counters(),
            numpy_generators={},
            torch_generators=torch_generators,
        )
    with pytest.raises(TrainingCheckpointError, match="torch_generators"):
        save_training_checkpoint(
            tmp_path / "no-torch.pt",
            config=config,
            policy_seed=config.training.policy_seeds[0],
            updater=updater,
            dual_ascent=dual,
            normalizer=normalizer,
            counters=_counters(),
            numpy_generators=numpy_generators,
            torch_generators={},
        )


def test_checkpoint_rejects_stale_optimizer_counter(tmp_path: Path) -> None:
    config, updater, dual, normalizer = _components()
    _populate_optimizer_state(updater, dual)
    numpy_generators, torch_generators = _random_generators()
    counters = TrainingCounters(
        completed_iterations=1,
        environment_transitions=1_024,
        learning_transitions=1_000,
        episodes_completed=6,
        optimizer_steps=2,
    )

    with pytest.raises(TrainingCheckpointError, match="optimizer step"):
        save_training_checkpoint(
            tmp_path / "stale-counter.pt",
            config=config,
            policy_seed=config.training.policy_seeds[0],
            updater=updater,
            dual_ascent=dual,
            normalizer=normalizer,
            counters=counters,
            numpy_generators=numpy_generators,
            torch_generators=torch_generators,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"completed_iterations": -1},
        {"optimizer_steps": -1},
        {"learning_transitions": 11, "environment_transitions": 10},
    ],
)
def test_training_counters_reject_invalid_resume_positions(kwargs: dict[str, int]) -> None:
    values = {
        "completed_iterations": 1,
        "environment_transitions": 10,
        "learning_transitions": 9,
        "episodes_completed": 2,
        "optimizer_steps": 4,
    }
    values.update(kwargs)

    with pytest.raises(TrainingCheckpointError):
        TrainingCounters(**values)
