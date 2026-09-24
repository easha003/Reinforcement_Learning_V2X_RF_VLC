"""Atomic, complete Phase 7 training-checkpoint snapshots.

The checkpoint is deliberately a single PyTorch ``weights_only``-compatible
artifact.  It owns every mutable quantity needed to continue optimization,
while the caller remains responsible for choosing a unique checkpoint path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Final, cast

import numpy as np
import torch
from numpy.typing import NDArray

from hybrid_v2x_rl.agents.dual_ascent import PerDensityDualAscent
from hybrid_v2x_rl.agents.ppo import PPOUpdater
from hybrid_v2x_rl.config.hashing import canonical_data, config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizer

TRAINING_CHECKPOINT_SCHEMA: Final = "hybrid-rf-vlc-rl.training-checkpoint.v1"
TRAINING_COUNTERS_SCHEMA: Final = "hybrid-rf-vlc-rl.training-counters.v1"
TRAINING_RANDOM_STATE_SCHEMA: Final = "hybrid-rf-vlc-rl.training-random-state.v1"
DUAL_STATE_SCHEMA: Final = "hybrid-rf-vlc-rl.density-duals.v1"


class TrainingCheckpointError(HybridV2XError):
    """Checkpoint state or an attempted atomic write is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingCounters:
    """Cumulative positions needed to resume the training schedule exactly."""

    completed_iterations: int
    environment_transitions: int
    learning_transitions: int
    episodes_completed: int
    optimizer_steps: int

    def __post_init__(self) -> None:
        for name, value in (
            ("completed_iterations", self.completed_iterations),
            ("environment_transitions", self.environment_transitions),
            ("learning_transitions", self.learning_transitions),
            ("episodes_completed", self.episodes_completed),
            ("optimizer_steps", self.optimizer_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TrainingCheckpointError(f"{name} must be a nonnegative integer")
        if self.learning_transitions > self.environment_transitions:
            raise TrainingCheckpointError(
                "learning_transitions cannot exceed environment_transitions"
            )

    def as_dict(self) -> dict[str, object]:
        """Return a stable, primitive-only checkpoint representation."""

        return {
            "schema": TRAINING_COUNTERS_SCHEMA,
            "completed_iterations": self.completed_iterations,
            "environment_transitions": self.environment_transitions,
            "learning_transitions": self.learning_transitions,
            "episodes_completed": self.episodes_completed,
            "optimizer_steps": self.optimizer_steps,
        }


@dataclass(frozen=True, slots=True)
class TrainingCheckpointSummary:
    """Identity and size of one successfully published checkpoint."""

    path: Path
    config_hash: str
    policy_seed: int
    counters: TrainingCounters
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path.is_file() or self.path.is_symlink():
            raise TrainingCheckpointError(
                "published checkpoint must be a regular non-symlink file",
                artifact_path=self.path,
            )
        if len(self.config_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.config_hash
        ):
            raise TrainingCheckpointError("checkpoint summary config_hash is invalid")
        if self.size_bytes <= 0:
            raise TrainingCheckpointError("checkpoint summary size must be positive")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise TrainingCheckpointError("checkpoint summary SHA-256 is invalid")


def save_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    config: ProjectConfig,
    policy_seed: int,
    updater: PPOUpdater,
    dual_ascent: PerDensityDualAscent,
    normalizer: ObservationNormalizer,
    counters: TrainingCounters,
    numpy_generators: Mapping[str, np.random.Generator],
    torch_generators: Mapping[str, torch.Generator],
) -> TrainingCheckpointSummary:
    """Validate, snapshot, and atomically publish one immutable checkpoint.

    The destination must not already exist.  Every tensor is detached, cloned,
    and moved to CPU before serialization, making checkpoints portable across
    CPU, MPS, and CUDA training hosts.  Both generator mappings must contain
    every live explicit generator owned by the training driver.
    """

    destination = _checkpoint_path(path)
    if not isinstance(config, ProjectConfig):
        raise TrainingCheckpointError("checkpoint requires a resolved ProjectConfig")
    if not isinstance(updater, PPOUpdater):
        raise TrainingCheckpointError("checkpoint requires a PPOUpdater")
    if not isinstance(dual_ascent, PerDensityDualAscent):
        raise TrainingCheckpointError("checkpoint requires PerDensityDualAscent state")
    if not isinstance(normalizer, ObservationNormalizer):
        raise TrainingCheckpointError("checkpoint requires an ObservationNormalizer")
    if not isinstance(counters, TrainingCounters):
        raise TrainingCheckpointError("checkpoint requires validated TrainingCounters")

    validated_seed = _policy_seed(policy_seed, config=config)
    digest = config_hash(config)
    _validate_component_contracts(
        config=config,
        updater=updater,
        dual_ascent=dual_ascent,
        normalizer=normalizer,
        counters=counters,
    )
    payload = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "config_hash": digest,
        "configuration": canonical_data(config),
        "policy_seed": validated_seed,
        "counters": counters.as_dict(),
        "network_contract": {
            "actor_observation_width": updater.actor.observation_width,
            "critic_observation_width": updater.reward_critic.observation_width,
            "actor_hidden_units": list(updater.actor.hidden_units),
            "reward_critic_hidden_units": list(updater.reward_critic.hidden_units),
            "cost_critic_hidden_units": list(updater.cost_critic.hidden_units),
        },
        "models": {
            "actor": _snapshot_tree(updater.actor.state_dict(), name="actor model"),
            "reward_critic": _snapshot_tree(
                updater.reward_critic.state_dict(),
                name="reward critic model",
            ),
            "cost_critic": _snapshot_tree(
                updater.cost_critic.state_dict(),
                name="cost critic model",
            ),
        },
        "optimizers": {
            "actor": _snapshot_tree(
                updater.actor_optimizer.state_dict(),
                name="actor optimizer",
            ),
            "reward_critic": _snapshot_tree(
                updater.reward_critic_optimizer.state_dict(),
                name="reward critic optimizer",
            ),
            "cost_critic": _snapshot_tree(
                updater.cost_critic_optimizer.state_dict(),
                name="cost critic optimizer",
            ),
        },
        "duals": _dual_payload(
            config=config,
            dual_ascent=dual_ascent,
            completed_iterations=counters.completed_iterations,
        ),
        "normalization": dict(normalizer.state_dict()),
        "random_state": _random_state_payload(
            numpy_generators=numpy_generators,
            torch_generators=torch_generators,
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
    }
    return _atomic_save(
        destination,
        payload=payload,
        digest=digest,
        policy_seed=validated_seed,
        counters=counters,
    )


def _checkpoint_path(path: str | os.PathLike[str]) -> Path:
    try:
        destination = Path(path)
    except TypeError as exc:
        raise TrainingCheckpointError("checkpoint path must be path-like") from exc
    if not destination.name or destination.name in {".", ".."}:
        raise TrainingCheckpointError("checkpoint path must name a file")
    if destination.exists() or destination.is_symlink():
        raise TrainingCheckpointError(
            "checkpoint destination already exists; checkpoints are immutable",
            artifact_path=destination,
        )
    return destination


def _policy_seed(policy_seed: object, *, config: ProjectConfig) -> int:
    if isinstance(policy_seed, (bool, np.bool_)):
        raise TrainingCheckpointError("policy_seed must be an unsigned 64-bit integer")
    if not isinstance(policy_seed, Integral) or int(policy_seed) < 0 or int(policy_seed) >= 2**64:
        raise TrainingCheckpointError("policy_seed must be an unsigned 64-bit integer")
    validated = int(policy_seed)
    if validated not in config.training.policy_seeds:
        raise TrainingCheckpointError(
            "policy_seed is not declared by the resolved configuration",
            context={"policy_seed": validated},
        )
    return validated


def _validate_component_contracts(
    *,
    config: ProjectConfig,
    updater: PPOUpdater,
    dual_ascent: PerDensityDualAscent,
    normalizer: ObservationNormalizer,
    counters: TrainingCounters,
) -> None:
    training = config.training
    architecture = training.architecture
    mismatches: dict[str, object] = {}
    expected_actor_width = len(normalizer.columns)
    for name, actual, expected in (
        ("actor_observation_width", updater.actor.observation_width, expected_actor_width),
        ("actor_hidden_units", updater.actor.hidden_units, architecture.actor_hidden_units),
        (
            "reward_critic_hidden_units",
            updater.reward_critic.hidden_units,
            architecture.reward_critic_hidden_units,
        ),
        (
            "cost_critic_hidden_units",
            updater.cost_critic.hidden_units,
            architecture.cost_critic_hidden_units,
        ),
        ("learning_rate", updater.learning_rate, training.learning_rate),
        ("clip_ratio", updater.clip_ratio, training.clip_ratio),
        (
            "entropy_coefficient",
            updater.entropy_coefficient,
            training.entropy_coefficient,
        ),
    ):
        if actual != expected:
            mismatches[name] = {"checkpoint": actual, "config": expected}
    if updater.reward_critic.observation_width != updater.cost_critic.observation_width:
        mismatches["critic_observation_width"] = {
            "reward": updater.reward_critic.observation_width,
            "cost": updater.cost_critic.observation_width,
        }
    if counters.environment_transitions > training.total_transitions_per_seed:
        mismatches["environment_transitions"] = {
            "checkpoint": counters.environment_transitions,
            "configured_maximum": training.total_transitions_per_seed,
        }
    if mismatches:
        raise TrainingCheckpointError(
            "training components do not match the resolved configuration",
            config_hash=config_hash(config),
            context=mismatches,
        )
    _validate_optimizer_steps(updater=updater, expected=counters.optimizer_steps)

    # This also rejects an open decision frame, which cannot be resumed from a
    # checkpoint without serializing half of an environment transition.
    normalization_state = normalizer.snapshot()
    expected_normalizer = ObservationNormalizer.from_config(config)
    for normalization_name, normalization_actual, normalization_expected in (
        (
            "contract_version",
            normalization_state.contract_version,
            expected_normalizer.contract_version,
        ),
        ("columns", normalization_state.columns, expected_normalizer.columns),
        ("standardized", normalization_state.standardized, expected_normalizer.standardized),
        ("epsilon", normalization_state.epsilon, expected_normalizer.epsilon),
        ("clip_abs", normalization_state.clip_abs, expected_normalizer.clip_abs),
    ):
        if normalization_actual != normalization_expected:
            raise TrainingCheckpointError(
                "normalization state does not match the resolved configuration",
                config_hash=config_hash(config),
                context={
                    normalization_name: {
                        "checkpoint": normalization_actual,
                        "config": normalization_expected,
                    }
                },
            )


def _dual_payload(
    *,
    config: ProjectConfig,
    dual_ascent: PerDensityDualAscent,
    completed_iterations: int,
) -> dict[str, object]:
    snapshot = dual_ascent.snapshot()
    definitions = tuple(
        sorted(
            config.training.density_multipliers,
            key=lambda item: item.density_veh_per_lane_km,
        )
    )
    expected_densities = tuple(item.density_veh_per_lane_km for item in definitions)
    if snapshot.densities_veh_per_lane_km != expected_densities:
        raise TrainingCheckpointError(
            "dual densities do not match the resolved configuration",
            config_hash=config_hash(config),
            context={
                "checkpoint": snapshot.densities_veh_per_lane_km,
                "config": expected_densities,
            },
        )
    for definition, multiplier, updates in zip(
        definitions,
        snapshot.multipliers,
        snapshot.update_counts,
        strict=True,
    ):
        if not math.isfinite(multiplier) or not 0.0 <= multiplier <= definition.maximum:
            raise TrainingCheckpointError(
                "dual multiplier lies outside its configured projection interval",
                context={
                    "density_veh_per_lane_km": definition.density_veh_per_lane_km,
                    "multiplier": multiplier,
                    "maximum": definition.maximum,
                },
            )
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
            raise TrainingCheckpointError("dual update count must be a nonnegative integer")
        if updates > completed_iterations:
            raise TrainingCheckpointError(
                "dual update count cannot exceed completed training iterations",
                context={
                    "density_veh_per_lane_km": definition.density_veh_per_lane_km,
                    "update_count": updates,
                    "completed_iterations": completed_iterations,
                },
            )
    return {
        "schema": DUAL_STATE_SCHEMA,
        "densities_veh_per_lane_km": list(snapshot.densities_veh_per_lane_km),
        "multipliers": list(snapshot.multipliers),
        "update_counts": list(snapshot.update_counts),
    }


def _validate_optimizer_steps(*, updater: PPOUpdater, expected: int) -> None:
    for name, optimizer in (
        ("actor", updater.actor_optimizer),
        ("reward critic", updater.reward_critic_optimizer),
        ("cost critic", updater.cost_critic_optimizer),
    ):
        parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        observed: list[int] = []
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if not state:
                if expected != 0:
                    raise TrainingCheckpointError(
                        f"{name} optimizer state is missing at a nonzero optimizer step",
                        context={"expected_optimizer_steps": expected},
                    )
                continue
            observed.append(_optimizer_step(state.get("step"), name=name))
        if any(step != expected for step in observed):
            raise TrainingCheckpointError(
                f"{name} optimizer step does not match training counters",
                context={"observed": tuple(observed), "expected": expected},
            )


def _optimizer_step(value: object, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise TrainingCheckpointError(f"{name} optimizer step must be one finite scalar")
        numeric = float(value.detach().cpu().item())
    elif isinstance(value, bool):
        raise TrainingCheckpointError(f"{name} optimizer state has no numeric step")
    elif isinstance(value, Real):
        numeric = float(value)
    else:
        raise TrainingCheckpointError(f"{name} optimizer state has no numeric step")
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise TrainingCheckpointError(f"{name} optimizer step must be a nonnegative integer")
    return int(numeric)


def _random_state_payload(
    *,
    numpy_generators: Mapping[str, np.random.Generator],
    torch_generators: Mapping[str, torch.Generator],
) -> dict[str, object]:
    numpy_states = _numpy_generator_states(numpy_generators)
    torch_states = _torch_generator_states(torch_generators)
    python_state = random.getstate()
    numpy_global = cast(
        tuple[str, NDArray[np.uint32], int, int, float],
        np.random.get_state(legacy=True),
    )
    torch_global: dict[str, object] = {
        "cpu": torch.get_rng_state().detach().cpu().clone(),
        "cuda": [],
        "mps": None,
    }
    if torch.cuda.is_available():
        torch_global["cuda"] = [
            state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()
        ]
    if torch.backends.mps.is_available():
        torch_global["mps"] = torch.mps.get_rng_state().detach().cpu().clone()

    gauss_next = python_state[2]
    if gauss_next is not None and not math.isfinite(gauss_next):
        raise TrainingCheckpointError("Python global Gaussian RNG cache is non-finite")
    cached_gaussian = float(numpy_global[4])
    if not math.isfinite(cached_gaussian):
        raise TrainingCheckpointError("NumPy global Gaussian RNG cache is non-finite")
    return {
        "schema": TRAINING_RANDOM_STATE_SCHEMA,
        "python_global": {
            "version": python_state[0],
            "state": list(python_state[1]),
            "gauss_next": gauss_next,
        },
        "numpy_global": {
            "bit_generator": str(numpy_global[0]),
            "keys": torch.from_numpy(numpy_global[1].copy()),
            "position": int(numpy_global[2]),
            "has_gauss": int(numpy_global[3]),
            "cached_gaussian": cached_gaussian,
        },
        "torch_global": torch_global,
        "numpy_generators": numpy_states,
        "torch_generators": torch_states,
    }


def _numpy_generator_states(
    generators: Mapping[str, np.random.Generator],
) -> dict[str, object]:
    if not isinstance(generators, Mapping) or not generators:
        raise TrainingCheckpointError("numpy_generators must contain every live generator")
    states: dict[str, object] = {}
    for name in sorted(generators):
        _generator_name(name)
        generator = generators[name]
        if not isinstance(generator, np.random.Generator):
            raise TrainingCheckpointError(
                "numpy_generators values must be NumPy Generator instances",
                context={"name": name},
            )
        try:
            serialized = json.dumps(
                generator.bit_generator.state,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TrainingCheckpointError(
                "NumPy generator state is not primitive JSON data",
                context={"name": name},
            ) from exc
        states[name] = {
            "bit_generator": type(generator.bit_generator).__name__,
            "state": json.loads(serialized),
        }
    return states


def _torch_generator_states(
    generators: Mapping[str, torch.Generator],
) -> dict[str, object]:
    if not isinstance(generators, Mapping) or not generators:
        raise TrainingCheckpointError("torch_generators must contain every live generator")
    states: dict[str, object] = {}
    for name in sorted(generators):
        _generator_name(name)
        generator = generators[name]
        if not isinstance(generator, torch.Generator):
            raise TrainingCheckpointError(
                "torch_generators values must be torch.Generator instances",
                context={"name": name},
            )
        states[name] = {
            "device": str(generator.device),
            "state": generator.get_state().detach().cpu().clone(),
        }
    return states


def _generator_name(name: object) -> None:
    if not isinstance(name, str) or not name or name != name.strip():
        raise TrainingCheckpointError(
            "generator names must be non-empty strings without surrounding whitespace"
        )


def _snapshot_tree(value: object, *, name: str) -> object:
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise TrainingCheckpointError(f"{name} contains a non-finite tensor")
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        result: dict[object, object] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TrainingCheckpointError(f"{name} contains an unsupported mapping key")
            result[key] = _snapshot_tree(item, name=name)
        return result
    if isinstance(value, tuple):
        return tuple(_snapshot_tree(item, name=name) for item in value)
    if isinstance(value, list):
        return [_snapshot_tree(item, name=name) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TrainingCheckpointError(f"{name} contains a non-finite scalar")
        return value
    raise TrainingCheckpointError(f"{name} contains unsupported state type {type(value).__name__}")


def _atomic_save(
    destination: Path,
    *,
    payload: dict[str, object],
    digest: str,
    policy_seed: int,
    counters: TrainingCounters,
) -> TrainingCheckpointSummary:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise TrainingCheckpointError(
            "checkpoint parent must be a regular directory, not a symlink",
            artifact_path=parent,
        )
    if destination.exists() or destination.is_symlink():
        raise TrainingCheckpointError(
            "checkpoint destination already exists; checkpoints are immutable",
            artifact_path=destination,
        )

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_temporary_checkpoint(temporary, digest=digest)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise TrainingCheckpointError(
                "checkpoint destination was created concurrently",
                config_hash=digest,
                artifact_path=destination,
            ) from exc
        temporary.unlink()
        temporary = None
        _fsync_directory(parent)
    except TrainingCheckpointError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingCheckpointError(
            "failed to serialize training checkpoint",
            config_hash=digest,
            artifact_path=destination,
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return TrainingCheckpointSummary(
        path=destination,
        config_hash=digest,
        policy_seed=policy_seed,
        counters=counters,
        size_bytes=destination.stat().st_size,
        sha256=_sha256_file(destination),
    )


def _verify_temporary_checkpoint(path: Path, *, digest: str) -> None:
    loaded: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise TrainingCheckpointError("serialized checkpoint is not a mapping")
    if loaded.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
        raise TrainingCheckpointError("serialized checkpoint schema changed during write")
    if loaded.get("config_hash") != digest:
        raise TrainingCheckpointError("serialized checkpoint config hash changed during write")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DUAL_STATE_SCHEMA",
    "TRAINING_CHECKPOINT_SCHEMA",
    "TRAINING_COUNTERS_SCHEMA",
    "TRAINING_RANDOM_STATE_SCHEMA",
    "TrainingCheckpointError",
    "TrainingCheckpointSummary",
    "TrainingCounters",
    "save_training_checkpoint",
]
