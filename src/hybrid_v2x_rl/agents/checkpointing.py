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
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray

from hybrid_v2x_rl.agents.dual_ascent import DensityDualSnapshot, PerDensityDualAscent
from hybrid_v2x_rl.agents.ppo import PPOUpdater
from hybrid_v2x_rl.config.hashing import canonical_data, config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.normalization import (
    ObservationNormalizationError,
    ObservationNormalizer,
)

TRAINING_CHECKPOINT_SCHEMA: Final = "hybrid-rf-vlc-rl.training-checkpoint.v1"
TRAINING_COUNTERS_SCHEMA: Final = "hybrid-rf-vlc-rl.training-counters.v1"
TRAINING_RANDOM_STATE_SCHEMA: Final = "hybrid-rf-vlc-rl.training-random-state.v1"
DUAL_STATE_SCHEMA: Final = "hybrid-rf-vlc-rl.density-duals.v1"
_CHECKPOINT_FIELDS: Final = {
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

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrainingCounters:
        """Parse a counters payload without accepting missing or extra fields."""

        values = _exact_mapping(
            payload,
            name="training counters",
            expected={
                "schema",
                "completed_iterations",
                "environment_transitions",
                "learning_transitions",
                "episodes_completed",
                "optimizer_steps",
            },
        )
        if values["schema"] != TRAINING_COUNTERS_SCHEMA:
            raise TrainingCheckpointError("training counters schema is unsupported")
        return cls(
            completed_iterations=_nonnegative_integer(
                values["completed_iterations"],
                name="completed_iterations",
            ),
            environment_transitions=_nonnegative_integer(
                values["environment_transitions"],
                name="environment_transitions",
            ),
            learning_transitions=_nonnegative_integer(
                values["learning_transitions"],
                name="learning_transitions",
            ),
            episodes_completed=_nonnegative_integer(
                values["episodes_completed"],
                name="episodes_completed",
            ),
            optimizer_steps=_nonnegative_integer(
                values["optimizer_steps"],
                name="optimizer_steps",
            ),
        )


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


@dataclass(frozen=True, slots=True)
class RestoredTrainingState:
    """Fully reconstructed mutable training owners plus checkpoint identity."""

    updater: PPOUpdater
    dual_ascent: PerDensityDualAscent
    normalizer: ObservationNormalizer
    counters: TrainingCounters
    policy_seed: int
    numpy_generators: Mapping[str, np.random.Generator]
    torch_generators: Mapping[str, torch.Generator]
    config_hash: str
    sha256: str
    software: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _GlobalRandomState:
    python: tuple[int, tuple[int, ...], float | None]
    numpy: tuple[str, NDArray[np.uint32], int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    torch_mps: torch.Tensor | None


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


def restore_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    config: ProjectConfig,
    device: str | torch.device = "cpu",
    expected_sha256: str | None = None,
    restore_global_rng: bool = True,
) -> RestoredTrainingState:
    """Reconstruct a complete learner from a trusted weights-only checkpoint.

    Validation happens before any saved global RNG state is installed.  Model
    construction consumes PyTorch's global CPU generator, so its entry state is
    restored on every failure and whenever ``restore_global_rng`` is false.
    """

    source = _existing_checkpoint_path(path)
    if not isinstance(config, ProjectConfig):
        raise TrainingCheckpointError("checkpoint restoration requires a ProjectConfig")
    if type(restore_global_rng) is not bool:
        raise TrainingCheckpointError("restore_global_rng must be boolean")
    target_device = _target_device(device)
    expected_artifact_hash = _optional_sha256(expected_sha256)
    payload, artifact_hash = _load_checkpoint(source)
    if expected_artifact_hash is not None and artifact_hash != expected_artifact_hash:
        raise TrainingCheckpointError(
            "checkpoint artifact SHA-256 does not match the expected digest",
            artifact_path=source,
            context={"actual": artifact_hash, "expected": expected_artifact_hash},
        )

    top = _exact_mapping(payload, name="training checkpoint", expected=_CHECKPOINT_FIELDS)
    if top["schema"] != TRAINING_CHECKPOINT_SCHEMA:
        raise TrainingCheckpointError(
            "training checkpoint schema is unsupported",
            artifact_path=source,
            context={"actual": top["schema"], "expected": TRAINING_CHECKPOINT_SCHEMA},
        )
    expected_config_hash = config_hash(config)
    if top["config_hash"] != expected_config_hash:
        raise TrainingCheckpointError(
            "checkpoint configuration hash does not match the resolved configuration",
            config_hash=expected_config_hash,
            artifact_path=source,
            context={"checkpoint": top["config_hash"]},
        )
    if top["configuration"] != canonical_data(config):
        raise TrainingCheckpointError(
            "checkpoint canonical configuration does not match exactly",
            config_hash=expected_config_hash,
            artifact_path=source,
        )

    policy_seed = _policy_seed(top["policy_seed"], config=config)
    counters = TrainingCounters.from_dict(
        _string_mapping(top["counters"], name="training counters")
    )
    if counters.environment_transitions > config.training.total_transitions_per_seed:
        raise TrainingCheckpointError(
            "checkpoint environment transitions exceed the configured budget",
            config_hash=expected_config_hash,
            artifact_path=source,
        )
    try:
        normalizer = ObservationNormalizer.from_state_dict(
            config,
            _string_mapping(top["normalization"], name="normalization state"),
        )
    except ObservationNormalizationError as exc:
        raise TrainingCheckpointError(
            "checkpoint normalization state is invalid",
            config_hash=expected_config_hash,
            artifact_path=source,
        ) from exc
    actor_width, critic_width = _network_widths(
        top["network_contract"],
        config=config,
        normalizer=normalizer,
    )
    dual_ascent = _restore_duals(
        top["duals"],
        config=config,
        completed_iterations=counters.completed_iterations,
    )
    global_random, numpy_generators, torch_generators = _restore_random_state(top["random_state"])
    software = _software_versions(top["software"])
    models = _exact_mapping(
        _string_mapping(top["models"], name="model states"),
        name="model states",
        expected={"actor", "reward_critic", "cost_critic"},
    )
    optimizers = _exact_mapping(
        _string_mapping(top["optimizers"], name="optimizer states"),
        name="optimizer states",
        expected={"actor", "reward_critic", "cost_critic"},
    )

    entry_global_random = _capture_global_random_state()
    try:
        updater = PPOUpdater.from_config(
            actor_observation_width=actor_width,
            critic_observation_width=critic_width,
            training=config.training,
        )
        updater.actor.to(target_device)
        updater.reward_critic.to(target_device)
        updater.cost_critic.to(target_device)
        _load_model_state(updater.actor, models["actor"], name="actor")
        _load_model_state(
            updater.reward_critic,
            models["reward_critic"],
            name="reward critic",
        )
        _load_model_state(updater.cost_critic, models["cost_critic"], name="cost critic")
        _load_optimizer_state(
            updater.actor_optimizer,
            optimizers["actor"],
            name="actor optimizer",
        )
        _load_optimizer_state(
            updater.reward_critic_optimizer,
            optimizers["reward_critic"],
            name="reward critic optimizer",
        )
        _load_optimizer_state(
            updater.cost_critic_optimizer,
            optimizers["cost_critic"],
            name="cost critic optimizer",
        )
        _validate_component_contracts(
            config=config,
            updater=updater,
            dual_ascent=dual_ascent,
            normalizer=normalizer,
            counters=counters,
        )
        _install_global_random_state(global_random if restore_global_rng else entry_global_random)
    except TrainingCheckpointError:
        _install_global_random_state(entry_global_random)
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _install_global_random_state(entry_global_random)
        raise TrainingCheckpointError(
            "failed to restore training checkpoint state",
            config_hash=expected_config_hash,
            artifact_path=source,
        ) from exc

    return RestoredTrainingState(
        updater=updater,
        dual_ascent=dual_ascent,
        normalizer=normalizer,
        counters=counters,
        policy_seed=policy_seed,
        numpy_generators=MappingProxyType(numpy_generators),
        torch_generators=MappingProxyType(torch_generators),
        config_hash=expected_config_hash,
        sha256=artifact_hash,
        software=MappingProxyType(software),
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


def _existing_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    try:
        source = Path(path)
    except TypeError as exc:
        raise TrainingCheckpointError("checkpoint path must be path-like") from exc
    if source.is_symlink() or not source.is_file():
        raise TrainingCheckpointError(
            "checkpoint source must be a regular non-symlink file",
            artifact_path=source,
        )
    return source


def _target_device(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise TrainingCheckpointError("checkpoint target device is invalid") from exc
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise TrainingCheckpointError(
            "checkpoint target device must be cpu, cuda, or mps",
            context={"device": str(resolved)},
        )
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainingCheckpointError("CUDA checkpoint target is unavailable")
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        if index < 0 or index >= torch.cuda.device_count():
            raise TrainingCheckpointError("CUDA checkpoint target index is unavailable")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise TrainingCheckpointError("MPS checkpoint target is unavailable")
    return resolved


def _optional_sha256(value: object) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingCheckpointError("expected_sha256 must be a lowercase SHA-256 digest")
    return value


def _load_checkpoint(path: Path) -> tuple[dict[str, object], str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
            stream.seek(0)
            loaded: object = torch.load(stream, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingCheckpointError(
            "checkpoint cannot be read in weights-only mode",
            artifact_path=path,
        ) from exc
    return _string_mapping(loaded, name="training checkpoint"), digest.hexdigest()


def _string_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingCheckpointError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TrainingCheckpointError(f"{name} keys must be strings")
        result[key] = item
    return result


def _exact_mapping(
    value: object,
    *,
    name: str,
    expected: set[str],
) -> dict[str, object]:
    result = _string_mapping(value, name=name)
    actual = set(result)
    if actual != expected:
        raise TrainingCheckpointError(
            f"{name} fields do not match the schema",
            context={
                "missing": tuple(sorted(expected - actual)),
                "unexpected": tuple(sorted(actual - expected)),
            },
        )
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingCheckpointError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    converted = _nonnegative_integer(value, name=name)
    if converted == 0:
        raise TrainingCheckpointError(f"{name} must be positive")
    return converted


def _network_widths(
    value: object,
    *,
    config: ProjectConfig,
    normalizer: ObservationNormalizer,
) -> tuple[int, int]:
    contract = _exact_mapping(
        value,
        name="network contract",
        expected={
            "actor_observation_width",
            "critic_observation_width",
            "actor_hidden_units",
            "reward_critic_hidden_units",
            "cost_critic_hidden_units",
        },
    )
    actor_width = _positive_integer(
        contract["actor_observation_width"],
        name="actor_observation_width",
    )
    critic_width = _positive_integer(
        contract["critic_observation_width"],
        name="critic_observation_width",
    )
    architecture = config.training.architecture
    expected = {
        "actor_hidden_units": architecture.actor_hidden_units,
        "reward_critic_hidden_units": architecture.reward_critic_hidden_units,
        "cost_critic_hidden_units": architecture.cost_critic_hidden_units,
    }
    mismatches: dict[str, object] = {}
    for name, configured in expected.items():
        restored = _positive_integer_array(contract[name], name=name)
        if restored != configured:
            mismatches[name] = {"checkpoint": restored, "config": configured}
    if actor_width != len(normalizer.columns):
        mismatches["actor_observation_width"] = {
            "checkpoint": actor_width,
            "normalization": len(normalizer.columns),
        }
    if mismatches:
        raise TrainingCheckpointError(
            "checkpoint network contract does not match the resolved configuration",
            context=mismatches,
        )
    return actor_width, critic_width


def _positive_integer_array(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TrainingCheckpointError(f"{name} must be a nonempty integer array")
    return tuple(_positive_integer(item, name=name) for item in value)


def _restore_duals(
    value: object,
    *,
    config: ProjectConfig,
    completed_iterations: int,
) -> PerDensityDualAscent:
    payload = _exact_mapping(
        value,
        name="dual state",
        expected={
            "schema",
            "densities_veh_per_lane_km",
            "multipliers",
            "update_counts",
        },
    )
    if payload["schema"] != DUAL_STATE_SCHEMA:
        raise TrainingCheckpointError("dual checkpoint schema is unsupported")
    densities = _finite_real_array(
        payload["densities_veh_per_lane_km"],
        name="dual densities",
    )
    multipliers = _finite_real_array(payload["multipliers"], name="dual multipliers")
    update_counts = _nonnegative_integer_array(
        payload["update_counts"],
        name="dual update counts",
    )
    if any(value <= 0.0 for value in densities):
        raise TrainingCheckpointError("dual densities must be positive")
    if any(value < 0.0 for value in multipliers):
        raise TrainingCheckpointError("dual multipliers must be nonnegative")
    if any(value > completed_iterations for value in update_counts):
        raise TrainingCheckpointError(
            "dual update count cannot exceed completed training iterations"
        )
    controller = PerDensityDualAscent.from_config(config.training)
    try:
        controller.restore(
            DensityDualSnapshot(
                densities_veh_per_lane_km=densities,
                multipliers=multipliers,
                update_counts=update_counts,
            )
        )
    except HybridV2XError as exc:
        raise TrainingCheckpointError("checkpoint dual state is invalid") from exc
    return controller


def _finite_real_array(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TrainingCheckpointError(f"{name} must be a nonempty numeric array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TrainingCheckpointError(f"{name} must contain real numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise TrainingCheckpointError(f"{name} must contain finite numbers")
        result.append(converted)
    return tuple(result)


def _nonnegative_integer_array(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TrainingCheckpointError(f"{name} must be a nonempty integer array")
    return tuple(_nonnegative_integer(item, name=name) for item in value)


def _restore_random_state(
    value: object,
) -> tuple[
    _GlobalRandomState,
    dict[str, np.random.Generator],
    dict[str, torch.Generator],
]:
    payload = _exact_mapping(
        value,
        name="random state",
        expected={
            "schema",
            "python_global",
            "numpy_global",
            "torch_global",
            "numpy_generators",
            "torch_generators",
        },
    )
    if payload["schema"] != TRAINING_RANDOM_STATE_SCHEMA:
        raise TrainingCheckpointError("random-state checkpoint schema is unsupported")
    python_state = _python_random_state(payload["python_global"])
    numpy_state = _numpy_global_state(payload["numpy_global"])
    torch_cpu, torch_cuda, torch_mps = _torch_global_state(payload["torch_global"])
    numpy_generators = _restored_numpy_generators(payload["numpy_generators"])
    torch_generators = _restored_torch_generators(payload["torch_generators"])
    return (
        _GlobalRandomState(
            python=python_state,
            numpy=numpy_state,
            torch_cpu=torch_cpu,
            torch_cuda=torch_cuda,
            torch_mps=torch_mps,
        ),
        numpy_generators,
        torch_generators,
    )


def _python_random_state(value: object) -> tuple[int, tuple[int, ...], float | None]:
    payload = _exact_mapping(
        value,
        name="Python global RNG state",
        expected={"version", "state", "gauss_next"},
    )
    version = _nonnegative_integer(payload["version"], name="Python RNG version")
    raw_state = payload["state"]
    if not isinstance(raw_state, list | tuple) or not raw_state:
        raise TrainingCheckpointError("Python RNG state must be a nonempty integer array")
    state = tuple(_nonnegative_integer(item, name="Python RNG state") for item in raw_state)
    raw_gauss = payload["gauss_next"]
    if raw_gauss is None:
        gauss = None
    elif isinstance(raw_gauss, bool) or not isinstance(raw_gauss, Real):
        raise TrainingCheckpointError("Python Gaussian RNG cache must be numeric or null")
    else:
        gauss = float(raw_gauss)
        if not math.isfinite(gauss):
            raise TrainingCheckpointError("Python Gaussian RNG cache must be finite")
    restored = (version, state, gauss)
    try:
        random.Random().setstate(restored)
    except (TypeError, ValueError) as exc:
        raise TrainingCheckpointError("Python global RNG state is invalid") from exc
    return restored


def _numpy_global_state(
    value: object,
) -> tuple[str, NDArray[np.uint32], int, int, float]:
    payload = _exact_mapping(
        value,
        name="NumPy global RNG state",
        expected={"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"},
    )
    if payload["bit_generator"] != "MT19937":
        raise TrainingCheckpointError("NumPy global RNG must use MT19937")
    keys_tensor = _rng_tensor(payload["keys"], name="NumPy global RNG keys", dtype=torch.uint32)
    keys = keys_tensor.numpy().copy()
    position = _nonnegative_integer(payload["position"], name="NumPy RNG position")
    has_gauss = _nonnegative_integer(payload["has_gauss"], name="NumPy has_gauss")
    if has_gauss not in {0, 1}:
        raise TrainingCheckpointError("NumPy has_gauss must be zero or one")
    raw_cache = payload["cached_gaussian"]
    if isinstance(raw_cache, bool) or not isinstance(raw_cache, Real):
        raise TrainingCheckpointError("NumPy Gaussian RNG cache must be numeric")
    cached = float(raw_cache)
    if not math.isfinite(cached):
        raise TrainingCheckpointError("NumPy Gaussian RNG cache must be finite")
    restored = ("MT19937", keys, position, has_gauss, cached)
    try:
        np.random.RandomState().set_state(restored)
    except (TypeError, ValueError) as exc:
        raise TrainingCheckpointError("NumPy global RNG state is invalid") from exc
    return restored


def _torch_global_state(
    value: object,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor | None]:
    payload = _exact_mapping(
        value,
        name="PyTorch global RNG state",
        expected={"cpu", "cuda", "mps"},
    )
    cpu = _rng_tensor(payload["cpu"], name="PyTorch CPU RNG state", dtype=torch.uint8)
    try:
        torch.Generator(device="cpu").set_state(cpu)
    except RuntimeError as exc:
        raise TrainingCheckpointError("PyTorch CPU RNG state is invalid") from exc
    raw_cuda = payload["cuda"]
    if not isinstance(raw_cuda, list | tuple):
        raise TrainingCheckpointError("PyTorch CUDA RNG states must be an array")
    cuda = tuple(
        _rng_tensor(item, name="PyTorch CUDA RNG state", dtype=torch.uint8) for item in raw_cuda
    )
    raw_mps = payload["mps"]
    mps = (
        None
        if raw_mps is None
        else _rng_tensor(raw_mps, name="PyTorch MPS RNG state", dtype=torch.uint8)
    )
    return cpu, cuda, mps


def _restored_numpy_generators(value: object) -> dict[str, np.random.Generator]:
    payload = _string_mapping(value, name="named NumPy generators")
    if not payload:
        raise TrainingCheckpointError("named NumPy generators cannot be empty")
    result: dict[str, np.random.Generator] = {}
    for name in sorted(payload):
        _generator_name(name)
        record = _exact_mapping(
            payload[name],
            name=f"NumPy generator {name}",
            expected={"bit_generator", "state"},
        )
        if record["bit_generator"] != "PCG64":
            raise TrainingCheckpointError(
                "named NumPy generator uses an unsupported bit generator",
                context={"name": name, "actual": record["bit_generator"]},
            )
        state = _string_mapping(record["state"], name=f"NumPy generator {name} state")
        generator = np.random.Generator(np.random.PCG64())
        try:
            generator.bit_generator.state = cast(dict[str, Any], state)
        except (TypeError, ValueError) as exc:
            raise TrainingCheckpointError(
                "named NumPy generator state is invalid",
                context={"name": name},
            ) from exc
        result[name] = generator
    return result


def _restored_torch_generators(value: object) -> dict[str, torch.Generator]:
    payload = _string_mapping(value, name="named PyTorch generators")
    if not payload:
        raise TrainingCheckpointError("named PyTorch generators cannot be empty")
    result: dict[str, torch.Generator] = {}
    for name in sorted(payload):
        _generator_name(name)
        record = _exact_mapping(
            payload[name],
            name=f"PyTorch generator {name}",
            expected={"device", "state"},
        )
        raw_device = record["device"]
        if not isinstance(raw_device, str) or not raw_device:
            raise TrainingCheckpointError("named PyTorch generator device must be text")
        state = _rng_tensor(
            record["state"],
            name=f"PyTorch generator {name} state",
            dtype=torch.uint8,
        )
        try:
            generator = torch.Generator(device=raw_device)
            generator.set_state(state)
        except (RuntimeError, TypeError) as exc:
            raise TrainingCheckpointError(
                "named PyTorch generator state or device is unavailable",
                context={"name": name, "device": raw_device},
            ) from exc
        result[name] = generator
    return result


def _rng_tensor(value: object, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != dtype
        or value.ndim != 1
        or value.numel() == 0
    ):
        raise TrainingCheckpointError(
            f"{name} must be a nonempty one-dimensional CPU {dtype} tensor"
        )
    return value.detach().clone()


def _software_versions(value: object) -> dict[str, str]:
    payload = _exact_mapping(
        value,
        name="software versions",
        expected={"python", "numpy", "torch"},
    )
    result: dict[str, str] = {}
    for name in ("python", "numpy", "torch"):
        item = payload[name]
        if not isinstance(item, str) or not item:
            raise TrainingCheckpointError("software versions must be nonempty strings")
        result[name] = item
    return result


def _load_model_state(module: torch.nn.Module, value: object, *, name: str) -> None:
    payload = _string_mapping(value, name=f"{name} model state")
    state: dict[str, torch.Tensor] = {}
    for key, item in payload.items():
        if not isinstance(item, torch.Tensor):
            raise TrainingCheckpointError(f"{name} model state values must be tensors")
        if (item.is_floating_point() or item.is_complex()) and not bool(
            torch.isfinite(item).all().item()
        ):
            raise TrainingCheckpointError(f"{name} model state contains a non-finite tensor")
        state[key] = item.detach().clone()
    try:
        module.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise TrainingCheckpointError(f"{name} model state is incompatible") from exc


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    value: object,
    *,
    name: str,
) -> None:
    payload = _exact_mapping(
        value,
        name=name,
        expected={"state", "param_groups"},
    )
    validated = _snapshot_tree(payload, name=name)
    if not isinstance(validated, dict):
        raise TrainingCheckpointError(f"{name} must be a mapping")
    try:
        optimizer.load_state_dict(cast(dict[str, Any], validated))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingCheckpointError(f"{name} is incompatible") from exc


def _capture_global_random_state() -> _GlobalRandomState:
    python_state = cast(tuple[int, tuple[int, ...], float | None], random.getstate())
    numpy_state = cast(
        tuple[str, NDArray[np.uint32], int, int, float],
        np.random.get_state(legacy=True),
    )
    cuda = (
        tuple(state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    mps = (
        torch.mps.get_rng_state().detach().cpu().clone()
        if torch.backends.mps.is_available()
        else None
    )
    return _GlobalRandomState(
        python=python_state,
        numpy=(
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch_cpu=torch.get_rng_state().detach().cpu().clone(),
        torch_cuda=cuda,
        torch_mps=mps,
    )


def _install_global_random_state(state: _GlobalRandomState) -> None:
    if state.torch_cuda and torch.cuda.is_available():
        if len(state.torch_cuda) != torch.cuda.device_count():
            raise TrainingCheckpointError(
                "checkpoint CUDA RNG state count does not match available devices"
            )
        torch.cuda.set_rng_state_all(list(state.torch_cuda))
    if state.torch_mps is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state.torch_mps)
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)


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
        if not isinstance(generator.bit_generator, np.random.PCG64):
            raise TrainingCheckpointError(
                "numpy_generators must use the project PCG64 bit generator",
                context={
                    "name": name,
                    "actual": type(generator.bit_generator).__name__,
                },
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
    "RestoredTrainingState",
    "restore_training_checkpoint",
    "save_training_checkpoint",
]
