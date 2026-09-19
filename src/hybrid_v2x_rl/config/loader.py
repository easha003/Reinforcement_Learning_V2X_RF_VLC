"""Deterministic layered YAML loading for Hybrid RF/VLC RL configurations."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from pydantic import ValidationError
from yaml.constructor import ConstructorError

from hybrid_v2x_rl.config.models import PathConfig, ProjectConfig
from hybrid_v2x_rl.config.validation import validate_project_config
from hybrid_v2x_rl.core.errors import ConfigurationError

PathLike: TypeAlias = str | Path


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings, with later layers taking precedence.

    Mapping values merge recursively.  Lists, tuples, scalars, and explicit
    ``null`` values replace the earlier value as one atomic unit.  Neither input
    is mutated.
    """

    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        previous = result.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(previous, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_file(path: PathLike) -> dict[str, Any]:
    """Load one YAML mapping with duplicate-key protection."""

    source = Path(path).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration layer {source}: {exc}") from exc

    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in configuration layer {source}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"configuration layer {source} must contain a top-level mapping")
    if any(not isinstance(key, str) for key in loaded):
        raise ConfigurationError(f"configuration layer {source} has a non-string top-level key")
    return loaded


def _absolute(path: Path, root: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = root / expanded
    return expanded.resolve(strict=False)


def _resolve_paths(config: ProjectConfig, project_root: Path) -> ProjectConfig:
    root = project_root.expanduser().resolve(strict=False)
    old_paths = config.paths
    resolved_paths = PathConfig(
        project_root=root,
        artifact_root=_absolute(old_paths.artifact_root, root),
        calibration_root=_absolute(old_paths.calibration_root, root),
        trace_root=_absolute(old_paths.trace_root, root),
        checkpoint_root=_absolute(old_paths.checkpoint_root, root),
        evaluation_root=_absolute(old_paths.evaluation_root, root),
        figure_root=_absolute(old_paths.figure_root, root),
    )
    mobility = config.mobility
    if mobility.network_artifact is not None:
        mobility = mobility.model_copy(
            update={
                "network_artifact": _absolute(
                    mobility.network_artifact,
                    root,
                )
            }
        )
    rf = config.rf.model_copy(
        update={
            "calibration_artifact": _absolute(
                config.rf.calibration_artifact,
                root,
            )
        }
    )
    vlc = config.vlc.model_copy(
        update={
            "pattern_artifact": _absolute(
                config.vlc.pattern_artifact,
                root,
            )
        }
    )
    return config.model_copy(
        update={
            "paths": resolved_paths,
            "mobility": mobility,
            "rf": rf,
            "vlc": vlc,
        }
    )


def _normalize_layers(paths: PathLike | Sequence[PathLike]) -> tuple[Path, ...]:
    normalized: tuple[Path, ...]
    if isinstance(paths, (str, Path)):
        normalized = (Path(paths),)
    else:
        normalized = tuple(Path(path) for path in paths)
    if not normalized:
        raise ConfigurationError("at least one configuration layer is required")
    return normalized


def load_config(
    paths: PathLike | Sequence[PathLike],
    *,
    project_root: PathLike | None = None,
) -> ProjectConfig:
    """Load, resolve, and validate explicit YAML layers in listed order."""

    layers = _normalize_layers(paths)
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, load_yaml_file(layer))

    try:
        parsed = ProjectConfig.model_validate(merged)
    except ValidationError as exc:
        layer_list = ", ".join(str(layer) for layer in layers)
        raise ConfigurationError(
            f"configuration validation failed for layers [{layer_list}]: {exc}"
        ) from exc

    if project_root is not None:
        root = Path(project_root)
    elif parsed.paths.project_root.is_absolute():
        root = parsed.paths.project_root
    else:
        root = Path.cwd() / parsed.paths.project_root

    resolved = _resolve_paths(parsed, root)
    return validate_project_config(resolved)


def headline_config_layers(project_root: PathLike) -> tuple[Path, ...]:
    """Return the canonical ordered layer list for the headline experiment."""

    root = Path(project_root).expanduser().resolve(strict=False)
    config_root = root / "configs"
    return (
        config_root / "project" / "default.yaml",
        config_root / "mobility" / "synthetic_manhattan.yaml",
        config_root / "service" / "ev2x_300B_3ms_1e-4.yaml",
        config_root / "channel" / "rf_nr_v2x.yaml",
        config_root / "channel" / "vlc_vehicle.yaml",
        config_root / "observation" / "causal_200ms_forecast.yaml",
        config_root / "training" / "primal_dual_ppo.yaml",
        config_root / "evaluation" / "primary.yaml",
    )


def load_headline_config(
    project_root: PathLike | None = None,
) -> ProjectConfig:
    """Load the repository's frozen headline configuration."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]
    return load_config(
        headline_config_layers(root),
        project_root=root,
    )
