"""Deterministic, named NumPy random streams.

Every generator is constructed explicitly from a stable hash of experiment
identity.  This prevents worker order, policy action choice, or unused-link
evaluation from shifting unrelated random outcomes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

SeedIdentifier: TypeAlias = str | int | None

SEED_DERIVATION_SCHEMA: Final = "hybrid-rf-vlc-rl.seed.v1"
MAX_ROOT_SEED: Final = (1 << 64) - 1
RANDOM_STREAM_NAMES: Final = (
    "mobility",
    "sensor",
    "shadowing",
    "fading",
    "collision",
    "decoding",
    "policy",
    "bootstrap",
)


def _validated_integer(
    value: int | np.integer,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be no greater than {maximum}")
    return normalized


def _validated_identifier(value: SeedIdentifier, *, name: str) -> SeedIdentifier:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a string, integer, or None")
    if isinstance(value, (int, np.integer)):
        return _validated_integer(value, name=name)
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{name} must not be empty")
        return value
    raise TypeError(f"{name} must be a string, integer, or None")


def _validated_stream_name(stream_name: str) -> str:
    if not isinstance(stream_name, str):
        raise TypeError("stream_name must be a string")
    if not stream_name or stream_name != stream_name.strip():
        raise ValueError("stream_name must be non-empty and have no surrounding whitespace")
    return stream_name


def derive_child_seed(
    root_seed: int | np.integer,
    stream_name: str,
    *,
    trace_id: SeedIdentifier = None,
    episode_id: SeedIdentifier = None,
    packet_index: int | np.integer | None = None,
) -> int:
    """Derive a stable unsigned 64-bit child seed from experiment identity.

    SHA-256 is applied to a versioned canonical JSON tuple.  The procedure is
    independent of Python's randomized ``hash()`` and NumPy's global RNG.
    Identifier types are retained, so integer ``1`` and string ``"1"`` belong
    to different namespaces.
    """

    normalized_root = _validated_integer(root_seed, name="root_seed", maximum=MAX_ROOT_SEED)
    normalized_stream = _validated_stream_name(stream_name)
    normalized_trace = _validated_identifier(trace_id, name="trace_id")
    normalized_episode = _validated_identifier(episode_id, name="episode_id")
    normalized_packet = (
        None if packet_index is None else _validated_integer(packet_index, name="packet_index")
    )

    identity = (
        SEED_DERIVATION_SCHEMA,
        normalized_root,
        normalized_stream,
        normalized_trace,
        normalized_episode,
        normalized_packet,
    )
    serialized = json.dumps(
        identity,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def derive_seed(
    root_seed: int | np.integer,
    stream_name: str,
    *,
    trace_id: SeedIdentifier = None,
    episode_id: SeedIdentifier = None,
    packet_index: int | np.integer | None = None,
) -> int:
    """Compatibility name for :func:`derive_child_seed`."""

    return derive_child_seed(
        root_seed,
        stream_name,
        trace_id=trace_id,
        episode_id=episode_id,
        packet_index=packet_index,
    )


def make_generator(
    root_seed: int | np.integer,
    stream_name: str,
    *,
    trace_id: SeedIdentifier = None,
    episode_id: SeedIdentifier = None,
    packet_index: int | np.integer | None = None,
) -> np.random.Generator:
    """Construct an explicit PCG64 generator for one deterministic namespace."""

    child_seed = derive_child_seed(
        root_seed,
        stream_name,
        trace_id=trace_id,
        episode_id=episode_id,
        packet_index=packet_index,
    )
    return np.random.Generator(np.random.PCG64(child_seed))


def derive_stream_seeds(
    root_seed: int | np.integer,
    *,
    trace_id: SeedIdentifier = None,
    episode_id: SeedIdentifier = None,
    packet_index: int | np.integer | None = None,
) -> dict[str, int]:
    """Return child seeds for every canonical stream without advancing an RNG."""

    return {
        stream_name: derive_child_seed(
            root_seed,
            stream_name,
            trace_id=trace_id,
            episode_id=episode_id,
            packet_index=packet_index,
        )
        for stream_name in RANDOM_STREAM_NAMES
    }


@dataclass(frozen=True, slots=True)
class RandomStreams:
    """Canonical set of independent explicit NumPy generators."""

    mobility: np.random.Generator
    sensor: np.random.Generator
    shadowing: np.random.Generator
    fading: np.random.Generator
    collision: np.random.Generator
    decoding: np.random.Generator
    policy: np.random.Generator
    bootstrap: np.random.Generator

    @classmethod
    def from_root_seed(
        cls,
        root_seed: int | np.integer,
        *,
        trace_id: SeedIdentifier = None,
        episode_id: SeedIdentifier = None,
        packet_index: int | np.integer | None = None,
    ) -> RandomStreams:
        """Build all canonical streams from one root and experiment context."""

        generators = {
            stream_name: make_generator(
                root_seed,
                stream_name,
                trace_id=trace_id,
                episode_id=episode_id,
                packet_index=packet_index,
            )
            for stream_name in RANDOM_STREAM_NAMES
        }
        return cls(**generators)

    def as_dict(self) -> dict[str, np.random.Generator]:
        """Return a new name-to-generator mapping without copying RNG state."""

        return {
            "mobility": self.mobility,
            "sensor": self.sensor,
            "shadowing": self.shadowing,
            "fading": self.fading,
            "collision": self.collision,
            "decoding": self.decoding,
            "policy": self.policy,
            "bootstrap": self.bootstrap,
        }


__all__ = [
    "MAX_ROOT_SEED",
    "RANDOM_STREAM_NAMES",
    "SEED_DERIVATION_SCHEMA",
    "RandomStreams",
    "SeedIdentifier",
    "derive_child_seed",
    "derive_seed",
    "derive_stream_seeds",
    "make_generator",
]
