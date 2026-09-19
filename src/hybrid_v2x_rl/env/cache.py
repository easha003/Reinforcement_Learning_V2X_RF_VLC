"""The half of a transition that no policy can change, measured once.

Training this environment the obvious way is dominated by work that has
nothing to do with the policy. Sensing, tracking, forecasting, the spatial
hash, the blockage estimate and both channel models together run at a few
hundred packets a second, and the frozen profile asks for ten million
transitions on each of five seeds. That is days of recomputing identical
numbers.

They are identical for a reason that is a design decision rather than an
accident. Every random draw a packet consumes is seeded from the packet's own
identity -- the matched-tape rule in :mod:`hybrid_v2x_rl.env.packet` -- so the risk
each action carries, the outcome it would produce and the quality it would
report are all functions of the trace alone. The policy cannot move them. Only
the link history depends on what was chosen, and that is a pair of bounded
deques.

So a cache stores the trace-derived columns and the per-action truth, and a
replay reconstructs the rest as a trajectory unfolds. This is not an
approximation of the environment; it is the same environment with the
policy-independent part hoisted out of the loop.

**What makes it safe** is that the reconstruction calls
:func:`hybrid_v2x_rl.observation.builder.link_feature_values` rather than
reimplementing it, and that :mod:`tests.unit.test_agents_cache` replays a
trajectory through both paths and compares the vectors element by element. A
cache that silently disagreed with the evaluator would train a policy against
an observation the reported results were never produced from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.observation.builder import (
    HISTORY_FEATURES,
    LINK_FEATURES,
    ObservationSchema,
    link_feature_values,
)
from hybrid_v2x_rl.observation.link_state import LinkStateTracker

#: Bumped when the on-disk layout changes in a way a reader cannot detect.
CACHE_FORMAT_VERSION = 1

#: Column order of ``risk``, ``delivered`` and ``quality``. Both legs are
#: stored for every packet regardless of any action, because the cache exists
#: precisely so that any policy can be replayed against it.
LEGS = ("rf", "vlc")


class CacheError(HybridV2XError):
    """A cache is missing, malformed, or built for a different observation."""


@dataclass(frozen=True, slots=True)
class ColumnPlan:
    """Where each half of the observation lands in the assembled vector.

    Computed from the schema rather than hardcoded, so reordering the feature
    list in config moves the columns here too instead of silently transposing
    the observation.
    """

    width: int
    #: Indices of the trace-derived columns, in the order the cache stores them.
    trace_columns: np.ndarray
    #: Per link-derived feature name, the slice it occupies in the vector.
    link_slices: dict[str, slice]
    #: Feature names whose cached columns are stored, in cache order.
    trace_features: tuple[str, ...]

    @classmethod
    def from_schema(cls, schema: ObservationSchema) -> ColumnPlan:
        trace_columns: list[int] = []
        trace_features: list[str] = []
        link_slices: dict[str, slice] = {}

        cursor = 0
        for feature in schema.features:
            span = schema.history_packets if feature in HISTORY_FEATURES else 1
            if feature in LINK_FEATURES:
                link_slices[feature] = slice(cursor, cursor + span)
            else:
                trace_columns.extend(range(cursor, cursor + span))
                trace_features.append(feature)
            cursor += span

        if cursor != schema.width:
            raise CacheError(
                "schema width does not match its own feature list",
                context={"walked": cursor, "width": schema.width},
            )
        missing = set(LINK_FEATURES) - set(link_slices)
        if missing and missing != set(LINK_FEATURES) - set(schema.features):
            raise CacheError(
                "a link feature is configured but was not placed",
                context={"missing": sorted(missing)},
            )
        return cls(
            width=schema.width,
            trace_columns=np.asarray(trace_columns, dtype=np.int64),
            link_slices=link_slices,
            trace_features=tuple(trace_features),
        )

    def assemble(
        self, trace_row: np.ndarray, links: LinkStateTracker, now_s: float,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """One full observation from a cached row and a live link history."""

        vector = np.empty(self.width, dtype=np.float32) if out is None else out
        vector[self.trace_columns] = trace_row
        values = link_feature_values(links, now_s)
        for name, span in self.link_slices.items():
            value = values[name]
            if span.stop - span.start == 1:
                vector[span.start] = float(value)  # type: ignore[arg-type]
            else:
                vector[span] = np.asarray(value, dtype=np.float32)
        return vector


@dataclass(frozen=True, slots=True)
class TransitionCache:
    """One trace's policy-independent transitions, memory mapped.

    Arrays are parallel: row ``i`` of every one of them describes the same
    packet.
    """

    #: Trace-derived observation columns, ``[packets, len(trace_columns)]``.
    trace: np.ndarray
    #: ``[packets, 2]`` marginal failure probability of each leg.
    risk: np.ndarray
    #: ``[packets, 2]`` whether each leg would deliver, on this packet's tape.
    delivered: np.ndarray
    #: ``[packets, 2]`` the quality each leg would report if it were spent.
    quality: np.ndarray
    #: ``[packets]`` trace time, needed for the ages in the link features.
    time_s: np.ndarray
    #: ``[packets]`` contiguous episode index; rows of one episode are adjacent.
    episode: np.ndarray
    #: ``[packets]`` whether this row ends its episode.
    final: np.ndarray
    manifest: dict

    @property
    def packets(self) -> int:
        return int(self.trace.shape[0])

    @property
    def density(self) -> float:
        return float(self.manifest["density"])

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        schema: ObservationSchema | None = None,
        expected_config_hash: str | None = None,
    ) -> TransitionCache:
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise CacheError(f"no cache manifest at {manifest_path}")
        manifest = json.loads(manifest_path.read_text())

        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise CacheError(
                "cache was written by a different format version; rebuild it",
                context={"found": manifest.get("format_version"),
                         "expected": CACHE_FORMAT_VERSION},
            )
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            raise CacheError(
                "cache was built for a different configuration; rebuild it",
                context={
                    "cached": manifest.get("config_hash"),
                    "configured": expected_config_hash,
                },
            )
        if schema is not None:
            expected = ColumnPlan.from_schema(schema)
            if tuple(manifest["trace_features"]) != expected.trace_features:
                raise CacheError(
                    "cache was built for a different observation; rebuild it",
                    context={"cached": manifest["trace_features"],
                             "configured": list(expected.trace_features)},
                )

        def read(name: str) -> np.ndarray:
            path = directory / f"{name}.npy"
            if not path.exists():
                raise CacheError(f"cache is missing {path.name}")
            return np.load(path, mmap_mode="r")

        return cls(
            trace=read("trace"), risk=read("risk"), delivered=read("delivered"),
            quality=read("quality"), time_s=read("time_s"), episode=read("episode"),
            final=read("final"), manifest=manifest,
        )

    @staticmethod
    def write(
        directory: Path, *, trace: np.ndarray, risk: np.ndarray, delivered: np.ndarray,
        quality: np.ndarray, time_s: np.ndarray, episode: np.ndarray, final: np.ndarray,
        manifest: dict,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            "trace": np.asarray(trace, dtype=np.float32),
            "risk": np.asarray(risk, dtype=np.float32),
            "delivered": np.asarray(delivered, dtype=np.uint8),
            "quality": np.asarray(quality, dtype=np.float32),
            "time_s": np.asarray(time_s, dtype=np.float64),
            "episode": np.asarray(episode, dtype=np.int32),
            "final": np.asarray(final, dtype=np.uint8),
        }
        counts = {name: len(array) for name, array in arrays.items()}
        if len(set(counts.values())) != 1:
            raise CacheError("cache arrays disagree on length", context=counts)
        for name, array in arrays.items():
            np.save(directory / f"{name}.npy", array)
        (directory / "manifest.json").write_text(
            json.dumps({**manifest, "format_version": CACHE_FORMAT_VERSION,
                        "packets": int(len(time_s))}, indent=1)
        )


__all__ = [
    "CACHE_FORMAT_VERSION",
    "LEGS",
    "CacheError",
    "ColumnPlan",
    "TransitionCache",
]
