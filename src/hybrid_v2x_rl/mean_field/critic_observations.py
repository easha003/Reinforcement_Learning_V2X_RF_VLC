"""Training-only centralized critic inputs kept outside the actor API.

The decentralized actor receives one normalized 37-column row and its action
mask.  During training only, each reward/cost critic receives that same local
row followed by a 41-column population summary:

``mean actor row (37) + density one-hot (3) + log1p(population size) (1)``.

This module materializes the resulting 78-column matrix in a separate type.
It never mutates or extends :class:`~hybrid_v2x_rl.mean_field.environment_api.FrameObservation`,
so decentralized execution has no critic suffix to remove or impute.  The
population frame supplies the density label and stable identities; no caller
may relabel a trace by passing a free-standing density value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.congestion_feedback import ActorObservationSchema
from hybrid_v2x_rl.mean_field.environment_api import (
    CONTRACT_ACTOR_WIDTH,
    FRAME_API_VERSION,
    FrameObservation,
)
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.observation.builder import ObservationBuilder

CONTRACT_DENSITY_COUNT: Final = 3
CONTRACT_GLOBAL_SUMMARY_WIDTH: Final = CONTRACT_ACTOR_WIDTH + CONTRACT_DENSITY_COUNT + 1
CONTRACT_CRITIC_WIDTH: Final = CONTRACT_ACTOR_WIDTH + CONTRACT_GLOBAL_SUMMARY_WIDTH

GlobalSummaryArray: TypeAlias = NDArray[np.float32]
CriticObservationArray: TypeAlias = NDArray[np.float32]


class CriticObservationError(HybridV2XError):
    """A training-only centralized observation violates contract 1.0.0."""


def _freeze_float32(value: NDArray[np.float32]) -> NDArray[np.float32]:
    frozen = np.ascontiguousarray(value, dtype=np.float32).copy()
    frozen.setflags(write=False)
    return frozen


def _validate_pair_ids(pair_ids: tuple[str, ...]) -> None:
    if not isinstance(pair_ids, tuple):
        raise CriticObservationError("critic pair_ids must be an immutable tuple")
    if any(not isinstance(pair_id, str) or not pair_id.strip() for pair_id in pair_ids):
        raise CriticObservationError("critic pair_ids must be non-empty strings")
    if pair_ids != tuple(sorted(pair_ids)):
        raise CriticObservationError("critic pair_ids must use canonical stable-ID order")
    if len(pair_ids) != len(set(pair_ids)):
        raise CriticObservationError("critic pair_ids cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class CentralizedCriticSchema:
    """Versioned actor, population-summary, and critic column contracts."""

    contract_version: str
    actor_columns: tuple[str, ...]
    density_levels: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.contract_version != FRAME_API_VERSION:
            raise CriticObservationError(
                "critic schema version does not match the environment contract",
                context={
                    "actual": self.contract_version,
                    "expected": FRAME_API_VERSION,
                },
            )
        if len(self.actor_columns) != CONTRACT_ACTOR_WIDTH:
            raise CriticObservationError(
                "critic schema requires the frozen 37 actor columns",
                context={
                    "actual": len(self.actor_columns),
                    "expected": CONTRACT_ACTOR_WIDTH,
                },
            )
        if len(set(self.actor_columns)) != len(self.actor_columns):
            raise CriticObservationError("actor column names must be unique")
        if len(self.density_levels) != CONTRACT_DENSITY_COUNT:
            raise CriticObservationError(
                "critic schema requires exactly three configured density levels",
                context={
                    "actual": len(self.density_levels),
                    "expected": CONTRACT_DENSITY_COUNT,
                },
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in self.density_levels):
            raise CriticObservationError("critic density levels must be finite and positive")
        if self.density_levels != tuple(sorted(self.density_levels)):
            raise CriticObservationError("critic density levels must use ascending order")
        if len(set(self.density_levels)) != len(self.density_levels):
            raise CriticObservationError("critic density levels must be unique")

    @classmethod
    def from_config(cls, config: ProjectConfig) -> CentralizedCriticSchema:
        """Derive every column and density position from resolved configuration."""

        if not isinstance(config, ProjectConfig):
            raise CriticObservationError(
                "critic schema requires a resolved ProjectConfig"
            )
        local = ObservationBuilder.from_config(config.observation).schema
        actor = ActorObservationSchema(local=local)
        return cls(
            contract_version=config.environment.contract_version,
            actor_columns=actor.columns,
            density_levels=tuple(
                float(value)
                for value in config.mobility.target_densities_veh_per_lane_km
            ),
        )

    @property
    def actor_width(self) -> int:
        return len(self.actor_columns)

    @property
    def global_columns(self) -> tuple[str, ...]:
        population_mean = tuple(
            f"population_mean[{column}]" for column in self.actor_columns
        )
        density = tuple(
            f"density_{level:g}_veh_per_lane_km" for level in self.density_levels
        )
        return (*population_mean, *density, "log1p_population_size")

    @property
    def global_width(self) -> int:
        return len(self.global_columns)

    @property
    def critic_columns(self) -> tuple[str, ...]:
        return (*self.actor_columns, *self.global_columns)

    @property
    def critic_width(self) -> int:
        return len(self.critic_columns)

    def density_one_hot(self, density: float) -> GlobalSummaryArray:
        """Encode only an exact configured density label in frozen order."""

        if not math.isfinite(density) or density <= 0.0:
            raise CriticObservationError("frame density must be finite and positive")
        try:
            index = self.density_levels.index(float(density))
        except ValueError as error:
            raise CriticObservationError(
                "frame density is not one of the configured critic levels",
                context={"density": density, "configured": self.density_levels},
            ) from error
        encoded = np.zeros(CONTRACT_DENSITY_COUNT, dtype=np.float32)
        encoded[index] = 1.0
        return encoded


@dataclass(frozen=True, slots=True)
class CriticObservationFrame:
    """Immutable training-only critic tensor aligned to actor pair IDs."""

    trace_id: str
    frame_index: int
    time_s: float
    density: float
    schema: CentralizedCriticSchema
    pair_ids: tuple[str, ...]
    global_summary: GlobalSummaryArray | None
    critic_observations: CriticObservationArray

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise CriticObservationError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise CriticObservationError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise CriticObservationError("time_s must be finite and non-negative")
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise CriticObservationError("density must be finite and positive")
        if not isinstance(self.schema, CentralizedCriticSchema):
            raise CriticObservationError(
                "critic frame requires its versioned training-only schema"
            )
        _validate_pair_ids(self.pair_ids)
        density_one_hot = self.schema.density_one_hot(self.density)

        critic = self.critic_observations
        if not isinstance(critic, np.ndarray) or critic.dtype != np.dtype(np.float32):
            raise CriticObservationError("critic observations must be a float32 ndarray")
        expected_shape = (len(self.pair_ids), self.schema.critic_width)
        if critic.shape != expected_shape:
            raise CriticObservationError(
                "critic observations do not match the pair-aligned contract shape",
                context={"actual": critic.shape, "expected": expected_shape},
            )
        if not bool(np.all(np.isfinite(critic))):
            raise CriticObservationError("critic observations must be finite")

        summary = self.global_summary
        if not self.pair_ids:
            if summary is not None:
                raise CriticObservationError(
                    "an empty population has no defined population-mean summary"
                )
        else:
            if not isinstance(summary, np.ndarray) or summary.dtype != np.dtype(np.float32):
                raise CriticObservationError("global summary must be a float32 ndarray")
            if summary.shape != (self.schema.global_width,):
                raise CriticObservationError(
                    "global summary does not match the 41-column contract",
                    context={
                        "actual": summary.shape,
                        "expected": (self.schema.global_width,),
                    },
                )
            if not bool(np.all(np.isfinite(summary))):
                raise CriticObservationError("global summary must be finite")
            population_mean = np.mean(
                critic[:, : self.schema.actor_width],
                axis=0,
                dtype=np.float64,
            ).astype(np.float32)
            if not bool(
                np.array_equal(summary[: self.schema.actor_width], population_mean)
            ):
                raise CriticObservationError(
                    "global summary mean does not reconcile with the actor prefixes"
                )
            density_start = self.schema.actor_width
            density_stop = density_start + CONTRACT_DENSITY_COUNT
            if not bool(
                np.array_equal(summary[density_start:density_stop], density_one_hot)
            ):
                raise CriticObservationError(
                    "global summary density does not match the trace-carried label"
                )
            expected_population = np.float32(math.log1p(len(self.pair_ids)))
            if summary[-1] != expected_population:
                raise CriticObservationError(
                    "global summary population size does not match pair_ids"
                )
            if not bool(np.all(critic[:, self.schema.actor_width :] == summary)):
                raise CriticObservationError(
                    "every critic row must carry the same frame-global suffix"
                )
            object.__setattr__(self, "global_summary", _freeze_float32(summary))

        object.__setattr__(self, "critic_observations", _freeze_float32(critic))

    @property
    def population_size(self) -> int:
        return len(self.pair_ids)


@dataclass(frozen=True, slots=True)
class CentralizedCriticBuilder:
    """Construct critic rows from an actor-facing frame without changing it."""

    schema: CentralizedCriticSchema

    def __post_init__(self) -> None:
        if not isinstance(self.schema, CentralizedCriticSchema):
            raise CriticObservationError(
                "critic builder requires a CentralizedCriticSchema"
            )

    @classmethod
    def from_config(cls, config: ProjectConfig) -> CentralizedCriticBuilder:
        return cls(schema=CentralizedCriticSchema.from_config(config))

    def build(
        self,
        frame: PopulationFrame,
        normalized_actor_frame: FrameObservation,
    ) -> CriticObservationFrame:
        """Append a training-only global suffix to normalized actor rows.

        ``normalized_actor_frame`` must be the exact actor-facing frame used to
        sample actions.  Normalization is deliberately upstream: the population
        mean must be computed over those frozen normalized values, not over raw
        features or statistics updated with the current frame.
        """

        if not isinstance(frame, PopulationFrame):
            raise CriticObservationError("critic construction requires a PopulationFrame")
        if not isinstance(normalized_actor_frame, FrameObservation):
            raise CriticObservationError(
                "critic construction requires the actor-facing FrameObservation"
            )
        observation = normalized_actor_frame
        identity_mismatches: dict[str, object] = {}
        if observation.trace_id != frame.trace_id:
            identity_mismatches["trace_id"] = (
                observation.trace_id,
                frame.trace_id,
            )
        if observation.frame_index != frame.index:
            identity_mismatches["frame_index"] = (
                observation.frame_index,
                frame.index,
            )
        if not math.isclose(
            observation.time_s,
            frame.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            identity_mismatches["time_s"] = (observation.time_s, frame.time_s)
        if observation.pair_ids != frame.active_pair_ids:
            identity_mismatches["pair_ids"] = (
                observation.pair_ids,
                frame.active_pair_ids,
            )
        if identity_mismatches:
            raise CriticObservationError(
                "actor and population frames are not the same decision frame",
                context=identity_mismatches,
            )
        if observation.actor_observations.shape[1] != self.schema.actor_width:
            raise CriticObservationError(
                "actor observation width does not match the critic schema",
                context={
                    "actual": observation.actor_observations.shape[1],
                    "expected": self.schema.actor_width,
                },
            )

        # Validate the trace-carried label even for empty frames.  Empty frames
        # emit no critic rows, but they cannot silently bypass split metadata.
        density_one_hot = self.schema.density_one_hot(frame.density)
        population = observation.population_size
        if population == 0:
            empty = np.empty((0, self.schema.critic_width), dtype=np.float32)
            return CriticObservationFrame(
                trace_id=frame.trace_id,
                frame_index=frame.index,
                time_s=frame.time_s,
                density=frame.density,
                schema=self.schema,
                pair_ids=observation.pair_ids,
                global_summary=None,
                critic_observations=empty,
            )

        actor = observation.actor_observations
        population_mean = np.mean(actor, axis=0, dtype=np.float64).astype(np.float32)
        population_size = np.asarray([math.log1p(population)], dtype=np.float32)
        global_summary = np.concatenate(
            (population_mean, density_one_hot, population_size)
        ).astype(np.float32, copy=False)
        suffix = np.broadcast_to(global_summary, (population, self.schema.global_width))
        critic = np.concatenate((actor, suffix), axis=1).astype(np.float32, copy=False)

        return CriticObservationFrame(
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            density=frame.density,
            schema=self.schema,
            pair_ids=observation.pair_ids,
            global_summary=global_summary,
            critic_observations=critic,
        )


__all__ = [
    "CONTRACT_CRITIC_WIDTH",
    "CONTRACT_DENSITY_COUNT",
    "CONTRACT_GLOBAL_SUMMARY_WIDTH",
    "CentralizedCriticBuilder",
    "CentralizedCriticSchema",
    "CriticObservationArray",
    "CriticObservationError",
    "CriticObservationFrame",
    "GlobalSummaryArray",
]
