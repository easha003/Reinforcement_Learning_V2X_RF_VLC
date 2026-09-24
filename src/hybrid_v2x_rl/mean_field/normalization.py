"""Frame-frozen, train-only normalization for contract-1.0.0 actor rows.

The normalizer is a decision-frame state machine rather than a free-standing
``transform`` function.  It freezes Welford statistics before any actor in a
frame is transformed, then requires the selected joint action before it will
batch-update from that frame's valid raw rows.  Validation and test frames are
accepted only after the state has been frozen, and frozen state is immutable.

Four encoded columns are passed through unchanged.  Missing finite sentinels
and history padding in every other column are ordinary training samples, as the
environment contract requires.  Rows with no causal observation remain an
explicit non-learning zero placeholder in the rectangular API tensor; their
configured fallback action is checked before a frame can close, and they never
update statistics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.actor_observations import CausalActorFrame
from hybrid_v2x_rl.mean_field.congestion_feedback import ActorObservationSchema
from hybrid_v2x_rl.mean_field.environment_api import (
    ActionArray,
    FrameAPISchema,
    FrameObservation,
)
from hybrid_v2x_rl.mean_field.frames import PopulationFrame, TraceSplit
from hybrid_v2x_rl.observation.builder import ObservationBuilder

NORMALIZATION_STATE_SCHEMA: Final = "hybrid-rf-vlc-rl.observation-normalization.v1"
PASSTHROUGH_COLUMNS: Final = (
    "path_spans_junction",
    "previous_action",
    "last_delivery_outcome",
    "mean_field_valid",
)

BoolArray = NDArray[np.bool_]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


class ObservationNormalizationError(HybridV2XError):
    """Normalization state, timing, split, or checkpoint data is invalid."""


def _finite_tuple(values: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, list | tuple):
        raise ObservationNormalizationError(f"{name} must be an array")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ObservationNormalizationError(f"{name} must contain finite values")
    return converted


def _integer_tuple(values: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(values, list | tuple):
        raise ObservationNormalizationError(f"{name} must be an array")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ObservationNormalizationError(f"{name} must contain non-negative integers")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class ObservationNormalizationState:
    """JSON-safe checkpoint state for every actor column."""

    schema: str
    contract_version: str
    columns: tuple[str, ...]
    standardized: tuple[bool, ...]
    epsilon: float
    clip_abs: float
    frozen: bool
    count: tuple[int, ...]
    mean: tuple[float, ...]
    second_moment: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema != NORMALIZATION_STATE_SCHEMA:
            raise ObservationNormalizationError(
                "normalization checkpoint schema is unsupported",
                context={"actual": self.schema, "expected": NORMALIZATION_STATE_SCHEMA},
            )
        if not isinstance(self.contract_version, str) or not self.contract_version:
            raise ObservationNormalizationError("normalization contract_version must be non-empty")
        if (
            not isinstance(self.columns, tuple)
            or not self.columns
            or any(not isinstance(name, str) or not name for name in self.columns)
            or len(set(self.columns)) != len(self.columns)
        ):
            raise ObservationNormalizationError(
                "normalization columns must be unique non-empty names"
            )
        width = len(self.columns)
        if (
            not isinstance(self.standardized, tuple)
            or len(self.standardized) != width
            or any(type(value) is not bool for value in self.standardized)
        ):
            raise ObservationNormalizationError(
                "standardized mask must be boolean and column-aligned"
            )
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ObservationNormalizationError("epsilon must be finite and positive")
        if not math.isfinite(self.clip_abs) or self.clip_abs <= 0.0:
            raise ObservationNormalizationError("clip_abs must be finite and positive")
        if type(self.frozen) is not bool:
            raise ObservationNormalizationError("frozen must be boolean")
        if not all(
            isinstance(values, tuple) and len(values) == width
            for values in (self.count, self.mean, self.second_moment)
        ):
            raise ObservationNormalizationError(
                "checkpoint statistics must align with actor columns"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.count
        ):
            raise ObservationNormalizationError(
                "normalization counts must be non-negative integers"
            )
        if not all(math.isfinite(value) for value in (*self.mean, *self.second_moment)):
            raise ObservationNormalizationError(
                "normalization moments must contain only finite values"
            )
        if any(value < 0.0 for value in self.second_moment):
            raise ObservationNormalizationError("normalization second moments must be non-negative")
        for index, is_standardized in enumerate(self.standardized):
            if not is_standardized and (
                self.count[index] != 0
                or self.mean[index] != 0.0
                or self.second_moment[index] != 0.0
            ):
                raise ObservationNormalizationError(
                    "pass-through columns must have canonical zero statistics",
                    context={"column": self.columns[index]},
                )
            if self.count[index] == 0 and (
                self.mean[index] != 0.0 or self.second_moment[index] != 0.0
            ):
                raise ObservationNormalizationError(
                    "unseen columns must have canonical zero moments",
                    context={"column": self.columns[index]},
                )
            if self.count[index] == 1 and self.second_moment[index] != 0.0:
                raise ObservationNormalizationError(
                    "a one-row column must have zero second moment",
                    context={"column": self.columns[index]},
                )

    def as_dict(self) -> dict[str, object]:
        """Return a stable JSON-serializable checkpoint payload."""

        return {
            "schema": self.schema,
            "contract_version": self.contract_version,
            "columns": list(self.columns),
            "standardized": list(self.standardized),
            "epsilon": self.epsilon,
            "clip_abs": self.clip_abs,
            "frozen": self.frozen,
            "count": list(self.count),
            "mean": list(self.mean),
            "second_moment": list(self.second_moment),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObservationNormalizationState:
        """Parse checkpoint data without accepting missing or unknown fields."""

        if not isinstance(payload, Mapping):
            raise ObservationNormalizationError("normalization checkpoint must be a mapping")
        expected = {
            "schema",
            "contract_version",
            "columns",
            "standardized",
            "epsilon",
            "clip_abs",
            "frozen",
            "count",
            "mean",
            "second_moment",
        }
        actual = set(payload)
        if actual != expected:
            raise ObservationNormalizationError(
                "normalization checkpoint fields do not match the schema",
                context={
                    "missing": tuple(sorted(expected - actual)),
                    "unexpected": tuple(sorted(actual - expected)),
                },
            )
        columns_raw = payload["columns"]
        standardized_raw = payload["standardized"]
        if not isinstance(columns_raw, list | tuple) or any(
            not isinstance(value, str) for value in columns_raw
        ):
            raise ObservationNormalizationError("columns must be a string array")
        if not isinstance(standardized_raw, list | tuple) or any(
            type(value) is not bool for value in standardized_raw
        ):
            raise ObservationNormalizationError("standardized must be a boolean array")
        schema = payload["schema"]
        contract_version = payload["contract_version"]
        epsilon = payload["epsilon"]
        clip_abs = payload["clip_abs"]
        frozen = payload["frozen"]
        if not isinstance(schema, str) or not isinstance(contract_version, str):
            raise ObservationNormalizationError(
                "checkpoint schema and contract_version must be strings"
            )
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, int | float)
            or isinstance(clip_abs, bool)
            or not isinstance(clip_abs, int | float)
            or type(frozen) is not bool
        ):
            raise ObservationNormalizationError("checkpoint constants have invalid types")
        return cls(
            schema=schema,
            contract_version=contract_version,
            columns=tuple(columns_raw),
            standardized=tuple(standardized_raw),
            epsilon=float(epsilon),
            clip_abs=float(clip_abs),
            frozen=frozen,
            count=_integer_tuple(payload["count"], name="count"),
            mean=_finite_tuple(payload["mean"], name="mean"),
            second_moment=_finite_tuple(payload["second_moment"], name="second_moment"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedActorFrame:
    """Actor-facing normalized frame plus the explicit policy-control mask."""

    split: TraceSplit
    observation: FrameObservation
    usable_mask: BoolArray
    statistics_count_before: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ObservationNormalizationError("normalized frame split is invalid")
        if not isinstance(self.observation, FrameObservation):
            raise ObservationNormalizationError(
                "normalized actor frame requires a FrameObservation"
            )
        usable = self.usable_mask
        if not isinstance(usable, np.ndarray) or usable.dtype != np.dtype(np.bool_):
            raise ObservationNormalizationError("usable_mask must be a bool ndarray")
        if usable.shape != (self.observation.population_size,):
            raise ObservationNormalizationError("usable_mask must align with normalized actor rows")
        if len(self.statistics_count_before) != self.observation.actor_observations.shape[1]:
            raise ObservationNormalizationError(
                "statistics_count_before must align with actor columns"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.statistics_count_before
        ):
            raise ObservationNormalizationError(
                "statistics_count_before must contain non-negative integers"
            )
        if bool(np.any(self.observation.actor_observations[~usable] != np.float32(0.0))):
            raise ObservationNormalizationError(
                "unusable API rows must use the internal zero placeholder"
            )
        frozen = usable.copy(order="C")
        frozen.setflags(write=False)
        object.__setattr__(self, "usable_mask", frozen)

    @property
    def learn_mask(self) -> BoolArray:
        return self.usable_mask


@dataclass(frozen=True, slots=True)
class _PendingNormalizationFrame:
    normalized: NormalizedActorFrame
    raw_valid_rows: Float64Array


@dataclass(slots=True)
class ObservationNormalizer:
    """Training-state owner enforcing frame timing and trace-split isolation."""

    contract_version: str
    columns: tuple[str, ...]
    standardized: tuple[bool, ...]
    epsilon: float
    clip_abs: float
    api_schema: FrameAPISchema
    action_space: MaskedActionSpace
    _count: Int64Array = field(init=False, repr=False)
    _mean: Float64Array = field(init=False, repr=False)
    _second_moment: Float64Array = field(init=False, repr=False)
    _frozen: bool = field(default=False, init=False, repr=False)
    _pending: _PendingNormalizationFrame | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.contract_version != self.api_schema.contract_version:
            raise ObservationNormalizationError("normalizer and frame API contract versions differ")
        if len(self.columns) != self.api_schema.actor_width:
            raise ObservationNormalizationError(
                "normalizer columns do not match the actor API width"
            )
        if len(set(self.columns)) != len(self.columns):
            raise ObservationNormalizationError("normalizer columns must be unique")
        expected_standardized = tuple(column not in PASSTHROUGH_COLUMNS for column in self.columns)
        if self.standardized != expected_standardized:
            raise ObservationNormalizationError(
                "normalizer pass-through columns do not match contract 1.0.0",
                context={"passthrough": PASSTHROUGH_COLUMNS},
            )
        missing_passthrough = tuple(
            column for column in PASSTHROUGH_COLUMNS if column not in self.columns
        )
        if missing_passthrough:
            raise ObservationNormalizationError(
                "actor schema is missing contract pass-through columns",
                context={"missing": missing_passthrough},
            )
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ObservationNormalizationError("epsilon must be finite and positive")
        if not math.isfinite(self.clip_abs) or self.clip_abs <= 0.0:
            raise ObservationNormalizationError("clip_abs must be finite and positive")
        width = len(self.columns)
        self._count = np.zeros(width, dtype=np.int64)
        self._mean = np.zeros(width, dtype=np.float64)
        self._second_moment = np.zeros(width, dtype=np.float64)

    @classmethod
    def from_config(cls, config: ProjectConfig) -> ObservationNormalizer:
        """Bind schemas, constants, hardware masks, and fallback to configuration."""

        if not isinstance(config, ProjectConfig):
            raise ObservationNormalizationError("normalizer requires a resolved ProjectConfig")
        normalization = config.environment.normalization
        if normalization.method != "running_standardization":
            raise ObservationNormalizationError("normalizer method must be running_standardization")
        if normalization.update_scope != "training_only":
            raise ObservationNormalizationError("normalizer update_scope must be training_only")
        local = ObservationBuilder.from_config(config.observation).schema
        actor = ActorObservationSchema(local=local)
        columns = actor.columns
        return cls(
            contract_version=config.environment.contract_version,
            columns=columns,
            standardized=tuple(column not in PASSTHROUGH_COLUMNS for column in columns),
            epsilon=normalization.epsilon,
            clip_abs=normalization.clip_abs,
            api_schema=FrameAPISchema.from_config(config),
            action_space=MaskedActionSpace.from_config(
                config.environment,
                config.rf,
                config.vlc,
            ),
        )

    @classmethod
    def from_state_dict(
        cls,
        config: ProjectConfig,
        payload: Mapping[str, object],
    ) -> ObservationNormalizer:
        """Restore statistics only when checkpoint metadata matches configuration."""

        normalizer = cls.from_config(config)
        state = ObservationNormalizationState.from_dict(payload)
        mismatches: dict[str, object] = {}
        for name, actual, expected in (
            ("contract_version", state.contract_version, normalizer.contract_version),
            ("columns", state.columns, normalizer.columns),
            ("standardized", state.standardized, normalizer.standardized),
            ("epsilon", state.epsilon, normalizer.epsilon),
            ("clip_abs", state.clip_abs, normalizer.clip_abs),
        ):
            if actual != expected:
                mismatches[name] = {"checkpoint": actual, "config": expected}
        if mismatches:
            raise ObservationNormalizationError(
                "normalization checkpoint does not match resolved configuration",
                context=mismatches,
            )
        normalizer._count = np.asarray(state.count, dtype=np.int64)
        normalizer._mean = np.asarray(state.mean, dtype=np.float64)
        normalizer._second_moment = np.asarray(
            state.second_moment,
            dtype=np.float64,
        )
        normalizer._frozen = state.frozen
        return normalizer

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def training_rows(self) -> int:
        standardized_counts = self._count[np.asarray(self.standardized, dtype=np.bool_)]
        if standardized_counts.size == 0:
            return 0
        if not bool(np.all(standardized_counts == standardized_counts[0])):
            raise ObservationNormalizationError("standardized column counts have diverged")
        return int(standardized_counts[0])

    def snapshot(self) -> ObservationNormalizationState:
        """Return immutable state only between decision frames."""

        if self._pending is not None:
            raise ObservationNormalizationError(
                "cannot snapshot normalization while a decision frame is open"
            )
        return ObservationNormalizationState(
            schema=NORMALIZATION_STATE_SCHEMA,
            contract_version=self.contract_version,
            columns=self.columns,
            standardized=self.standardized,
            epsilon=self.epsilon,
            clip_abs=self.clip_abs,
            frozen=self._frozen,
            count=tuple(int(value) for value in self._count),
            mean=tuple(float(value) for value in self._mean),
            second_moment=tuple(float(value) for value in self._second_moment),
        )

    def state_dict(self) -> Mapping[str, object]:
        """Return an immutable checkpoint mapping."""

        return MappingProxyType(self.snapshot().as_dict())

    def freeze(self) -> ObservationNormalizationState:
        """Irreversibly disable updates before validation/test or evaluation."""

        if self._pending is not None:
            raise ObservationNormalizationError(
                "cannot freeze normalization while a decision frame is open"
            )
        self._frozen = True
        return self.snapshot()

    def _validate_identity(
        self,
        frame: PopulationFrame,
        actor_frame: CausalActorFrame,
    ) -> None:
        mismatches: dict[str, object] = {}
        if frame.trace_id != actor_frame.trace_id:
            mismatches["trace_id"] = (frame.trace_id, actor_frame.trace_id)
        if frame.index != actor_frame.frame_index:
            mismatches["frame_index"] = (frame.index, actor_frame.frame_index)
        if not math.isclose(
            frame.time_s,
            actor_frame.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            mismatches["time_s"] = (frame.time_s, actor_frame.time_s)
        if frame.active_pair_ids != actor_frame.pair_ids:
            mismatches["pair_ids"] = (frame.active_pair_ids, actor_frame.pair_ids)
        if actor_frame.schema.columns != self.columns:
            mismatches["columns"] = (actor_frame.schema.columns, self.columns)
        if mismatches:
            raise ObservationNormalizationError(
                "population and causal actor frames do not share one identity",
                context=mismatches,
            )

    def _transform_valid_rows(self, raw: Float64Array) -> NDArray[np.float32]:
        transformed = raw.copy(order="C")
        mask = np.asarray(self.standardized, dtype=np.bool_)
        historical = self._count >= 2
        means = np.where(historical, self._mean, 0.0)
        variances = np.ones(len(self.columns), dtype=np.float64)
        variances[historical] = self._second_moment[historical] / (self._count[historical] - 1)
        standardized = (raw[:, mask] - means[mask]) / np.sqrt(variances[mask] + self.epsilon)
        transformed[:, mask] = np.clip(
            standardized,
            -self.clip_abs,
            self.clip_abs,
        )
        result = transformed.astype(np.float32)
        if not bool(np.all(np.isfinite(result))):
            raise ObservationNormalizationError("normalized actor observations must be finite")
        return result

    def begin_frame(
        self,
        frame: PopulationFrame,
        actor_frame: CausalActorFrame,
    ) -> NormalizedActorFrame:
        """Transform with pre-frame statistics and hold raw rows until actions exist."""

        if self._pending is not None:
            raise ObservationNormalizationError(
                "the open normalization frame must complete before another begins"
            )
        if not isinstance(frame, PopulationFrame):
            raise ObservationNormalizationError("normalization requires a PopulationFrame")
        if not isinstance(actor_frame, CausalActorFrame):
            raise ObservationNormalizationError("normalization requires a CausalActorFrame")
        self._validate_identity(frame, actor_frame)
        if not self._frozen and frame.source.split != "train":
            raise ObservationNormalizationError(
                "validation/test normalization requires frozen training state",
                context={"split": frame.source.split, "trace_id": frame.trace_id},
            )

        usable = np.asarray(actor_frame.usable_mask, dtype=np.bool_)
        raw_valid = np.asarray(
            [row.values for row in actor_frame.rows if row.values is not None],
            dtype=np.float64,
        )
        if raw_valid.size == 0:
            raw_valid = np.empty((0, len(self.columns)), dtype=np.float64)
        if raw_valid.shape != (int(np.count_nonzero(usable)), len(self.columns)):
            raise ObservationNormalizationError(
                "raw valid rows do not match the causal usable mask"
            )
        normalized_valid = self._transform_valid_rows(raw_valid)
        actor_values = np.zeros(
            (len(actor_frame.rows), len(self.columns)),
            dtype=np.float32,
        )
        actor_values[usable] = normalized_valid
        mask_row = np.asarray(self.action_space.mask.values, dtype=np.bool_)
        action_masks = np.tile(mask_row, (len(actor_frame.rows), 1))
        observation = FrameObservation(
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            pair_ids=frame.active_pair_ids,
            actor_observations=actor_values,
            action_masks=action_masks,
        )
        normalized = NormalizedActorFrame(
            split=frame.source.split,
            observation=observation,
            usable_mask=usable,
            statistics_count_before=tuple(int(value) for value in self._count),
        )
        self._pending = _PendingNormalizationFrame(
            normalized=normalized,
            raw_valid_rows=raw_valid.copy(order="C"),
        )
        return normalized

    def transform_final_observations(
        self,
        next_frame: PopulationFrame,
        actor_frame: CausalActorFrame,
    ) -> NormalizedActorFrame:
        """Normalize internal-truncation rows without updating state.

        These rows are critic-only observations at the next physical instant;
        they are not actions and therefore must neither open a normalization
        frame nor become additional training-statistics samples.
        """

        if self._pending is not None:
            raise ObservationNormalizationError(
                "final observations require the acted normalization frame to be complete"
            )
        if not isinstance(next_frame, PopulationFrame):
            raise ObservationNormalizationError(
                "final-observation normalization requires a PopulationFrame"
            )
        if not isinstance(actor_frame, CausalActorFrame):
            raise ObservationNormalizationError(
                "final-observation normalization requires a CausalActorFrame"
            )
        mismatches: dict[str, object] = {}
        if next_frame.trace_id != actor_frame.trace_id:
            mismatches["trace_id"] = (next_frame.trace_id, actor_frame.trace_id)
        if next_frame.index != actor_frame.frame_index:
            mismatches["frame_index"] = (next_frame.index, actor_frame.frame_index)
        if not math.isclose(
            next_frame.time_s,
            actor_frame.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            mismatches["time_s"] = (next_frame.time_s, actor_frame.time_s)
        if actor_frame.schema.columns != self.columns:
            mismatches["columns"] = (actor_frame.schema.columns, self.columns)
        if mismatches:
            raise ObservationNormalizationError(
                "final actor rows do not belong to the next physical frame",
                context=mismatches,
            )
        if not self._frozen and next_frame.source.split != "train":
            raise ObservationNormalizationError(
                "validation/test normalization requires frozen training state",
                context={
                    "split": next_frame.source.split,
                    "trace_id": next_frame.trace_id,
                },
            )

        usable = np.asarray(actor_frame.usable_mask, dtype=np.bool_)
        raw_valid = np.asarray(
            [row.values for row in actor_frame.rows if row.values is not None],
            dtype=np.float64,
        )
        if raw_valid.size == 0:
            raw_valid = np.empty((0, len(self.columns)), dtype=np.float64)
        if raw_valid.shape != (int(np.count_nonzero(usable)), len(self.columns)):
            raise ObservationNormalizationError(
                "raw final rows do not match the causal usable mask"
            )
        actor_values = np.zeros(
            (len(actor_frame.rows), len(self.columns)),
            dtype=np.float32,
        )
        actor_values[usable] = self._transform_valid_rows(raw_valid)
        mask_row = np.asarray(self.action_space.mask.values, dtype=np.bool_)
        observation = FrameObservation(
            trace_id=actor_frame.trace_id,
            frame_index=actor_frame.frame_index,
            time_s=actor_frame.time_s,
            pair_ids=actor_frame.pair_ids,
            actor_observations=actor_values,
            action_masks=np.tile(mask_row, (len(actor_frame.rows), 1)),
        )
        return NormalizedActorFrame(
            split=next_frame.source.split,
            observation=observation,
            usable_mask=usable,
            statistics_count_before=tuple(int(value) for value in self._count),
        )

    def _batch_update(self, raw_rows: Float64Array) -> None:
        if raw_rows.shape[0] == 0:
            return
        mask = np.asarray(self.standardized, dtype=np.bool_)
        batch = raw_rows[:, mask]
        batch_count = batch.shape[0]
        batch_mean = np.mean(batch, axis=0, dtype=np.float64)
        differences = batch - batch_mean
        batch_second_moment = np.sum(
            differences * differences,
            axis=0,
            dtype=np.float64,
        )
        prior_count = self._count[mask].astype(np.float64)
        total_count = prior_count + batch_count
        delta = batch_mean - self._mean[mask]
        self._mean[mask] += delta * batch_count / total_count
        self._second_moment[mask] += (
            batch_second_moment + delta * delta * prior_count * batch_count / total_count
        )
        self._count[mask] += batch_count

    def complete_frame(
        self,
        normalized_frame: NormalizedActorFrame,
        actions: ActionArray,
    ) -> ObservationNormalizationState:
        """Validate selected actions, then update once from valid training rows."""

        pending = self._pending
        if pending is None:
            raise ObservationNormalizationError("no normalization frame is open")
        if normalized_frame is not pending.normalized:
            raise ObservationNormalizationError(
                "normalization completion does not match the open frame"
            )
        validated = self.api_schema.validate_actions(
            normalized_frame.observation,
            actions,
        )
        fallback_index = int(self.action_space.fallback_action)
        wrong_fallback = np.flatnonzero(
            (~normalized_frame.usable_mask) & (validated != fallback_index)
        )
        if wrong_fallback.size:
            rows = tuple(int(value) for value in wrong_fallback)
            raise ObservationNormalizationError(
                "unusable observations must select the configured fallback action",
                context={
                    "rows": rows,
                    "pair_ids": tuple(normalized_frame.observation.pair_ids[row] for row in rows),
                    "fallback_action": self.action_space.fallback_action.label,
                },
            )
        if not self._frozen:
            if normalized_frame.split != "train":
                raise ObservationNormalizationError(
                    "only training frames may update normalization state"
                )
            self._batch_update(pending.raw_valid_rows)
        self._pending = None
        return self.snapshot()


__all__ = [
    "NORMALIZATION_STATE_SCHEMA",
    "PASSTHROUGH_COLUMNS",
    "NormalizedActorFrame",
    "ObservationNormalizationError",
    "ObservationNormalizationState",
    "ObservationNormalizer",
]
