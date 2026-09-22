"""Train-only, frame-frozen observation-normalization invariants."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.mean_field.actor_observations import (
    CausalActorFrame,
    CausalActorRow,
)
from hybrid_v2x_rl.mean_field.congestion_feedback import (
    ActorObservationSchema,
    MeanFieldSignal,
)
from hybrid_v2x_rl.mean_field.critic_observations import CentralizedCriticBuilder
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
    TraceSplit,
)
from hybrid_v2x_rl.mean_field.normalization import (
    NORMALIZATION_STATE_SCHEMA,
    PASSTHROUGH_COLUMNS,
    ObservationNormalizationError,
    ObservationNormalizationState,
    ObservationNormalizer,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord
from hybrid_v2x_rl.observation.builder import ObservationBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAIR_ENDPOINTS = {
    "pair-a": ("veh-1", "veh-2"),
    "pair-b": ("veh-3", "veh-4"),
}


@pytest.fixture(scope="module")
def config():
    return load_headline_config(PROJECT_ROOT)


def _trace_id(split: TraceSplit) -> str:
    return f"synthetic-d20-{split}-900"


def _vehicle(trace_id: str, time_s: float, vehicle_id: str) -> VehicleTraceRecord:
    index = int(vehicle_id.removeprefix("veh-"))
    return VehicleTraceRecord(
        trace_id=trace_id,
        time_s=time_s,
        vehicle_id=vehicle_id,
        x_m=10.0 * index,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=5.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


def _population_frame(
    *,
    split: TraceSplit = "train",
    pair_ids: tuple[str, ...] = ("pair-a", "pair-b"),
    index: int = 3,
) -> PopulationFrame:
    trace_id = _trace_id(split)
    time_s = 0.1 * index
    endpoint_ids = tuple(
        sorted({endpoint for pair_id in pair_ids for endpoint in PAIR_ENDPOINTS[pair_id]})
    )
    vehicles = {vehicle_id: _vehicle(trace_id, time_s, vehicle_id) for vehicle_id in endpoint_ids}
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=index,
            transmitter=vehicles[PAIR_ENDPOINTS[pair_id][0]],
            receiver=vehicles[PAIR_ENDPOINTS[pair_id][1]],
            lifecycle=PairLifecycle(born=False),
        )
        for pair_id in pair_ids
    )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path("/tmp") / trace_id,
            trace_id=trace_id,
            split=split,
            density=20.0,
            replicate=900,
        ),
        index=index,
        time_s=time_s,
        vehicles=tuple(vehicles[vehicle_id] for vehicle_id in endpoint_ids),
        pairs=pairs,
    )


def _schema(config) -> ActorObservationSchema:
    return ActorObservationSchema(local=ObservationBuilder.from_config(config.observation).schema)


def _row_values(config, offset: float) -> tuple[float, ...]:
    columns = _schema(config).columns
    values = np.arange(len(columns), dtype=np.float64) + offset
    values[columns.index("path_spans_junction")] = float(int(offset) % 2)
    values[columns.index("previous_action")] = -1.0 if offset < 1.0 else 3.0
    values[columns.index("last_delivery_outcome")] = -1.0 if offset < 1.0 else 1.0
    values[columns.index("delayed_mean_rf_attempt_fraction")] = 0.25
    values[columns.index("mean_field_valid")] = 1.0
    return tuple(float(value) for value in values)


def _actor_frame(
    config,
    frame: PopulationFrame,
    values: tuple[tuple[float, ...] | None, ...],
) -> CausalActorFrame:
    signal = MeanFieldSignal(
        mean_rf_attempt_fraction=0.25,
        valid=True,
        source_trace_id=frame.trace_id,
        source_frame_index=max(0, frame.index - 1),
    )
    return CausalActorFrame(
        trace_id=frame.trace_id,
        frame_index=frame.index,
        time_s=frame.time_s,
        schema=_schema(config),
        signal=signal,
        rows=tuple(
            CausalActorRow(pair_id=pair_id, values=row)
            for pair_id, row in zip(frame.active_pair_ids, values, strict=True)
        ),
    )


def _actions(*indices: int) -> np.ndarray:
    return np.asarray(indices, dtype=np.int64)


def test_schema_excludes_only_the_four_encoded_contract_columns(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    excluded = tuple(
        column
        for column, standardized in zip(
            normalizer.columns,
            normalizer.standardized,
            strict=True,
        )
        if not standardized
    )

    assert excluded == PASSTHROUGH_COLUMNS
    assert len(normalizer.columns) == 37
    assert normalizer.epsilon == config.environment.normalization.epsilon
    assert normalizer.clip_abs == config.environment.normalization.clip_abs


def test_cold_frame_uses_default_statistics_and_updates_only_after_actions(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    frame = _population_frame()
    raw = (_row_values(config, 0.0), _row_values(config, 2.0))
    actor = _actor_frame(config, frame, raw)

    normalized = normalizer.begin_frame(frame, actor)

    assert normalizer.training_rows == 0
    assert normalized.statistics_count_before == (0,) * 37
    for column_index, is_standardized in enumerate(normalizer.standardized):
        expected = np.asarray(raw, dtype=np.float64)[:, column_index]
        if is_standardized:
            expected = expected / math.sqrt(1.0 + normalizer.epsilon)
            expected = np.clip(expected, -normalizer.clip_abs, normalizer.clip_abs)
        assert normalized.observation.actor_observations[:, column_index] == pytest.approx(
            expected.astype(np.float32)
        )
    with pytest.raises(
        ObservationNormalizationError,
        match="cannot snapshot normalization while a decision frame is open",
    ):
        normalizer.snapshot()

    state = normalizer.complete_frame(normalized, _actions(0, 1))

    assert normalizer.training_rows == 2
    for index, is_standardized in enumerate(normalizer.standardized):
        if is_standardized:
            assert state.count[index] == 2
            assert state.mean[index] == pytest.approx(np.mean(np.asarray(raw)[:, index]))
        else:
            assert state.count[index] == 0
            assert state.mean[index] == 0.0
            assert state.second_moment[index] == 0.0


def test_every_row_in_a_frame_uses_the_same_historical_snapshot(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    first = _population_frame(index=3)
    initial_raw = (_row_values(config, 1.0), _row_values(config, 3.0))
    initial = normalizer.begin_frame(
        first,
        _actor_frame(config, first, initial_raw),
    )
    normalizer.complete_frame(initial, _actions(0, 0))

    second = _population_frame(index=4)
    current_raw = (_row_values(config, 5.0), _row_values(config, 9.0))
    current = normalizer.begin_frame(
        second,
        _actor_frame(config, second, current_raw),
    )
    column = normalizer.columns.index("pair_distance")
    historical_values = np.asarray(initial_raw, dtype=np.float64)[:, column]
    expected_mean = float(np.mean(historical_values))
    expected_variance = float(np.var(historical_values, ddof=1))
    expected = (np.asarray(current_raw, dtype=np.float64)[:, column] - expected_mean) / math.sqrt(
        expected_variance + normalizer.epsilon
    )

    assert current.statistics_count_before[column] == 2
    assert current.observation.actor_observations[:, column] == pytest.approx(
        np.clip(expected, -10.0, 10.0).astype(np.float32)
    )
    assert normalizer.training_rows == 2

    completed = normalizer.complete_frame(current, _actions(1, 2))
    all_values = np.asarray((*initial_raw, *current_raw), dtype=np.float64)[:, column]
    assert completed.count[column] == 4
    assert completed.mean[column] == pytest.approx(float(np.mean(all_values)))
    assert completed.second_moment[column] == pytest.approx(
        float(np.sum((all_values - np.mean(all_values)) ** 2))
    )


def test_missing_sentinels_and_history_padding_are_training_samples(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    frame = _population_frame()
    columns = normalizer.columns
    rows = [list(_row_values(config, offset)) for offset in (0.0, 2.0)]
    age_index = columns.index("rf_quality_age")
    history_indices = tuple(
        index
        for index, column in enumerate(columns)
        if column.startswith("rf_quality_history[")
    )
    for row in rows:
        row[age_index] = -1.0
        for index in history_indices:
            row[index] = 0.0
    actor = _actor_frame(
        config,
        frame,
        tuple(tuple(value for value in row) for row in rows),
    )
    normalized = normalizer.begin_frame(frame, actor)

    state = normalizer.complete_frame(normalized, _actions(0, 0))

    assert state.count[age_index] == 2
    assert state.mean[age_index] == -1.0
    for index in history_indices:
        assert state.count[index] == 2
        assert state.mean[index] == 0.0


def test_unusable_rows_force_fallback_and_never_update_statistics(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    frame = _population_frame()
    actor = _actor_frame(config, frame, (None, _row_values(config, 0.0)))
    normalized = normalizer.begin_frame(frame, actor)

    assert normalized.learn_mask.tolist() == [False, True]
    assert np.all(normalized.observation.actor_observations[0] == 0.0)
    with pytest.raises(
        ObservationNormalizationError,
        match="must select the configured fallback action",
    ):
        normalizer.complete_frame(normalized, _actions(0, 0))
    assert normalizer.training_rows == 0

    state = normalizer.complete_frame(normalized, _actions(8, 0))
    assert normalizer.training_rows == 1
    assert set(state.count) == {0, 1}


def test_validation_and_test_cannot_influence_frozen_training_state(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    validation = _population_frame(split="validation")
    validation_actor = _actor_frame(
        config,
        validation,
        (_row_values(config, 100.0), _row_values(config, 200.0)),
    )
    with pytest.raises(
        ObservationNormalizationError,
        match="requires frozen training state",
    ):
        normalizer.begin_frame(validation, validation_actor)

    train = _population_frame()
    train_frame = normalizer.begin_frame(
        train,
        _actor_frame(
            config,
            train,
            (_row_values(config, 0.0), _row_values(config, 2.0)),
        ),
    )
    normalizer.complete_frame(train_frame, _actions(0, 0))
    frozen = normalizer.freeze()

    for split in ("validation", "test"):
        evaluation = _population_frame(split=split)
        evaluation_frame = normalizer.begin_frame(
            evaluation,
            _actor_frame(
                config,
                evaluation,
                (_row_values(config, 1000.0), _row_values(config, 2000.0)),
            ),
        )
        normalizer.complete_frame(evaluation_frame, _actions(0, 0))

    assert normalizer.snapshot() == frozen
    assert normalizer.frozen


def test_checkpoint_round_trip_preserves_transform_and_frozen_mode(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    train = _population_frame()
    normalized = normalizer.begin_frame(
        train,
        _actor_frame(
            config,
            train,
            (_row_values(config, 0.0), _row_values(config, 2.0)),
        ),
    )
    normalizer.complete_frame(normalized, _actions(0, 0))
    expected_state = normalizer.freeze()
    payload = json.loads(json.dumps(dict(normalizer.state_dict())))

    restored = ObservationNormalizer.from_state_dict(config, payload)

    assert restored.snapshot() == expected_state
    evaluation = _population_frame(split="test")
    actor = _actor_frame(
        config,
        evaluation,
        (_row_values(config, 4.0), _row_values(config, 6.0)),
    )
    original_frame = normalizer.begin_frame(evaluation, actor)
    restored_frame = restored.begin_frame(evaluation, actor)
    assert np.array_equal(
        original_frame.observation.actor_observations,
        restored_frame.observation.actor_observations,
    )
    normalizer.complete_frame(original_frame, _actions(0, 0))
    restored.complete_frame(restored_frame, _actions(0, 0))
    assert restored.snapshot() == expected_state


def test_checkpoint_metadata_drift_and_corruption_fail_closed(config) -> None:
    state = ObservationNormalizer.from_config(config).snapshot().as_dict()
    changed_columns = dict(state)
    changed_columns["columns"] = [*state["columns"][:-1], "wrong-column"]
    with pytest.raises(
        ObservationNormalizationError,
        match="does not match resolved configuration",
    ):
        ObservationNormalizer.from_state_dict(config, changed_columns)

    corrupt = dict(state)
    corrupt["second_moment"] = [-1.0] * 37
    with pytest.raises(
        ObservationNormalizationError,
        match="second moments must be non-negative",
    ):
        ObservationNormalizationState.from_dict(corrupt)

    incomplete = dict(state)
    del incomplete["mean"]
    with pytest.raises(
        ObservationNormalizationError,
        match="fields do not match the schema",
    ):
        ObservationNormalizationState.from_dict(incomplete)


def test_freeze_requires_a_closed_frame(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    frame = _population_frame()
    normalized = normalizer.begin_frame(
        frame,
        _actor_frame(
            config,
            frame,
            (_row_values(config, 0.0), _row_values(config, 2.0)),
        ),
    )

    with pytest.raises(
        ObservationNormalizationError,
        match="cannot freeze normalization while a decision frame is open",
    ):
        normalizer.freeze()
    normalizer.complete_frame(normalized, _actions(0, 0))
    assert normalizer.freeze().frozen


def test_normalized_rows_feed_the_existing_centralized_critic(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    frame = _population_frame()
    normalized = normalizer.begin_frame(
        frame,
        _actor_frame(
            config,
            frame,
            (_row_values(config, 0.0), _row_values(config, 2.0)),
        ),
    )

    critic = CentralizedCriticBuilder.from_config(config).build(
        frame,
        normalized.observation,
    )

    assert critic.critic_observations.shape == (2, 78)
    assert np.array_equal(
        critic.critic_observations[:, :37],
        normalized.observation.actor_observations,
    )
    normalizer.complete_frame(normalized, _actions(0, 0))


def test_empty_training_frame_has_contract_shapes_and_no_statistics(config) -> None:
    normalizer = ObservationNormalizer.from_config(config)
    frame = _population_frame(pair_ids=())
    normalized = normalizer.begin_frame(frame, _actor_frame(config, frame, ()))

    assert normalized.observation.actor_observations.shape == (0, 37)
    assert normalized.observation.action_masks.shape == (0, 9)
    state = normalizer.complete_frame(normalized, _actions())
    assert normalizer.training_rows == 0
    assert state.count == (0,) * 37


def test_state_schema_name_is_pinned() -> None:
    assert NORMALIZATION_STATE_SCHEMA.endswith(".v1")
