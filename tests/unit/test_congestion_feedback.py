"""One-frame-delayed RF congestion at the decentralized actor boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand, headline_parameters
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.mean_field.congestion_feedback import (
    MEAN_FIELD_COLUMNS,
    ActorObservationSchema,
    CongestionFeedbackError,
    DelayedCongestionFeedback,
    MeanFieldSignal,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel, RFPoolResponse
from hybrid_v2x_rl.observation.builder import ObservationBuilder

TRACE_ID = "synthetic-d20-train-000"
OTHER_TRACE_ID = "synthetic-d20-train-001"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _pool_model() -> RFPoolModel:
    return RFPoolModel(
        parameters=headline_parameters(),
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=0.0005,
    )


def _response(
    frame_index: int,
    attempts: tuple[int, ...],
    *,
    trace_id: str = TRACE_ID,
    sensed_fraction: float = 1.0,
) -> RFPoolResponse:
    rows = tuple(
        (f"pair-{index:03d}", reserved)
        for index, reserved in enumerate(attempts)
    )
    demand = RFPoolDemand(
        trace_id=trace_id,
        frame_index=frame_index,
        time_s=0.1 * frame_index,
        active_pairs=len(rows),
        reserved_rf_attempts_by_pair=rows,
        offered_rf_attempts=sum(attempts),
        rf_using_pairs=sum(attempt > 0 for attempt in attempts),
    )
    return _pool_model().evaluate(demand, sensed_fraction=sensed_fraction)


def _feedback() -> DelayedCongestionFeedback:
    config = load_headline_config(PROJECT_ROOT)
    return DelayedCongestionFeedback.from_config(
        config.environment.mean_field,
        max_rf_attempts=config.environment.max_rf_attempts,
    )


def _actor_schema() -> ActorObservationSchema:
    config = load_headline_config(PROJECT_ROOT)
    return ActorObservationSchema(
        local=ObservationBuilder.from_config(config.observation).schema
    )


def _local_observation(schema: ActorObservationSchema) -> tuple[float, ...]:
    return (0.0,) * schema.local.width


def test_actor_schema_appends_only_the_two_frozen_mean_field_columns() -> None:
    schema = _actor_schema()

    assert schema.local.width == 35
    assert schema.columns[-2:] == MEAN_FIELD_COLUMNS
    assert schema.width == 37
    assert schema.columns[35:] == (
        "delayed_mean_rf_attempt_fraction",
        "mean_field_valid",
    )
    assert "offered_rf_attempts" not in schema.columns
    assert "per_attempt_collision_probability" not in schema.columns


def test_reset_is_missing_history_not_a_zero_load_measurement() -> None:
    feedback = _feedback()
    schema = _actor_schema()
    feedback.reset(TRACE_ID)

    signal = feedback.begin_frame(TRACE_ID, 0)
    actor = feedback.actor_observation(schema, _local_observation(schema))

    assert signal == MeanFieldSignal.reset()
    assert signal.vector == (0.0, 0.0)
    assert actor[-2:] == (0.0, 0.0)


def test_current_demand_becomes_visible_exactly_one_frame_later() -> None:
    feedback = _feedback()
    schema = _actor_schema()
    local = _local_observation(schema)
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)

    before_close = feedback.actor_observation(schema, local)
    feedback.close_frame(_response(0, (4, 2, 0)))

    assert before_close[-2:] == (0.0, 0.0)
    assert feedback.visible_signal.vector == (0.0, 0.0)
    with pytest.raises(CongestionFeedbackError, match="before the frame closes"):
        feedback.actor_observation(schema, local)

    next_signal = feedback.begin_frame(TRACE_ID, 1)
    next_actor = feedback.actor_observation(schema, local)
    assert next_signal.mean_rf_attempt_fraction == pytest.approx(6 / 12)
    assert next_signal.valid
    assert next_signal.source_frame_index == 0
    assert next_actor[-2:] == pytest.approx((0.5, 1.0))


def test_closed_frame_signal_can_be_previewed_without_consuming_it() -> None:
    feedback = _feedback()
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    feedback.close_frame(_response(0, (4, 0)))

    preview = feedback.preview_next_frame(TRACE_ID, 1)
    repeated = feedback.preview_next_frame(TRACE_ID, 1)

    assert repeated is preview
    assert preview.vector == pytest.approx((0.5, 1.0))
    assert feedback.begin_frame(TRACE_ID, 1) is preview


def test_next_frame_preview_requires_closed_exact_identity_and_queued_state() -> None:
    feedback = _feedback()
    feedback.reset(TRACE_ID)
    with pytest.raises(CongestionFeedbackError, match="no next-frame feedback"):
        feedback.preview_next_frame(TRACE_ID, 0)
    feedback.begin_frame(TRACE_ID, 0)
    with pytest.raises(CongestionFeedbackError, match="only after the frame closes"):
        feedback.preview_next_frame(TRACE_ID, 1)
    feedback.close_frame(_response(0, (1,)))
    with pytest.raises(CongestionFeedbackError, match="preview trace"):
        feedback.preview_next_frame(OTHER_TRACE_ID, 1)
    with pytest.raises(CongestionFeedbackError, match="expected frame index"):
        feedback.preview_next_frame(TRACE_ID, 2)


def test_same_frame_cbr_and_collision_never_enter_the_delayed_suffix() -> None:
    feedback = _feedback()
    schema = _actor_schema()
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    feedback.close_frame(_response(0, (2, 2)))
    feedback.begin_frame(TRACE_ID, 1)

    local = list(_local_observation(schema))
    local_cbr_index = schema.local.columns.index("rf_channel_busy_ratio")
    local[local_cbr_index] = 0.71
    actor_before_actions = feedback.actor_observation(schema, local)
    current_response = _response(1, (4, 4), sensed_fraction=0.25)

    assert actor_before_actions[local_cbr_index] == pytest.approx(0.71)
    assert actor_before_actions[-2:] == pytest.approx((0.5, 1.0))
    assert current_response.channel_busy_ratio != actor_before_actions[-2]
    assert (
        current_response.per_attempt_collision_probability
        != actor_before_actions[-2]
    )

    feedback.close_frame(current_response)
    assert feedback.visible_signal.mean_rf_attempt_fraction == pytest.approx(0.5)


def test_valid_empty_frame_is_distinct_from_reset() -> None:
    feedback = _feedback()
    schema = _actor_schema()
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    feedback.close_frame(_response(0, ()))

    signal = feedback.begin_frame(TRACE_ID, 1)
    actor = feedback.actor_observation(schema, _local_observation(schema))

    assert signal.vector == (0.0, 1.0)
    assert signal.source_frame_index == 0
    assert actor[-2:] == (0.0, 1.0)


def test_full_population_reservation_maps_to_one() -> None:
    feedback = _feedback()
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    feedback.close_frame(_response(0, (4, 4, 4, 4)))

    signal = feedback.begin_frame(TRACE_ID, 1)

    assert signal.mean_rf_attempt_fraction == pytest.approx(1.0)
    assert signal.valid


def test_every_actor_in_a_frame_receives_the_same_frozen_signal() -> None:
    feedback = _feedback()
    schema = _actor_schema()
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    feedback.close_frame(_response(0, (1, 0)))
    feedback.begin_frame(TRACE_ID, 1)

    first_local = list(_local_observation(schema))
    second_local = list(_local_observation(schema))
    first_local[0] = 0.2
    second_local[0] = 0.9
    first = feedback.actor_observation(schema, first_local)
    second = feedback.actor_observation(schema, second_local)

    assert first[0] != second[0]
    assert first[-2:] == second[-2:] == pytest.approx((0.125, 1.0))


def test_reset_at_a_new_trace_or_sampled_episode_clears_prior_load() -> None:
    feedback = _feedback()
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    feedback.close_frame(_response(0, (4, 4)))

    feedback.reset(OTHER_TRACE_ID, start_frame_index=300)
    signal = feedback.begin_frame(OTHER_TRACE_ID, 300)

    assert signal.vector == (0.0, 0.0)
    assert signal.source_trace_id is None
    assert signal.source_frame_index is None


def test_response_must_match_the_open_trace_and_frame() -> None:
    feedback = _feedback()
    feedback.reset(TRACE_ID, start_frame_index=4)
    feedback.begin_frame(TRACE_ID, 4)

    with pytest.raises(CongestionFeedbackError, match="does not belong"):
        feedback.close_frame(_response(5, (1,)))
    with pytest.raises(CongestionFeedbackError, match="does not belong"):
        feedback.close_frame(_response(4, (1,), trace_id=OTHER_TRACE_ID))

    feedback.close_frame(_response(4, (1,)))
    assert feedback.begin_frame(TRACE_ID, 5).valid


def test_frames_cannot_skip_reorder_or_overlap() -> None:
    feedback = _feedback()
    feedback.reset(TRACE_ID)

    with pytest.raises(CongestionFeedbackError, match="gaps or reordering"):
        feedback.begin_frame(TRACE_ID, 1)
    feedback.begin_frame(TRACE_ID, 0)
    with pytest.raises(CongestionFeedbackError, match="preceding frame"):
        feedback.begin_frame(TRACE_ID, 1)
    with pytest.raises(CongestionFeedbackError, match="while a frame is open"):
        feedback.reset(OTHER_TRACE_ID)


def test_actor_assembly_refuses_width_and_finiteness_drift() -> None:
    schema = _actor_schema()
    signal = MeanFieldSignal.reset()

    with pytest.raises(CongestionFeedbackError, match="width"):
        schema.assemble((0.0,), signal)
    invalid = list(_local_observation(schema))
    invalid[0] = float("nan")
    with pytest.raises(CongestionFeedbackError, match="finite"):
        schema.assemble(invalid, signal)


def test_signal_and_configuration_fail_closed_on_contract_drift() -> None:
    feedback = _feedback()

    with pytest.raises(CongestionFeedbackError, match="source frame"):
        MeanFieldSignal(0.5, True)
    with pytest.raises(CongestionFeedbackError, match="reset encoding"):
        MeanFieldSignal(0.5, False)
    with pytest.raises(CongestionFeedbackError, match="contract 1.0.0"):
        replace(feedback, delay_frames=0)
