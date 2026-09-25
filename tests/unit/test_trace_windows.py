"""Deterministic trace-window schedule contract."""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.agents.trace_windows import (
    TRACE_WINDOW_ITERATION_STRIDE,
    TRACE_WINDOW_ROUND_STRIDE,
    TRACE_WINDOW_SCHEDULE_SCHEMA,
    TraceWindowScheduleError,
    select_trace_window,
)


def test_headline_schedule_spreads_rounds_and_advances_across_iterations() -> None:
    first = select_trace_window(
        available_frames=9_000,
        requested_frames=3,
        completed_iterations=0,
        balanced_round=0,
    )
    middle = select_trace_window(
        available_frames=9_000,
        requested_frames=20,
        completed_iterations=0,
        balanced_round=1,
    )
    resumed = select_trace_window(
        available_frames=9_000,
        requested_frames=3,
        completed_iterations=1,
        balanced_round=0,
    )

    assert first.start_frame_index == 0
    assert first.end_frame_index == 2
    assert middle.start_frame_index == TRACE_WINDOW_ROUND_STRIDE
    assert middle.end_frame_index == TRACE_WINDOW_ROUND_STRIDE + 19
    assert resumed.start_frame_index == TRACE_WINDOW_ITERATION_STRIDE
    assert resumed.end_frame_index == TRACE_WINDOW_ITERATION_STRIDE + 2
    assert first.as_dict()["schema"] == TRACE_WINDOW_SCHEDULE_SCHEMA


def test_iteration_stride_traverses_every_headline_start_before_repeating() -> None:
    for requested_frames in (3, 20):
        span = 9_000 - requested_frames + 1
        assert math.gcd(TRACE_WINDOW_ITERATION_STRIDE, span) == 1
        starts = {
            select_trace_window(
                available_frames=9_000,
                requested_frames=requested_frames,
                completed_iterations=index,
                balanced_round=0,
            ).start_frame_index
            for index in range(span)
        }
        assert len(starts) == span


def test_rotating_replicates_retain_full_round_zero_coverage() -> None:
    for requested_frames in (3, 20):
        span = 9_000 - requested_frames + 1
        assert math.gcd(3 * TRACE_WINDOW_ITERATION_STRIDE, span) == 1
        for replicate_index in range(3):
            starts = {
                select_trace_window(
                    available_frames=9_000,
                    requested_frames=requested_frames,
                    completed_iterations=replicate_index + 3 * appearance,
                    balanced_round=0,
                ).start_frame_index
                for appearance in range(span)
            }
            assert len(starts) == span


def test_headline_schedule_records_wraps() -> None:
    wrapped = select_trace_window(
        available_frames=9_000,
        requested_frames=3,
        completed_iterations=10,
        balanced_round=0,
    )

    assert wrapped.start_frame_index == 972
    assert wrapped.schedule_cycle == 1
    assert wrapped.wrapped


def test_schedule_bounds_short_trace_and_reports_modulo_cycle() -> None:
    full = select_trace_window(
        available_frames=2,
        requested_frames=20,
        completed_iterations=4,
        balanced_round=3,
    )

    assert full.window_frames == 2
    assert full.start_frame_index == 0
    assert full.end_frame_index == 1
    assert full.schedule_cycle == full.schedule_position
    assert full.wrapped


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("available_frames", 0),
        ("requested_frames", 0),
        ("completed_iterations", -1),
        ("balanced_round", True),
    ),
)
def test_schedule_rejects_invalid_coordinates(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "available_frames": 9_000,
        "requested_frames": 3,
        "completed_iterations": 0,
        "balanced_round": 0,
    }
    arguments[field] = value

    with pytest.raises(TraceWindowScheduleError):
        select_trace_window(**arguments)  # type: ignore[arg-type]
