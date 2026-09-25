"""Deterministic temporal-window schedule for trace-backed PPO collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from hybrid_v2x_rl.core.errors import HybridV2XError

TRACE_WINDOW_SCHEDULE_SCHEMA: Final = "hybrid-rf-vlc-rl.trace-window-schedule.v1"
TRACE_WINDOW_ITERATION_STRIDE: Final = 997
TRACE_WINDOW_ROUND_STRIDE: Final = 4099


class TraceWindowScheduleError(HybridV2XError):
    """A temporal trace window cannot be selected from the declared schedule."""


@dataclass(frozen=True, slots=True)
class TraceWindowSelection:
    """One reproducible bounded window within a physical mobility trace."""

    available_frames: int
    requested_frames: int
    window_frames: int
    start_frame_index: int
    end_frame_index: int
    schedule_position: int
    schedule_cycle: int
    wrapped: bool

    def __post_init__(self) -> None:
        for name in (
            "available_frames",
            "requested_frames",
            "window_frames",
            "start_frame_index",
            "end_frame_index",
            "schedule_position",
            "schedule_cycle",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TraceWindowScheduleError(f"trace-window {name} must be a nonnegative integer")
        if self.available_frames <= 0 or self.requested_frames <= 0:
            raise TraceWindowScheduleError(
                "trace-window available and requested frame counts must be positive"
            )
        if self.window_frames != min(self.available_frames, self.requested_frames):
            raise TraceWindowScheduleError(
                "trace-window size must be the bounded requested frame count"
            )
        span = self.available_frames - self.window_frames + 1
        if not 0 <= self.start_frame_index < span:
            raise TraceWindowScheduleError("trace-window start lies outside its valid span")
        if self.end_frame_index != self.start_frame_index + self.window_frames - 1:
            raise TraceWindowScheduleError("trace-window end does not match its start and size")
        if self.schedule_cycle != self.schedule_position // span:
            raise TraceWindowScheduleError("trace-window cycle is inconsistent")
        if type(self.wrapped) is not bool or self.wrapped != (self.schedule_cycle > 0):
            raise TraceWindowScheduleError("trace-window wrap flag is inconsistent")

    def as_dict(self) -> dict[str, object]:
        """Return the versioned primitive report representation."""

        return {
            "schema": TRACE_WINDOW_SCHEDULE_SCHEMA,
            "available_frames": self.available_frames,
            "requested_frames": self.requested_frames,
            "window_frames": self.window_frames,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "schedule_position": self.schedule_position,
            "schedule_cycle": self.schedule_cycle,
            "wrapped": self.wrapped,
        }


def select_trace_window(
    *,
    available_frames: int,
    requested_frames: int,
    completed_iterations: int,
    balanced_round: int,
) -> TraceWindowSelection:
    """Select one window from checkpointed counters and the local round index.

    The two prime strides are fixed public experiment constants. The iteration
    stride is coprime with both valid headline spans (8,998 for a three-frame
    window and 8,981 for a twenty-frame window), so repeated iterations traverse
    every possible start before repeating when the local round is fixed. The
    round stride spreads the windows within one density-balanced update.
    """

    for name, value in (
        ("available_frames", available_frames),
        ("requested_frames", requested_frames),
        ("completed_iterations", completed_iterations),
        ("balanced_round", balanced_round),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TraceWindowScheduleError(f"{name} must be a nonnegative integer")
    if available_frames <= 0 or requested_frames <= 0:
        raise TraceWindowScheduleError("available_frames and requested_frames must be positive")

    window_frames = min(available_frames, requested_frames)
    span = available_frames - window_frames + 1
    position = (
        completed_iterations * TRACE_WINDOW_ITERATION_STRIDE
        + balanced_round * TRACE_WINDOW_ROUND_STRIDE
    )
    start = position % span
    return TraceWindowSelection(
        available_frames=available_frames,
        requested_frames=requested_frames,
        window_frames=window_frames,
        start_frame_index=start,
        end_frame_index=start + window_frames - 1,
        schedule_position=position,
        schedule_cycle=position // span,
        wrapped=position >= span,
    )


__all__ = [
    "TRACE_WINDOW_ITERATION_STRIDE",
    "TRACE_WINDOW_ROUND_STRIDE",
    "TRACE_WINDOW_SCHEDULE_SCHEMA",
    "TraceWindowScheduleError",
    "TraceWindowSelection",
    "select_trace_window",
]
