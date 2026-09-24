"""One-frame-delayed population congestion for decentralized actors.

The current frame's RF demand is known only after every active pair has acted.
It therefore cannot appear in any observation used to choose those actions.
This module makes that timing a state machine rather than a convention:

1. :meth:`DelayedCongestionFeedback.begin_frame` promotes the response from the
   preceding frame and freezes it for every actor in the new frame;
2. actors receive only that frozen signal through
   :meth:`DelayedCongestionFeedback.actor_observation`;
3. :meth:`DelayedCongestionFeedback.close_frame` accepts the audited current
   RF-pool response only after observation/action selection and queues it for
   the *next* frame.

Reset is ``[0, 0]``: zero value with an invalid flag.  A genuinely empty frame
queues ``[0, 1]`` instead, preserving the contract's distinction between
missing history and a valid zero-load measurement.  Current collision, CBR,
action histograms, and hidden pool state are never appended to actor input.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import MAX_RESERVED_RF_ATTEMPTS
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolResponse
from hybrid_v2x_rl.observation.builder import ObservationSchema

MEAN_FIELD_COLUMNS: Final = (
    "delayed_mean_rf_attempt_fraction",
    "mean_field_valid",
)


class CongestionFeedbackError(HybridV2XError):
    """Congestion feedback violated the frozen timing or observation contract."""


@runtime_checkable
class MeanFieldConfigSource(Protocol):
    """Configuration fields needed by the delayed-feedback state machine."""

    @property
    def signal(self) -> str: ...

    @property
    def delay_frames(self) -> int: ...

    @property
    def initial_value(self) -> float: ...

    @property
    def include_validity_flag(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class MeanFieldSignal:
    """The population signal visible during one decision frame."""

    mean_rf_attempt_fraction: float
    valid: bool
    source_trace_id: str | None = None
    source_frame_index: int | None = None

    def __post_init__(self) -> None:
        value = self.mean_rf_attempt_fraction
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise CongestionFeedbackError(
                "mean RF-attempt fraction must be finite and lie in [0, 1]",
                context={"mean_rf_attempt_fraction": value},
            )
        if not isinstance(self.valid, bool):
            raise CongestionFeedbackError("mean-field validity must be boolean")
        if self.valid:
            if (
                not isinstance(self.source_trace_id, str)
                or not self.source_trace_id.strip()
                or not isinstance(self.source_frame_index, int)
                or isinstance(self.source_frame_index, bool)
                or self.source_frame_index < 0
            ):
                raise CongestionFeedbackError(
                    "valid mean-field feedback requires a source frame identity"
                )
        elif (
            value != 0.0
            or self.source_trace_id is not None
            or self.source_frame_index is not None
        ):
            raise CongestionFeedbackError(
                "invalid mean-field feedback must use the reset encoding [0, 0]"
            )

    @classmethod
    def reset(cls) -> MeanFieldSignal:
        """Return missing-history encoding, distinct from valid zero load."""

        return cls(mean_rf_attempt_fraction=0.0, valid=False)

    @property
    def vector(self) -> tuple[float, float]:
        return self.mean_rf_attempt_fraction, float(self.valid)

    def as_dict(self) -> dict[str, object]:
        """Return stable diagnostics including the preceding source frame."""

        return {
            MEAN_FIELD_COLUMNS[0]: self.mean_rf_attempt_fraction,
            MEAN_FIELD_COLUMNS[1]: self.valid,
            "source_trace_id": self.source_trace_id,
            "source_frame_index": self.source_frame_index,
        }


@dataclass(frozen=True, slots=True)
class ActorObservationSchema:
    """Local causal columns followed by the two delayed mean-field columns."""

    local: ObservationSchema

    def __post_init__(self) -> None:
        if not isinstance(self.local, ObservationSchema):
            raise CongestionFeedbackError(
                "actor observation schema requires a local observation schema"
            )

    @property
    def columns(self) -> tuple[str, ...]:
        return (*self.local.columns, *MEAN_FIELD_COLUMNS)

    @property
    def width(self) -> int:
        return self.local.width + len(MEAN_FIELD_COLUMNS)

    def assemble(
        self,
        local_observation: Sequence[float],
        signal: MeanFieldSignal,
    ) -> tuple[float, ...]:
        """Append the frozen delayed signal to one validated local vector."""

        if not isinstance(signal, MeanFieldSignal):
            raise CongestionFeedbackError("actor observation requires mean-field state")
        if isinstance(local_observation, str | bytes):
            raise CongestionFeedbackError("local observation must be a numeric sequence")
        local = tuple(float(value) for value in local_observation)
        if len(local) != self.local.width:
            raise CongestionFeedbackError(
                "local observation width does not match its schema",
                context={"actual": len(local), "expected": self.local.width},
            )
        actor = (*local, *signal.vector)
        if len(actor) != self.width or not all(math.isfinite(value) for value in actor):
            raise CongestionFeedbackError(
                "actor observation must have the configured finite width"
            )
        return actor


@dataclass(slots=True)
class DelayedCongestionFeedback:
    """Queue audited frame demand for visibility exactly one frame later."""

    signal_name: str = MEAN_FIELD_COLUMNS[0]
    delay_frames: int = 1
    initial_value: float = 0.0
    include_validity_flag: bool = True
    max_rf_attempts: int = MAX_RESERVED_RF_ATTEMPTS
    _trace_id: str | None = field(default=None, init=False, repr=False)
    _expected_frame_index: int | None = field(default=None, init=False, repr=False)
    _open_frame_index: int | None = field(default=None, init=False, repr=False)
    _visible: MeanFieldSignal = field(
        default_factory=MeanFieldSignal.reset,
        init=False,
        repr=False,
    )
    _pending: MeanFieldSignal | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        failures: dict[str, object] = {}
        if self.signal_name != MEAN_FIELD_COLUMNS[0]:
            failures["signal"] = self.signal_name
        if self.delay_frames != 1:
            failures["delay_frames"] = self.delay_frames
        if self.initial_value != 0.0:
            failures["initial_value"] = self.initial_value
        if self.include_validity_flag is not True:
            failures["include_validity_flag"] = self.include_validity_flag
        if self.max_rf_attempts != MAX_RESERVED_RF_ATTEMPTS:
            failures["max_rf_attempts"] = self.max_rf_attempts
        if failures:
            raise CongestionFeedbackError(
                "mean-field configuration does not match contract 1.0.0",
                context=failures,
            )

    @classmethod
    def from_config(
        cls,
        config: MeanFieldConfigSource,
        *,
        max_rf_attempts: int,
    ) -> DelayedCongestionFeedback:
        """Bind the state machine to validated environment configuration."""

        if not isinstance(config, MeanFieldConfigSource):
            raise CongestionFeedbackError("mean-field configuration is incomplete")
        return cls(
            signal_name=config.signal,
            delay_frames=config.delay_frames,
            initial_value=config.initial_value,
            include_validity_flag=config.include_validity_flag,
            max_rf_attempts=max_rf_attempts,
        )

    @property
    def visible_signal(self) -> MeanFieldSignal:
        """Return the signal frozen for the currently open decision frame."""

        return self._visible

    def reset(self, trace_id: str, *, start_frame_index: int = 0) -> None:
        """Clear history at a trace or sampled-episode boundary."""

        if self._open_frame_index is not None:
            raise CongestionFeedbackError("cannot reset while a frame is open")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise CongestionFeedbackError("trace_id must be a non-empty string")
        if (
            not isinstance(start_frame_index, int)
            or isinstance(start_frame_index, bool)
            or start_frame_index < 0
        ):
            raise CongestionFeedbackError(
                "start_frame_index must be a non-negative integer"
            )
        self._trace_id = trace_id
        self._expected_frame_index = start_frame_index
        self._open_frame_index = None
        self._visible = MeanFieldSignal.reset()
        self._pending = None

    def begin_frame(self, trace_id: str, frame_index: int) -> MeanFieldSignal:
        """Freeze the preceding frame's signal before any current action."""

        if self._trace_id is None or self._expected_frame_index is None:
            raise CongestionFeedbackError("mean-field feedback must be reset first")
        if self._open_frame_index is not None:
            raise CongestionFeedbackError(
                "the preceding frame must close before another frame begins"
            )
        if trace_id != self._trace_id:
            raise CongestionFeedbackError(
                "frame trace does not match the active mean-field episode",
                context={"actual": trace_id, "expected": self._trace_id},
            )
        if frame_index != self._expected_frame_index:
            raise CongestionFeedbackError(
                "mean-field frames must be processed without gaps or reordering",
                context={"actual": frame_index, "expected": self._expected_frame_index},
            )
        if self._pending is not None:
            self._visible = self._pending
            self._pending = None
        self._open_frame_index = frame_index
        return self._visible

    def actor_observation(
        self,
        schema: ActorObservationSchema,
        local_observation: Sequence[float],
    ) -> tuple[float, ...]:
        """Append only the signal frozen when the current frame began."""

        if self._open_frame_index is None:
            raise CongestionFeedbackError(
                "actor observations are available only before the frame closes"
            )
        if not isinstance(schema, ActorObservationSchema):
            raise CongestionFeedbackError("actor observation schema is invalid")
        return schema.assemble(local_observation, self._visible)

    def preview_next_frame(
        self,
        trace_id: str,
        frame_index: int,
    ) -> MeanFieldSignal:
        """Return queued next-frame feedback without consuming it.

        Internal time-limit truncations need a final physical observation after
        current packet feedback has closed.  This read-only view gives that
        observation the same delayed signal the ordinary next population will
        receive, while leaving :meth:`begin_frame` as the sole consumer.
        """

        if self._trace_id is None or self._expected_frame_index is None:
            raise CongestionFeedbackError("mean-field feedback must be reset first")
        if self._open_frame_index is not None:
            raise CongestionFeedbackError(
                "next-frame feedback is available only after the frame closes"
            )
        if trace_id != self._trace_id:
            raise CongestionFeedbackError(
                "preview trace does not match the active mean-field episode",
                context={"actual": trace_id, "expected": self._trace_id},
            )
        if frame_index != self._expected_frame_index:
            raise CongestionFeedbackError(
                "next-frame feedback preview must use the expected frame index",
                context={"actual": frame_index, "expected": self._expected_frame_index},
            )
        if self._pending is None:
            raise CongestionFeedbackError("no next-frame feedback is queued")
        return self._pending

    def close_frame(self, response: RFPoolResponse) -> None:
        """Queue current audited demand for the next decision frame."""

        if self._trace_id is None or self._open_frame_index is None:
            raise CongestionFeedbackError("no decision frame is open")
        if not isinstance(response, RFPoolResponse):
            raise CongestionFeedbackError(
                "closing congestion feedback requires an RF-pool response"
            )
        demand = response.demand
        if (
            demand.trace_id != self._trace_id
            or demand.frame_index != self._open_frame_index
        ):
            raise CongestionFeedbackError(
                "RF-pool response does not belong to the open decision frame",
                context={
                    "response": (demand.trace_id, demand.frame_index),
                    "open": (self._trace_id, self._open_frame_index),
                },
            )

        fraction = (
            demand.offered_rf_attempts
            / (self.max_rf_attempts * demand.active_pairs)
            if demand.active_pairs > 0
            else 0.0
        )
        pending = MeanFieldSignal(
            mean_rf_attempt_fraction=fraction,
            valid=True,
            source_trace_id=demand.trace_id,
            source_frame_index=demand.frame_index,
        )
        self._pending = pending
        self._expected_frame_index = self._open_frame_index + 1
        self._open_frame_index = None


__all__ = [
    "MEAN_FIELD_COLUMNS",
    "ActorObservationSchema",
    "CongestionFeedbackError",
    "DelayedCongestionFeedback",
    "MeanFieldConfigSource",
    "MeanFieldSignal",
]
