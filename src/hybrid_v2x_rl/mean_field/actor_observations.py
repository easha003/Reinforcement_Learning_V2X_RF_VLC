"""Causal actor rows for simultaneous variable-population decision frames.

This is the Phase 5 seam between policy-independent trace replay and the
decentralized actor.  It deliberately accepts a :class:`PopulationFrame`, not
channel results or simulator geometry.  The existing perception boundary turns
the current trace frame into noisy, delayed tracks and a 35-column local row;
this module appends only the population signal frozen from the preceding
closed frame.

Action-dependent receiver reports may be recorded only after every row for the
current frame has been materialized.  They therefore affect a later frame, not
the tuple already returned to the policy.  An unavailable tagged-end track is
represented by ``values=None`` rather than a plausible numeric row.  The full
environment can consequently apply its declared fallback action and exclude
that transition from policy learning without training on invented state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, cast, runtime_checkable

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import ActionKey, PolicyAction, action_resources
from hybrid_v2x_rl.env.episodes import PairInstant, VehiclePose
from hybrid_v2x_rl.env.perception import build_perception
from hybrid_v2x_rl.mean_field.congestion_feedback import (
    ActorObservationSchema,
    DelayedCongestionFeedback,
    MeanFieldSignal,
)
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolResponse
from hybrid_v2x_rl.observation.builder import ObservationBuilder


class CausalObservationError(HybridV2XError):
    """The actor-observation timing, identity, or feedback contract was violated."""


@runtime_checkable
class CausalPerception(Protocol):
    """Narrow policy-side interface used by the population assembler."""

    builder: ObservationBuilder

    def reset(self) -> None: ...

    def initialize_pair_history(self, pair_id: str) -> None: ...

    def observe(self, instant: PairInstant) -> tuple[float, ...] | None: ...

    def record_policy_feedback(
        self,
        pair_id: str,
        *,
        action: PolicyAction,
        at_s: float,
        delivered: bool,
        measurements: dict[Link, float] | None = None,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CausalActorRow:
    """One stable pair's actor row, or an explicit unusable observation."""

    pair_id: str
    values: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise CausalObservationError("pair_id must be a non-empty string")
        if self.values is not None and not all(math.isfinite(value) for value in self.values):
            raise CausalObservationError(
                "actor observation must contain only finite values",
                context={"pair_id": self.pair_id},
            )

    @property
    def usable(self) -> bool:
        return self.values is not None


@dataclass(frozen=True, slots=True)
class CausalActorFrame:
    """Immutable pre-action observations in canonical active-pair order."""

    trace_id: str
    frame_index: int
    time_s: float
    schema: ActorObservationSchema
    signal: MeanFieldSignal
    rows: tuple[CausalActorRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise CausalObservationError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise CausalObservationError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise CausalObservationError("time_s must be finite and non-negative")
        if not isinstance(self.schema, ActorObservationSchema):
            raise CausalObservationError("actor frame requires its versioned schema")
        if not isinstance(self.signal, MeanFieldSignal):
            raise CausalObservationError("actor frame requires frozen mean-field state")

        pair_ids = self.pair_ids
        if pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(set(pair_ids)):
            raise CausalObservationError(
                "actor rows must use unique canonical stable-pair order"
            )
        wrong_width = tuple(
            row.pair_id
            for row in self.rows
            if row.values is not None and len(row.values) != self.schema.width
        )
        if wrong_width:
            raise CausalObservationError(
                "actor row width does not match the versioned schema",
                context={"pair_ids": wrong_width, "expected": self.schema.width},
            )
        wrong_suffix = tuple(
            row.pair_id
            for row in self.rows
            if row.values is not None and row.values[-2:] != self.signal.vector
        )
        if wrong_suffix:
            raise CausalObservationError(
                "every usable actor row must carry the same frozen delayed signal",
                context={"pair_ids": wrong_suffix},
            )

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(row.pair_id for row in self.rows)

    @property
    def usable_mask(self) -> tuple[bool, ...]:
        """Pair-aligned policy-control mask; false rows require fallback."""

        return tuple(row.usable for row in self.rows)

    @property
    def unusable_pair_ids(self) -> tuple[str, ...]:
        return tuple(row.pair_id for row in self.rows if not row.usable)

    @property
    def usable_actor_rows(self) -> Mapping[str, tuple[float, ...]]:
        """Stable-ID rows safe to send to the policy; unavailable rows are absent."""

        return MappingProxyType(
            {
                row.pair_id: row.values
                for row in self.rows
                if row.values is not None
            }
        )


@dataclass(slots=True)
class CausalActorObservationAssembler:
    """Enforce observation-before-action-before-feedback frame ordering."""

    perception: CausalPerception
    congestion: DelayedCongestionFeedback
    schema: ActorObservationSchema
    feedback_deadline_s: float
    _trace_id: str | None = field(default=None, init=False, repr=False)
    _expected_frame_index: int | None = field(default=None, init=False, repr=False)
    _open_frame: PopulationFrame | None = field(default=None, init=False, repr=False)
    _recorded_pair_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _current_pair_ids: tuple[str, ...] = field(default=(), init=False, repr=False)
    _retired_pair_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _has_population_frame: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.perception, CausalPerception):
            raise CausalObservationError("assembler requires a causal perception source")
        if not isinstance(self.congestion, DelayedCongestionFeedback):
            raise CausalObservationError("assembler requires delayed congestion feedback")
        if not isinstance(self.schema, ActorObservationSchema):
            raise CausalObservationError("assembler requires an actor observation schema")
        if self.perception.builder.schema != self.schema.local:
            raise CausalObservationError(
                "perception and actor schemas must share the same local columns"
            )
        if (
            not math.isfinite(self.feedback_deadline_s)
            or self.feedback_deadline_s <= 0.0
        ):
            raise CausalObservationError(
                "feedback_deadline_s must be finite and positive"
            )

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        *,
        root_seed: int = 0,
    ) -> CausalActorObservationAssembler:
        """Assemble the causal observation boundary from resolved configuration."""

        if not isinstance(config, ProjectConfig):
            raise CausalObservationError("assembler requires a resolved ProjectConfig")
        if not isinstance(root_seed, int) or isinstance(root_seed, bool) or root_seed < 0:
            raise CausalObservationError("root_seed must be a non-negative integer")
        perception = build_perception(config, root_seed=root_seed)
        schema = ActorObservationSchema(local=perception.builder.schema)
        congestion = DelayedCongestionFeedback.from_config(
            config.environment.mean_field,
            max_rf_attempts=config.environment.max_rf_attempts,
        )
        return cls(
            perception=perception,
            congestion=congestion,
            schema=schema,
            feedback_deadline_s=config.service.deadline_s,
        )

    def reset(self, trace_id: str, *, start_frame_index: int = 0) -> None:
        """Clear track, link, and delayed-population history together."""

        if self._open_frame is not None:
            raise CausalObservationError("cannot reset while a decision frame is open")
        self.congestion.reset(trace_id, start_frame_index=start_frame_index)
        self.perception.reset()
        self._trace_id = trace_id
        self._expected_frame_index = start_frame_index
        self._recorded_pair_ids.clear()
        self._current_pair_ids = ()
        self._retired_pair_ids.clear()
        self._has_population_frame = False

    def _fresh_pair_ids(self, frame: PopulationFrame) -> tuple[str, ...]:
        """Validate entry semantics and identify histories initialized now."""

        current = set(frame.active_pair_ids)
        if not self._has_population_frame:
            # A sampled segment may begin inside an already-running physical
            # pair episode.  Its pre-reset feedback is outside the rollout and
            # must not be reconstructed, so every first-frame pair starts with
            # fresh local history whether or not its physical ``born`` flag is
            # true on this frame.
            return frame.active_pair_ids

        previous = set(self._current_pair_ids)
        entered = tuple(sorted(current - previous))
        declared_births = tuple(
            pair.pair_id for pair in frame.pairs if pair.lifecycle.born
        )
        if entered != declared_births:
            raise CausalObservationError(
                "mid-episode population entries must match declared pair births",
                context={
                    "entered_pair_ids": entered,
                    "declared_birth_pair_ids": declared_births,
                },
            )
        reappeared = tuple(sorted(current & self._retired_pair_ids))
        if reappeared:
            raise CausalObservationError(
                "a retired pair ID cannot reappear without a new episode identity",
                context={"reappeared_pair_ids": reappeared},
            )
        return entered

    def begin_frame(self, frame: PopulationFrame) -> CausalActorFrame:
        """Materialize every pre-action row from current trace and prior feedback."""

        if self._trace_id is None or self._expected_frame_index is None:
            raise CausalObservationError("causal observations must be reset first")
        if self._open_frame is not None:
            raise CausalObservationError(
                "the current decision frame must close before another begins"
            )
        if not isinstance(frame, PopulationFrame):
            raise CausalObservationError("begin_frame requires a PopulationFrame")
        if frame.trace_id != self._trace_id:
            raise CausalObservationError(
                "population frame trace does not match the sampled episode",
                context={"actual": frame.trace_id, "expected": self._trace_id},
            )
        if frame.index != self._expected_frame_index:
            raise CausalObservationError(
                "population frames must arrive without gaps or reordering",
                context={"actual": frame.index, "expected": self._expected_frame_index},
            )

        fresh_pair_ids = self._fresh_pair_ids(frame)
        for pair_id in fresh_pair_ids:
            try:
                self.perception.initialize_pair_history(pair_id)
            except (TypeError, ValueError) as error:
                raise CausalObservationError(
                    "pair birth could not create a fresh link history",
                    context={"pair_id": pair_id},
                ) from error

        local_rows: list[tuple[str, tuple[float, ...] | None]] = []
        for pair in frame.pairs:
            instant = PairInstant(
                trace_id=frame.trace_id,
                pair_id=pair.pair_id,
                index=pair.episode_step,
                time_s=frame.time_s,
                # Both source records satisfy the exact pose interface.  The
                # inherited PairInstant annotation predates trace records and
                # names its concrete streaming implementation.
                transmitter=cast(VehiclePose, pair.transmitter),
                receiver=cast(VehiclePose, pair.receiver),
                neighbours=cast(tuple[VehiclePose, ...], frame.vehicles),
                index_of_frame=frame.spatial_index,
                final=pair.lifecycle.final,
            )
            local = self.perception.observe(instant)
            if local is not None and len(local) != self.schema.local.width:
                raise CausalObservationError(
                    "perception produced a local row with the wrong width",
                    context={
                        "pair_id": pair.pair_id,
                        "actual": len(local),
                        "expected": self.schema.local.width,
                    },
                )
            local_rows.append((pair.pair_id, local))

        signal = self.congestion.begin_frame(frame.trace_id, frame.index)
        rows = tuple(
            CausalActorRow(
                pair_id=pair_id,
                values=(
                    None
                    if local is None
                    else self.congestion.actor_observation(self.schema, local)
                ),
            )
            for pair_id, local in local_rows
        )
        actor_frame = CausalActorFrame(
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            schema=self.schema,
            signal=signal,
            rows=rows,
        )
        if actor_frame.pair_ids != frame.active_pair_ids:
            raise CausalObservationError(
                "causal actor rows lost population-frame identity alignment"
            )
        previous = set(self._current_pair_ids) if self._has_population_frame else set()
        self._retired_pair_ids.update(previous - set(frame.active_pair_ids))
        self._current_pair_ids = frame.active_pair_ids
        self._has_population_frame = True
        self._open_frame = frame
        self._recorded_pair_ids.clear()
        return actor_frame

    def record_feedback(
        self,
        pair_id: str,
        *,
        action: ActionKey,
        at_s: float,
        delivered: bool,
        measurements: Mapping[Link, float] | None = None,
    ) -> None:
        """Store one completed packet's causal reports for later observations."""

        frame = self._open_frame
        if frame is None:
            raise CausalObservationError("pair feedback requires an open decision frame")
        if pair_id not in set(frame.active_pair_ids):
            raise CausalObservationError(
                "pair feedback does not belong to the open population",
                context={"pair_id": pair_id},
            )
        if pair_id in self._recorded_pair_ids:
            raise CausalObservationError(
                "pair feedback may be recorded only once per decision frame",
                context={"pair_id": pair_id},
            )
        if not math.isfinite(at_s) or at_s < frame.time_s:
            raise CausalObservationError(
                "feedback availability time cannot precede its decision frame",
                context={"pair_id": pair_id, "at_s": at_s, "frame_time_s": frame.time_s},
            )
        latest_feedback_s = frame.time_s + self.feedback_deadline_s
        if at_s > latest_feedback_s + 1e-12:
            raise CausalObservationError(
                "feedback availability time exceeds the packet deadline",
                context={
                    "pair_id": pair_id,
                    "at_s": at_s,
                    "latest_feedback_s": latest_feedback_s,
                },
            )
        if type(delivered) is not bool:
            raise CausalObservationError("delivered must be boolean")
        spec = action_resources(action)
        reports = dict(measurements or {})
        self.perception.record_policy_feedback(
            pair_id,
            action=spec.action,
            at_s=at_s,
            delivered=delivered,
            measurements=reports,
        )
        self._recorded_pair_ids.add(pair_id)

    def close_frame(self, response: RFPoolResponse) -> None:
        """Close current action feedback and queue its load for the next frame."""

        frame = self._open_frame
        if frame is None:
            raise CausalObservationError("no causal observation frame is open")
        if set(self._recorded_pair_ids) != set(frame.active_pair_ids):
            missing = tuple(
                pair_id
                for pair_id in frame.active_pair_ids
                if pair_id not in self._recorded_pair_ids
            )
            raise CausalObservationError(
                "every active pair requires outcome feedback before frame close",
                context={"missing_pair_ids": missing},
            )
        if not isinstance(response, RFPoolResponse):
            raise CausalObservationError("frame close requires an RFPoolResponse")
        response_ids = tuple(
            pair_id for pair_id, _ in response.demand.reserved_rf_attempts_by_pair
        )
        if response_ids != frame.active_pair_ids:
            raise CausalObservationError(
                "RF-pool response IDs do not match the open population",
                context={"actual": response_ids, "expected": frame.active_pair_ids},
            )
        self.congestion.close_frame(response)
        self._expected_frame_index = frame.index + 1
        self._open_frame = None
        self._recorded_pair_ids.clear()


__all__ = [
    "CausalActorFrame",
    "CausalActorObservationAssembler",
    "CausalActorRow",
    "CausalObservationError",
    "CausalPerception",
]
