"""Causal pre-action rows for the Phase 5 population environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand, headline_parameters
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.actor_observations import (
    CausalActorObservationAssembler,
    CausalObservationError,
)
from hybrid_v2x_rl.mean_field.congestion_feedback import (
    ActorObservationSchema,
    DelayedCongestionFeedback,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel, RFPoolResponse
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord
from hybrid_v2x_rl.observation.builder import ObservationBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d20-train-000"
SOURCE = FrameTraceSource(
    path=Path("/tmp") / TRACE_ID,
    trace_id=TRACE_ID,
    split="train",
    density=20.0,
    replicate=0,
)
ENDPOINTS = {
    "pair-a": ("veh-1", "veh-2"),
    "pair-b": ("veh-3", "veh-4"),
}


def _vehicle(time_s: float, vehicle_id: str) -> VehicleTraceRecord:
    index = int(vehicle_id.removeprefix("veh-"))
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
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


def _frame(index: int, pair_ids: tuple[str, ...]) -> PopulationFrame:
    time_s = 0.1 * index
    endpoint_ids = tuple(
        sorted({endpoint for pair_id in pair_ids for endpoint in ENDPOINTS[pair_id]})
    )
    vehicles = {
        vehicle_id: _vehicle(time_s, vehicle_id) for vehicle_id in endpoint_ids
    }
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=index,
            transmitter=vehicles[ENDPOINTS[pair_id][0]],
            receiver=vehicles[ENDPOINTS[pair_id][1]],
            lifecycle=PairLifecycle(born=index == 0),
        )
        for pair_id in pair_ids
    )
    return PopulationFrame(
        source=SOURCE,
        index=index,
        time_s=time_s,
        vehicles=tuple(vehicles[vehicle_id] for vehicle_id in sorted(vehicles)),
        pairs=pairs,
    )


def _response(frame: PopulationFrame, attempts: tuple[int, ...]) -> RFPoolResponse:
    rows = tuple(zip(frame.active_pair_ids, attempts, strict=True))
    demand = RFPoolDemand(
        trace_id=frame.trace_id,
        frame_index=frame.index,
        time_s=frame.time_s,
        active_pairs=len(rows),
        reserved_rf_attempts_by_pair=rows,
        offered_rf_attempts=sum(attempts),
        rf_using_pairs=sum(value > 0 for value in attempts),
    )
    model = RFPoolModel(
        parameters=headline_parameters(),
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=0.0005,
    )
    return model.evaluate(demand, sensed_fraction=1.0)


class _PerceptionDouble:
    def __init__(self, builder: ObservationBuilder) -> None:
        self.builder = builder
        self.missing: set[str] = set()
        self.previous: dict[str, tuple[PolicyAction, bool, dict[Link, float]]] = {}
        self.instants = []

    def reset(self) -> None:
        self.previous.clear()
        self.instants.clear()

    def observe(self, instant):
        self.instants.append(instant)
        if instant.pair_id in self.missing:
            return None
        local = [0.0] * self.builder.schema.width
        columns = self.builder.schema.columns
        local[columns.index("pair_distance")] = float(
            int(instant.pair_id.removeprefix("pair-") == "b") + 1
        )
        prior = self.previous.get(instant.pair_id)
        local[columns.index("previous_action")] = -1.0
        local[columns.index("last_delivery_outcome")] = -1.0
        if prior is not None:
            action, delivered, reports = prior
            local[columns.index("previous_action")] = float(action)
            local[columns.index("last_delivery_outcome")] = float(delivered)
            if Link.RF in reports:
                local[columns.index("rf_quality")] = reports[Link.RF]
        return tuple(local)

    def record_policy_feedback(
        self,
        pair_id,
        *,
        action,
        at_s,
        delivered,
        measurements=None,
    ) -> None:
        del at_s
        self.previous[pair_id] = (action, delivered, dict(measurements or {}))


def _assembler() -> tuple[CausalActorObservationAssembler, _PerceptionDouble]:
    config = load_headline_config(PROJECT_ROOT)
    builder = ObservationBuilder.from_config(config.observation)
    perception = _PerceptionDouble(builder)
    congestion = DelayedCongestionFeedback.from_config(
        config.environment.mean_field,
        max_rf_attempts=config.environment.max_rf_attempts,
    )
    assembler = CausalActorObservationAssembler(
        perception=perception,
        congestion=congestion,
        schema=ActorObservationSchema(local=builder.schema),
        feedback_deadline_s=config.service.deadline_s,
    )
    return assembler, perception


def test_population_rows_use_current_trace_state_and_a_frozen_past_signal() -> None:
    assembler, perception = _assembler()
    assembler.reset(TRACE_ID)
    frame = _frame(0, ("pair-a", "pair-b"))

    actor = assembler.begin_frame(frame)

    assert actor.pair_ids == frame.active_pair_ids
    assert actor.usable_mask == (True, True)
    assert tuple(actor.usable_actor_rows) == frame.active_pair_ids
    assert [row.values[-2:] for row in actor.rows if row.values is not None] == [
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert [instant.pair_id for instant in perception.instants] == list(
        frame.active_pair_ids
    )
    assert all(tuple(instant.neighbours) == frame.vehicles for instant in perception.instants)


def test_an_absent_causal_track_is_not_replaced_by_a_plausible_row() -> None:
    assembler, perception = _assembler()
    perception.missing.add("pair-b")
    assembler.reset(TRACE_ID)

    actor = assembler.begin_frame(_frame(0, ("pair-a", "pair-b")))

    assert actor.usable_mask == (True, False)
    assert actor.unusable_pair_ids == ("pair-b",)
    assert tuple(actor.usable_actor_rows) == ("pair-a",)
    assert actor.rows[1].values is None


def test_current_action_feedback_changes_only_the_next_actor_row() -> None:
    assembler, _ = _assembler()
    assembler.reset(TRACE_ID)
    frame_0 = _frame(0, ("pair-a",))
    actor_0 = assembler.begin_frame(frame_0)
    previous_index = actor_0.schema.columns.index("previous_action")
    quality_index = actor_0.schema.columns.index("rf_quality")

    assert actor_0.rows[0].values[previous_index] == -1.0
    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.RF_4,
        at_s=0.001,
        delivered=True,
        measurements={Link.RF: 0.8},
    )
    # The returned tuple is immutable pre-action state, so same-frame feedback
    # cannot rewrite what selected the action.
    assert actor_0.rows[0].values[previous_index] == -1.0
    assembler.close_frame(_response(frame_0, (4,)))

    actor_1 = assembler.begin_frame(_frame(1, ("pair-a",)))
    assert actor_1.rows[0].values[previous_index] == float(PolicyAction.RF_4)
    assert actor_1.rows[0].values[quality_index] == pytest.approx(0.8)
    assert actor_1.rows[0].values[-2:] == (1.0, 1.0)
    assert actor_1.signal.source_frame_index == 0


def test_close_requires_one_feedback_record_and_the_exact_open_population() -> None:
    assembler, _ = _assembler()
    assembler.reset(TRACE_ID)
    frame = _frame(0, ("pair-a", "pair-b"))
    assembler.begin_frame(frame)
    response = _response(frame, (1, 0))

    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.RF_1,
        at_s=0.001,
        delivered=False,
    )
    with pytest.raises(CausalObservationError, match="every active pair"):
        assembler.close_frame(response)
    with pytest.raises(CausalObservationError, match="only once"):
        assembler.record_feedback(
            "pair-a",
            action=PolicyAction.RF_1,
            at_s=0.001,
            delivered=False,
        )
    with pytest.raises(CausalObservationError, match="packet deadline"):
        assembler.record_feedback(
            "pair-b",
            action=PolicyAction.VLC,
            at_s=0.1,
            delivered=True,
        )
    assembler.record_feedback(
        "pair-b",
        action=PolicyAction.VLC,
        at_s=0.002,
        delivered=True,
    )
    assembler.close_frame(response)


def test_empty_frame_advances_and_queues_a_valid_zero_load_signal() -> None:
    assembler, _ = _assembler()
    assembler.reset(TRACE_ID)
    empty = _frame(0, ())
    actor_0 = assembler.begin_frame(empty)
    assert actor_0.rows == ()
    assembler.close_frame(_response(empty, ()))

    actor_1 = assembler.begin_frame(_frame(1, ("pair-a",)))
    assert actor_1.signal.vector == (0.0, 1.0)
    assert actor_1.rows[0].values[-2:] == (0.0, 1.0)


def test_reset_clears_both_pair_history_and_population_history() -> None:
    assembler, _ = _assembler()
    assembler.reset(TRACE_ID)
    frame = _frame(0, ("pair-a",))
    assembler.begin_frame(frame)
    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.DUP_3,
        at_s=0.001,
        delivered=True,
        measurements={Link.RF: 0.7, Link.VLC: 0.6},
    )
    assembler.close_frame(_response(frame, (3,)))

    assembler.reset(TRACE_ID)
    fresh = assembler.begin_frame(frame)
    previous_index = fresh.schema.columns.index("previous_action")
    assert fresh.rows[0].values[previous_index] == -1.0
    assert fresh.signal.vector == (0.0, 0.0)


def test_real_perception_records_the_exact_nine_action_index() -> None:
    config = load_headline_config(PROJECT_ROOT)
    assembler = CausalActorObservationAssembler.from_config(config, root_seed=11)
    assembler.reset(TRACE_ID)
    perception = assembler.perception

    perception.record_policy_feedback(
        "pair-a",
        action=PolicyAction.DUP_3,
        at_s=0.1,
        delivered=True,
        measurements={Link.RF: 0.7, Link.VLC: 0.6},
    )

    tracker = perception._links["pair-a"]
    assert tracker.previous_action is PolicyAction.DUP_3
    assert tracker.rf.latest.value == pytest.approx(0.7)
    assert tracker.vlc.latest.value == pytest.approx(0.6)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        perception.record_policy_feedback(
            "pair-b",
            action=PolicyAction.RF_1,
            at_s=0.1,
            delivered=True,
            measurements={Link.RF: 20.0},
        )


def test_configured_assembler_runs_the_real_causal_perception_chain() -> None:
    config = load_headline_config(PROJECT_ROOT)
    at_trace_start = CausalActorObservationAssembler.from_config(config, root_seed=17)
    at_trace_start.reset(TRACE_ID)

    unavailable = at_trace_start.begin_frame(_frame(0, ("pair-a",)))
    assert unavailable.usable_mask == (False,)
    assert unavailable.rows[0].values is None

    assembler = CausalActorObservationAssembler.from_config(config, root_seed=17)
    assembler.reset(TRACE_ID, start_frame_index=1)

    actor = assembler.begin_frame(_frame(1, ("pair-a",)))

    assert actor.usable_mask == (True,)
    assert actor.rows[0].values is not None
    assert len(actor.rows[0].values) == 37
    previous_index = actor.schema.columns.index("previous_action")
    assert actor.rows[0].values[previous_index] == -1.0
    assert actor.rows[0].values[-2:] == (0.0, 0.0)
