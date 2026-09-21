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
from hybrid_v2x_rl.observation.builder import (
    UNMEASURED_AGE_S,
    UNMEASURED_QUALITY,
    ObservationBuilder,
)

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


def _frame(
    index: int,
    pair_ids: tuple[str, ...],
    *,
    born_pair_ids: frozenset[str] | None = None,
    episode_steps: dict[str, int] | None = None,
    lifecycles: dict[str, PairLifecycle] | None = None,
    extra_vehicle_ids: tuple[str, ...] = (),
) -> PopulationFrame:
    time_s = 0.1 * index
    births = (
        frozenset(pair_ids) if index == 0 and born_pair_ids is None
        else frozenset() if born_pair_ids is None
        else born_pair_ids
    )
    steps = dict(episode_steps or {})
    lifecycle_by_pair = dict(lifecycles or {})
    endpoint_ids = tuple(
        sorted(
            {
                *(endpoint for pair_id in pair_ids for endpoint in ENDPOINTS[pair_id]),
                *extra_vehicle_ids,
            }
        )
    )
    vehicles = {
        vehicle_id: _vehicle(time_s, vehicle_id) for vehicle_id in endpoint_ids
    }
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=steps.get(pair_id, 0 if pair_id in births else index),
            transmitter=vehicles[ENDPOINTS[pair_id][0]],
            receiver=vehicles[ENDPOINTS[pair_id][1]],
            lifecycle=lifecycle_by_pair.get(
                pair_id,
                PairLifecycle(born=pair_id in births),
            ),
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
        self.initialized: set[str] = set()
        self.released: list[str] = []

    def reset(self) -> None:
        self.previous.clear()
        self.instants.clear()
        self.initialized.clear()
        self.released.clear()

    def initialize_pair_history(self, pair_id: str) -> None:
        if pair_id in self.initialized or pair_id in self.previous:
            raise ValueError("pair already has history")
        self.initialized.add(pair_id)

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

    def release(self, pair_id: str) -> None:
        self.released.append(pair_id)
        self.previous.pop(pair_id, None)
        self.initialized.discard(pair_id)

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
    assert perception.initialized == {"pair-a", "pair-b"}


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

    actor_1 = assembler.begin_frame(
        _frame(
            1,
            ("pair-a",),
            born_pair_ids=frozenset({"pair-a"}),
            episode_steps={"pair-a": 0},
        )
    )
    assert actor_1.signal.vector == (0.0, 1.0)
    assert actor_1.rows[0].values[-2:] == (0.0, 1.0)


def test_mid_episode_birth_has_fresh_local_history_and_valid_global_history() -> None:
    config = load_headline_config(PROJECT_ROOT)
    assembler = CausalActorObservationAssembler.from_config(config, root_seed=23)
    assembler.reset(TRACE_ID, start_frame_index=1)
    all_vehicles = ("veh-1", "veh-2", "veh-3", "veh-4")
    frame_1 = _frame(
        1,
        ("pair-a",),
        extra_vehicle_ids=all_vehicles,
    )
    actor_1 = assembler.begin_frame(frame_1)
    assert actor_1.usable_mask == (True,)
    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.RF_4,
        at_s=frame_1.time_s + 0.001,
        delivered=False,
        measurements={Link.RF: 0.8},
    )
    assembler.close_frame(_response(frame_1, (4,)))

    frame_2 = _frame(
        2,
        ("pair-a", "pair-b"),
        born_pair_ids=frozenset({"pair-b"}),
        episode_steps={"pair-a": 2, "pair-b": 0},
        extra_vehicle_ids=all_vehicles,
    )
    actor_2 = assembler.begin_frame(frame_2)
    rows = {
        row.pair_id: row.values
        for row in actor_2.rows
    }
    continuing = rows["pair-a"]
    newborn = rows["pair-b"]
    assert continuing is not None
    assert newborn is not None
    columns = actor_2.schema.columns

    assert continuing[columns.index("previous_action")] == float(
        PolicyAction.RF_4
    )
    assert continuing[columns.index("last_delivery_outcome")] == 0.0
    assert continuing[columns.index("consecutive_miss_count")] == 1.0
    assert continuing[columns.index("rf_quality")] == pytest.approx(0.8)

    assert newborn[columns.index("rf_quality")] == UNMEASURED_QUALITY
    assert newborn[columns.index("vlc_quality")] == UNMEASURED_QUALITY
    assert newborn[columns.index("rf_quality_age")] == UNMEASURED_AGE_S
    assert newborn[columns.index("vlc_quality_age")] == UNMEASURED_AGE_S
    assert newborn[columns.index("previous_action")] == -1.0
    assert newborn[columns.index("last_delivery_outcome")] == -1.0
    assert newborn[columns.index("consecutive_miss_count")] == 0.0
    for prefix in ("rf_quality_history", "vlc_quality_history"):
        history = tuple(
            newborn[index]
            for index, name in enumerate(columns)
            if name.startswith(f"{prefix}[")
        )
        assert history == (UNMEASURED_QUALITY,) * config.observation.history_packets

    # Birth resets only pair-local history. The valid frame-1 population signal
    # is shared by both current actors and remains available to the newborn.
    assert actor_2.signal.source_frame_index == 1
    assert actor_2.signal.vector == (1.0, 1.0)
    assert continuing[-2:] == newborn[-2:] == (1.0, 1.0)
    perception = assembler.perception
    assert perception._links["pair-a"].packets_seen == 1
    assert perception._links["pair-b"].packets_seen == 0


def test_mid_episode_entry_requires_a_declared_birth_before_state_changes() -> None:
    assembler, perception = _assembler()
    assembler.reset(TRACE_ID)
    first = _frame(0, ("pair-a",))
    assembler.begin_frame(first)
    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.VLC,
        at_s=0.001,
        delivered=True,
    )
    assembler.close_frame(_response(first, (0,)))

    undeclared = _frame(1, ("pair-a", "pair-b"))
    with pytest.raises(CausalObservationError, match="declared pair births"):
        assembler.begin_frame(undeclared)
    assert perception.initialized == {"pair-a"}

    declared = _frame(
        1,
        ("pair-a", "pair-b"),
        born_pair_ids=frozenset({"pair-b"}),
        episode_steps={"pair-a": 1, "pair-b": 0},
    )
    actor = assembler.begin_frame(declared)
    assert actor.pair_ids == ("pair-a", "pair-b")
    assert perception.initialized == {"pair-a", "pair-b"}


def test_continuing_pair_cannot_repeat_its_birth_transition() -> None:
    assembler, perception = _assembler()
    assembler.reset(TRACE_ID)
    first = _frame(0, ("pair-a",))
    assembler.begin_frame(first)
    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.VLC,
        at_s=0.001,
        delivered=True,
    )
    assembler.close_frame(_response(first, (0,)))

    repeated_birth = _frame(
        1,
        ("pair-a",),
        born_pair_ids=frozenset({"pair-a"}),
        episode_steps={"pair-a": 0},
    )
    with pytest.raises(CausalObservationError, match="declared pair births"):
        assembler.begin_frame(repeated_birth)
    assert perception.initialized == {"pair-a"}


def test_population_pair_birth_flag_matches_episode_step_zero() -> None:
    with pytest.raises(ValueError, match="episode step zero"):
        _frame(
            2,
            ("pair-b",),
            born_pair_ids=frozenset({"pair-b"}),
            episode_steps={"pair-b": 1},
        )


def test_birth_rejects_preexisting_pair_local_history() -> None:
    assembler, perception = _assembler()
    assembler.reset(TRACE_ID)
    perception.initialize_pair_history("pair-a")

    with pytest.raises(CausalObservationError, match="fresh link history"):
        assembler.begin_frame(_frame(0, ("pair-a",)))


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


def test_final_pair_history_is_released_only_after_its_feedback_is_processed() -> None:
    assembler, perception = _assembler()
    assembler.reset(TRACE_ID, start_frame_index=1)
    frame = _frame(
        1,
        ("pair-a",),
        episode_steps={"pair-a": 1},
        lifecycles={
            "pair-a": PairLifecycle(
                born=False,
                terminated=True,
                end_reason="route_diverged",
            )
        },
    )
    assembler.begin_frame(frame)

    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.RF_1,
        at_s=frame.time_s + 0.001,
        delivered=False,
        measurements={Link.RF: 0.4},
    )
    assert "pair-a" in perception.previous
    assert perception.released == []

    assembler.close_frame(_response(frame, (1,)))

    assert perception.released == ["pair-a"]
    assert "pair-a" not in perception.previous
    assert "pair-a" not in perception.initialized


def test_finalized_pair_cannot_remain_active_on_the_next_frame() -> None:
    assembler, perception = _assembler()
    assembler.reset(TRACE_ID, start_frame_index=1)
    final = _frame(
        1,
        ("pair-a",),
        episode_steps={"pair-a": 1},
        lifecycles={
            "pair-a": PairLifecycle(
                born=False,
                truncated=True,
                bootstrap_valid=True,
                end_reason="max_duration",
            )
        },
    )
    assembler.begin_frame(final)
    assembler.record_feedback(
        "pair-a",
        action=PolicyAction.VLC,
        at_s=final.time_s + 0.001,
        delivered=True,
    )
    assembler.close_frame(_response(final, (0,)))

    illegal = _frame(
        2,
        ("pair-a",),
        episode_steps={"pair-a": 2},
    )
    with pytest.raises(CausalObservationError, match="finalized pair"):
        assembler.begin_frame(illegal)
    assert perception.instants[-1].index == 1


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
