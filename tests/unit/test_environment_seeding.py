"""One reproducible seed authority for the Phase 5 population environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import FailureCause, Link
from hybrid_v2x_rl.core.policy_actions import PolicyAction
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.packet_outcomes import PairPacketOutcome
from hybrid_v2x_rl.mean_field.seeding import (
    ENVIRONMENT_SEED_SCHEMA,
    RUNTIME_RANDOM_COMPONENTS,
    EnvironmentSeedError,
    EnvironmentSeedState,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d20-train-000"


def _vehicle(vehicle_id: str) -> VehicleTraceRecord:
    index = int(vehicle_id.removeprefix("veh-"))
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=0.1,
        vehicle_id=vehicle_id,
        x_m=10.0 * index,
        y_m=float(index % 2),
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


def _frame() -> PopulationFrame:
    vehicles = tuple(_vehicle(f"veh-{index}") for index in range(1, 5))
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    pairs = (
        PopulationPair(
            pair_id="pair-a",
            episode_step=1,
            transmitter=by_id["veh-1"],
            receiver=by_id["veh-2"],
            lifecycle=PairLifecycle(born=False),
        ),
        PopulationPair(
            pair_id="pair-b",
            episode_step=1,
            transmitter=by_id["veh-3"],
            receiver=by_id["veh-4"],
            lifecycle=PairLifecycle(born=False),
        ),
    )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="train",
            density=20.0,
            replicate=0,
        ),
        index=1,
        time_s=0.1,
        vehicles=vehicles,
        pairs=pairs,
    )


def _vlc_outcome(pair_id: str) -> PairPacketOutcome:
    result = VLCChannelResult(
        received_power_w=1e-6,
        electrical_snr=100.0,
        within_field_of_view=True,
        occluded=False,
        bit_error_rate=0.01,
        decoding_failure_probability=0.2,
        total_failure_probability=0.2,
        success=True,
        failure_cause=FailureCause.NONE,
        beam_aimed=True,
    )
    return PairPacketOutcome(
        pair_id=pair_id,
        action=PolicyAction.VLC,
        reward=-1.0,
        delivered=True,
        sampled_miss_cost=0,
        conditional_miss_probability=0.2,
        failure_cause=FailureCause.NONE,
        rf_delivered=False,
        vlc_delivered=True,
        rf_attempts=(),
        rf_attempt_risk=None,
        rf_packet_miss_probability=None,
        vlc_result=result,
        vlc_miss_probability=0.2,
    )


def test_configured_seed_is_default_and_reset_seed_is_an_explicit_override() -> None:
    config = load_headline_config(PROJECT_ROOT)

    configured = EnvironmentSeedState.from_config(config)
    overridden = EnvironmentSeedState.from_config(config, reset_seed=17)

    assert configured.configured_root_seed == config.training.root_seed
    assert configured.active_root_seed == config.training.root_seed
    assert not configured.explicit_reset_seed
    assert overridden.configured_root_seed == config.training.root_seed
    assert overridden.active_root_seed == 17
    assert overridden.explicit_reset_seed


def test_same_reset_seed_reproduces_sensing_tapes_and_feedback() -> None:
    config = load_headline_config(PROJECT_ROOT)
    frame = _frame()

    first = EnvironmentSeedState.from_config(config, reset_seed=91).for_trace(TRACE_ID)
    repeated = EnvironmentSeedState.from_config(config, reset_seed=91).for_trace(TRACE_ID)
    first_actor = first.actor_assembler(config, start_frame_index=1).begin_frame(frame)
    repeated_actor = repeated.actor_assembler(config, start_frame_index=1).begin_frame(frame)
    first_tapes = first.packet_tapes(frame)
    repeated_tapes = repeated.packet_tapes(frame)

    assert first_actor == repeated_actor
    assert first_tapes == repeated_tapes
    outcome = _vlc_outcome("pair-a")
    assert first.feedback_measurements(outcome, first_tapes["pair-a"]) == (
        repeated.feedback_measurements(outcome, repeated_tapes["pair-a"])
    )


def test_explicit_reset_seed_changes_all_runtime_random_paths() -> None:
    config = load_headline_config(PROJECT_ROOT)
    frame = _frame()
    first = EnvironmentSeedState.from_config(config, reset_seed=91).for_trace(TRACE_ID)
    second = EnvironmentSeedState.from_config(config, reset_seed=92).for_trace(TRACE_ID)

    first_actor = first.actor_assembler(config, start_frame_index=1).begin_frame(frame)
    second_actor = second.actor_assembler(config, start_frame_index=1).begin_frame(frame)
    first_tapes = first.packet_tapes(frame)
    second_tapes = second.packet_tapes(frame)

    assert first_actor.rows != second_actor.rows
    assert first_tapes != second_tapes
    reports_first = tuple(
        first.feedback_measurements(
            _vlc_outcome("pair-a"),
            first_tapes["pair-a"],
        )[Link.VLC]
        for _ in range(2)
    )
    reports_second = tuple(
        second.feedback_measurements(
            _vlc_outcome("pair-a"),
            second_tapes["pair-a"],
        )[Link.VLC]
        for _ in range(2)
    )
    assert reports_first != reports_second


def test_global_numpy_rng_and_population_lookup_order_cannot_shift_tapes() -> None:
    config = load_headline_config(PROJECT_ROOT)
    frame = _frame()
    trace = EnvironmentSeedState.from_config(config, reset_seed=123).for_trace(TRACE_ID)

    before = trace.packet_tapes(frame)
    np.random.seed(999)
    np.random.random(10_000)
    after = trace.packet_tapes(frame)

    assert before == after
    assert tuple(before) == frame.active_pair_ids
    assert before["pair-a"] == after["pair-a"]


def test_reset_info_is_immutable_complete_seed_provenance() -> None:
    config = load_headline_config(PROJECT_ROOT)
    trace = EnvironmentSeedState.from_config(config, reset_seed=77).for_trace(TRACE_ID)

    info = trace.as_reset_info()

    assert info["seed_schema"] == ENVIRONMENT_SEED_SCHEMA
    assert info["seed"] == 77
    assert info["active_root_seed"] == 77
    assert info["trace_id"] == TRACE_ID
    assert info["runtime_random_components"] == RUNTIME_RANDOM_COMPONENTS
    with pytest.raises(TypeError):
        info["seed"] = 78  # type: ignore[index]


def test_trace_context_and_unsigned_seed_fail_closed() -> None:
    config = load_headline_config(PROJECT_ROOT)
    wrong_trace = EnvironmentSeedState.from_config(config).for_trace(
        "synthetic-d20-train-001"
    )

    with pytest.raises(EnvironmentSeedError, match="does not belong"):
        wrong_trace.packet_tapes(_frame())
    with pytest.raises(EnvironmentSeedError, match="unsigned 64-bit"):
        EnvironmentSeedState(
            configured_root_seed=0,
            active_root_seed=2**64,
            explicit_reset_seed=True,
        )


def test_feedback_rejects_a_tape_from_another_reset_seed() -> None:
    config = load_headline_config(PROJECT_ROOT)
    frame = _frame()
    active = EnvironmentSeedState.from_config(config, reset_seed=10).for_trace(TRACE_ID)
    other = EnvironmentSeedState.from_config(config, reset_seed=11).for_trace(TRACE_ID)

    with pytest.raises(EnvironmentSeedError, match="active reset seed"):
        active.feedback_measurements(
            _vlc_outcome("pair-a"),
            other.packet_tapes(frame)["pair-a"],
        )
