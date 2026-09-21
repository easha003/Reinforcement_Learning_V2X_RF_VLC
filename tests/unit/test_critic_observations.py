"""Training-only centralized critic rows remain outside actor observations."""

from __future__ import annotations

import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.mean_field.critic_observations import (
    CONTRACT_CRITIC_WIDTH,
    CONTRACT_GLOBAL_SUMMARY_WIDTH,
    CentralizedCriticBuilder,
    CentralizedCriticSchema,
    CriticObservationError,
    CriticObservationFrame,
)
from hybrid_v2x_rl.mean_field.environment_api import (
    CONTRACT_ACTION_COUNT,
    CONTRACT_ACTOR_WIDTH,
    FrameObservation,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d20-train-000"
PAIR_ENDPOINTS = {
    "pair-a": ("veh-1", "veh-2"),
    "pair-b": ("veh-3", "veh-4"),
    "pair-c": ("veh-5", "veh-6"),
}


def _source(*, density: float = 20.0, trace_id: str = TRACE_ID) -> FrameTraceSource:
    return FrameTraceSource(
        path=Path("/tmp") / trace_id,
        trace_id=trace_id,
        split="train",
        density=density,
        replicate=0,
    )


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
    pair_ids: tuple[str, ...] = ("pair-a", "pair-b"),
    *,
    density: float = 20.0,
    trace_id: str = TRACE_ID,
    index: int = 3,
) -> PopulationFrame:
    source = _source(density=density, trace_id=trace_id)
    time_s = 0.1 * index
    endpoint_ids = tuple(
        sorted({endpoint for pair_id in pair_ids for endpoint in PAIR_ENDPOINTS[pair_id]})
    )
    vehicles = {
        vehicle_id: _vehicle(trace_id, time_s, vehicle_id)
        for vehicle_id in endpoint_ids
    }
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
        source=source,
        index=index,
        time_s=time_s,
        vehicles=tuple(vehicles[vehicle_id] for vehicle_id in sorted(vehicles)),
        pairs=pairs,
    )


def _actor_frame(
    frame: PopulationFrame,
    *,
    values: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> FrameObservation:
    population = len(frame.active_pair_ids)
    if values is None:
        values = np.stack(
            (
                np.linspace(-1.0, 1.0, CONTRACT_ACTOR_WIDTH, dtype=np.float32),
                np.linspace(1.0, 3.0, CONTRACT_ACTOR_WIDTH, dtype=np.float32),
            )[:population]
        ) if population else np.empty((0, CONTRACT_ACTOR_WIDTH), dtype=np.float32)
    if masks is None:
        masks = np.ones((population, CONTRACT_ACTION_COUNT), dtype=np.bool_)
    return FrameObservation(
        trace_id=frame.trace_id,
        frame_index=frame.index,
        time_s=frame.time_s,
        pair_ids=frame.active_pair_ids,
        actor_observations=values,
        action_masks=masks,
    )


def _builder() -> CentralizedCriticBuilder:
    return CentralizedCriticBuilder.from_config(load_headline_config(PROJECT_ROOT))


def test_schema_derives_the_frozen_37_plus_41_equals_78_contract() -> None:
    schema = _builder().schema

    assert schema.density_levels == (10.0, 20.0, 30.0)
    assert schema.actor_width == CONTRACT_ACTOR_WIDTH == 37
    assert schema.global_width == CONTRACT_GLOBAL_SUMMARY_WIDTH == 41
    assert schema.critic_width == CONTRACT_CRITIC_WIDTH == 78
    assert schema.global_columns[:37] == tuple(
        f"population_mean[{column}]" for column in schema.actor_columns
    )
    assert schema.global_columns[-4:] == (
        "density_10_veh_per_lane_km",
        "density_20_veh_per_lane_km",
        "density_30_veh_per_lane_km",
        "log1p_population_size",
    )


def test_critic_rows_append_the_exact_population_summary() -> None:
    frame = _population_frame()
    actor_frame = _actor_frame(frame)

    critic_frame = _builder().build(frame, actor_frame)

    expected_mean = np.mean(
        actor_frame.actor_observations,
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    expected_summary = np.concatenate(
        (
            expected_mean,
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([math.log1p(2)], dtype=np.float32),
        )
    )
    assert critic_frame.pair_ids == actor_frame.pair_ids
    assert critic_frame.critic_observations.shape == (2, 78)
    assert np.array_equal(critic_frame.global_summary, expected_summary)
    assert np.array_equal(
        critic_frame.critic_observations[:, :37],
        actor_frame.actor_observations,
    )
    assert np.array_equal(
        critic_frame.critic_observations[:, 37:],
        np.broadcast_to(expected_summary, (2, 41)),
    )


@pytest.mark.parametrize(
    ("density", "expected"),
    [
        (10.0, (1.0, 0.0, 0.0)),
        (20.0, (0.0, 1.0, 0.0)),
        (30.0, (0.0, 0.0, 1.0)),
    ],
)
def test_density_one_hot_uses_frozen_configured_order(
    density: float,
    expected: tuple[float, float, float],
) -> None:
    frame = _population_frame(density=density)
    critic = _builder().build(frame, _actor_frame(frame))

    assert tuple(critic.global_summary[37:40]) == expected


def test_critic_materialization_cannot_modify_or_extend_the_actor_frame() -> None:
    frame = _population_frame()
    actor_frame = _actor_frame(frame)
    actor_before = actor_frame.actor_observations.copy()
    critic = _builder().build(frame, actor_frame)

    assert np.array_equal(actor_frame.actor_observations, actor_before)
    assert actor_frame.actor_observations.shape == (2, 37)
    assert not np.shares_memory(
        actor_frame.actor_observations,
        critic.critic_observations,
    )
    assert not critic.global_summary.flags.writeable
    assert not critic.critic_observations.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        critic.critic_observations[0, 0] = 99.0

    actor_fields = {item.name for item in fields(FrameObservation)}
    assert "critic_observations" not in actor_fields
    assert "global_summary" not in actor_fields
    assert "density" not in actor_fields


def test_action_masks_are_not_critic_features() -> None:
    frame = _population_frame()
    all_actions = _actor_frame(frame)
    sparse_masks = np.zeros((2, CONTRACT_ACTION_COUNT), dtype=np.bool_)
    sparse_masks[:, 0] = True
    vlc_only = _actor_frame(frame, masks=sparse_masks)

    all_critic = _builder().build(frame, all_actions)
    vlc_critic = _builder().build(frame, vlc_only)

    assert np.array_equal(
        all_critic.critic_observations,
        vlc_critic.critic_observations,
    )


def test_empty_population_has_no_invented_mean_and_no_critic_rows() -> None:
    frame = _population_frame(())
    critic = _builder().build(frame, _actor_frame(frame))

    assert critic.pair_ids == ()
    assert critic.population_size == 0
    assert critic.global_summary is None
    assert critic.critic_observations.shape == (0, 78)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"trace_id": "synthetic-d20-train-001"}, id="trace"),
        pytest.param({"frame_index": 4}, id="index"),
        pytest.param({"time_s": 0.4}, id="time"),
        pytest.param({"pair_ids": ("pair-a", "pair-c")}, id="pair-ids"),
    ],
)
def test_actor_and_population_identity_must_match(
    changed: dict[str, object],
) -> None:
    frame = _population_frame()
    actor_frame = replace(_actor_frame(frame), **changed)

    with pytest.raises(CriticObservationError, match="same decision frame"):
        _builder().build(frame, actor_frame)


@pytest.mark.parametrize("pair_ids", [("pair-a", "pair-b"), ()])
def test_unknown_density_cannot_be_smuggled_into_training_state(
    pair_ids: tuple[str, ...],
) -> None:
    frame = _population_frame(pair_ids, density=25.0)

    with pytest.raises(CriticObservationError, match="configured critic levels"):
        _builder().build(frame, _actor_frame(frame))


def test_schema_rejects_density_or_actor_width_drift() -> None:
    columns = tuple(f"actor-{index}" for index in range(37))
    with pytest.raises(CriticObservationError, match="three configured density"):
        CentralizedCriticSchema("1.0.0", columns, (10.0, 20.0))
    with pytest.raises(CriticObservationError, match="37 actor columns"):
        CentralizedCriticSchema("1.0.0", columns[:-1], (10.0, 20.0, 30.0))


def test_direct_critic_frame_rejects_a_nonshared_global_suffix() -> None:
    summary = np.zeros(41, dtype=np.float32)
    summary[38] = 1.0
    summary[-1] = np.float32(math.log1p(1))
    critic = np.zeros((1, 78), dtype=np.float32)
    critic[0, -1] = 1.0

    with pytest.raises(CriticObservationError, match="same frame-global suffix"):
        CriticObservationFrame(
            trace_id=TRACE_ID,
            frame_index=0,
            time_s=0.0,
            density=20.0,
            schema=_builder().schema,
            pair_ids=("pair-a",),
            global_summary=summary,
            critic_observations=critic,
        )


@pytest.mark.parametrize(
    ("summary_index", "value", "match"),
    [
        (0, 1.0, "mean"),
        (37, 1.0, "density"),
        (40, 2.0, "population size"),
    ],
)
def test_direct_critic_frame_reconciles_every_global_component(
    summary_index: int,
    value: float,
    match: str,
) -> None:
    summary = np.zeros(41, dtype=np.float32)
    summary[38] = 1.0
    summary[-1] = np.float32(math.log1p(1))
    summary[summary_index] = value
    critic = np.concatenate(
        (
            np.zeros((1, 37), dtype=np.float32),
            summary.reshape(1, 41),
        ),
        axis=1,
    )

    with pytest.raises(CriticObservationError, match=match):
        CriticObservationFrame(
            trace_id=TRACE_ID,
            frame_index=0,
            time_s=0.0,
            density=20.0,
            schema=_builder().schema,
            pair_ids=("pair-a",),
            global_summary=summary,
            critic_observations=critic,
        )
