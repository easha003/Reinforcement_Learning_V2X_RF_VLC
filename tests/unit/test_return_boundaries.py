"""Return-estimation masks at natural, internal, and trace boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.policy_actions import ActionResourceMap, PolicyAction
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.actor_observations import (
    CausalActorFrame,
    CausalActorRow,
)
from hybrid_v2x_rl.mean_field.congestion_feedback import (
    ActorObservationSchema,
    MeanFieldSignal,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.return_boundaries import (
    FrameReturnBoundary,
    ReturnBoundaryError,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord
from hybrid_v2x_rl.observation.builder import ObservationBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d10-train-000"
PAIR_IDS = (
    "pair-continuing",
    "pair-internal",
    "pair-natural",
    "pair-trace-end",
)


def _vehicle(vehicle_id: str) -> VehicleTraceRecord:
    index = int(vehicle_id.removeprefix("veh-"))
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=1.0,
        vehicle_id=vehicle_id,
        x_m=float(index),
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


def _inputs() -> tuple[FrameActionLedger, CausalActorFrame]:
    vehicles = tuple(_vehicle(f"veh-{index}") for index in range(1, 9))
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    lifecycles = {
        "pair-continuing": PairLifecycle(born=False),
        "pair-internal": PairLifecycle(
            born=False,
            truncated=True,
            bootstrap_valid=True,
            end_reason="max_duration",
        ),
        "pair-natural": PairLifecycle(
            born=False,
            terminated=True,
            end_reason="outside_range_1s",
        ),
        "pair-trace-end": PairLifecycle(
            born=False,
            truncated=True,
            end_reason="trace_end",
        ),
    }
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=10,
            transmitter=by_id[f"veh-{2 * index + 1}"],
            receiver=by_id[f"veh-{2 * index + 2}"],
            lifecycle=lifecycles[pair_id],
        )
        for index, pair_id in enumerate(PAIR_IDS)
    )
    frame = PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="train",
            density=10.0,
            replicate=0,
        ),
        index=10,
        time_s=1.0,
        vehicles=vehicles,
        pairs=pairs,
    )
    config = load_headline_config(PROJECT_ROOT)
    resource_map = ActionResourceMap.from_config(config.environment, config.cost)
    ledger = FrameActionLedger.from_frame(
        frame,
        {pair_id: PolicyAction.RF_1 for pair_id in PAIR_IDS},
        resource_map=resource_map,
    )
    schema = ActorObservationSchema(local=ObservationBuilder.from_config(config.observation).schema)
    signal = MeanFieldSignal.reset()
    actor_frame = CausalActorFrame(
        trace_id=TRACE_ID,
        frame_index=10,
        time_s=1.0,
        schema=schema,
        signal=signal,
        rows=tuple(
            CausalActorRow(
                pair_id=pair_id,
                values=None if pair_id == "pair-trace-end" else (0.0,) * schema.width,
            )
            for pair_id in PAIR_IDS
        ),
    )
    return ledger, actor_frame


def test_boundaries_separate_value_bootstrap_from_gae_continuation() -> None:
    ledger, actor_frame = _inputs()

    boundary = FrameReturnBoundary.from_frame(ledger, actor_frame)

    assert boundary.pair_ids == PAIR_IDS
    assert boundary.terminated.tolist() == [False, False, True, False]
    assert boundary.truncated.tolist() == [False, True, False, True]
    assert boundary.bootstrap_valid.tolist() == [False, True, False, False]
    assert boundary.value_bootstrap_mask.tolist() == [True, True, False, False]
    assert boundary.gae_continuation_mask.tolist() == [True, False, False, False]
    assert boundary.learn_mask.tolist() == [True, True, True, False]
    assert boundary.bootstrap_pair_ids == ("pair-internal",)
    assert boundary.zero_bootstrap_final_pair_ids == (
        "pair-natural",
        "pair-trace-end",
    )
    assert boundary.final_pair_ids == (
        "pair-internal",
        "pair-natural",
        "pair-trace-end",
    )
    assert not boundary.terminated.flags.writeable


def test_step_info_requires_exact_internal_truncation_final_observations() -> None:
    ledger, actor_frame = _inputs()
    boundary = FrameReturnBoundary.from_frame(ledger, actor_frame)

    with pytest.raises(ReturnBoundaryError, match="cover bootstrap-valid"):
        boundary.as_step_info()
    with pytest.raises(ReturnBoundaryError, match="cover bootstrap-valid"):
        boundary.as_step_info(
            final_observation={
                "pair-internal": ("next-row",),
                "pair-trace-end": ("forbidden",),
            }
        )

    info = boundary.as_step_info(final_observation={"pair-internal": ("next-row",)})
    assert info["end_reason"] == (
        None,
        "max_duration",
        "outside_range_1s",
        "trace_end",
    )
    assert info["final_observation"]["pair-internal"] == ("next-row",)
    assert info["release_after_frame_pair_ids"] == boundary.final_pair_ids


def test_action_and_actor_identity_drift_fails_before_masks_are_emitted() -> None:
    ledger, actor_frame = _inputs()
    wrong_frame = replace(actor_frame, frame_index=11)

    with pytest.raises(ReturnBoundaryError, match="same decision frame"):
        FrameReturnBoundary.from_frame(ledger, wrong_frame)


def test_direct_construction_rejects_recursive_gae_across_truncation() -> None:
    ledger, actor_frame = _inputs()
    boundary = FrameReturnBoundary.from_frame(ledger, actor_frame)

    with pytest.raises(ReturnBoundaryError, match="stop at every pair boundary"):
        replace(
            boundary,
            gae_continuation_mask=np.array([True, True, False, False], dtype=np.bool_),
        )
