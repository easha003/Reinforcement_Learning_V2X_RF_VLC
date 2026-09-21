"""Stable-ID binding for variable-population Phase 5 observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.mean_field.action_masks import ActionMask, MaskedActionSpace
from hybrid_v2x_rl.mean_field.environment_api import FrameAPISchema
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.population_binding import (
    PopulationBindingError,
    VariablePopulationBinding,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d20-train-000"
OTHER_TRACE_ID = "synthetic-d20-train-001"
SOURCE = FrameTraceSource(
    path=Path("/tmp") / TRACE_ID,
    trace_id=TRACE_ID,
    split="train",
    density=20.0,
    replicate=0,
)
OTHER_SOURCE = FrameTraceSource(
    path=Path("/tmp") / OTHER_TRACE_ID,
    trace_id=OTHER_TRACE_ID,
    split="train",
    density=20.0,
    replicate=1,
)
ENDPOINTS = {
    "pair-a": ("veh-1", "veh-2"),
    "pair-b": ("veh-3", "veh-4"),
    "pair-c": ("veh-5", "veh-6"),
    "pair-d": ("veh-7", "veh-8"),
}
ROW_MARKERS = {"pair-a": 0.1, "pair-b": 0.2, "pair-c": 0.3, "pair-d": 0.4}


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


def _frame(
    index: int,
    pair_ids: tuple[str, ...],
    *,
    born: frozenset[str] | None = None,
    source: FrameTraceSource = SOURCE,
) -> PopulationFrame:
    time_s = 0.1 * index
    births = (
        frozenset(pair_ids) if index == 0 and born is None
        else frozenset() if born is None
        else born
    )
    endpoint_ids = tuple(
        sorted({endpoint for pair_id in pair_ids for endpoint in ENDPOINTS[pair_id]})
    )
    vehicles = {
        vehicle.vehicle_id: vehicle
        for vehicle in (
            _vehicle(source.trace_id, time_s, vehicle_id)
            for vehicle_id in endpoint_ids
        )
    }
    pairs = tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=0 if pair_id in births else index,
            transmitter=vehicles[ENDPOINTS[pair_id][0]],
            receiver=vehicles[ENDPOINTS[pair_id][1]],
            lifecycle=PairLifecycle(born=pair_id in births),
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


def _rows(pair_ids: tuple[str, ...], *, reverse: bool = False) -> dict[str, tuple[float, ...]]:
    ordered = tuple(reversed(pair_ids)) if reverse else pair_ids
    return {
        pair_id: (ROW_MARKERS[pair_id], *(0.0 for _ in range(36)))
        for pair_id in ordered
    }


def _binding() -> VariablePopulationBinding:
    config = load_headline_config(PROJECT_ROOT)
    rf_only = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=False,
        max_reserved_rf_attempts=4,
    )
    return VariablePopulationBinding(
        api_schema=FrameAPISchema.from_config(config),
        action_mask=rf_only,
    )


def test_binding_from_config_reuses_the_authoritative_schema_and_mask() -> None:
    config = load_headline_config(PROJECT_ROOT)
    binding = VariablePopulationBinding.from_config(config)
    expected = MaskedActionSpace.from_config(
        config.environment,
        config.rf,
        config.vlc,
    )

    assert binding.api_schema == FrameAPISchema.from_config(config)
    assert binding.action_mask == expected.mask


def test_changing_populations_keep_rows_masks_and_deltas_on_stable_ids() -> None:
    binding = _binding()
    binding.reset(TRACE_ID)
    sequence = (
        _frame(0, ("pair-a", "pair-c"), born=frozenset({"pair-a", "pair-c"})),
        _frame(1, ("pair-a", "pair-b", "pair-c"), born=frozenset({"pair-b"})),
        _frame(2, ("pair-b",)),
        _frame(3, ()),
        _frame(4, ("pair-d",), born=frozenset({"pair-d"})),
    )
    bound = tuple(
        binding.bind_frame(frame, _rows(frame.active_pair_ids, reverse=True))
        for frame in sequence
    )

    assert [item.observation.pair_ids for item in bound] == [
        ("pair-a", "pair-c"),
        ("pair-a", "pair-b", "pair-c"),
        ("pair-b",),
        (),
        ("pair-d",),
    ]
    assert [item.delta.entered_pair_ids for item in bound] == [
        ("pair-a", "pair-c"),
        ("pair-b",),
        (),
        (),
        ("pair-d",),
    ]
    assert [item.delta.continuing_pair_ids for item in bound] == [
        (),
        ("pair-a", "pair-c"),
        ("pair-b",),
        (),
        (),
    ]
    assert [item.delta.exited_pair_ids for item in bound] == [
        (),
        (),
        ("pair-a", "pair-c"),
        ("pair-b",),
        (),
    ]

    for item in bound:
        ids = item.observation.pair_ids
        assert item.observation.actor_observations[:, 0].tolist() == [
            pytest.approx(ROW_MARKERS[pair_id]) for pair_id in ids
        ]
        assert item.observation.action_masks.shape == (len(ids), 9)
        assert np.all(
            item.observation.action_masks
            == np.asarray(binding.action_mask.values, dtype=np.bool_)
        )
    assert bound[3].observation.actor_observations.shape == (0, 37)
    assert bound[3].observation.action_masks.shape == (0, 9)
    assert binding.current_pair_ids == ("pair-d",)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ({"pair-a": (0.0,) * 37}, "active population"),
        (
            {"pair-a": (0.0,) * 37, "pair-b": (0.0,) * 37, "pair-x": (0.0,) * 37},
            "active population",
        ),
        (
            {"pair-a": (0.0,) * 36, "pair-b": (0.0,) * 37},
            "width",
        ),
        (
            {"pair-a": (np.nan,) + (0.0,) * 36, "pair-b": (0.0,) * 37},
            "finite",
        ),
    ],
)
def test_binding_rejects_missing_extra_or_invalid_actor_rows(
    rows: dict[str, tuple[float, ...]],
    match: str,
) -> None:
    binding = _binding()
    binding.reset(TRACE_ID)
    frame = _frame(0, ("pair-a", "pair-b"))

    with pytest.raises(PopulationBindingError, match=match):
        binding.bind_frame(frame, rows)

    # A rejected candidate must not consume the expected frame index.
    accepted = binding.bind_frame(frame, _rows(frame.active_pair_ids))
    assert accepted.observation.frame_index == 0


def test_frames_must_match_reset_trace_and_advance_without_gaps() -> None:
    binding = _binding()
    frame = _frame(0, ("pair-a",))

    with pytest.raises(PopulationBindingError, match="reset first"):
        binding.bind_frame(frame, _rows(frame.active_pair_ids))

    binding.reset(TRACE_ID)
    wrong_trace = _frame(0, ("pair-a",), source=OTHER_SOURCE)
    with pytest.raises(PopulationBindingError, match="trace"):
        binding.bind_frame(wrong_trace, _rows(wrong_trace.active_pair_ids))
    with pytest.raises(PopulationBindingError, match="gaps or reordering"):
        skipped = _frame(1, ("pair-a",))
        binding.bind_frame(skipped, _rows(skipped.active_pair_ids))

    binding.bind_frame(frame, _rows(frame.active_pair_ids))
    with pytest.raises(PopulationBindingError, match="gaps or reordering"):
        binding.bind_frame(frame, _rows(frame.active_pair_ids))


def test_retired_id_requires_a_new_episode_identity_until_reset() -> None:
    binding = _binding()
    binding.reset(TRACE_ID)
    first = _frame(0, ("pair-a",))
    empty = _frame(1, ())
    reappeared = _frame(2, ("pair-a",))
    binding.bind_frame(first, _rows(first.active_pair_ids))
    binding.bind_frame(empty, {})

    with pytest.raises(PopulationBindingError, match="retired pair ID"):
        binding.bind_frame(reappeared, _rows(reappeared.active_pair_ids))

    binding.reset(TRACE_ID, start_frame_index=2)
    fresh = binding.bind_frame(reappeared, _rows(reappeared.active_pair_ids))
    assert fresh.delta.entered_pair_ids == ("pair-a",)
    assert fresh.delta.continuing_pair_ids == ()
    assert fresh.delta.exited_pair_ids == ()


def test_mid_episode_entries_and_declared_births_must_match() -> None:
    binding = _binding()
    binding.reset(TRACE_ID)
    first = _frame(0, ("pair-a",))
    binding.bind_frame(first, _rows(first.active_pair_ids))

    undeclared = _frame(1, ("pair-a", "pair-b"))
    with pytest.raises(PopulationBindingError, match="declared pair births"):
        binding.bind_frame(undeclared, _rows(undeclared.active_pair_ids))

    repeated = _frame(
        1,
        ("pair-a",),
        born=frozenset({"pair-a"}),
    )
    with pytest.raises(PopulationBindingError, match="declared pair births"):
        binding.bind_frame(repeated, _rows(repeated.active_pair_ids))

    declared = _frame(
        1,
        ("pair-a", "pair-b"),
        born=frozenset({"pair-b"}),
    )
    accepted = binding.bind_frame(declared, _rows(declared.active_pair_ids))
    assert accepted.delta.entered_pair_ids == ("pair-b",)
    assert accepted.delta.continuing_pair_ids == ("pair-a",)
