"""End to end: exact state in, degraded observation out, reproducibly.

The individual modules are tested beside themselves.  This file asserts the
three properties that only exist once they are wired together:

1. the pipeline is deterministic, so a replay reproduces the run it replays;
2. the observation genuinely *differs from truth* -- without this, wiring the
   sensor to pass exact state through would leave every other test green;
3. the action changes what the next observation contains, which is §6.3's
   sequential claim measured at the output rather than asserted at the input.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.config.validation import FORBIDDEN_OBSERVATION_FIELDS
from hybrid_v2x_rl.core.enums import Action, Link
from hybrid_v2x_rl.core.geometry import Point, Segment
from hybrid_v2x_rl.core.intersection_context import (
    IntersectionContext,
    JunctionGrid,
    intersection_context,
)
from hybrid_v2x_rl.observation.blockage import BlockerShape, blockage_probability
from hybrid_v2x_rl.observation.builder import (
    UNMEASURED_AGE_S,
    ObservationBuilder,
    PairObservationInputs,
)
from hybrid_v2x_rl.observation.forecast import ConstantVelocityForecaster, PredictedState
from hybrid_v2x_rl.observation.link_state import LinkStateTracker
from hybrid_v2x_rl.observation.sensing import SensorModel, sense_frame
from hybrid_v2x_rl.observation.tracks import TrackStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_ROOT = PROJECT_ROOT / "src" / "hybrid_v2x_rl" / "observation"

EAST = 0.0
NORTH = 0.5 * math.pi
ROOT_SEED = 20260728


@dataclass(frozen=True, slots=True)
class Vehicle:
    vehicle_id: str
    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float
    length_m: float = 4.5
    width_m: float = 1.8
    height_m: float = 1.5


def scene() -> list[Vehicle]:
    """A tagged pair 20 m apart with cross traffic approaching a junction."""

    return [
        Vehicle("tx", 0.0, 0.0, EAST, 9.0),
        Vehicle("rx", 20.0, 0.0, EAST, 9.5),
        Vehicle("cross", 10.0, -6.0, NORTH, 7.0),
        Vehicle("far", 120.0, 40.0, EAST, 11.0),
    ]



def grid() -> JunctionGrid:
    return JunctionGrid.from_grid(
        avenues=6, cross_streets=12, avenue_spacing_m=244.0,
        cross_street_spacing_m=61.0, road_half_width_m=3.5,
    )


def context(tx: PredictedState, rx: PredictedState) -> IntersectionContext:
    return intersection_context(
        tx.x_m, tx.y_m, tx.heading_rad, rx.x_m, rx.y_m, rx.heading_rad,
        Segment(Point(tx.x_m, tx.y_m), Point(rx.x_m, rx.y_m)), grid(),
    )

def observe(
    vehicles: list[Vehicle],
    *,
    now_s: float = 1.05,
    seed: int = ROOT_SEED,
    trace_id: str = "synthetic-d20-train-000",
    links: LinkStateTracker | None = None,
) -> tuple[float, ...]:
    """Run the whole chain: sense, track, forecast, predict, assemble."""

    config = load_headline_config(PROJECT_ROOT)
    sensor = SensorModel.from_config(config.observation)
    forecaster = ConstantVelocityForecaster.from_config(config.observation)
    builder = ObservationBuilder.from_config(config.observation)

    store = TrackStore()
    store.update(
        sense_frame(vehicles, now_s=now_s, sensor=sensor, root_seed=seed, trace_id=trace_id),
        now_s=now_s,
    )

    shapes = {v.vehicle_id: BlockerShape(v.length_m, v.width_m, v.height_m) for v in vehicles}
    transmitter = forecaster.predict(store.require("tx"), now_s=now_s)
    receiver = forecaster.predict(store.require("rx"), now_s=now_s)
    blockers = [
        (forecaster.predict(track, now_s=now_s), shapes[track.vehicle_id])
        for track in store.tracks()
        if track.vehicle_id not in ("tx", "rx")
    ]

    return builder.build(
        PairObservationInputs(
            now_s=now_s,
            transmitter=transmitter,
            receiver=receiver,
            transmitter_track=store.require("tx"),
            receiver_track=store.require("rx"),
            blockage=blockage_probability(
                transmitter, receiver, blockers, horizon_s=forecaster.horizon_s
            ),
            links=links or LinkStateTracker(history_packets=config.observation.history_packets),
            neighbour_count=store.neighbour_count(
                store.require("tx"), radius_m=200.0, now_s=now_s
            ),
            channel_busy_ratio=0.3,
            fov_half_angle_rad=math.radians(60.0),
            intersection=context(transmitter, receiver),
        )
    )


# -- 1. determinism -----------------------------------------------------------


def test_the_same_inputs_produce_a_bit_identical_observation() -> None:
    assert observe(scene()) == observe(scene())


def test_a_different_seed_produces_a_different_observation() -> None:
    """If it did not, the sensor noise would not be reaching the output."""

    assert observe(scene(), seed=ROOT_SEED) != observe(scene(), seed=ROOT_SEED + 1)


def test_a_different_trace_produces_a_different_observation() -> None:
    assert observe(scene(), trace_id="synthetic-d10-train-000") != observe(scene())


def test_the_vector_is_finite_and_matches_the_configured_width() -> None:
    config = load_headline_config(PROJECT_ROOT)
    builder = ObservationBuilder.from_config(config.observation)
    vector = observe(scene())

    assert len(vector) == builder.schema.width
    assert all(math.isfinite(value) for value in vector)


def test_adding_an_unrelated_vehicle_does_not_perturb_the_pair() -> None:
    """Noise is keyed by identity, so the scene composition cannot leak in.

    Without this the observation would depend on how many vehicles happened to
    be sensed first, and a replay could differ from the run it replays.
    """

    config = load_headline_config(PROJECT_ROOT)
    columns = ObservationBuilder.from_config(config.observation).schema.columns
    stable = [columns.index(name) for name in ("pair_distance", "pair_bearing", "relative_speed")]

    base = observe(scene())
    with_extra = observe([*scene(), Vehicle("extra", 300.0, 300.0, EAST, 5.0)])

    for index in stable:
        assert base[index] == pytest.approx(with_extra[index])


# -- 2. the observation is not the truth --------------------------------------


def test_the_observed_separation_differs_from_the_true_separation() -> None:
    """The load-bearing test of the whole barrier.

    Wiring the sensor to pass exact state through would leave every other test
    in this package green.  This one fails.
    """

    config = load_headline_config(PROJECT_ROOT)
    columns = ObservationBuilder.from_config(config.observation).schema.columns
    observed = observe(scene())[columns.index("pair_distance")]

    true_separation = 20.0
    assert observed != pytest.approx(true_separation, abs=1e-9)
    # Still recognisably the same pair: degraded, not scrambled.
    assert abs(observed - true_separation) < 10.0


def test_the_error_has_the_declared_scale_rather_than_being_token() -> None:
    """Averaged over seeds the bias is small, but any single view is wrong."""

    config = load_headline_config(PROJECT_ROOT)
    columns = ObservationBuilder.from_config(config.observation).schema.columns
    index = columns.index("pair_distance")

    errors = [observe(scene(), seed=ROOT_SEED + step)[index] - 20.0 for step in range(200)]
    mean = sum(errors) / len(errors)
    spread = math.sqrt(sum(error**2 for error in errors) / len(errors) - mean**2)

    assert abs(mean) < 0.3, "position noise is zero-mean"
    # Two endpoints, each 0.5 m: the separation error is larger than one.
    assert 0.3 < spread < 2.0


def test_no_observed_column_carries_a_forbidden_name() -> None:
    config = load_headline_config(PROJECT_ROOT)
    columns = set(ObservationBuilder.from_config(config.observation).schema.columns)

    assert not columns & set(FORBIDDEN_OBSERVATION_FIELDS)


def test_no_observation_module_imports_the_geometry_engine() -> None:
    """Restated here against the package as it now stands, not as it was empty.

    ``hybrid_v2x_rl.core.geometry`` and its siblings are fine: pure mathematics over
    whatever coordinates they are handed.  ``hybrid_v2x_rl.geometry`` is not, because
    everything in it reasons about exact vehicle state.
    """

    offenders: list[str] = []
    for source in sorted(OBSERVATION_ROOT.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.startswith("hybrid_v2x_rl.geometry") for name in names):
                offenders.append(str(source.relative_to(PROJECT_ROOT)))
                break

    assert not offenders, f"observation reached the exact-state layer: {offenders}"


def test_the_observation_package_is_not_empty() -> None:
    """The guard above is only meaningful if there is something to guard."""

    modules = [path.name for path in OBSERVATION_ROOT.glob("*.py") if path.name != "__init__.py"]
    assert len(modules) >= 5, modules


# -- 3. the action changes the next observation -------------------------------


def test_choosing_one_leg_leaves_the_other_unmeasured_in_the_vector() -> None:
    """§6.3 measured at the output, not asserted at the input."""

    config = load_headline_config(PROJECT_ROOT)
    builder = ObservationBuilder.from_config(config.observation)
    columns = builder.schema.columns

    links = LinkStateTracker(history_packets=config.observation.history_packets)
    for step in range(5):
        links.record(
            action=Action.RF, at_s=float(step) * 0.1, delivered=True,
            measurements={Link.RF: 1.0 + step},
        )

    vector = observe(scene(), links=links)
    assert vector[columns.index("rf_quality_age")] >= 0.0
    assert vector[columns.index("vlc_quality_age")] == UNMEASURED_AGE_S


def test_duplicating_makes_both_legs_measurable() -> None:
    config = load_headline_config(PROJECT_ROOT)
    builder = ObservationBuilder.from_config(config.observation)
    columns = builder.schema.columns

    links = LinkStateTracker(history_packets=config.observation.history_packets)
    links.record(action=Action.DUP, at_s=1.0, delivered=True,
                 measurements={Link.RF: 2.0, Link.VLC: 3.0})

    vector = observe(scene(), links=links)
    assert vector[columns.index("rf_quality_age")] >= 0.0
    assert vector[columns.index("vlc_quality_age")] >= 0.0
    assert vector[columns.index("vlc_quality")] == pytest.approx(3.0)


def test_the_history_columns_carry_the_recorded_sequence() -> None:
    config = load_headline_config(PROJECT_ROOT)
    builder = ObservationBuilder.from_config(config.observation)
    columns = builder.schema.columns

    links = LinkStateTracker(history_packets=config.observation.history_packets)
    for step in range(3):
        links.record(action=Action.RF, at_s=float(step), delivered=True,
                     measurements={Link.RF: float(step) + 1.0})

    vector = observe(scene(), links=links)
    tail = [vector[columns.index(f"rf_quality_history[{index}]")] for index in (5, 6, 7)]
    assert tail == pytest.approx([1.0, 2.0, 3.0])
