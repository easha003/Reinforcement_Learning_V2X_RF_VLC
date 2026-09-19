"""The causal view: what a deployable radio could actually know.

The load-bearing test here is the import guard. Everything else in this file
checks that the chain produces sensible numbers; that one checks it cannot
produce *privileged* numbers, and it is the only test in the suite whose
failure would invalidate every policy result rather than one of them.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.env.episodes import TraceSource, iter_pair_instants
from hybrid_v2x_rl.env.packet import DUP, RF_ONLY
from hybrid_v2x_rl.env.perception import (
    NOMINAL_BLOCKER,
    _near_segment,
    build_perception,
)
from hybrid_v2x_rl.observation.link_state import Link

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE = PROJECT_ROOT / "artifacts" / "traces" / "synthetic-d20-test-000"


# -- the barrier --------------------------------------------------------------


def test_perception_cannot_reach_the_geometry_engine() -> None:
    """The policy side must not import the exact side, and this parses to check.

    Grepping the file would match the prose in its own docstring, which
    discusses precisely the modules it must not import. Parsing the imports is
    the difference between a test that reads the code and one that reads about
    it.

    If this fails, every result produced by a learned policy is void: it would
    have been conditioning on hidden simulator state, and no amount of
    downstream statistics recovers from that.
    """

    source = (PROJECT_ROOT / "src" / "hybrid_v2x_rl" / "env" / "perception.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        name for name in imported
        if name.startswith("hybrid_v2x_rl.geometry")
        or name == "hybrid_v2x_rl.core.pair_geometry"
        or name.startswith("hybrid_v2x_rl.channels.vlc")
    }
    assert not forbidden, f"perception reached the exact side: {sorted(forbidden)}"


def test_the_observation_never_sees_a_vehicle_body() -> None:
    """Sensing reports where a vehicle is, not how large it is.

    Every blocker is one nominal body, so the policy cannot distinguish a
    hatchback from a bus -- and leader length is worth 4.7 dB optical, which
    makes this the largest single unobservable in the system. Asserting the
    constant is a passenger car rather than something conveniently large keeps
    the gap honest.
    """

    assert NOMINAL_BLOCKER.length_m == pytest.approx(4.5)
    assert NOMINAL_BLOCKER.height_m == pytest.approx(1.5)


# -- geometry helper ----------------------------------------------------------


class _P:
    def __init__(self, x, y):
        self.x_m, self.y_m = x, y


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (5.0, 0.0, True),    # on the segment
        (5.0, 1.5, True),    # within the margin
        (5.0, 4.0, False),   # beyond it
        (-3.0, 0.0, False),  # behind the transmitter
        (13.0, 0.0, False),  # past the receiver
    ],
)
def test_the_blocker_filter_keeps_only_what_lies_along_the_path(x, y, expected) -> None:
    """The filter exists to remove work, not candidates, so it must not clip
    anything the estimator would have considered obstructing."""

    assert _near_segment(x, y, _P(0.0, 0.0), _P(10.0, 0.0), 2.0) is expected


def test_a_degenerate_path_falls_back_to_a_radius() -> None:
    """Transmitter and receiver at the same point has no direction to project
    onto, and dividing by that length is how a rare frame becomes a crash."""

    assert _near_segment(0.5, 0.0, _P(1.0, 1.0), _P(1.0, 1.0), 2.0) is True
    assert _near_segment(9.0, 9.0, _P(1.0, 1.0), _P(1.0, 1.0), 2.0) is False


# -- the chain end to end -----------------------------------------------------


pytestmark_trace = pytest.mark.skipif(
    not TRACE.exists(), reason="campaign traces are not present in this checkout"
)


@pytest.fixture(scope="module")
def config():
    return load_headline_config(PROJECT_ROOT)


def instants(limit: int):
    source = TraceSource.discover(TRACE)
    return list(iter_pair_instants(
        source, generation_period_s=0.1, max_packets=limit, warmup_s=400.0))


@pytestmark_trace
def test_the_observation_matches_the_declared_schema(config) -> None:
    """History features expand, so the vector is longer than the feature list."""

    perception = build_perception(config, root_seed=3)
    vectors = [perception.observe(i) for i in instants(60)]
    usable = [v for v in vectors if v is not None]
    assert usable, "no usable observation was produced"
    assert len({len(v) for v in usable}) == 1, "the vector must have a fixed width"
    expected = len(config.observation.features) - 2 + 2 * config.observation.history_packets
    assert len(usable[0]) == expected


@pytestmark_trace
def test_every_observation_is_finite(config) -> None:
    """A NaN here trains a policy on nothing and reports no error."""

    perception = build_perception(config, root_seed=4)
    for instant in instants(200):
        vector = perception.observe(instant)
        if vector is None:
            continue
        assert all(math.isfinite(x) for x in vector)


@pytestmark_trace
def test_the_sensed_neighbour_count_lags_the_true_one(config) -> None:
    """The whole point of the causal path, in one number.

    The rollout tells the channel how many contenders are really in range; this
    tells the policy how many it can currently hear. If they were equal the
    sensor model would not be doing anything.
    """

    from hybrid_v2x_rl.env.perception import CONTENTION_RADIUS_M

    perception = build_perception(config, root_seed=5)
    gaps = []
    for instant in instants(400):
        if perception.observe(instant) is None:
            continue
        track = perception.tracks.get(instant.transmitter.vehicle_id)
        sensed = perception.tracks.neighbour_count(
            track, radius_m=CONTENTION_RADIUS_M, now_s=instant.time_s, limit_s=1.0)
        truth = sum(
            1 for other in instant.neighbours
            if other.vehicle_id != instant.transmitter.vehicle_id
            and math.hypot(other.x_m - instant.transmitter.x_m,
                           other.y_m - instant.transmitter.y_m) <= CONTENTION_RADIUS_M
        )
        gaps.append(truth - sensed)
    assert gaps, "expected some usable instants"
    assert max(gaps) > 0, "the sensed count must ever lag the true one"


@pytestmark_trace
def test_sensing_happens_once_per_frame(config) -> None:
    """Sensing per pair would hand a vehicle a fresh measurement for each of
    its neighbours and quietly delete the latency this layer models."""

    perception = build_perception(config, root_seed=6)
    batch = instants(300)
    by_time: dict[float, list] = {}
    for instant in batch:
        by_time.setdefault(instant.time_s, []).append(instant)
    shared = next(v for v in by_time.values() if len(v) > 2)

    for instant in shared:
        perception.observe(instant)
    stamps = {t.sample.observed_at_s for t in perception.tracks.tracks()}
    assert len(stamps) == 1, "one sweep per frame, so one observation time"


@pytestmark_trace
def test_link_history_is_per_pair_and_released(config) -> None:
    perception = build_perception(config, root_seed=8)
    batch = instants(80)
    first = batch[0]
    perception.observe(first)
    perception.record(first.pair_id, action=RF_ONLY, at_s=first.time_s,
                      delivered=True, measurements={Link.RF: 0.9})
    assert perception._links[first.pair_id].packets_seen == 1
    other = next(i for i in batch if i.pair_id != first.pair_id)
    assert perception._links.get(other.pair_id) is None or \
        perception._links[other.pair_id].packets_seen == 0
    perception.release(first.pair_id)
    assert first.pair_id not in perception._links


@pytestmark_trace
def test_every_environment_action_translates_to_the_observation_layer() -> None:
    """The two layers name the actions with different types, and the seam must
    cover all of them.

    M5's Action is a dataclass carrying an activation cost; M2's is an enum
    used as a dictionary key. An action added to one and not mapped here fails
    at runtime inside a training loop, which is the worst place to find it.
    """

    from hybrid_v2x_rl.env.packet import ACTIONS
    from hybrid_v2x_rl.env.perception import _OBSERVED_ACTION

    for action in ACTIONS:
        assert action.name in _OBSERVED_ACTION, f"{action.name} has no translation"
    assert len(_OBSERVED_ACTION) == len(ACTIONS), "stale entries in the mapping"


@pytestmark_trace
def test_feedback_is_action_dependent(config) -> None:
    """A packet that spent only the radio teaches nothing about the light.

    That asymmetry is why the observation carries a quality *age* beside every
    quality: an unused leg's reading gets older rather than being refreshed.
    """

    perception = build_perception(config, root_seed=9)
    batch = instants(40)
    pair = batch[0]
    perception.observe(pair)

    # A radio-only packet refreshes the radio's reading and leaves the light's
    # untouched, so at any later instant the optical age exceeds the radio's.
    perception.record(pair.pair_id, action=RF_ONLY, at_s=1.0, delivered=True,
                      measurements={Link.RF: 0.8})
    ages = perception._links[pair.pair_id].ages(now_s=2.0)
    assert ages[Link.RF] == pytest.approx(1.0)
    assert ages[Link.VLC] is None, "the light was never used, so it has no reading"

    # Duplication spends both legs and therefore measures both.
    perception.record(pair.pair_id, action=DUP, at_s=2.0, delivered=True,
                      measurements={Link.RF: 0.8, Link.VLC: 0.7})
    ages = perception._links[pair.pair_id].ages(now_s=2.5)
    assert ages[Link.RF] == pytest.approx(0.5)
    assert ages[Link.VLC] == pytest.approx(0.5)


@pytestmark_trace
def test_the_spatial_hash_finds_exactly_the_brute_force_blockers(config) -> None:
    """The hash is an optimisation, so it must change no answer.

    Both the contender count and the blocker filter were scanning every live
    track for every pair and together took 88% of the chain. Bucketing them is
    only legitimate if the set it returns is the set brute force returns --
    a filter that quietly dropped a candidate would weaken the blockage
    estimate in exactly the crowded frames where it matters.
    """

    from hybrid_v2x_rl.env.perception import NOMINAL_BLOCKER, _near_segment

    perception = build_perception(config, root_seed=11)
    compared = 0
    for instant in instants(400):
        if perception.observe(instant) is None:
            continue
        tx = perception._predicted[instant.transmitter.vehicle_id]
        rx = perception._predicted[instant.receiver.vehicle_id]
        margin = 0.5 * NOMINAL_BLOCKER.length_m + 3.0 * max(
            tx.cross_track_std_m, rx.cross_track_std_m, 1.0)
        exclude = (tx.vehicle_id, rx.vehicle_id)
        brute = {
            s.vehicle_id for s in perception._predicted.values()
            if s.vehicle_id not in exclude
            and _near_segment(s.x_m, s.y_m, tx, rx, margin)
        }
        hashed = {
            s.vehicle_id
            for s in perception._cells_near(tx.x_m, tx.y_m, rx.x_m, rx.y_m, margin)
            if s.vehicle_id not in exclude
            and _near_segment(s.x_m, s.y_m, tx, rx, margin)
        }
        assert hashed == brute
        compared += 1
    assert compared > 100, "expected a meaningful number of comparisons"


@pytestmark_trace
def test_the_contender_count_is_taken_at_the_forecast_horizon(config) -> None:
    """Consistent with the rest of the observation, and asserted as a choice.

    The blockage estimate and the pair geometry are both evaluated at the
    horizon because the decision applies to a packet sent later. Counting
    contenders at measurement time instead would describe a different instant
    from the rest of the vector.
    """

    from hybrid_v2x_rl.env.perception import CONTENTION_RADIUS_M

    perception = build_perception(config, root_seed=12)
    instant = next(i for i in instants(60) if perception.observe(i) is not None)
    tx = perception._predicted[instant.transmitter.vehicle_id]

    expected = sum(
        1 for other in perception._predicted.values()
        if other.vehicle_id != tx.vehicle_id
        and math.hypot(other.x_m - tx.x_m, other.y_m - tx.y_m) <= CONTENTION_RADIUS_M
    )
    assert perception._contenders(tx) == expected
