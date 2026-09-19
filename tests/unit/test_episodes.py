"""Streaming a stored trace into pair instants.

The load-bearing tests are the ones about *what gets sampled*: the packet
cadence and the warm-up. Both are places where a plausible-looking loop reports
a number computed on the wrong population, and neither failure announces itself
-- the run completes, the table prints, and the rate is wrong by a factor
nobody can see.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.env.episodes import EpisodeError, TraceSource, iter_pair_instants

TRACES = Path.cwd() / "artifacts" / "traces"
TRACE = TRACES / "synthetic-d10-train-000"

pytestmark = pytest.mark.skipif(
    not TRACE.exists(), reason="campaign traces are not present in this checkout"
)


@pytest.fixture(scope="module")
def source() -> TraceSource:
    return TraceSource.discover(TRACE)


def take(source, n, **kwargs):
    out = []
    for instant in iter_pair_instants(source, generation_period_s=0.1, **kwargs):
        out.append(instant)
        if len(out) >= n:
            break
    return out


# -- identity -----------------------------------------------------------------


def test_the_density_group_is_read_from_the_trace_id(source) -> None:
    """It has to come from somewhere durable: the constraint is enforced per
    density group, so a trace joining the wrong one moves a constraint without
    changing a number anywhere."""

    assert source.density == 10.0
    assert source.trace_id == "synthetic-d10-train-000"


def test_an_unlabelled_trace_is_refused_rather_than_defaulted(tmp_path) -> None:
    with pytest.raises(EpisodeError, match="density group"):
        TraceSource.discover(tmp_path / "some-trace-without-a-density")


# -- what gets sampled --------------------------------------------------------


def test_packets_follow_the_service_period_not_the_trace_timestep(source) -> None:
    """The trace resolves car-following; the service generates every 100 ms.

    One packet per trace frame would inflate every count by the ratio between
    them and report a 1e-4 budget as met on twice the evidence.
    """

    times = sorted({instant.time_s for instant in take(source, 400)})
    gaps = {round(b - a, 6) for a, b in zip(times, times[1:], strict=False)}
    assert gaps, "expected more than one distinct instant"
    assert min(gaps) >= 0.1 - 1e-9


def test_the_warmup_skips_the_seeded_formation(source) -> None:
    """Measured on the first packets, the geometry is the one the simulator was
    seeded with -- evenly spaced, aligned, every pair straight down its lane.
    P_out reads 4.6% there against 17.4% over the whole trace."""

    late = take(source, 50, warmup_s=400.0)
    assert late, "the trace should run well past the warm-up"
    assert min(instant.time_s for instant in late) >= 400.0


def test_a_warmup_past_the_end_yields_nothing_rather_than_the_start(source) -> None:
    assert take(source, 5, warmup_s=1e9) == []


def test_max_packets_truncates_at_a_frame_boundary(source) -> None:
    """It overshoots, and the overshoot is the point.

    The cut lands between frames, never inside one, so every instant the run
    reports belongs to a frame all of whose pairs were also reported. Stopping
    mid-frame would leave one timestamp represented by whichever pairs happened
    to be enumerated first, and any per-frame statistic -- contender counts most
    of all -- would be computed on a partial population without saying so.
    """

    got = list(iter_pair_instants(source, generation_period_s=0.1, max_packets=50))
    assert len(got) >= 50
    reference = list(iter_pair_instants(source, generation_period_s=0.1, max_packets=100_000))
    assert [(i.pair_id, i.time_s) for i in got] == [
        (i.pair_id, i.time_s) for i in reference[: len(got)]
    ], "a truncated run must be a prefix of a longer one"
    boundary = got[-1].time_s
    assert sum(1 for i in got if i.time_s == boundary) == sum(
        1 for i in reference if i.time_s == boundary
    ), "the final frame must be complete"


# -- what each instant carries ------------------------------------------------


def test_a_pair_indexes_its_own_packets_from_zero(source) -> None:
    """The index keys the tape, so it must count this pair's packets and not
    the loop's iterations -- otherwise two pairs in the same frame would be the
    same packet identity."""

    by_pair: dict[str, list[int]] = {}
    for instant in take(source, 3000):
        by_pair.setdefault(instant.pair_id, []).append(instant.index)
    repeated = [v for v in by_pair.values() if len(v) > 2]
    assert repeated, "expected at least one pair to persist across instants"
    for indices in repeated:
        assert indices == list(range(len(indices)))


def test_the_pair_endpoints_are_present_in_the_frame(source) -> None:
    for instant in take(source, 200):
        ids = {v.vehicle_id for v in instant.neighbours}
        assert instant.transmitter.vehicle_id in ids
        assert instant.receiver.vehicle_id in ids


def test_pairs_in_one_frame_share_one_spatial_index(source) -> None:
    """Rebuilding it per pair would dominate the run at density 30, where a
    frame holds a thousand vehicles and hundreds of pairs read the same one."""

    instants = take(source, 400)
    per_time: dict[float, set[int]] = {}
    for instant in instants:
        per_time.setdefault(instant.time_s, set()).add(id(instant.index_of_frame))
    assert all(len(indices) == 1 for indices in per_time.values())


def test_the_last_instant_of_a_pair_is_flagged(source) -> None:
    """The rollout releases correlated state on it; without the flag a pair
    that re-forms inherits shadowing from an unrelated earlier encounter."""

    finals = [i for i in take(source, 5000) if i.final]
    assert finals, "expected some pair episodes to end within the sample"
