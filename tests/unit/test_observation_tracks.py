"""Work plan §6.1: track age is an observed feature, not diagnostics."""

from __future__ import annotations

import math

import pytest

from hybrid_v2x_rl.observation.sensing import TrackSample
from hybrid_v2x_rl.observation.tracks import Track, TrackStore


def sample(
    vehicle_id: str,
    measured_at_s: float,
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    speed_mps: float = 10.0,
    heading_rad: float = 0.0,
    latency_s: float = 0.05,
) -> TrackSample:
    return TrackSample(
        vehicle_id=vehicle_id,
        measured_at_s=measured_at_s,
        observed_at_s=measured_at_s + latency_s,
        x_m=x_m,
        y_m=y_m,
        speed_mps=speed_mps,
        heading_rad=heading_rad,
    )


# -- ageing, which is the point ----------------------------------------------


def test_a_vehicle_that_stops_updating_goes_stale_rather_than_vanishing() -> None:
    """Dropping the track would hand the policy a clean "unknown".

    What a real receiver has instead is a confident-looking estimate that is
    quietly going wrong, and the age is the only warning.
    """

    store = TrackStore()
    store.update([sample("v1", 1.0)], now_s=1.05)

    assert "v1" in store
    for now in (1.05, 2.0, 5.0):
        track = store.require("v1")
        assert track.age_s(now) == pytest.approx(now - 1.0)


def test_age_is_measured_from_truth_not_from_arrival() -> None:
    """A policy treating a 50 ms-late sample as current is wrong by 50 ms."""

    store = TrackStore()
    store.update([sample("v1", 1.0, latency_s=0.05)], now_s=1.05)

    assert store.require("v1").age_s(1.05) == pytest.approx(0.05)


def test_age_never_goes_negative() -> None:
    store = TrackStore()
    store.update([sample("v1", 1.0)], now_s=1.05)
    assert store.require("v1").age_s(0.5) == 0.0


def test_a_stale_sample_never_overwrites_a_fresher_one() -> None:
    """Out-of-order arrival must not make an observed feature move backwards."""

    store = TrackStore()
    store.update([sample("v1", 2.0, x_m=20.0)], now_s=2.05)
    store.update([sample("v1", 1.0, x_m=10.0)], now_s=2.05)

    track = store.require("v1")
    assert track.sample.measured_at_s == pytest.approx(2.0)
    assert track.sample.x_m == pytest.approx(20.0)
    assert track.update_count == 1, "a discarded sample is not an update"


def test_a_newer_sample_replaces_and_counts() -> None:
    store = TrackStore()
    store.update([sample("v1", 1.0)], now_s=1.05)
    store.update([sample("v1", 1.05)], now_s=1.10)

    track = store.require("v1")
    assert track.update_count == 2
    assert track.first_seen_s == pytest.approx(1.0), "first_seen must not drift"


def test_re_appearing_after_a_gap_keeps_the_original_first_seen() -> None:
    store = TrackStore()
    store.update([sample("v1", 1.0)], now_s=1.05)
    store.update([sample("v1", 9.0)], now_s=9.05)

    track = store.require("v1")
    assert track.first_seen_s == pytest.approx(1.0)
    assert track.age_s(9.05) == pytest.approx(0.05)


# -- forgetting ---------------------------------------------------------------


def test_a_store_built_without_configuration_forgets_nothing() -> None:
    """A caller with no configuration cannot inherit a number nobody chose.

    The declared timeout arrives through :meth:`TrackStore.from_config`; the
    bare constructor stays inert so a missing configuration is visible as
    unbounded growth rather than as a silently plausible default.
    """

    store = TrackStore()
    store.update([sample("v1", 1.0)], now_s=1.05)
    store.update([sample("v2", 500.0)], now_s=500.05)

    assert len(store) == 2


def test_a_configured_lifetime_drops_tracks_past_it() -> None:
    store = TrackStore(forget_after_s=2.0)
    store.update([sample("v1", 1.0)], now_s=1.05)
    store.update([sample("v2", 5.0)], now_s=5.05)

    assert "v1" not in store
    assert "v2" in store


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_an_unusable_lifetime_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="forget_after_s"):
        TrackStore(forget_after_s=bad)


# -- derived quantities -------------------------------------------------------


@pytest.mark.parametrize(
    "heading,expected",
    [
        (0.0, (10.0, 0.0)),
        (0.5 * math.pi, (0.0, 10.0)),
        (math.pi, (-10.0, 0.0)),
        (1.5 * math.pi, (0.0, -10.0)),
    ],
)
def test_velocity_follows_measured_speed_and_heading(
    heading: float, expected: tuple[float, float]
) -> None:
    track = Track(
        sample=sample("v1", 1.0, speed_mps=10.0, heading_rad=heading),
        first_seen_s=1.0,
        update_count=1,
    )
    vx, vy = track.velocity_mps
    assert vx == pytest.approx(expected[0], abs=1e-9)
    assert vy == pytest.approx(expected[1], abs=1e-9)


def test_neighbour_count_uses_tracks_rather_than_truth() -> None:
    """The RF-load proxy must degrade with the rest of the observation."""

    store = TrackStore()
    store.update(
        [
            sample("ego", 1.0, x_m=0.0, y_m=0.0),
            sample("near", 1.0, x_m=150.0, y_m=0.0),
            sample("edge", 1.0, x_m=200.0, y_m=0.0),
            sample("far", 1.0, x_m=250.0, y_m=0.0),
        ],
        now_s=1.05,
    )
    ego = store.require("ego")

    assert store.neighbour_count(ego, radius_m=200.0, now_s=1.05) == 2
    assert store.neighbour_count(ego, radius_m=100.0, now_s=1.05) == 0


def test_neighbour_count_can_ignore_stale_tracks() -> None:
    store = TrackStore()
    store.update([sample("ego", 5.0, x_m=0.0)], now_s=5.05)
    store.update([sample("old", 1.0, x_m=10.0)], now_s=5.05)
    ego = store.require("ego")

    assert store.neighbour_count(ego, radius_m=200.0, now_s=5.05) == 1
    assert store.neighbour_count(ego, radius_m=200.0, now_s=5.05, limit_s=1.0) == 0


def test_neighbour_count_excludes_the_centre() -> None:
    store = TrackStore()
    store.update([sample("ego", 1.0)], now_s=1.05)
    ego = store.require("ego")

    assert store.neighbour_count(ego, radius_m=200.0, now_s=1.05) == 0


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan])
def test_an_unusable_radius_is_rejected(bad: float) -> None:
    store = TrackStore()
    store.update([sample("ego", 1.0)], now_s=1.05)
    with pytest.raises(ValueError, match="radius_m"):
        store.neighbour_count(store.require("ego"), radius_m=bad, now_s=1.05)


# -- reproducibility ----------------------------------------------------------


def test_iteration_order_is_stable() -> None:
    store = TrackStore()
    store.update([sample(f"v{i}", 1.0) for i in range(20)], now_s=1.05)

    once = [track.vehicle_id for track in store.tracks()]
    twice = [track.vehicle_id for track in store.tracks()]
    assert once == twice


def test_a_missing_track_raises_rather_than_returning_a_default() -> None:
    store = TrackStore()
    assert store.get("absent") is None
    with pytest.raises(KeyError, match="absent"):
        store.require("absent")


# -- the configured awareness-message timeout ---------------------------------


def test_the_headline_configuration_supplies_a_track_lifetime() -> None:
    """Unbounded, the RF-load proxy only ever rises: a vehicle that drives
    away is counted for the rest of the episode."""

    from pathlib import Path

    from hybrid_v2x_rl.config import load_headline_config

    config = load_headline_config(Path(__file__).resolve().parents[2])
    store = TrackStore.from_config(config.observation)

    assert store.forget_after_s == pytest.approx(1.0)


def test_a_neighbour_that_stops_broadcasting_leaves_the_count() -> None:
    from pathlib import Path

    from hybrid_v2x_rl.config import load_headline_config

    config = load_headline_config(Path(__file__).resolve().parents[2])
    store = TrackStore.from_config(config.observation)

    store.update([sample("ego", 0.0, x_m=0.0), sample("gone", 0.0, x_m=50.0)], now_s=0.05)
    assert store.neighbour_count(store.require("ego"), radius_m=200.0, now_s=0.05) == 1

    # Only the ego keeps being heard from.
    for step in range(1, 40):
        store.update([sample("ego", step * 0.05, x_m=0.0)], now_s=step * 0.05)

    assert "gone" not in store
    assert store.neighbour_count(store.require("ego"), radius_m=200.0, now_s=2.0) == 0
