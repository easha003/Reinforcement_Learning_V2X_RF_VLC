"""Work plan §6.3: the action decides what the policy will know next."""

from __future__ import annotations

import pytest

from hybrid_v2x_rl.core.enums import Action, Link
from hybrid_v2x_rl.observation.link_state import (
    LEGS_USED,
    LinkHistory,
    LinkStateTracker,
    replay,
)


def tracker(history_packets: int = 8) -> LinkStateTracker:
    return LinkStateTracker(history_packets=history_packets)


# -- the sequential claim of the paper ----------------------------------------


def test_using_one_leg_refreshes_it_and_lets_the_other_age() -> None:
    """This is the whole action-dependence of §6.3.

    If it did not hold, the problem would be contextual rather than an MDP and
    the paper would have to say so.
    """

    state = tracker()
    state.record(action=Action.DUP, at_s=0.0, delivered=True,
                 measurements={Link.RF: 1.0, Link.VLC: 2.0})
    state.record(action=Action.RF, at_s=1.0, delivered=True, measurements={Link.RF: 1.5})

    ages = state.ages(now_s=1.0)
    assert ages[Link.RF] == pytest.approx(0.0), "the leg that transmitted is fresh"
    assert ages[Link.VLC] == pytest.approx(1.0), "the unused leg aged"


def test_duplication_refreshes_both_legs() -> None:
    state = tracker()
    state.record(action=Action.DUP, at_s=2.0, delivered=True,
                 measurements={Link.RF: 1.0, Link.VLC: 2.0})

    ages = state.ages(now_s=2.0)
    assert ages[Link.RF] == pytest.approx(0.0)
    assert ages[Link.VLC] == pytest.approx(0.0)


def test_a_measurement_for_an_unused_leg_is_rejected() -> None:
    """Silently accepting it would dissolve §6.3 with every test still green."""

    state = tracker()
    with pytest.raises(ValueError, match="does not transmit on vlc"):
        state.record(action=Action.RF, at_s=0.0, delivered=True,
                     measurements={Link.VLC: 1.0})


def test_never_duplicating_leaves_one_leg_permanently_unmeasured() -> None:
    state = tracker()
    for index in range(10):
        state.record(action=Action.RF, at_s=float(index), delivered=True,
                     measurements={Link.RF: 1.0})

    assert state.ages(now_s=9.0)[Link.VLC] is None
    assert state.stalest_link(now_s=9.0) is Link.VLC


@pytest.mark.parametrize(
    "action,expected",
    [
        (Action.RF, {Link.RF}),
        (Action.VLC, {Link.VLC}),
        (Action.DUP, {Link.RF, Link.VLC}),
    ],
)
def test_each_action_refreshes_exactly_the_legs_it_transmits_on(
    action: Action, expected: set[Link]
) -> None:
    assert set(LEGS_USED[action]) == expected


# -- unmeasured is not the same as old ----------------------------------------


def test_an_unmeasured_leg_has_no_age_rather_than_a_large_one() -> None:
    """Encoding "never measured" as a big number conflates two different states."""

    state = tracker()
    assert state.ages(now_s=5.0) == {Link.RF: None, Link.VLC: None}
    assert state.stalest_link(now_s=5.0) is None


def test_an_unmeasured_leg_outranks_any_measured_one_for_staleness() -> None:
    state = tracker()
    state.record(action=Action.RF, at_s=0.0, delivered=True, measurements={Link.RF: 1.0})

    assert state.stalest_link(now_s=1000.0) is Link.VLC


# -- history ------------------------------------------------------------------


def test_history_is_bounded_and_keeps_the_newest() -> None:
    history = LinkHistory(capacity=3)
    for index in range(6):
        history.record(float(index), at_s=float(index))

    assert history.values() == (3.0, 4.0, 5.0)
    assert len(history) == 3


def test_history_pads_to_a_fixed_width_before_it_fills() -> None:
    """The policy needs a fixed-width vector from the first packet."""

    history = LinkHistory(capacity=4)
    history.record(7.0, at_s=0.0)

    assert history.padded_values() == (0.0, 0.0, 0.0, 7.0)
    assert history.padded_values(fill=-1.0) == (-1.0, -1.0, -1.0, 7.0)


def test_a_full_history_needs_no_padding() -> None:
    history = LinkHistory(capacity=2)
    history.record(1.0, at_s=0.0)
    history.record(2.0, at_s=1.0)

    assert history.padded_values() == (1.0, 2.0)


def test_the_history_length_follows_the_configured_packet_count() -> None:
    state = tracker(history_packets=8)
    assert state.rf.capacity == 8
    assert len(state.rf.padded_values()) == 8


def test_a_non_finite_quality_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        LinkHistory(capacity=2).record(float("nan"), at_s=0.0)


@pytest.mark.parametrize("bad", [0, -1])
def test_an_unusable_capacity_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError, match="capacity"):
        LinkHistory(capacity=bad)
    with pytest.raises(ValueError, match="history_packets"):
        LinkStateTracker(history_packets=bad)


# -- outcome bookkeeping ------------------------------------------------------


def test_consecutive_misses_accumulate_and_reset_on_delivery() -> None:
    state = tracker()
    for index in range(3):
        state.record(action=Action.RF, at_s=float(index), delivered=False,
                     measurements={Link.RF: 1.0})
    assert state.consecutive_misses == 3

    state.record(action=Action.RF, at_s=3.0, delivered=True, measurements={Link.RF: 1.0})
    assert state.consecutive_misses == 0


def test_the_previous_action_and_last_outcome_are_carried() -> None:
    state = tracker()
    assert state.previous_action is None
    assert state.last_delivered is None

    state.record(action=Action.VLC, at_s=0.0, delivered=False, measurements={Link.VLC: 1.0})
    assert state.previous_action is Action.VLC
    assert state.last_delivered is False


def test_packets_are_counted_even_without_measurements() -> None:
    """A leg can transmit and yield nothing; the packet still happened."""

    state = tracker()
    state.record(action=Action.RF, at_s=0.0, delivered=False)

    assert state.packets_seen == 1
    assert state.ages(now_s=0.0)[Link.RF] is None


def test_replay_applies_a_sequence_in_order() -> None:
    state = replay(
        tracker(),
        [
            (Action.DUP, 0.0, True, {Link.RF: 1.0, Link.VLC: 2.0}),
            (Action.VLC, 1.0, False, {Link.VLC: 2.5}),
            (Action.VLC, 2.0, False, {Link.VLC: 3.0}),
        ],
    )

    assert state.consecutive_misses == 2
    assert state.previous_action is Action.VLC
    assert state.vlc.values() == (2.0, 2.5, 3.0)
    assert state.ages(now_s=2.0)[Link.RF] == pytest.approx(2.0)
