"""What the policy knows about each leg, and how the action changes that.

Work plan §6.3 is the reason this module is not just a pair of numbers:

    RF busy ratio remains continuously observable.  A fresh post-transmission
    quality measurement is obtained on whichever legs were used; the unused
    leg's most recent measurement ages.  The current action therefore affects
    future information freshness.

That last sentence is the entire sequential claim of the paper.  If the action
did not change what the policy will know next, the problem would be contextual
rather than an MDP, and §6.3 commits to saying so if a contextual baseline
matches PPO.  **This module is where the dependence is created**, so it is
implemented literally: recording an outcome refreshes the legs the action used
and leaves the others to age.

Choosing VLC therefore buys a fresh optical reading at the price of letting the
radio estimate go stale, and choosing DUP refreshes both at cost 2.  A policy
that never duplicates flies half-blind on whichever leg it stopped using, and
the ages are observable so it can notice.

Quality is deliberately an opaque scalar here.  M3 and M4 decide what it means
-- an SINR proxy, a received-power proxy -- and this module only has to store
it, age it, and hand back a bounded history.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from hybrid_v2x_rl.core.enums import Action, Link
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources

#: Legs each action actually transmits on, and therefore refreshes.
LEGS_USED: dict[Action, frozenset[Link]] = {
    Action.RF: frozenset({Link.RF}),
    Action.VLC: frozenset({Link.VLC}),
    Action.DUP: frozenset({Link.RF, Link.VLC}),
}

ObservedAction = Action | PolicyAction


@dataclass(frozen=True, slots=True)
class QualityReading:
    """One post-transmission measurement of one leg."""

    value: float
    at_s: float

    def age_s(self, now_s: float) -> float:
        return max(0.0, now_s - self.at_s)


@dataclass(slots=True)
class LinkHistory:
    """Recent readings for one leg, newest last.

    ``capacity`` is §6.1's eight-packet history.  The deque is bounded, so an
    episode cannot grow the observation, and the oldest reading is dropped
    rather than aggregated -- a summary statistic would hide exactly the
    pattern the history exists to expose.
    """

    capacity: int
    _readings: deque[QualityReading] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be at least one")
        self._readings = deque(self._readings, maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._readings)

    def record(self, value: float, *, at_s: float) -> None:
        if not math.isfinite(value):
            raise ValueError("quality must be finite")
        self._readings.append(QualityReading(value=value, at_s=at_s))

    @property
    def latest(self) -> QualityReading | None:
        return self._readings[-1] if self._readings else None

    def age_s(self, now_s: float) -> float | None:
        """Staleness of the newest reading, or ``None`` if the leg is unmeasured.

        ``None`` rather than a sentinel: a leg that has never been used has no
        age, and encoding that as a large number would let the policy treat
        "never measured" and "measured a long time ago" as the same state when
        they are not.  Turning it into a finite feature is the observation
        builder's job, and it must do so explicitly.
        """

        newest = self.latest
        return None if newest is None else newest.age_s(now_s)

    def values(self) -> tuple[float, ...]:
        """Readings newest last, shorter than ``capacity`` until it fills."""

        return tuple(reading.value for reading in self._readings)

    def padded_values(self, fill: float = 0.0) -> tuple[float, ...]:
        """Exactly ``capacity`` values, oldest first, left-padded with ``fill``.

        The policy needs a fixed-width vector from the first packet of an
        episode, before any history exists.
        """

        recorded = self.values()
        missing = self.capacity - len(recorded)
        return (fill,) * missing + recorded


@dataclass(slots=True)
class LinkStateTracker:
    """Per-leg quality, freshness, and the outcome history §6.1 observes."""

    history_packets: int
    rf: LinkHistory = field(init=False)
    vlc: LinkHistory = field(init=False)
    previous_action: ObservedAction | None = None
    last_delivered: bool | None = None
    consecutive_misses: int = 0
    packets_seen: int = 0

    def __post_init__(self) -> None:
        if self.history_packets < 1:
            raise ValueError("history_packets must be at least one")
        self.rf = LinkHistory(capacity=self.history_packets)
        self.vlc = LinkHistory(capacity=self.history_packets)

    def history(self, link: Link) -> LinkHistory:
        return self.rf if link is Link.RF else self.vlc

    def record(
        self,
        *,
        action: Action,
        at_s: float,
        delivered: bool,
        measurements: dict[Link, float] | None = None,
    ) -> None:
        """Apply one packet's outcome.

        Only legs in ``LEGS_USED[action]`` may be refreshed.  A measurement
        offered for an unused leg is rejected rather than ignored: silently
        accepting it would dissolve the action-dependence of §6.3 while every
        test still passed, and the paper's sequential claim with it.
        """

        self._record(
            action=action,
            used=LEGS_USED[action],
            at_s=at_s,
            delivered=delivered,
            measurements=measurements,
        )

    def record_policy(
        self,
        *,
        action: PolicyAction,
        at_s: float,
        delivered: bool,
        measurements: dict[Link, float] | None = None,
    ) -> None:
        """Apply feedback for one action in the nine-action RL contract.

        Unlike the inherited three-action interface, the exact persistent
        policy index is retained in ``previous_action``.  RF-1 through RF-4
        therefore remain distinguishable to the next actor observation even
        though they refresh the same physical leg.
        """

        if not isinstance(action, PolicyAction):
            raise TypeError("policy feedback requires a PolicyAction")
        resources = action_resources(action)
        used = frozenset(
            link
            for link, active in (
                (Link.RF, resources.uses_rf),
                (Link.VLC, resources.uses_vlc),
            )
            if active
        )
        self._record(
            action=action,
            used=used,
            at_s=at_s,
            delivered=delivered,
            measurements=measurements,
        )

    def _record(
        self,
        *,
        action: ObservedAction,
        used: frozenset[Link],
        at_s: float,
        delivered: bool,
        measurements: dict[Link, float] | None,
    ) -> None:
        """Shared state update after either action vocabulary is resolved."""

        offered = dict(measurements or {})
        for link in offered:
            if link not in used:
                raise ValueError(
                    f"action {action.name} does not transmit on {link.value}, "
                    "so it cannot produce a fresh measurement for it"
                )

        for link, value in offered.items():
            self.history(link).record(value, at_s=at_s)

        self.previous_action = action
        self.last_delivered = delivered
        self.consecutive_misses = 0 if delivered else self.consecutive_misses + 1
        self.packets_seen += 1

    def ages(self, now_s: float) -> dict[Link, float | None]:
        return {Link.RF: self.rf.age_s(now_s), Link.VLC: self.vlc.age_s(now_s)}

    def stalest_link(self, now_s: float) -> Link | None:
        """Which leg the policy currently knows least about.

        ``None`` when both are unmeasured; an unmeasured leg beats any measured
        one, since no information is worse than old information.
        """

        rf_age, vlc_age = self.rf.age_s(now_s), self.vlc.age_s(now_s)
        if rf_age is None and vlc_age is None:
            return None
        if rf_age is None:
            return Link.RF
        if vlc_age is None:
            return Link.VLC
        return Link.RF if rf_age > vlc_age else Link.VLC


def replay(
    tracker: LinkStateTracker,
    outcomes: Iterable[tuple[Action, float, bool, dict[Link, float]]],
) -> LinkStateTracker:
    """Apply a sequence of packet outcomes, for tests and for warm starts."""

    for action, at_s, delivered, measurements in outcomes:
        tracker.record(action=action, at_s=at_s, delivered=delivered, measurements=measurements)
    return tracker


__all__ = [
    "LEGS_USED",
    "LinkHistory",
    "LinkStateTracker",
    "ObservedAction",
    "QualityReading",
    "replay",
]
