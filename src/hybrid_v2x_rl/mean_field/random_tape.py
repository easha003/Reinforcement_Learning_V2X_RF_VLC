"""Identity-addressed random tapes for the nine-action RL environment.

The tape is generated before an action is known.  One packet always receives
four RF entries and one VLC entry, so RF-n reads the first ``n`` RF entries and
DUP-n reads that identical prefix plus the identical optical entry.  An action
therefore selects a view of already-fixed randomness; it never changes which
draws exist or advances a shared generator.

Every scalar draw has its own namespace containing the physical link,
mechanism, and attempt index.  Adding a new mechanism cannot shift an existing
draw, and generating another pair first cannot change this packet.  The
remaining address components are the root seed, trace ID, stable pair-episode
ID, and packet index required by the frozen environment contract.

Counterfactual views are simulator/oracle data.  This module deliberately does
not expose them as observations or critic features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from hybrid_v2x_rl.channels.rf.model import RFPacketRandomness
from hybrid_v2x_rl.channels.vlc.model import VLCPacketRandomness
from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    MAX_RESERVED_RF_ATTEMPTS,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.core.randomness import derive_child_seed, make_generator
from hybrid_v2x_rl.mean_field.frames import PopulationFrame, PopulationPair

MATCHED_TAPE_SCHEMA: Final = "hybrid-rf-vlc-rl.matched-packet-tape.v1"


class RandomTapeError(HybridV2XError):
    """A random-tape identity or action view violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class PacketRandomnessIdentity:
    """Stable address of one pair episode's packet."""

    trace_id: str
    pair_episode_id: str
    packet_index: int

    def __post_init__(self) -> None:
        for name in ("trace_id", "pair_episode_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RandomTapeError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.packet_index, int)
            or isinstance(self.packet_index, bool)
            or self.packet_index < 0
        ):
            raise RandomTapeError("packet_index must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ActionRandomness:
    """The immutable prefix and optional VLC entry selected by one action."""

    identity: PacketRandomnessIdentity
    action: PolicyAction
    rf_attempts: tuple[RFPacketRandomness, ...]
    vlc: VLCPacketRandomness | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PacketRandomnessIdentity):
            raise RandomTapeError("action randomness requires a packet identity")
        if type(self.action) is not PolicyAction:
            raise RandomTapeError("action randomness requires an exact PolicyAction")
        if not isinstance(self.rf_attempts, tuple) or not all(
            isinstance(attempt, RFPacketRandomness) for attempt in self.rf_attempts
        ):
            raise RandomTapeError("RF action randomness must be an immutable RF tuple")

        spec = action_resources(self.action)
        if len(self.rf_attempts) != spec.reserved_rf_attempts:
            raise RandomTapeError(
                "RF random prefix length does not match the action reservation",
                context={
                    "action": spec.name,
                    "actual": len(self.rf_attempts),
                    "expected": spec.reserved_rf_attempts,
                },
            )
        if spec.uses_vlc != (self.vlc is not None):
            raise RandomTapeError(
                "VLC randomness presence does not match the action reservation",
                context={"action": spec.name, "uses_vlc": spec.uses_vlc},
            )
        if self.vlc is not None and not isinstance(self.vlc, VLCPacketRandomness):
            raise RandomTapeError("VLC action randomness has an invalid entry")


@dataclass(frozen=True, slots=True)
class MatchedPacketTape:
    """All action-independent outcome draws owned by one packet."""

    identity: PacketRandomnessIdentity
    rf_attempts: tuple[RFPacketRandomness, ...]
    vlc: VLCPacketRandomness

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PacketRandomnessIdentity):
            raise RandomTapeError("matched tape requires a packet identity")
        if not isinstance(self.rf_attempts, tuple) or len(
            self.rf_attempts
        ) != MAX_RESERVED_RF_ATTEMPTS:
            raise RandomTapeError(
                "matched tape must contain exactly four RF attempts",
                context={
                    "actual": (
                        len(self.rf_attempts)
                        if isinstance(self.rf_attempts, tuple)
                        else None
                    ),
                    "expected": MAX_RESERVED_RF_ATTEMPTS,
                },
            )
        if not all(
            isinstance(attempt, RFPacketRandomness) for attempt in self.rf_attempts
        ):
            raise RandomTapeError("matched tape contains an invalid RF entry")
        if not isinstance(self.vlc, VLCPacketRandomness):
            raise RandomTapeError("matched tape contains an invalid VLC entry")

    def view_for_action(self, action: PolicyAction) -> ActionRandomness:
        """Select one action's RF prefix and optional shared optical entry."""

        if type(action) is not PolicyAction:
            raise RandomTapeError("random-tape views require an exact PolicyAction")
        spec = action_resources(action)
        return ActionRandomness(
            identity=self.identity,
            action=action,
            rf_attempts=self.rf_attempts[: spec.reserved_rf_attempts],
            vlc=self.vlc if spec.uses_vlc else None,
        )

    def counterfactual_views(self) -> dict[PolicyAction, ActionRandomness]:
        """Return every action view without drawing or mutating randomness."""

        return {action: self.view_for_action(action) for action in PolicyAction}


@dataclass(frozen=True, slots=True)
class MatchedPacketTapeFactory:
    """Generate packet tapes from stable identity rather than evaluation order."""

    root_seed: int

    def __post_init__(self) -> None:
        try:
            derive_child_seed(
                self.root_seed,
                f"{MATCHED_TAPE_SCHEMA}.root-validation",
            )
        except (TypeError, ValueError) as error:
            raise RandomTapeError("root_seed is not a valid unsigned seed") from error

    def _uniform(
        self,
        identity: PacketRandomnessIdentity,
        *,
        link: Link,
        mechanism: str,
        attempt_index: int,
    ) -> float:
        stream_name = (
            f"{MATCHED_TAPE_SCHEMA}.{link.value}.attempt-{attempt_index}."
            f"{mechanism}"
        )
        generator = make_generator(
            self.root_seed,
            stream_name,
            trace_id=identity.trace_id,
            episode_id=identity.pair_episode_id,
            packet_index=identity.packet_index,
        )
        return float(generator.random())

    def build(self, identity: PacketRandomnessIdentity) -> MatchedPacketTape:
        """Generate all four RF entries and the VLC entry before action choice."""

        if not isinstance(identity, PacketRandomnessIdentity):
            raise RandomTapeError("matched tape generation requires a packet identity")
        rf_attempts = tuple(
            RFPacketRandomness(
                collision_draw=self._uniform(
                    identity,
                    link=Link.RF,
                    mechanism="collision",
                    attempt_index=attempt_index,
                ),
                decoding_draw=self._uniform(
                    identity,
                    link=Link.RF,
                    mechanism="decoding",
                    attempt_index=attempt_index,
                ),
                half_duplex_draw=self._uniform(
                    identity,
                    link=Link.RF,
                    mechanism="half-duplex",
                    attempt_index=attempt_index,
                ),
            )
            for attempt_index in range(MAX_RESERVED_RF_ATTEMPTS)
        )
        vlc = VLCPacketRandomness(
            decoding_draw=self._uniform(
                identity,
                link=Link.VLC,
                mechanism="decoding",
                attempt_index=0,
            )
        )
        return MatchedPacketTape(
            identity=identity,
            rf_attempts=rf_attempts,
            vlc=vlc,
        )

    def for_population_pair(
        self,
        frame: PopulationFrame,
        pair: PopulationPair,
    ) -> MatchedPacketTape:
        """Address a tape from the authoritative frame and pair lifecycle."""

        if not isinstance(frame, PopulationFrame) or not isinstance(pair, PopulationPair):
            raise RandomTapeError(
                "population tape generation requires a frame and one of its pairs"
            )
        if pair not in frame.pairs:
            raise RandomTapeError(
                "pair does not belong to the supplied population frame",
                context={"pair_id": pair.pair_id},
            )
        return self.build(
            PacketRandomnessIdentity(
                trace_id=frame.source.trace_id,
                pair_episode_id=pair.episode_id,
                packet_index=pair.episode_step,
            )
        )


__all__ = [
    "MATCHED_TAPE_SCHEMA",
    "ActionRandomness",
    "MatchedPacketTape",
    "MatchedPacketTapeFactory",
    "PacketRandomnessIdentity",
    "RandomTapeError",
]
