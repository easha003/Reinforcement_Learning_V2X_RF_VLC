"""Fixed-time traffic signals for the analytic Manhattan-grid model.

Implements section 4.4 of `research/MOBILITY_MODEL_V2_PROPOSAL.md`: a
deterministic two-phase program with staggered offsets, so greens do not
arrive simultaneously across the grid.

Signal state is a pure function of ``(time, junction)``.  There is no adaptive
or actuated control, and no vehicle influences a signal.  That keeps the whole
mobility model reproducible from a seed and a clock.

The offset pattern reproduces the half-offset stagger the SUMO backend
requested from ``netgenerate`` (applied where ``avenue + cross_street`` is
odd), so the two backends remain comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from hybrid_v2x_rl.mobility.grid_network import Direction, GridNetwork, Junction


class SignalPhase(Enum):
    """Which movement currently holds right of way."""

    NORTH_SOUTH_GREEN = "ns_green"
    NORTH_SOUTH_YELLOW = "ns_yellow"
    EAST_WEST_GREEN = "ew_green"
    EAST_WEST_YELLOW = "ew_yellow"

    @property
    def is_green(self) -> bool:
        return self in (SignalPhase.NORTH_SOUTH_GREEN, SignalPhase.EAST_WEST_GREEN)

    @property
    def serves_avenue(self) -> bool:
        """Whether this phase belongs to north-south (avenue) movement."""

        return self in (
            SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.NORTH_SOUTH_YELLOW,
        )


@dataclass(frozen=True, slots=True)
class SignalProgram:
    """Two-phase fixed-time program shared by every signalized junction."""

    #: Matched to block traversal rather than inherited, and matched to
    #: ``configs/mobility/synthetic_manhattan.yaml`` so that a bare
    #: ``SignalProgram()`` measures the *configured* model.  It did not: the
    #: default was 90 s while the configuration said 45 s, nothing wires the
    #: two together, and only ``GridTracePipeline`` reads the configured value.
    #: Every probe and unit test that constructs a simulator directly was
    #: therefore measuring a different model from the one the traces come from,
    #: which produced one wrong version of the §4.5 fundamental diagram before
    #: it was caught.  A default that disagrees with the shipped configuration
    #: is a trap whatever its value.
    #:
    #: The value itself: a vehicle arriving at random waits
    #: ``E[wait] = (1-g)^2 C/2``, about ``C/8`` at a two-phase signal.  That is
    #: 11.3 s at 90 s against a 5.5 s free-flow block traversal, a mismatch
    #: that left vehicles stopped two-thirds of the time; at 45 s the wait is
    #: comparable to the traversal.
    cycle_s: float = 45.0
    yellow_s: float = 3.0
    stagger_offsets: bool = True
    #: Progression speed for green-wave offsets, in m/s.  ``None`` keeps the
    #: half-cycle stagger, which is what work plan §4.7 item 2 records as the
    #: largest realism gap: free-flow speed tops out at 52% of the limit and
    #: 38% of vehicles are stopped in near-empty traffic, because a platoon
    #: released by one junction arrives at the next during its red.
    #:
    #: A green wave sets each junction's offset from its distance along the
    #: progressed axis, so a platoon travelling at this speed meets a green at
    #: every junction.  Real Manhattan avenues are timed this way.
    progression_speed_mps: float | None = None
    #: Spacing between junctions along the progressed (avenue) axis, needed to
    #: turn a progression speed into an offset.
    progression_spacing_m: float = 61.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.cycle_s) or self.cycle_s <= 0.0:
            raise ValueError("cycle_s must be finite and positive")
        if not math.isfinite(self.yellow_s) or self.yellow_s < 0.0:
            raise ValueError("yellow_s must be finite and non-negative")
        if self.progression_speed_mps is not None and (
            not math.isfinite(self.progression_speed_mps) or self.progression_speed_mps <= 0.0
        ):
            raise ValueError("progression_speed_mps must be finite and positive when set")
        if not math.isfinite(self.progression_spacing_m) or self.progression_spacing_m <= 0.0:
            raise ValueError("progression_spacing_m must be finite and positive")
        if 2.0 * self.yellow_s >= self.cycle_s:
            raise ValueError("yellow intervals must leave positive green time")

    @property
    def green_s(self) -> float:
        """Green duration of each of the two phases."""

        return (self.cycle_s - 2.0 * self.yellow_s) / 2.0

    def offset_s(self, junction: Junction) -> float:
        """When this junction's cycle begins, relative to the network clock.

        With ``progression_speed_mps`` set, the offset is the travel time a
        platoon needs to reach this junction along the progressed axis, so the
        green arrives with it.  Otherwise the original half-cycle stagger
        applies, which deliberately does *not* progress: it alternates, so a
        platoon meets a red at every second junction.
        """

        if self.progression_speed_mps is not None:
            # Progress along the avenue axis, in the direction that avenue
            # actually runs.  Avenues alternate, so a single increasing offset
            # would progress the northbound ones and anti-progress the
            # southbound ones -- the same failure a two-way street has, moved
            # from between directions to between adjacent avenues.
            northbound = junction.avenue_index % 2 == 0
            steps = (
                junction.cross_street_index
                if northbound
                else -junction.cross_street_index
            )
            travel_s = steps * self.progression_spacing_m / self.progression_speed_mps
            return travel_s % self.cycle_s

        if not self.stagger_offsets:
            return 0.0
        odd = (junction.avenue_index + junction.cross_street_index) % 2 == 1
        return self.cycle_s / 2.0 if odd else 0.0

    def phase_at(self, junction: Junction, time_s: float) -> SignalPhase:
        """Return the phase in force at ``time_s``."""

        position = (time_s + self.offset_s(junction)) % self.cycle_s
        green = self.green_s
        if position < green:
            return SignalPhase.NORTH_SOUTH_GREEN
        if position < green + self.yellow_s:
            return SignalPhase.NORTH_SOUTH_YELLOW
        if position < 2.0 * green + self.yellow_s:
            return SignalPhase.EAST_WEST_GREEN
        return SignalPhase.EAST_WEST_YELLOW


class SignalController:
    """Evaluates entry permission for approaches to signalized junctions."""

    def __init__(self, network: GridNetwork, program: SignalProgram | None = None) -> None:
        self.network = network
        self.program = program or SignalProgram()

    def may_enter(self, junction_id: str, direction: Direction, time_s: float) -> bool:
        """Whether a vehicle approaching in ``direction`` may cross now.

        Unsignalized boundary junctions always permit entry.  Yellow is treated
        as stop for the purpose of *entering*: a vehicle already past the stop
        line has advanced onto its next edge and is unaffected.
        """

        junction = self.network.junction(junction_id)
        if not junction.signalized:
            return True
        phase = self.program.phase_at(junction, time_s)
        if not phase.is_green:
            return False
        return phase.serves_avenue == direction.is_avenue

    def phase_records(self, time_s: float) -> tuple[tuple[str, str], ...]:
        """Return ``(junction_id, phase)`` pairs for trace signal records."""

        return tuple(
            (jid, self.program.phase_at(self.network.junction(jid), time_s).value)
            for jid in self.network.signalized_junction_ids
        )


__all__ = [
    "SignalController",
    "SignalPhase",
    "SignalProgram",
]
