"""Intelligent Driver Model longitudinal motion.

Implements the car-following law of
`research/MOBILITY_MODEL_V2_PROPOSAL.md` section 4.3.

Reference: M. Treiber, A. Hennecke and D. Helbing, "Congested traffic states in
empirical observations and microscopic simulations", *Physical Review E* 62(2),
pp. 1805-1824, 2000.

The model is chosen because it is a standard, citable reference law, is
collision-free by construction under its own dynamics, and reproduces
stop-and-go queueing.  Queue formation at signals therefore needs no separate
mechanism: a red light is supplied as a stationary leader at the stop line
(see :func:`stop_line_gap`), and the same equation produces the queue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Acceleration exponent from the original formulation.
IDM_DELTA = 4.0


@dataclass(frozen=True, slots=True)
class IDMParameters:
    """Driver-behaviour parameters.

    Defaults align with the vehicle mixture already configured for the SUMO
    backend (``accel`` 2.6, ``decel`` 4.5, ``minGap`` 2.5) so the two mobility
    backends remain comparable.
    """

    desired_speed_mps: float = 11.18
    max_acceleration_mps2: float = 2.6
    comfortable_deceleration_mps2: float = 4.5
    minimum_gap_m: float = 2.5
    #: Desired time headway.  CITED, and deliberately below the reference.
    #: Treiber, Hennecke and Helbing (2000) use 1.6 s for freeway traffic, and
    #: Treiber's own published defaults give 1.5 s for cars with the note that
    #: city traffic should adapt the desired *speed* while the other parameters
    #: "essentially can be left unchanged".  1.2 s is therefore more aggressive
    #: than the reference: shorter gaps, higher capacity, and -- since the
    #: optical budget divides by the bumper gap -- a *better* V-VLC link than a
    #: Treiber-parameterised model would give.  Declared rather than silently
    #: favourable.
    #:
    #: Measured across T in {1.0, 1.2, 1.5} s, the optical budget improves from
    #: rho=10 to rho=30 by 10.88, 9.97 and 9.50 dB respectively.  The claim that
    #: V-VLC strengthens with density therefore does **not** rest on this
    #: parameter: 1.4 dB of spread across the whole band, against a 9.5 dB
    #: effect.  The gap is set by density, not by the car-following model.
    desired_time_headway_s: float = 1.2

    def __post_init__(self) -> None:
        values = (
            self.desired_speed_mps,
            self.max_acceleration_mps2,
            self.comfortable_deceleration_mps2,
            self.desired_time_headway_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("IDM parameters must be finite and positive")
        if not math.isfinite(self.minimum_gap_m) or self.minimum_gap_m < 0.0:
            raise ValueError("minimum_gap_m must be finite and non-negative")


def desired_gap_m(
    speed_mps: float,
    approach_rate_mps: float,
    parameters: IDMParameters,
) -> float:
    """Return the IDM desired dynamic gap ``s*``.

    ``approach_rate_mps`` is the follower's speed minus the leader's, so it is
    positive when closing.  The interaction term is clamped at zero: a vehicle
    falling behind its leader does not want a gap smaller than ``s0``.
    """

    interaction = (
        speed_mps
        * approach_rate_mps
        / (
            2.0
            * math.sqrt(parameters.max_acceleration_mps2 * parameters.comfortable_deceleration_mps2)
        )
    )
    dynamic = speed_mps * parameters.desired_time_headway_s + interaction
    return parameters.minimum_gap_m + max(0.0, dynamic)


def acceleration_mps2(
    speed_mps: float,
    gap_m: float | None,
    leader_speed_mps: float | None,
    parameters: IDMParameters,
) -> float:
    """Return the IDM acceleration.

    ``gap_m`` is the bumper-to-bumper distance to the leader; pass ``None`` for
    free flow.  A non-positive gap yields maximum braking rather than a
    singularity, which keeps the integrator stable if a step ever overshoots.
    """

    free_flow = 1.0 - float((speed_mps / parameters.desired_speed_mps) ** IDM_DELTA)
    if gap_m is None:
        return parameters.max_acceleration_mps2 * free_flow

    if gap_m <= 0.0:
        return -parameters.comfortable_deceleration_mps2 * 2.0

    approach_rate = speed_mps - (leader_speed_mps or 0.0)
    star = desired_gap_m(speed_mps, approach_rate, parameters)
    return parameters.max_acceleration_mps2 * (free_flow - float((star / gap_m) ** 2))


def stop_line_gap(
    distance_to_stop_line_m: float,
    vehicle_front_offset_m: float = 0.0,
) -> tuple[float, float]:
    """Return ``(gap, leader_speed)`` describing a red signal as a stopped leader.

    Modelling the stop line this way is what makes queues emerge from the same
    equation that governs car-following, rather than from a bespoke rule.
    """

    return max(0.0, distance_to_stop_line_m - vehicle_front_offset_m), 0.0


def integrate(
    speed_mps: float,
    acceleration: float,
    step_s: float,
) -> tuple[float, float]:
    """Advance one ballistic step, returning ``(new_speed, distance_travelled)``.

    Speed is floored at zero and the travelled distance is corrected for the
    partial step when a vehicle comes to rest, so a decelerating vehicle never
    moves backwards.
    """

    if step_s <= 0.0:
        raise ValueError("step_s must be positive")

    new_speed = speed_mps + acceleration * step_s
    if new_speed <= 0.0:
        # Vehicle stops partway through the step.
        stop_time = -speed_mps / acceleration if acceleration < 0.0 else 0.0
        stop_time = min(max(stop_time, 0.0), step_s)
        return 0.0, max(0.0, speed_mps * stop_time + 0.5 * acceleration * stop_time**2)

    distance = speed_mps * step_s + 0.5 * acceleration * step_s**2
    return new_speed, max(0.0, distance)


__all__ = [
    "IDM_DELTA",
    "IDMParameters",
    "acceleration_mps2",
    "desired_gap_m",
    "integrate",
    "stop_line_gap",
]


@dataclass(frozen=True, slots=True)
class MOBILParameters:
    """Lane-change decision parameters (Kesting, Treiber and Helbing).

    Lane changing is what gives V-VLC a blockage that is **not derivable from
    a map**.  Junction cross-traffic is predictable from own pose plus a road
    layout; a vehicle deciding to overtake is not, and work plan §4.9 records
    that distinction as the reason RQ3 and H7 survive.
    """

    #: Weight given to the vehicles a change inconveniences.  Zero is purely
    #: selfish, one weights others equally with oneself.
    politeness: float = 0.3
    #: Advantage a change must yield before it is worth making, in m/s^2.
    threshold_mps2: float = 0.1
    #: A change is refused outright if it would force the new follower to brake
    #: harder than this.  Safety is a veto, never traded against incentive.
    safe_deceleration_mps2: float = 4.0
    #: Keep-right bias: added when moving towards the kerb and subtracted when
    #: moving inward, so the inside lane is vacated when it is not needed.
    keep_right_bias_mps2: float = 0.3
    #: A manoeuvre occupies the vehicle for this long, and it cannot begin
    #: another until it completes.  Without it the criterion is re-evaluated
    #: every 50 ms and vehicles oscillate between lanes at 20 Hz, which
    #: produced 937 changes per vehicle-hour against a plausible few dozen.
    manoeuvre_duration_s: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.politeness <= 1.0:
            raise ValueError("politeness must lie in [0, 1]")
        for name in ("threshold_mps2", "safe_deceleration_mps2", "keep_right_bias_mps2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.manoeuvre_duration_s) or self.manoeuvre_duration_s <= 0.0:
            raise ValueError("manoeuvre_duration_s must be finite and positive")


def mobil_accepts(
    *,
    own_before: float,
    own_after: float,
    new_follower_before: float,
    new_follower_after: float,
    old_follower_before: float,
    old_follower_after: float,
    parameters: MOBILParameters,
    bias_mps2: float = 0.0,
) -> bool:
    """Whether MOBIL sanctions a lane change, given accelerations either side.

    The caller supplies the six accelerations because only it knows the lane
    neighbourhood; keeping the criterion free of that lookup is what makes it
    testable without a simulator.

    Two clauses, and the order matters.  **Safety is a veto**: if the vehicle
    that would end up behind must brake harder than
    ``safe_deceleration_mps2``, the change is refused whatever it gains.  Only
    then is the incentive weighed, and the politeness term is what stops a
    vehicle from buying its own small gain with a large loss imposed on
    someone else.
    """

    if new_follower_after < -parameters.safe_deceleration_mps2:
        return False

    gain = (own_after - own_before) + parameters.politeness * (
        (new_follower_after - new_follower_before) + (old_follower_after - old_follower_before)
    )
    return gain + bias_mps2 > parameters.threshold_mps2
