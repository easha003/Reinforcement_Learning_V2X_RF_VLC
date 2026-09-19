"""Small immutable value objects shared across Hybrid RF/VLC RL modules.

The types in this module deliberately contain no simulator, NumPy, Arrow, or
PyTorch objects.  They are safe to pass between the mobility, channel, PHY,
environment, and evaluation layers without giving any layer ownership of
mutable state.
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_v2x_rl.core.enums import Action, FailureCause, Link


@dataclass(frozen=True, slots=True)
class VehicleState:
    """One vehicle's ground-truth kinematic state at a simulation instant."""

    vehicle_id: str
    time_s: float
    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float
    acceleration_mps2: float
    length_m: float
    width_m: float
    height_m: float
    lane_id: str
    edge_id: str
    route_id: str


@dataclass(frozen=True, slots=True)
class PairState:
    """Ground-truth state of a tagged transmitter/receiver pair."""

    time_s: float
    tx: VehicleState
    rx: VehicleState
    distance_m: float
    same_route: bool


@dataclass(frozen=True, slots=True)
class ServiceProfile:
    """Resolved packet-generation and reliability contract."""

    name: str
    payload_bytes: int
    generation_period_s: float
    deadline_s: float
    miss_budget: float


@dataclass(frozen=True, slots=True)
class LinkAttemptResult:
    """Terminal result of one selected or unselected physical-link leg."""

    link: Link
    selected: bool
    success: bool
    arrival_time_s: float | None
    conditional_failure_probability: float
    failure_cause: FailureCause


@dataclass(frozen=True, slots=True)
class PacketOutcome:
    """Single source-of-truth outcome after both possible link legs resolve."""

    packet_id: str
    generation_time_s: float
    action: Action
    delivered: bool
    deadline_missed: bool
    delivery_time_s: float | None
    rf: LinkAttemptResult
    vlc: LinkAttemptResult
    activation_cost: float


__all__ = [
    "LinkAttemptResult",
    "PacketOutcome",
    "PairState",
    "ServiceProfile",
    "VehicleState",
]
