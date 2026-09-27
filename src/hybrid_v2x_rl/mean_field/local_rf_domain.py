"""Pair-local RF contention topology and action-coupled demand contract.

This module deliberately stops before collision, sensing, and half-duplex
evaluation.  It establishes the identity-preserving spatial boundary that
those mechanisms must consume, without partially changing rollout behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import MAX_RESERVED_RF_ATTEMPTS
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.frames import PopulationFrame

LOCAL_RF_DOMAIN_CONTRACT_VERSION: Final = "1.0.0"
LOCAL_RF_CONTENTION_RADIUS_M: Final = 200.0
_TIME_TOLERANCE_S: Final = 1e-9


class LocalRFDomainError(HybridV2XError):
    """Pair-local RF topology or demand violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class PairLocalRFDomain:
    """Active service flows inside one transmitter-centred RF domain."""

    focal_pair_id: str
    focal_transmitter_id: str
    member_pair_ids: tuple[str, ...]
    member_transmitter_ids: tuple[str, ...]
    radius_m: float

    def __post_init__(self) -> None:
        if not self.focal_pair_id or not self.focal_transmitter_id:
            raise LocalRFDomainError("local RF focal identities must be nonempty")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise LocalRFDomainError("local RF radius must be finite and positive")
        if not self.member_pair_ids or self.member_pair_ids != tuple(
            sorted(self.member_pair_ids)
        ):
            raise LocalRFDomainError(
                "local RF member pairs must be nonempty and canonically ordered"
            )
        if len(self.member_pair_ids) != len(set(self.member_pair_ids)):
            raise LocalRFDomainError("local RF member pairs cannot repeat")
        if len(self.member_transmitter_ids) != len(self.member_pair_ids) or any(
            not transmitter_id for transmitter_id in self.member_transmitter_ids
        ):
            raise LocalRFDomainError(
                "local RF member transmitters must align with member pairs"
            )
        try:
            focal_index = self.member_pair_ids.index(self.focal_pair_id)
        except ValueError as error:
            raise LocalRFDomainError(
                "a local RF domain must include its focal flow"
            ) from error
        if self.member_transmitter_ids[focal_index] != self.focal_transmitter_id:
            raise LocalRFDomainError(
                "focal transmitter must align with the focal member flow"
            )

    @property
    def member_flows(self) -> int:
        return len(self.member_pair_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "focal_pair_id": self.focal_pair_id,
            "focal_transmitter_id": self.focal_transmitter_id,
            "radius_m": self.radius_m,
            "member_flows": self.member_flows,
            "members": [
                {"pair_id": pair_id, "transmitter_id": transmitter_id}
                for pair_id, transmitter_id in zip(
                    self.member_pair_ids,
                    self.member_transmitter_ids,
                    strict=True,
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class FrameLocalRFTopology:
    """Stable pair-local interference graph for one population frame."""

    contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    radius_m: float
    pair_ids: tuple[str, ...]
    domains: tuple[PairLocalRFDomain, ...]

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_DOMAIN_CONTRACT_VERSION:
            raise LocalRFDomainError("local RF topology contract version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise LocalRFDomainError("local RF topology frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise LocalRFDomainError("local RF topology time is invalid")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise LocalRFDomainError("local RF topology radius is invalid")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise LocalRFDomainError(
                "local RF topology pair IDs must be unique and canonical"
            )
        if len(self.domains) != len(self.pair_ids) or tuple(
            domain.focal_pair_id for domain in self.domains
        ) != self.pair_ids:
            raise LocalRFDomainError(
                "local RF domains must align exactly with frame pair IDs"
            )
        pair_set = set(self.pair_ids)
        members_by_pair = {
            domain.focal_pair_id: set(domain.member_pair_ids)
            for domain in self.domains
        }
        for domain in self.domains:
            if domain.radius_m != self.radius_m:
                raise LocalRFDomainError("local RF domain radius drifted within a frame")
            if not set(domain.member_pair_ids).issubset(pair_set):
                raise LocalRFDomainError(
                    "local RF domain references a flow outside the frame"
                )
            for member_pair_id in domain.member_pair_ids:
                if domain.focal_pair_id not in members_by_pair[member_pair_id]:
                    raise LocalRFDomainError(
                        "transmitter-distance membership must be reciprocal"
                    )

    @classmethod
    def from_frame(
        cls,
        frame: PopulationFrame,
        *,
        radius_m: float = LOCAL_RF_CONTENTION_RADIUS_M,
    ) -> FrameLocalRFTopology:
        """Build exact transmitter-distance membership from simulator truth."""

        if not isinstance(frame, PopulationFrame):
            raise LocalRFDomainError("local RF topology requires a PopulationFrame")
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise LocalRFDomainError("local RF radius must be finite and positive")
        if not frame.pairs:
            return cls(
                contract_version=LOCAL_RF_DOMAIN_CONTRACT_VERSION,
                trace_id=frame.trace_id,
                frame_index=frame.index,
                time_s=frame.time_s,
                radius_m=radius_m,
                pair_ids=(),
                domains=(),
            )

        pair_ids = frame.active_pair_ids
        transmitter_ids = tuple(
            pair.transmitter.vehicle_id for pair in frame.pairs
        )
        positions = np.asarray(
            [(pair.transmitter.x_m, pair.transmitter.y_m) for pair in frame.pairs],
            dtype=np.float64,
        )
        squared_distances = np.sum(
            (positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        membership = squared_distances <= radius_m * radius_m
        domains = tuple(
            PairLocalRFDomain(
                focal_pair_id=pair_id,
                focal_transmitter_id=transmitter_ids[row],
                member_pair_ids=tuple(
                    candidate_id
                    for candidate_id, included in zip(
                        pair_ids, membership[row].tolist(), strict=True
                    )
                    if included
                ),
                member_transmitter_ids=tuple(
                    transmitter_id
                    for transmitter_id, included in zip(
                        transmitter_ids, membership[row].tolist(), strict=True
                    )
                    if included
                ),
                radius_m=radius_m,
            )
            for row, pair_id in enumerate(pair_ids)
        )
        return cls(
            contract_version=LOCAL_RF_DOMAIN_CONTRACT_VERSION,
            trace_id=frame.trace_id,
            frame_index=frame.index,
            time_s=frame.time_s,
            radius_m=radius_m,
            pair_ids=pair_ids,
            domains=domains,
        )

    def domain_for(self, pair_id: str) -> PairLocalRFDomain:
        try:
            index = self.pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFDomainError(
                "pair is absent from the local RF topology",
                context={"pair_id": pair_id},
            ) from error
        return self.domains[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "radius_m": self.radius_m,
            "active_pairs": len(self.pair_ids),
            "domains": [domain.as_dict() for domain in self.domains],
        }


@dataclass(frozen=True, slots=True)
class PairLocalRFLoad:
    """Selected RF reservations visible inside one focal pair's domain."""

    focal_pair_id: str
    reserved_rf_attempts_by_pair: tuple[tuple[str, int], ...]
    offered_rf_attempts: int
    rf_using_pairs: int

    def __post_init__(self) -> None:
        if not self.focal_pair_id:
            raise LocalRFDomainError("local RF load focal pair must be nonempty")
        pair_ids = tuple(pair_id for pair_id, _ in self.reserved_rf_attempts_by_pair)
        if not pair_ids or pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(
            set(pair_ids)
        ):
            raise LocalRFDomainError(
                "local RF reservations must be nonempty, unique, and canonical"
            )
        if self.focal_pair_id not in pair_ids:
            raise LocalRFDomainError("local RF reservations must include the focal flow")
        if any(
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or not 0 <= attempts <= MAX_RESERVED_RF_ATTEMPTS
            for _, attempts in self.reserved_rf_attempts_by_pair
        ):
            raise LocalRFDomainError("local RF reservation attempts are invalid")
        expected_attempts = sum(
            attempts for _, attempts in self.reserved_rf_attempts_by_pair
        )
        expected_users = sum(
            attempts > 0 for _, attempts in self.reserved_rf_attempts_by_pair
        )
        if self.offered_rf_attempts != expected_attempts:
            raise LocalRFDomainError("local RF offered attempts do not conserve")
        if self.rf_using_pairs != expected_users:
            raise LocalRFDomainError("local RF-using pair count does not conserve")

    @property
    def member_pair_ids(self) -> tuple[str, ...]:
        return tuple(pair_id for pair_id, _ in self.reserved_rf_attempts_by_pair)

    def as_dict(self) -> dict[str, object]:
        return {
            "focal_pair_id": self.focal_pair_id,
            "offered_rf_attempts": self.offered_rf_attempts,
            "rf_using_pairs": self.rf_using_pairs,
            "reservations": [
                {"pair_id": pair_id, "reserved_rf_attempts": attempts}
                for pair_id, attempts in self.reserved_rf_attempts_by_pair
            ],
        }


@dataclass(frozen=True, slots=True)
class FrameLocalRFLoads:
    """One action-coupled local demand row for every active focal pair."""

    contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    radius_m: float
    pair_ids: tuple[str, ...]
    loads: tuple[PairLocalRFLoad, ...]

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_DOMAIN_CONTRACT_VERSION:
            raise LocalRFDomainError("local RF load contract version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise LocalRFDomainError("local RF load frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise LocalRFDomainError("local RF load time is invalid")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise LocalRFDomainError("local RF load radius is invalid")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise LocalRFDomainError("local RF load pair IDs are not canonical")
        if len(self.loads) != len(self.pair_ids) or tuple(
            load.focal_pair_id for load in self.loads
        ) != self.pair_ids:
            raise LocalRFDomainError("local RF loads must align with frame pair IDs")

    @classmethod
    def from_topology_and_ledger(
        cls,
        topology: FrameLocalRFTopology,
        ledger: FrameActionLedger,
    ) -> FrameLocalRFLoads:
        """Project one complete action ledger into every overlapping domain."""

        if not isinstance(topology, FrameLocalRFTopology):
            raise LocalRFDomainError("local RF loads require FrameLocalRFTopology")
        if not isinstance(ledger, FrameActionLedger):
            raise LocalRFDomainError("local RF loads require FrameActionLedger")
        if (
            topology.trace_id != ledger.trace_id
            or topology.frame_index != ledger.frame_index
            or not math.isclose(
                topology.time_s,
                ledger.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or topology.pair_ids != ledger.pair_ids
        ):
            raise LocalRFDomainError(
                "local RF topology and action ledger do not identify the same frame"
            )
        ledger.audit_accounting_conservation()
        attempts_by_pair = dict(ledger.reserved_rf_attempts_by_pair)
        loads = tuple(
            PairLocalRFLoad(
                focal_pair_id=domain.focal_pair_id,
                reserved_rf_attempts_by_pair=tuple(
                    (pair_id, attempts_by_pair[pair_id])
                    for pair_id in domain.member_pair_ids
                ),
                offered_rf_attempts=sum(
                    attempts_by_pair[pair_id] for pair_id in domain.member_pair_ids
                ),
                rf_using_pairs=sum(
                    attempts_by_pair[pair_id] > 0
                    for pair_id in domain.member_pair_ids
                ),
            )
            for domain in topology.domains
        )
        return cls(
            contract_version=topology.contract_version,
            trace_id=topology.trace_id,
            frame_index=topology.frame_index,
            time_s=topology.time_s,
            radius_m=topology.radius_m,
            pair_ids=topology.pair_ids,
            loads=loads,
        )

    def load_for(self, pair_id: str) -> PairLocalRFLoad:
        try:
            index = self.pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFDomainError(
                "pair is absent from the local RF loads",
                context={"pair_id": pair_id},
            ) from error
        return self.loads[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "radius_m": self.radius_m,
            "active_pairs": len(self.pair_ids),
            "loads": [load.as_dict() for load in self.loads],
        }


__all__ = [
    "LOCAL_RF_CONTENTION_RADIUS_M",
    "LOCAL_RF_DOMAIN_CONTRACT_VERSION",
    "FrameLocalRFLoads",
    "FrameLocalRFTopology",
    "LocalRFDomainError",
    "PairLocalRFDomain",
    "PairLocalRFLoad",
]
