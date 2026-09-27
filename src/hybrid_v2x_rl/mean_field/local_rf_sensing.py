"""Simulator-only pair-local sensing visibility and attempt-weighted exposure."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.geometry import OrientedRectangle
from hybrid_v2x_rl.core.link_endpoints import rf_link_path
from hybrid_v2x_rl.core.policy_actions import MAX_RESERVED_RF_ATTEMPTS
from hybrid_v2x_rl.geometry.building_geometry import BuildingLayout
from hybrid_v2x_rl.geometry.rf_visibility import building_obstructs
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.mean_field.local_rf_domain import (
    LOCAL_RF_DOMAIN_CONTRACT_VERSION,
    FrameLocalRFLoads,
    FrameLocalRFTopology,
)

LOCAL_RF_SENSING_CONTRACT_VERSION: Final = "1.0.0"
_TIME_TOLERANCE_S: Final = 1e-9


class LocalRFSensingError(HybridV2XError):
    """Pair-local sensing geometry or attempt partition is inconsistent."""


def sensing_buildings_from_config(
    config: ProjectConfig,
) -> tuple[OrientedRectangle, ...]:
    """Build the deterministic Manhattan obstruction set declared by config."""

    if not isinstance(config, ProjectConfig):
        raise LocalRFSensingError("sensing building construction requires ProjectConfig")
    if not config.geometry.include_building_nlos:
        return ()
    grid = config.mobility.grid
    if grid is None:
        raise LocalRFSensingError(
            "building NLOS is enabled without a configured grid layout"
        )
    return BuildingLayout.from_grid(
        avenues=grid.avenues,
        cross_streets=grid.cross_streets,
        avenue_spacing_m=grid.avenue_spacing_m,
        cross_street_spacing_m=grid.cross_street_spacing_m,
        lanes_per_direction=grid.lanes_per_direction,
        lane_width_m=grid.lane_width_m,
    ).rectangles


def _building_geometry_sha256(buildings: tuple[OrientedRectangle, ...]) -> str:
    payload = [
        {
            "centre_x_m": building.centre.x_m,
            "centre_y_m": building.centre.y_m,
            "heading_rad": building.heading_rad,
            "length_m": building.length_m,
            "width_m": building.width_m,
        }
        for building in buildings
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LocalRFSensingMember:
    """Geometric reservation visibility for one member flow."""

    pair_id: str
    transmitter_id: str
    colocated_with_focal_transmitter: bool
    building_blocked: bool
    geometrically_decodable: bool

    def __post_init__(self) -> None:
        if not self.pair_id or not self.transmitter_id:
            raise LocalRFSensingError("sensing member identities must be nonempty")
        if any(
            type(value) is not bool
            for value in (
                self.colocated_with_focal_transmitter,
                self.building_blocked,
                self.geometrically_decodable,
            )
        ):
            raise LocalRFSensingError("sensing member flags must be booleans")
        if self.colocated_with_focal_transmitter and self.building_blocked:
            raise LocalRFSensingError("a colocated reservation cannot be building-blocked")
        expected_decodable = (
            self.colocated_with_focal_transmitter or not self.building_blocked
        )
        if self.geometrically_decodable is not expected_decodable:
            raise LocalRFSensingError(
                "geometric decodability must be the complement of building blockage"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "transmitter_id": self.transmitter_id,
            "colocated_with_focal_transmitter": (
                self.colocated_with_focal_transmitter
            ),
            "building_blocked": self.building_blocked,
            "geometrically_decodable": self.geometrically_decodable,
        }


@dataclass(frozen=True, slots=True)
class PairLocalRFSensing:
    """Action-independent sensing geometry for one focal service flow."""

    focal_pair_id: str
    focal_transmitter_id: str
    members: tuple[LocalRFSensingMember, ...]

    def __post_init__(self) -> None:
        if not self.focal_pair_id or not self.focal_transmitter_id:
            raise LocalRFSensingError("focal sensing identities must be nonempty")
        pair_ids = self.member_pair_ids
        if not pair_ids or pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(
            set(pair_ids)
        ):
            raise LocalRFSensingError(
                "sensing members must be nonempty, unique, and canonical"
            )
        try:
            focal_index = pair_ids.index(self.focal_pair_id)
        except ValueError as error:
            raise LocalRFSensingError(
                "pair-local sensing must include its focal flow"
            ) from error
        focal = self.members[focal_index]
        if (
            focal.transmitter_id != self.focal_transmitter_id
            or not focal.colocated_with_focal_transmitter
            or not focal.geometrically_decodable
        ):
            raise LocalRFSensingError("focal sensing membership is invalid")
        if any(
            member.colocated_with_focal_transmitter
            != (member.transmitter_id == self.focal_transmitter_id)
            for member in self.members
        ):
            raise LocalRFSensingError(
                "colocated sensing flags do not follow physical transmitter identity"
            )

    @property
    def member_pair_ids(self) -> tuple[str, ...]:
        return tuple(member.pair_id for member in self.members)

    def member_for(self, pair_id: str) -> LocalRFSensingMember:
        try:
            index = self.member_pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFSensingError(
                "flow is absent from pair-local sensing",
                context={"pair_id": pair_id},
            ) from error
        return self.members[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "focal_pair_id": self.focal_pair_id,
            "focal_transmitter_id": self.focal_transmitter_id,
            "members": [member.as_dict() for member in self.members],
        }


@dataclass(frozen=True, slots=True)
class FrameLocalRFSensing:
    """Pair-aligned simulator-truth sensing geometry for one frame."""

    contract_version: str
    topology_contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    pair_ids: tuple[str, ...]
    antenna_height_m: float
    building_count: int
    building_geometry_sha256: str
    sensing: tuple[PairLocalRFSensing, ...]

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_SENSING_CONTRACT_VERSION:
            raise LocalRFSensingError("local RF sensing contract version is invalid")
        if self.topology_contract_version != LOCAL_RF_DOMAIN_CONTRACT_VERSION:
            raise LocalRFSensingError("local RF topology contract version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise LocalRFSensingError("local RF sensing frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise LocalRFSensingError("local RF sensing time is invalid")
        if not math.isfinite(self.antenna_height_m) or self.antenna_height_m <= 0.0:
            raise LocalRFSensingError("local RF sensing antenna height is invalid")
        if self.building_count < 0:
            raise LocalRFSensingError("local RF sensing building count is invalid")
        if len(self.building_geometry_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.building_geometry_sha256
        ):
            raise LocalRFSensingError("building geometry digest must be SHA-256")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise LocalRFSensingError("local RF sensing pair IDs are not canonical")
        if len(self.sensing) != len(self.pair_ids) or tuple(
            row.focal_pair_id for row in self.sensing
        ) != self.pair_ids:
            raise LocalRFSensingError(
                "local RF sensing rows must align with frame pair IDs"
            )

    @classmethod
    def from_frame_and_topology(
        cls,
        frame: PopulationFrame,
        topology: FrameLocalRFTopology,
        *,
        buildings: Iterable[OrientedRectangle],
        antenna_height_m: float,
    ) -> FrameLocalRFSensing:
        """Classify building visibility once per physical transmitter edge."""

        if not isinstance(frame, PopulationFrame):
            raise LocalRFSensingError("local RF sensing requires PopulationFrame")
        if not isinstance(topology, FrameLocalRFTopology):
            raise LocalRFSensingError("local RF sensing requires FrameLocalRFTopology")
        if (
            frame.trace_id != topology.trace_id
            or frame.index != topology.frame_index
            or not math.isclose(
                frame.time_s,
                topology.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or frame.active_pair_ids != topology.pair_ids
        ):
            raise LocalRFSensingError(
                "population frame and local RF topology do not identify the same frame"
            )
        if not math.isfinite(antenna_height_m) or antenna_height_m <= 0.0:
            raise LocalRFSensingError("antenna height must be finite and positive")
        building_tuple = tuple(buildings)
        if any(
            not isinstance(building, OrientedRectangle) for building in building_tuple
        ):
            raise LocalRFSensingError(
                "sensing buildings must be OrientedRectangle values"
            )
        transmitter_by_pair = {
            pair.pair_id: pair.transmitter for pair in frame.pairs
        }
        transmitter_by_id = {
            pair.transmitter.vehicle_id: pair.transmitter for pair in frame.pairs
        }
        blocked_by_edge: dict[tuple[str, str], bool] = {}
        sensing_rows: list[PairLocalRFSensing] = []
        for domain in topology.domains:
            focal_transmitter = transmitter_by_pair[domain.focal_pair_id]
            if focal_transmitter.vehicle_id != domain.focal_transmitter_id:
                raise LocalRFSensingError(
                    "topology focal transmitter differs from the population frame"
                )
            members: list[LocalRFSensingMember] = []
            for pair_id, transmitter_id in zip(
                domain.member_pair_ids,
                domain.member_transmitter_ids,
                strict=True,
            ):
                member_transmitter = transmitter_by_pair[pair_id]
                if member_transmitter.vehicle_id != transmitter_id:
                    raise LocalRFSensingError(
                        "topology member transmitter differs from the population frame"
                    )
                colocated = transmitter_id == focal_transmitter.vehicle_id
                if colocated:
                    building_blocked = False
                else:
                    lower_id = min(focal_transmitter.vehicle_id, transmitter_id)
                    upper_id = max(focal_transmitter.vehicle_id, transmitter_id)
                    edge = (lower_id, upper_id)
                    cached = blocked_by_edge.get(edge)
                    if cached is None:
                        cached = building_obstructs(
                            rf_link_path(
                                focal_transmitter,
                                transmitter_by_id[transmitter_id],
                                height_m=antenna_height_m,
                            ),
                            building_tuple,
                        )
                        blocked_by_edge[edge] = cached
                    building_blocked = cached
                members.append(
                    LocalRFSensingMember(
                        pair_id=pair_id,
                        transmitter_id=transmitter_id,
                        colocated_with_focal_transmitter=colocated,
                        building_blocked=building_blocked,
                        geometrically_decodable=not building_blocked,
                    )
                )
            sensing_rows.append(
                PairLocalRFSensing(
                    focal_pair_id=domain.focal_pair_id,
                    focal_transmitter_id=domain.focal_transmitter_id,
                    members=tuple(members),
                )
            )
        return cls(
            contract_version=LOCAL_RF_SENSING_CONTRACT_VERSION,
            topology_contract_version=topology.contract_version,
            trace_id=topology.trace_id,
            frame_index=topology.frame_index,
            time_s=topology.time_s,
            pair_ids=topology.pair_ids,
            antenna_height_m=antenna_height_m,
            building_count=len(building_tuple),
            building_geometry_sha256=_building_geometry_sha256(building_tuple),
            sensing=tuple(sensing_rows),
        )

    def sensing_for(self, pair_id: str) -> PairLocalRFSensing:
        try:
            index = self.pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFSensingError(
                "pair is absent from frame-local sensing",
                context={"pair_id": pair_id},
            ) from error
        return self.sensing[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "topology_contract_version": self.topology_contract_version,
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "active_pairs": len(self.pair_ids),
            "antenna_height_m": self.antenna_height_m,
            "building_count": self.building_count,
            "building_geometry_sha256": self.building_geometry_sha256,
            "sensing": [row.as_dict() for row in self.sensing],
        }


@dataclass(frozen=True, slots=True)
class LocalRFSensingReservation:
    """One selected reservation classified relative to a focal transmitter."""

    pair_id: str
    transmitter_id: str
    reserved_rf_attempts: int
    focal_flow: bool
    colocated_with_focal_transmitter: bool
    geometrically_decodable: bool

    def __post_init__(self) -> None:
        if not self.pair_id or not self.transmitter_id:
            raise LocalRFSensingError("sensing reservation identities must be nonempty")
        if (
            not isinstance(self.reserved_rf_attempts, int)
            or isinstance(self.reserved_rf_attempts, bool)
            or not 0 <= self.reserved_rf_attempts <= MAX_RESERVED_RF_ATTEMPTS
        ):
            raise LocalRFSensingError("sensing reservation attempt count is invalid")
        if any(
            type(value) is not bool
            for value in (
                self.focal_flow,
                self.colocated_with_focal_transmitter,
                self.geometrically_decodable,
            )
        ):
            raise LocalRFSensingError("sensing reservation flags must be booleans")
        if self.focal_flow and not self.colocated_with_focal_transmitter:
            raise LocalRFSensingError("focal flow must be transmitter-colocated")
        if self.colocated_with_focal_transmitter and not self.geometrically_decodable:
            raise LocalRFSensingError("colocated reservations must be locally known")

    @property
    def role(self) -> str:
        if self.focal_flow:
            return "focal"
        if self.colocated_with_focal_transmitter:
            return "colocated_other_flow"
        if self.geometrically_decodable:
            return "external_geometrically_sensed"
        return "external_geometrically_hidden"

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "transmitter_id": self.transmitter_id,
            "reserved_rf_attempts": self.reserved_rf_attempts,
            "role": self.role,
            "geometrically_decodable": self.geometrically_decodable,
        }


@dataclass(frozen=True, slots=True)
class PairLocalRFSensedLoad:
    """Attempt-weighted geometric sensing partition for one focal flow."""

    focal_pair_id: str
    focal_transmitter_id: str
    reservations: tuple[LocalRFSensingReservation, ...]
    local_offered_rf_attempts: int
    focal_rf_attempts: int
    colocated_other_rf_attempts: int
    external_contending_rf_attempts: int
    geometrically_sensed_external_rf_attempts: int
    geometrically_hidden_external_rf_attempts: int
    sensed_fraction: float

    def __post_init__(self) -> None:
        if not self.focal_pair_id or not self.focal_transmitter_id:
            raise LocalRFSensingError("sensed-load focal identities must be nonempty")
        pair_ids = tuple(reservation.pair_id for reservation in self.reservations)
        if not pair_ids or pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(
            set(pair_ids)
        ):
            raise LocalRFSensingError(
                "sensed-load reservations must be nonempty, unique, and canonical"
            )
        focal_rows = tuple(
            reservation for reservation in self.reservations if reservation.focal_flow
        )
        if len(focal_rows) != 1 or focal_rows[0].pair_id != self.focal_pair_id:
            raise LocalRFSensingError("sensed load must identify exactly one focal flow")
        if focal_rows[0].transmitter_id != self.focal_transmitter_id:
            raise LocalRFSensingError("sensed-load focal transmitter is misaligned")
        expected_local = sum(
            reservation.reserved_rf_attempts for reservation in self.reservations
        )
        expected_focal = focal_rows[0].reserved_rf_attempts
        expected_colocated = sum(
            reservation.reserved_rf_attempts
            for reservation in self.reservations
            if not reservation.focal_flow
            and reservation.colocated_with_focal_transmitter
        )
        external = tuple(
            reservation
            for reservation in self.reservations
            if not reservation.colocated_with_focal_transmitter
        )
        expected_external = sum(
            reservation.reserved_rf_attempts for reservation in external
        )
        expected_sensed = sum(
            reservation.reserved_rf_attempts
            for reservation in external
            if reservation.geometrically_decodable
        )
        expected_hidden = expected_external - expected_sensed
        expected_fraction = (
            expected_sensed / expected_external if expected_external else 1.0
        )
        actual = (
            self.local_offered_rf_attempts,
            self.focal_rf_attempts,
            self.colocated_other_rf_attempts,
            self.external_contending_rf_attempts,
            self.geometrically_sensed_external_rf_attempts,
            self.geometrically_hidden_external_rf_attempts,
        )
        expected = (
            expected_local,
            expected_focal,
            expected_colocated,
            expected_external,
            expected_sensed,
            expected_hidden,
        )
        if actual != expected:
            raise LocalRFSensingError("pair-local sensed-load totals do not conserve")
        if expected_local != expected_focal + expected_colocated + expected_external:
            raise LocalRFSensingError("pair-local attempt roles do not partition load")
        if (
            not math.isfinite(self.sensed_fraction)
            or not 0.0 <= self.sensed_fraction <= 1.0
            or not math.isclose(
                self.sensed_fraction,
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise LocalRFSensingError("pair-local sensed fraction does not reconcile")

    def as_dict(self) -> dict[str, object]:
        return {
            "focal_pair_id": self.focal_pair_id,
            "focal_transmitter_id": self.focal_transmitter_id,
            "local_offered_rf_attempts": self.local_offered_rf_attempts,
            "focal_rf_attempts": self.focal_rf_attempts,
            "colocated_other_rf_attempts": self.colocated_other_rf_attempts,
            "external_contending_rf_attempts": self.external_contending_rf_attempts,
            "geometrically_sensed_external_rf_attempts": (
                self.geometrically_sensed_external_rf_attempts
            ),
            "geometrically_hidden_external_rf_attempts": (
                self.geometrically_hidden_external_rf_attempts
            ),
            "sensed_fraction": self.sensed_fraction,
            "reservations": [
                reservation.as_dict() for reservation in self.reservations
            ],
        }


@dataclass(frozen=True, slots=True)
class FrameLocalRFSensedLoads:
    """Action-coupled sensing partitions aligned to every active pair."""

    contract_version: str
    trace_id: str
    frame_index: int
    time_s: float
    pair_ids: tuple[str, ...]
    rows: tuple[PairLocalRFSensedLoad, ...]

    def __post_init__(self) -> None:
        if self.contract_version != LOCAL_RF_SENSING_CONTRACT_VERSION:
            raise LocalRFSensingError("sensed-load contract version is invalid")
        if not self.trace_id or self.frame_index < 0:
            raise LocalRFSensingError("sensed-load frame identity is invalid")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise LocalRFSensingError("sensed-load frame time is invalid")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(
            self.pair_ids
        ) != len(set(self.pair_ids)):
            raise LocalRFSensingError("sensed-load pair IDs are not canonical")
        if len(self.rows) != len(self.pair_ids) or tuple(
            row.focal_pair_id for row in self.rows
        ) != self.pair_ids:
            raise LocalRFSensingError("sensed-load rows must align with frame pairs")

    @classmethod
    def from_sensing_and_loads(
        cls,
        sensing: FrameLocalRFSensing,
        loads: FrameLocalRFLoads,
    ) -> FrameLocalRFSensedLoads:
        """Weight action-independent visibility by selected RF reservations."""

        if not isinstance(sensing, FrameLocalRFSensing):
            raise LocalRFSensingError("sensed loads require FrameLocalRFSensing")
        if not isinstance(loads, FrameLocalRFLoads):
            raise LocalRFSensingError("sensed loads require FrameLocalRFLoads")
        if (
            sensing.trace_id != loads.trace_id
            or sensing.frame_index != loads.frame_index
            or not math.isclose(
                sensing.time_s,
                loads.time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
            or sensing.pair_ids != loads.pair_ids
        ):
            raise LocalRFSensingError(
                "pair-local sensing and loads do not identify the same frame"
            )
        rows: list[PairLocalRFSensedLoad] = []
        for sensing_row, load in zip(sensing.sensing, loads.loads, strict=True):
            if sensing_row.member_pair_ids != load.member_pair_ids:
                raise LocalRFSensingError(
                    "pair-local sensing and load membership differ"
                )
            attempts_by_pair = dict(load.reserved_rf_attempts_by_pair)
            reservations = tuple(
                LocalRFSensingReservation(
                    pair_id=member.pair_id,
                    transmitter_id=member.transmitter_id,
                    reserved_rf_attempts=attempts_by_pair[member.pair_id],
                    focal_flow=member.pair_id == sensing_row.focal_pair_id,
                    colocated_with_focal_transmitter=(
                        member.colocated_with_focal_transmitter
                    ),
                    geometrically_decodable=member.geometrically_decodable,
                )
                for member in sensing_row.members
            )
            focal_attempts = attempts_by_pair[sensing_row.focal_pair_id]
            colocated_attempts = sum(
                reservation.reserved_rf_attempts
                for reservation in reservations
                if not reservation.focal_flow
                and reservation.colocated_with_focal_transmitter
            )
            external = tuple(
                reservation
                for reservation in reservations
                if not reservation.colocated_with_focal_transmitter
            )
            external_attempts = sum(
                reservation.reserved_rf_attempts for reservation in external
            )
            sensed_attempts = sum(
                reservation.reserved_rf_attempts
                for reservation in external
                if reservation.geometrically_decodable
            )
            rows.append(
                PairLocalRFSensedLoad(
                    focal_pair_id=sensing_row.focal_pair_id,
                    focal_transmitter_id=sensing_row.focal_transmitter_id,
                    reservations=reservations,
                    local_offered_rf_attempts=load.offered_rf_attempts,
                    focal_rf_attempts=focal_attempts,
                    colocated_other_rf_attempts=colocated_attempts,
                    external_contending_rf_attempts=external_attempts,
                    geometrically_sensed_external_rf_attempts=sensed_attempts,
                    geometrically_hidden_external_rf_attempts=(
                        external_attempts - sensed_attempts
                    ),
                    sensed_fraction=(
                        sensed_attempts / external_attempts
                        if external_attempts
                        else 1.0
                    ),
                )
            )
        return cls(
            contract_version=sensing.contract_version,
            trace_id=sensing.trace_id,
            frame_index=sensing.frame_index,
            time_s=sensing.time_s,
            pair_ids=sensing.pair_ids,
            rows=tuple(rows),
        )

    def row_for(self, pair_id: str) -> PairLocalRFSensedLoad:
        try:
            index = self.pair_ids.index(pair_id)
        except ValueError as error:
            raise LocalRFSensingError(
                "pair is absent from frame sensed loads",
                context={"pair_id": pair_id},
            ) from error
        return self.rows[index]

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "trace_id": self.trace_id,
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "active_pairs": len(self.pair_ids),
            "rows": [row.as_dict() for row in self.rows],
        }


__all__ = [
    "LOCAL_RF_SENSING_CONTRACT_VERSION",
    "FrameLocalRFSensedLoads",
    "FrameLocalRFSensing",
    "LocalRFSensingError",
    "LocalRFSensingMember",
    "LocalRFSensingReservation",
    "PairLocalRFSensedLoad",
    "PairLocalRFSensing",
    "sensing_buildings_from_config",
]
