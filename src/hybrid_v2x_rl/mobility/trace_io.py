"""Manifest-backed Parquet I/O for simulator-independent mobility traces."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from hybrid_v2x_rl.artifacts.manifest import ArtifactReference
from hybrid_v2x_rl.artifacts.store import ArtifactStore, StoredArtifact, verify_artifact
from hybrid_v2x_rl.core.errors import TraceIntegrityError
from hybrid_v2x_rl.core.types import VehicleState

TRACE_ARTIFACT_FAMILY = "traces"
TRACE_ARTIFACT_TYPE = "mobility_trace"
TRACE_SCHEMA_VERSION = "1.0.0"
_SCHEMA_METADATA = {b"hybrid_v2x_rl.trace_schema_version": TRACE_SCHEMA_VERSION.encode()}

VEHICLE_TRACE_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("vehicle_id", pa.string(), nullable=False),
        pa.field("x_m", pa.float64(), nullable=False),
        pa.field("y_m", pa.float64(), nullable=False),
        pa.field("heading_rad", pa.float64(), nullable=False),
        pa.field("speed_mps", pa.float64(), nullable=False),
        pa.field("acceleration_mps2", pa.float64(), nullable=False),
        pa.field("length_m", pa.float64(), nullable=False),
        pa.field("width_m", pa.float64(), nullable=False),
        pa.field("height_m", pa.float64(), nullable=False),
        pa.field("lane_id", pa.string(), nullable=False),
        pa.field("edge_id", pa.string(), nullable=False),
        pa.field("route_id", pa.string(), nullable=False),
        pa.field("vehicle_type", pa.string(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)

#: Tagged-pair episode schema, per implementation spec §8.7.
PAIR_SEGMENT_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("pair_id", pa.string(), nullable=False),
        pa.field("tx_id", pa.string(), nullable=False),
        pa.field("rx_id", pa.string(), nullable=False),
        pa.field("start_s", pa.float64(), nullable=False),
        pa.field("end_s", pa.float64(), nullable=False),
        pa.field("duration_s", pa.float64(), nullable=False),
        pa.field("initial_distance_m", pa.float64(), nullable=False),
        pa.field("route_id", pa.string(), nullable=False),
        pa.field("eligibility_reason", pa.string(), nullable=False),
        pa.field("has_intervening_vehicle", pa.bool_(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)

SIGNAL_STATE_SCHEMA = pa.schema(
    [
        pa.field("trace_id", pa.string(), nullable=False),
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("signal_id", pa.string(), nullable=False),
        pa.field("program_id", pa.string(), nullable=False),
        pa.field("phase_index", pa.int32(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
    ],
    metadata=_SCHEMA_METADATA,
)

_REQUIRED_TRACE_FILES = (
    "network.json",
    "pairs.parquet",
    "resolved_config.yaml",
    "routes.json",
    "signals.parquet",
)
_FLOAT_VEHICLE_FIELDS = (
    "time_s",
    "x_m",
    "y_m",
    "heading_rad",
    "speed_mps",
    "acceleration_mps2",
    "length_m",
    "width_m",
    "height_m",
)


@dataclass(frozen=True, slots=True)
class VehicleTraceRecord:
    """Flat, immutable row in a vehicle-state trace partition."""

    trace_id: str
    time_s: float
    vehicle_id: str
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
    vehicle_type: str

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            trace_id=self.trace_id,
            vehicle_id=self.vehicle_id,
            lane_id=self.lane_id,
            edge_id=self.edge_id,
            route_id=self.route_id,
            vehicle_type=self.vehicle_type,
        )
        _require_finite_values(
            time_s=self.time_s,
            x_m=self.x_m,
            y_m=self.y_m,
            heading_rad=self.heading_rad,
            speed_mps=self.speed_mps,
            acceleration_mps2=self.acceleration_mps2,
            length_m=self.length_m,
            width_m=self.width_m,
            height_m=self.height_m,
        )
        if self.time_s < 0.0:
            raise ValueError("time_s must be non-negative")
        if self.speed_mps < 0.0:
            raise ValueError("speed_mps must be non-negative")
        if min(self.length_m, self.width_m, self.height_m) <= 0.0:
            raise ValueError("vehicle dimensions must be positive")

    @classmethod
    def from_vehicle_state(
        cls,
        trace_id: str,
        vehicle_type: str,
        state: VehicleState,
    ) -> VehicleTraceRecord:
        """Add trace/type identity to the universal vehicle-state object."""

        return cls(
            trace_id=trace_id,
            time_s=state.time_s,
            vehicle_id=state.vehicle_id,
            x_m=state.x_m,
            y_m=state.y_m,
            heading_rad=state.heading_rad,
            speed_mps=state.speed_mps,
            acceleration_mps2=state.acceleration_mps2,
            length_m=state.length_m,
            width_m=state.width_m,
            height_m=state.height_m,
            lane_id=state.lane_id,
            edge_id=state.edge_id,
            route_id=state.route_id,
            vehicle_type=vehicle_type,
        )

    def to_vehicle_state(self) -> VehicleState:
        """Drop trace-only metadata and return the shared state object."""

        return VehicleState(
            vehicle_id=self.vehicle_id,
            time_s=self.time_s,
            x_m=self.x_m,
            y_m=self.y_m,
            heading_rad=self.heading_rad,
            speed_mps=self.speed_mps,
            acceleration_mps2=self.acceleration_mps2,
            length_m=self.length_m,
            width_m=self.width_m,
            height_m=self.height_m,
            lane_id=self.lane_id,
            edge_id=self.edge_id,
            route_id=self.route_id,
        )

    def as_arrow_row(self) -> dict[str, str | float]:
        """Return a mapping ordered like :data:`VEHICLE_TRACE_SCHEMA`."""

        return {
            "trace_id": self.trace_id,
            "time_s": self.time_s,
            "vehicle_id": self.vehicle_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "heading_rad": self.heading_rad,
            "speed_mps": self.speed_mps,
            "acceleration_mps2": self.acceleration_mps2,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "lane_id": self.lane_id,
            "edge_id": self.edge_id,
            "route_id": self.route_id,
            "vehicle_type": self.vehicle_type,
        }


@dataclass(frozen=True, slots=True)
class SignalStateRecord:
    """One traffic-signal program state at a simulation instant."""

    trace_id: str
    time_s: float
    signal_id: str
    program_id: str
    phase_index: int
    state: str

    def __post_init__(self) -> None:
        _require_nonempty_strings(
            trace_id=self.trace_id,
            signal_id=self.signal_id,
            program_id=self.program_id,
            state=self.state,
        )
        _require_finite_values(time_s=self.time_s)
        if self.time_s < 0.0:
            raise ValueError("time_s must be non-negative")
        if (
            not isinstance(self.phase_index, int)
            or isinstance(self.phase_index, bool)
            or self.phase_index < 0
        ):
            raise ValueError("phase_index must be a non-negative integer")

    def as_arrow_row(self) -> dict[str, str | float | int]:
        """Return a mapping ordered like :data:`SIGNAL_STATE_SCHEMA`."""

        return {
            "trace_id": self.trace_id,
            "time_s": self.time_s,
            "signal_id": self.signal_id,
            "program_id": self.program_id,
            "phase_index": self.phase_index,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class TraceIntegrityReport:
    """Summary produced while validating a complete mobility trace."""

    trace_id: str
    vehicle_rows: int
    signal_rows: int
    vehicle_parts: int
    first_time_s: float
    last_time_s: float


TracePayload = str | bytes | Path


class MobilityTraceWriter:
    """Stream iterable records into one immutable trace artifact."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        rows_per_part: int = 250_000,
        compression: str = "zstd",
        family: str = TRACE_ARTIFACT_FAMILY,
    ) -> None:
        if (
            not isinstance(rows_per_part, int)
            or isinstance(rows_per_part, bool)
            or rows_per_part <= 0
        ):
            raise ValueError("rows_per_part must be a positive integer")
        if not pa.Codec.is_available(compression):
            raise ValueError(f"PyArrow codec is unavailable: {compression}")
        self.store = store
        self.rows_per_part = rows_per_part
        self.compression = compression
        self.family = family

    def write(
        self,
        *,
        trace_id: str,
        vehicles: Iterable[VehicleTraceRecord],
        signals: Iterable[SignalStateRecord],
        pairs: Iterable[Mapping[str, object]],
        network_definition: TracePayload,
        route_definition: TracePayload,
        resolved_config_yaml: TracePayload,
        config_hash: str,
        config_scope_hashes: dict[str, str] | None = None,
        code_version: str,
        input_artifacts: Iterable[ArtifactReference] = (),
        random_seeds: dict[str, int] | None = None,
        software_versions: Mapping[str, str] | None = None,
        notes: Iterable[str] = (),
    ) -> StoredArtifact:
        """Write source files and record streams, then publish after validation."""

        vehicle_count = 0
        signal_count = 0
        vehicle_part_count = 0

        def produce(root: Path) -> None:
            nonlocal vehicle_count, signal_count, vehicle_part_count
            _write_payload(root / "network.json", network_definition)
            _write_payload(root / "routes.json", route_definition)
            _write_payload(root / "resolved_config.yaml", resolved_config_yaml)

            vehicle_dir = root / "vehicles"
            vehicle_dir.mkdir()
            previous_time: float | None = None
            vehicle_iterator = iter(vehicles)
            while chunk := tuple(islice(vehicle_iterator, self.rows_per_part)):
                for record in chunk:
                    _validate_record_trace_id(record.trace_id, trace_id)
                    if previous_time is not None and record.time_s < previous_time:
                        raise ValueError("vehicle records must be ordered by non-decreasing time")
                    previous_time = record.time_s
                table = pa.Table.from_pylist(
                    [record.as_arrow_row() for record in chunk],
                    schema=VEHICLE_TRACE_SCHEMA,
                )
                part_path = vehicle_dir / f"part-{vehicle_part_count:05d}.parquet"
                pq.write_table(
                    table,
                    part_path,
                    compression=self.compression,
                    version="2.6",
                    write_statistics=True,
                )
                vehicle_count += len(chunk)
                vehicle_part_count += 1
            if vehicle_count == 0:
                raise ValueError("a mobility trace must contain at least one vehicle record")

            signal_count = _write_signal_stream(
                root / "signals.parquet",
                signals,
                trace_id=trace_id,
                rows_per_group=self.rows_per_part,
                compression=self.compression,
            )
            _write_pair_stream(
                root / "pairs.parquet",
                pairs,
                trace_id=trace_id,
                rows_per_group=self.rows_per_part,
                compression=self.compression,
            )
            generated = _scan_trace_parquet(
                root,
                trace_id,
                tuple(
                    root / "vehicles" / f"part-{index:05d}.parquet"
                    for index in range(vehicle_part_count)
                ),
            )
            if generated.vehicle_rows != vehicle_count or generated.signal_rows != signal_count:
                raise TraceIntegrityError(
                    "generated trace row counts failed pre-publication verification",
                    artifact_path=root,
                )

        base_versions = {"pyarrow": pa.__version__}
        base_versions.update(dict(software_versions or {}))
        caller_notes = list(notes)
        caller_notes.extend(
            [
                f"trace_schema_version={TRACE_SCHEMA_VERSION}",
                "vehicle_rows and partitions are verified after publication",
            ]
        )
        stored = self.store.create(
            family=self.family,
            artifact_id=trace_id,
            artifact_type=TRACE_ARTIFACT_TYPE,
            config_hash=config_hash,
            config_scope_hashes=config_scope_hashes,
            code_version=code_version,
            producer=produce,
            required_files=_REQUIRED_TRACE_FILES,
            input_artifacts=input_artifacts,
            random_seeds=random_seeds,
            software_versions=base_versions,
            notes=caller_notes,
        )
        report = _verify_trace_layout(stored)
        if (
            report.vehicle_rows != vehicle_count
            or report.signal_rows != signal_count
            or report.vehicle_parts != vehicle_part_count
        ):
            raise TraceIntegrityError(
                "published trace row counts changed during verification",
                artifact_path=stored.path,
            )
        return stored


class MobilityTraceReader:
    """Verified, lazy reader for an immutable mobility trace artifact."""

    def __init__(self, artifact_path: str | os.PathLike[str]) -> None:
        path = Path(artifact_path)
        self.artifact = verify_artifact(
            path,
            expected_artifact_type=TRACE_ARTIFACT_TYPE,
            expected_artifact_id=path.name,
        )
        self.report = _verify_trace_layout(self.artifact)

    @classmethod
    def from_store(
        cls,
        store: ArtifactStore,
        trace_id: str,
        *,
        family: str = TRACE_ARTIFACT_FAMILY,
    ) -> MobilityTraceReader:
        """Resolve the conventional trace location from an artifact store."""

        return cls(store.artifact_path(family, trace_id))

    @property
    def trace_id(self) -> str:
        return self.artifact.manifest.artifact_id

    @property
    def vehicle_part_paths(self) -> tuple[Path, ...]:
        return tuple(sorted((self.artifact.path / "vehicles").glob("part-*.parquet")))

    @property
    def signals_path(self) -> Path:
        return self.artifact.path / "signals.parquet"

    def iter_vehicles(self, *, batch_size: int = 65_536) -> Iterator[VehicleTraceRecord]:
        """Yield vehicle records in partition and row order without full loading."""

        _validate_batch_size(batch_size)
        for path in self.vehicle_part_paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                for row in batch.to_pylist():
                    yield _vehicle_from_mapping(row)

    def iter_signals(self, *, batch_size: int = 65_536) -> Iterator[SignalStateRecord]:
        """Yield traffic-signal states without loading the full table."""

        _validate_batch_size(batch_size)
        parquet = pq.ParquetFile(self.signals_path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield _signal_from_mapping(row)

    def read_vehicle_table(self) -> pa.Table:
        """Read and concatenate all vehicle partitions using the exact schema."""

        tables = [pq.read_table(path) for path in self.vehicle_part_paths]
        return pa.concat_tables(tables)

    def read_signal_table(self) -> pa.Table:
        """Read all traffic-signal rows using the exact schema."""

        return pq.read_table(self.signals_path)


def verify_mobility_trace(path: str | os.PathLike[str]) -> TraceIntegrityReport:
    """Verify an immutable trace artifact and return its row/time summary."""

    return MobilityTraceReader(path).report


def _write_payload(target: Path, payload: TracePayload) -> None:
    if isinstance(payload, Path):
        data = payload.read_bytes()
    elif isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8")
    else:
        raise TypeError("trace source payload must be str, bytes, or pathlib.Path")
    if not data:
        raise ValueError(f"trace source payload cannot be empty: {target.name}")
    target.write_bytes(data)


def _write_pair_stream(
    path: Path,
    records: Iterable[Mapping[str, object]],
    *,
    trace_id: str,
    rows_per_group: int,
    compression: str,
) -> int:
    """Persist tagged-pair episodes so they need not be re-derived.

    Extraction reads every vehicle row in the trace, so without this the
    episodes would have to be recomputed from tens of millions of Parquet rows
    each time an experiment needs them.
    """

    count = 0
    iterator = iter(records)
    writer = pq.ParquetWriter(
        path,
        PAIR_SEGMENT_SCHEMA,
        compression=compression,
        version="2.6",
        write_statistics=True,
    )
    try:
        while chunk := tuple(islice(iterator, rows_per_group)):
            for record in chunk:
                _validate_record_trace_id(str(record["trace_id"]), trace_id)
            writer.write_table(pa.Table.from_pylist(list(chunk), schema=PAIR_SEGMENT_SCHEMA))
            count += len(chunk)
    finally:
        writer.close()
    return count


def _write_signal_stream(
    path: Path,
    records: Iterable[SignalStateRecord],
    *,
    trace_id: str,
    rows_per_group: int,
    compression: str,
) -> int:
    count = 0
    previous_time: float | None = None
    iterator = iter(records)
    writer = pq.ParquetWriter(
        path,
        SIGNAL_STATE_SCHEMA,
        compression=compression,
        version="2.6",
        write_statistics=True,
    )
    try:
        while chunk := tuple(islice(iterator, rows_per_group)):
            for record in chunk:
                _validate_record_trace_id(record.trace_id, trace_id)
                if previous_time is not None and record.time_s < previous_time:
                    raise ValueError("signal records must be ordered by non-decreasing time")
                previous_time = record.time_s
            table = pa.Table.from_pylist(
                [record.as_arrow_row() for record in chunk],
                schema=SIGNAL_STATE_SCHEMA,
            )
            writer.write_table(table)
            count += len(chunk)
    finally:
        writer.close()
    return count


def _verify_trace_layout(artifact: StoredArtifact) -> TraceIntegrityReport:
    root = artifact.path
    vehicle_paths = tuple(sorted((root / "vehicles").glob("part-*.parquet")))
    if not vehicle_paths:
        raise TraceIntegrityError(
            "trace has no vehicle Parquet partitions",
            artifact_path=root,
        )
    expected_names = [
        root / "vehicles" / f"part-{index:05d}.parquet" for index in range(len(vehicle_paths))
    ]
    if list(vehicle_paths) != expected_names:
        raise TraceIntegrityError(
            "vehicle Parquet partitions are not contiguous from part-00000",
            artifact_path=root,
        )

    expected_inventory = set(_REQUIRED_TRACE_FILES)
    expected_inventory.update(path.relative_to(root).as_posix() for path in vehicle_paths)
    manifest_inventory = {digest.path for digest in artifact.manifest.files}
    if manifest_inventory != expected_inventory:
        raise TraceIntegrityError(
            "trace artifact contains files outside the versioned trace layout",
            artifact_path=root,
            context={
                "missing": sorted(expected_inventory - manifest_inventory),
                "unexpected": sorted(manifest_inventory - expected_inventory),
            },
        )

    return _scan_trace_parquet(
        root,
        artifact.manifest.artifact_id,
        vehicle_paths,
    )


def _scan_trace_parquet(
    root: Path,
    trace_id: str,
    vehicle_paths: tuple[Path, ...],
) -> TraceIntegrityReport:
    """Validate exact schemas and identity fields in unpublished or stored files."""

    vehicle_rows = 0
    first_time = math.inf
    last_time = -math.inf
    for path in vehicle_paths:
        parquet = pq.ParquetFile(path)
        _require_schema(path, parquet.schema_arrow, VEHICLE_TRACE_SCHEMA)
        if parquet.metadata.num_rows <= 0:
            raise TraceIntegrityError(
                "vehicle partition is empty",
                artifact_path=root,
                context={"file": path.relative_to(root).as_posix()},
            )
        vehicle_rows += parquet.metadata.num_rows
        identity = parquet.read(columns=["trace_id", "time_s"])
        _validate_identity_column(identity, trace_id, path, root)
        times = identity.column("time_s").to_numpy(zero_copy_only=False)
        if len(times):
            if not bool(pc.all(pc.is_finite(identity.column("time_s"))).as_py()):
                raise TraceIntegrityError(
                    "vehicle partition contains non-finite times",
                    artifact_path=root,
                    context={"file": path.relative_to(root).as_posix()},
                )
            first_time = min(first_time, float(times.min()))
            last_time = max(last_time, float(times.max()))

    signals_path = root / "signals.parquet"
    signal_parquet = pq.ParquetFile(signals_path)
    _require_schema(signals_path, signal_parquet.schema_arrow, SIGNAL_STATE_SCHEMA)
    signal_rows = signal_parquet.metadata.num_rows
    if signal_rows:
        identity = signal_parquet.read(columns=["trace_id", "time_s"])
        _validate_identity_column(identity, trace_id, signals_path, root)
        if not bool(pc.all(pc.is_finite(identity.column("time_s"))).as_py()):
            raise TraceIntegrityError(
                "signal trace contains non-finite times",
                artifact_path=root,
                context={"file": "signals.parquet"},
            )

    return TraceIntegrityReport(
        trace_id=trace_id,
        vehicle_rows=vehicle_rows,
        signal_rows=signal_rows,
        vehicle_parts=len(vehicle_paths),
        first_time_s=first_time,
        last_time_s=last_time,
    )


def _require_schema(path: Path, actual: pa.Schema, expected: pa.Schema) -> None:
    if not actual.equals(expected, check_metadata=True):
        raise TraceIntegrityError(
            "Parquet schema does not match the versioned trace schema",
            artifact_path=path,
            context={"actual": str(actual), "expected": str(expected)},
        )


def _validate_identity_column(
    table: pa.Table,
    trace_id: str,
    path: Path,
    root: Path,
) -> None:
    matches = pc.equal(table.column("trace_id"), trace_id)
    if not bool(pc.all(matches).as_py()):
        raise TraceIntegrityError(
            "Parquet row has the wrong trace_id",
            artifact_path=root,
            context={"file": path.relative_to(root).as_posix(), "expected": trace_id},
        )


def _vehicle_from_mapping(row: Mapping[str, Any]) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=str(row["trace_id"]),
        time_s=float(row["time_s"]),
        vehicle_id=str(row["vehicle_id"]),
        x_m=float(row["x_m"]),
        y_m=float(row["y_m"]),
        heading_rad=float(row["heading_rad"]),
        speed_mps=float(row["speed_mps"]),
        acceleration_mps2=float(row["acceleration_mps2"]),
        length_m=float(row["length_m"]),
        width_m=float(row["width_m"]),
        height_m=float(row["height_m"]),
        lane_id=str(row["lane_id"]),
        edge_id=str(row["edge_id"]),
        route_id=str(row["route_id"]),
        vehicle_type=str(row["vehicle_type"]),
    )


def _signal_from_mapping(row: Mapping[str, Any]) -> SignalStateRecord:
    return SignalStateRecord(
        trace_id=str(row["trace_id"]),
        time_s=float(row["time_s"]),
        signal_id=str(row["signal_id"]),
        program_id=str(row["program_id"]),
        phase_index=int(row["phase_index"]),
        state=str(row["state"]),
    )


def _validate_record_trace_id(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(
            f"record trace_id {actual!r} does not match artifact trace_id {expected!r}"
        )


def _validate_batch_size(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("batch_size must be a positive integer")


def _require_nonempty_strings(**values: str) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def _require_finite_values(**values: float) -> None:
    for name, value in values.items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")


__all__ = [
    "MobilityTraceReader",
    "MobilityTraceWriter",
    "SIGNAL_STATE_SCHEMA",
    "SignalStateRecord",
    "TRACE_ARTIFACT_FAMILY",
    "TRACE_ARTIFACT_TYPE",
    "TRACE_SCHEMA_VERSION",
    "TraceIntegrityReport",
    "VEHICLE_TRACE_SCHEMA",
    "VehicleTraceRecord",
    "verify_mobility_trace",
]
