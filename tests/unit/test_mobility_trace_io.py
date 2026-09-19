"""Tests for exact-schema, partitioned, manifest-backed mobility traces."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.core.errors import TraceIntegrityError
from hybrid_v2x_rl.core.types import VehicleState
from hybrid_v2x_rl.mobility.trace_io import (
    PAIR_SEGMENT_SCHEMA,
    SIGNAL_STATE_SCHEMA,
    TRACE_ARTIFACT_TYPE,
    VEHICLE_TRACE_SCHEMA,
    MobilityTraceReader,
    MobilityTraceWriter,
    SignalStateRecord,
    VehicleTraceRecord,
    verify_mobility_trace,
)


def _vehicles(trace_id: str = "trace-001") -> list[VehicleTraceRecord]:
    return [
        VehicleTraceRecord(
            trace_id=trace_id,
            time_s=float(index // 2) * 0.05,
            vehicle_id=f"veh-{index}",
            x_m=10.0 + index,
            y_m=20.0,
            heading_rad=0.0,
            speed_mps=11.18,
            acceleration_mps2=0.0,
            length_m=4.5,
            width_m=1.8,
            height_m=1.5,
            lane_id="edge-0_0",
            edge_id="edge-0",
            route_id="route-0",
            vehicle_type="passenger",
        )
        for index in range(5)
    ]


def _signals(trace_id: str = "trace-001") -> list[SignalStateRecord]:
    return [
        SignalStateRecord(
            trace_id=trace_id,
            time_s=0.0,
            signal_id="junction-0",
            program_id="program-0",
            phase_index=0,
            state="GrGr",
        ),
        SignalStateRecord(
            trace_id=trace_id,
            time_s=0.05,
            signal_id="junction-0",
            program_id="program-0",
            phase_index=0,
            state="GrGr",
        ),
    ]


def _write_trace(
    tmp_path: Path,
    *,
    trace_id: str = "trace-001",
    vehicles: list[VehicleTraceRecord] | None = None,
) -> tuple[ArtifactStore, Path]:
    store = ArtifactStore(tmp_path / "artifacts")
    writer = MobilityTraceWriter(store, rows_per_part=2)
    artifact = writer.write(
        trace_id=trace_id,
        vehicles=vehicles if vehicles is not None else _vehicles(trace_id),
        signals=_signals(trace_id),
        network_definition=b'{"avenues": 6}',
        route_definition="{}",
        pairs=[],
        resolved_config_yaml="service:\n  payload_bytes: 300\n",
        config_hash="a" * 64,
        code_version="test",
        random_seeds={"mobility": 123},
    )
    return store, artifact.path


def test_trace_writer_partitions_and_reader_round_trips_exact_records(
    tmp_path: Path,
) -> None:
    store, path = _write_trace(tmp_path)
    reader = MobilityTraceReader.from_store(store, "trace-001")

    assert reader.artifact.path == path
    assert reader.artifact.manifest.artifact_type == TRACE_ARTIFACT_TYPE
    assert [part.name for part in reader.vehicle_part_paths] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert list(reader.iter_vehicles(batch_size=1)) == _vehicles()
    assert list(reader.iter_signals(batch_size=1)) == _signals()
    assert reader.report.vehicle_rows == 5
    assert reader.report.signal_rows == 2
    assert reader.report.vehicle_parts == 3
    assert reader.report.first_time_s == 0.0
    assert reader.report.last_time_s == 0.1
    assert verify_mobility_trace(path) == reader.report


def test_persisted_schemas_have_exact_column_order_types_and_nullability(
    tmp_path: Path,
) -> None:
    _, path = _write_trace(tmp_path)
    actual_vehicle = pq.ParquetFile(path / "vehicles/part-00000.parquet").schema_arrow
    actual_signal = pq.ParquetFile(path / "signals.parquet").schema_arrow

    assert actual_vehicle.equals(VEHICLE_TRACE_SCHEMA, check_metadata=True)
    assert actual_signal.equals(SIGNAL_STATE_SCHEMA, check_metadata=True)
    assert actual_vehicle.names == [
        "trace_id",
        "time_s",
        "vehicle_id",
        "x_m",
        "y_m",
        "heading_rad",
        "speed_mps",
        "acceleration_mps2",
        "length_m",
        "width_m",
        "height_m",
        "lane_id",
        "edge_id",
        "route_id",
        "vehicle_type",
    ]
    assert all(not field.nullable for field in actual_vehicle)


def test_vehicle_trace_record_converts_to_and_from_core_state() -> None:
    state = VehicleState(
        vehicle_id="veh",
        time_s=3.0,
        x_m=1.0,
        y_m=2.0,
        heading_rad=1.57,
        speed_mps=8.0,
        acceleration_mps2=-0.5,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="lane",
        edge_id="edge",
        route_id="route",
    )
    record = VehicleTraceRecord.from_vehicle_state("trace", "passenger", state)
    assert record.to_vehicle_state() == state
    assert record.vehicle_type == "passenger"


def test_wrong_trace_identity_aborts_without_partial_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    writer = MobilityTraceWriter(store, rows_per_part=2)
    records = _vehicles()
    records[2] = VehicleTraceRecord(**{**records[2].as_arrow_row(), "trace_id": "other-trace"})

    with pytest.raises(ValueError, match="does not match"):
        writer.write(
            trace_id="trace-001",
            vehicles=records,
            signals=(),
            network_definition="{}",
            route_definition="{}",
            pairs=[],
            resolved_config_yaml="config: true",
            config_hash="hash",
            code_version="test",
        )
    assert list((tmp_path / "traces").iterdir()) == []


def test_reader_rejects_digest_tampering_before_reading_parquet(tmp_path: Path) -> None:
    _, path = _write_trace(tmp_path)
    part = path / "vehicles/part-00000.parquet"
    with part.open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(TraceIntegrityError, match="size|digest"):
        MobilityTraceReader(path)


def test_reader_rejects_manifest_valid_but_wrong_parquet_schema(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    def produce(root: Path) -> None:
        (root / "vehicles").mkdir()
        wrong = pa.table({"trace_id": ["trace-wrong"], "time_s": [0.0]})
        pq.write_table(wrong, root / "vehicles/part-00000.parquet")
        pq.write_table(
            pa.Table.from_pylist([], schema=SIGNAL_STATE_SCHEMA),
            root / "signals.parquet",
        )
        pq.write_table(
            pa.Table.from_pylist([], schema=PAIR_SEGMENT_SCHEMA),
            root / "pairs.parquet",
        )
        (root / "network.json").write_text("<net/>", encoding="utf-8")
        (root / "routes.json").write_text("<routes/>", encoding="utf-8")
        (root / "resolved_config.yaml").write_text("config: true", encoding="utf-8")

    artifact = store.create(
        family="traces",
        artifact_id="trace-wrong",
        artifact_type=TRACE_ARTIFACT_TYPE,
        config_hash="hash",
        code_version="test",
        producer=produce,
    )
    with pytest.raises(TraceIntegrityError, match="schema"):
        MobilityTraceReader(artifact.path)


def test_empty_signal_stream_is_a_valid_exact_schema_file(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = MobilityTraceWriter(store).write(
        trace_id="trace-no-signals",
        vehicles=_vehicles("trace-no-signals"),
        signals=(),
        network_definition="{}",
        route_definition="{}",
        pairs=[],
        resolved_config_yaml="config: true",
        config_hash="hash",
        code_version="test",
    )
    reader = MobilityTraceReader(artifact.path)
    assert reader.report.signal_rows == 0
    assert list(reader.iter_signals()) == []
