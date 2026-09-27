"""RF contention-domain geometry and decision artifact."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hybrid_v2x_rl.agents.regime_evaluation import EvaluationWindow
from hybrid_v2x_rl.mean_field.contention_domain_audit import (
    CONTENTION_DOMAIN_AUDIT_SCHEMA,
    ContentionDomainAuditError,
    ContentionDomainAuditReport,
    measure_frame_contention_domains,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord


def _vehicle(vehicle_id: str, x_m: float) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id="synthetic-d10-validation-000",
        time_s=0.0,
        vehicle_id=vehicle_id,
        x_m=x_m,
        y_m=0.0,
        heading_rad=0.0,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        length_m=4.5,
        width_m=1.8,
        height_m=1.5,
        lane_id="edge-0_0",
        edge_id="edge-0",
        route_id="route-0",
        vehicle_type="passenger",
    )


def _frame(*, empty: bool = False) -> PopulationFrame:
    vehicles = tuple(
        sorted(
            (
                _vehicle("a", 0.0),
                _vehicle("a-rx", 10.0),
                _vehicle("b", 100.0),
                _vehicle("b-rx", 110.0),
                _vehicle("c", 500.0),
                _vehicle("c-rx", 510.0),
            ),
            key=lambda vehicle: vehicle.vehicle_id,
        )
    )
    by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}
    pairs = () if empty else tuple(
        PopulationPair(
            pair_id=pair_id,
            episode_step=0,
            transmitter=by_id[transmitter],
            receiver=by_id[receiver],
            lifecycle=PairLifecycle(born=True),
        )
        for pair_id, transmitter, receiver in (
            ("pair-a", "a", "a-rx"),
            ("pair-b", "b", "b-rx"),
            ("pair-c", "c", "c-rx"),
        )
    )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path("synthetic-d10-validation-000"),
            trace_id="synthetic-d10-validation-000",
            split="validation",
            density=10.0,
            replicate=0,
        ),
        index=0,
        time_s=0.0,
        vehicles=vehicles,
        pairs=pairs,
    )


def test_pair_local_domains_exclude_distant_frame_flows() -> None:
    result = measure_frame_contention_domains(_frame(), radius_m=200.0)

    assert result.active_pairs == 3
    assert result.local_pair_flows == (2, 2, 1)
    assert result.local_vehicle_neighbours == (3, 3, 1)


def test_empty_frame_has_no_local_domain_rows() -> None:
    result = measure_frame_contention_domains(_frame(empty=True), radius_m=200.0)

    assert result.active_pairs == 0
    assert result.local_pair_flows == ()
    assert result.local_vehicle_neighbours == ()


def test_contention_radius_must_be_positive() -> None:
    with pytest.raises(ContentionDomainAuditError, match="radius"):
        measure_frame_contention_domains(_frame(), radius_m=0.0)


@pytest.mark.parametrize(("equal_fraction", "matches"), [(0.0, False), (1.0, True)])
def test_report_gate_follows_measured_domain_equality(
    tmp_path: Path,
    equal_fraction: float,
    matches: bool,
) -> None:
    report = ContentionDomainAuditReport(
        config_hash="a" * 64,
        policy_environment_scope_hash="b" * 64,
        audit_path=Path("audit.json"),
        audit_sha256="c" * 64,
        radius_m=200.0,
        windows=(
            EvaluationWindow(
                trace_id="synthetic-d10-validation-000",
                density=10.0,
                start_frame_index=0,
                frames=16,
            ),
        ),
        densities=({"density_vehicles_per_lane_km": 10.0},),
        campaign={
            "fraction_rows_whose_local_domain_equals_global_frame": equal_fraction
        },
        generated_at_utc=datetime(2026, 9, 27, tzinfo=UTC),
    )

    payload = report.as_dict()

    assert payload["schema"] == CONTENTION_DOMAIN_AUDIT_SCHEMA
    assert payload["test_split_opened"] is False
    decision = payload["decision"]
    assert isinstance(decision, Mapping)
    assert decision["global_frame_pool_matches_declared_local_domain"] is matches
    assert decision["rf_contention_model_repair_required"] is (not matches)
    output = report.write_json(tmp_path / "domain.json")
    assert json.loads(output.read_text(encoding="utf-8")) == payload
