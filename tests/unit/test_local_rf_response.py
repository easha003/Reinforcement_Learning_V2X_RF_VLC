"""Pair-specific RF collision/CBR response contract and analytical proofs."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import (
    MODEL_NAME,
    SensitivityBand,
)
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.local_rf_response import (
    LOCAL_RF_RESPONSE_CONTRACT_VERSION,
    LocalRFResponseError,
    LocalRFResponseModel,
    PairLocalRFResponse,
)
from hybrid_v2x_rl.mean_field.local_rf_sensing import (
    LOCAL_RF_SENSING_CONTRACT_VERSION,
    FrameLocalRFSensedLoads,
    LocalRFSensingReservation,
    PairLocalRFSensedLoad,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-local-response-contract"
ATTEMPT_AIRTIME_S = 0.0005
GENERATION_PERIOD_S = 0.1
SUBCHANNELS = 2
CANDIDATE_RESOURCES = 400
POOL_CAPACITY_AIRTIME_S = 0.2
SENSING_RELIABILITY = {
    SensitivityBand.OPTIMISTIC: 0.95,
    SensitivityBand.NOMINAL: 0.85,
    SensitivityBand.PESSIMISTIC: 0.70,
}


def _model(
    band: SensitivityBand = SensitivityBand.NOMINAL,
) -> LocalRFResponseModel:
    config = load_headline_config(PROJECT_ROOT)
    return LocalRFResponseModel(
        parameters=build_rf_channel(config, band=band).collision,
        sensitivity_band=band,
        attempt_airtime_s=float(config.rf.timing.airtime_s),
    )


def _split_attempts(total: int) -> tuple[int, ...]:
    full, remainder = divmod(total, 4)
    return (4,) * full + ((remainder,) if remainder else ())


def _load(
    *,
    focal_pair_id: str = "pair-000-focal",
    focal_attempts: int = 1,
    colocated_attempts: int = 0,
    sensed_external_attempts: int = 0,
    hidden_external_attempts: int = 0,
) -> PairLocalRFSensedLoad:
    if not 0 <= focal_attempts <= 4:
        raise ValueError("focal attempts must lie in [0, 4]")
    reservations = [
        LocalRFSensingReservation(
            pair_id=focal_pair_id,
            transmitter_id="tx-focal",
            reserved_rf_attempts=focal_attempts,
            focal_flow=True,
            colocated_with_focal_transmitter=True,
            geometrically_decodable=True,
        )
    ]
    reservations.extend(
        LocalRFSensingReservation(
            pair_id=f"pair-100-colocated-{index:03d}",
            transmitter_id="tx-focal",
            reserved_rf_attempts=attempts,
            focal_flow=False,
            colocated_with_focal_transmitter=True,
            geometrically_decodable=True,
        )
        for index, attempts in enumerate(_split_attempts(colocated_attempts))
    )
    reservations.extend(
        LocalRFSensingReservation(
            pair_id=f"pair-200-sensed-{index:03d}",
            transmitter_id=f"tx-sensed-{index:03d}",
            reserved_rf_attempts=attempts,
            focal_flow=False,
            colocated_with_focal_transmitter=False,
            geometrically_decodable=True,
        )
        for index, attempts in enumerate(
            _split_attempts(sensed_external_attempts)
        )
    )
    reservations.extend(
        LocalRFSensingReservation(
            pair_id=f"pair-300-hidden-{index:03d}",
            transmitter_id=f"tx-hidden-{index:03d}",
            reserved_rf_attempts=attempts,
            focal_flow=False,
            colocated_with_focal_transmitter=False,
            geometrically_decodable=False,
        )
        for index, attempts in enumerate(
            _split_attempts(hidden_external_attempts)
        )
    )
    canonical = tuple(sorted(reservations, key=lambda row: row.pair_id))
    external = sensed_external_attempts + hidden_external_attempts
    return PairLocalRFSensedLoad(
        focal_pair_id=focal_pair_id,
        focal_transmitter_id="tx-focal",
        reservations=canonical,
        local_offered_rf_attempts=(
            focal_attempts + colocated_attempts + external
        ),
        focal_rf_attempts=focal_attempts,
        colocated_other_rf_attempts=colocated_attempts,
        external_contending_rf_attempts=external,
        geometrically_sensed_external_rf_attempts=sensed_external_attempts,
        geometrically_hidden_external_rf_attempts=hidden_external_attempts,
        sensed_fraction=(
            sensed_external_attempts / external if external else 1.0
        ),
    )


def _frame_loads(
    *loads: PairLocalRFSensedLoad,
) -> FrameLocalRFSensedLoads:
    rows = tuple(sorted(loads, key=lambda row: row.focal_pair_id))
    return FrameLocalRFSensedLoads(
        contract_version=LOCAL_RF_SENSING_CONTRACT_VERSION,
        trace_id=TRACE_ID,
        frame_index=3,
        time_s=0.3,
        pair_ids=tuple(row.focal_pair_id for row in rows),
        rows=rows,
    )


def _response(
    load: PairLocalRFSensedLoad,
    band: SensitivityBand = SensitivityBand.NOMINAL,
) -> PairLocalRFResponse:
    return _model(band).evaluate(_frame_loads(load)).response_for(
        load.focal_pair_id
    )


def _stable_collision(hidden_attempts: float) -> float:
    return -math.expm1(
        hidden_attempts * math.log1p(-1.0 / CANDIDATE_RESOURCES)
    )


def test_local_response_matches_independent_attempt_equations() -> None:
    load = _load(
        focal_attempts=2,
        colocated_attempts=1,
        sensed_external_attempts=2,
        hidden_external_attempts=4,
    )

    response = _response(load)

    expected_effective_sensed = 2 * 0.85
    expected_effective_hidden = 4 + 2 * (1.0 - 0.85)
    assert response.model_name == MODEL_NAME
    assert response.candidate_resources == CANDIDATE_RESOURCES
    assert response.attempt_airtime_s == pytest.approx(ATTEMPT_AIRTIME_S)
    assert response.pool_capacity_airtime_s == pytest.approx(
        POOL_CAPACITY_AIRTIME_S
    )
    assert response.offered_airtime_s == pytest.approx(9 * ATTEMPT_AIRTIME_S)
    assert response.pool_utilization == pytest.approx(9 / CANDIDATE_RESOURCES)
    assert response.channel_busy_ratio == pytest.approx(9 / CANDIDATE_RESOURCES)
    assert response.external_contending_attempts == 6
    assert response.effective_sensed_external_attempts == pytest.approx(
        expected_effective_sensed
    )
    assert response.effective_hidden_external_attempts == pytest.approx(
        expected_effective_hidden
    )
    assert response.per_attempt_collision_probability == pytest.approx(
        _stable_collision(expected_effective_hidden),
        rel=1e-12,
        abs=1e-15,
    )
    assert response.has_focal_rf_reservation
    assert not response.oversubscribed
    assert response.load.as_dict()["focal_rf_attempts"] == 2


@pytest.mark.parametrize("band", tuple(SensitivityBand), ids=lambda band: band.value)
@pytest.mark.parametrize(
    ("sensed", "hidden"),
    ((0, 0), (8, 0), (2, 6), (0, 8)),
    ids=("no-external", "all-sensed", "mixed", "all-hidden"),
)
def test_every_band_matches_geometric_limiting_cases(
    band: SensitivityBand,
    sensed: int,
    hidden: int,
) -> None:
    response = _response(
        _load(
            focal_attempts=1,
            sensed_external_attempts=sensed,
            hidden_external_attempts=hidden,
        ),
        band,
    )
    reliability = SENSING_RELIABILITY[band]
    expected_sensed = sensed * reliability
    expected_hidden = hidden + sensed * (1.0 - reliability)

    assert response.sensing_reliability == reliability
    assert response.external_contending_attempts == sensed + hidden
    assert response.effective_sensed_external_attempts == pytest.approx(
        expected_sensed
    )
    assert response.effective_hidden_external_attempts == pytest.approx(
        expected_hidden
    )
    assert response.per_attempt_collision_probability == pytest.approx(
        _stable_collision(expected_hidden),
        rel=1e-12,
        abs=1e-15,
    )


def test_all_hidden_limit_is_band_independent() -> None:
    responses = tuple(
        _response(
            _load(focal_attempts=1, hidden_external_attempts=8),
            band,
        )
        for band in SensitivityBand
    )

    assert all(
        response.effective_hidden_external_attempts == 8.0
        for response in responses
    )
    assert len(
        {response.per_attempt_collision_probability for response in responses}
    ) == 1


def test_sensed_external_limit_orders_all_declared_bands() -> None:
    responses = {
        band: _response(
            _load(focal_attempts=1, sensed_external_attempts=8),
            band,
        )
        for band in SensitivityBand
    }

    assert (
        responses[SensitivityBand.OPTIMISTIC].per_attempt_collision_probability
        < responses[SensitivityBand.NOMINAL].per_attempt_collision_probability
        < responses[SensitivityBand.PESSIMISTIC].per_attempt_collision_probability
    )


@pytest.mark.parametrize("band", tuple(SensitivityBand), ids=lambda band: band.value)
def test_collision_is_strictly_monotone_in_external_load_for_every_band(
    band: SensitivityBand,
) -> None:
    model = _model(band)
    probabilities = tuple(
        model.evaluate(
            _frame_loads(
                _load(
                    focal_attempts=0,
                    sensed_external_attempts=external_attempts,
                )
            )
        ).responses[0].per_attempt_collision_probability
        for external_attempts in range(801)
    )

    assert probabilities[0] == 0.0
    assert all(higher > lower for lower, higher in pairwise(probabilities))


@pytest.mark.parametrize("band", tuple(SensitivityBand), ids=lambda band: band.value)
def test_collision_never_increases_as_more_external_attempts_become_sensed(
    band: SensitivityBand,
) -> None:
    probabilities = tuple(
        _response(
            _load(
                focal_attempts=1,
                sensed_external_attempts=sensed,
                hidden_external_attempts=8 - sensed,
            ),
            band,
        ).per_attempt_collision_probability
        for sensed in range(9)
    )

    assert all(higher < lower for lower, higher in pairwise(probabilities))


@pytest.mark.parametrize("band", tuple(SensitivityBand), ids=lambda band: band.value)
def test_cbr_saturates_but_utilization_preserves_overload_in_every_band(
    band: SensitivityBand,
) -> None:
    below = _response(
        _load(focal_attempts=0, hidden_external_attempts=399),
        band,
    )
    exact = _response(
        _load(focal_attempts=0, hidden_external_attempts=400),
        band,
    )
    above = _response(
        _load(focal_attempts=0, hidden_external_attempts=401),
        band,
    )

    assert below.pool_utilization == pytest.approx(399 / 400)
    assert below.channel_busy_ratio == pytest.approx(399 / 400)
    assert not below.oversubscribed
    assert exact.pool_utilization == pytest.approx(1.0)
    assert exact.channel_busy_ratio == pytest.approx(1.0)
    assert not exact.oversubscribed
    assert above.pool_utilization == pytest.approx(401 / 400)
    assert above.channel_busy_ratio == pytest.approx(1.0)
    assert above.oversubscribed
    assert (
        below.per_attempt_collision_probability
        < exact.per_attempt_collision_probability
        < above.per_attempt_collision_probability
    )


def test_focal_and_colocated_attempts_raise_cbr_but_not_external_collision() -> None:
    external_only = _response(
        _load(
            focal_attempts=0,
            colocated_attempts=0,
            sensed_external_attempts=4,
            hidden_external_attempts=4,
        )
    )
    locally_loaded = _response(
        _load(
            focal_attempts=4,
            colocated_attempts=4,
            sensed_external_attempts=4,
            hidden_external_attempts=4,
        )
    )

    assert external_only.external_contending_attempts == 8
    assert locally_loaded.external_contending_attempts == 8
    assert external_only.per_attempt_collision_probability == pytest.approx(
        locally_loaded.per_attempt_collision_probability
    )
    assert external_only.channel_busy_ratio < locally_loaded.channel_busy_ratio
    assert not external_only.has_focal_rf_reservation
    assert locally_loaded.has_focal_rf_reservation


def test_frame_response_is_pair_aligned_empty_safe_and_queryable() -> None:
    first = _load(
        focal_pair_id="pair-a",
        focal_attempts=1,
        sensed_external_attempts=2,
    )
    second = _load(
        focal_pair_id="pair-b",
        focal_attempts=0,
        hidden_external_attempts=4,
    )

    frame = _model().evaluate(_frame_loads(second, first))
    empty = _model().evaluate(_frame_loads())

    assert frame.contract_version == LOCAL_RF_RESPONSE_CONTRACT_VERSION
    assert frame.sensing_contract_version == LOCAL_RF_SENSING_CONTRACT_VERSION
    assert frame.pair_ids == ("pair-a", "pair-b")
    assert frame.response_for("pair-a").load is first
    assert frame.response_for("pair-b").load is second
    assert (
        frame.response_for("pair-a").per_attempt_collision_probability
        < frame.response_for("pair-b").per_attempt_collision_probability
    )
    assert frame.as_dict()["active_pairs"] == 2
    assert empty.pair_ids == ()
    assert empty.responses == ()
    with pytest.raises(LocalRFResponseError, match="absent"):
        frame.response_for("missing")


def test_model_normalizes_packet_airtime_without_an_rf_usage_multiplier() -> None:
    model = _model()
    config = load_headline_config(PROJECT_ROOT)

    assert model.parameters.airtime_s == pytest.approx(
        config.rf.timing.airtime_s * config.service.rf_attempts_per_packet
    )
    assert model.attempt_parameters.airtime_s == pytest.approx(
        config.rf.timing.airtime_s
    )
    assert model.attempt_parameters.rf_usage_fraction == 1.0


def test_model_and_response_fail_closed_on_inconsistent_fields() -> None:
    model = _model()
    response = model.evaluate(
        _frame_loads(_load(focal_attempts=1, sensed_external_attempts=4))
    ).responses[0]

    with pytest.raises(LocalRFResponseError, match="do not reconcile"):
        replace(response, channel_busy_ratio=0.9)
    with pytest.raises(LocalRFResponseError, match="external attempts"):
        replace(response, external_contending_attempts=3)
    with pytest.raises(LocalRFResponseError, match="sensitivity band"):
        LocalRFResponseModel(
            parameters=model.parameters,
            sensitivity_band=SensitivityBand.OPTIMISTIC,
            attempt_airtime_s=model.attempt_airtime_s,
        )
    with pytest.raises(LocalRFResponseError, match="requires FrameLocalRFSensedLoads"):
        model.evaluate(object())  # type: ignore[arg-type]


def test_frame_rejects_response_order_or_model_drift() -> None:
    model = _model()
    valid = model.evaluate(
        _frame_loads(
            _load(focal_pair_id="pair-a"),
            _load(focal_pair_id="pair-b"),
        )
    )

    with pytest.raises(LocalRFResponseError, match="align exactly"):
        replace(valid, responses=tuple(reversed(valid.responses)))
    with pytest.raises(LocalRFResponseError, match="frame model"):
        replace(
            valid,
            responses=(
                replace(
                    valid.responses[0],
                    sensitivity_band=SensitivityBand.PESSIMISTIC,
                    parameters=_model(SensitivityBand.PESSIMISTIC).attempt_parameters,
                    effective_sensed_external_attempts=0.0,
                    effective_hidden_external_attempts=0.0,
                    per_attempt_collision_probability=0.0,
                ),
                valid.responses[1],
            ),
        )


def test_headline_primitives_used_by_independent_proofs_are_frozen() -> None:
    model = _model()
    parameters = model.attempt_parameters

    assert parameters.subchannels == SUBCHANNELS
    assert parameters.selection_window_slots == 200
    assert parameters.candidate_resources == CANDIDATE_RESOURCES
    assert parameters.generation_period_s == pytest.approx(GENERATION_PERIOD_S)
    assert parameters.airtime_s == pytest.approx(ATTEMPT_AIRTIME_S)
    assert (
        parameters.subchannels * parameters.generation_period_s
        == pytest.approx(POOL_CAPACITY_AIRTIME_S)
    )
