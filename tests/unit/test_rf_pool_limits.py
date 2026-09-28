"""Independent analytical limiting-case checks for the Phase 4 RF pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-analytical-limit-contract"
ACTIVE_PAIRS = 200
MAX_RF_ATTEMPTS_PER_PAIR = 4

# Independent headline primitives. These deliberately do not call the
# collision module's resource-demand, CBR, hidden-contender, or collision
# helpers: the point is to cross-check the adapter against the equations.
SUBCHANNELS = 1
SELECTION_WINDOW_SLOTS = 200
ATTEMPT_AIRTIME_S = 0.0005
GENERATION_PERIOD_S = 0.1
CANDIDATE_RESOURCES = SUBCHANNELS * SELECTION_WINDOW_SLOTS
POOL_CAPACITY_AIRTIME_S = SUBCHANNELS * GENERATION_PERIOD_S
SENSING_RELIABILITY = {
    SensitivityBand.OPTIMISTIC: 0.95,
    SensitivityBand.NOMINAL: 0.85,
    SensitivityBand.PESSIMISTIC: 0.70,
}


@dataclass(frozen=True, slots=True)
class AnalyticalCase:
    """A boundary load evaluated under one fixed sensing condition."""

    name: str
    offered_rf_attempts: int
    band: SensitivityBand
    sensed_fraction: float


ANALYTICAL_CASES = (
    AnalyticalCase("zero-load", 0, SensitivityBand.NOMINAL, 1.0),
    AnalyticalCase("focal-attempt-only", 1, SensitivityBand.PESSIMISTIC, 0.0),
    AnalyticalCase("one-below-saturation", 199, SensitivityBand.OPTIMISTIC, 0.25),
    AnalyticalCase("exact-saturation", 200, SensitivityBand.NOMINAL, 0.5),
    AnalyticalCase("first-overload", 201, SensitivityBand.PESSIMISTIC, 0.75),
    AnalyticalCase("twice-capacity", 400, SensitivityBand.NOMINAL, 1.0),
)


def _model(band: SensitivityBand) -> RFPoolModel:
    config = load_headline_config(PROJECT_ROOT)
    return RFPoolModel(
        parameters=build_rf_channel(config, band=band).collision,
        sensitivity_band=band,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )


def _demand(offered_rf_attempts: int) -> RFPoolDemand:
    assert 0 <= offered_rf_attempts <= ACTIVE_PAIRS * MAX_RF_ATTEMPTS_PER_PAIR
    full_rf4_pairs, remainder = divmod(
        offered_rf_attempts,
        MAX_RF_ATTEMPTS_PER_PAIR,
    )
    rows = tuple(
        (
            f"pair-{index:03d}",
            MAX_RF_ATTEMPTS_PER_PAIR
            if index < full_rf4_pairs
            else remainder
            if index == full_rf4_pairs
            else 0,
        )
        for index in range(ACTIVE_PAIRS)
    )
    return RFPoolDemand(
        trace_id=TRACE_ID,
        frame_index=0,
        time_s=0.0,
        active_pairs=ACTIVE_PAIRS,
        reserved_rf_attempts_by_pair=rows,
        offered_rf_attempts=offered_rf_attempts,
        rf_using_pairs=full_rf4_pairs + int(remainder > 0),
    )


def _stable_birthday_collision(hidden_contenders: float) -> float:
    """Compute ``1 - (1 - 1/M)^n`` without subtractive cancellation."""

    return -math.expm1(
        hidden_contenders * math.log1p(-1.0 / CANDIDATE_RESOURCES)
    )


@pytest.mark.parametrize(
    "case",
    ANALYTICAL_CASES,
    ids=lambda case: case.name,
)
def test_rf_pool_boundaries_match_independent_closed_form(
    case: AnalyticalCase,
) -> None:
    """Cross-check both sides of every physical boundary from primitives."""

    response = _model(case.band).evaluate(
        _demand(case.offered_rf_attempts),
        sensed_fraction=case.sensed_fraction,
    )

    expected_offered_airtime = case.offered_rf_attempts * ATTEMPT_AIRTIME_S
    expected_utilization = expected_offered_airtime / POOL_CAPACITY_AIRTIME_S
    expected_contenders = max(0, case.offered_rf_attempts - 1)
    expected_hidden = expected_contenders * (
        1.0 - SENSING_RELIABILITY[case.band] * case.sensed_fraction
    )
    expected_collision = _stable_birthday_collision(expected_hidden)

    assert response.parameters.subchannels == SUBCHANNELS
    assert response.parameters.selection_window_slots == SELECTION_WINDOW_SLOTS
    assert response.attempt_airtime_s == pytest.approx(ATTEMPT_AIRTIME_S)
    assert response.parameters.generation_period_s == pytest.approx(
        GENERATION_PERIOD_S
    )
    assert response.sensing_reliability == pytest.approx(
        SENSING_RELIABILITY[case.band]
    )
    assert response.candidate_resources == CANDIDATE_RESOURCES
    assert response.pool_capacity_airtime_s == pytest.approx(
        POOL_CAPACITY_AIRTIME_S
    )
    assert response.offered_airtime_s == pytest.approx(expected_offered_airtime)
    assert response.pool_utilization == pytest.approx(expected_utilization)
    assert response.channel_busy_ratio == pytest.approx(
        min(1.0, expected_utilization)
    )
    assert response.contending_attempts == expected_contenders
    assert response.hidden_contenders == pytest.approx(expected_hidden)
    assert response.per_attempt_collision_probability == pytest.approx(
        expected_collision,
        rel=1e-12,
        abs=1e-15,
    )
    assert response.oversubscribed is (expected_utilization > 1.0)


def test_no_sensed_contenders_removes_sensitivity_band_dependence() -> None:
    """At sensed fraction zero, the band multiplier has nothing to multiply."""

    demand = _demand(CANDIDATE_RESOURCES)
    responses = tuple(
        _model(band).evaluate(demand, sensed_fraction=0.0)
        for band in SensitivityBand
    )
    expected_hidden = CANDIDATE_RESOURCES - 1
    expected_collision = _stable_birthday_collision(expected_hidden)

    assert all(
        response.hidden_contenders == pytest.approx(expected_hidden)
        for response in responses
    )
    assert all(
        response.per_attempt_collision_probability
        == pytest.approx(expected_collision, rel=1e-12, abs=1e-15)
        for response in responses
    )
    assert len(
        {response.per_attempt_collision_probability for response in responses}
    ) == 1
