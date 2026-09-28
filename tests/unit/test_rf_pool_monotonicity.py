"""Monotonicity checks for action-coupled RF collision risk."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-monotonicity-contract"
ACTIVE_PAIRS = 200
MAX_OFFERED_RF_ATTEMPTS = 4 * ACTIVE_PAIRS
SENSED_FRACTIONS = (0.0, 0.5, 1.0)


def _model(band: SensitivityBand) -> RFPoolModel:
    config = load_headline_config(PROJECT_ROOT)
    return RFPoolModel(
        parameters=build_rf_channel(config, band=band).collision,
        sensitivity_band=band,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )


def _demand(offered_rf_attempts: int) -> RFPoolDemand:
    """Keep 200 actors fixed while their joint action raises demand 0..800."""

    full_rf4_pairs, remainder = divmod(offered_rf_attempts, 4)
    rows = tuple(
        (
            f"pair-{index:03d}",
            4
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


@pytest.mark.parametrize("band", tuple(SensitivityBand), ids=lambda band: band.value)
@pytest.mark.parametrize(
    "sensed_fraction",
    SENSED_FRACTIONS,
    ids=lambda value: f"sensed-{value:g}",
)
def test_collision_risk_never_decreases_at_any_adjacent_offered_load(
    band: SensitivityBand,
    sensed_fraction: float,
) -> None:
    """Sweep all reachable loads with population and channel inputs frozen."""

    model = _model(band)
    responses = tuple(
        model.evaluate(_demand(demand), sensed_fraction=sensed_fraction)
        for demand in range(MAX_OFFERED_RF_ATTEMPTS + 1)
    )
    probabilities = tuple(
        response.per_attempt_collision_probability for response in responses
    )

    decreases = tuple(
        (offered, lower, higher)
        for offered, (lower, higher) in enumerate(pairwise(probabilities), start=1)
        if higher < lower
    )
    assert decreases == ()

    # Zero and one offered attempt both have no contender. Every subsequent
    # attempt adds positive hidden-contender mass in every declared band.
    assert probabilities[0] == pytest.approx(0.0)
    assert probabilities[1] == pytest.approx(0.0)
    assert all(higher > lower for lower, higher in pairwise(probabilities[1:]))

    saturation = model.attempt_parameters.candidate_resources
    assert saturation == 200
    assert responses[saturation].channel_busy_ratio == pytest.approx(1.0)
    assert responses[-1].channel_busy_ratio == pytest.approx(1.0)
    assert (
        probabilities[-1]
        > probabilities[saturation]
    ), "collision risk must keep rising after the bounded CBR reaches one"

    assert all(response.sensitivity_band is band for response in responses)
    assert all(
        response.sensed_fraction == sensed_fraction for response in responses
    )
    assert all(
        response.demand.active_pairs == ACTIVE_PAIRS for response in responses
    )
