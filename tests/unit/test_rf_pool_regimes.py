"""Contract tests for the Phase 4 RF-pool load regimes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.congestion_feedback import (
    ActorObservationSchema,
    DelayedCongestionFeedback,
)
from hybrid_v2x_rl.mean_field.local_rf_response import (
    FrameLocalRFResponses,
    LocalRFResponseModel,
)
from hybrid_v2x_rl.mean_field.local_rf_sensing import (
    LOCAL_RF_SENSING_CONTRACT_VERSION,
    FrameLocalRFSensedLoads,
    LocalRFSensingReservation,
    PairLocalRFSensedLoad,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel
from hybrid_v2x_rl.observation.builder import ObservationBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-regime-contract"


@dataclass(frozen=True, slots=True)
class LoadRegime:
    """One reachable population demand and its physical-pool classification."""

    name: str
    active_pairs: int
    offered_rf_attempts: int
    expected_pool_utilization: float
    expected_channel_busy_ratio: float
    expected_oversubscribed: bool
    expected_delayed_mean_fraction: float


LOAD_REGIMES = (
    LoadRegime("empty", 0, 0, 0.0, 0.0, False, 0.0),
    # Forty attempts consume twenty percent of the corrected 200-attempt pool.
    LoadRegime("light-load", 100, 40, 0.2, 0.2, False, 0.1),
    LoadRegime("saturation", 50, 200, 1.0, 1.0, False, 1.0),
    # One hundred RF-4 actors offer twice the physical pool capacity.
    LoadRegime("overload", 100, 400, 2.0, 1.0, True, 1.0),
)


def _model() -> RFPoolModel:
    config = load_headline_config(PROJECT_ROOT)
    return RFPoolModel(
        parameters=build_rf_channel(
            config,
            band=SensitivityBand.NOMINAL,
        ).collision,
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )


def _demand(regime: LoadRegime) -> RFPoolDemand:
    remaining = regime.offered_rf_attempts
    reservations: list[tuple[str, int]] = []
    for index in range(regime.active_pairs):
        attempts = min(4, remaining)
        reservations.append((f"pair-{index:03d}", attempts))
        remaining -= attempts
    assert remaining == 0, "the regime must be reachable with at most RF-4 per actor"
    rows = tuple(reservations)
    return RFPoolDemand(
        trace_id=TRACE_ID,
        frame_index=0,
        time_s=0.0,
        active_pairs=regime.active_pairs,
        reserved_rf_attempts_by_pair=rows,
        offered_rf_attempts=regime.offered_rf_attempts,
        rf_using_pairs=sum(attempts > 0 for _, attempts in rows),
    )


def _feedback() -> DelayedCongestionFeedback:
    config = load_headline_config(PROJECT_ROOT)
    return DelayedCongestionFeedback.from_config(
        config.environment.mean_field,
        max_rf_attempts=config.environment.max_rf_attempts,
    )


def _local_responses(regime: LoadRegime) -> FrameLocalRFResponses:
    """Represent the same per-actor reservations at the live feedback boundary."""

    demand = _demand(regime)
    rows = tuple(
        PairLocalRFSensedLoad(
            focal_pair_id=pair_id,
            focal_transmitter_id=f"tx-{pair_id}",
            reservations=(
                LocalRFSensingReservation(
                    pair_id=pair_id,
                    transmitter_id=f"tx-{pair_id}",
                    reserved_rf_attempts=attempts,
                    focal_flow=True,
                    colocated_with_focal_transmitter=True,
                    geometrically_decodable=True,
                ),
            ),
            local_offered_rf_attempts=attempts,
            focal_rf_attempts=attempts,
            colocated_other_rf_attempts=0,
            external_contending_rf_attempts=0,
            geometrically_sensed_external_rf_attempts=0,
            geometrically_hidden_external_rf_attempts=0,
            sensed_fraction=1.0,
        )
        for pair_id, attempts in demand.reserved_rf_attempts_by_pair
    )
    config = load_headline_config(PROJECT_ROOT)
    collision = build_rf_channel(
        config,
        band=SensitivityBand.NOMINAL,
    ).collision
    return LocalRFResponseModel(
        parameters=collision,
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=config.rf.timing.airtime_s,
    ).evaluate(
        FrameLocalRFSensedLoads(
            contract_version=LOCAL_RF_SENSING_CONTRACT_VERSION,
            trace_id=TRACE_ID,
            frame_index=0,
            time_s=0.0,
            pair_ids=tuple(pair_id for pair_id, _ in demand.reserved_rf_attempts_by_pair),
            rows=rows,
        )
    )


def _actor_schema() -> ActorObservationSchema:
    config = load_headline_config(PROJECT_ROOT)
    return ActorObservationSchema(
        local=ObservationBuilder.from_config(config.observation).schema
    )


@pytest.mark.parametrize("regime", LOAD_REGIMES, ids=lambda regime: regime.name)
def test_named_load_regimes_have_auditable_pool_and_delayed_responses(
    regime: LoadRegime,
) -> None:
    """Exercise each named boundary from demand through next-frame actor input."""

    model = _model()
    response = model.evaluate(_demand(regime), sensed_fraction=1.0)

    assert response.candidate_resources == 200
    assert response.pool_capacity_airtime_s == pytest.approx(0.1)
    assert response.offered_airtime_s == pytest.approx(
        regime.offered_rf_attempts * 0.0005
    )
    assert response.pool_utilization == pytest.approx(
        regime.expected_pool_utilization
    )
    assert response.channel_busy_ratio == pytest.approx(
        regime.expected_channel_busy_ratio
    )
    assert response.oversubscribed is regime.expected_oversubscribed

    expected_contenders = max(0, regime.offered_rf_attempts - 1)
    expected_hidden = expected_contenders * (1.0 - 0.85)
    expected_collision = 1.0 - (1.0 - 1.0 / 200.0) ** expected_hidden
    assert response.contending_attempts == expected_contenders
    assert response.hidden_contenders == pytest.approx(expected_hidden)
    assert response.per_attempt_collision_probability == pytest.approx(
        expected_collision
    )

    feedback = _feedback()
    schema = _actor_schema()
    local = (0.0,) * schema.local.width
    feedback.reset(TRACE_ID)
    feedback.begin_frame(TRACE_ID, 0)
    assert feedback.actor_observation(schema, local)[-2:] == (0.0, 0.0)
    feedback.close_frame(_local_responses(regime))

    delayed = feedback.begin_frame(TRACE_ID, 1)
    actor = feedback.actor_observation(schema, local)
    assert delayed.source_frame_index == 0
    assert delayed.mean_rf_attempt_fraction == pytest.approx(
        regime.expected_delayed_mean_fraction
    )
    assert actor[-2:] == pytest.approx(
        (regime.expected_delayed_mean_fraction, 1.0)
    )


def test_overload_remains_visible_when_cbr_and_actor_feedback_are_bounded() -> None:
    """Unclipped utilization must retain overload after bounded signals hit one."""

    overload = LOAD_REGIMES[-1]
    response = _model().evaluate(_demand(overload))

    assert response.pool_utilization == pytest.approx(2.0)
    assert response.channel_busy_ratio == pytest.approx(1.0)
    assert overload.expected_delayed_mean_fraction == pytest.approx(1.0)
    assert response.oversubscribed
