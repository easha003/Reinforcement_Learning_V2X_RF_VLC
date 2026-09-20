"""Phase 4 boundary from complete joint actions to shared RF demand."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hybrid_v2x_rl.channels.rf.collision import MODEL_NAME, SensitivityBand
from hybrid_v2x_rl.channels.rf.model import (
    RFPropagationRequest,
    RFPropagationResult,
)
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
)
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.action_ledger import (
    FrameActionLedger,
    PairActionLifecycle,
    PairActionReservation,
)
from hybrid_v2x_rl.mean_field.rf_pool import (
    RFAttemptRisk,
    RFPoolDemand,
    RFPoolError,
    RFPoolModel,
)

TRACE_ID = "synthetic-d20-train-000"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=0.3,
    vlc_activation_cost=2.0,
)


def _model(
    band: SensitivityBand = SensitivityBand.NOMINAL,
) -> RFPoolModel:
    config = load_headline_config(PROJECT_ROOT)
    return RFPoolModel(
        parameters=build_rf_channel(config, band=band).collision,
        sensitivity_band=band,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )


def _demand_with_total_attempts(offered_rf_attempts: int) -> RFPoolDemand:
    rows: list[tuple[str, int]] = []
    remaining = offered_rf_attempts
    index = 0
    while remaining:
        attempts = min(4, remaining)
        rows.append((f"pair-{index:03d}", attempts))
        remaining -= attempts
        index += 1
    reservations = tuple(rows)
    return RFPoolDemand(
        trace_id=TRACE_ID,
        frame_index=4,
        time_s=0.4,
        active_pairs=len(reservations),
        reserved_rf_attempts_by_pair=reservations,
        offered_rf_attempts=offered_rf_attempts,
        rf_using_pairs=len(reservations),
    )


def _propagation(
    *,
    state: RFPropagationState = RFPropagationState.LOS,
    fading_power_gain: float = 1.0,
) -> RFPropagationResult:
    config = load_headline_config(PROJECT_ROOT)
    channel = build_rf_channel(config)
    return channel.evaluate_propagation(
        RFPropagationRequest(
            distance_m=100.0,
            propagation_state=state,
            blockage_db=0.0,
            shadowing_normalized=0.0,
            fading_power_gain=fading_power_gain,
        )
    )


def _ledger(
    actions: tuple[PolicyAction, ...],
    *,
    frame_index: int = 4,
) -> FrameActionLedger:
    lifecycle = PairActionLifecycle(
        episode_step=frame_index,
        born=frame_index == 0,
        terminated=False,
        truncated=False,
        bootstrap_valid=False,
        end_reason=None,
    )
    return FrameActionLedger(
        trace_id=TRACE_ID,
        frame_index=frame_index,
        time_s=frame_index * 0.1,
        resource_map=RESOURCE_MAP,
        reservations=tuple(
            PairActionReservation(
                pair_id=f"pair-{index}",
                action=action,
                lifecycle=lifecycle,
            )
            for index, action in enumerate(actions)
        ),
    )


def test_pool_boundary_consumes_all_nine_actions_after_ledger_conservation() -> None:
    ledger = _ledger(tuple(PolicyAction))

    demand = RFPoolDemand.from_ledger(ledger)

    assert demand.trace_id == TRACE_ID
    assert demand.frame_index == ledger.frame_index
    assert demand.time_s == pytest.approx(ledger.time_s)
    assert demand.active_pairs == 9
    assert demand.reserved_rf_attempts_by_pair == (
        ("pair-0", 0),
        ("pair-1", 1),
        ("pair-2", 2),
        ("pair-3", 3),
        ("pair-4", 4),
        ("pair-5", 1),
        ("pair-6", 2),
        ("pair-7", 3),
        ("pair-8", 4),
    )
    assert demand.offered_rf_attempts == 20
    assert demand.rf_using_pairs == 8
    assert demand.as_dict()["offered_rf_attempts"] == 20
    assert demand.as_dict()["reserved_rf_attempts_by_pair"][0] == {
        "pair_id": "pair-0",
        "reserved_rf_attempts": 0,
    }


def test_pool_demand_is_recomputed_from_each_current_joint_action() -> None:
    first = RFPoolDemand.from_ledger(
        _ledger(
            (PolicyAction.VLC, PolicyAction.RF_1, PolicyAction.DUP_2),
            frame_index=4,
        )
    )
    second = RFPoolDemand.from_ledger(
        _ledger(
            (PolicyAction.RF_4, PolicyAction.DUP_4, PolicyAction.DUP_4),
            frame_index=5,
        )
    )

    assert first.offered_rf_attempts == 3
    assert first.rf_using_pairs == 2
    assert second.offered_rf_attempts == 12
    assert second.rf_using_pairs == 3
    assert first.reserved_rf_attempts_by_pair == (
        ("pair-0", 0),
        ("pair-1", 1),
        ("pair-2", 2),
    )
    assert second.reserved_rf_attempts_by_pair == (
        ("pair-0", 4),
        ("pair-1", 4),
        ("pair-2", 4),
    )


@pytest.mark.parametrize(
    ("actions", "active_pairs"),
    [
        ((), 0),
        ((PolicyAction.VLC,), 1),
        ((PolicyAction.VLC, PolicyAction.VLC, PolicyAction.VLC), 3),
    ],
)
def test_empty_and_vlc_only_populations_offer_zero_rf_load(
    actions: tuple[PolicyAction, ...],
    active_pairs: int,
) -> None:
    demand = RFPoolDemand.from_ledger(_ledger(actions))

    assert demand.active_pairs == active_pairs
    assert demand.offered_rf_attempts == 0
    assert demand.rf_using_pairs == 0
    assert all(attempts == 0 for _, attempts in demand.reserved_rf_attempts_by_pair)


def test_pool_demand_rejects_aggregate_or_identity_drift() -> None:
    valid = RFPoolDemand.from_ledger(
        _ledger((PolicyAction.VLC, PolicyAction.DUP_4))
    )

    with pytest.raises(RFPoolError, match="reservation sum"):
        replace(valid, offered_rf_attempts=5)
    with pytest.raises(RFPoolError, match="RF-using pair count"):
        replace(valid, rf_using_pairs=2)
    with pytest.raises(RFPoolError, match="canonical"):
        replace(valid, reserved_rf_attempts_by_pair=tuple(
            reversed(valid.reserved_rf_attempts_by_pair)
        ))
    with pytest.raises(RFPoolError, match="invalid rows"):
        replace(
            valid,
            reserved_rf_attempts_by_pair=(("pair-0", 0), ("pair-1", 5)),
            offered_rf_attempts=5,
        )


def test_pool_boundary_refuses_inputs_that_bypass_the_complete_ledger() -> None:
    with pytest.raises(RFPoolError, match="complete FrameActionLedger"):
        RFPoolDemand.from_ledger(object())  # type: ignore[arg-type]


def test_pool_model_counts_each_reserved_attempt_once() -> None:
    model = _model()
    config = load_headline_config(PROJECT_ROOT)

    assert model.parameters.airtime_s == pytest.approx(
        config.rf.timing.airtime_s * config.service.rf_attempts_per_packet
    )
    assert model.attempt_parameters.airtime_s == pytest.approx(
        config.rf.timing.airtime_s
    )
    assert model.attempt_parameters.rf_usage_fraction == 1.0


@pytest.mark.parametrize(
    (
        "offered_rf_attempts",
        "expected_utilization",
        "expected_cbr",
        "expected_contenders",
        "expected_oversubscribed",
    ),
    [
        (0, 0.0, 0.0, 0, False),
        (1, 1.0 / 400.0, 1.0 / 400.0, 0, False),
        (400, 1.0, 1.0, 399, False),
        (800, 2.0, 1.0, 799, True),
    ],
)
def test_pool_response_separates_unclipped_utilization_from_clipped_cbr(
    offered_rf_attempts: int,
    expected_utilization: float,
    expected_cbr: float,
    expected_contenders: int,
    expected_oversubscribed: bool,
) -> None:
    response = _model().evaluate(
        _demand_with_total_attempts(offered_rf_attempts)
    )

    assert response.candidate_resources == 400
    assert response.offered_airtime_s == pytest.approx(
        offered_rf_attempts * 0.0005
    )
    assert response.pool_capacity_airtime_s == pytest.approx(0.2)
    assert response.pool_utilization == pytest.approx(expected_utilization)
    assert response.channel_busy_ratio == pytest.approx(expected_cbr)
    assert response.contending_attempts == expected_contenders
    assert response.oversubscribed is expected_oversubscribed


def test_collision_response_matches_the_validated_birthday_model() -> None:
    response = _model().evaluate(
        _demand_with_total_attempts(20),
        sensed_fraction=0.75,
    )
    expected_hidden = 19 * (1.0 - 0.85 * 0.75)
    expected_collision = 1.0 - (1.0 - 1.0 / 400.0) ** expected_hidden

    assert response.sensing_reliability == pytest.approx(0.85)
    assert response.hidden_contenders == pytest.approx(expected_hidden)
    assert response.per_attempt_collision_probability == pytest.approx(
        expected_collision
    )
    assert response.model_name == MODEL_NAME
    assert response.as_dict()["model_name"] == MODEL_NAME


def test_declared_sensitivity_bands_order_collision_risk() -> None:
    demand = _demand_with_total_attempts(80)
    responses = {
        band: _model(band).evaluate(demand)
        for band in SensitivityBand
    }

    assert (
        responses[SensitivityBand.OPTIMISTIC].per_attempt_collision_probability
        < responses[SensitivityBand.NOMINAL].per_attempt_collision_probability
        < responses[SensitivityBand.PESSIMISTIC].per_attempt_collision_probability
    )
    assert responses[SensitivityBand.OPTIMISTIC].sensing_reliability == 0.95
    assert responses[SensitivityBand.NOMINAL].sensing_reliability == 0.85
    assert responses[SensitivityBand.PESSIMISTIC].sensing_reliability == 0.70


def test_response_and_model_fail_closed_on_inconsistent_fields() -> None:
    model = _model()
    response = model.evaluate(_demand_with_total_attempts(4))

    with pytest.raises(RFPoolError, match="do not reconcile"):
        replace(response, channel_busy_ratio=0.9)
    with pytest.raises(RFPoolError, match="sensitivity band"):
        RFPoolModel(
            parameters=model.parameters,
            sensitivity_band=SensitivityBand.OPTIMISTIC,
            attempt_airtime_s=model.attempt_airtime_s,
        )


def test_attempt_risk_combines_current_contention_with_fixed_propagation() -> None:
    model = _model()
    propagation = _propagation()
    low = model.combine_attempt_risk(
        model.evaluate(_demand_with_total_attempts(4)),
        pair_id="pair-000",
        propagation=propagation,
    )
    high = model.combine_attempt_risk(
        model.evaluate(_demand_with_total_attempts(80)),
        pair_id="pair-000",
        propagation=propagation,
    )

    assert low.propagation is propagation
    assert high.propagation is propagation
    assert low.decoding_failure_probability == pytest.approx(
        high.decoding_failure_probability
    )
    assert low.half_duplex_probability == pytest.approx(
        high.half_duplex_probability
    )
    assert low.collision_probability < high.collision_probability
    assert low.access_failure_probability < high.access_failure_probability
    assert low.total_failure_probability < high.total_failure_probability


def test_attempt_risk_changes_decoding_without_changing_contention() -> None:
    model = _model()
    response = model.evaluate(_demand_with_total_attempts(20))
    clear = model.combine_attempt_risk(
        response,
        pair_id="pair-000",
        propagation=_propagation(),
    )
    faded = model.combine_attempt_risk(
        response,
        pair_id="pair-000",
        propagation=_propagation(
            state=RFPropagationState.NLOS,
            fading_power_gain=1e-4,
        ),
    )

    assert clear.collision_probability == pytest.approx(
        faded.collision_probability
    )
    assert clear.half_duplex_probability == pytest.approx(
        faded.half_duplex_probability
    )
    assert clear.decoding_failure_probability < faded.decoding_failure_probability
    assert clear.total_failure_probability < faded.total_failure_probability


def test_attempt_risk_preserves_each_failure_mechanism_and_exact_composition() -> None:
    model = _model()
    response = model.evaluate(
        _demand_with_total_attempts(20),
        sensed_fraction=0.75,
    )
    risk = model.combine_attempt_risk(
        response,
        pair_id="pair-000",
        propagation=_propagation(fading_power_gain=1e-4),
    )

    expected_half_duplex = (20 / 5) * 0.0005 / 0.1
    expected_access = 1.0 - (1.0 - risk.collision_probability) * (
        1.0 - expected_half_duplex
    )
    expected_total = 1.0 - (1.0 - expected_access) * (
        1.0 - risk.decoding_failure_probability
    )
    diagnostics = risk.as_dict()

    assert risk.reserved_rf_attempts == 4
    assert risk.half_duplex_probability == pytest.approx(expected_half_duplex)
    assert risk.access_failure_probability == pytest.approx(expected_access)
    assert risk.total_failure_probability == pytest.approx(expected_total)
    assert diagnostics["collision_probability"] == pytest.approx(
        risk.collision_probability
    )
    assert diagnostics["decoding_failure_probability"] == pytest.approx(
        risk.decoding_failure_probability
    )
    assert diagnostics["total_failure_probability"] == pytest.approx(
        risk.total_failure_probability
    )
    assert "success" not in diagnostics
    assert "failure_cause" not in diagnostics


def test_attempt_risk_rejects_absent_and_vlc_only_pairs() -> None:
    model = _model()
    response = model.evaluate(
        RFPoolDemand.from_ledger(
            _ledger((PolicyAction.VLC, PolicyAction.RF_1))
        )
    )
    propagation = _propagation()

    with pytest.raises(RFPoolError, match="VLC-only"):
        model.combine_attempt_risk(
            response,
            pair_id="pair-0",
            propagation=propagation,
        )
    with pytest.raises(RFPoolError, match="absent"):
        model.combine_attempt_risk(
            response,
            pair_id="not-active",
            propagation=propagation,
        )


def test_attempt_risk_fails_closed_if_composed_fields_drift() -> None:
    model = _model()
    risk = model.combine_attempt_risk(
        model.evaluate(_demand_with_total_attempts(4)),
        pair_id="pair-000",
        propagation=_propagation(),
    )

    with pytest.raises(RFPoolError, match="do not reconcile"):
        replace(risk, access_failure_probability=0.9)
    with pytest.raises(RFPoolError, match="reserved attempts"):
        replace(risk, reserved_rf_attempts=3)

    assert isinstance(risk, RFAttemptRisk)
