"""Phase 4 boundary from complete joint actions to shared RF demand."""

from __future__ import annotations

from dataclasses import replace

import pytest

from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
)
from hybrid_v2x_rl.mean_field.action_ledger import (
    FrameActionLedger,
    PairActionLifecycle,
    PairActionReservation,
)
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolError

TRACE_ID = "synthetic-d20-train-000"
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=0.3,
    vlc_activation_cost=2.0,
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
