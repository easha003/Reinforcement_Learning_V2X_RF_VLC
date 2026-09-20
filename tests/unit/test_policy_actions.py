"""Phase 3 authoritative policy-action resource semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.config.models import CostConfig
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ACTION_RESOURCE_SPECS,
    MAX_RESERVED_RF_ATTEMPTS,
    POLICY_ACTION_ORDER,
    ActionMappingError,
    ActionResourceMap,
    PolicyAction,
    action_resources,
    validate_action_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED = (
    (0, "VLC", 1, 0),
    (1, "RF-1", 0, 1),
    (2, "RF-2", 0, 2),
    (3, "RF-3", 0, 3),
    (4, "RF-4", 0, 4),
    (5, "DUP-1", 1, 1),
    (6, "DUP-2", 1, 2),
    (7, "DUP-3", 1, 3),
    (8, "DUP-4", 1, 4),
)


def test_persistent_indices_and_resources_match_contract_1_0_0() -> None:
    assert tuple(
        (
            spec.index,
            spec.name,
            spec.vlc_activations,
            spec.reserved_rf_attempts,
        )
        for spec in ACTION_RESOURCE_SPECS
    ) == EXPECTED
    assert tuple(PolicyAction) == tuple(spec.action for spec in ACTION_RESOURCE_SPECS)
    assert tuple(action.label for action in PolicyAction) == POLICY_ACTION_ORDER
    assert MAX_RESERVED_RF_ATTEMPTS == 4


def test_every_supported_key_resolves_to_the_same_authoritative_spec() -> None:
    for spec in ACTION_RESOURCE_SPECS:
        assert action_resources(spec.action) is spec
        assert action_resources(spec.index) is spec
        assert action_resources(np.int64(spec.index)) is spec
        assert action_resources(spec.name) is spec
        assert spec.as_dict() == {
            "index": spec.index,
            "name": spec.name,
            "vlc_activations": spec.vlc_activations,
            "reserved_rf_attempts": spec.reserved_rf_attempts,
        }


@pytest.mark.parametrize("invalid", [True, -1, 9, "RF", "dup-4", 1.5])
def test_invalid_action_keys_fail_closed(invalid: object) -> None:
    with pytest.raises(ActionMappingError):
        action_resources(invalid)  # type: ignore[arg-type]


def test_headline_cost_and_reward_are_derived_from_configured_coefficients() -> None:
    config = load_headline_config(PROJECT_ROOT)
    mapping = ActionResourceMap.from_config(config.environment, config.cost)

    assert config.environment.actions == POLICY_ACTION_ORDER
    assert [mapping.activation_cost(action) for action in PolicyAction] == [
        1.0,
        1.0,
        2.0,
        3.0,
        4.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    assert [mapping.reward(action) for action in PolicyAction] == [
        -1.0,
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
    ]
    assert mapping.resolve("VLC").reserved_rf_attempts == 0


def test_cost_sensitivity_changes_prices_without_changing_resources() -> None:
    config = load_headline_config(PROJECT_ROOT)
    costs = CostConfig(rf_activation=0.3, vlc_activation=2.0)
    mapping = ActionResourceMap.from_config(config.environment, costs)

    assert mapping.activation_cost("VLC") == pytest.approx(2.0)
    assert mapping.activation_cost("RF-4") == pytest.approx(1.2)
    assert mapping.activation_cost("DUP-4") == pytest.approx(3.2)
    assert mapping.resolve("DUP-4").reserved_rf_attempts == 4
    assert mapping.resolve("DUP-4").vlc_activations == 1


@pytest.mark.parametrize(
    ("version", "names", "attempts"),
    [
        ("2.0.0", POLICY_ACTION_ORDER, 4),
        (ACTION_CONTRACT_VERSION, tuple(reversed(POLICY_ACTION_ORDER)), 4),
        (ACTION_CONTRACT_VERSION, POLICY_ACTION_ORDER, 3),
    ],
)
def test_contract_drift_is_rejected(
    version: str,
    names: tuple[str, ...],
    attempts: int,
) -> None:
    with pytest.raises(ActionMappingError, match="does not match"):
        validate_action_contract(
            contract_version=version,
            action_names=names,
            max_rf_attempts=attempts,
        )


def test_medium_flags_are_derived_from_reserved_resources() -> None:
    vlc = action_resources("VLC")
    rf = action_resources("RF-3")
    duplicate = action_resources("DUP-2")

    assert vlc.uses_vlc and not vlc.uses_rf and not vlc.duplicates
    assert rf.uses_rf and not rf.uses_vlc and not rf.duplicates
    assert duplicate.uses_rf and duplicate.uses_vlc and duplicate.duplicates
