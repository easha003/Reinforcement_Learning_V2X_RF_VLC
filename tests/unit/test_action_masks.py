"""Phase 3 action-mask and missing-observation fallback rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER, PolicyAction
from hybrid_v2x_rl.mean_field.action_masks import (
    ActionMask,
    ActionMaskError,
    MaskedActionSpace,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_headline_hardware_profile_exposes_all_nine_actions() -> None:
    config = load_headline_config(PROJECT_ROOT)
    action_space = MaskedActionSpace.from_config(
        config.environment,
        config.rf,
        config.vlc,
    )

    assert action_space.mask.values == (True,) * 9
    assert action_space.mask.allowed_actions == tuple(PolicyAction)
    assert action_space.mask.allowed_names == POLICY_ACTION_ORDER
    assert action_space.fallback_action is PolicyAction.DUP_4


def test_absent_rf_hardware_leaves_only_vlc() -> None:
    mask = ActionMask.from_availability(
        rf_hardware_available=False,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=0,
    )

    assert mask.allowed_actions == (PolicyAction.VLC,)
    assert mask.values == (True, False, False, False, False, False, False, False, False)


def test_absent_vlc_hardware_leaves_rf_reservations() -> None:
    mask = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=False,
        max_reserved_rf_attempts=4,
    )

    assert mask.allowed_names == ("RF-1", "RF-2", "RF-3", "RF-4")
    assert mask.values == (False, True, True, True, True, False, False, False, False)


def test_reservation_limit_masks_larger_rf_and_dup_actions() -> None:
    mask = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=2,
    )

    assert mask.allowed_names == ("VLC", "RF-1", "RF-2", "DUP-1", "DUP-2")
    assert mask.values == (True, True, True, False, False, True, True, False, False)


def test_zero_rf_reservations_with_both_media_available_leaves_vlc() -> None:
    mask = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=0,
    )

    assert mask.allowed_actions == (PolicyAction.VLC,)


def test_profile_with_no_communication_medium_is_rejected() -> None:
    with pytest.raises(ActionMaskError, match="no selectable"):
        ActionMask.from_availability(
            rf_hardware_available=False,
            vlc_hardware_available=False,
            max_reserved_rf_attempts=0,
        )


@pytest.mark.parametrize("attempts", [-1, 5, True, 1.5])
def test_invalid_reservation_limits_are_rejected(attempts: object) -> None:
    with pytest.raises(ActionMaskError, match="outside"):
        ActionMask.from_availability(
            rf_hardware_available=True,
            vlc_hardware_available=True,
            max_reserved_rf_attempts=attempts,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "values",
    [
        (True,) * 8,
        (False,) * 9,
        (True, True, True, True, True, True, True, True, 1),
    ],
)
def test_malformed_masks_are_rejected(values: tuple[object, ...]) -> None:
    with pytest.raises(ActionMaskError):
        ActionMask(values=values)  # type: ignore[arg-type]


def test_masked_policy_action_is_rejected_before_accounting() -> None:
    mask = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=2,
    )

    assert mask.require_allowed("DUP-2") is PolicyAction.DUP_2
    with pytest.raises(ActionMaskError, match="masked"):
        mask.require_allowed("DUP-3")


def test_fallback_applies_only_when_the_causal_observation_is_unusable() -> None:
    config = load_headline_config(PROJECT_ROOT)
    action_space = MaskedActionSpace.from_config(
        config.environment,
        config.rf,
        config.vlc,
    )

    assert action_space.select("RF-2", observation_usable=True) is PolicyAction.RF_2
    assert action_space.select(None, observation_usable=False) is PolicyAction.DUP_4
    assert action_space.select("VLC", observation_usable=False) is PolicyAction.DUP_4
    with pytest.raises(ActionMaskError, match="explicit"):
        action_space.select(None, observation_usable=True)


@pytest.mark.parametrize("disabled_medium", ["rf", "vlc"])
def test_configuration_rejects_a_fallback_that_hardware_cannot_execute(
    disabled_medium: str,
) -> None:
    config = load_headline_config(PROJECT_ROOT)
    rf = config.rf.model_copy(update={"enabled": disabled_medium != "rf"})
    vlc = config.vlc.model_copy(update={"enabled": disabled_medium != "vlc"})

    with pytest.raises(ActionMaskError, match="fallback is masked"):
        MaskedActionSpace.from_config(config.environment, rf, vlc)


def test_mask_depends_on_profile_not_transient_channel_quality() -> None:
    """Poor quality and predicted blockage remain policy inputs, not masks."""

    first = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=4,
    )
    second = ActionMask.from_availability(
        rf_hardware_available=True,
        vlc_hardware_available=True,
        max_reserved_rf_attempts=4,
    )

    assert first == second
    assert first.values == (True,) * 9
