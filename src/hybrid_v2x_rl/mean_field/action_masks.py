"""Causal action masks for the population-policy action space.

Masks encode only hardware/profile feasibility: whether RF and VLC exist and
how many RF attempts can be reserved.  They deliberately accept no channel
state, predicted blockage, occlusion truth, failure probability, or sampled
outcome, because using any of those to suppress an action would leak the answer
the policy is meant to learn under uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_v2x_rl.config.models import EnvironmentConfig, RFConfig, VLCConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    MAX_RESERVED_RF_ATTEMPTS,
    ActionKey,
    PolicyAction,
    action_resources,
    validate_action_contract,
)


class ActionMaskError(HybridV2XError):
    """A hardware profile or selected action violates its action mask."""


@dataclass(frozen=True, slots=True)
class ActionMask:
    """One boolean per persistent action index; ``True`` means selectable."""

    values: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(PolicyAction):
            raise ActionMaskError(
                "action mask has the wrong width",
                context={"actual": len(self.values), "expected": len(PolicyAction)},
            )
        if any(type(value) is not bool for value in self.values):
            raise ActionMaskError("action mask entries must be booleans")
        if not any(self.values):
            raise ActionMaskError("hardware profile leaves no selectable policy action")

    @classmethod
    def from_availability(
        cls,
        *,
        rf_hardware_available: bool,
        vlc_hardware_available: bool,
        max_reserved_rf_attempts: int,
    ) -> ActionMask:
        """Build a mask without consulting transient link or channel state."""

        for name, value in (
            ("rf_hardware_available", rf_hardware_available),
            ("vlc_hardware_available", vlc_hardware_available),
        ):
            if type(value) is not bool:
                raise ActionMaskError(f"{name} must be boolean")
        if (
            not isinstance(max_reserved_rf_attempts, int)
            or isinstance(max_reserved_rf_attempts, bool)
            or not 0 <= max_reserved_rf_attempts <= MAX_RESERVED_RF_ATTEMPTS
        ):
            raise ActionMaskError(
                "maximum reservable RF attempts lies outside the action contract",
                context={
                    "actual": max_reserved_rf_attempts,
                    "minimum": 0,
                    "maximum": MAX_RESERVED_RF_ATTEMPTS,
                },
            )
        values = tuple(
            (not spec.uses_rf or rf_hardware_available)
            and (not spec.uses_vlc or vlc_hardware_available)
            and spec.reserved_rf_attempts <= max_reserved_rf_attempts
            for spec in (action_resources(action) for action in PolicyAction)
        )
        return cls(values=values)

    def allows(self, action: ActionKey) -> bool:
        """Whether the action may be selected under this hardware profile."""

        spec = action_resources(action)
        return self.values[spec.index]

    def require_allowed(self, action: ActionKey) -> PolicyAction:
        """Resolve an action and reject it if its mask entry is false."""

        spec = action_resources(action)
        if not self.values[spec.index]:
            raise ActionMaskError(
                "selected policy action is masked by the hardware profile",
                context={"index": spec.index, "action": spec.name},
            )
        return spec.action

    @property
    def allowed_actions(self) -> tuple[PolicyAction, ...]:
        return tuple(action for action in PolicyAction if self.values[int(action)])

    @property
    def allowed_names(self) -> tuple[str, ...]:
        return tuple(action.label for action in self.allowed_actions)


@dataclass(frozen=True, slots=True)
class MaskedActionSpace:
    """A verified hardware mask plus the missing-observation fallback."""

    mask: ActionMask
    fallback_action: PolicyAction

    def __post_init__(self) -> None:
        if not self.mask.allows(self.fallback_action):
            spec = action_resources(self.fallback_action)
            raise ActionMaskError(
                "configured missing-observation fallback is masked",
                context={"index": spec.index, "action": spec.name},
            )

    @classmethod
    def from_config(
        cls,
        environment: EnvironmentConfig,
        rf: RFConfig,
        vlc: VLCConfig,
    ) -> MaskedActionSpace:
        """Bind masks and fallback to the frozen loaded environment profile."""

        validate_action_contract(
            contract_version=environment.contract_version,
            action_names=environment.actions,
            max_rf_attempts=environment.max_rf_attempts,
        )
        mask = ActionMask.from_availability(
            rf_hardware_available=rf.enabled,
            vlc_hardware_available=vlc.enabled,
            max_reserved_rf_attempts=(
                environment.max_rf_attempts if rf.enabled else 0
            ),
        )
        fallback = action_resources(environment.no_observation_fallback_action).action
        return cls(mask=mask, fallback_action=fallback)

    def select(
        self,
        proposed_action: ActionKey | None,
        *,
        observation_usable: bool,
    ) -> PolicyAction:
        """Validate a proposal or apply fallback only for a missing observation."""

        if type(observation_usable) is not bool:
            raise ActionMaskError("observation_usable must be boolean")
        if not observation_usable:
            return self.fallback_action
        if proposed_action is None:
            raise ActionMaskError(
                "a usable observation requires an explicit policy action"
            )
        return self.mask.require_allowed(proposed_action)


__all__ = [
    "ActionMask",
    "ActionMaskError",
    "MaskedActionSpace",
]
