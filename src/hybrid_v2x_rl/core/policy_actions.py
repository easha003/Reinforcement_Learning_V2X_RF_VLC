"""Authoritative nine-action resource mapping for environment contract 1.0.0.

The inherited packet simulator has a separate three-action interface.  This
module defines the persistent action indices used by the population RL
environment and is the only source of RF-attempt and VLC-activation semantics
for those indices.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import TYPE_CHECKING, Literal, SupportsIndex, TypeAlias, cast

from hybrid_v2x_rl.core.errors import HybridV2XError

if TYPE_CHECKING:
    from hybrid_v2x_rl.config.models import CostConfig, EnvironmentConfig

ACTION_CONTRACT_VERSION = "1.0.0"

PolicyActionName: TypeAlias = Literal[
    "VLC",
    "RF-1",
    "RF-2",
    "RF-3",
    "RF-4",
    "DUP-1",
    "DUP-2",
    "DUP-3",
    "DUP-4",
]


class ActionMappingError(HybridV2XError):
    """An action value or loaded action contract does not match version 1.0.0."""


@unique
class PolicyAction(IntEnum):
    """Persistent population-policy action indices; never reorder in place."""

    VLC = 0
    RF_1 = 1
    RF_2 = 2
    RF_3 = 3
    RF_4 = 4
    DUP_1 = 5
    DUP_2 = 6
    DUP_3 = 7
    DUP_4 = 8

    @property
    def label(self) -> PolicyActionName:
        """Return the contract-facing name stored in configuration and artifacts."""

        return ACTION_RESOURCE_SPECS[int(self)].name


@dataclass(frozen=True, slots=True)
class ActionResourceSpec:
    """Policy-independent resources reserved by one action."""

    action: PolicyAction
    name: PolicyActionName
    vlc_activations: Literal[0, 1]
    reserved_rf_attempts: int

    def __post_init__(self) -> None:
        if self.vlc_activations not in (0, 1):
            raise ValueError("vlc_activations must be zero or one")
        if not 0 <= self.reserved_rf_attempts <= 4:
            raise ValueError("reserved_rf_attempts must lie in [0, 4]")
        if self.vlc_activations == 0 and self.reserved_rf_attempts == 0:
            raise ValueError("an action must reserve at least one communication resource")

    @property
    def index(self) -> int:
        return int(self.action)

    @property
    def uses_rf(self) -> bool:
        return self.reserved_rf_attempts > 0

    @property
    def uses_vlc(self) -> bool:
        return self.vlc_activations == 1

    @property
    def duplicates(self) -> bool:
        return self.uses_rf and self.uses_vlc

    def as_dict(self) -> dict[str, object]:
        """Return stable fields suitable for logs and persisted metadata."""

        return {
            "index": self.index,
            "name": self.name,
            "vlc_activations": self.vlc_activations,
            "reserved_rf_attempts": self.reserved_rf_attempts,
        }


# This tuple is the single authoritative index-to-resource table.  Every other
# view (configuration order, name lookup, costs, rewards, and later ledgers) is
# derived from it.
ACTION_RESOURCE_SPECS: tuple[ActionResourceSpec, ...] = (
    ActionResourceSpec(PolicyAction.VLC, "VLC", 1, 0),
    ActionResourceSpec(PolicyAction.RF_1, "RF-1", 0, 1),
    ActionResourceSpec(PolicyAction.RF_2, "RF-2", 0, 2),
    ActionResourceSpec(PolicyAction.RF_3, "RF-3", 0, 3),
    ActionResourceSpec(PolicyAction.RF_4, "RF-4", 0, 4),
    ActionResourceSpec(PolicyAction.DUP_1, "DUP-1", 1, 1),
    ActionResourceSpec(PolicyAction.DUP_2, "DUP-2", 1, 2),
    ActionResourceSpec(PolicyAction.DUP_3, "DUP-3", 1, 3),
    ActionResourceSpec(PolicyAction.DUP_4, "DUP-4", 1, 4),
)

if tuple(spec.action for spec in ACTION_RESOURCE_SPECS) != tuple(PolicyAction):
    raise RuntimeError("action resource table must follow contiguous PolicyAction indices")

POLICY_ACTION_ORDER: tuple[PolicyActionName, ...] = tuple(
    spec.name for spec in ACTION_RESOURCE_SPECS
)
MAX_RESERVED_RF_ATTEMPTS = max(
    spec.reserved_rf_attempts for spec in ACTION_RESOURCE_SPECS
)
_SPECS_BY_NAME = {spec.name: spec for spec in ACTION_RESOURCE_SPECS}

ActionKey: TypeAlias = PolicyAction | SupportsIndex | str


def action_resources(action: ActionKey) -> ActionResourceSpec:
    """Resolve an exact action enum, integer index, or contract name."""

    if isinstance(action, str):
        try:
            return _SPECS_BY_NAME[cast(PolicyActionName, action)]
        except KeyError as error:
            raise ActionMappingError(
                "unknown policy action name",
                context={"action": action, "known": POLICY_ACTION_ORDER},
            ) from error
    if isinstance(action, bool):
        raise ActionMappingError("boolean values are not policy action indices")
    try:
        index = operator.index(action)
        member = PolicyAction(index)
    except (TypeError, ValueError) as error:
        raise ActionMappingError(
            "policy action index lies outside the frozen action space",
            context={"action": repr(action), "valid_indices": (0, len(PolicyAction) - 1)},
        ) from error
    return ACTION_RESOURCE_SPECS[int(member)]


def validate_action_contract(
    *,
    contract_version: str,
    action_names: tuple[str, ...],
    max_rf_attempts: int,
) -> None:
    """Fail closed when loaded configuration drifts from the frozen mapping."""

    failures: dict[str, object] = {}
    if contract_version != ACTION_CONTRACT_VERSION:
        failures["contract_version"] = {
            "actual": contract_version,
            "expected": ACTION_CONTRACT_VERSION,
        }
    if action_names != POLICY_ACTION_ORDER:
        failures["action_names"] = {
            "actual": action_names,
            "expected": POLICY_ACTION_ORDER,
        }
    if max_rf_attempts != MAX_RESERVED_RF_ATTEMPTS:
        failures["max_rf_attempts"] = {
            "actual": max_rf_attempts,
            "expected": MAX_RESERVED_RF_ATTEMPTS,
        }
    if failures:
        raise ActionMappingError(
            "loaded environment does not match the frozen action-resource contract",
            context=failures,
        )


@dataclass(frozen=True, slots=True)
class ActionResourceMap:
    """Frozen action resources combined with run-specific cost coefficients."""

    contract_version: str
    rf_activation_cost: float
    vlc_activation_cost: float

    def __post_init__(self) -> None:
        for name, value in (
            ("rf_activation_cost", self.rf_activation_cost),
            ("vlc_activation_cost", self.vlc_activation_cost),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.contract_version != ACTION_CONTRACT_VERSION:
            raise ActionMappingError(
                "resource map contract version does not match its action table",
                context={
                    "actual": self.contract_version,
                    "expected": ACTION_CONTRACT_VERSION,
                },
            )

    @classmethod
    def from_config(
        cls,
        environment: EnvironmentConfig,
        costs: CostConfig,
    ) -> ActionResourceMap:
        """Bind the mapping to validated environment order and configured costs."""

        validate_action_contract(
            contract_version=environment.contract_version,
            action_names=environment.actions,
            max_rf_attempts=environment.max_rf_attempts,
        )
        return cls(
            contract_version=environment.contract_version,
            rf_activation_cost=costs.rf_activation,
            vlc_activation_cost=costs.vlc_activation,
        )

    @property
    def specs(self) -> tuple[ActionResourceSpec, ...]:
        return ACTION_RESOURCE_SPECS

    def resolve(self, action: ActionKey) -> ActionResourceSpec:
        return action_resources(action)

    def activation_cost(self, action: ActionKey) -> float:
        """Compute configured normalized resource cost for one action."""

        spec = self.resolve(action)
        return (
            self.rf_activation_cost * spec.reserved_rf_attempts
            + self.vlc_activation_cost * spec.vlc_activations
        )

    def reward(self, action: ActionKey) -> float:
        """Return the Phase 1 resource reward, the negative activation cost."""

        return -self.activation_cost(action)


__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ACTION_RESOURCE_SPECS",
    "MAX_RESERVED_RF_ATTEMPTS",
    "POLICY_ACTION_ORDER",
    "ActionKey",
    "ActionMappingError",
    "ActionResourceMap",
    "ActionResourceSpec",
    "PolicyAction",
    "PolicyActionName",
    "action_resources",
    "validate_action_contract",
]
