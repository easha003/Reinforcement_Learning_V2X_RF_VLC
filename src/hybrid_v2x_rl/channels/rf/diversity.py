"""Explicit receive-diversity profile and instantaneous MRC arithmetic.

The profile changes only the receiver-side small-scale link budget. It does
not create another transmission, collision opportunity, shadowing state, or
half-duplex event. Those mechanisms belong to the packet and receiving vehicle,
not to an antenna branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from hybrid_v2x_rl.core.errors import HybridV2XError


class ReceiveDiversityError(HybridV2XError):
    """A receive-diversity profile or branch realization is invalid."""


class ReceiveCombiningRule(StrEnum):
    """Implemented receiver-side combining rules."""

    NONE = "none"
    MAXIMUM_RATIO = "maximum-ratio-combining"


@dataclass(frozen=True, slots=True)
class RFReceiveDiversity:
    """One physically explicit SISO or two-branch MRC receiver profile.

    ``branch_correlation`` is the magnitude of correlation between the two
    unit-power diffuse complex fading processes. ``implementation_loss_db`` is
    applied only to the second branch before MRC, matching the frozen frontier
    equation rather than reducing the established primary SISO budget.
    """

    antenna_count: int
    combining_rule: ReceiveCombiningRule
    branch_correlation: float
    implementation_loss_db: float

    def __post_init__(self) -> None:
        if not isinstance(self.combining_rule, ReceiveCombiningRule):
            raise ReceiveDiversityError(
                "receive combining rule must be a ReceiveCombiningRule"
            )
        if (
            not isinstance(self.antenna_count, int)
            or isinstance(self.antenna_count, bool)
        ):
            raise ReceiveDiversityError("receive antenna count must be an integer")
        if (
            not isinstance(self.branch_correlation, int | float)
            or isinstance(self.branch_correlation, bool)
            or not math.isfinite(float(self.branch_correlation))
            or not 0.0 <= float(self.branch_correlation) <= 1.0
        ):
            raise ReceiveDiversityError(
                "branch correlation must be finite and lie in [0, 1]"
            )
        if (
            not isinstance(self.implementation_loss_db, int | float)
            or isinstance(self.implementation_loss_db, bool)
            or not math.isfinite(float(self.implementation_loss_db))
            or float(self.implementation_loss_db) < 0.0
        ):
            raise ReceiveDiversityError(
                "implementation loss must be finite and non-negative"
            )
        if self.combining_rule is ReceiveCombiningRule.NONE:
            if (
                self.antenna_count != 1
                or self.branch_correlation != 0.0
                or self.implementation_loss_db != 0.0
            ):
                raise ReceiveDiversityError(
                    "SISO requires one antenna, zero branch correlation, and zero loss"
                )
        elif self.antenna_count != 2:
            raise ReceiveDiversityError("MRC v1 requires exactly two receive antennas")

    @classmethod
    def siso(cls) -> RFReceiveDiversity:
        return cls(
            antenna_count=1,
            combining_rule=ReceiveCombiningRule.NONE,
            branch_correlation=0.0,
            implementation_loss_db=0.0,
        )

    @classmethod
    def two_branch_mrc(
        cls,
        *,
        branch_correlation: float,
        implementation_loss_db: float,
    ) -> RFReceiveDiversity:
        return cls(
            antenna_count=2,
            combining_rule=ReceiveCombiningRule.MAXIMUM_RATIO,
            branch_correlation=branch_correlation,
            implementation_loss_db=implementation_loss_db,
        )

    @property
    def is_siso(self) -> bool:
        return self.combining_rule is ReceiveCombiningRule.NONE

    @property
    def secondary_branch_power_factor(self) -> float:
        return math.pow(10.0, -self.implementation_loss_db / 10.0)

    def combined_fading_power(
        self,
        primary_power_gain: float,
        secondary_power_gain: float | None = None,
    ) -> float:
        """Return the effective fading multiplier seen by the decoder."""

        primary = _power_gain(primary_power_gain, name="primary branch")
        if self.is_siso:
            if secondary_power_gain is not None:
                raise ReceiveDiversityError(
                    "SISO propagation cannot carry a secondary branch gain"
                )
            return primary
        if secondary_power_gain is None:
            raise ReceiveDiversityError(
                "two-branch MRC requires a secondary branch gain"
            )
        secondary = _power_gain(secondary_power_gain, name="secondary branch")
        return primary + self.secondary_branch_power_factor * secondary

    def as_dict(self) -> dict[str, object]:
        return {
            "antenna_count": self.antenna_count,
            "combining_rule": self.combining_rule.value,
            "branch_correlation": self.branch_correlation,
            "implementation_loss_db": self.implementation_loss_db,
            "secondary_branch_power_factor": self.secondary_branch_power_factor,
        }


def _power_gain(value: object, *, name: str) -> float:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ReceiveDiversityError(f"{name} power gain must be finite and positive")
    return float(value)


SISO_RECEIVE_DIVERSITY = RFReceiveDiversity.siso()


__all__ = [
    "RFReceiveDiversity",
    "ReceiveCombiningRule",
    "ReceiveDiversityError",
    "SISO_RECEIVE_DIVERSITY",
]
