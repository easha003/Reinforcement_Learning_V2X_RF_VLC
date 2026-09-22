"""Executable Phase 6 ordering checks for fixed baselines and limiting cases.

The checks in this module are deliberately limited to relations that the
model guarantees.  They do not impose a total order on adaptive policies:
geometry, contextual, supervised, and truth-risk policies can exchange cost
and reliability as channel state and action-coupled RF load change.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Final, Literal

from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.baselines import (
    BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_DUPLICATE_ALL,
)
from hybrid_v2x_rl.mean_field.matched_campaign import MatchedPolicyCampaignReport

BASELINE_ORDERING_SCHEMA: Final = "hybrid-rf-vlc-rl.baseline-ordering.v1"
REQUIRED_FIXED_POLICIES: Final = (
    BASELINE_ALWAYS_VLC,
    *BASELINE_ALWAYS_RF,
    BASELINE_DUPLICATE_ALL,
)
_FIXED_ACTIONS: Final = {
    BASELINE_ALWAYS_VLC: PolicyAction.VLC,
    **{
        name: PolicyAction(attempts)
        for attempts, name in enumerate(BASELINE_ALWAYS_RF, start=1)
    },
    BASELINE_DUPLICATE_ALL: PolicyAction.DUP_4,
}
_RELATIVE_TOLERANCE: Final = 1e-12


class BaselineOrderingError(HybridV2XError):
    """An ordering audit cannot be constructed from the supplied campaign."""


@dataclass(frozen=True, slots=True)
class BaselineOrderingCheck:
    """One numeric relation with both operands retained as evidence."""

    name: str
    scenario: str
    relation: Literal["==", "<=", ">="]
    left: float
    right: float
    passed: bool

    def __post_init__(self) -> None:
        if not self.name or not self.scenario:
            raise BaselineOrderingError("ordering check labels must be non-empty")
        if self.relation not in ("==", "<=", ">="):
            raise BaselineOrderingError("ordering check relation is unsupported")
        if not math.isfinite(self.left) or not math.isfinite(self.right):
            raise BaselineOrderingError("ordering check operands must be finite")
        if type(self.passed) is not bool:
            raise BaselineOrderingError("ordering check verdict must be boolean")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scenario": self.scenario,
            "relation": self.relation,
            "left": self.left,
            "right": self.right,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class BaselineOrderingReport:
    """Auditable result of all guaranteed Phase 6 ordering relations."""

    config_hash: str
    environment_seed: int
    requested_max_frames: int | None
    trace_count: int
    usable_transitions: int
    checks: tuple[BaselineOrderingCheck, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        if len(self.config_hash) != 64:
            raise BaselineOrderingError("ordering config_hash must be SHA-256")
        if self.trace_count <= 0:
            raise BaselineOrderingError("ordering report requires campaign traces")
        if self.usable_transitions <= 0:
            raise BaselineOrderingError(
                "ordering report requires at least one usable policy transition"
            )
        if not self.checks:
            raise BaselineOrderingError("ordering report requires checks")
        if self.generated_at_utc.tzinfo is None:
            raise BaselineOrderingError("ordering timestamp must be timezone-aware")

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[BaselineOrderingCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": BASELINE_ORDERING_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "passed": self.passed,
            "config_hash": self.config_hash,
            "environment_seed": self.environment_seed,
            "requested_max_frames": self.requested_max_frames,
            "trace_count": self.trace_count,
            "usable_transitions": self.usable_transitions,
            "check_count": len(self.checks),
            "failed_check_count": len(self.failed_checks),
            "checks": [check.as_dict() for check in self.checks],
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the ordering evidence."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target


def _check(
    name: str,
    scenario: str,
    relation: Literal["==", "<=", ">="],
    left: int | float,
    right: int | float,
) -> BaselineOrderingCheck:
    left_value = float(left)
    right_value = float(right)
    margin = _RELATIVE_TOLERANCE * max(1.0, abs(left_value), abs(right_value))
    if relation == "==":
        passed = abs(left_value - right_value) <= margin
    elif relation == "<=":
        passed = left_value <= right_value + margin
    else:
        passed = left_value + margin >= right_value
    return BaselineOrderingCheck(
        name=name,
        scenario=scenario,
        relation=relation,
        left=left_value,
        right=right_value,
        passed=passed,
    )


def _fixed_load_risk(
    action: PolicyAction,
    *,
    rf_attempt_risk: float,
    vlc_risk: float,
) -> float:
    """Conditional action risk when shared RF load is held fixed."""

    spec = action_resources(action)
    risk = 1.0
    if spec.uses_rf:
        risk *= rf_attempt_risk**spec.reserved_rf_attempts
    if spec.uses_vlc:
        risk *= vlc_risk
    return risk


def _limiting_case_checks() -> tuple[BaselineOrderingCheck, ...]:
    checks: list[BaselineOrderingCheck] = []
    interior_rf = 0.2
    interior_vlc = 0.3
    rf_risks = tuple(
        _fixed_load_risk(
            PolicyAction(attempts),
            rf_attempt_risk=interior_rf,
            vlc_risk=interior_vlc,
        )
        for attempts in range(1, 5)
    )
    for attempts in range(1, 4):
        checks.append(
            _check(
                f"rf-retry-{attempts + 1}-no-worse-than-{attempts}",
                "fixed-load-interior-risk",
                "<=",
                rf_risks[attempts],
                rf_risks[attempts - 1],
            )
        )
    duplicate = _fixed_load_risk(
        PolicyAction.DUP_4,
        rf_attempt_risk=interior_rf,
        vlc_risk=interior_vlc,
    )
    checks.extend(
        (
            _check(
                "duplicate-no-worse-than-rf-leg",
                "fixed-load-interior-risk",
                "<=",
                duplicate,
                rf_risks[-1],
            ),
            _check(
                "duplicate-no-worse-than-vlc-leg",
                "fixed-load-interior-risk",
                "<=",
                duplicate,
                interior_vlc,
            ),
            _check(
                "perfect-vlc-makes-vlc-risk-zero",
                "perfect-vlc",
                "==",
                _fixed_load_risk(
                    PolicyAction.VLC,
                    rf_attempt_risk=interior_rf,
                    vlc_risk=0.0,
                ),
                0.0,
            ),
            _check(
                "perfect-vlc-makes-duplicate-risk-zero",
                "perfect-vlc",
                "==",
                _fixed_load_risk(
                    PolicyAction.DUP_4,
                    rf_attempt_risk=interior_rf,
                    vlc_risk=0.0,
                ),
                0.0,
            ),
            _check(
                "failed-vlc-reduces-duplicate-to-rf",
                "certain-vlc-failure",
                "==",
                _fixed_load_risk(
                    PolicyAction.DUP_4,
                    rf_attempt_risk=interior_rf,
                    vlc_risk=1.0,
                ),
                _fixed_load_risk(
                    PolicyAction.RF_4,
                    rf_attempt_risk=interior_rf,
                    vlc_risk=1.0,
                ),
            ),
            _check(
                "perfect-rf-makes-rf-risk-zero",
                "perfect-rf-attempt",
                "==",
                _fixed_load_risk(
                    PolicyAction.RF_1,
                    rf_attempt_risk=0.0,
                    vlc_risk=interior_vlc,
                ),
                0.0,
            ),
            _check(
                "failed-rf-reduces-duplicate-to-vlc",
                "certain-rf-attempt-failure",
                "==",
                _fixed_load_risk(
                    PolicyAction.DUP_4,
                    rf_attempt_risk=1.0,
                    vlc_risk=interior_vlc,
                ),
                interior_vlc,
            ),
        )
    )
    return tuple(checks)


def verify_baseline_ordering(
    config: ProjectConfig,
    campaign: MatchedPolicyCampaignReport,
) -> BaselineOrderingReport:
    """Check guaranteed fixed-policy relations on one matched campaign."""

    if not isinstance(config, ProjectConfig):
        raise BaselineOrderingError("ordering audit requires a ProjectConfig")
    if not isinstance(campaign, MatchedPolicyCampaignReport):
        raise BaselineOrderingError("ordering audit requires a matched campaign")
    resolved_hash = config_hash(config)
    if campaign.config_hash != resolved_hash:
        raise BaselineOrderingError("campaign and ordering config hashes differ")
    missing = tuple(name for name in REQUIRED_FIXED_POLICIES if name not in campaign.policies)
    if missing:
        raise BaselineOrderingError(
            "ordering campaign is missing fixed baselines",
            context={"missing": missing},
        )

    resource_map = ActionResourceMap.from_config(config.environment, config.cost)
    fallback = action_resources(config.environment.no_observation_fallback_action).action
    fallback_spec = action_resources(fallback)
    fallback_cost = resource_map.activation_cost(fallback)
    checks: list[BaselineOrderingCheck] = list(_limiting_case_checks())
    usable_total = 0

    for comparison in campaign.comparisons:
        reports = {report.policy: report for report in comparison.reports}
        reference = reports[REQUIRED_FIXED_POLICIES[0]]
        usable_total += reference.usable_transitions
        scenario = f"matched-trace:{comparison.source.trace_id}"

        for name, action in _FIXED_ACTIONS.items():
            report = reports[name]
            spec = action_resources(action)
            expected_rf = (
                report.usable_transitions * spec.reserved_rf_attempts
                + report.fallback_transitions * fallback_spec.reserved_rf_attempts
            )
            expected_vlc = (
                report.usable_transitions * spec.vlc_activations
                + report.fallback_transitions * fallback_spec.vlc_activations
            )
            expected_reward = -(
                report.usable_transitions * resource_map.activation_cost(action)
                + report.fallback_transitions * fallback_cost
            )
            checks.extend(
                (
                    _check(
                        f"{name}:reserved-rf-attempts",
                        scenario,
                        "==",
                        report.reserved_rf_attempts,
                        expected_rf,
                    ),
                    _check(
                        f"{name}:vlc-activations",
                        scenario,
                        "==",
                        report.vlc_activations,
                        expected_vlc,
                    ),
                    _check(
                        f"{name}:resource-reward",
                        scenario,
                        "==",
                        report.reward_sum,
                        expected_reward,
                    ),
                )
            )

        for left_name, right_name in combinations(REQUIRED_FIXED_POLICIES, 2):
            left_cost = resource_map.activation_cost(_FIXED_ACTIONS[left_name])
            right_cost = resource_map.activation_cost(_FIXED_ACTIONS[right_name])
            left_reward = reports[left_name].reward_sum
            right_reward = reports[right_name].reward_sum
            if math.isclose(left_cost, right_cost, rel_tol=0.0, abs_tol=1e-15):
                checks.append(
                    _check(
                        f"equal-cost:{left_name}:{right_name}",
                        scenario,
                        "==",
                        left_reward,
                        right_reward,
                    )
                )
            elif left_cost < right_cost:
                checks.append(
                    _check(
                        f"lower-cost:{left_name}:{right_name}",
                        scenario,
                        ">=",
                        left_reward,
                        right_reward,
                    )
                )
            else:
                checks.append(
                    _check(
                        f"lower-cost:{right_name}:{left_name}",
                        scenario,
                        ">=",
                        right_reward,
                        left_reward,
                    )
                )

        duplicate = reports[BASELINE_DUPLICATE_ALL]
        for component in (BASELINE_ALWAYS_VLC, BASELINE_ALWAYS_RF[-1]):
            component_report = reports[component]
            checks.extend(
                (
                    _check(
                        f"duplicate-conditional-risk-no-worse-than:{component}",
                        scenario,
                        "<=",
                        duplicate.conditional_risk_sum,
                        component_report.conditional_risk_sum,
                    ),
                    _check(
                        f"duplicate-sampled-misses-no-worse-than:{component}",
                        scenario,
                        "<=",
                        duplicate.misses,
                        component_report.misses,
                    ),
                )
            )

    return BaselineOrderingReport(
        config_hash=resolved_hash,
        environment_seed=campaign.environment_seed,
        requested_max_frames=campaign.requested_max_frames,
        trace_count=len(campaign.comparisons),
        usable_transitions=usable_total,
        checks=tuple(checks),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "BASELINE_ORDERING_SCHEMA",
    "REQUIRED_FIXED_POLICIES",
    "BaselineOrderingCheck",
    "BaselineOrderingError",
    "BaselineOrderingReport",
    "verify_baseline_ordering",
]
