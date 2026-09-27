"""Predeclared comparison gate for the bounded constraint-recovery arms."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

from hybrid_v2x_rl.agents.regime_evaluation import (
    POLICY_INDUCED_LOAD,
    PPO_REGIME_EVALUATION_SCHEMA,
)
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.state_regime_audit import REGIME_NAMES

CONSTRAINT_RECOVERY_ANALYSIS_SCHEMA: Final = "hybrid-rf-vlc-rl.constraint-recovery-analysis.v1"
RECOVERY_ARMS: Final = (
    "control",
    "dual_lr_5",
    "dual_init_10",
    "entropy_005",
)
_MIN_WEIGHTED_FEASIBLE_GAIN: Final = 0.10
_MIN_REGIME_FEASIBLE_GAIN: Final = 0.05
_MIN_IMPROVED_REGIMES: Final = 3
_MAX_REGIME_FEASIBLE_LOSS: Final = 0.02
_MAX_EXPECTED_RISK_RATIO: Final = 1.05


class ConstraintRecoveryAnalysisError(HybridV2XError):
    """Recovery evidence is absent, malformed, or incompatible with the gate."""


def _finite_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConstraintRecoveryAnalysisError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ConstraintRecoveryAnalysisError(f"{name} must lie in [0, 1]")
    return result


def _load_evaluation(path: Path) -> tuple[dict[str, object], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ConstraintRecoveryAnalysisError(
            "cannot load recovery evaluation", artifact_path=path
        ) from error
    if not isinstance(payload, dict):
        raise ConstraintRecoveryAnalysisError("recovery evaluation must be a mapping")
    if payload.get("schema") != PPO_REGIME_EVALUATION_SCHEMA:
        raise ConstraintRecoveryAnalysisError("recovery evaluation schema is not frozen")
    if payload.get("test_split_opened") is not False:
        raise ConstraintRecoveryAnalysisError("recovery evaluation opened the test split")
    return payload, hashlib.sha256(raw).hexdigest()


def _arm_summary(payload: Mapping[str, object]) -> dict[str, object]:
    raw_rows = payload.get("campaign_regimes")
    if not isinstance(raw_rows, list):
        raise ConstraintRecoveryAnalysisError("campaign regime rows are absent")
    rows_by_name: dict[str, dict[str, object]] = {}
    weighted_feasible = 0.0
    weighted_risk = 0.0
    total_rows = 0
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ConstraintRecoveryAnalysisError("campaign regime row must be a mapping")
        regime = raw_row.get("regime")
        rows = raw_row.get("rows")
        if regime not in REGIME_NAMES or regime in rows_by_name:
            raise ConstraintRecoveryAnalysisError("campaign regimes are unknown or duplicated")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ConstraintRecoveryAnalysisError("campaign regime row count must be positive")
        profiles = raw_row.get("counterfactuals")
        if not isinstance(profiles, Mapping):
            raise ConstraintRecoveryAnalysisError("counterfactual profiles are absent")
        induced = profiles.get(POLICY_INDUCED_LOAD)
        if not isinstance(induced, Mapping):
            raise ConstraintRecoveryAnalysisError("policy-induced profile is absent")
        feasible = _finite_probability(
            "feasible action probability mass",
            induced.get("mean_feasible_action_probability_mass"),
        )
        risk = _finite_probability(
            "policy expected conditional miss risk",
            induced.get("mean_policy_expected_conditional_miss_risk"),
        )
        rows_by_name[cast(str, regime)] = {
            "rows": rows,
            "mean_feasible_action_probability_mass": feasible,
            "mean_policy_expected_conditional_miss_risk": risk,
        }
        total_rows += rows
        weighted_feasible += rows * feasible
        weighted_risk += rows * risk
    if set(rows_by_name) != set(REGIME_NAMES):
        raise ConstraintRecoveryAnalysisError("campaign regime set is incomplete")
    return {
        "regime_label_rows": total_rows,
        "weighted_mean_feasible_action_probability_mass": (weighted_feasible / total_rows),
        "weighted_mean_policy_expected_conditional_miss_risk": (weighted_risk / total_rows),
        "regimes": rows_by_name,
    }


@dataclass(frozen=True, slots=True)
class ConstraintRecoveryAnalysisReport:
    arms: Mapping[str, Mapping[str, object]]
    comparisons: Mapping[str, Mapping[str, object]]
    selected_arm: str | None
    generated_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CONSTRAINT_RECOVERY_ANALYSIS_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "scope": "bounded validation-only comparison; test split unopened",
            "test_split_opened": False,
            "gate": {
                "minimum_weighted_feasible_mass_gain": _MIN_WEIGHTED_FEASIBLE_GAIN,
                "minimum_per_regime_feasible_mass_gain": _MIN_REGIME_FEASIBLE_GAIN,
                "minimum_improved_regimes": _MIN_IMPROVED_REGIMES,
                "maximum_per_regime_feasible_mass_loss": _MAX_REGIME_FEASIBLE_LOSS,
                "maximum_weighted_expected_risk_ratio": _MAX_EXPECTED_RISK_RATIO,
                "selection": ("largest weighted feasible-mass gain, then lower weighted risk"),
            },
            "arms": {name: dict(value) for name, value in self.arms.items()},
            "comparisons_to_control": {
                name: dict(value) for name, value in self.comparisons.items()
            },
            "selected_arm": self.selected_arm,
            "full_seed_1001_authorized": self.selected_arm is not None,
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target


def build_constraint_recovery_analysis(
    evaluation_paths: Mapping[str, str | Path],
) -> ConstraintRecoveryAnalysisReport:
    """Apply the frozen recovery gate to exactly four evaluation artifacts."""

    if set(evaluation_paths) != set(RECOVERY_ARMS):
        raise ConstraintRecoveryAnalysisError(
            "recovery analysis requires exactly the four predeclared arms"
        )
    summaries: dict[str, dict[str, object]] = {}
    for arm in RECOVERY_ARMS:
        path = Path(evaluation_paths[arm]).expanduser().resolve(strict=True)
        payload, sha256 = _load_evaluation(path)
        summaries[arm] = {
            "evaluation_path": str(path),
            "evaluation_sha256": sha256,
            "config_hash": payload.get("config_hash"),
            **_arm_summary(payload),
        }

    control = summaries["control"]
    control_feasible = cast(float, control["weighted_mean_feasible_action_probability_mass"])
    control_risk = cast(float, control["weighted_mean_policy_expected_conditional_miss_risk"])
    control_regimes = cast(Mapping[str, Mapping[str, object]], control["regimes"])
    comparisons: dict[str, dict[str, object]] = {}
    passing: list[tuple[str, float, float]] = []
    for arm in RECOVERY_ARMS[1:]:
        summary = summaries[arm]
        feasible = cast(float, summary["weighted_mean_feasible_action_probability_mass"])
        risk = cast(float, summary["weighted_mean_policy_expected_conditional_miss_risk"])
        regimes = cast(Mapping[str, Mapping[str, object]], summary["regimes"])
        gains = {
            regime: cast(float, regimes[regime]["mean_feasible_action_probability_mass"])
            - cast(
                float,
                control_regimes[regime]["mean_feasible_action_probability_mass"],
            )
            for regime in REGIME_NAMES
        }
        weighted_gain = feasible - control_feasible
        improved = sum(gain >= _MIN_REGIME_FEASIBLE_GAIN for gain in gains.values())
        risk_ratio = risk / control_risk if control_risk > 0.0 else (1.0 if risk == 0.0 else None)
        passes = bool(
            weighted_gain >= _MIN_WEIGHTED_FEASIBLE_GAIN
            and improved >= _MIN_IMPROVED_REGIMES
            and min(gains.values()) >= -_MAX_REGIME_FEASIBLE_LOSS
            and risk_ratio is not None
            and risk_ratio <= _MAX_EXPECTED_RISK_RATIO
        )
        comparisons[arm] = {
            "weighted_feasible_mass_gain": weighted_gain,
            "per_regime_feasible_mass_gains": gains,
            "regimes_improved_by_at_least_0_05": improved,
            "weighted_expected_risk_ratio": risk_ratio,
            "passes_primary_gate": passes,
        }
        if passes:
            passing.append((arm, weighted_gain, risk))
    selected = sorted(passing, key=lambda row: (-row[1], row[2], row[0]))[0][0] if passing else None
    return ConstraintRecoveryAnalysisReport(
        arms=summaries,
        comparisons=comparisons,
        selected_arm=selected,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "CONSTRAINT_RECOVERY_ANALYSIS_SCHEMA",
    "RECOVERY_ARMS",
    "ConstraintRecoveryAnalysisError",
    "ConstraintRecoveryAnalysisReport",
    "build_constraint_recovery_analysis",
]
