"""Residual reliability-floor and policy-regret diagnosis for frozen PPO evaluation."""

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
    ALL_USABLE_ROWS,
    POLICY_INDUCED_LOAD,
    PPO_REGIME_EVALUATION_SCHEMA,
)
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.state_regime_audit import LOAD_PROFILES

RESIDUAL_FEASIBILITY_SCHEMA: Final = "hybrid-rf-vlc-rl.residual-feasibility.v1"
_DECOMPOSITION_TOLERANCE: Final = 1e-10


class ResidualFeasibilityError(HybridV2XError):
    """A source evaluation or risk decomposition is invalid."""


def _finite_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResidualFeasibilityError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ResidualFeasibilityError(f"{name} must lie in [0, 1]")
    return result


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResidualFeasibilityError(f"{name} must be a positive integer")
    return value


def _profile_summary(
    raw: Mapping[str, object],
    *,
    miss_budget: float,
) -> dict[str, object]:
    rows = _positive_int("profile rows", raw.get("rows"))
    floor = _finite_probability(
        "minimum-action conditional miss risk",
        raw.get("mean_minimum_action_conditional_miss_risk"),
    )
    expected = _finite_probability(
        "policy-expected conditional miss risk",
        raw.get("mean_policy_expected_conditional_miss_risk"),
    )
    selected = _finite_probability(
        "deterministic-selected conditional miss risk",
        raw.get("mean_deterministic_selected_conditional_miss_risk"),
    )
    expected_regret = _finite_probability(
        "policy-expected action risk regret",
        raw.get("mean_policy_expected_action_risk_regret"),
    )
    selected_regret = _finite_probability(
        "deterministic-selected action risk regret",
        raw.get("mean_deterministic_selected_action_risk_regret"),
    )
    expected_residual = expected - floor - expected_regret
    selected_residual = selected - floor - selected_regret
    if abs(expected_residual) > _DECOMPOSITION_TOLERANCE:
        raise ResidualFeasibilityError("policy-expected risk decomposition does not close")
    if abs(selected_residual) > _DECOMPOSITION_TOLERANCE:
        raise ResidualFeasibilityError("deterministic-selected risk decomposition does not close")
    any_feasible = _finite_probability(
        "statewise any-feasible fraction",
        raw.get("any_feasible_fraction"),
    )
    minimum_counts = raw.get("minimum_risk_action_counts")
    if not isinstance(minimum_counts, Mapping):
        raise ResidualFeasibilityError("minimum-risk action counts are absent")
    return {
        "rows": rows,
        "mean_minimum_action_conditional_miss_risk": floor,
        "mean_policy_expected_conditional_miss_risk": expected,
        "mean_deterministic_selected_conditional_miss_risk": selected,
        "mean_policy_expected_action_risk_regret": expected_regret,
        "mean_deterministic_selected_action_risk_regret": selected_regret,
        "policy_expected_decomposition_residual": expected_residual,
        "deterministic_selected_decomposition_residual": selected_residual,
        "minimum_action_risk_budget_multiple": floor / miss_budget,
        "policy_expected_risk_budget_multiple": expected / miss_budget,
        "deterministic_selected_risk_budget_multiple": selected / miss_budget,
        "minimum_action_mean_meets_budget": floor <= miss_budget,
        "statewise_any_feasible_fraction": any_feasible,
        "selected_risk_fraction_attributable_to_action_regret": (
            selected_regret / selected if selected > 0.0 else 0.0
        ),
        "minimum_risk_action_counts": {
            str(action): int(count) for action, count in minimum_counts.items()
        },
    }


def _scope_summary(
    raw: Mapping[str, object],
    *,
    miss_budget: float,
) -> dict[str, object]:
    if raw.get("regime") != ALL_USABLE_ROWS:
        raise ResidualFeasibilityError("residual diagnosis requires all usable rows")
    profiles = raw.get("counterfactuals")
    if not isinstance(profiles, Mapping):
        raise ResidualFeasibilityError("counterfactual profiles are absent")
    expected_names = {POLICY_INDUCED_LOAD, *LOAD_PROFILES}
    if set(profiles) != expected_names:
        raise ResidualFeasibilityError("counterfactual profile set is incomplete")
    summarized: dict[str, dict[str, object]] = {}
    for name in (POLICY_INDUCED_LOAD, *LOAD_PROFILES):
        profile = profiles[name]
        if not isinstance(profile, Mapping):
            raise ResidualFeasibilityError("counterfactual profile must be a mapping")
        summarized[name] = _profile_summary(profile, miss_budget=miss_budget)

    induced_floor = cast(float, summarized[POLICY_INDUCED_LOAD][
        "mean_minimum_action_conditional_miss_risk"
    ])
    offload_floor = cast(float, summarized["vlc_offload"][
        "mean_minimum_action_conditional_miss_risk"
    ])
    if offload_floor > miss_budget:
        diagnosis = "physical_or_action_floor_exceeds_target_even_with_vlc_offload"
    elif induced_floor > miss_budget:
        diagnosis = "policy_load_or_population_coordination_floor_exceeds_target"
    else:
        diagnosis = "fixed_policy_load_floor_meets_target_action_selection_regret_remains"
    return {
        "density_vehicles_per_lane_km": raw.get("density_vehicles_per_lane_km"),
        "rows": _positive_int("scope rows", raw.get("rows")),
        "diagnosis": diagnosis,
        "counterfactuals": summarized,
    }


@dataclass(frozen=True, slots=True)
class ResidualFeasibilityReport:
    source_path: Path
    source_sha256: str
    config_hash: str
    checkpoint: Mapping[str, object]
    miss_budget: float
    campaign: Mapping[str, object]
    densities: tuple[Mapping[str, object], ...]
    next_action: str
    standard_ppo_recovery_arm_authorized: bool
    generated_at_utc: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": RESIDUAL_FEASIBILITY_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "scope": (
                "bounded validation-only decomposition under policy-induced and "
                "declared population RF-load profiles"
            ),
            "test_split_opened": False,
            "source_evaluation": {
                "path": str(self.source_path),
                "sha256": self.source_sha256,
                "schema": PPO_REGIME_EVALUATION_SCHEMA,
                "config_hash": self.config_hash,
                "checkpoint": dict(self.checkpoint),
            },
            "reliability_miss_budget": self.miss_budget,
            "interpretation_boundary": {
                "minimum_action_risk": (
                    "non-deployable per-row lower bound with other-pair RF load fixed"
                ),
                "vlc_offload": (
                    "optimistic zero-other-pair-RF-load probe; failure here proves the "
                    "current action/physical model cannot meet the aggregate target"
                ),
                "joint_equilibrium": (
                    "not claimed; a passing fixed-load probe does not prove that a stable "
                    "population action assignment exists"
                ),
            },
            "all_usable_campaign": dict(self.campaign),
            "all_usable_by_density": [dict(row) for row in self.densities],
            "decision": {
                "standard_ppo_recovery_arm_authorized": (
                    self.standard_ppo_recovery_arm_authorized
                ),
                "next_action": self.next_action,
            },
        }

    def write_json(self, path: str | Path) -> Path:
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


def build_residual_feasibility_report(
    evaluation_path: str | Path,
) -> ResidualFeasibilityReport:
    """Load one v2 frozen evaluation and diagnose its remaining reliability gap."""

    source = Path(evaluation_path).expanduser().resolve()
    try:
        raw = source.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ResidualFeasibilityError(
            "cannot load PPO regime evaluation",
            artifact_path=source,
        ) from error
    if not isinstance(payload, Mapping):
        raise ResidualFeasibilityError("PPO regime evaluation must be a mapping")
    if payload.get("schema") != PPO_REGIME_EVALUATION_SCHEMA:
        raise ResidualFeasibilityError("residual diagnosis requires PPO evaluation v2")
    if payload.get("test_split_opened") is not False:
        raise ResidualFeasibilityError("source evaluation opened the test split")
    miss_budget = _finite_probability(
        "reliability miss budget",
        payload.get("reliability_miss_budget"),
    )
    if miss_budget <= 0.0:
        raise ResidualFeasibilityError("reliability miss budget must be positive")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ResidualFeasibilityError("checkpoint provenance is absent")
    config_digest = payload.get("config_hash")
    if not isinstance(config_digest, str) or len(config_digest) != 64:
        raise ResidualFeasibilityError("configuration hash is invalid")
    campaign_raw = payload.get("all_usable_campaign")
    densities_raw = payload.get("all_usable_by_density")
    if not isinstance(campaign_raw, Mapping) or not isinstance(densities_raw, list):
        raise ResidualFeasibilityError("all-usable summaries are absent")
    campaign = _scope_summary(campaign_raw, miss_budget=miss_budget)
    densities = tuple(
        _scope_summary(row, miss_budget=miss_budget)
        for row in densities_raw
        if isinstance(row, Mapping)
    )
    if len(densities) != len(densities_raw) or not densities:
        raise ResidualFeasibilityError("density summaries are empty or invalid")

    diagnoses = {cast(str, row["diagnosis"]) for row in densities}
    if "physical_or_action_floor_exceeds_target_even_with_vlc_offload" in diagnoses:
        authorized = False
        next_action = (
            "revise the physical/action model for failing densities before another "
            "standard PPO recovery arm"
        )
    elif "policy_load_or_population_coordination_floor_exceeds_target" in diagnoses:
        authorized = False
        next_action = (
            "test a population-coupled coordination or load-shaping mechanism before "
            "another standard action-pressure arm"
        )
    else:
        authorized = True
        next_action = (
            "predeclare a bounded optimizer recovery arm targeting the measured action regret"
        )
    return ResidualFeasibilityReport(
        source_path=source,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        config_hash=config_digest,
        checkpoint=checkpoint,
        miss_budget=miss_budget,
        campaign=campaign,
        densities=densities,
        next_action=next_action,
        standard_ppo_recovery_arm_authorized=authorized,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "RESIDUAL_FEASIBILITY_SCHEMA",
    "ResidualFeasibilityError",
    "ResidualFeasibilityReport",
    "build_residual_feasibility_report",
]
