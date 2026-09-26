"""Read-only pre-update diagnostics for constrained PPO pressure and policy mix."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final

import torch

from hybrid_v2x_rl.agents.masked_categorical import (
    ACTION_COUNT,
    MaskedCategorical,
    SharedCategoricalActor,
)
from hybrid_v2x_rl.agents.ppo import PPOBatch
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction

CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA: Final = (
    "hybrid-rf-vlc-rl.constraint-pressure-diagnostics.v1"
)
_LOG_PROBABILITY_TOLERANCE: Final = 1e-6


class ConstraintDiagnosticsError(HybridV2XError):
    """A pre-update diagnostic input or derived statistic is invalid."""


@dataclass(frozen=True, slots=True)
class ScalarDistributionSummary:
    """Finite signed and magnitude statistics for one aligned vector."""

    sample_count: int
    mean: float
    standard_deviation: float
    mean_absolute: float
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        _positive_integer("sample_count", self.sample_count)
        for name in (
            "mean",
            "standard_deviation",
            "mean_absolute",
            "minimum",
            "maximum",
        ):
            _finite_real(name, getattr(self, name))
        if self.standard_deviation < 0.0 or self.mean_absolute < 0.0:
            raise ConstraintDiagnosticsError(
                "distribution dispersion and absolute magnitude must be nonnegative"
            )
        if self.minimum > self.mean or self.mean > self.maximum:
            raise ConstraintDiagnosticsError(
                "distribution mean must lie within its observed extrema"
            )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "mean_absolute": self.mean_absolute,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class DensityConstraintPressureDiagnostics:
    """Pre-update advantage scales and policy behavior for one density."""

    density_veh_per_lane_km: float
    learning_rows: int
    dual_multiplier_before_update: float
    reward_advantage: ScalarDistributionSummary
    cost_advantage: ScalarDistributionSummary
    dual_weighted_cost_advantage: ScalarDistributionSummary
    combined_actor_advantage: ScalarDistributionSummary
    entropy: ScalarDistributionSummary
    mean_absolute_constraint_to_reward_ratio: float | None
    positive_combined_advantage_fraction: float
    mean_action_probabilities: tuple[float, ...]
    action_availability_fractions: tuple[float, ...]
    selected_action_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        density = _finite_real(
            "density_veh_per_lane_km",
            self.density_veh_per_lane_km,
        )
        if density <= 0.0:
            raise ConstraintDiagnosticsError("diagnostic density must be positive")
        _positive_integer("learning_rows", self.learning_rows)
        multiplier = _finite_real(
            "dual_multiplier_before_update",
            self.dual_multiplier_before_update,
        )
        if multiplier < 0.0:
            raise ConstraintDiagnosticsError("diagnostic dual multiplier cannot be negative")
        for name in (
            "reward_advantage",
            "cost_advantage",
            "dual_weighted_cost_advantage",
            "combined_actor_advantage",
            "entropy",
        ):
            summary = getattr(self, name)
            if not isinstance(summary, ScalarDistributionSummary):
                raise ConstraintDiagnosticsError(f"{name} must be a distribution summary")
            if summary.sample_count != self.learning_rows:
                raise ConstraintDiagnosticsError(
                    f"{name} does not cover every density learning row"
                )
        if self.entropy.minimum < -1e-12:
            raise ConstraintDiagnosticsError("categorical entropy cannot be negative")
        ratio = self.mean_absolute_constraint_to_reward_ratio
        if (
            ratio is not None
            and _finite_real(
                "mean_absolute_constraint_to_reward_ratio",
                ratio,
            )
            < 0.0
        ):
            raise ConstraintDiagnosticsError("constraint-to-reward ratio cannot be negative")
        positive = _finite_real(
            "positive_combined_advantage_fraction",
            self.positive_combined_advantage_fraction,
        )
        if not 0.0 <= positive <= 1.0:
            raise ConstraintDiagnosticsError(
                "positive combined-advantage fraction must lie in [0, 1]"
            )
        if (
            len(self.mean_action_probabilities) != ACTION_COUNT
            or len(self.action_availability_fractions) != ACTION_COUNT
            or len(self.selected_action_counts) != ACTION_COUNT
        ):
            raise ConstraintDiagnosticsError(
                "action diagnostics must follow the complete canonical action space"
            )
        probabilities = tuple(
            _unit_interval("mean action probability", value)
            for value in self.mean_action_probabilities
        )
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ConstraintDiagnosticsError("mean action probabilities must sum to one")
        availability = tuple(
            _unit_interval("action availability fraction", value)
            for value in self.action_availability_fractions
        )
        for index, count in enumerate(self.selected_action_counts):
            _nonnegative_integer("selected action count", count)
            if availability[index] == 0.0 and (count != 0 or probabilities[index] != 0.0):
                raise ConstraintDiagnosticsError(
                    "an unavailable action cannot be selected or receive probability"
                )
        if sum(self.selected_action_counts) != self.learning_rows:
            raise ConstraintDiagnosticsError(
                "selected action counts must partition density learning rows"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "density_veh_per_lane_km": self.density_veh_per_lane_km,
            "learning_rows": self.learning_rows,
            "dual_multiplier_before_update": self.dual_multiplier_before_update,
            "reward_advantage": self.reward_advantage.as_dict(),
            "cost_advantage": self.cost_advantage.as_dict(),
            "dual_weighted_cost_advantage": (self.dual_weighted_cost_advantage.as_dict()),
            "combined_actor_advantage": self.combined_actor_advantage.as_dict(),
            "entropy": self.entropy.as_dict(),
            "mean_absolute_constraint_to_reward_ratio": (
                self.mean_absolute_constraint_to_reward_ratio
            ),
            "positive_combined_advantage_fraction": (self.positive_combined_advantage_fraction),
            "mean_action_probabilities": {
                action.label: self.mean_action_probabilities[int(action)] for action in PolicyAction
            },
            "action_availability_fractions": {
                action.label: self.action_availability_fractions[int(action)]
                for action in PolicyAction
            },
            "selected_action_counts": {
                action.label: self.selected_action_counts[int(action)] for action in PolicyAction
            },
        }


@dataclass(frozen=True, slots=True)
class ConstraintPressureDiagnostics:
    """Complete immutable pre-update diagnostics for one PPO iteration."""

    learning_rows: int
    densities: tuple[DensityConstraintPressureDiagnostics, ...]

    def __post_init__(self) -> None:
        _positive_integer("learning_rows", self.learning_rows)
        if (
            not isinstance(self.densities, tuple)
            or not self.densities
            or any(
                not isinstance(row, DensityConstraintPressureDiagnostics) for row in self.densities
            )
        ):
            raise ConstraintDiagnosticsError("constraint-pressure diagnostics require density rows")
        labels = tuple(row.density_veh_per_lane_km for row in self.densities)
        if labels != tuple(sorted(set(labels))):
            raise ConstraintDiagnosticsError(
                "constraint-pressure density rows must be unique and sorted"
            )
        if sum(row.learning_rows for row in self.densities) != self.learning_rows:
            raise ConstraintDiagnosticsError(
                "density diagnostics must partition all PPO learning rows"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA,
            "scope": "read-only pre-update rollout-policy diagnostics",
            "learning_rows": self.learning_rows,
            "densities": [row.as_dict() for row in self.densities],
        }


def build_constraint_pressure_diagnostics(
    *,
    actor: SharedCategoricalActor,
    batch: PPOBatch,
    learning_densities: torch.Tensor,
) -> ConstraintPressureDiagnostics:
    """Summarize constraint pressure without sampling or mutating learner state."""

    if not isinstance(actor, SharedCategoricalActor):
        raise ConstraintDiagnosticsError("constraint diagnostics require an actor module")
    if not isinstance(batch, PPOBatch):
        raise ConstraintDiagnosticsError("constraint diagnostics require a PPOBatch")
    _validate_learning_densities(learning_densities, batch=batch)

    with torch.no_grad():
        logits = actor(batch.actor_observations)
        distribution = MaskedCategorical(logits, batch.action_masks)
        evaluation = distribution.evaluate_actions(batch.actions)
        probabilities = distribution.probabilities.detach()
        entropies = evaluation.entropy.detach()
    if not torch.allclose(
        evaluation.log_probabilities,
        batch.old_log_probabilities,
        rtol=0.0,
        atol=_LOG_PROBABILITY_TOLERANCE,
    ):
        raise ConstraintDiagnosticsError("diagnostic actor no longer matches the rollout policy")

    rows: list[DensityConstraintPressureDiagnostics] = []
    for raw_density in sorted(set(float(value) for value in learning_densities.tolist())):
        selected = learning_densities == raw_density
        count = int(torch.count_nonzero(selected).item())
        weights = batch.cost_penalty_weights[selected]
        multiplier = float(weights[0].item())
        if not torch.allclose(weights, torch.full_like(weights, multiplier)):
            raise ConstraintDiagnosticsError(
                "one density has inconsistent pre-update dual multipliers"
            )
        reward = batch.reward_advantages[selected]
        cost = batch.cost_advantages[selected]
        weighted_cost = weights * cost
        combined = reward - weighted_cost
        probability_rows = probabilities[selected]
        mask_rows = batch.action_masks[selected]
        actions = batch.actions[selected]
        reward_summary = _summarize(reward)
        weighted_summary = _summarize(weighted_cost)
        ratio = (
            None
            if reward_summary.mean_absolute == 0.0
            else weighted_summary.mean_absolute / reward_summary.mean_absolute
        )
        rows.append(
            DensityConstraintPressureDiagnostics(
                density_veh_per_lane_km=raw_density,
                learning_rows=count,
                dual_multiplier_before_update=multiplier,
                reward_advantage=reward_summary,
                cost_advantage=_summarize(cost),
                dual_weighted_cost_advantage=weighted_summary,
                combined_actor_advantage=_summarize(combined),
                entropy=_summarize(entropies[selected]),
                mean_absolute_constraint_to_reward_ratio=ratio,
                positive_combined_advantage_fraction=float(
                    (combined > 0.0).to(torch.float64).mean().item()
                ),
                mean_action_probabilities=tuple(
                    float(value)
                    for value in probability_rows.to(torch.float64).mean(dim=0).tolist()
                ),
                action_availability_fractions=tuple(
                    float(value) for value in mask_rows.to(torch.float64).mean(dim=0).tolist()
                ),
                selected_action_counts=tuple(
                    int(value) for value in torch.bincount(actions, minlength=ACTION_COUNT).tolist()
                ),
            )
        )
    return ConstraintPressureDiagnostics(
        learning_rows=batch.batch_size,
        densities=tuple(rows),
    )


def _summarize(values: torch.Tensor) -> ScalarDistributionSummary:
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 1
        or values.numel() == 0
        or not values.is_floating_point()
        or values.requires_grad
        or not bool(torch.isfinite(values).all().item())
    ):
        raise ConstraintDiagnosticsError(
            "diagnostic vectors must be nonempty, detached, finite, and floating point"
        )
    values64 = values.detach().to(device="cpu", dtype=torch.float64)
    return ScalarDistributionSummary(
        sample_count=int(values64.numel()),
        mean=float(values64.mean().item()),
        standard_deviation=float(torch.std(values64, correction=0).item()),
        mean_absolute=float(torch.abs(values64).mean().item()),
        minimum=float(values64.min().item()),
        maximum=float(values64.max().item()),
    )


def _validate_learning_densities(
    values: torch.Tensor,
    *,
    batch: PPOBatch,
) -> None:
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != (batch.batch_size,)
        or not values.is_floating_point()
        or values.dtype != batch.actor_observations.dtype
        or values.device != batch.actor_observations.device
        or values.requires_grad
        or not bool(torch.isfinite(values).all().item())
        or not bool((values > 0.0).all().item())
    ):
        raise ConstraintDiagnosticsError(
            "learning densities must be a positive finite vector aligned with PPO rows"
        )


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ConstraintDiagnosticsError(f"{name} must be finite")
    return float(value)


def _unit_interval(name: str, value: object) -> float:
    validated = _finite_real(name, value)
    if not 0.0 <= validated <= 1.0:
        raise ConstraintDiagnosticsError(f"{name} must lie in [0, 1]")
    return validated


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConstraintDiagnosticsError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ConstraintDiagnosticsError(f"{name} must be a nonnegative integer")
    return value


__all__ = [
    "CONSTRAINT_PRESSURE_DIAGNOSTICS_SCHEMA",
    "ConstraintDiagnosticsError",
    "ConstraintPressureDiagnostics",
    "DensityConstraintPressureDiagnostics",
    "ScalarDistributionSummary",
    "build_constraint_pressure_diagnostics",
]
