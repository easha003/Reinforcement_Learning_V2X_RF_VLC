"""Versioned rollout/update metrics for primal-dual PPO training."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final, NoReturn

import torch

from hybrid_v2x_rl.agents.cost_signal import CONDITIONAL_MISS_PROBABILITY
from hybrid_v2x_rl.agents.dual_ascent import (
    DensityDualSnapshot,
    DensityDualUpdate,
    DensityDualUpdateReport,
)
from hybrid_v2x_rl.agents.ppo import PPOUpdateMetrics
from hybrid_v2x_rl.core.errors import HybridV2XError

TRAINING_METRICS_SCHEMA: Final = "hybrid-rf-vlc-rl.training-metrics.v1"
_PPO_SCALAR_FIELDS: Final = (
    "actor_loss",
    "policy_loss",
    "reward_value_loss",
    "cost_value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "ratio_mean",
)


class TrainingMetricsError(HybridV2XError):
    """A training metric, aggregation, or JSONL log violates its contract."""


@dataclass(frozen=True, slots=True)
class AggregatedPPOMetrics:
    """Row-weighted optimizer diagnostics across minibatches and epochs."""

    minibatch_updates: int
    optimizer_rows: int
    actor_loss: float
    policy_loss: float
    reward_value_loss: float
    cost_value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    ratio_mean: float

    def __post_init__(self) -> None:
        _positive_integer("minibatch_updates", self.minibatch_updates)
        _positive_integer("optimizer_rows", self.optimizer_rows)
        for field in _PPO_SCALAR_FIELDS:
            _finite_real(field, getattr(self, field))
        if self.reward_value_loss < 0.0 or self.cost_value_loss < 0.0:
            raise TrainingMetricsError("aggregated value losses must be nonnegative")
        if self.entropy < 0.0 or self.approximate_kl < 0.0:
            raise TrainingMetricsError("aggregated entropy and KL must be nonnegative")
        if not 0.0 <= self.clip_fraction <= 1.0:
            raise TrainingMetricsError("aggregated clip fraction must lie in [0, 1]")
        if self.ratio_mean <= 0.0:
            raise TrainingMetricsError("aggregated probability-ratio mean must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "minibatch_updates": self.minibatch_updates,
            "optimizer_rows": self.optimizer_rows,
            "actor_loss": self.actor_loss,
            "policy_loss": self.policy_loss,
            "reward_value_loss": self.reward_value_loss,
            "cost_value_loss": self.cost_value_loss,
            "entropy": self.entropy,
            "approximate_kl": self.approximate_kl,
            "clip_fraction": self.clip_fraction,
            "ratio_mean": self.ratio_mean,
        }


@dataclass(frozen=True, slots=True)
class DensityConstraintMetrics:
    """One density's rollout constraint estimate and projected dual state."""

    density_veh_per_lane_km: float
    sample_count: int
    conditional_miss_estimate: float | None
    miss_budget: float
    violation: float | None
    dual_before: float
    dual_after: float
    dual_learning_rate: float
    dual_maximum: float
    dual_update_count: int

    def __post_init__(self) -> None:
        density = _finite_real(
            "density_veh_per_lane_km",
            self.density_veh_per_lane_km,
        )
        if density <= 0.0:
            raise TrainingMetricsError("metric density must be positive")
        _nonnegative_integer("sample_count", self.sample_count)
        _nonnegative_integer("dual_update_count", self.dual_update_count)
        budget = _finite_real("miss_budget", self.miss_budget)
        if not 0.0 < budget < 1.0:
            raise TrainingMetricsError("metric miss budget must lie in (0, 1)")
        before = _finite_real("dual_before", self.dual_before)
        after = _finite_real("dual_after", self.dual_after)
        rate = _finite_real("dual_learning_rate", self.dual_learning_rate)
        maximum = _finite_real("dual_maximum", self.dual_maximum)
        if before < 0.0 or after < 0.0 or rate <= 0.0 or maximum <= 0.0:
            raise TrainingMetricsError("dual metrics violate nonnegative/positive bounds")
        if before > maximum or after > maximum:
            raise TrainingMetricsError("dual metric exceeds its configured maximum")

        if self.sample_count == 0:
            if self.conditional_miss_estimate is not None or self.violation is not None:
                raise TrainingMetricsError("an absent density cannot have a constraint estimate")
            if before != after:
                raise TrainingMetricsError("an absent density cannot change its dual")
            return
        if self.conditional_miss_estimate is None or self.violation is None:
            raise TrainingMetricsError("a represented density requires estimate and violation")
        if self.dual_update_count == 0:
            raise TrainingMetricsError("a represented density requires a dual update count")
        estimate = _finite_real(
            "conditional_miss_estimate",
            self.conditional_miss_estimate,
        )
        violation = _finite_real("violation", self.violation)
        if not 0.0 <= estimate <= 1.0:
            raise TrainingMetricsError("constraint estimate must lie in [0, 1]")
        if not math.isclose(violation, estimate - budget, rel_tol=0.0, abs_tol=1e-12):
            raise TrainingMetricsError("constraint violation does not equal estimate minus budget")
        expected_after = min(maximum, max(0.0, before + rate * violation))
        if not math.isclose(after, expected_after, rel_tol=0.0, abs_tol=1e-12):
            raise TrainingMetricsError("logged dual value does not match projected ascent")

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "density_veh_per_lane_km": self.density_veh_per_lane_km,
            "sample_count": self.sample_count,
            "conditional_miss_estimate": self.conditional_miss_estimate,
            "miss_budget": self.miss_budget,
            "violation": self.violation,
            "dual_before": self.dual_before,
            "dual_after": self.dual_after,
            "dual_learning_rate": self.dual_learning_rate,
            "dual_maximum": self.dual_maximum,
            "dual_update_count": self.dual_update_count,
        }


@dataclass(frozen=True, slots=True)
class TrainingIterationMetrics:
    """One complete machine-readable rollout and optimizer metric record."""

    config_hash: str
    policy_seed: int
    iteration: int
    environment_transitions: int
    rollout_transitions: int
    learning_rows: int
    ppo: AggregatedPPOMetrics
    reward_explained_variance: float
    cost_explained_variance: float
    densities: tuple[DensityConstraintMetrics, ...]

    def __post_init__(self) -> None:
        _sha256("config_hash", self.config_hash)
        _seed("policy_seed", self.policy_seed)
        _nonnegative_integer("iteration", self.iteration)
        _positive_integer("environment_transitions", self.environment_transitions)
        _positive_integer("rollout_transitions", self.rollout_transitions)
        _positive_integer("learning_rows", self.learning_rows)
        if self.rollout_transitions > self.environment_transitions:
            raise TrainingMetricsError(
                "rollout transitions cannot exceed cumulative environment transitions"
            )
        if self.learning_rows > self.rollout_transitions:
            raise TrainingMetricsError("learning rows cannot exceed collected rollout transitions")
        if not isinstance(self.ppo, AggregatedPPOMetrics):
            raise TrainingMetricsError("training record requires aggregated PPO metrics")
        reward_ev = _finite_real(
            "reward_explained_variance",
            self.reward_explained_variance,
        )
        cost_ev = _finite_real("cost_explained_variance", self.cost_explained_variance)
        if reward_ev > 1.0 or cost_ev > 1.0:
            raise TrainingMetricsError("explained variance cannot exceed one")
        if (
            not isinstance(self.densities, tuple)
            or not self.densities
            or any(not isinstance(row, DensityConstraintMetrics) for row in self.densities)
        ):
            raise TrainingMetricsError("density metrics must be nonempty, unique, and sorted")
        density_labels = tuple(row.density_veh_per_lane_km for row in self.densities)
        if density_labels != tuple(sorted(set(density_labels))):
            raise TrainingMetricsError("density metrics must be nonempty, unique, and sorted")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": TRAINING_METRICS_SCHEMA,
            "config_hash": self.config_hash,
            "policy_seed": self.policy_seed,
            "iteration": self.iteration,
            "counters": {
                "environment_transitions": self.environment_transitions,
                "rollout_transitions": self.rollout_transitions,
                "learning_rows": self.learning_rows,
            },
            "reliability_cost_signal": CONDITIONAL_MISS_PROBABILITY,
            "ppo": self.ppo.as_dict(),
            "critics": {
                "reward_explained_variance": self.reward_explained_variance,
                "cost_explained_variance": self.cost_explained_variance,
            },
            "densities": [row.as_dict() for row in self.densities],
        }

    def to_json_line(self) -> str:
        """Serialize one compact deterministic JSONL record."""

        return (
            json.dumps(
                self.as_dict(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def aggregate_ppo_updates(
    updates: tuple[PPOUpdateMetrics, ...],
) -> AggregatedPPOMetrics:
    """Aggregate minibatch means using their actual number of optimizer rows."""

    if not isinstance(updates, tuple) or not updates:
        raise TrainingMetricsError("PPO metric aggregation requires a nonempty tuple")
    if any(not isinstance(update, PPOUpdateMetrics) for update in updates):
        raise TrainingMetricsError("PPO metric aggregation received an invalid update")
    optimizer_rows = sum(update.minibatch_size for update in updates)
    weighted = {
        field: math.fsum(getattr(update, field) * update.minibatch_size for update in updates)
        / optimizer_rows
        for field in _PPO_SCALAR_FIELDS
    }
    return AggregatedPPOMetrics(
        minibatch_updates=len(updates),
        optimizer_rows=optimizer_rows,
        actor_loss=weighted["actor_loss"],
        policy_loss=weighted["policy_loss"],
        reward_value_loss=weighted["reward_value_loss"],
        cost_value_loss=weighted["cost_value_loss"],
        entropy=weighted["entropy"],
        approximate_kl=weighted["approximate_kl"],
        clip_fraction=weighted["clip_fraction"],
        ratio_mean=weighted["ratio_mean"],
    )


def explained_variance(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Return ``1 - Var(target - prediction) / Var(target)``.

    A constant target has no explainable variance, so this implementation
    returns finite sentinel ``0.0`` rather than emitting NaN into the log.
    """

    _validate_metric_vector("predictions", predictions)
    _validate_metric_vector("targets", targets, reference=predictions)
    prediction64 = predictions.detach().to(device="cpu", dtype=torch.float64)
    target64 = targets.detach().to(device="cpu", dtype=torch.float64)
    target_variance = torch.var(target64, correction=0)
    if float(target_variance.item()) == 0.0:
        return 0.0
    residual_variance = torch.var(target64 - prediction64, correction=0)
    result = 1.0 - float((residual_variance / target_variance).item())
    if not math.isfinite(result):
        raise TrainingMetricsError("explained variance is non-finite")
    return result


def build_training_iteration_metrics(
    *,
    config_hash: str,
    policy_seed: int,
    iteration: int,
    environment_transitions: int,
    rollout_transitions: int,
    ppo_updates: tuple[PPOUpdateMetrics, ...],
    reward_predictions: torch.Tensor,
    reward_targets: torch.Tensor,
    cost_predictions: torch.Tensor,
    cost_targets: torch.Tensor,
    dual_report: DensityDualUpdateReport,
    dual_snapshot: DensityDualSnapshot,
) -> TrainingIterationMetrics:
    """Combine optimizer, critic, constraint, and dual diagnostics."""

    if not isinstance(dual_report, DensityDualUpdateReport):
        raise TrainingMetricsError("training metrics require a dual update report")
    if not isinstance(dual_snapshot, DensityDualSnapshot):
        raise TrainingMetricsError("training metrics require a dual snapshot")
    if any(not isinstance(update, DensityDualUpdate) for update in dual_report.updates):
        raise TrainingMetricsError("dual update report contains an invalid row")
    if any(
        not isinstance(values, torch.Tensor)
        for values in (
            reward_predictions,
            reward_targets,
            cost_predictions,
            cost_targets,
        )
    ):
        raise TrainingMetricsError("critic metrics must be torch tensors")
    if (
        reward_predictions.shape != cost_predictions.shape
        or reward_predictions.dtype != cost_predictions.dtype
        or reward_predictions.device != cost_predictions.device
    ):
        raise TrainingMetricsError("reward and cost critic metrics must use aligned rows")
    report_densities = tuple(update.density_veh_per_lane_km for update in dual_report.updates)
    if (
        not dual_report.updates
        or report_densities != tuple(sorted(set(report_densities)))
        or report_densities != dual_snapshot.densities_veh_per_lane_km
        or len(dual_snapshot.multipliers) != len(report_densities)
        or len(dual_snapshot.update_counts) != len(report_densities)
    ):
        raise TrainingMetricsError("dual report and snapshot densities do not align")

    density_metrics: list[DensityConstraintMetrics] = []
    for index, update in enumerate(dual_report.updates):
        snapshot_value = dual_snapshot.multipliers[index]
        if not math.isclose(
            update.multiplier_after,
            snapshot_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TrainingMetricsError("dual report does not match current snapshot")
        density_metrics.append(
            DensityConstraintMetrics(
                density_veh_per_lane_km=update.density_veh_per_lane_km,
                sample_count=update.sample_count,
                conditional_miss_estimate=update.estimated_cost,
                miss_budget=update.miss_budget,
                violation=update.violation,
                dual_before=update.multiplier_before,
                dual_after=update.multiplier_after,
                dual_learning_rate=update.learning_rate,
                dual_maximum=update.maximum,
                dual_update_count=dual_snapshot.update_counts[index],
            )
        )

    if sum(row.sample_count for row in density_metrics) != rollout_transitions:
        raise TrainingMetricsError("density constraint samples must partition rollout transitions")
    learning_rows = int(reward_predictions.numel())
    if learning_rows > rollout_transitions:
        raise TrainingMetricsError("critic metric rows cannot exceed collected rollout transitions")

    return TrainingIterationMetrics(
        config_hash=config_hash,
        policy_seed=policy_seed,
        iteration=iteration,
        environment_transitions=environment_transitions,
        rollout_transitions=rollout_transitions,
        learning_rows=learning_rows,
        ppo=aggregate_ppo_updates(ppo_updates),
        reward_explained_variance=explained_variance(
            predictions=reward_predictions,
            targets=reward_targets,
        ),
        cost_explained_variance=explained_variance(
            predictions=cost_predictions,
            targets=cost_targets,
        ),
        densities=tuple(density_metrics),
    )


class TrainingMetricsJSONL:
    """Single-writer append-only JSONL sink with resume-order validation."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._last_iteration: int | None = None
        self._last_environment_transitions: int | None = None
        self._config_hash: str | None = None
        self._policy_seed: int | None = None
        if self.path.exists() or self.path.is_symlink():
            if self.path.is_symlink() or not self.path.is_file():
                raise TrainingMetricsError(
                    "training metrics path must be a regular non-symlink file",
                    artifact_path=self.path,
                )
            (
                self._last_iteration,
                self._last_environment_transitions,
                self._config_hash,
                self._policy_seed,
            ) = _inspect_existing_log(self.path)

    def append(self, record: TrainingIterationMetrics) -> Path:
        """Append and fsync one record after enforcing monotonic counters."""

        if not isinstance(record, TrainingIterationMetrics):
            raise TrainingMetricsError("training metrics logger requires a validated record")
        if self._config_hash is not None and (
            record.config_hash != self._config_hash or record.policy_seed != self._policy_seed
        ):
            raise TrainingMetricsError(
                "training metrics log belongs to a different configuration or policy seed"
            )
        if self._last_iteration is not None and record.iteration <= self._last_iteration:
            raise TrainingMetricsError("training metric iterations must increase strictly")
        if (
            self._last_environment_transitions is not None
            and record.environment_transitions <= self._last_environment_transitions
        ):
            raise TrainingMetricsError(
                "training metric environment transitions must increase strictly"
            )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise TrainingMetricsError(
                "training metrics path cannot be a symlink",
                artifact_path=self.path,
            )
        payload = record.to_json_line().encode("utf-8")
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
            with os.fdopen(descriptor, "ab", buffering=0) as stream:
                written = stream.write(payload)
                if written != len(payload):
                    raise OSError("short write while appending training metrics")
                os.fsync(stream.fileno())
        except OSError as exc:
            raise TrainingMetricsError(
                "cannot append training metrics",
                artifact_path=self.path,
                context={"reason": str(exc)},
            ) from exc
        self._last_iteration = record.iteration
        self._last_environment_transitions = record.environment_transitions
        self._config_hash = record.config_hash
        self._policy_seed = record.policy_seed
        return self.path


def _inspect_existing_log(
    path: Path,
) -> tuple[int | None, int | None, str | None, int | None]:
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrainingMetricsError(
            "cannot read existing training metrics",
            artifact_path=path,
            context={"reason": str(exc)},
        ) from exc
    if not contents:
        return None, None, None, None
    if not contents.endswith("\n"):
        raise TrainingMetricsError(
            "existing training metrics end with a partial JSONL record",
            artifact_path=path,
        )

    last_iteration: int | None = None
    last_transitions: int | None = None
    config_hash: str | None = None
    policy_seed: int | None = None
    for line_number, line in enumerate(contents.splitlines(), start=1):
        try:
            payload = json.loads(line, parse_constant=_reject_nonfinite_json)
        except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
            raise TrainingMetricsError(
                "existing training metrics contain invalid JSON",
                artifact_path=path,
                context={"line": line_number},
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema") != TRAINING_METRICS_SCHEMA:
            raise TrainingMetricsError(
                "existing training metrics have an unsupported schema",
                artifact_path=path,
                context={"line": line_number},
            )
        iteration = payload.get("iteration")
        row_config_hash = payload.get("config_hash")
        row_policy_seed = payload.get("policy_seed")
        counters = payload.get("counters")
        transitions = (
            counters.get("environment_transitions") if isinstance(counters, dict) else None
        )
        if (
            not isinstance(iteration, int)
            or isinstance(iteration, bool)
            or not isinstance(transitions, int)
            or isinstance(transitions, bool)
            or not isinstance(row_config_hash, str)
            or len(row_config_hash) != 64
            or any(character not in "0123456789abcdef" for character in row_config_hash)
            or not isinstance(row_policy_seed, int)
            or isinstance(row_policy_seed, bool)
            or not 0 <= row_policy_seed < 2**64
            or iteration < 0
            or transitions <= 0
            or (last_iteration is not None and iteration <= last_iteration)
            or (last_transitions is not None and transitions <= last_transitions)
        ):
            raise TrainingMetricsError(
                "existing training metric counters are invalid or non-monotonic",
                artifact_path=path,
                context={"line": line_number},
            )
        if config_hash is not None and (
            row_config_hash != config_hash or row_policy_seed != policy_seed
        ):
            raise TrainingMetricsError(
                "existing training metrics mix configurations or policy seeds",
                artifact_path=path,
                context={"line": line_number},
            )
        last_iteration = iteration
        last_transitions = transitions
        config_hash = row_config_hash
        policy_seed = row_policy_seed
    return last_iteration, last_transitions, config_hash, policy_seed


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value}")


def _validate_metric_vector(
    name: str,
    values: torch.Tensor,
    *,
    reference: torch.Tensor | None = None,
) -> None:
    if not isinstance(values, torch.Tensor) or values.ndim != 1 or values.numel() == 0:
        raise TrainingMetricsError(f"{name} must be a nonempty rank-one torch.Tensor")
    if not values.is_floating_point():
        raise TrainingMetricsError(f"{name} must be floating point")
    if values.requires_grad:
        raise TrainingMetricsError(f"{name} must be detached")
    if not bool(torch.isfinite(values).all().item()):
        raise TrainingMetricsError(f"{name} contains a non-finite value")
    if reference is not None:
        if values.shape != reference.shape:
            raise TrainingMetricsError(f"{name} must match prediction shape")
        if values.dtype != reference.dtype:
            raise TrainingMetricsError(f"{name} must match prediction dtype")
        if values.device != reference.device:
            raise TrainingMetricsError(f"{name} must use the prediction device")


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise TrainingMetricsError(f"{name} must be finite")
    return float(value)


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingMetricsError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrainingMetricsError(f"{name} must be a nonnegative integer")
    return value


def _seed(name: str, value: object) -> int:
    validated = _nonnegative_integer(name, value)
    if validated >= 2**64:
        raise TrainingMetricsError(f"{name} must be an unsigned 64-bit integer")
    return validated


def _sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingMetricsError(f"{name} must be a lowercase SHA-256 digest")
    return value


__all__ = [
    "TRAINING_METRICS_SCHEMA",
    "AggregatedPPOMetrics",
    "DensityConstraintMetrics",
    "TrainingIterationMetrics",
    "TrainingMetricsError",
    "TrainingMetricsJSONL",
    "aggregate_ppo_updates",
    "build_training_iteration_metrics",
    "explained_variance",
]
