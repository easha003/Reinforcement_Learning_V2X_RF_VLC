"""Projected per-density dual ascent for reliability constraints.

The actor remains shared across traffic densities, but each configured density
owns an independent Lagrange multiplier.  This module deliberately consumes a
generic probability-valued cost signal; selection of sampled misses versus
conditional risk belongs to the rollout/training-signal boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch

from hybrid_v2x_rl.config.models import DensityMultiplierConfig, TrainingConfig
from hybrid_v2x_rl.core.errors import HybridV2XError


class DualAscentError(HybridV2XError):
    """A dual definition, density assignment, or projected update is invalid."""


@dataclass(frozen=True, slots=True)
class DensityDualSnapshot:
    """Immutable, canonically ordered dual state for inspection/checkpointing."""

    densities_veh_per_lane_km: tuple[float, ...]
    multipliers: tuple[float, ...]
    update_counts: tuple[int, ...]

    def multiplier_for_density(self, density_veh_per_lane_km: float) -> float:
        """Return the exact configured density's multiplier."""

        density = _finite_real("density_veh_per_lane_km", density_veh_per_lane_km)
        try:
            index = self.densities_veh_per_lane_km.index(density)
        except ValueError as exc:
            raise DualAscentError(
                "density does not have a configured dual multiplier",
                context={"density_veh_per_lane_km": density},
            ) from exc
        return self.multipliers[index]


@dataclass(frozen=True, slots=True)
class DensityDualUpdate:
    """One density row in an atomic projected-ascent update report."""

    density_veh_per_lane_km: float
    sample_count: int
    estimated_cost: float | None
    miss_budget: float
    violation: float | None
    multiplier_before: float
    multiplier_after: float
    learning_rate: float
    maximum: float


@dataclass(frozen=True, slots=True)
class DensityDualUpdateReport:
    """Projected updates in canonical ascending-density order."""

    updates: tuple[DensityDualUpdate, ...]

    def for_density(self, density_veh_per_lane_km: float) -> DensityDualUpdate:
        """Return the report row for an exact configured density."""

        density = _finite_real("density_veh_per_lane_km", density_veh_per_lane_km)
        for update in self.updates:
            if update.density_veh_per_lane_km == density:
                return update
        raise DualAscentError(
            "density is absent from the dual update report",
            context={"density_veh_per_lane_km": density},
        )


@dataclass(frozen=True, slots=True)
class _DensityDualDefinition:
    density_veh_per_lane_km: float
    learning_rate: float
    maximum: float


def projected_dual_ascent(
    *,
    current: float,
    learning_rate: float,
    estimated_cost: float,
    miss_budget: float,
    maximum: float,
) -> float:
    """Apply ``clip(current + alpha * (estimate - budget), 0, maximum)``."""

    value = _nonnegative_real("current", current)
    rate = _positive_real("learning_rate", learning_rate)
    estimate = _closed_probability("estimated_cost", estimated_cost)
    budget = _open_probability("miss_budget", miss_budget)
    cap = _positive_real("maximum", maximum)
    if value > cap:
        raise DualAscentError(
            "current dual multiplier exceeds its configured maximum",
            context={"current": value, "maximum": cap},
        )
    return min(cap, max(0.0, value + rate * (estimate - budget)))


class PerDensityDualAscent:
    """Own one projected multiplier for every configured traffic density.

    Density labels are matched exactly.  The caller must supply the configured
    target-density label, not a noisy realized-density measurement.  Updates
    use the arithmetic mean of this rollout's cost samples independently for
    every represented density; an absent density is left unchanged.
    """

    def __init__(
        self,
        definitions: tuple[DensityMultiplierConfig, ...],
    ) -> None:
        if not isinstance(definitions, tuple) or not definitions:
            raise DualAscentError("dual definitions must be a nonempty tuple")

        records: list[tuple[_DensityDualDefinition, float]] = []
        seen: set[float] = set()
        for configured in definitions:
            if not isinstance(configured, DensityMultiplierConfig):
                raise DualAscentError("every dual definition must be a DensityMultiplierConfig")
            density = _positive_real(
                "density_veh_per_lane_km",
                configured.density_veh_per_lane_km,
            )
            initial = _nonnegative_real("initial_value", configured.initial_value)
            learning_rate = _positive_real("learning_rate", configured.learning_rate)
            maximum = _positive_real("maximum", configured.maximum)
            if density in seen:
                raise DualAscentError(
                    "dual density definitions must be unique",
                    context={"density_veh_per_lane_km": density},
                )
            if initial > maximum:
                raise DualAscentError(
                    "initial dual multiplier exceeds its configured maximum",
                    context={
                        "density_veh_per_lane_km": density,
                        "initial_value": initial,
                        "maximum": maximum,
                    },
                )
            seen.add(density)
            records.append(
                (
                    _DensityDualDefinition(
                        density_veh_per_lane_km=density,
                        learning_rate=learning_rate,
                        maximum=maximum,
                    ),
                    initial,
                )
            )

        records.sort(key=lambda item: item[0].density_veh_per_lane_km)
        self._definitions = tuple(record[0] for record in records)
        self._multipliers = [record[1] for record in records]
        self._update_counts = [0 for _ in records]
        self._index_by_density = {
            definition.density_veh_per_lane_km: index
            for index, definition in enumerate(self._definitions)
        }

    @classmethod
    def from_config(cls, training: TrainingConfig) -> PerDensityDualAscent:
        """Build canonical dual state from a validated training configuration."""

        if not isinstance(training, TrainingConfig):
            raise DualAscentError("dual ascent requires a validated TrainingConfig")
        return cls(training.density_multipliers)

    @property
    def densities_veh_per_lane_km(self) -> tuple[float, ...]:
        """Configured density labels in canonical order."""

        return tuple(definition.density_veh_per_lane_km for definition in self._definitions)

    def snapshot(self) -> DensityDualSnapshot:
        """Return an immutable copy of current values and update counters."""

        return DensityDualSnapshot(
            densities_veh_per_lane_km=self.densities_veh_per_lane_km,
            multipliers=tuple(self._multipliers),
            update_counts=tuple(self._update_counts),
        )

    def restore(self, snapshot: DensityDualSnapshot) -> None:
        """Atomically restore multiplier values and update counters.

        The receiver's configured density definitions remain authoritative.
        Checkpoint state may change only the mutable multiplier and counter
        arrays, and every proposed value is validated before either array is
        replaced.
        """

        if not isinstance(snapshot, DensityDualSnapshot):
            raise DualAscentError("dual restoration requires a DensityDualSnapshot")
        expected_densities = self.densities_veh_per_lane_km
        if snapshot.densities_veh_per_lane_km != expected_densities:
            raise DualAscentError(
                "checkpoint dual densities do not match configured densities",
                context={
                    "checkpoint": snapshot.densities_veh_per_lane_km,
                    "configured": expected_densities,
                },
            )
        width = len(self._definitions)
        if len(snapshot.multipliers) != width or len(snapshot.update_counts) != width:
            raise DualAscentError("checkpoint dual arrays are not density-aligned")

        multipliers: list[float] = []
        update_counts: list[int] = []
        for definition, multiplier, updates in zip(
            self._definitions,
            snapshot.multipliers,
            snapshot.update_counts,
            strict=True,
        ):
            restored = _nonnegative_real("checkpoint multiplier", multiplier)
            if restored > definition.maximum:
                raise DualAscentError(
                    "checkpoint dual multiplier exceeds its configured maximum",
                    context={
                        "density_veh_per_lane_km": definition.density_veh_per_lane_km,
                        "multiplier": restored,
                        "maximum": definition.maximum,
                    },
                )
            if not isinstance(updates, int) or isinstance(updates, bool) or updates < 0:
                raise DualAscentError("checkpoint dual update counts must be nonnegative integers")
            multipliers.append(restored)
            update_counts.append(updates)

        self._multipliers = multipliers
        self._update_counts = update_counts

    def multiplier_for_density(self, density_veh_per_lane_km: float) -> float:
        """Return the multiplier for an exact configured density label."""

        density = _finite_real("density_veh_per_lane_km", density_veh_per_lane_km)
        try:
            return self._multipliers[self._index_by_density[density]]
        except KeyError as exc:
            raise DualAscentError(
                "density does not have a configured dual multiplier",
                context={"density_veh_per_lane_km": density},
            ) from exc

    def penalty_weights(self, densities_veh_per_lane_km: torch.Tensor) -> torch.Tensor:
        """Assign the current detached multiplier to every PPO row."""

        self._validate_density_vector(densities_veh_per_lane_km)
        weights = torch.empty_like(densities_veh_per_lane_km)
        for definition, multiplier in zip(
            self._definitions,
            self._multipliers,
            strict=True,
        ):
            mask = densities_veh_per_lane_km == definition.density_veh_per_lane_km
            weights[mask] = multiplier
        return weights.detach()

    def update(
        self,
        *,
        densities_veh_per_lane_km: torch.Tensor,
        costs: torch.Tensor,
        miss_budget: float,
    ) -> DensityDualUpdateReport:
        """Atomically update represented densities from undiscounted mean costs."""

        self._validate_density_vector(densities_veh_per_lane_km)
        _validate_cost_vector(costs, reference=densities_veh_per_lane_km)
        budget = _open_probability("miss_budget", miss_budget)

        proposed = list(self._multipliers)
        proposed_counts = list(self._update_counts)
        reports: list[DensityDualUpdate] = []
        for index, definition in enumerate(self._definitions):
            mask = densities_veh_per_lane_km == definition.density_veh_per_lane_km
            sample_count = int(mask.sum().item())
            before = self._multipliers[index]
            if sample_count == 0:
                reports.append(
                    DensityDualUpdate(
                        density_veh_per_lane_km=definition.density_veh_per_lane_km,
                        sample_count=0,
                        estimated_cost=None,
                        miss_budget=budget,
                        violation=None,
                        multiplier_before=before,
                        multiplier_after=before,
                        learning_rate=definition.learning_rate,
                        maximum=definition.maximum,
                    )
                )
                continue

            # A dual update occurs once per rollout, so a CPU float64 reduction
            # provides stable accounting without participating in autograd.
            estimate = float(
                costs[mask].detach().to(device="cpu", dtype=torch.float64).mean().item()
            )
            after = projected_dual_ascent(
                current=before,
                learning_rate=definition.learning_rate,
                estimated_cost=estimate,
                miss_budget=budget,
                maximum=definition.maximum,
            )
            proposed[index] = after
            proposed_counts[index] += 1
            reports.append(
                DensityDualUpdate(
                    density_veh_per_lane_km=definition.density_veh_per_lane_km,
                    sample_count=sample_count,
                    estimated_cost=estimate,
                    miss_budget=budget,
                    violation=estimate - budget,
                    multiplier_before=before,
                    multiplier_after=after,
                    learning_rate=definition.learning_rate,
                    maximum=definition.maximum,
                )
            )

        self._multipliers = proposed
        self._update_counts = proposed_counts
        return DensityDualUpdateReport(updates=tuple(reports))

    def _validate_density_vector(self, densities: torch.Tensor) -> None:
        if not isinstance(densities, torch.Tensor) or densities.ndim != 1:
            raise DualAscentError("density labels must be a one-dimensional torch.Tensor")
        if densities.numel() == 0:
            raise DualAscentError("density labels cannot be empty")
        if not densities.is_floating_point():
            raise DualAscentError("density labels must use a floating-point dtype")
        if densities.requires_grad:
            raise DualAscentError("density labels must be detached rollout metadata")
        if not bool(torch.isfinite(densities).all().item()):
            raise DualAscentError("density labels contain a non-finite value")

        recognized = torch.zeros_like(densities, dtype=torch.bool)
        for definition in self._definitions:
            recognized |= densities == definition.density_veh_per_lane_km
        if not bool(recognized.all().item()):
            unknown = tuple(
                float(value)
                for value in torch.unique(densities[~recognized]).detach().cpu().tolist()
            )
            raise DualAscentError(
                "density labels must exactly match configured dual densities",
                context={"unknown_densities_veh_per_lane_km": unknown},
            )


def _validate_cost_vector(costs: torch.Tensor, *, reference: torch.Tensor) -> None:
    if not isinstance(costs, torch.Tensor) or costs.ndim != 1:
        raise DualAscentError("dual costs must be a one-dimensional torch.Tensor")
    if costs.shape != reference.shape:
        raise DualAscentError(
            "dual costs and density labels must have the same shape",
            context={"costs": tuple(costs.shape), "densities": tuple(reference.shape)},
        )
    if not costs.is_floating_point():
        raise DualAscentError("dual costs must use a floating-point dtype")
    if costs.device != reference.device:
        raise DualAscentError("dual costs and density labels must use the same device")
    if costs.requires_grad:
        raise DualAscentError("dual costs must be detached rollout data")
    if not bool(torch.isfinite(costs).all().item()):
        raise DualAscentError("dual costs contain a non-finite value")
    if bool(((costs < 0.0) | (costs > 1.0)).any().item()):
        raise DualAscentError("dual costs must lie in the closed interval [0, 1]")


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DualAscentError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise DualAscentError(f"{name} must be finite")
    return converted


def _nonnegative_real(name: str, value: object) -> float:
    converted = _finite_real(name, value)
    if converted < 0.0:
        raise DualAscentError(f"{name} must be nonnegative")
    return converted


def _positive_real(name: str, value: object) -> float:
    converted = _finite_real(name, value)
    if converted <= 0.0:
        raise DualAscentError(f"{name} must be positive")
    return converted


def _closed_probability(name: str, value: object) -> float:
    converted = _finite_real(name, value)
    if not 0.0 <= converted <= 1.0:
        raise DualAscentError(f"{name} must lie in [0, 1]")
    return converted


def _open_probability(name: str, value: object) -> float:
    converted = _finite_real(name, value)
    if not 0.0 < converted < 1.0:
        raise DualAscentError(f"{name} must lie in (0, 1)")
    return converted


__all__ = [
    "DensityDualSnapshot",
    "DensityDualUpdate",
    "DensityDualUpdateReport",
    "DualAscentError",
    "PerDensityDualAscent",
    "projected_dual_ascent",
]
