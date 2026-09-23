"""Matched deployable-to-oracle gap measurement for Phase 6."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np

from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources
from hybrid_v2x_rl.mean_field.baselines import (
    BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_CONTEXTUAL,
    BASELINE_DUPLICATE_ALL,
    BASELINE_GEOMETRY_THRESHOLD,
    BASELINE_ORACLE,
    BASELINE_SUPERVISED,
)
from hybrid_v2x_rl.mean_field.density_metrics import (
    DensityMetricsReport,
    EvaluationSplit,
    PolicyDensityMetrics,
)
from hybrid_v2x_rl.mean_field.matched_campaign import MatchedPolicyCampaignReport

ORACLE_GAP_SCHEMA: Final = "hybrid-rf-vlc-rl.deployable-oracle-gap.v1"
REQUIRED_DEPLOYABLE_BASELINES: Final = (
    *BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_DUPLICATE_ALL,
    BASELINE_GEOMETRY_THRESHOLD,
    BASELINE_CONTEXTUAL,
    BASELINE_SUPERVISED,
)
REQUIRED_GAP_POLICIES: Final = (*REQUIRED_DEPLOYABLE_BASELINES, BASELINE_ORACLE)
GapStatus = Literal[
    "diagnostic",
    "comparable",
    "oracle-only-feasible",
    "oracle-infeasible",
    "no-feasible-policy",
]


class OracleGapError(HybridV2XError):
    """The best-deployable-to-oracle comparison is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class GapEstimate:
    """One paired policy-minus-oracle metric difference and bootstrap interval."""

    metric: str
    deployable_value: float
    oracle_value: float
    gap: float
    confidence_lower: float
    confidence_upper: float
    relative_gap: float | None

    def __post_init__(self) -> None:
        if not self.metric:
            raise OracleGapError("gap metric name must be non-empty")
        values = (
            self.deployable_value,
            self.oracle_value,
            self.gap,
            self.confidence_lower,
            self.confidence_upper,
        )
        if any(not math.isfinite(value) for value in values):
            raise OracleGapError("gap estimates must be finite")
        if not math.isclose(
            self.gap,
            self.deployable_value - self.oracle_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise OracleGapError("gap must equal deployable minus oracle")
        if self.confidence_lower > self.confidence_upper:
            raise OracleGapError("gap confidence interval is reversed")
        if self.relative_gap is not None and not math.isfinite(self.relative_gap):
            raise OracleGapError("relative gap must be finite when available")

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "deployable_value": self.deployable_value,
            "oracle_value": self.oracle_value,
            "gap_deployable_minus_oracle": self.gap,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "relative_gap_to_oracle": self.relative_gap,
        }


@dataclass(frozen=True, slots=True)
class OracleGapDensity:
    """Selection and gap status at one traffic density."""

    density: float
    status: GapStatus
    closest_deployable_policy: str
    feasible_deployable_policies: tuple[str, ...]
    best_deployable_policy: str | None
    oracle_meets_budget: bool | None
    estimates: tuple[GapEstimate, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise OracleGapError("gap density must be finite and positive")
        if self.status not in (
            "diagnostic",
            "comparable",
            "oracle-only-feasible",
            "oracle-infeasible",
            "no-feasible-policy",
        ):
            raise OracleGapError("oracle-gap status is invalid")
        if self.closest_deployable_policy not in REQUIRED_DEPLOYABLE_BASELINES:
            raise OracleGapError("closest deployable policy is not a required baseline")
        if any(
            policy not in REQUIRED_DEPLOYABLE_BASELINES
            for policy in self.feasible_deployable_policies
        ) or len(self.feasible_deployable_policies) != len(
            set(self.feasible_deployable_policies)
        ):
            raise OracleGapError("feasible deployable policies are invalid")
        if self.best_deployable_policy is not None and (
            self.best_deployable_policy not in self.feasible_deployable_policies
        ):
            raise OracleGapError("best deployable must belong to the feasible set")
        if self.oracle_meets_budget is not None and type(self.oracle_meets_budget) is not bool:
            raise OracleGapError("oracle reliability verdict must be boolean or unavailable")
        if self.status == "diagnostic":
            if (
                self.oracle_meets_budget is not None
                or self.feasible_deployable_policies
                or self.best_deployable_policy is not None
                or self.estimates
            ):
                raise OracleGapError("diagnostic gap rows cannot make comparison claims")
        elif self.oracle_meets_budget is None:
            raise OracleGapError("ready gap rows require an oracle verdict")
        if self.status == "comparable":
            if (
                not self.oracle_meets_budget
                or self.best_deployable_policy is None
                or not self.estimates
            ):
                raise OracleGapError("comparable gap row lacks a feasible paired estimate")
        elif self.estimates:
            raise OracleGapError("only comparable rows may carry numeric gap estimates")

    @property
    def comparison_ready(self) -> bool:
        return self.status != "diagnostic"

    @property
    def comparable(self) -> bool:
        return self.status == "comparable"

    def as_dict(self) -> dict[str, object]:
        return {
            "density_vehicles_per_lane_km": self.density,
            "status": self.status,
            "comparison_ready": self.comparison_ready,
            "comparable": self.comparable,
            "selection_rule": (
                "conditional upper bound <= budget; then minimum activation cost; "
                "then conditional upper bound, RF attempts, and policy name"
            ),
            "closest_deployable_policy": self.closest_deployable_policy,
            "feasible_deployable_policies": list(self.feasible_deployable_policies),
            "best_deployable_policy": self.best_deployable_policy,
            "oracle_meets_budget": self.oracle_meets_budget,
            "estimates": [estimate.as_dict() for estimate in self.estimates],
        }


@dataclass(frozen=True, slots=True)
class OracleGapReport:
    """Versioned matched comparison of the best deployable and truth oracle."""

    config_hash: str
    environment_seed: int
    split: EvaluationSplit
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    densities: tuple[OracleGapDensity, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        if len(self.config_hash) != 64:
            raise OracleGapError("oracle-gap config hash must be SHA-256")
        if not 0.5 < self.confidence_level < 1.0:
            raise OracleGapError("oracle-gap confidence must lie in (0.5, 1)")
        if self.bootstrap_replicates < 1_000:
            raise OracleGapError("oracle-gap report requires at least 1,000 bootstraps")
        if self.bootstrap_seed < 0:
            raise OracleGapError("oracle-gap bootstrap seed must be non-negative")
        values = tuple(row.density for row in self.densities)
        if not values or values != tuple(sorted(set(values))):
            raise OracleGapError("oracle-gap densities must be unique and sorted")
        if self.generated_at_utc.tzinfo is None:
            raise OracleGapError("oracle-gap timestamp must be timezone-aware")

    @property
    def comparison_ready(self) -> bool:
        return all(row.comparison_ready for row in self.densities)

    @property
    def all_densities_comparable(self) -> bool:
        return all(row.comparable for row in self.densities)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": ORACLE_GAP_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "config_hash": self.config_hash,
            "environment_seed": self.environment_seed,
            "split": self.split,
            "required_deployable_baselines": list(REQUIRED_DEPLOYABLE_BASELINES),
            "oracle_policy": BASELINE_ORACLE,
            "confidence_level": self.confidence_level,
            "interval_method": "paired_trajectory_pair_cluster_bootstrap",
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "comparison_ready": self.comparison_ready,
            "all_densities_comparable": self.all_densities_comparable,
            "densities": [row.as_dict() for row in self.densities],
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the oracle-gap report."""

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


@dataclass(frozen=True, slots=True)
class _ClusterGapRow:
    key: str
    packets: int
    activation_cost: float
    reserved_rf_attempts: int
    rf_uses: int
    vlc_uses: int
    duplications: int
    conditional_misses: float
    sampled_misses: int


def _cluster_gap_rows(
    campaign: MatchedPolicyCampaignReport,
    *,
    split: EvaluationSplit,
    density: float,
    policy: str,
) -> tuple[_ClusterGapRow, ...]:
    rows: list[_ClusterGapRow] = []
    for comparison in campaign.comparisons:
        if comparison.source.split != split or comparison.source.density != density:
            continue
        report = next(item for item in comparison.reports if item.policy == policy)
        for cluster in report.episode_clusters:
            reserved = rf_uses = vlc_uses = duplications = 0
            for action in PolicyAction:
                count = cluster.action_counts[int(action)]
                spec = action_resources(action)
                reserved += count * spec.reserved_rf_attempts
                rf_uses += count * int(spec.uses_rf)
                vlc_uses += count * int(spec.uses_vlc)
                duplications += count * int(spec.duplicates)
            rows.append(
                _ClusterGapRow(
                    key=f"{report.trace_id}/{cluster.pair_id}",
                    packets=cluster.packets,
                    activation_cost=-cluster.reward_sum,
                    reserved_rf_attempts=reserved,
                    rf_uses=rf_uses,
                    vlc_uses=vlc_uses,
                    duplications=duplications,
                    conditional_misses=cluster.conditional_risk_sum,
                    sampled_misses=cluster.misses,
                )
            )
    return tuple(rows)


def _paired_estimates(
    campaign: MatchedPolicyCampaignReport,
    *,
    split: EvaluationSplit,
    density: float,
    deployable_policy: str,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[GapEstimate, ...]:
    deployable = _cluster_gap_rows(
        campaign,
        split=split,
        density=density,
        policy=deployable_policy,
    )
    oracle = _cluster_gap_rows(
        campaign,
        split=split,
        density=density,
        policy=BASELINE_ORACLE,
    )
    deployable_structure = tuple((row.key, row.packets) for row in deployable)
    oracle_structure = tuple((row.key, row.packets) for row in oracle)
    if not deployable or deployable_structure != oracle_structure:
        raise OracleGapError("paired oracle-gap clusters are not exactly aligned")

    packets = np.asarray([row.packets for row in deployable], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(packets), size=(replicates, len(packets)))
    denominators = packets[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    metric_fields = (
        ("mean_activation_cost", "activation_cost"),
        ("mean_reserved_rf_attempts", "reserved_rf_attempts"),
        ("rf_use_fraction", "rf_uses"),
        ("vlc_use_fraction", "vlc_uses"),
        ("duplication_fraction", "duplications"),
        ("conditional_miss_rate", "conditional_misses"),
        ("sampled_miss_rate", "sampled_misses"),
    )
    estimates: list[GapEstimate] = []
    packet_total = float(packets.sum())
    for metric, field in metric_fields:
        deployable_values = np.asarray(
            [float(getattr(row, field)) for row in deployable],
            dtype=np.float64,
        )
        oracle_values = np.asarray(
            [float(getattr(row, field)) for row in oracle],
            dtype=np.float64,
        )
        deployable_value = float(deployable_values.sum() / packet_total)
        oracle_value = float(oracle_values.sum() / packet_total)
        gap = deployable_value - oracle_value
        resampled = (deployable_values - oracle_values)[draws].sum(axis=1) / denominators
        lower, upper = np.quantile(resampled, (alpha, 1.0 - alpha))
        estimates.append(
            GapEstimate(
                metric=metric,
                deployable_value=deployable_value,
                oracle_value=oracle_value,
                gap=gap,
                confidence_lower=float(lower),
                confidence_upper=float(upper),
                relative_gap=(gap / oracle_value if abs(oracle_value) > 1e-15 else None),
            )
        )
    return tuple(estimates)


def _metric_map(
    report: DensityMetricsReport,
    density: float,
) -> dict[str, PolicyDensityMetrics]:
    block = next(block for block in report.densities if block.density == density)
    return {metric.policy: metric for metric in block.policies}


def _require_density_report_binding(
    campaign: MatchedPolicyCampaignReport,
    density_report: DensityMetricsReport,
) -> None:
    """Reject a same-config report derived from different campaign content."""

    for block in density_report.densities:
        comparisons = tuple(
            comparison
            for comparison in campaign.comparisons
            if comparison.source.split == density_report.split
            and comparison.source.density == block.density
        )
        expected_traces = tuple(comparison.source.trace_id for comparison in comparisons)
        if not comparisons:
            raise OracleGapError("density report is absent from the matched campaign")
        for metric in block.policies:
            reports = tuple(
                next(report for report in comparison.reports if report.policy == metric.policy)
                for comparison in comparisons
            )
            if (
                metric.trace_ids != expected_traces
                or metric.packets != sum(report.transitions for report in reports)
                or metric.source_exhausted
                != all(report.source_exhausted for report in reports)
            ):
                raise OracleGapError(
                    "density metrics do not bind to the supplied matched campaign",
                    context={"density": block.density, "policy": metric.policy},
                )


def build_oracle_gap_report(
    config: ProjectConfig,
    campaign: MatchedPolicyCampaignReport,
    density_report: DensityMetricsReport,
) -> OracleGapReport:
    """Select the strongest reliable deployable and measure its oracle gap."""

    if not isinstance(config, ProjectConfig):
        raise OracleGapError("oracle gap requires a ProjectConfig")
    if not isinstance(campaign, MatchedPolicyCampaignReport):
        raise OracleGapError("oracle gap requires a matched campaign")
    if not isinstance(density_report, DensityMetricsReport):
        raise OracleGapError("oracle gap requires a density metrics report")
    resolved_hash = config_hash(config)
    if campaign.config_hash != resolved_hash or density_report.config_hash != resolved_hash:
        raise OracleGapError("config hashes differ across oracle-gap inputs")
    if campaign.environment_seed != density_report.environment_seed:
        raise OracleGapError("environment seeds differ across oracle-gap inputs")
    if set(campaign.policies) != set(density_report.policies):
        raise OracleGapError("policy sets differ across oracle-gap inputs")
    missing_campaign = tuple(
        policy for policy in REQUIRED_GAP_POLICIES if policy not in campaign.policies
    )
    missing_metrics = tuple(
        policy for policy in REQUIRED_GAP_POLICIES if policy not in density_report.policies
    )
    if missing_campaign or missing_metrics:
        raise OracleGapError(
            "oracle-gap comparison requires the complete baseline suite",
            context={
                "missing_campaign": missing_campaign,
                "missing_density_metrics": missing_metrics,
            },
        )
    _require_density_report_binding(campaign, density_report)

    density_rows: list[OracleGapDensity] = []
    for block in density_report.densities:
        metrics = _metric_map(density_report, block.density)
        deployable = tuple(metrics[policy] for policy in REQUIRED_DEPLOYABLE_BASELINES)
        oracle = metrics[BASELINE_ORACLE]
        closest = min(
            deployable,
            key=lambda metric: (
                metric.conditional_bootstrap_upper,
                metric.mean_activation_cost,
                metric.mean_reserved_rf_attempts,
                metric.policy,
            ),
        )
        ready = oracle.evaluation_ready and all(metric.evaluation_ready for metric in deployable)
        if not ready:
            density_rows.append(
                OracleGapDensity(
                    density=block.density,
                    status="diagnostic",
                    closest_deployable_policy=closest.policy,
                    feasible_deployable_policies=(),
                    best_deployable_policy=None,
                    oracle_meets_budget=None,
                    estimates=(),
                )
            )
            continue

        feasible = tuple(metric for metric in deployable if metric.meets_miss_budget)
        best = (
            min(
                feasible,
                key=lambda metric: (
                    metric.mean_activation_cost,
                    metric.conditional_bootstrap_upper,
                    metric.mean_reserved_rf_attempts,
                    metric.policy,
                ),
            )
            if feasible
            else None
        )
        oracle_meets = bool(oracle.meets_miss_budget)
        if oracle_meets and best is not None:
            status: GapStatus = "comparable"
            estimates = _paired_estimates(
                campaign,
                split=density_report.split,
                density=block.density,
                deployable_policy=best.policy,
                confidence=density_report.confidence_level,
                replicates=density_report.bootstrap_replicates,
                seed=density_report.bootstrap_seed,
            )
        elif oracle_meets:
            status = "oracle-only-feasible"
            estimates = ()
        elif best is not None:
            status = "oracle-infeasible"
            estimates = ()
        else:
            status = "no-feasible-policy"
            estimates = ()
        density_rows.append(
            OracleGapDensity(
                density=block.density,
                status=status,
                closest_deployable_policy=closest.policy,
                feasible_deployable_policies=tuple(metric.policy for metric in feasible),
                best_deployable_policy=best.policy if best is not None else None,
                oracle_meets_budget=oracle_meets,
                estimates=estimates,
            )
        )

    return OracleGapReport(
        config_hash=resolved_hash,
        environment_seed=campaign.environment_seed,
        split=density_report.split,
        confidence_level=density_report.confidence_level,
        bootstrap_replicates=density_report.bootstrap_replicates,
        bootstrap_seed=density_report.bootstrap_seed,
        densities=tuple(density_rows),
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "ORACLE_GAP_SCHEMA",
    "REQUIRED_DEPLOYABLE_BASELINES",
    "REQUIRED_GAP_POLICIES",
    "GapEstimate",
    "GapStatus",
    "OracleGapDensity",
    "OracleGapError",
    "OracleGapReport",
    "build_oracle_gap_report",
]
