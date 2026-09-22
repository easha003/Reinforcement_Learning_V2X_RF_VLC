"""Per-density reliability and resource summaries for matched policy campaigns."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources
from hybrid_v2x_rl.env.statistics import ClusteredRate, ClusterTally
from hybrid_v2x_rl.mean_field.matched_campaign import MatchedPolicyCampaignReport

DENSITY_METRICS_SCHEMA: Final = "hybrid-rf-vlc-rl.policy-density-metrics.v1"
EvaluationSplit = Literal["train", "validation", "test"]


class DensityMetricsError(HybridV2XError):
    """A density report is invalid or cannot be formed from its campaign."""


def _is_probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


@dataclass(frozen=True, slots=True)
class PolicyDensityMetrics:
    """Reliability, resources, and evidence sufficiency for one policy/density."""

    density: float
    split: EvaluationSplit
    policy: str
    trace_ids: tuple[str, ...]
    source_exhausted: bool
    frames: int
    packets: int
    usable_packets: int
    fallback_packets: int
    pair_episode_clusters: int
    sampled_misses: int
    sampled_miss_rate: float
    sampled_bootstrap_upper: float
    sampled_bootstrap_informative: bool
    conditional_miss_rate: float
    conditional_bootstrap_upper: float
    miss_budget: float
    sufficient_packets: bool
    sufficient_clusters: bool
    evaluation_ready: bool
    meets_miss_budget: bool | None
    mean_activation_cost: float
    mean_reserved_rf_attempts: float
    rf_use_fraction: float
    vlc_use_fraction: float
    duplication_fraction: float
    fallback_fraction: float
    mean_population: float
    mean_pool_utilization: float
    max_pool_utilization: float
    action_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise DensityMetricsError("density must be finite and positive")
        if self.split not in ("train", "validation", "test"):
            raise DensityMetricsError("density split is invalid")
        if not self.policy or not self.trace_ids or len(set(self.trace_ids)) != len(
            self.trace_ids
        ):
            raise DensityMetricsError("policy and unique trace IDs are required")
        integer_fields = (
            "frames",
            "packets",
            "usable_packets",
            "fallback_packets",
            "pair_episode_clusters",
            "sampled_misses",
        )
        if any(
            not isinstance(getattr(self, name), int)
            or isinstance(getattr(self, name), bool)
            or getattr(self, name) < 0
            for name in integer_fields
        ):
            raise DensityMetricsError("density metric counts must be non-negative")
        if self.frames <= 0 or self.packets <= 0 or self.pair_episode_clusters <= 0:
            raise DensityMetricsError("density metrics require frames, packets, and clusters")
        if self.usable_packets + self.fallback_packets != self.packets:
            raise DensityMetricsError("usable and fallback packets must partition packets")
        if self.sampled_misses > self.packets:
            raise DensityMetricsError("sampled misses cannot exceed packets")
        for name in (
            "sampled_miss_rate",
            "sampled_bootstrap_upper",
            "conditional_miss_rate",
            "conditional_bootstrap_upper",
            "miss_budget",
            "rf_use_fraction",
            "vlc_use_fraction",
            "duplication_fraction",
            "fallback_fraction",
        ):
            if not _is_probability(getattr(self, name)):
                raise DensityMetricsError(f"{name} must be a finite probability")
        for name in (
            "mean_activation_cost",
            "mean_reserved_rf_attempts",
            "mean_population",
            "mean_pool_utilization",
            "max_pool_utilization",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise DensityMetricsError(f"{name} must be finite and non-negative")
        boolean_fields = (
            "source_exhausted",
            "sampled_bootstrap_informative",
            "sufficient_packets",
            "sufficient_clusters",
            "evaluation_ready",
        )
        if any(type(getattr(self, name)) is not bool for name in boolean_fields):
            raise DensityMetricsError("density metric flags must be booleans")
        expected_ready = (
            self.source_exhausted
            and self.sufficient_packets
            and self.sufficient_clusters
        )
        if self.evaluation_ready != expected_ready:
            raise DensityMetricsError("evaluation-ready flag does not match evidence")
        if self.meets_miss_budget is not None and type(self.meets_miss_budget) is not bool:
            raise DensityMetricsError("budget verdict must be boolean or unavailable")
        if (self.meets_miss_budget is None) == self.evaluation_ready:
            raise DensityMetricsError(
                "a budget verdict is available exactly when evaluation evidence is ready"
            )
        if (
            not isinstance(self.action_counts, tuple)
            or len(self.action_counts) != len(PolicyAction)
            or any(
                not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                for count in self.action_counts
            )
            or sum(self.action_counts) != self.packets
        ):
            raise DensityMetricsError("action counts must partition policy packets")

    def as_dict(self) -> dict[str, object]:
        return {
            "density_vehicles_per_lane_km": self.density,
            "split": self.split,
            "policy": self.policy,
            "trace_ids": list(self.trace_ids),
            "source_exhausted": self.source_exhausted,
            "frames": self.frames,
            "packets": self.packets,
            "usable_packets": self.usable_packets,
            "fallback_packets": self.fallback_packets,
            "pair_episode_clusters": self.pair_episode_clusters,
            "reliability": {
                "sampled_misses": self.sampled_misses,
                "sampled_miss_rate": self.sampled_miss_rate,
                "sampled_bootstrap_upper": self.sampled_bootstrap_upper,
                "sampled_bootstrap_informative": (
                    self.sampled_bootstrap_informative
                ),
                "conditional_miss_rate": self.conditional_miss_rate,
                "conditional_bootstrap_upper": self.conditional_bootstrap_upper,
                "miss_budget": self.miss_budget,
                "meets_miss_budget": self.meets_miss_budget,
            },
            "evidence": {
                "sufficient_packets": self.sufficient_packets,
                "sufficient_clusters": self.sufficient_clusters,
                "evaluation_ready": self.evaluation_ready,
            },
            "resources": {
                "mean_activation_cost": self.mean_activation_cost,
                "mean_reserved_rf_attempts": self.mean_reserved_rf_attempts,
                "rf_use_fraction": self.rf_use_fraction,
                "vlc_use_fraction": self.vlc_use_fraction,
                "duplication_fraction": self.duplication_fraction,
                "fallback_fraction": self.fallback_fraction,
                "mean_population": self.mean_population,
                "mean_pool_utilization": self.mean_pool_utilization,
                "max_pool_utilization": self.max_pool_utilization,
                "action_counts": {
                    action.label: self.action_counts[int(action)]
                    for action in PolicyAction
                },
            },
        }


@dataclass(frozen=True, slots=True)
class DensityMetricsBlock:
    """All matched policy summaries at one traffic density."""

    density: float
    policies: tuple[PolicyDensityMetrics, ...]

    def __post_init__(self) -> None:
        if not self.policies:
            raise DensityMetricsError("density block requires policy metrics")
        if any(metric.density != self.density for metric in self.policies):
            raise DensityMetricsError("density block contains a mismatched density")
        names = tuple(metric.policy for metric in self.policies)
        if len(names) != len(set(names)):
            raise DensityMetricsError("density block repeats a policy")

    def as_dict(self) -> dict[str, object]:
        return {
            "density_vehicles_per_lane_km": self.density,
            "policies": [metric.as_dict() for metric in self.policies],
        }


@dataclass(frozen=True, slots=True)
class DensityMetricsReport:
    """Versioned per-density report derived from one matched campaign."""

    config_hash: str
    environment_seed: int
    split: EvaluationSplit
    policies: tuple[str, ...]
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    minimum_packets: int
    minimum_clusters: int
    densities: tuple[DensityMetricsBlock, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        if len(self.config_hash) != 64:
            raise DensityMetricsError("density-report config hash must be SHA-256")
        if len(self.policies) < 2 or len(self.policies) != len(set(self.policies)):
            raise DensityMetricsError("density report requires unique compared policies")
        if not 0.5 < self.confidence_level < 1.0:
            raise DensityMetricsError("density-report confidence must lie in (0.5, 1)")
        if self.bootstrap_replicates < 1_000:
            raise DensityMetricsError("density report requires at least 1,000 bootstraps")
        if self.bootstrap_seed < 0:
            raise DensityMetricsError("bootstrap seed must be non-negative")
        density_values = tuple(block.density for block in self.densities)
        if not density_values or density_values != tuple(sorted(set(density_values))):
            raise DensityMetricsError("density blocks must be unique and sorted")
        if any(
            tuple(metric.policy for metric in block.policies) != self.policies
            for block in self.densities
        ):
            raise DensityMetricsError("density blocks must preserve campaign policy order")
        if self.generated_at_utc.tzinfo is None:
            raise DensityMetricsError("density-report timestamp must be timezone-aware")

    @property
    def evaluation_ready(self) -> bool:
        return all(
            metric.evaluation_ready
            for block in self.densities
            for metric in block.policies
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": DENSITY_METRICS_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "config_hash": self.config_hash,
            "environment_seed": self.environment_seed,
            "split": self.split,
            "policies": list(self.policies),
            "confidence_level": self.confidence_level,
            "one_sided_upper_bound": True,
            "interval_method": "trajectory_pair_cluster_bootstrap",
            "matched_bootstrap_across_policies": True,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "minimum_packets_per_policy_density": self.minimum_packets,
            "minimum_pair_episode_clusters": self.minimum_clusters,
            "evaluation_ready": self.evaluation_ready,
            "densities": [block.as_dict() for block in self.densities],
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the density report."""

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


def _policy_density_metrics(
    config: ProjectConfig,
    campaign: MatchedPolicyCampaignReport,
    *,
    split: EvaluationSplit,
    density: float,
    policy: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> PolicyDensityMetrics:
    comparisons = tuple(
        comparison
        for comparison in campaign.comparisons
        if comparison.source.split == split and comparison.source.density == density
    )
    reports = tuple(
        next(report for report in comparison.reports if report.policy == policy)
        for comparison in comparisons
    )
    clusters: dict[str, ClusterTally] = {}
    for report in reports:
        for cluster in report.episode_clusters:
            cluster_id = f"{report.trace_id}/{cluster.pair_id}"
            if cluster_id in clusters:  # pragma: no cover - validated source defense
                raise DensityMetricsError("density report repeats a pair-episode cluster")
            clusters[cluster_id] = ClusterTally(
                packets=cluster.packets,
                misses=cluster.misses,
                expected_misses=cluster.conditional_risk_sum,
                cost=-cluster.reward_sum,
            )
    rate = ClusteredRate(clusters=clusters)
    packets = sum(report.transitions for report in reports)
    frames = sum(report.frames for report in reports)
    action_counts = tuple(
        sum(report.action_counts[index] for report in reports)
        for index in range(len(PolicyAction))
    )
    reserved_rf_attempts = sum(
        action_counts[int(action)] * action_resources(action).reserved_rf_attempts
        for action in PolicyAction
    )
    rf_uses = sum(
        action_counts[int(action)] * int(action_resources(action).uses_rf)
        for action in PolicyAction
    )
    vlc_uses = sum(
        action_counts[int(action)] * int(action_resources(action).uses_vlc)
        for action in PolicyAction
    )
    duplications = sum(
        action_counts[int(action)] * int(action_resources(action).duplicates)
        for action in PolicyAction
    )
    if rate.packets != packets:
        raise DensityMetricsError("episode clusters do not cover density packets")
    if reserved_rf_attempts != sum(report.reserved_rf_attempts for report in reports):
        raise DensityMetricsError("density RF attempts do not reconcile")
    if vlc_uses != sum(report.vlc_activations for report in reports):
        raise DensityMetricsError("density VLC activations do not reconcile")

    source_exhausted = all(report.source_exhausted for report in reports)
    sufficient_packets = packets >= config.evaluation.min_packets_per_policy_density
    sufficient_clusters = (
        rate.cluster_count >= config.evaluation.min_trajectory_pair_clusters
    )
    evaluation_ready = source_exhausted and sufficient_packets and sufficient_clusters
    conditional_upper = rate.expected_upper(
        replicates=bootstrap_replicates,
        confidence=config.evaluation.confidence_level,
        seed=bootstrap_seed,
    )
    sampled_upper = rate.bootstrap_upper(
        replicates=bootstrap_replicates,
        confidence=config.evaluation.confidence_level,
        seed=bootstrap_seed,
        statistic="realized",
    )
    return PolicyDensityMetrics(
        density=density,
        split=split,
        policy=policy,
        trace_ids=tuple(report.trace_id for report in reports),
        source_exhausted=source_exhausted,
        frames=frames,
        packets=packets,
        usable_packets=sum(report.usable_transitions for report in reports),
        fallback_packets=sum(report.fallback_transitions for report in reports),
        pair_episode_clusters=rate.cluster_count,
        sampled_misses=rate.misses,
        sampled_miss_rate=rate.realized_rate,
        sampled_bootstrap_upper=sampled_upper,
        sampled_bootstrap_informative=rate.realized_bound_is_informative,
        conditional_miss_rate=rate.expected_rate,
        conditional_bootstrap_upper=conditional_upper,
        miss_budget=config.service.miss_budget,
        sufficient_packets=sufficient_packets,
        sufficient_clusters=sufficient_clusters,
        evaluation_ready=evaluation_ready,
        meets_miss_budget=(
            conditional_upper <= config.service.miss_budget
            if evaluation_ready
            else None
        ),
        mean_activation_cost=rate.mean_cost,
        mean_reserved_rf_attempts=reserved_rf_attempts / packets,
        rf_use_fraction=rf_uses / packets,
        vlc_use_fraction=vlc_uses / packets,
        duplication_fraction=duplications / packets,
        fallback_fraction=sum(report.fallback_transitions for report in reports)
        / packets,
        mean_population=packets / frames,
        mean_pool_utilization=math.fsum(
            report.pool_utilization_sum for report in reports
        )
        / frames,
        max_pool_utilization=max(report.max_pool_utilization for report in reports),
        action_counts=action_counts,
    )


def build_density_metrics_report(
    config: ProjectConfig,
    campaign: MatchedPolicyCampaignReport,
    *,
    split: EvaluationSplit = "test",
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int = 0,
) -> DensityMetricsReport:
    """Aggregate a matched campaign by traffic density and policy."""

    if not isinstance(config, ProjectConfig):
        raise DensityMetricsError("density metrics require a ProjectConfig")
    if not isinstance(campaign, MatchedPolicyCampaignReport):
        raise DensityMetricsError("density metrics require a matched campaign")
    if campaign.config_hash != config_hash(config):
        raise DensityMetricsError("campaign and density-report config hashes differ")
    if split not in ("train", "validation", "test"):
        raise DensityMetricsError("density-report split is invalid")
    replicates = (
        config.evaluation.bootstrap_replicates
        if bootstrap_replicates is None
        else bootstrap_replicates
    )
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1_000:
        raise DensityMetricsError("at least 1,000 bootstrap replicates are required")
    if (
        not isinstance(bootstrap_seed, int)
        or isinstance(bootstrap_seed, bool)
        or bootstrap_seed < 0
    ):
        raise DensityMetricsError("bootstrap seed must be a non-negative integer")
    selected = tuple(
        comparison
        for comparison in campaign.comparisons
        if comparison.source.split == split
    )
    if not selected:
        raise DensityMetricsError("campaign contains no traces for the requested split")
    densities = tuple(sorted({comparison.source.density for comparison in selected}))
    blocks = tuple(
        DensityMetricsBlock(
            density=density,
            policies=tuple(
                _policy_density_metrics(
                    config,
                    campaign,
                    split=split,
                    density=density,
                    policy=policy,
                    bootstrap_replicates=replicates,
                    bootstrap_seed=bootstrap_seed,
                )
                for policy in campaign.policies
            ),
        )
        for density in densities
    )
    return DensityMetricsReport(
        config_hash=config_hash(config),
        environment_seed=campaign.environment_seed,
        split=split,
        policies=campaign.policies,
        confidence_level=config.evaluation.confidence_level,
        bootstrap_replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_packets=config.evaluation.min_packets_per_policy_density,
        minimum_clusters=config.evaluation.min_trajectory_pair_clusters,
        densities=blocks,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "DENSITY_METRICS_SCHEMA",
    "DensityMetricsBlock",
    "DensityMetricsError",
    "DensityMetricsReport",
    "EvaluationSplit",
    "PolicyDensityMetrics",
    "build_density_metrics_report",
]
