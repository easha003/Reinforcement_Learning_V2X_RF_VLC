"""Matched-split, matched-randomness policy comparison campaigns.

Every policy visits the configuration-authoritative traces in the same order,
with one environment seed and the same diagnostic cutoff.  The rollout report
fingerprints every complete pre-action packet tape, so matching is verified
from the actual draws rather than inferred only from equal seed arguments.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    DeterministicRolloutReport,
    run_policy_rollout_with_state,
)
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource, TraceCatalog
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizationState
from hybrid_v2x_rl.mean_field.policy_interface import PopulationPolicy
from hybrid_v2x_rl.mean_field.seeding import EnvironmentSeedState

MATCHED_CAMPAIGN_SCHEMA: Final = "hybrid-rf-vlc-rl.matched-policy-campaign.v1"
_SPLITS: Final = ("train", "validation", "test")
_MATCHED_STRUCTURE_FIELDS: Final = (
    "requested_max_frames",
    "available_frames",
    "frames",
    "source_exhausted",
    "nonempty_frames",
    "transitions",
    "usable_transitions",
    "fallback_transitions",
    "births",
    "natural_terminations",
    "internal_truncations",
    "trace_end_truncations",
    "max_population",
)


class MatchedCampaignError(HybridV2XError):
    """A comparison campaign does not use identical splits or randomness."""


@dataclass(frozen=True, slots=True)
class MatchedTraceComparison:
    """All policy reports for one immutable trace and packet-tape realization."""

    source: FrameTraceSource
    reports: tuple[DeterministicRolloutReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, FrameTraceSource):
            raise MatchedCampaignError("trace comparison requires a FrameTraceSource")
        if len(self.reports) < 2:
            raise MatchedCampaignError("matched comparison requires at least two policies")
        if any(not isinstance(report, DeterministicRolloutReport) for report in self.reports):
            raise MatchedCampaignError("trace comparison contains an invalid report")
        names = tuple(report.policy for report in self.reports)
        if len(names) != len(set(names)):
            raise MatchedCampaignError("trace comparison repeats a policy name")
        for report in self.reports:
            if report.trace_id != self.source.trace_id or report.split != self.source.split:
                raise MatchedCampaignError(
                    "policy report does not belong to the comparison trace",
                    context={
                        "trace_id": self.source.trace_id,
                        "policy": report.policy,
                        "report_trace_id": report.trace_id,
                        "report_split": report.split,
                    },
                )
        reference = self.reports[0]
        mismatches: dict[str, object] = {}
        for report in self.reports[1:]:
            fields = tuple(
                field
                for field in _MATCHED_STRUCTURE_FIELDS
                if getattr(report, field) != getattr(reference, field)
            )
            if fields:
                mismatches[report.policy] = fields
        if mismatches:
            raise MatchedCampaignError(
                "policies did not replay the same trace structure",
                context={"trace_id": self.source.trace_id, "mismatches": mismatches},
            )
        reference_clusters = tuple(
            (cluster.pair_id, cluster.packets)
            for cluster in reference.episode_clusters
        )
        cluster_mismatches = tuple(
            report.policy
            for report in self.reports[1:]
            if tuple(
                (cluster.pair_id, cluster.packets)
                for cluster in report.episode_clusters
            )
            != reference_clusters
        )
        if cluster_mismatches:
            raise MatchedCampaignError(
                "policies did not replay identical pair-episode clusters",
                context={
                    "trace_id": self.source.trace_id,
                    "policies": cluster_mismatches,
                },
            )
        seeds = {report.environment_seed for report in self.reports}
        if len(seeds) != 1:
            raise MatchedCampaignError(
                "policies did not use one matched environment seed",
                context={"trace_id": self.source.trace_id, "seeds": tuple(sorted(seeds))},
            )
        tape_fingerprints = {
            report.matched_tape_fingerprint for report in self.reports
        }
        if len(tape_fingerprints) != 1:
            raise MatchedCampaignError(
                "policy packet tapes do not match",
                context={"trace_id": self.source.trace_id},
            )

    @property
    def matched_tape_fingerprint(self) -> str:
        return self.reports[0].matched_tape_fingerprint

    def as_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.source.trace_id,
            "split": self.source.split,
            "density": self.source.density,
            "replicate": self.source.replicate,
            "matched_tape_fingerprint": self.matched_tape_fingerprint,
            "policies": [report.as_dict() for report in self.reports],
        }


@dataclass(frozen=True, slots=True)
class PolicyNormalizationCheckpoint:
    """Final frozen train-only normalization state for one compared policy."""

    policy: str
    state: ObservationNormalizationState

    def __post_init__(self) -> None:
        if not isinstance(self.policy, str) or not self.policy:
            raise MatchedCampaignError("normalization checkpoint policy must be non-empty")
        if not isinstance(self.state, ObservationNormalizationState):
            raise MatchedCampaignError("campaign checkpoint requires normalization state")
        if not self.state.frozen:
            raise MatchedCampaignError("final campaign normalization must be frozen")

    def as_dict(self) -> dict[str, object]:
        return {"policy": self.policy, "state": self.state.as_dict()}


@dataclass(frozen=True, slots=True)
class MatchedPolicyCampaignReport:
    """Auditable comparisons over the exact configured train/validation/test catalog."""

    config_hash: str
    environment_seed: int
    policy_seed: int
    requested_max_frames: int | None
    policies: tuple[str, ...]
    comparisons: tuple[MatchedTraceComparison, ...]
    normalization_checkpoints: tuple[PolicyNormalizationCheckpoint, ...]
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.config_hash, str) or len(self.config_hash) != 64:
            raise MatchedCampaignError("campaign config_hash must be SHA-256")
        if len(self.policies) < 2 or len(self.policies) != len(set(self.policies)):
            raise MatchedCampaignError("campaign requires unique names for at least two policies")
        if not self.comparisons:
            raise MatchedCampaignError("campaign requires at least one trace comparison")
        expected = self.policies
        for comparison in self.comparisons:
            if tuple(report.policy for report in comparison.reports) != expected:
                raise MatchedCampaignError(
                    "comparison policy order differs from the campaign contract",
                    context={"trace_id": comparison.source.trace_id},
                )
        checkpoint_names = tuple(item.policy for item in self.normalization_checkpoints)
        if checkpoint_names != expected:
            raise MatchedCampaignError(
                "normalization checkpoints do not align with campaign policies"
            )
        if self.generated_at_utc.tzinfo is None:
            raise MatchedCampaignError("campaign timestamp must be timezone-aware")

    @property
    def passed(self) -> bool:
        return bool(self.comparisons)

    def as_dict(self) -> dict[str, object]:
        split_membership = {
            split: [
                item.source.trace_id
                for item in self.comparisons
                if item.source.split == split
            ]
            for split in _SPLITS
        }
        return {
            "schema": MATCHED_CAMPAIGN_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "passed": self.passed,
            "config_hash": self.config_hash,
            "environment_seed": self.environment_seed,
            "policy_seed": self.policy_seed,
            "requested_max_frames": self.requested_max_frames,
            "policies": list(self.policies),
            "trace_membership": split_membership,
            "trace_count": len(self.comparisons),
            "comparisons": [item.as_dict() for item in self.comparisons],
            "normalization_checkpoints": [
                item.as_dict() for item in self.normalization_checkpoints
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the matched campaign report."""

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


def _configured_membership(config: ProjectConfig) -> tuple[tuple[str, str], ...]:
    return tuple(
        (split, trace_id)
        for split in _SPLITS
        for trace_id in getattr(config.environment.splits, split)
    )


def run_matched_policy_campaign(
    config: ProjectConfig,
    catalog: TraceCatalog,
    *,
    policies: tuple[PopulationPolicy, ...],
    environment_seed: int | None = None,
    policy_seed: int = 0,
    max_frames: int | None = None,
) -> MatchedPolicyCampaignReport:
    """Replay every policy on the exact same configured traces and packet tapes."""

    if not isinstance(config, ProjectConfig):
        raise MatchedCampaignError("matched campaign requires a ProjectConfig")
    if not isinstance(catalog, TraceCatalog):
        raise MatchedCampaignError("matched campaign requires a TraceCatalog")
    if not config.environment.matched_random_tapes:
        raise MatchedCampaignError("environment must enable matched_random_tapes")
    if not config.evaluation.matched_across_policies:
        raise MatchedCampaignError("evaluation must require matched_across_policies")
    if max_frames is not None and (
        not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0
    ):
        raise MatchedCampaignError("max_frames must be positive or None")
    if len(policies) < 2 or any(not isinstance(policy, PopulationPolicy) for policy in policies):
        raise MatchedCampaignError("campaign requires at least two PopulationPolicy objects")
    policy_names = tuple(policy.name for policy in policies)
    if any(not isinstance(name, str) or not name for name in policy_names):
        raise MatchedCampaignError("campaign policy names must be non-empty")
    if len(policy_names) != len(set(policy_names)):
        raise MatchedCampaignError("campaign policy names must be unique")

    expected_membership = _configured_membership(config)
    actual_membership = tuple(
        (source.split, source.trace_id) for source in catalog.sources
    )
    if actual_membership != expected_membership:
        raise MatchedCampaignError(
            "trace catalog does not exactly match configured split membership",
            context={
                "expected": expected_membership,
                "actual": actual_membership,
            },
        )
    training_sources = catalog.for_split("train")
    if not training_sources:
        raise MatchedCampaignError("campaign requires at least one training trace")
    last_training_trace = training_sources[-1].trace_id

    active_seed = EnvironmentSeedState.from_config(
        config,
        reset_seed=environment_seed,
    ).active_root_seed
    states: dict[str, ObservationNormalizationState | None] = {
        name: None for name in policy_names
    }
    comparisons: list[MatchedTraceComparison] = []
    for source in catalog.sources:
        reports: list[DeterministicRolloutReport] = []
        for policy in policies:
            result = run_policy_rollout_with_state(
                config,
                source,
                policy=policy,
                environment_seed=active_seed,
                policy_seed=policy_seed,
                max_frames=max_frames,
                normalization_state=states[policy.name],
                freeze_normalization_at_end=(
                    source.trace_id == last_training_trace
                ),
            )
            states[policy.name] = result.normalization_state
            reports.append(result.report)
        comparisons.append(
            MatchedTraceComparison(source=source, reports=tuple(reports))
        )

    checkpoints = tuple(
        PolicyNormalizationCheckpoint(policy=name, state=state)
        for name in policy_names
        if (state := states[name]) is not None
    )
    return MatchedPolicyCampaignReport(
        config_hash=config_hash(config),
        environment_seed=active_seed,
        policy_seed=policy_seed,
        requested_max_frames=max_frames,
        policies=policy_names,
        comparisons=tuple(comparisons),
        normalization_checkpoints=checkpoints,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "MATCHED_CAMPAIGN_SCHEMA",
    "MatchedCampaignError",
    "MatchedPolicyCampaignReport",
    "MatchedTraceComparison",
    "PolicyNormalizationCheckpoint",
    "run_matched_policy_campaign",
]
