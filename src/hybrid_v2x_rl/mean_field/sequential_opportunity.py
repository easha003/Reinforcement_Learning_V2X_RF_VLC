"""Phase 6 evidence for history and population-coupling opportunities."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config.hashing import config_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.enums import Action, Link
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.env.cache import ColumnPlan, TransitionCache
from hybrid_v2x_rl.mean_field.baselines import CONTEXTUAL_FEATURES
from hybrid_v2x_rl.mean_field.frame_cache import PopulationFrameCacheReader
from hybrid_v2x_rl.mean_field.rf_pool import RFPoolDemand, RFPoolModel
from hybrid_v2x_rl.observation.builder import (
    LINK_FEATURES,
    ObservationSchema,
)
from hybrid_v2x_rl.observation.link_state import LinkStateTracker

SEQUENTIAL_OPPORTUNITY_SCHEMA: Final = "hybrid-rf-vlc-rl.sequential-opportunity.v1"
_BEHAVIOR_CYCLE: Final = (Action.RF, Action.VLC, Action.DUP)
_LINK_NAMES: Final = ("rf", "vlc")
OpportunityStatus = Literal["diagnostic", "opportunity", "no-detected-opportunity"]
GateDecision = Literal["go", "no-go", "insufficient-evidence"]


class SequentialOpportunityError(HybridV2XError):
    """Sequential-opportunity evidence is incomplete, invalid, or unbound."""


def _probability(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


@dataclass(frozen=True, slots=True)
class HistoryLinkEstimate:
    """Held-out predictive value of action-dependent history for one link."""

    link: str
    contextual_brier: float
    history_brier: float
    brier_gain: float
    confidence_lower: float
    confidence_upper: float

    def __post_init__(self) -> None:
        if self.link not in _LINK_NAMES:
            raise SequentialOpportunityError("history estimate link is invalid")
        if not _probability(self.contextual_brier) or not _probability(self.history_brier):
            raise SequentialOpportunityError("Brier scores must be probabilities")
        values = (self.brier_gain, self.confidence_lower, self.confidence_upper)
        if any(not math.isfinite(value) for value in values):
            raise SequentialOpportunityError("history gains must be finite")
        if not math.isclose(
            self.brier_gain,
            self.contextual_brier - self.history_brier,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SequentialOpportunityError(
                "history Brier gain must equal contextual minus history score"
            )
        if self.confidence_lower > self.confidence_upper:
            raise SequentialOpportunityError("history confidence interval is reversed")

    @property
    def positive_interval(self) -> bool:
        return self.confidence_lower > 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "link": self.link,
            "contextual_brier": self.contextual_brier,
            "history_brier": self.history_brier,
            "brier_gain_contextual_minus_history": self.brier_gain,
            "confidence_lower": self.confidence_lower,
            "confidence_upper": self.confidence_upper,
            "positive_interval": self.positive_interval,
        }


@dataclass(frozen=True, slots=True)
class HistoryOpportunity:
    """Train-only fit and held-out test of causal link-history information."""

    status: OpportunityStatus
    evidence_ready: bool
    opportunity_detected: bool | None
    train_sources: tuple[str, ...]
    evaluation_sources: tuple[str, ...]
    missing_train_sources: tuple[str, ...]
    missing_evaluation_sources: tuple[str, ...]
    training_rows: int
    evaluation_rows: int
    evaluation_clusters: int
    minimum_evaluation_rows: int
    minimum_evaluation_clusters: int
    behavior_action_counts: tuple[tuple[str, int], ...]
    estimates: tuple[HistoryLinkEstimate, ...]

    def __post_init__(self) -> None:
        if self.status not in (
            "diagnostic",
            "opportunity",
            "no-detected-opportunity",
        ):
            raise SequentialOpportunityError("history status is invalid")
        if type(self.evidence_ready) is not bool:
            raise SequentialOpportunityError("history readiness must be boolean")
        if self.opportunity_detected is not None and type(self.opportunity_detected) is not bool:
            raise SequentialOpportunityError("history verdict must be boolean or null")
        counts = (
            self.training_rows,
            self.evaluation_rows,
            self.evaluation_clusters,
            self.minimum_evaluation_rows,
            self.minimum_evaluation_clusters,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in counts
        ):
            raise SequentialOpportunityError("history evidence counts must be positive")
        expected_ready = (
            not self.missing_train_sources
            and not self.missing_evaluation_sources
            and self.evaluation_rows >= self.minimum_evaluation_rows
            and self.evaluation_clusters >= self.minimum_evaluation_clusters
        )
        if self.evidence_ready != expected_ready:
            raise SequentialOpportunityError("history readiness does not match evidence")
        if self.evidence_ready != (self.opportunity_detected is not None):
            raise SequentialOpportunityError(
                "history verdict is available exactly when evidence is ready"
            )
        expected_status: OpportunityStatus
        if not self.evidence_ready:
            expected_status = "diagnostic"
        elif self.opportunity_detected:
            expected_status = "opportunity"
        else:
            expected_status = "no-detected-opportunity"
        if self.status != expected_status:
            raise SequentialOpportunityError("history status and verdict disagree")
        if tuple(estimate.link for estimate in self.estimates) != _LINK_NAMES:
            raise SequentialOpportunityError("history estimates must cover RF then VLC")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "evidence_ready": self.evidence_ready,
            "opportunity_detected": self.opportunity_detected,
            "claim_rule": (
                "at least one held-out link Brier-gain cluster-bootstrap interval "
                "is strictly above zero"
            ),
            "train_sources": list(self.train_sources),
            "evaluation_sources": list(self.evaluation_sources),
            "missing_train_sources": list(self.missing_train_sources),
            "missing_evaluation_sources": list(self.missing_evaluation_sources),
            "training_rows": self.training_rows,
            "evaluation_rows": self.evaluation_rows,
            "evaluation_clusters": self.evaluation_clusters,
            "minimum_evaluation_rows": self.minimum_evaluation_rows,
            "minimum_evaluation_clusters": self.minimum_evaluation_clusters,
            "behavior_policy": "within-episode RF/VLC/DUP cycle",
            "behavior_action_counts": dict(self.behavior_action_counts),
            "contextual_features": list(CONTEXTUAL_FEATURES),
            "history_features": sorted(LINK_FEATURES),
            "estimates": [estimate.as_dict() for estimate in self.estimates],
        }


@dataclass(frozen=True, slots=True)
class PopulationAttemptOpportunity:
    """Feasibility flip for one fixed focal RF reservation level."""

    rf_attempts: int
    first_flip_population: int | None
    flip_frames: int
    flip_frame_fraction: float
    median_population: int
    low_demand_risk_at_median: float
    all_rf_risk_at_median: float

    def __post_init__(self) -> None:
        if not 1 <= self.rf_attempts <= 4:
            raise SequentialOpportunityError("RF attempt probe is outside [1, 4]")
        if self.first_flip_population is not None and self.first_flip_population < 1:
            raise SequentialOpportunityError("first flip population must be positive")
        if self.flip_frames < 0 or self.median_population < 1:
            raise SequentialOpportunityError("population probe counts are invalid")
        if not _probability(self.flip_frame_fraction):
            raise SequentialOpportunityError("flip frame fraction is invalid")
        if not _probability(self.low_demand_risk_at_median) or not _probability(
            self.all_rf_risk_at_median
        ):
            raise SequentialOpportunityError("population risks must be probabilities")

    @property
    def decision_relevant(self) -> bool:
        return self.flip_frames > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "rf_attempts": self.rf_attempts,
            "first_flip_population": self.first_flip_population,
            "flip_frames": self.flip_frames,
            "flip_frame_fraction": self.flip_frame_fraction,
            "median_population": self.median_population,
            "low_demand_packet_risk_at_median": self.low_demand_risk_at_median,
            "all_rf_packet_risk_at_median": self.all_rf_risk_at_median,
            "decision_relevant": self.decision_relevant,
        }


@dataclass(frozen=True, slots=True)
class PopulationDensityOpportunity:
    """Observed population support and RF externality at one density."""

    density: float
    frames: int
    minimum_population: int
    median_population: int
    maximum_population: int
    attempts: tuple[PopulationAttemptOpportunity, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise SequentialOpportunityError("population density must be positive")
        if self.frames < 1 or not (
            1 <= self.minimum_population <= self.median_population <= self.maximum_population
        ):
            raise SequentialOpportunityError("population support is invalid")
        if tuple(item.rf_attempts for item in self.attempts) != tuple(
            range(1, len(self.attempts) + 1)
        ):
            raise SequentialOpportunityError("population attempt probes are incomplete")

    @property
    def opportunity_detected(self) -> bool:
        return any(item.decision_relevant for item in self.attempts)

    def as_dict(self) -> dict[str, object]:
        return {
            "density_vehicles_per_lane_km": self.density,
            "nonempty_frames": self.frames,
            "minimum_population": self.minimum_population,
            "median_population": self.median_population,
            "maximum_population": self.maximum_population,
            "opportunity_detected": self.opportunity_detected,
            "attempts": [item.as_dict() for item in self.attempts],
        }


@dataclass(frozen=True, slots=True)
class PopulationOpportunity:
    """Held-out frame evidence that current joint RF demand changes feasibility."""

    status: OpportunityStatus
    evidence_ready: bool
    opportunity_detected: bool | None
    evaluation_sources: tuple[str, ...]
    missing_evaluation_sources: tuple[str, ...]
    densities: tuple[PopulationDensityOpportunity, ...]

    def __post_init__(self) -> None:
        expected_ready = not self.missing_evaluation_sources
        if self.evidence_ready != expected_ready:
            raise SequentialOpportunityError("population readiness does not match sources")
        if self.evidence_ready != (self.opportunity_detected is not None):
            raise SequentialOpportunityError(
                "population verdict is available exactly when evidence is ready"
            )
        detected = any(row.opportunity_detected for row in self.densities)
        if self.evidence_ready and self.opportunity_detected != detected:
            raise SequentialOpportunityError("population verdict does not match probes")
        expected_status: OpportunityStatus
        if not self.evidence_ready:
            expected_status = "diagnostic"
        elif detected:
            expected_status = "opportunity"
        else:
            expected_status = "no-detected-opportunity"
        if self.status != expected_status:
            raise SequentialOpportunityError("population status and verdict disagree")

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "evidence_ready": self.evidence_ready,
            "opportunity_detected": self.opportunity_detected,
            "claim_rule": (
                "under fixed focal propagation and action, observed population "
                "support contains frames where changing only other agents' RF "
                "reservations crosses the miss budget"
            ),
            "evaluation_sources": list(self.evaluation_sources),
            "missing_evaluation_sources": list(self.missing_evaluation_sources),
            "counterfactuals": {
                "low_demand": "focal RF-n; every other pair VLC-only",
                "high_demand": "every pair RF-n",
                "fixed": "population, focal action, and optimistic RF decoding risk",
            },
            "densities": [row.as_dict() for row in self.densities],
        }


@dataclass(frozen=True, slots=True)
class SequentialOpportunityReport:
    """Versioned Phase 6 opportunity evidence and PPO gate decision."""

    config_hash: str
    miss_budget: float
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    history: HistoryOpportunity
    population: PopulationOpportunity
    gate_decision: GateDecision
    generated_at_utc: datetime

    def __post_init__(self) -> None:
        if len(self.config_hash) != 64 or not _probability(self.miss_budget):
            raise SequentialOpportunityError("report configuration identity is invalid")
        if not 0.5 < self.confidence_level < 1.0:
            raise SequentialOpportunityError("report confidence must lie in (0.5, 1)")
        if self.bootstrap_replicates < 1_000 or self.bootstrap_seed < 0:
            raise SequentialOpportunityError("report bootstrap settings are invalid")
        evidence = (
            self.history.opportunity_detected,
            self.population.opportunity_detected,
        )
        expected: GateDecision
        if True in evidence:
            expected = "go"
        elif all(value is not None for value in evidence):
            expected = "no-go"
        else:
            expected = "insufficient-evidence"
        if self.gate_decision != expected:
            raise SequentialOpportunityError("PPO gate does not match opportunity evidence")
        if self.generated_at_utc.tzinfo is None:
            raise SequentialOpportunityError("report timestamp must be timezone-aware")

    def as_dict(self) -> dict[str, object]:
        basis: list[str] = []
        if self.history.opportunity_detected is True:
            basis.append("causal link history improves held-out risk prediction")
        if self.population.opportunity_detected is True:
            basis.append("joint RF demand changes action feasibility on held-out frames")
        return {
            "schema": SEQUENTIAL_OPPORTUNITY_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "config_hash": self.config_hash,
            "miss_budget": self.miss_budget,
            "confidence_level": self.confidence_level,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "gate_decision": self.gate_decision,
            "gate_basis": basis,
            "gate_scope": (
                "pre-training opportunity gate, not evidence that PPO outperforms "
                "the strongest feasible deployable baseline"
            ),
            "history": self.history.as_dict(),
            "population_coupling": self.population.as_dict(),
        }

    def write_json(self, path: str | Path) -> Path:
        """Atomically persist the opportunity report."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(self.as_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
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
class _HistoryRows:
    context: NDArray[np.float64]
    context_and_history: NDArray[np.float64]
    targets: NDArray[np.float64]
    clusters: NDArray[np.str_]
    sources: tuple[str, ...]
    action_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _RidgeModel:
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    weights: NDArray[np.float64]


def _cache_identity(
    cache: TransitionCache,
    *,
    expected_split: str,
    expected_hash: str,
) -> str:
    split = cache.manifest.get("split")
    trace = cache.manifest.get("trace")
    if split != expected_split or not isinstance(trace, str) or not trace:
        raise SequentialOpportunityError("transition cache split or trace identity is invalid")
    if cache.manifest.get("config_hash") != expected_hash:
        raise SequentialOpportunityError("transition cache was built for another configuration")
    return trace


def _history_rows(
    paths: tuple[Path, ...],
    *,
    schema: ObservationSchema,
    split: str,
    expected_hash: str,
) -> _HistoryRows:
    if not paths:
        raise SequentialOpportunityError(f"{split} history caches are required")
    plan = ColumnPlan.from_schema(schema)
    columns = schema.columns
    try:
        contextual_indices = tuple(columns.index(name) for name in CONTEXTUAL_FEATURES)
    except ValueError as error:  # pragma: no cover - config validation guards this
        raise SequentialOpportunityError("observation schema lacks contextual features") from error
    history_indices = tuple(
        index
        for index, name in enumerate(columns)
        if name.split("[", maxsplit=1)[0] in LINK_FEATURES
    )
    contexts: list[NDArray[np.float64]] = []
    enhanced: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    clusters: list[str] = []
    sources: list[str] = []
    actions: Counter[str] = Counter()

    for path in paths:
        cache = TransitionCache.load(path, schema=schema, expected_config_hash=expected_hash)
        trace_id = _cache_identity(
            cache,
            expected_split=split,
            expected_hash=expected_hash,
        )
        if trace_id in sources:
            raise SequentialOpportunityError("history caches repeat a trace")
        sources.append(trace_id)
        trackers: dict[int, LinkStateTracker] = {}
        episode_steps: dict[int, int] = {}
        for row_index in range(cache.packets):
            episode = int(cache.episode[row_index])
            tracker = trackers.setdefault(
                episode,
                LinkStateTracker(schema.history_packets),
            )
            step = episode_steps.get(episode, 0)
            now_s = float(cache.time_s[row_index])
            full = plan.assemble(
                np.asarray(cache.trace[row_index], dtype=np.float64),
                tracker,
                now_s,
            ).astype(np.float64, copy=False)
            if tracker.packets_seen > 0:
                contexts.append(full[list(contextual_indices)])
                enhanced.append(full[list((*contextual_indices, *history_indices))])
                targets.append(np.asarray(cache.risk[row_index], dtype=np.float64))
                clusters.append(f"{trace_id}/{episode}")

            # The behavior action depends only on within-episode step, never on
            # the risk label, trace identity, or episode identity.
            action = _BEHAVIOR_CYCLE[step % len(_BEHAVIOR_CYCLE)]
            actions[action.name] += 1
            used: tuple[tuple[Link, int], ...]
            if action is Action.RF:
                used = ((Link.RF, 0),)
                delivered = bool(cache.delivered[row_index, 0])
            elif action is Action.VLC:
                used = ((Link.VLC, 1),)
                delivered = bool(cache.delivered[row_index, 1])
            else:
                used = ((Link.RF, 0), (Link.VLC, 1))
                delivered = bool(cache.delivered[row_index, 0] or cache.delivered[row_index, 1])
            tracker.record(
                action=action,
                at_s=now_s,
                delivered=delivered,
                measurements={link: float(cache.quality[row_index, leg]) for link, leg in used},
            )
            episode_steps[episode] = step + 1

    if not contexts:
        raise SequentialOpportunityError("history caches contain no revisited episodes")
    return _HistoryRows(
        context=np.vstack(contexts),
        context_and_history=np.vstack(enhanced),
        targets=np.vstack(targets),
        clusters=np.asarray(clusters, dtype=np.str_),
        sources=tuple(sorted(sources)),
        action_counts=tuple(sorted(actions.items())),
    )


def _fit_ridge(
    features: NDArray[np.float64],
    targets: NDArray[np.float64],
    *,
    ridge: float,
) -> _RidgeModel:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-9] = 1.0
    design = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    clipped = np.clip(targets, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ logits)
    return _RidgeModel(mean=mean, scale=scale, weights=weights)


def _predict(
    model: _RidgeModel,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    design = np.column_stack(((features - model.mean) / model.scale, np.ones(len(features))))
    logits = np.clip(design @ model.weights, -40.0, 40.0)
    return np.asarray(1.0 / (1.0 + np.exp(-logits)), dtype=np.float64)


def measure_history_opportunity(
    config: ProjectConfig,
    *,
    training_cache_paths: tuple[Path, ...],
    evaluation_cache_paths: tuple[Path, ...],
    confidence_level: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    minimum_evaluation_rows: int,
    minimum_evaluation_clusters: int,
    ridge: float = 1e-3,
) -> HistoryOpportunity:
    """Fit on training caches and test history value on untouched caches."""

    if not math.isfinite(ridge) or ridge <= 0.0:
        raise SequentialOpportunityError("history ridge must be finite and positive")
    schema = ObservationSchema(
        features=tuple(config.observation.features),
        history_packets=config.observation.history_packets,
    )
    resolved_hash = config_hash(config)
    train = _history_rows(
        training_cache_paths,
        schema=schema,
        split="train",
        expected_hash=resolved_hash,
    )
    evaluation = _history_rows(
        evaluation_cache_paths,
        schema=schema,
        split="test",
        expected_hash=resolved_hash,
    )
    expected_train = tuple(sorted(config.environment.splits.train))
    expected_test = tuple(sorted(config.environment.splits.test))
    unexpected_train = tuple(name for name in train.sources if name not in expected_train)
    unexpected_test = tuple(name for name in evaluation.sources if name not in expected_test)
    if unexpected_train or unexpected_test:
        raise SequentialOpportunityError(
            "history caches contain sources outside the configured splits",
            context={
                "unexpected_train": unexpected_train,
                "unexpected_test": unexpected_test,
            },
        )
    missing_train = tuple(name for name in expected_train if name not in train.sources)
    missing_test = tuple(name for name in expected_test if name not in evaluation.sources)
    unique_clusters = np.unique(evaluation.clusters)
    ready = (
        not missing_train
        and not missing_test
        and len(evaluation.targets) >= minimum_evaluation_rows
        and len(unique_clusters) >= minimum_evaluation_clusters
    )
    alpha = (1.0 - confidence_level) / 2.0
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(
        0,
        len(unique_clusters),
        size=(bootstrap_replicates, len(unique_clusters)),
    )
    estimates: list[HistoryLinkEstimate] = []
    for leg, name in enumerate(_LINK_NAMES):
        contextual = _fit_ridge(train.context, train.targets[:, leg], ridge=ridge)
        history = _fit_ridge(
            train.context_and_history,
            train.targets[:, leg],
            ridge=ridge,
        )
        contextual_error = (
            _predict(contextual, evaluation.context) - evaluation.targets[:, leg]
        ) ** 2
        history_error = (
            _predict(history, evaluation.context_and_history) - evaluation.targets[:, leg]
        ) ** 2
        differences = contextual_error - history_error
        cluster_sums = np.asarray(
            [differences[evaluation.clusters == cluster].sum() for cluster in unique_clusters],
            dtype=np.float64,
        )
        cluster_counts = np.asarray(
            [np.count_nonzero(evaluation.clusters == cluster) for cluster in unique_clusters],
            dtype=np.float64,
        )
        sampled = cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
        lower, upper = np.quantile(sampled, (alpha, 1.0 - alpha))
        estimates.append(
            HistoryLinkEstimate(
                link=name,
                contextual_brier=float(contextual_error.mean()),
                history_brier=float(history_error.mean()),
                brier_gain=float(differences.mean()),
                confidence_lower=float(lower),
                confidence_upper=float(upper),
            )
        )
    detected = any(estimate.positive_interval for estimate in estimates) if ready else None
    status: OpportunityStatus = (
        "diagnostic" if not ready else "opportunity" if detected else "no-detected-opportunity"
    )
    combined_actions = Counter(dict(train.action_counts))
    combined_actions.update(dict(evaluation.action_counts))
    return HistoryOpportunity(
        status=status,
        evidence_ready=ready,
        opportunity_detected=detected,
        train_sources=train.sources,
        evaluation_sources=evaluation.sources,
        missing_train_sources=missing_train,
        missing_evaluation_sources=missing_test,
        training_rows=len(train.targets),
        evaluation_rows=len(evaluation.targets),
        evaluation_clusters=len(unique_clusters),
        minimum_evaluation_rows=minimum_evaluation_rows,
        minimum_evaluation_clusters=minimum_evaluation_clusters,
        behavior_action_counts=tuple(sorted(combined_actions.items())),
        estimates=tuple(estimates),
    )


def load_test_populations(
    config: ProjectConfig,
    frame_cache_root: str | Path,
) -> tuple[dict[float, tuple[int, ...]], tuple[str, ...], tuple[str, ...]]:
    """Verify held-out frame caches and return nonempty population counts."""

    root = Path(frame_cache_root)
    expected = tuple(sorted(config.environment.splits.test))
    found: list[str] = []
    grouped: dict[float, list[int]] = {}
    resolved_hash = config_hash(config)
    for trace_id in expected:
        path = root / trace_id
        if not path.exists():
            continue
        reader = PopulationFrameCacheReader(path)
        reader.verify()
        report = reader.summary["report"]
        if (
            reader.artifact.manifest.config_hash != resolved_hash
            or reader.summary.get("split") != "test"
            or not isinstance(report, dict)
            or report.get("trace_id") != trace_id
        ):
            raise SequentialOpportunityError(
                "frame cache is not bound to the configured test campaign"
            )
        density = float(report["density"])
        counts = grouped.setdefault(density, [])
        counts.extend(
            cast(int, row["active_pairs"])
            for row in reader.iter_frame_rows()
            if cast(int, row["active_pairs"]) > 0
        )
        found.append(trace_id)
    missing = tuple(trace_id for trace_id in expected if trace_id not in found)
    if not grouped:
        raise SequentialOpportunityError("no held-out frame caches were found")
    return (
        {density: tuple(values) for density, values in grouped.items()},
        tuple(found),
        missing,
    )


def _counterfactual_packet_risk(
    model: RFPoolModel,
    *,
    population: int,
    attempts: int,
    all_pairs_use_rf: bool,
) -> float:
    reservations = tuple(
        (
            f"pair-{index:05d}",
            attempts if all_pairs_use_rf or index == 0 else 0,
        )
        for index in range(population)
    )
    demand = RFPoolDemand(
        trace_id="phase6-population-probe",
        frame_index=0,
        time_s=0.0,
        active_pairs=population,
        reserved_rf_attempts_by_pair=reservations,
        offered_rf_attempts=sum(value for _, value in reservations),
        rf_using_pairs=sum(value > 0 for _, value in reservations),
    )
    per_attempt = model.access_failure_probability(model.evaluate(demand))
    return float(per_attempt**attempts)


def _cached_counterfactual_risk(
    cache: dict[tuple[int, int, bool], float],
    model: RFPoolModel,
    *,
    population: int,
    attempts: int,
    high: bool,
) -> float:
    key = (population, attempts, high)
    if key not in cache:
        cache[key] = _counterfactual_packet_risk(
            model,
            population=population,
            attempts=attempts,
            all_pairs_use_rf=high,
        )
    return cache[key]


def measure_population_opportunity(
    config: ProjectConfig,
    *,
    populations_by_density: dict[float, tuple[int, ...]],
    evaluation_sources: tuple[str, ...],
    missing_evaluation_sources: tuple[str, ...],
) -> PopulationOpportunity:
    """Measure focal feasibility flips caused only by other agents' demand."""

    if not populations_by_density or any(
        not values or any(value < 1 for value in values)
        for values in populations_by_density.values()
    ):
        raise SequentialOpportunityError("population evidence must be nonempty and positive")
    physical = build_rf_channel(config, band=SensitivityBand.NOMINAL)
    model = RFPoolModel(
        parameters=physical.collision,
        sensitivity_band=SensitivityBand.NOMINAL,
        attempt_airtime_s=config.rf.timing.airtime_s,
    )
    rows: list[PopulationDensityOpportunity] = []
    for density in sorted(populations_by_density):
        populations = np.asarray(populations_by_density[density], dtype=np.int64)
        median = int(np.median(populations))
        maximum = int(populations.max())
        risk_cache: dict[tuple[int, int, bool], float] = {}

        attempt_rows: list[PopulationAttemptOpportunity] = []
        for attempts in range(1, config.environment.max_rf_attempts + 1):
            flips_by_population = {
                population: (
                    _cached_counterfactual_risk(
                        risk_cache,
                        model,
                        population=population,
                        attempts=attempts,
                        high=False,
                    )
                    <= config.service.miss_budget
                    < _cached_counterfactual_risk(
                        risk_cache,
                        model,
                        population=population,
                        attempts=attempts,
                        high=True,
                    )
                )
                for population in range(1, maximum + 1)
            }
            first_flip = next(
                (population for population, flipped in flips_by_population.items() if flipped),
                None,
            )
            flip_frames = sum(flips_by_population[int(population)] for population in populations)
            attempt_rows.append(
                PopulationAttemptOpportunity(
                    rf_attempts=attempts,
                    first_flip_population=first_flip,
                    flip_frames=int(flip_frames),
                    flip_frame_fraction=float(flip_frames / len(populations)),
                    median_population=median,
                    low_demand_risk_at_median=_cached_counterfactual_risk(
                        risk_cache,
                        model,
                        population=median,
                        attempts=attempts,
                        high=False,
                    ),
                    all_rf_risk_at_median=_cached_counterfactual_risk(
                        risk_cache,
                        model,
                        population=median,
                        attempts=attempts,
                        high=True,
                    ),
                )
            )
        rows.append(
            PopulationDensityOpportunity(
                density=density,
                frames=len(populations),
                minimum_population=int(populations.min()),
                median_population=median,
                maximum_population=maximum,
                attempts=tuple(attempt_rows),
            )
        )
    ready = not missing_evaluation_sources
    detected = any(row.opportunity_detected for row in rows) if ready else None
    status: OpportunityStatus = (
        "diagnostic" if not ready else "opportunity" if detected else "no-detected-opportunity"
    )
    return PopulationOpportunity(
        status=status,
        evidence_ready=ready,
        opportunity_detected=detected,
        evaluation_sources=evaluation_sources,
        missing_evaluation_sources=missing_evaluation_sources,
        densities=tuple(rows),
    )


def build_sequential_opportunity_report(
    config: ProjectConfig,
    *,
    training_cache_paths: tuple[Path, ...],
    evaluation_cache_paths: tuple[Path, ...],
    frame_cache_root: str | Path,
    confidence_level: float | None = None,
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int = 0,
    minimum_history_rows: int | None = None,
    minimum_history_clusters: int | None = None,
) -> SequentialOpportunityReport:
    """Build both necessary-condition probes and the Phase 6 PPO gate."""

    confidence = (
        confidence_level if confidence_level is not None else config.evaluation.confidence_level
    )
    replicates = (
        bootstrap_replicates
        if bootstrap_replicates is not None
        else config.evaluation.bootstrap_replicates
    )
    minimum_rows = (
        minimum_history_rows
        if minimum_history_rows is not None
        else config.evaluation.min_packets_per_policy_density
    )
    minimum_clusters = (
        minimum_history_clusters
        if minimum_history_clusters is not None
        else config.evaluation.min_trajectory_pair_clusters
    )
    if not 0.5 < confidence < 1.0 or replicates < 1_000 or bootstrap_seed < 0:
        raise SequentialOpportunityError("opportunity statistical settings are invalid")
    if minimum_rows < 1 or minimum_clusters < 1:
        raise SequentialOpportunityError("opportunity evidence minima must be positive")
    history = measure_history_opportunity(
        config,
        training_cache_paths=training_cache_paths,
        evaluation_cache_paths=evaluation_cache_paths,
        confidence_level=confidence,
        bootstrap_replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        minimum_evaluation_rows=minimum_rows,
        minimum_evaluation_clusters=minimum_clusters,
    )
    populations, sources, missing = load_test_populations(config, frame_cache_root)
    population = measure_population_opportunity(
        config,
        populations_by_density=populations,
        evaluation_sources=sources,
        missing_evaluation_sources=missing,
    )
    evidence = (history.opportunity_detected, population.opportunity_detected)
    gate: GateDecision
    if True in evidence:
        gate = "go"
    elif all(value is not None for value in evidence):
        gate = "no-go"
    else:
        gate = "insufficient-evidence"
    return SequentialOpportunityReport(
        config_hash=config_hash(config),
        miss_budget=config.service.miss_budget,
        confidence_level=confidence,
        bootstrap_replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        history=history,
        population=population,
        gate_decision=gate,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "SEQUENTIAL_OPPORTUNITY_SCHEMA",
    "GateDecision",
    "HistoryLinkEstimate",
    "HistoryOpportunity",
    "OpportunityStatus",
    "PopulationAttemptOpportunity",
    "PopulationDensityOpportunity",
    "PopulationOpportunity",
    "SequentialOpportunityError",
    "SequentialOpportunityReport",
    "build_sequential_opportunity_report",
    "load_test_populations",
    "measure_history_opportunity",
    "measure_population_opportunity",
]
