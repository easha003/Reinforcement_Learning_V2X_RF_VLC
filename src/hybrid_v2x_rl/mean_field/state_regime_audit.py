"""Causal state coverage and counterfactual action-feasibility audit.

The audit answers two questions that aggregate PPO metrics cannot answer:

* do the training and validation traces actually expose the decision contexts
  used in the proposed policy interpretation; and
* in those contexts, which of the nine actions are feasible and cheapest under
  declared population RF-load counterfactuals?

Regime labels use only raw actor-visible, action-independent columns.  Exact
RF/VLC risks are kept on the audit side and never enter the actor tensor.  The
test split is deliberately unsupported by the campaign builder so threshold
or experiment choices cannot leak from final evidence.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import numpy as np

from hybrid_v2x_rl.channels.rf.model import RFPropagationResult
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import (
    POLICY_ACTION_ORDER,
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.actor_observations import (
    CausalActorObservationAssembler,
)
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    run_policy_rollout_with_state,
)
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PopulationFrameReader,
    TraceCatalog,
)
from hybrid_v2x_rl.mean_field.local_rf_pipeline import LocalRFPhysicsModel
from hybrid_v2x_rl.mean_field.normalization import ObservationNormalizationState
from hybrid_v2x_rl.mean_field.policy_interface import (
    OracleChannelTruth,
    PolicyProposal,
    PopulationPolicyFrame,
)

STATE_REGIME_AUDIT_SCHEMA: Final = "hybrid-rf-vlc-rl.state-regime-audit.v3"

RegimeName = Literal[
    "easy_state",
    "moderate_rf_conditions",
    "poor_vlc_usable_rf",
    "uncertain_mixed_state",
    "heavy_rf_contention_optical_permitted",
]
REGIME_NAMES: Final[tuple[RegimeName, ...]] = (
    "easy_state",
    "moderate_rf_conditions",
    "poor_vlc_usable_rf",
    "uncertain_mixed_state",
    "heavy_rf_contention_optical_permitted",
)

LoadProfileName = Literal["vlc_offload", "rf1_pressure", "rf4_pressure"]
LOAD_PROFILES: Final[Mapping[LoadProfileName, int]] = {
    "vlc_offload": 0,
    "rf1_pressure": 1,
    "rf4_pressure": 4,
}

_STRUCTURAL_FEATURES: Final = (
    "rf_channel_busy_ratio",
    "neighbor_count",
    "optical_fov_margin",
    "predicted_blockage_probability",
    "predictor_confidence",
    "track_age",
)


class StateRegimeAuditError(HybridV2XError):
    """A state-regime audit request or evidence record is invalid."""


def _probability(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise StateRegimeAuditError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Training-fitted causal thresholds frozen before validation."""

    cbr_low: float
    cbr_high: float
    neighbor_low: float
    neighbor_high: float
    blockage_low: float
    blockage_high: float
    confidence_low: float
    track_age_high: float
    fit_rows_seen: int
    fit_rows_retained: int
    quantile_low: float = 0.25
    quantile_high: float = 0.75

    def __post_init__(self) -> None:
        for name in (
            "cbr_low",
            "cbr_high",
            "blockage_low",
            "blockage_high",
            "confidence_low",
        ):
            _probability(name, float(getattr(self, name)))
        for name in ("neighbor_low", "neighbor_high", "track_age_high"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise StateRegimeAuditError(f"{name} must be finite and non-negative")
        if self.cbr_low > self.cbr_high:
            raise StateRegimeAuditError("CBR thresholds are reversed")
        if self.neighbor_low > self.neighbor_high:
            raise StateRegimeAuditError("neighbor thresholds are reversed")
        if self.blockage_low > self.blockage_high:
            raise StateRegimeAuditError("blockage thresholds are reversed")
        if not 0.0 < self.quantile_low < self.quantile_high < 1.0:
            raise StateRegimeAuditError("threshold quantiles must be ordered in (0, 1)")
        if self.fit_rows_seen < 1 or not 1 <= self.fit_rows_retained <= self.fit_rows_seen:
            raise StateRegimeAuditError("threshold fit row counts are invalid")

    def as_dict(self) -> dict[str, object]:
        return {
            "fit_split": "train",
            "fit_rows_seen": self.fit_rows_seen,
            "fit_rows_retained": self.fit_rows_retained,
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "rf_channel_busy_ratio": {"low": self.cbr_low, "high": self.cbr_high},
            "neighbor_count": {"low": self.neighbor_low, "high": self.neighbor_high},
            "predicted_blockage_probability": {
                "low": self.blockage_low,
                "high": self.blockage_high,
            },
            "predictor_confidence_low": self.confidence_low,
            "track_age_high": self.track_age_high,
            "optical_fov_boundary_rad": 0.0,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimeThresholds:
        """Restore the exact training-fitted threshold contract from an audit."""

        expected = {
            "fit_split",
            "fit_rows_seen",
            "fit_rows_retained",
            "quantile_low",
            "quantile_high",
            "rf_channel_busy_ratio",
            "neighbor_count",
            "predicted_blockage_probability",
            "predictor_confidence_low",
            "track_age_high",
            "optical_fov_boundary_rad",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise StateRegimeAuditError(
                "persisted regime thresholds do not match their schema"
            )
        if payload["fit_split"] != "train" or payload["optical_fov_boundary_rad"] != 0.0:
            raise StateRegimeAuditError(
                "persisted regime thresholds violate the frozen causal contract"
            )

        def bounds(name: str) -> tuple[float, float]:
            value = payload[name]
            if not isinstance(value, Mapping) or set(value) != {"low", "high"}:
                raise StateRegimeAuditError(f"persisted {name} bounds are invalid")
            try:
                return float(value["low"]), float(value["high"])
            except (TypeError, ValueError) as error:
                raise StateRegimeAuditError(
                    f"persisted {name} bounds must be numeric"
                ) from error

        cbr_low, cbr_high = bounds("rf_channel_busy_ratio")
        neighbor_low, neighbor_high = bounds("neighbor_count")
        blockage_low, blockage_high = bounds("predicted_blockage_probability")
        def integer(name: str) -> int:
            value = payload[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise StateRegimeAuditError(f"persisted {name} must be an integer")
            return value

        def numeric(name: str) -> float:
            value = payload[name]
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise StateRegimeAuditError(f"persisted {name} must be numeric")
            return float(value)

        fit_rows_seen = integer("fit_rows_seen")
        fit_rows_retained = integer("fit_rows_retained")
        quantile_low = numeric("quantile_low")
        quantile_high = numeric("quantile_high")
        confidence_low = numeric("predictor_confidence_low")
        track_age_high = numeric("track_age_high")
        return cls(
            cbr_low=cbr_low,
            cbr_high=cbr_high,
            neighbor_low=neighbor_low,
            neighbor_high=neighbor_high,
            blockage_low=blockage_low,
            blockage_high=blockage_high,
            confidence_low=confidence_low,
            track_age_high=track_age_high,
            fit_rows_seen=fit_rows_seen,
            fit_rows_retained=fit_rows_retained,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )


@dataclass(slots=True)
class ThresholdReservoir:
    """Bounded deterministic reservoir of training-only structural rows."""

    maximum_rows: int
    seed: int
    _rows: list[tuple[float, ...]] = field(default_factory=list, init=False)
    _seen: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.maximum_rows < 1 or self.seed < 0:
            raise StateRegimeAuditError("reservoir size must be positive and seed non-negative")
        self._rng = np.random.default_rng(self.seed)

    @property
    def rows_seen(self) -> int:
        return self._seen

    @property
    def rows_retained(self) -> int:
        return len(self._rows)

    def observe(self, values: Sequence[float], columns: Sequence[str]) -> None:
        indices = _column_indices(columns)
        row = tuple(float(values[indices[name]]) for name in _STRUCTURAL_FEATURES)
        if not all(math.isfinite(value) for value in row):
            raise StateRegimeAuditError("threshold row contains non-finite values")
        self._seen += 1
        if len(self._rows) < self.maximum_rows:
            self._rows.append(row)
            return
        replacement = int(self._rng.integers(0, self._seen))
        if replacement < self.maximum_rows:
            self._rows[replacement] = row

    def fit(
        self,
        *,
        quantile_low: float = 0.25,
        quantile_high: float = 0.75,
    ) -> RegimeThresholds:
        if not self._rows:
            raise StateRegimeAuditError("no usable training rows were retained")
        if not 0.0 < quantile_low < quantile_high < 1.0:
            raise StateRegimeAuditError("fit quantiles must be ordered in (0, 1)")
        matrix = np.asarray(self._rows, dtype=np.float64)
        positions = {name: index for index, name in enumerate(_STRUCTURAL_FEATURES)}

        def quantile(name: str, level: float) -> float:
            return float(np.quantile(matrix[:, positions[name]], level))

        return RegimeThresholds(
            cbr_low=quantile("rf_channel_busy_ratio", quantile_low),
            cbr_high=quantile("rf_channel_busy_ratio", quantile_high),
            neighbor_low=quantile("neighbor_count", quantile_low),
            neighbor_high=quantile("neighbor_count", quantile_high),
            blockage_low=quantile("predicted_blockage_probability", quantile_low),
            blockage_high=quantile("predicted_blockage_probability", quantile_high),
            confidence_low=quantile("predictor_confidence", quantile_low),
            track_age_high=quantile("track_age", quantile_high),
            fit_rows_seen=self.rows_seen,
            fit_rows_retained=self.rows_retained,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
        )


def _column_indices(columns: Sequence[str]) -> dict[str, int]:
    mapping = {name: index for index, name in enumerate(columns)}
    missing = tuple(name for name in _STRUCTURAL_FEATURES if name not in mapping)
    if missing:
        raise StateRegimeAuditError(
            "actor schema lacks structural audit features",
            context={"missing": missing},
        )
    return mapping


def classify_regimes(
    values: Sequence[float],
    columns: Sequence[str],
    thresholds: RegimeThresholds,
) -> tuple[RegimeName, ...]:
    """Return overlapping decision contexts from causal raw actor columns."""

    indices = _column_indices(columns)

    def value(name: str) -> float:
        result = float(values[indices[name]])
        if not math.isfinite(result):
            raise StateRegimeAuditError("actor regime feature is non-finite")
        return result

    cbr = value("rf_channel_busy_ratio")
    neighbors = value("neighbor_count")
    fov = value("optical_fov_margin")
    blockage = value("predicted_blockage_probability")
    confidence = value("predictor_confidence")
    track_age = value("track_age")

    rf_light = cbr <= thresholds.cbr_low and neighbors <= thresholds.neighbor_low
    rf_heavy = cbr >= thresholds.cbr_high or neighbors >= thresholds.neighbor_high
    rf_moderate = not rf_light and not rf_heavy
    # Strict tails keep a quantile tie (for example every 50 ms causal track)
    # from labeling the entire dataset uncertain.  Exact equality belongs to
    # the observed central mass, not both tails at once.
    uncertain = (
        confidence < thresholds.confidence_low
        or track_age > thresholds.track_age_high
    )
    optical_favorable = (
        fov > 0.0
        and blockage <= thresholds.blockage_low
    )
    optical_impaired = fov <= 0.0 or blockage >= thresholds.blockage_high
    optical_ambiguous = not optical_favorable and not optical_impaired

    labels: list[RegimeName] = []
    if optical_favorable and rf_light and not uncertain:
        labels.append("easy_state")
    if rf_moderate and optical_ambiguous and not uncertain:
        labels.append("moderate_rf_conditions")
    if optical_impaired and not rf_heavy:
        labels.append("poor_vlc_usable_rf")
    if uncertain:
        labels.append("uncertain_mixed_state")
    if rf_heavy and optical_favorable:
        labels.append("heavy_rf_contention_optical_permitted")
    return tuple(labels)


@dataclass(frozen=True, slots=True)
class CounterfactualAction:
    action: PolicyAction
    conditional_miss_probability: float
    activation_cost: float
    feasible: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action.label,
            "conditional_miss_probability": self.conditional_miss_probability,
            "activation_cost": self.activation_cost,
            "feasible": self.feasible,
        }


@dataclass(frozen=True, slots=True)
class CounterfactualActionSet:
    load_profile: str
    other_pair_rf_attempts: int
    actions: tuple[CounterfactualAction, ...]
    selected_action: PolicyAction
    any_feasible: bool


def assess_counterfactual_actions(
    decision: PopulationPolicyFrame,
    *,
    pair_id: str,
    rf_decoding_failure_probability: float,
    vlc_failure_probability: float,
    load_profile: LoadProfileName,
) -> CounterfactualActionSet:
    """Evaluate all allowed focal actions under one declared other-pair load."""

    _probability("rf_decoding_failure_probability", rf_decoding_failure_probability)
    _probability("vlc_failure_probability", vlc_failure_probability)
    try:
        other_attempts = LOAD_PROFILES[load_profile]
    except KeyError as error:
        raise StateRegimeAuditError("unknown RF-load profile") from error
    other_action = (
        PolicyAction.VLC
        if other_attempts == 0
        else PolicyAction(other_attempts)
    )
    return assess_counterfactual_actions_at_load(
        decision,
        pair_id=pair_id,
        rf_decoding_failure_probability=rf_decoding_failure_probability,
        vlc_failure_probability=vlc_failure_probability,
        other_actions_by_pair={
            candidate_id: other_action
            for candidate_id in decision.frame.active_pair_ids
            if candidate_id != pair_id
        },
        load_label=load_profile,
    )


def assess_counterfactual_actions_at_load(
    decision: PopulationPolicyFrame,
    *,
    pair_id: str,
    rf_decoding_failure_probability: float,
    vlc_failure_probability: float,
    other_actions_by_pair: Mapping[str, PolicyAction],
    load_label: str = "policy_induced_load",
) -> CounterfactualActionSet:
    """Evaluate focal actions while holding exact other-pair actions fixed."""

    _probability("rf_decoding_failure_probability", rf_decoding_failure_probability)
    _probability("vlc_failure_probability", vlc_failure_probability)
    if not isinstance(load_label, str) or not load_label.strip():
        raise StateRegimeAuditError("counterfactual load label must be non-empty")
    population = decision.population_size
    if population < 1:
        raise StateRegimeAuditError("counterfactual action audit requires a nonempty frame")
    if pair_id not in decision.frame.active_pair_ids:
        raise StateRegimeAuditError("counterfactual focal pair is absent")
    expected_other_ids = set(decision.frame.active_pair_ids) - {pair_id}
    if set(other_actions_by_pair) != expected_other_ids or any(
        type(action) is not PolicyAction
        or action not in decision.action_space.mask.allowed_actions
        for action in other_actions_by_pair.values()
    ):
        raise StateRegimeAuditError(
            "counterfactual other actions must cover the remaining population"
        )

    propagation = {
        candidate_id: RFPropagationResult(
            propagation_state=RFPropagationState.LOS,
            pathloss_db=0.0,
            shadowing_db=0.0,
            fading_gain_linear=1.0,
            sinr_db=0.0,
            decoding_failure_probability=rf_decoding_failure_probability,
        )
        for candidate_id in decision.frame.active_pair_ids
    }
    other_pair_rf_attempts = sum(
        action_resources(action).reserved_rf_attempts
        for action in other_actions_by_pair.values()
    )

    rows: list[CounterfactualAction] = []
    for action in decision.action_space.mask.allowed_actions:
        spec = action_resources(action)
        actions = dict(other_actions_by_pair)
        actions[pair_id] = action
        ledger = FrameActionLedger.from_frame(
            decision.frame,
            actions,
            resource_map=decision.resource_map,
        )
        physics = decision.local_rf_model.evaluate(
            decision.local_rf_context,
            ledger,
            propagation_by_pair=propagation,
        )
        risk = 1.0
        if spec.uses_rf:
            per_attempt = physics.attempt_risks.risk_for(
                pair_id
            ).total_failure_probability
            risk *= per_attempt**spec.reserved_rf_attempts
        if spec.uses_vlc:
            risk *= vlc_failure_probability
        clipped = float(min(1.0, max(0.0, risk)))
        rows.append(
            CounterfactualAction(
                action=action,
                conditional_miss_probability=clipped,
                activation_cost=decision.resource_map.activation_cost(action),
                feasible=clipped <= decision.miss_budget,
            )
        )
    feasible = tuple(row for row in rows if row.feasible)
    selected = min(
        feasible or tuple(rows),
        key=(
            (lambda row: (row.activation_cost, row.conditional_miss_probability, int(row.action)))
            if feasible
            else (lambda row: (row.conditional_miss_probability, row.activation_cost, int(row.action)))
        ),
    )
    return CounterfactualActionSet(
        load_profile=load_label,
        other_pair_rf_attempts=other_pair_rf_attempts,
        actions=tuple(rows),
        selected_action=selected.action,
        any_feasible=bool(feasible),
    )


@dataclass(slots=True)
class _ProfileTally:
    evaluated_rows: int = 0
    feasible_rows: int = 0
    selected_actions: Counter[str] = field(default_factory=Counter)
    feasible_actions: Counter[str] = field(default_factory=Counter)

    def observe(self, assessment: CounterfactualActionSet) -> None:
        self.evaluated_rows += 1
        self.feasible_rows += int(assessment.any_feasible)
        self.selected_actions[assessment.selected_action.label] += 1
        for row in assessment.actions:
            self.feasible_actions[row.action.label] += int(row.feasible)

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluated_rows": self.evaluated_rows,
            "any_feasible_rows": self.feasible_rows,
            "no_feasible_rows": self.evaluated_rows - self.feasible_rows,
            "any_feasible_fraction": (
                self.feasible_rows / self.evaluated_rows if self.evaluated_rows else None
            ),
            "selected_cheapest_feasible_or_minimum_risk": {
                action: int(self.selected_actions[action]) for action in POLICY_ACTION_ORDER
            },
            "feasible_action_rows": {
                action: int(self.feasible_actions[action]) for action in POLICY_ACTION_ORDER
            },
        }


@dataclass(slots=True)
class _RegimeTally:
    rows: int = 0
    clusters: set[str] = field(default_factory=set)
    traces: set[str] = field(default_factory=set)
    profiles: dict[LoadProfileName, _ProfileTally] = field(
        default_factory=lambda: {
            name: _ProfileTally() for name in LOAD_PROFILES
        }
    )


@dataclass(slots=True)
class CoverageAccumulator:
    """Mutable bounded-pass tally converted to a versioned immutable report."""

    thresholds: RegimeThresholds
    _cells: dict[tuple[str, float, RegimeName], _RegimeTally] = field(
        default_factory=dict,
        init=False,
    )
    _usable: Counter[tuple[str, float]] = field(default_factory=Counter, init=False)
    _unusable: Counter[tuple[str, float]] = field(default_factory=Counter, init=False)
    _unclassified: Counter[tuple[str, float]] = field(default_factory=Counter, init=False)

    def observe_unusable(self, *, split: str, density: float) -> None:
        self._unusable[(split, density)] += 1

    def observe(
        self,
        decision: PopulationPolicyFrame,
        *,
        pair_id: str,
        values: Sequence[float],
        rf_decoding_failure_probability: float,
        vlc_failure_probability: float,
    ) -> None:
        split = decision.frame.source.split
        density = decision.frame.source.density
        identity = (split, density)
        self._usable[identity] += 1
        labels = classify_regimes(values, decision.columns, self.thresholds)
        if not labels:
            self._unclassified[identity] += 1
            return
        assessments = {
            name: assess_counterfactual_actions(
                decision,
                pair_id=pair_id,
                rf_decoding_failure_probability=rf_decoding_failure_probability,
                vlc_failure_probability=vlc_failure_probability,
                load_profile=name,
            )
            for name in LOAD_PROFILES
        }
        cluster = f"{decision.frame.trace_id}/{pair_id}"
        for label in labels:
            tally = self._cells.setdefault((split, density, label), _RegimeTally())
            tally.rows += 1
            tally.clusters.add(cluster)
            tally.traces.add(decision.frame.trace_id)
            for name, assessment in assessments.items():
                tally.profiles[name].observe(assessment)

    def report_rows(
        self,
        *,
        splits: Sequence[str],
        densities: Sequence[float],
        minimum_rows: int,
        minimum_clusters: int,
    ) -> tuple[dict[str, object], ...]:
        rows: list[dict[str, object]] = []
        for split in splits:
            for density in sorted(set(densities)):
                identity = (split, density)
                for regime in REGIME_NAMES:
                    tally = self._cells.get((split, density, regime), _RegimeTally())
                    supported = (
                        tally.rows >= minimum_rows
                        and len(tally.clusters) >= minimum_clusters
                        and bool(tally.traces)
                    )
                    rows.append(
                        {
                            "split": split,
                            "density_vehicles_per_lane_km": density,
                            "regime": regime,
                            "rows": tally.rows,
                            "pair_episode_clusters": len(tally.clusters),
                            "trace_count": len(tally.traces),
                            "trace_ids": sorted(tally.traces),
                            "supported": supported,
                            "counterfactuals": {
                                name: tally.profiles[name].as_dict()
                                for name in LOAD_PROFILES
                            },
                            "split_density_usable_rows": self._usable[identity],
                            "split_density_unusable_rows": self._unusable[identity],
                            "split_density_unclassified_rows": self._unclassified[identity],
                        }
                    )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class WindowSample:
    trace_id: str
    split: str
    density: float
    start_frame_index: int
    frames: int

    def as_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "split": self.split,
            "density_vehicles_per_lane_km": self.density,
            "start_frame_index": self.start_frame_index,
            "frames": self.frames,
        }


@dataclass(frozen=True, slots=True)
class StateRegimeAuditReport:
    config_hash: str
    policy_environment_scope_hash: str
    environment_seed: int
    thresholds: RegimeThresholds
    windows: tuple[WindowSample, ...]
    rows: tuple[dict[str, object], ...]
    minimum_rows: int
    minimum_clusters: int
    generated_at_utc: datetime

    @property
    def all_regimes_supported(self) -> bool:
        return bool(self.rows) and all(bool(row["supported"]) for row in self.rows)

    @property
    def campaign_rows(self) -> tuple[dict[str, object], ...]:
        """Aggregate density cells without weakening the persisted cell gate."""

        summary: list[dict[str, object]] = []
        for split in ("train", "validation"):
            for regime in REGIME_NAMES:
                selected = tuple(
                    row
                    for row in self.rows
                    if row["split"] == split and row["regime"] == regime
                )
                row_count = sum(cast(int, row["rows"]) for row in selected)
                clusters = sum(
                    cast(int, row["pair_episode_clusters"]) for row in selected
                )
                trace_ids: set[str] = set()
                for row in selected:
                    trace_ids.update(cast(list[str], row["trace_ids"]))
                summary.append(
                    {
                        "split": split,
                        "regime": regime,
                        "rows": row_count,
                        "pair_episode_clusters": clusters,
                        "trace_count": len(trace_ids),
                        "trace_ids": sorted(trace_ids),
                        "observed": row_count > 0,
                        "supported": (
                            row_count >= self.minimum_rows
                            and clusters >= self.minimum_clusters
                            and bool(trace_ids)
                        ),
                    }
                )
        return tuple(summary)

    @property
    def all_campaign_regimes_observed(self) -> bool:
        return bool(self.campaign_rows) and all(
            bool(row["observed"]) for row in self.campaign_rows
        )

    @property
    def all_campaign_regimes_supported(self) -> bool:
        return bool(self.campaign_rows) and all(
            bool(row["supported"]) for row in self.campaign_rows
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": STATE_REGIME_AUDIT_SCHEMA,
            "generated_at_utc": self.generated_at_utc.isoformat(),
            "config_hash": self.config_hash,
            "policy_environment_scope_hash": self.policy_environment_scope_hash,
            "environment_seed": self.environment_seed,
            "test_split_opened": False,
            "thresholds": self.thresholds.as_dict(),
            "regime_semantics": {
                "easy_state": "optical favorable AND RF light AND not uncertain",
                "moderate_rf_conditions": (
                    "RF moderate AND optical ambiguous AND not uncertain"
                ),
                "poor_vlc_usable_rf": "optical impaired AND RF not heavy",
                "uncertain_mixed_state": "low confidence OR high track age",
                "heavy_rf_contention_optical_permitted": (
                    "RF heavy AND optical favorable"
                ),
                "labels_may_overlap": True,
                "actor_visible_features_only": True,
            },
            "coverage_claim": {
                "level": "campaign",
                "density_conditioned_support_reported": True,
                "every_regime_at_every_density_claimed": False,
                "all_campaign_regimes_observed": self.all_campaign_regimes_observed,
                "all_campaign_regimes_supported": self.all_campaign_regimes_supported,
                "interpretation": (
                    "declared causal regimes are assessed across the campaign; "
                    "per-density support is reported separately and is not assumed uniform"
                ),
            },
            "counterfactual_load_profiles": {
                name: {
                    "other_pair_reserved_rf_attempts": attempts,
                    "description": (
                        "all other active pairs are held to this RF-attempt count; "
                        "the focal action replaces only its own count"
                    ),
                }
                for name, attempts in LOAD_PROFILES.items()
            },
            "coverage_gate": {
                "minimum_rows_per_split_density_regime": self.minimum_rows,
                "minimum_pair_episode_clusters_per_split_density_regime": (
                    self.minimum_clusters
                ),
                "all_split_density_regimes_supported": self.all_regimes_supported,
                "all_campaign_regimes_observed": self.all_campaign_regimes_observed,
                "all_campaign_regimes_supported": self.all_campaign_regimes_supported,
            },
            "sampled_windows": [window.as_dict() for window in self.windows],
            "campaign_coverage": list(self.campaign_rows),
            "coverage": list(self.rows),
        }

    def write_json(self, path: str | Path) -> Path:
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


@dataclass(slots=True)
class _CoveragePolicy:
    accumulator: CoverageAccumulator
    name: str = "state-regime-audit-cycle-probe"
    requires_oracle_truth: bool = True

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        if channel_truth is None:
            raise StateRegimeAuditError("coverage audit requires isolated channel truth")
        allowed = decision.action_space.mask.allowed_actions
        proposals: list[PolicyProposal] = []
        for pair, row in zip(
            decision.frame.pairs,
            decision.actor_frame.rows,
            strict=True,
        ):
            if row.values is None:
                self.accumulator.observe_unusable(
                    split=decision.frame.source.split,
                    density=decision.frame.source.density,
                )
                proposals.append(None)
                continue
            truth = channel_truth[pair.pair_id]
            self.accumulator.observe(
                decision,
                pair_id=pair.pair_id,
                values=row.values,
                rf_decoding_failure_probability=(
                    truth.rf_propagation.decoding_failure_probability
                ),
                vlc_failure_probability=truth.vlc_result.total_failure_probability,
            )
            proposals.append(allowed[pair.episode_step % len(allowed)])
        return tuple(proposals)


def _window_starts(
    reader: PopulationFrameReader,
    *,
    windows_per_trace: int,
    frames_per_window: int,
) -> tuple[int, ...]:
    if windows_per_trace < 1 or frames_per_window < 1:
        raise StateRegimeAuditError("window counts must be positive")
    maximum = max(0, reader.decision_frame_count - frames_per_window)
    if windows_per_trace == 1:
        return (maximum // 2,)
    raw = np.linspace(0, maximum, num=windows_per_trace)
    return tuple(sorted(set(int(round(value)) for value in raw)))


def _collect_threshold_window(
    config: ProjectConfig,
    source: FrameTraceSource,
    *,
    start_frame_index: int,
    frames: int,
    root_seed: int,
    reservoir: ThresholdReservoir,
) -> None:
    """Collect action-independent causal rows without evaluating channels."""

    reader = PopulationFrameReader(
        source,
        generation_period_s=config.service.generation_period_s,
        expected_config_hash=config_hash(config),
        expected_config_scope_hashes={
            "mobility_trace": scope_hash(config, "mobility_trace")
        },
    )
    assembler = CausalActorObservationAssembler.from_config(
        config,
        root_seed=root_seed,
    )
    assembler.reset(source.trace_id, start_frame_index=start_frame_index)
    action_space = MaskedActionSpace.from_config(config.environment, config.rf, config.vlc)
    resource_map = ActionResourceMap.from_config(config.environment, config.cost)
    local_rf_model = LocalRFPhysicsModel.from_config(config)
    iterator = iter(
        reader.iter_frames(
            start_frame_index=start_frame_index,
            max_frames=min(frames + 1, reader.decision_frame_count - start_frame_index),
        )
    )
    frame = next(iterator, None)
    processed = 0
    while frame is not None and processed < frames:
        next_frame = next(iterator, None)
        actor = assembler.begin_frame(frame)
        for row in actor.rows:
            if row.values is not None:
                reservoir.observe(row.values, actor.schema.columns)
        actions: dict[str, PolicyAction] = {}
        for pair, row in zip(frame.pairs, actor.rows, strict=True):
            proposal = PolicyAction.VLC if row.usable else None
            action = action_space.select(proposal, observation_usable=row.usable)
            actions[pair.pair_id] = action
            spec = action_resources(action)
            completion_s = max(
                spec.reserved_rf_attempts * config.rf.timing.airtime_s,
                config.vlc.timing.airtime_s if spec.uses_vlc else 0.0,
            )
            assembler.record_feedback(
                pair.pair_id,
                action=action,
                at_s=frame.time_s + completion_s,
                delivered=True,
            )
        ledger = FrameActionLedger.from_frame(frame, actions, resource_map=resource_map)
        context = local_rf_model.context_for(frame)
        propagation = {
            pair_id: RFPropagationResult(
                propagation_state=RFPropagationState.LOS,
                pathloss_db=0.0,
                shadowing_db=0.0,
                fading_gain_linear=1.0,
                sinr_db=0.0,
                decoding_failure_probability=0.0,
            )
            for pair_id in frame.active_pair_ids
        }
        physics = local_rf_model.evaluate(
            context,
            ledger,
            propagation_by_pair=propagation,
        )
        assembler.close_frame(physics, next_frame=next_frame)
        frame = next_frame
        processed += 1


def _selected_sources(
    catalog: TraceCatalog,
    split: Literal["train", "validation"],
    *,
    replicates_per_density: int | None,
) -> tuple[FrameTraceSource, ...]:
    sources = catalog.for_split(split)
    if replicates_per_density is None:
        return sources
    if replicates_per_density < 1:
        raise StateRegimeAuditError("replicates_per_density must be positive or None")
    selected: list[FrameTraceSource] = []
    counts: Counter[float] = Counter()
    for source in sources:
        if counts[source.density] < replicates_per_density:
            selected.append(source)
            counts[source.density] += 1
    return tuple(selected)


def build_state_regime_audit(
    config: ProjectConfig,
    *,
    windows_per_trace: int = 3,
    frames_per_window: int = 16,
    threshold_max_rows: int = 250_000,
    threshold_seed: int = 73,
    environment_seed: int | None = None,
    minimum_rows: int = 10_000,
    minimum_clusters: int = 200,
    replicates_per_density: int | None = None,
) -> StateRegimeAuditReport:
    """Build a bounded train/validation audit without touching the test split."""

    if not isinstance(config, ProjectConfig):
        raise StateRegimeAuditError("audit requires a resolved ProjectConfig")
    if minimum_rows < 1 or minimum_clusters < 1:
        raise StateRegimeAuditError("coverage minima must be positive")
    seed = config.training.root_seed if environment_seed is None else environment_seed
    if seed < 0:
        raise StateRegimeAuditError("environment_seed must be non-negative")
    catalog = TraceCatalog.from_splits(config.paths.trace_root, config.environment.splits)
    training = _selected_sources(
        catalog,
        "train",
        replicates_per_density=replicates_per_density,
    )
    validation = _selected_sources(
        catalog,
        "validation",
        replicates_per_density=replicates_per_density,
    )
    if not training or not validation:
        raise StateRegimeAuditError("audit requires configured training and validation traces")

    reservoir = ThresholdReservoir(maximum_rows=threshold_max_rows, seed=threshold_seed)
    windows: list[WindowSample] = []
    for source in training:
        reader = PopulationFrameReader(
            source,
            generation_period_s=config.service.generation_period_s,
            expected_config_hash=config_hash(config),
            expected_config_scope_hashes={
                "mobility_trace": scope_hash(config, "mobility_trace")
            },
        )
        for start in _window_starts(
            reader,
            windows_per_trace=windows_per_trace,
            frames_per_window=frames_per_window,
        ):
            selected_frames = min(
                frames_per_window,
                reader.decision_frame_count - start,
            )
            _collect_threshold_window(
                config,
                source,
                start_frame_index=start,
                frames=selected_frames,
                root_seed=seed,
                reservoir=reservoir,
            )
    thresholds = reservoir.fit()

    accumulator = CoverageAccumulator(thresholds)
    policy = _CoveragePolicy(accumulator)

    def schedule(
        sources: Sequence[FrameTraceSource],
    ) -> tuple[tuple[FrameTraceSource, int, int], ...]:
        selected: list[tuple[FrameTraceSource, int, int]] = []
        for source in sources:
            reader = PopulationFrameReader(
                source,
                generation_period_s=config.service.generation_period_s,
                expected_config_hash=config_hash(config),
                expected_config_scope_hashes={
                    "mobility_trace": scope_hash(config, "mobility_trace")
                },
            )
            for start in _window_starts(
                reader,
                windows_per_trace=windows_per_trace,
                frames_per_window=frames_per_window,
            ):
                selected.append(
                    (
                        source,
                        start,
                        min(frames_per_window, reader.decision_frame_count - start),
                    )
                )
        return tuple(selected)

    training_schedule = schedule(training)
    validation_schedule = schedule(validation)
    normalization_state: ObservationNormalizationState | None = None
    for index, (source, start, selected_frames) in enumerate(training_schedule):
        windows.append(
            WindowSample(
                trace_id=source.trace_id,
                split=source.split,
                density=source.density,
                start_frame_index=start,
                frames=selected_frames,
            )
        )
        result = run_policy_rollout_with_state(
            config,
            source,
            policy=policy,
            environment_seed=seed,
            policy_seed=threshold_seed,
            start_frame_index=start,
            max_frames=selected_frames,
            normalization_state=normalization_state,
            freeze_normalization_at_end=index == len(training_schedule) - 1,
        )
        normalization_state = result.normalization_state
    if normalization_state is None or not normalization_state.frozen:
        raise StateRegimeAuditError(
            "training audit did not produce frozen normalization state"
        )

    for source, start, selected_frames in validation_schedule:
        windows.append(
            WindowSample(
                trace_id=source.trace_id,
                split=source.split,
                density=source.density,
                start_frame_index=start,
                frames=selected_frames,
            )
        )
        run_policy_rollout_with_state(
            config,
            source,
            policy=policy,
            environment_seed=seed,
            policy_seed=threshold_seed,
            start_frame_index=start,
            max_frames=selected_frames,
            normalization_state=normalization_state,
        )

    densities = tuple(source.density for source in (*training, *validation))
    rows = accumulator.report_rows(
        splits=("train", "validation"),
        densities=densities,
        minimum_rows=minimum_rows,
        minimum_clusters=minimum_clusters,
    )
    return StateRegimeAuditReport(
        config_hash=config_hash(config),
        policy_environment_scope_hash=scope_hash(config, "policy_environment"),
        environment_seed=seed,
        thresholds=thresholds,
        windows=tuple(windows),
        rows=rows,
        minimum_rows=minimum_rows,
        minimum_clusters=minimum_clusters,
        generated_at_utc=datetime.now(UTC),
    )


__all__ = [
    "LOAD_PROFILES",
    "REGIME_NAMES",
    "STATE_REGIME_AUDIT_SCHEMA",
    "CounterfactualAction",
    "CounterfactualActionSet",
    "CoverageAccumulator",
    "RegimeName",
    "RegimeThresholds",
    "StateRegimeAuditError",
    "StateRegimeAuditReport",
    "ThresholdReservoir",
    "WindowSample",
    "assess_counterfactual_actions",
    "assess_counterfactual_actions_at_load",
    "build_state_regime_audit",
    "classify_regimes",
]
