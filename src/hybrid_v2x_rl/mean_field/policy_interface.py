"""Information boundary between population rollouts and action policies.

Every policy receives the same causal frame object.  Channel truth is carried
separately and is supplied only to policies that explicitly declare themselves
non-deployable oracles.  Keeping the two inputs distinct makes an accidental
oracle feature visible at the call site and keeps baseline comparisons on the
same action-accounting and outcome path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import ActionResourceMap, PolicyAction
from hybrid_v2x_rl.env.rollout import PairChannelEvaluation
from hybrid_v2x_rl.mean_field.action_masks import MaskedActionSpace
from hybrid_v2x_rl.mean_field.actor_observations import CausalActorFrame
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.mean_field.local_rf_pipeline import (
    FrameLocalRFContext,
    LocalRFPhysicsModel,
)

PolicyProposal: TypeAlias = PolicyAction | None
OracleChannelTruth: TypeAlias = Mapping[str, PairChannelEvaluation]


class PopulationPolicyError(HybridV2XError):
    """A policy request crosses information or population-frame boundaries."""


@dataclass(frozen=True, slots=True)
class PopulationPolicyFrame:
    """One causal simultaneous decision presented to any population policy."""

    frame: PopulationFrame
    actor_frame: CausalActorFrame
    observation: FrameObservation
    action_space: MaskedActionSpace
    resource_map: ActionResourceMap
    local_rf_model: LocalRFPhysicsModel
    local_rf_context: FrameLocalRFContext
    miss_budget: float

    def __post_init__(self) -> None:
        if (
            self.frame.trace_id != self.actor_frame.trace_id
            or self.frame.trace_id != self.observation.trace_id
            or self.frame.index != self.actor_frame.frame_index
            or self.frame.index != self.observation.frame_index
            or self.frame.active_pair_ids != self.actor_frame.pair_ids
            or self.frame.active_pair_ids != self.observation.pair_ids
        ):
            raise PopulationPolicyError(
                "policy inputs must describe one population frame"
            )
        if not math.isclose(
            self.frame.time_s,
            self.observation.time_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise PopulationPolicyError("policy inputs must share one decision time")
        if not isinstance(self.local_rf_model, LocalRFPhysicsModel):
            raise PopulationPolicyError(
                "policy frame requires the pair-local RF physics model"
            )
        if (
            not isinstance(self.local_rf_context, FrameLocalRFContext)
            or self.local_rf_context.frame != self.frame
        ):
            raise PopulationPolicyError(
                "policy frame requires pair-local context for the same population"
            )
        if not 0.0 < self.miss_budget < 1.0:
            raise PopulationPolicyError("policy miss_budget must lie in (0, 1)")

    @property
    def columns(self) -> tuple[str, ...]:
        """Stable names for raw and normalized actor columns."""

        return self.actor_frame.schema.columns

    @property
    def population_size(self) -> int:
        return len(self.frame.pairs)


@runtime_checkable
class PopulationPolicy(Protocol):
    """A joint-action producer consumed by the shared rollout engine."""

    @property
    def name(self) -> str: ...

    @property
    def requires_oracle_truth(self) -> bool: ...

    def select_actions(
        self,
        decision: PopulationPolicyFrame,
        *,
        channel_truth: OracleChannelTruth | None,
    ) -> tuple[PolicyProposal, ...]:
        """Return one proposal per active pair; ``None`` requests fallback."""


__all__ = [
    "OracleChannelTruth",
    "PolicyProposal",
    "PopulationPolicy",
    "PopulationPolicyError",
    "PopulationPolicyFrame",
]
