"""One reset-seed authority for every stochastic Phase 5 environment path.

The environment has several kinds of randomness, but it must not have several
unrelated notions of a seed.  :class:`EnvironmentSeedState` resolves the
configured root against an optional ``reset(seed=...)`` override once.
:class:`TraceRandomness` then passes that exact root to causal sensing, matched
packet tapes, and noisy receiver feedback while binding all draws to the
immutable trace and stable pair-episode identity.

RF shadowing and fading are stateful rather than packet-local.  Their keyed
stream construction lives in :mod:`hybrid_v2x_rl.env.rollout`, which consumes
the same root and trace identity. Mobility is already frozen inside immutable
trace artifacts. Policy action sampling and statistical bootstrap resampling
belong to the trainer and analysis layers and use their own named streams; they
are intentionally not advanced by an environment step.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.randomness import derive_child_seed
from hybrid_v2x_rl.env.feedback import reported_quality
from hybrid_v2x_rl.mean_field.actor_observations import (
    CausalActorObservationAssembler,
)
from hybrid_v2x_rl.mean_field.frames import PopulationFrame
from hybrid_v2x_rl.mean_field.packet_outcomes import PairPacketOutcome
from hybrid_v2x_rl.mean_field.random_tape import (
    MatchedPacketTape,
    MatchedPacketTapeFactory,
)

ENVIRONMENT_SEED_SCHEMA: Final = "hybrid-rf-vlc-rl.environment-seed.v1"
RUNTIME_RANDOM_COMPONENTS: Final = (
    "sensor_noise",
    "rf_shadowing",
    "rf_blockage_residual",
    "rf_fading",
    "packet_half_duplex",
    "packet_collision",
    "packet_decoding",
    "feedback_rf",
    "feedback_vlc",
)


class EnvironmentSeedError(HybridV2XError):
    """The configured/reset seed or stochastic identity is inconsistent."""


def _validate_root_seed(value: int, *, name: str) -> int:
    try:
        derive_child_seed(value, f"{ENVIRONMENT_SEED_SCHEMA}.root-validation")
    except (TypeError, ValueError) as error:
        raise EnvironmentSeedError(
            f"{name} is not a valid unsigned 64-bit seed"
        ) from error
    return value


@dataclass(frozen=True, slots=True)
class EnvironmentSeedState:
    """Configured and active roots resolved exactly once at environment reset."""

    configured_root_seed: int
    active_root_seed: int
    explicit_reset_seed: bool

    def __post_init__(self) -> None:
        _validate_root_seed(
            self.configured_root_seed,
            name="configured_root_seed",
        )
        _validate_root_seed(self.active_root_seed, name="active_root_seed")
        if type(self.explicit_reset_seed) is not bool:
            raise EnvironmentSeedError("explicit_reset_seed must be boolean")

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        *,
        reset_seed: int | None = None,
    ) -> EnvironmentSeedState:
        """Use the configured root unless ``reset`` supplied an explicit seed."""

        if not isinstance(config, ProjectConfig):
            raise EnvironmentSeedError("seed state requires a resolved ProjectConfig")
        configured = int(config.training.root_seed)
        active = configured if reset_seed is None else reset_seed
        return cls(
            configured_root_seed=configured,
            active_root_seed=active,
            explicit_reset_seed=reset_seed is not None,
        )

    def for_trace(self, trace_id: str) -> TraceRandomness:
        """Bind the active reset root to one immutable trace identity."""

        return TraceRandomness(seed_state=self, trace_id=trace_id)


@dataclass(frozen=True, slots=True)
class TraceRandomness:
    """Factories and reports sharing one root and immutable trace identity."""

    seed_state: EnvironmentSeedState
    trace_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.seed_state, EnvironmentSeedState):
            raise EnvironmentSeedError(
                "trace randomness requires an EnvironmentSeedState"
            )
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise EnvironmentSeedError("trace_id must be a non-empty string")

    @property
    def root_seed(self) -> int:
        return self.seed_state.active_root_seed

    def actor_assembler(
        self,
        config: ProjectConfig,
        *,
        start_frame_index: int = 0,
    ) -> CausalActorObservationAssembler:
        """Build and reset causal sensing from this trace's active root."""

        if not isinstance(config, ProjectConfig):
            raise EnvironmentSeedError(
                "actor seeding requires a resolved ProjectConfig"
            )
        if int(config.training.root_seed) != self.seed_state.configured_root_seed:
            raise EnvironmentSeedError(
                "seed state was created from a different project configuration"
            )
        assembler = CausalActorObservationAssembler.from_config(
            config,
            root_seed=self.root_seed,
        )
        assembler.reset(self.trace_id, start_frame_index=start_frame_index)
        return assembler

    def packet_tapes(
        self,
        frame: PopulationFrame,
    ) -> Mapping[str, MatchedPacketTape]:
        """Generate one identity-addressed tape for every current pair."""

        if not isinstance(frame, PopulationFrame):
            raise EnvironmentSeedError("packet tapes require a PopulationFrame")
        if frame.trace_id != self.trace_id:
            raise EnvironmentSeedError(
                "population frame does not belong to this trace seed context",
                context={"actual": frame.trace_id, "expected": self.trace_id},
            )
        factory = MatchedPacketTapeFactory(self.root_seed)
        return MappingProxyType(
            {
                pair.pair_id: factory.for_population_pair(frame, pair)
                for pair in frame.pairs
            }
        )

    def feedback_measurements(
        self,
        outcome: PairPacketOutcome,
        tape: MatchedPacketTape,
    ) -> Mapping[Link, float]:
        """Degrade selected-link truth with independently seeded report noise."""

        if not isinstance(outcome, PairPacketOutcome):
            raise EnvironmentSeedError(
                "feedback seeding requires a PairPacketOutcome"
            )
        if not isinstance(tape, MatchedPacketTape):
            raise EnvironmentSeedError("feedback seeding requires a matched tape")
        identity = tape.identity
        if identity.trace_id != self.trace_id or identity.pair_episode_id != outcome.pair_id:
            raise EnvironmentSeedError(
                "feedback outcome and packet tape identities do not match",
                context={
                    "trace_id": identity.trace_id,
                    "pair_id": identity.pair_episode_id,
                    "outcome_pair_id": outcome.pair_id,
                },
            )
        expected_tape = MatchedPacketTapeFactory(self.root_seed).build(identity)
        if tape != expected_tape:
            raise EnvironmentSeedError(
                "packet tape was not generated from the active reset seed",
                context={
                    "trace_id": identity.trace_id,
                    "pair_id": identity.pair_episode_id,
                    "packet_index": identity.packet_index,
                },
            )

        exact: dict[Link, float] = {}
        if outcome.rf_attempt_risk is not None:
            exact[Link.RF] = outcome.rf_attempt_risk.propagation.sinr_db
        if outcome.vlc_result is not None:
            exact[Link.VLC] = outcome.vlc_result.snr_db
        return MappingProxyType(
            {
                link: reported_quality(
                    value_db,
                    link=link,
                    root_seed=self.root_seed,
                    trace_id=self.trace_id,
                    pair_id=outcome.pair_id,
                    packet_index=identity.packet_index,
                )
                for link, value_db in exact.items()
            }
        )

    def as_reset_info(self) -> Mapping[str, object]:
        """Expose enough immutable seed provenance to reproduce this reset."""

        return MappingProxyType(
            {
                "seed_schema": ENVIRONMENT_SEED_SCHEMA,
                "seed": self.root_seed,
                "active_root_seed": self.root_seed,
                "configured_root_seed": self.seed_state.configured_root_seed,
                "reset_seed_explicit": self.seed_state.explicit_reset_seed,
                "trace_id": self.trace_id,
                "runtime_random_components": RUNTIME_RANDOM_COMPONENTS,
            }
        )


__all__ = [
    "ENVIRONMENT_SEED_SCHEMA",
    "RUNTIME_RANDOM_COMPONENTS",
    "EnvironmentSeedError",
    "EnvironmentSeedState",
    "TraceRandomness",
]
