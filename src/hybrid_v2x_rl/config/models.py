"""Typed configuration models for Hybrid RF/VLC RL.

The configuration tree is deliberately strict: unknown keys are rejected and
instances are immutable.  This prevents a misspelled research parameter from
silently falling back to a default and makes resolved configurations safe to
hash and attach to artifacts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER, PolicyActionName

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveInt = Annotated[int, Field(gt=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
OpenProbability = Annotated[float, Field(gt=0.0, lt=1.0)]

#: Subcarriers per NR resource block (TS 38.211).
SUBCARRIERS_PER_RESOURCE_BLOCK = 12

#: OFDM symbols per slot with normal cyclic prefix (TS 38.211).
OFDM_SYMBOLS_PER_SLOT = 14

#: Coded bits carried by one resource element, by modulation order.
BITS_PER_RESOURCE_ELEMENT: dict[str, int] = {"qpsk": 2, "16qam": 4, "64qam": 6}


class ConfigModel(BaseModel):
    """Base class shared by all strict, immutable configuration sections."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class ProjectMetadataConfig(ConfigModel):
    """Human-facing experiment identity."""

    name: str = "hybrid-rf-vlc-rl"
    experiment: str
    description: str = ""


class PathConfig(ConfigModel):
    """Project-relative artifact locations.

    ``loader.load_config`` resolves every value in this section against
    ``project_root``.  Paths are not required to exist during M0 because most
    point to artifacts produced by later milestones.
    """

    project_root: Path = Path(".")
    artifact_root: Path = Path("artifacts")
    calibration_root: Path = Path("artifacts/calibration")
    trace_root: Path = Path("artifacts/traces")
    checkpoint_root: Path = Path("artifacts/checkpoints")
    evaluation_root: Path = Path("artifacts/evaluations")
    figure_root: Path = Path("artifacts/figures")


class ServiceConfig(ConfigModel):
    """Packet generation, deadline, and reliability contract."""

    payload_bytes: PositiveInt
    generation_period_s: PositiveFloat
    deadline_s: PositiveFloat
    miss_budget: OpenProbability
    predecision_lead_s: NonNegativeFloat = 0.0001
    harq_enabled: bool = False
    rf_attempts_per_packet: PositiveInt = 1
    vlc_attempts_per_packet: PositiveInt = 1
    dup_simultaneous: bool = True
    rf_resource_assumption: Literal["pre_reserved", "semi_persistent"] = "pre_reserved"

    @property
    def information_bits(self) -> int:
        """Number of information bits handed to each link-specific PHY."""

        return self.payload_bytes * 8


class GridConfig(ConfigModel):
    """Controlled rectangular Manhattan-style topology."""

    avenues: Annotated[int, Field(ge=2)]
    cross_streets: Annotated[int, Field(ge=2)]
    avenue_spacing_m: PositiveFloat
    cross_street_spacing_m: PositiveFloat
    evaluation_avenues: Annotated[int, Field(ge=1)]
    evaluation_cross_streets: Annotated[int, Field(ge=1)]
    lanes_per_direction: PositiveInt = 2
    #: Manhattan's grid is predominantly one-way, alternating street by
    #: street.  It is not incidental: a two-way street can only carry a
    #: green wave at v = 2L/(kC), which is 1.36 m/s for 61 m blocks on a
    #: 90 s cycle, so the one-way conversion is what makes progression
    #: possible at all.  See work plan §4.1.
    one_way: bool = True
    lane_width_m: PositiveFloat = 3.5
    """Sets how far each direction runs from the street centreline.

    Physics-bearing: it moves every vehicle, so it belongs in the config hash.
    Opposing carriageways end up ``lanes_per_direction * lane_width_m`` apart.
    """

    @model_validator(mode="after")
    def evaluation_region_fits(self) -> GridConfig:
        if self.evaluation_avenues > self.avenues:
            raise ValueError("evaluation_avenues cannot exceed avenues")
        if self.evaluation_cross_streets > self.cross_streets:
            raise ValueError("evaluation_cross_streets cannot exceed cross_streets")
        return self


class TurnProbabilityConfig(ConfigModel):
    """Route-choice probabilities at an eligible intersection."""

    left: Probability
    straight: Probability
    right: Probability

    @model_validator(mode="after")
    def sums_to_one(self) -> TurnProbabilityConfig:
        total = self.left + self.straight + self.right
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"turn probabilities must sum to 1, got {total}")
        return self


class VehicleClassConfig(ConfigModel):
    """One vehicle class in the microscopic mobility mixture."""

    name: str
    share: Probability
    length_m: PositiveFloat
    width_m: PositiveFloat
    height_m: PositiveFloat


class MobilityConfig(ConfigModel):
    """Grid topology, timing, demand, and vehicle-mixture settings."""

    kind: Literal["synthetic_manhattan"]
    grid: GridConfig | None = None
    network_artifact: Path | None = None
    step_s: PositiveFloat
    speed_limit_mps: PositiveFloat
    target_densities_veh_per_lane_km: tuple[PositiveFloat, ...] = Field(min_length=1)
    density_tolerance_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    signal_cycle_s: PositiveFloat = 45.0
    stagger_signal_offsets: bool = True
    turn_probabilities: TurnProbabilityConfig
    # Reserved for a future arterial-flow variant; validated but read by no
    # code.  See configs/mobility/synthetic_manhattan.yaml for the rationale
    # and for why enabling it is an experimental decision, not a cleanup.
    straight_heavy_turn_probabilities: TurnProbabilityConfig
    warmup_s: NonNegativeFloat = 300.0
    trace_duration_s: PositiveFloat = 900.0
    boundary_arrivals: Literal["poisson"] = "poisson"
    vehicle_classes: tuple[VehicleClassConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_mobility_variant(self) -> MobilityConfig:
        if self.kind == "synthetic_manhattan" and self.grid is None:
            raise ValueError("synthetic_manhattan mobility requires grid")

        densities = self.target_densities_veh_per_lane_km
        if len(set(densities)) != len(densities):
            raise ValueError("target densities must be unique")
        if tuple(sorted(densities)) != densities:
            raise ValueError("target densities must be in ascending order")

        class_names = [vehicle.name for vehicle in self.vehicle_classes]
        if len(set(class_names)) != len(class_names):
            raise ValueError("vehicle class names must be unique")
        share = sum(vehicle.share for vehicle in self.vehicle_classes)
        if not math.isclose(share, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"vehicle class shares must sum to 1, got {share}")
        return self


class GeometryConfig(ConfigModel):
    """Tagged-pair and shared-occluder geometry."""

    tagged_pair_mode: Literal["longitudinal_following"] = "longitudinal_following"
    min_separation_m: PositiveFloat = 10.0
    max_separation_m: PositiveFloat = 60.0
    rf_antenna_height_m: PositiveFloat = 1.5
    headlamp_height_m: PositiveFloat = 0.7
    photodiode_height_m: PositiveFloat = 0.7
    exact_geometry_hidden_from_policy: bool = True
    include_building_nlos: bool = True

    @model_validator(mode="after")
    def valid_separation_range(self) -> GeometryConfig:
        if self.max_separation_m <= self.min_separation_m:
            raise ValueError("max_separation_m must exceed min_separation_m")
        return self


class PhyTimingConfig(ConfigModel):
    """Minimal link timing data used by the pre-training feasibility gate."""

    gross_bit_rate_bps: PositiveFloat
    airtime_s: PositiveFloat
    framing_overhead_bytes: Annotated[int, Field(ge=0)] = 0
    code_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0

    def coded_block_bits(self, payload_bytes: int) -> float:
        """Return coded bits including configured framing overhead."""

        uncoded_bits = 8.0 * (payload_bytes + self.framing_overhead_bytes)
        return uncoded_bits / self.code_rate

    @property
    def capacity_bits(self) -> float:
        """Gross bits available in the committed transmission interval."""

        return self.gross_bit_rate_bps * self.airtime_s


class RFConfig(ConfigModel):
    """NR sidelink channel and calibrated-surrogate configuration."""

    enabled: bool = True
    carrier_hz: PositiveFloat
    bandwidth_hz: PositiveFloat
    subcarrier_spacing_hz: PositiveFloat
    slot_s: PositiveFloat
    resource_blocks: PositiveInt
    modulation: Literal["qpsk", "16qam", "64qam"]
    resource_element_overhead: OpenProbability
    tx_power_dbm: float
    antenna_height_m: PositiveFloat
    resource_mode: Literal["nr_sidelink_mode2"]
    opportunity: Literal["pre_reserved", "semi_persistent"]
    propagation_family: Literal["3gpp_tr_37_885_urban_grid"]
    spatially_correlated_shadowing: bool = True
    temporally_correlated_fading: bool = True
    calibration_artifact: Path
    timing: PhyTimingConfig

    @property
    def slots_per_transmission(self) -> float:
        """Committed airtime expressed in slots."""

        return self.timing.airtime_s / self.slot_s

    def available_coded_bits(self) -> float:
        """Coded bits the resource grid can carry in the committed airtime.

        This is the physical ceiling that ``timing.gross_bit_rate_bps`` must not
        exceed.  Overhead covers the AGC symbol, the transmit/receive guard
        symbol, PSSCH DMRS, and PSCCH.
        """

        elements_per_slot = (
            self.resource_blocks * SUBCARRIERS_PER_RESOURCE_BLOCK * OFDM_SYMBOLS_PER_SLOT
        )
        usable_per_slot = elements_per_slot * (1.0 - self.resource_element_overhead)
        bits_per_element = BITS_PER_RESOURCE_ELEMENT[self.modulation]
        return usable_per_slot * self.slots_per_transmission * bits_per_element


class VLCConfig(ConfigModel):
    """Automotive IM/DD optical-link configuration."""

    enabled: bool = True
    electrical_bandwidth_hz: PositiveFloat
    modulation: Literal["ook", "pam"]
    modulation_order: Annotated[int, Field(ge=2)] = 2
    pulse_roll_off: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    headlamp_pattern: Literal["measured_non_lambertian"]
    pattern_artifact: Path
    receiver_fov_deg: Annotated[float, Field(gt=0.0, le=180.0)]
    ambient_condition: Literal["clear_night", "clear_day"]
    complete_blockage_main: bool = True
    residual_optical_floor_enabled: bool = False
    timing: PhyTimingConfig

    @model_validator(mode="after")
    def ook_is_binary(self) -> VLCConfig:
        if self.modulation == "ook" and self.modulation_order != 2:
            raise ValueError("ook requires modulation_order 2")
        if self.modulation_order & (self.modulation_order - 1):
            raise ValueError("modulation_order must be a power of two")
        return self

    @property
    def bits_per_symbol(self) -> float:
        return math.log2(self.modulation_order)

    def occupied_bandwidth_hz(self) -> float:
        """Baseband bandwidth the configured gross rate actually occupies.

        Raised-cosine shaping with roll-off ``beta`` occupies
        ``(1 + beta) * symbol_rate / 2``.  A configuration whose occupied
        bandwidth exceeds ``electrical_bandwidth_hz`` is not realizable: it
        would require zero excess bandwidth (ideal sinc pulses).
        """

        symbol_rate = self.timing.gross_bit_rate_bps / self.bits_per_symbol
        return (1.0 + self.pulse_roll_off) * symbol_rate / 2.0


class ObservationConfig(ConfigModel):
    """Causal policy observation and tracking assumptions."""

    history_packets: PositiveInt
    track_update_s: PositiveFloat
    track_latency_s: NonNegativeFloat
    position_noise_std_m: NonNegativeFloat
    speed_noise_std_mps: NonNegativeFloat
    #: Heading is *broadcast* rather than inferred, so its error is the
    #: transmitting vehicle's own GNSS/IMU attitude error rather than anything a
    #: tracker derives from successive positions.  Non-zero matters: work plan
    #: §4.6.1 found that a leader turning out of the acceptance cone causes most
    #: junction unavailability, and a policy given exact heading would see every
    #: turn perfectly at exactly the moment that decides the result.
    heading_noise_std_deg: NonNegativeFloat = 2.0
    #: How long a neighbour is retained after its last awareness message.
    #: Without a bound the RF-load proxy only ever rises, since a vehicle that
    #: drives away is counted for the rest of the episode.
    track_lifetime_s: PositiveFloat = 1.0
    forecast_horizon_s: PositiveFloat
    interpolate_tracks: bool = False
    action_dependent_link_feedback: bool = True
    features: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def features_are_unique(self) -> ObservationConfig:
        normalized = [feature.casefold() for feature in self.features]
        if len(set(normalized)) != len(normalized):
            raise ValueError("observation features must be unique")
        return self


class CostConfig(ConfigModel):
    """Normalized resource costs used by the population environment.

    ``rf_activation`` is the cost of one *reserved RF attempt*, not one RF
    packet.  The nine-action environment therefore derives every action cost
    as ``rf_activation * attempts + vlc_activation * uses_vlc``.  Keeping the
    two primitive coefficients here avoids a separate DUP price that could
    drift away from the resources the action actually reserves.
    """

    units: Literal["normalized_activation"] = "normalized_activation"
    rf_activation: PositiveFloat
    vlc_activation: PositiveFloat
    vlc_to_rf_sensitivity: tuple[PositiveFloat, ...] = (0.3, 0.5, 1.0)

class TraceSplitConfig(ConfigModel):
    """Immutable full-trajectory split identifiers."""

    train: tuple[str, ...] = Field(min_length=1)
    validation: tuple[str, ...] = Field(min_length=1)
    test: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def no_duplicates_within_split(self) -> TraceSplitConfig:
        for split_name in ("train", "validation", "test"):
            trace_ids = getattr(self, split_name)
            if len(set(trace_ids)) != len(trace_ids):
                raise ValueError(f"duplicate trace ID in {split_name} split")
        return self


class MeanFieldConfig(ConfigModel):
    """Causal population signal appended to every local observation."""

    signal: Literal["delayed_mean_rf_attempt_fraction"]
    delay_frames: Literal[1] = 1
    initial_value: Probability = 0.0
    include_validity_flag: bool = True

    @model_validator(mode="after")
    def frozen_reset_encoding(self) -> MeanFieldConfig:
        if self.initial_value != 0.0:
            raise ValueError("mean-field initial_value must be 0.0")
        if not self.include_validity_flag:
            raise ValueError("mean-field signal requires an explicit validity flag")
        return self


class ObservationNormalizationConfig(ConfigModel):
    """Training-only running normalization persisted with each checkpoint."""

    method: Literal["running_standardization"]
    update_scope: Literal["training_only"] = "training_only"
    epsilon: PositiveFloat = 1e-8
    clip_abs: PositiveFloat = 10.0


class EnvironmentConfig(ConfigModel):
    """Versioned population-environment and action semantics."""

    contract_version: Literal["1.0.0"]
    actions: tuple[PolicyActionName, ...] = POLICY_ACTION_ORDER
    max_rf_attempts: Literal[4] = 4
    no_observation_fallback_action: Literal["DUP-4"] = "DUP-4"
    mean_field: MeanFieldConfig
    normalization: ObservationNormalizationConfig
    episode_duration_s: PositiveFloat = 60.0
    packets_per_episode: PositiveInt = 600
    matched_random_tapes: bool = True
    sample_binary_outcomes: bool = True
    splits: TraceSplitConfig

    @model_validator(mode="after")
    def complete_action_set(self) -> EnvironmentConfig:
        if self.actions != POLICY_ACTION_ORDER:
            raise ValueError(
                "environment contract 1.0.0 requires the canonical nine-action order"
            )
        return self


class NetworkArchitectureConfig(ConfigModel):
    """Feed-forward actor and critic architecture."""

    actor_hidden_units: tuple[PositiveInt, ...] = (64, 64)
    reward_critic_hidden_units: tuple[PositiveInt, ...] = (64, 64)
    cost_critic_hidden_units: tuple[PositiveInt, ...] = (64, 64)
    activation: Literal["tanh"] = "tanh"
    recurrent: bool = False


class CurriculumStageConfig(ConfigModel):
    """One contiguous fraction of primal-dual training."""

    fraction: Annotated[float, Field(gt=0.0, le=1.0)]
    miss_budget: OpenProbability


class DensityMultiplierConfig(ConfigModel):
    """One density-specific nonnegative Lagrange multiplier."""

    density_veh_per_lane_km: PositiveFloat
    initial_value: NonNegativeFloat = 0.0
    learning_rate: PositiveFloat
    maximum: PositiveFloat


class TrainingConfig(ConfigModel):
    """Primal-dual PPO and reproducibility settings."""

    algorithm: Literal["primal_dual_ppo"]
    root_seed: Annotated[int, Field(ge=0, lt=2**64)]
    policy_seeds: tuple[Annotated[int, Field(ge=0, lt=2**64)], ...] = Field(
        min_length=1
    )
    total_transitions_per_seed: PositiveInt
    architecture: NetworkArchitectureConfig
    rollout_packets: PositiveInt
    minibatch_size: PositiveInt
    update_epochs: PositiveInt
    learning_rate: PositiveFloat
    gamma: Annotated[float, Field(gt=0.0, le=1.0)]
    gae_lambda: Annotated[float, Field(gt=0.0, le=1.0)]
    clip_ratio: Annotated[float, Field(gt=0.0, lt=1.0)]
    entropy_coefficient: NonNegativeFloat
    cost_signal: Literal["conditional_miss_probability"]
    curriculum: tuple[CurriculumStageConfig, ...] = Field(min_length=1)
    density_multipliers: tuple[DensityMultiplierConfig, ...] = Field(min_length=1)
    checkpoint_selection: Literal["reliability_then_cost"]
    normalize_observations_on_training_only: bool = True

    @model_validator(mode="after")
    def validate_training_schedule(self) -> TrainingConfig:
        if len(set(self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique")
        if self.minibatch_size > self.rollout_packets:
            raise ValueError("minibatch_size cannot exceed rollout_packets")
        fractions = sum(stage.fraction for stage in self.curriculum)
        if not math.isclose(fractions, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"curriculum stage fractions must sum to 1, got {fractions}")
        densities = [multiplier.density_veh_per_lane_km for multiplier in self.density_multipliers]
        if len(set(densities)) != len(densities):
            raise ValueError("density multiplier definitions must be unique")
        return self


class EvaluationConfig(ConfigModel):
    """Rare-event sampling and uncertainty requirements."""

    profile: Literal["primary", "ultra_reliability"]
    policies: tuple[str, ...] = Field(min_length=1)
    min_packets_per_policy_density: PositiveInt
    min_trajectory_pair_clusters: PositiveInt
    confidence_level: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.95
    one_sided_upper_bound: bool = True
    interval_method: Literal["trajectory_block_bootstrap"]
    bootstrap_replicates: Annotated[int, Field(ge=1_000)]
    matched_across_policies: bool = True
    report_raw_counts: bool = True
    require_upper_bound_below_budget: bool = True


class ProjectConfig(ConfigModel):
    """Complete, resolved Hybrid RF/VLC RL experiment configuration."""

    schema_version: Literal["1.1"]
    project: ProjectMetadataConfig
    mobility: MobilityConfig
    service: ServiceConfig
    geometry: GeometryConfig
    rf: RFConfig
    vlc: VLCConfig
    observation: ObservationConfig
    cost: CostConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathConfig
