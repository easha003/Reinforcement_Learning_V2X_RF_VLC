"""End-to-end synthetic mobility campaign for the analytic grid model.

Replaces the SUMO campaign driver.  The flow is the one declared in
`CODE_IMPLEMENTATION_SPEC.md` §24.1, minus the calibration stage, which no
longer exists because the analytic model sets the active vehicle count
directly:

    mobility config -> build network -> run warm-up and trace
      -> validate mobility -> extract tagged pairs -> immutable trace artifact

Orchestration only.  Every physical step is delegated to the module that
implements it, so `scripts/` stays thin per spec §23 and the Gate-1 verdict is
produced by :class:`MobilityValidator` rather than by checks invented here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from hybrid_v2x_rl.artifacts.store import ArtifactStore
from hybrid_v2x_rl.config.hashing import config_hash, scope_hash
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.core.errors import ConfigurationError
from hybrid_v2x_rl.core.randomness import derive_seed
from hybrid_v2x_rl.mobility.car_following import IDMParameters
from hybrid_v2x_rl.mobility.grid_network import GridNetwork, GridNetworkSpec
from hybrid_v2x_rl.mobility.grid_signals import SignalController, SignalProgram
from hybrid_v2x_rl.mobility.grid_simulator import (
    GridMobilitySimulator,
    GridMobilitySpec,
    TurnProbabilities,
)
from hybrid_v2x_rl.mobility.tagged_pairs import (
    TaggedPairConfig,
    TaggedPairExtractor,
    TaggedPairSegment,
    VehicleRecord,
)
from hybrid_v2x_rl.mobility.trace_io import (
    MobilityTraceWriter,
    SignalStateRecord,
    VehicleTraceRecord,
)
from hybrid_v2x_rl.mobility.validation import (
    MobilityFrame,
    MobilityValidationCriteria,
    MobilityValidationReport,
    MobilityValidator,
)
from hybrid_v2x_rl.mobility.vehicle_types import headline_vehicle_distribution

#: Splits a campaign generates, in the order they are produced.
SPLIT_NAMES = ("train", "validation", "test")


def trace_id_for(density: float, split: str, replicate: int, *, prefix: str = "synthetic") -> str:
    """Canonical trace identifier.

    Density alone is not enough: work-plan §13.1 splits by complete trajectory
    and seed, so each split needs its own mobility realizations.  Encoding
    split and replicate here keeps `configs/project/default.yaml` and the
    generator agreed on names without either hard-coding a list.
    """

    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    return f"{prefix}-d{density:g}-{split}-{replicate:03d}"


@dataclass(frozen=True, slots=True)
class SplitCounts:
    """How many independent mobility realizations each split receives."""

    train: int = 3
    validation: int = 1
    test: int = 3

    def __post_init__(self) -> None:
        if min(self.train, self.validation, self.test) < 1:
            raise ValueError("every split needs at least one realization")

    def as_pairs(self) -> tuple[tuple[str, int], ...]:
        return (("train", self.train), ("validation", self.validation), ("test", self.test))


@dataclass(frozen=True, slots=True)
class TraceCampaignPlan:
    """Inputs that are deliberately not part of the frozen project config."""

    output_root: Path
    code_version: str
    validation_sample_period_s: float = 1.0
    trace_id_prefix: str = "synthetic"

    def __post_init__(self) -> None:
        if not self.code_version.strip():
            raise ValueError("code_version must be non-empty")
        if not self.trace_id_prefix.strip():
            raise ValueError("trace_id_prefix must be non-empty")
        if self.validation_sample_period_s <= 0.0:
            raise ValueError("validation_sample_period_s must be positive")


@dataclass(frozen=True, slots=True)
class DensityCampaignResult:
    """One target density carried through to its Gate-1 verdict."""

    target_density_veh_per_lane_km: float
    trace_id: str
    split: str
    replicate: int
    artifact_path: Path
    realized_density_veh_per_lane_km: float
    pair_segments: tuple[TaggedPairSegment, ...]
    validation: MobilityValidationReport

    @property
    def degenerate_pair_count(self) -> int:
        """Episodes too short to carry a packet.

        A pair whose endpoints are observed once and then lose a vehicle to the
        boundary yields a zero-length episode.  It is a legitimate detection but
        useless for training, so it is counted rather than silently averaged
        into the duration statistics.
        """

        return sum(1 for segment in self.pair_segments if segment.duration_s <= 0.0)

    @property
    def passed(self) -> bool:
        return self.validation.gate1_passed

    def to_record(self) -> dict[str, object]:
        return {
            "target_density_veh_per_lane_km": self.target_density_veh_per_lane_km,
            "realized_density_veh_per_lane_km": self.realized_density_veh_per_lane_km,
            "trace_id": self.trace_id,
            "split": self.split,
            "replicate": self.replicate,
            "artifact_path": str(self.artifact_path),
            "passed": self.passed,
            "pair_count": len(self.pair_segments),
            "degenerate_pair_count": self.degenerate_pair_count,
            "usable_pair_count": len(self.pair_segments) - self.degenerate_pair_count,
            "validation": self.validation.to_record(),
        }


@dataclass(frozen=True, slots=True)
class TraceCampaignResult:
    """Every density in one campaign plus the aggregate Gate-1 verdict."""

    densities: tuple[DensityCampaignResult, ...]
    config_hash: str
    manifest_path: Path

    @property
    def gate1_passed(self) -> bool:
        return bool(self.densities) and all(item.passed for item in self.densities)


def _network_from_config(config: ProjectConfig) -> GridNetwork:
    grid = config.mobility.grid
    if grid is None:
        raise ConfigurationError("synthetic trace generation requires mobility.grid")
    return GridNetwork(
        GridNetworkSpec(
            avenues=grid.avenues,
            cross_streets=grid.cross_streets,
            avenue_spacing_m=grid.avenue_spacing_m,
            cross_street_spacing_m=grid.cross_street_spacing_m,
            lanes_per_direction=grid.lanes_per_direction,
            one_way=grid.one_way,
            lane_width_m=grid.lane_width_m,
            speed_limit_mps=config.mobility.speed_limit_mps,
        )
    )


class GridTracePipeline:
    """Drive one analytic mobility campaign across every target density."""

    def __init__(self, config: ProjectConfig, plan: TraceCampaignPlan) -> None:
        if config.mobility.kind != "synthetic_manhattan":
            raise ConfigurationError(
                f"analytic trace generation requires synthetic_manhattan mobility, "
                f"got {config.mobility.kind!r}"
            )
        self.config = config
        self.plan = plan
        self.network = _network_from_config(config)
        self.config_hash = config_hash(config)
        # Retain the legacy scope as provenance while writing the narrower,
        # consumer-facing replay contract used by current training code.
        self.mobility_config_hash = scope_hash(config, "mobility")
        self.mobility_trace_config_hash = scope_hash(config, "mobility_trace")

        turns = config.mobility.turn_probabilities
        self.turns = TurnProbabilities(left=turns.left, straight=turns.straight, right=turns.right)
        self.signals = SignalController(
            self.network,
            SignalProgram(
                cycle_s=config.mobility.signal_cycle_s,
                stagger_offsets=config.mobility.stagger_signal_offsets,
            ),
        )

    # -- validation criteria ---------------------------------------------

    def _criteria(self, target_density: float) -> MobilityValidationCriteria:
        mobility = self.config.mobility
        geometry = self.config.geometry
        return MobilityValidationCriteria(
            target_density_veh_per_lane_km=target_density,
            speed_limit_mps=mobility.speed_limit_mps,
            density_tolerance_fraction=mobility.density_tolerance_fraction,
            min_pair_separation_m=geometry.min_separation_m,
            max_pair_separation_m=geometry.max_separation_m,
            max_pair_duration_s=self.config.environment.episode_duration_s,
        )

    # -- one density ------------------------------------------------------

    def run_density(
        self,
        target_density: float,
        *,
        index: int = 0,
        split: str = "train",
        replicate: int = 0,
    ) -> DensityCampaignResult:
        """Run, archive, extract pairs, and produce a Gate-1 report."""

        mobility = self.config.mobility
        trace_id = trace_id_for(target_density, split, replicate, prefix=self.plan.trace_id_prefix)
        # Seeding on the full identity makes every (density, split, replicate)
        # an independent realization, which is what lets a held-out split be
        # genuinely held out rather than a relabelled copy.
        seed = derive_seed(self.config.training.root_seed, "mobility", trace_id=trace_id)

        simulator = GridMobilitySimulator(
            self.network,
            signals=self.signals,
            idm=IDMParameters(desired_speed_mps=mobility.speed_limit_mps),
            vehicle_distribution=headline_vehicle_distribution(),
        )
        spec = GridMobilitySpec(
            trace_id=trace_id,
            target_density_veh_per_lane_km=target_density,
            seed=seed % (2**63),
            step_s=mobility.step_s,
            warmup_s=mobility.warmup_s,
            duration_s=mobility.trace_duration_s,
            turn_probabilities=self.turns,
        )

        lane_km = self.network.total_lane_length_m / 1000.0
        sample_stride = max(1, round(self.plan.validation_sample_period_s / mobility.step_s))

        # Two passes, because a full-length campaign cannot be held in memory.
        # At 60 veh/lane-km a 900 s trace is 2,240 vehicles x 18,000 steps =
        # 40.3 million records, roughly 29 GB as Python objects.
        #
        # Pass one is cheap and builds only the small things the writer needs
        # up front (the route table) plus what validation and pair extraction
        # consume.  Pass two re-runs the simulation and streams records into
        # the writer, which chunks them to Parquet, so peak memory stays flat.
        # Re-running is safe precisely because the model is deterministic: the
        # same seed reproduces the same trajectories exactly.
        signal_records: list[SignalStateRecord] = []
        frames: list[MobilityFrame] = []
        routes: dict[str, list[str]] = {}
        pair_inputs: list[VehicleRecord] = []

        # ``run`` yields time measured from the end of warm-up, but signals are a
        # function of absolute simulation time.  Evaluating them at the recording
        # clock would archive phases the vehicles never experienced.
        for step, (time_s, vehicles) in enumerate(simulator.run(spec)):
            absolute_s = time_s + spec.warmup_s
            routes.update(simulator.route_table(vehicles))
            if step % sample_stride:
                continue

            signal_records.extend(simulator.signal_records(spec, absolute_s))
            speeds = tuple(vehicle.speed_mps for vehicle in vehicles)
            frames.append(
                MobilityFrame(
                    trace_id=trace_id,
                    time_s=time_s,
                    total_lane_length_km=lane_km,
                    vehicle_speeds_mps=speeds,
                    departed_vehicle_count=len(vehicles),
                    arrived_vehicle_count=len(vehicles),
                    signal_states=self.signals.phase_records(absolute_s),
                )
            )
            for vehicle in vehicles:
                edge = vehicle.edge
                x_m, y_m = edge.position_at(vehicle.offset_m, vehicle.lane_index)
                pair_inputs.append(
                    VehicleRecord(
                        trace_id=trace_id,
                        time_s=time_s,
                        vehicle_id=vehicle.vehicle_id,
                        x_m=x_m,
                        y_m=y_m,
                        heading_rad=edge.heading_rad,
                        route_id=vehicle.route_id,
                        lane_id=edge.lane_id(vehicle.lane_index),
                        planned_route=vehicle.planned_route_ids,
                    )
                )

        def _stream_vehicle_records() -> Iterator[VehicleTraceRecord]:
            """Second pass: yield records lazily so none are retained."""

            replay = GridMobilitySimulator(
                self.network,
                signals=self.signals,
                idm=IDMParameters(desired_speed_mps=mobility.speed_limit_mps),
                vehicle_distribution=headline_vehicle_distribution(),
            )
            for replay_time_s, replay_vehicles in replay.run(spec):
                yield from replay.vehicle_records(spec, replay_time_s, replay_vehicles)

        pairs = TaggedPairExtractor(
            TaggedPairConfig(
                min_separation_m=self.config.geometry.min_separation_m,
                max_separation_m=self.config.geometry.max_separation_m,
            )
        ).extract(pair_inputs)

        artifact = MobilityTraceWriter(ArtifactStore(self.plan.output_root)).write(
            trace_id=trace_id,
            vehicles=_stream_vehicle_records(),
            signals=signal_records,
            pairs=[{**segment.to_record(), "duration_s": segment.duration_s} for segment in pairs],
            network_definition=json.dumps(asdict(self.network.spec), indent=2, sort_keys=True),
            route_definition=json.dumps(routes, indent=2, sort_keys=True),
            resolved_config_yaml=self.config.model_dump_json(indent=2),
            config_hash=self.config_hash,
            config_scope_hashes={
                "mobility": self.mobility_config_hash,
                "mobility_trace": self.mobility_trace_config_hash,
            },
            code_version=self.plan.code_version,
            random_seeds={"mobility": spec.seed},
            software_versions={"mobility_model": "analytic-manhattan-grid"},
            notes=(
                f"warmup_s={spec.warmup_s:.12g}",
                f"saved_duration_s={spec.duration_s:.12g}",
                f"total_lane_length_m={self.network.total_lane_length_m:.12g}",
                "density is set directly; no closed-loop flow calibration is performed",
            ),
        )

        report = MobilityValidator(self._criteria(target_density)).validate(
            frames, pair_segments=pairs, manifest_archived=True
        )
        realized = sum(frame.density_veh_per_lane_km for frame in frames) / len(frames)

        return DensityCampaignResult(
            target_density_veh_per_lane_km=target_density,
            trace_id=trace_id,
            split=split,
            replicate=replicate,
            artifact_path=artifact.path,
            realized_density_veh_per_lane_km=realized,
            pair_segments=pairs,
            validation=report,
        )

    # -- whole campaign ---------------------------------------------------

    def run(
        self,
        *,
        target_densities: Sequence[float] | None = None,
        splits: SplitCounts | None = None,
    ) -> TraceCampaignResult:
        densities = tuple(
            target_densities
            if target_densities is not None
            else self.config.mobility.target_densities_veh_per_lane_km
        )
        if not densities:
            raise ConfigurationError("at least one target density is required")

        counts = splits or SplitCounts()
        results = tuple(
            self.run_density(density, split=split, replicate=replicate)
            for split, number in counts.as_pairs()
            for replicate in range(number)
            for density in densities
        )

        manifest_path = self.plan.output_root / "campaign_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": "hybrid-rf-vlc-rl.mobility-campaign.v2",
                    "mobility_model": "analytic-manhattan-grid",
                    "config_hash": self.config_hash,
                    "code_version": self.plan.code_version,
                    "network": {
                        "spec": asdict(self.network.spec),
                        "total_lane_length_m": self.network.total_lane_length_m,
                        "signalized_junction_count": len(self.network.signalized_junction_ids),
                    },
                    "splits": {
                        name: [r.trace_id for r in results if r.split == name]
                        for name in SPLIT_NAMES
                    },
                    "gate1_passed": all(item.passed for item in results),
                    "densities": [item.to_record() for item in results],
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        return TraceCampaignResult(
            densities=results,
            config_hash=self.config_hash,
            manifest_path=manifest_path,
        )


__all__ = [
    "SPLIT_NAMES",
    "DensityCampaignResult",
    "SplitCounts",
    "GridTracePipeline",
    "TraceCampaignPlan",
    "TraceCampaignResult",
    "trace_id_for",
]
