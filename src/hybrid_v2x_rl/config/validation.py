"""Cross-section configuration validation.

Pydantic validates local section invariants in ``models.py``.  This module owns
checks that require two or more sections, including the frozen headline
contract and pre-training PHY feasibility.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from hybrid_v2x_rl.config.models import PhyTimingConfig, ProjectConfig
from hybrid_v2x_rl.core.errors import ConfigurationError
from hybrid_v2x_rl.core.policy_actions import POLICY_ACTION_ORDER

FORBIDDEN_OBSERVATION_FIELDS = frozenset(
    {
        "counterfactual_outcomes",
        "exact_blocker_flag",
        "exact_future_blockage",
        "exact_rf_channel_state",
        "exact_simulator_geometry",
        "exact_vehicle_footprints",
        "exact_vlc_channel_state",
        "future_blocker_state",
        "simulator_future",
    }
)


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return bool(left == right)


def _require_headline(field: str, actual: Any, expected: Any) -> None:
    if not _equal(actual, expected):
        raise ConfigurationError(
            f"headline contract violation: {field} must be {expected!r}, got {actual!r}"
        )


def _validate_headline_contract(config: ProjectConfig) -> None:
    """Protect the assumptions frozen in the implementation specification."""

    if config.project.experiment != "headline":
        return

    _require_headline("service.payload_bytes", config.service.payload_bytes, 300)
    _require_headline(
        "service.generation_period_s",
        config.service.generation_period_s,
        0.1,
    )
    _require_headline("service.deadline_s", config.service.deadline_s, 0.003)
    _require_headline("service.miss_budget", config.service.miss_budget, 1e-4)
    _require_headline("service.harq_enabled", config.service.harq_enabled, False)
    # Amended 2026-08-10 from 1 to 3, deliberately and with the arithmetic
    # recorded, because this validator exists to make exactly that a decision
    # rather than a drift.  Collision alone puts a 1.5-12% floor on a single
    # attempt, so one attempt could not meet 1e-4 at any density or any end of
    # the declared sensitivity band. Three pre-reserved attempts fit the timing
    # budget because 16QAM carries the block in one 0.5 ms slot. Under the
    # corrected full-carrier contract they repeat on the same carrier at the
    # 10 MHz headline point; the four-attempt policy actions and subsequent
    # feasibility gate determine whether that architecture is sufficient.
    _require_headline(
        "service.rf_attempts_per_packet",
        config.service.rf_attempts_per_packet,
        3,
    )
    _require_headline(
        "service.vlc_attempts_per_packet",
        config.service.vlc_attempts_per_packet,
        1,
    )
    _require_headline("service.dup_simultaneous", config.service.dup_simultaneous, True)
    _require_headline(
        "service.rf_resource_assumption",
        config.service.rf_resource_assumption,
        "pre_reserved",
    )

    grid = config.mobility.grid
    if grid is None:
        raise ConfigurationError("headline contract requires a synthetic grid")
    _require_headline("mobility.kind", config.mobility.kind, "synthetic_manhattan")
    _require_headline("mobility.grid.avenues", grid.avenues, 6)
    _require_headline("mobility.grid.cross_streets", grid.cross_streets, 12)
    _require_headline("mobility.grid.avenue_spacing_m", grid.avenue_spacing_m, 244.0)
    _require_headline(
        "mobility.grid.cross_street_spacing_m",
        grid.cross_street_spacing_m,
        61.0,
    )
    _require_headline("mobility.step_s", config.mobility.step_s, 0.05)
    _require_headline("mobility.speed_limit_mps", config.mobility.speed_limit_mps, 11.18)
    _require_headline(
        "mobility.target_densities_veh_per_lane_km",
        config.mobility.target_densities_veh_per_lane_km,
        (10.0, 20.0, 30.0),
    )

    _require_headline("cost.rf_activation", config.cost.rf_activation, 1.0)
    _require_headline("cost.vlc_activation", config.cost.vlc_activation, 1.0)
    _require_headline("environment.contract_version", config.environment.contract_version, "1.0.0")
    _require_headline("environment.actions", config.environment.actions, POLICY_ACTION_ORDER)
    _require_headline("environment.max_rf_attempts", config.environment.max_rf_attempts, 4)
    _require_headline(
        "geometry.tagged_pair_mode",
        config.geometry.tagged_pair_mode,
        "longitudinal_following",
    )
    _require_headline(
        "geometry.exact_geometry_hidden_from_policy",
        config.geometry.exact_geometry_hidden_from_policy,
        True,
    )


def _validate_phy_timing(
    *,
    link_name: str,
    timing: PhyTimingConfig,
    payload_bytes: int,
    deadline_s: float,
) -> None:
    if timing.airtime_s > deadline_s + 1e-15:
        raise ConfigurationError(
            f"{link_name} airtime {timing.airtime_s:g} s exceeds service deadline {deadline_s:g} s"
        )
    required_bits = timing.coded_block_bits(payload_bytes)
    if required_bits > timing.capacity_bits + 1e-9:
        raise ConfigurationError(
            f"{link_name} PHY is infeasible: coded block requires "
            f"{required_bits:g} bits but configured airtime carries only "
            f"{timing.capacity_bits:g} bits"
        )


def _validate_committed_airtime(config: ProjectConfig) -> None:
    """Refuse a profile whose granted attempts cannot fit inside the deadline.

    :func:`_validate_phy_timing` compares *one* attempt against the deadline,
    which was sufficient while the profile granted one.  Now that the attempt
    count is the parameter reliability is bought with, the binding quantity is
    what a packet commits in total, and a profile that overran the deadline
    would load cleanly and fail much later at
    :class:`~hybrid_v2x_rl.env.packet.PacketLifecycle` construction.

    The two media run concurrently -- different physical channels -- so the
    binding quantity is the longer leg rather than the sum.  That is the same
    inequality ``env.packet.Timing.check_feasible`` applies.  It is repeated
    here rather than imported because configuration validation sits below the
    environment and must not depend on it; the two are pinned to each other by
    a test, which is where a cross-layer invariant belongs.
    """

    service = config.service
    available = service.deadline_s - service.predecision_lead_s
    if available <= 0.0:
        raise ConfigurationError(
            f"service.predecision_lead_s {service.predecision_lead_s:g} s leaves no "
            f"time inside the {service.deadline_s:g} s deadline"
        )

    # ``Timing`` carries no VLC attempt count, so a profile asking for more than
    # one optical attempt would have it silently ignored -- the packet would be
    # evaluated with one.  Rejected rather than tolerated: the field is in the
    # schema, and a research parameter that does nothing is worse than one that
    # is absent.
    if service.vlc_attempts_per_packet != 1:
        raise ConfigurationError(
            "service.vlc_attempts_per_packet must be 1: the packet lifecycle "
            "grants the optical leg a single attempt and would ignore "
            f"{service.vlc_attempts_per_packet}"
        )

    committed = 0.0
    rf_attempts = 0
    if config.rf.enabled:
        # The inherited three-action lifecycle still reads the service-level
        # attempt count.  Environment contract 1.0.0 adds RF-1 through RF-4,
        # so both interfaces must fit until Phase 3 replaces the former.
        rf_attempts = max(
            service.rf_attempts_per_packet,
            config.environment.max_rf_attempts,
        )
        committed = max(
            committed, config.rf.timing.airtime_s * rf_attempts
        )
    if config.vlc.enabled:
        committed = max(committed, config.vlc.timing.airtime_s)
    if committed > available + 1e-15:
        raise ConfigurationError(
            f"committed airtime {committed:g} s exceeds the {available:g} s the "
            f"{service.deadline_s:g} s deadline leaves after a "
            f"{service.predecision_lead_s:g} s pre-decision lead "
            f"(maximum_rf_attempts={rf_attempts})"
        )


def _validate_rf_resource_grid(config: ProjectConfig) -> None:
    """Reject an RF rate the resource grid cannot physically deliver.

    ``_validate_phy_timing`` only compares the coded block against
    ``gross_bit_rate_bps * airtime_s``.  That check passes for any declared
    rate, however large.  This check bounds the declared rate by the resource
    elements the configured bandwidth, numerology, and modulation actually
    provide, so an unachievable MCS choice fails at load time.

    Bounding the rate is sufficient.  ``_validate_phy_timing`` has already
    established ``coded_block <= capacity``, so ``capacity <= available``
    transitively guarantees the block fits the grid.
    """

    rf = config.rf
    available = rf.available_coded_bits()
    if rf.timing.capacity_bits > available + 1e-9:
        raise ConfigurationError(
            f"RF gross rate is unachievable: configured airtime claims "
            f"{rf.timing.capacity_bits:g} coded bits but the resource grid "
            f"({rf.resource_blocks} RB, {rf.modulation}, "
            f"{rf.resource_element_overhead:.0%} overhead, "
            f"{rf.slots_per_transmission:g} slot(s)) provides only "
            f"{available:g} bits"
        )


def _validate_vlc_bandwidth(config: ProjectConfig) -> None:
    """Reject a VLC rate that does not fit the declared optical front end."""

    vlc = config.vlc
    occupied = vlc.occupied_bandwidth_hz()
    if occupied > vlc.electrical_bandwidth_hz + 1e-6:
        raise ConfigurationError(
            f"VLC gross rate is unrealizable: {vlc.timing.gross_bit_rate_bps:g} bit/s "
            f"at {vlc.modulation} with roll-off {vlc.pulse_roll_off:g} occupies "
            f"{occupied:g} Hz, exceeding the {vlc.electrical_bandwidth_hz:g} Hz "
            f"electrical bandwidth"
        )


def _overlap(left: Iterable[str], right: Iterable[str]) -> set[str]:
    return set(left).intersection(right)


def validate_project_config(config: ProjectConfig) -> ProjectConfig:
    """Validate all cross-section invariants and return ``config`` unchanged.

    Returning the configuration makes this function convenient in loader
    pipelines without weakening its read-only semantics.
    """

    _validate_headline_contract(config)

    if (
        config.mobility.step_s > config.observation.track_update_s
        and not config.observation.interpolate_tracks
    ):
        raise ConfigurationError(
            "mobility.step_s exceeds observation.track_update_s without track interpolation"
        )

    features = {feature.casefold() for feature in config.observation.features}
    forbidden = sorted(features.intersection(FORBIDDEN_OBSERVATION_FIELDS))
    if forbidden:
        raise ConfigurationError(
            "policy observation requests forbidden hidden field(s): " + ", ".join(forbidden)
        )

    splits = config.environment.splits
    overlaps = {
        "train/validation": _overlap(splits.train, splits.validation),
        "train/test": _overlap(splits.train, splits.test),
        "validation/test": _overlap(splits.validation, splits.test),
    }
    for split_pair, trace_ids in overlaps.items():
        if trace_ids:
            joined = ", ".join(sorted(trace_ids))
            raise ConfigurationError(f"trace split overlap in {split_pair}: {joined}")

    training_densities = {
        multiplier.density_veh_per_lane_km for multiplier in config.training.density_multipliers
    }
    mobility_densities = set(config.mobility.target_densities_veh_per_lane_km)
    if training_densities != mobility_densities:
        missing = sorted(mobility_densities - training_densities)
        unexpected = sorted(training_densities - mobility_densities)
        raise ConfigurationError(
            "density multiplier definitions do not match training densities; "
            f"missing={missing}, unexpected={unexpected}"
        )

    final_budget = config.training.curriculum[-1].miss_budget
    if not math.isclose(
        final_budget,
        config.service.miss_budget,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ConfigurationError("final curriculum miss budget must equal service.miss_budget")

    if config.rf.enabled:
        _validate_phy_timing(
            link_name="RF",
            timing=config.rf.timing,
            payload_bytes=config.service.payload_bytes,
            deadline_s=config.service.deadline_s,
        )
        _validate_rf_resource_grid(config)
        if config.rf.opportunity != config.service.rf_resource_assumption:
            raise ConfigurationError("rf.opportunity must match service.rf_resource_assumption")
    if config.vlc.enabled:
        _validate_phy_timing(
            link_name="VLC",
            timing=config.vlc.timing,
            payload_bytes=config.service.payload_bytes,
            deadline_s=config.service.deadline_s,
        )
        _validate_vlc_bandwidth(config)

    _validate_committed_airtime(config)

    expected_packets = round(
        config.environment.episode_duration_s / config.service.generation_period_s
    )
    if config.environment.packets_per_episode != expected_packets:
        raise ConfigurationError(
            "environment.packets_per_episode does not match episode duration "
            "and packet generation period"
        )

    is_ultra = config.service.miss_budget <= 1e-5 + 1e-15
    if is_ultra:
        if config.evaluation.profile != "ultra_reliability":
            raise ConfigurationError(
                "a 1e-5 reliability target requires the ultra_reliability evaluation profile"
            )
        if config.evaluation.min_packets_per_policy_density < 10_000_000:
            raise ConfigurationError(
                "a 1e-5 reliability target requires at least 10,000,000 "
                "packets per policy-density condition"
            )
    elif config.evaluation.min_packets_per_policy_density < 1_000_000:
        raise ConfigurationError(
            "primary evaluation requires at least 1,000,000 packets per policy-density condition"
        )

    if config.evaluation.min_trajectory_pair_clusters < 200:
        raise ConfigurationError("evaluation requires at least 200 trajectory/pair clusters")
    if not config.evaluation.one_sided_upper_bound:
        raise ConfigurationError("evaluation must use a one-sided reliability upper bound")

    return config
