"""Pair-aligned reward, sampled miss, and conditional-risk outcomes.

This is the Phase 5 boundary at which an action ledger, the current shared
RF-pool response, selected-link channel evaluations, and one matched packet
tape per active pair become training targets.  Resource reward remains an
action-accounting quantity: every reserved RF attempt and VLC activation is
charged even when a packet succeeds early.  Reliability is represented twice:

* ``sampled_miss_cost`` is the realized binary CMDP cost used for evaluation;
* ``conditional_miss_probability`` is the selected action's miss probability
  after current load and physical state are known, used as the lower-variance
  training signal named by environment contract 1.0.0.

The conditional RF packet risk is the product of the identical per-attempt
risks across the reserved retry count.  Each retry consumes its own independent
mechanism draws from the matched tape.  A DUP action misses only when both its
RF prefix and its single VLC leg miss, so its conditional risk is their product.
All raw risk and mechanism records remain simulator diagnostics; this module
does not modify either actor or critic observation tensors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult
from hybrid_v2x_rl.core.enums import FailureCause
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.core.policy_actions import PolicyAction, action_resources
from hybrid_v2x_rl.mean_field.action_ledger import (
    FrameActionLedger,
    PairResourceAccounting,
)
from hybrid_v2x_rl.mean_field.random_tape import MatchedPacketTape
from hybrid_v2x_rl.mean_field.rf_pool import (
    RFAttemptRisk,
    RFPoolDemand,
    RFPoolResponse,
)

OutcomeArray: TypeAlias = NDArray[np.float32]
OutcomeInfo: TypeAlias = Mapping[str, object]


class PacketOutcomeError(HybridV2XError):
    """Selected packet outcomes do not reconcile with their frame inputs."""


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _freeze_float32(values: tuple[float, ...]) -> OutcomeArray:
    array = np.asarray(values, dtype=np.float32)
    array.setflags(write=False)
    return array


def _validate_exact_keys(
    supplied: Mapping[str, object],
    expected: tuple[str, ...],
    *,
    name: str,
) -> None:
    if not isinstance(supplied, Mapping):
        raise PacketOutcomeError(f"{name} must be a pair-ID mapping")
    invalid = tuple(repr(key) for key in supplied if not isinstance(key, str) or not key)
    if invalid:
        raise PacketOutcomeError(
            f"{name} contains invalid pair IDs",
            context={"invalid_pair_ids": invalid},
        )
    supplied_ids = set(supplied)
    expected_ids = set(expected)
    missing = tuple(pair_id for pair_id in expected if pair_id not in supplied_ids)
    unexpected = tuple(sorted(supplied_ids - expected_ids))
    if missing or unexpected:
        raise PacketOutcomeError(
            f"{name} does not cover its selected pairs exactly",
            context={"missing_pair_ids": missing, "unexpected_pair_ids": unexpected},
        )


@dataclass(frozen=True, slots=True)
class RFAttemptOutcome:
    """One actually evaluated RF retry and the mechanism that decided it."""

    attempt_index: int
    success: bool
    half_duplex_failure: bool
    collision_failure: bool
    decoding_failure: bool
    failure_cause: FailureCause

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt_index, int)
            or isinstance(self.attempt_index, bool)
            or not 0 <= self.attempt_index < 4
        ):
            raise PacketOutcomeError("RF attempt_index must lie in [0, 3]")
        flags = (
            self.success,
            self.half_duplex_failure,
            self.collision_failure,
            self.decoding_failure,
        )
        if any(type(flag) is not bool for flag in flags):
            raise PacketOutcomeError("RF attempt outcome flags must be booleans")
        failures = sum(flags[1:])
        if self.success != (failures == 0) or failures > 1:
            raise PacketOutcomeError(
                "an RF attempt must be either successful or have one terminal mechanism"
            )
        expected_cause = FailureCause.NONE
        if self.half_duplex_failure or self.collision_failure:
            expected_cause = FailureCause.RF_COLLISION
        elif self.decoding_failure:
            expected_cause = FailureCause.RF_CHANNEL
        if self.failure_cause is not expected_cause:
            raise PacketOutcomeError(
                "RF attempt failure cause does not match its terminal mechanism"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "success": self.success,
            "half_duplex_failure": self.half_duplex_failure,
            "collision_failure": self.collision_failure,
            "decoding_failure": self.decoding_failure,
            "failure_cause": self.failure_cause.value,
        }


@dataclass(frozen=True, slots=True)
class PairPacketOutcome:
    """One selected action's immutable CMDP targets and simulator diagnostics."""

    pair_id: str
    action: PolicyAction
    reward: float
    delivered: bool
    sampled_miss_cost: int
    conditional_miss_probability: float
    failure_cause: FailureCause
    rf_delivered: bool
    vlc_delivered: bool
    rf_attempts: tuple[RFAttemptOutcome, ...]
    rf_attempt_risk: RFAttemptRisk | None
    rf_packet_miss_probability: float | None
    vlc_result: VLCChannelResult | None
    vlc_miss_probability: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id.strip():
            raise PacketOutcomeError("pair_id must be a non-empty string")
        if type(self.action) is not PolicyAction:
            raise PacketOutcomeError("pair outcome requires an exact PolicyAction")
        if not math.isfinite(self.reward) or self.reward >= 0.0:
            raise PacketOutcomeError("packet reward must be finite and negative")
        if any(
            type(flag) is not bool
            for flag in (self.delivered, self.rf_delivered, self.vlc_delivered)
        ):
            raise PacketOutcomeError("packet delivery fields must be booleans")
        if self.sampled_miss_cost not in (0, 1) or isinstance(
            self.sampled_miss_cost, bool
        ):
            raise PacketOutcomeError("sampled_miss_cost must be binary")
        if self.sampled_miss_cost != int(not self.delivered):
            raise PacketOutcomeError("sampled miss cost must be one exactly on a miss")
        if self.delivered != (self.rf_delivered or self.vlc_delivered):
            raise PacketOutcomeError("packet delivery must equal delivery by either selected leg")
        if not _is_probability(self.conditional_miss_probability):
            raise PacketOutcomeError(
                "conditional_miss_probability must be finite and lie in [0, 1]"
            )

        spec = action_resources(self.action)
        if spec.uses_rf:
            if not isinstance(self.rf_attempt_risk, RFAttemptRisk) or not _is_probability(
                self.rf_packet_miss_probability
            ):
                raise PacketOutcomeError("an RF action requires RF risk diagnostics")
            if not self.rf_attempts or len(self.rf_attempts) > spec.reserved_rf_attempts:
                raise PacketOutcomeError(
                    "RF evaluated-attempt count must lie within the reservation"
                )
            if tuple(row.attempt_index for row in self.rf_attempts) != tuple(
                range(len(self.rf_attempts))
            ):
                raise PacketOutcomeError("RF evaluated attempts must form a zero-based prefix")
            if self.rf_delivered:
                if not self.rf_attempts[-1].success or any(
                    row.success for row in self.rf_attempts[:-1]
                ):
                    raise PacketOutcomeError("RF delivery must stop at the first successful retry")
            elif len(self.rf_attempts) != spec.reserved_rf_attempts or any(
                row.success for row in self.rf_attempts
            ):
                raise PacketOutcomeError("an RF miss must exhaust every reserved retry")
            expected_rf = self.rf_attempt_risk.total_failure_probability ** (
                spec.reserved_rf_attempts
            )
            if not math.isclose(
                cast(float, self.rf_packet_miss_probability),
                expected_rf,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise PacketOutcomeError("RF packet risk does not match its reserved retries")
        elif (
            self.rf_attempt_risk is not None
            or self.rf_packet_miss_probability is not None
            or self.rf_attempts
            or self.rf_delivered
        ):
            raise PacketOutcomeError("a VLC-only action cannot carry RF outcomes")

        if spec.uses_vlc:
            if not isinstance(self.vlc_result, VLCChannelResult) or not _is_probability(
                self.vlc_miss_probability
            ):
                raise PacketOutcomeError("a VLC action requires VLC diagnostics")
            if self.vlc_delivered != self.vlc_result.success:
                raise PacketOutcomeError("VLC delivery does not match its channel result")
        elif (
            self.vlc_result is not None
            or self.vlc_miss_probability is not None
            or self.vlc_delivered
        ):
            raise PacketOutcomeError("an RF-only action cannot carry VLC outcomes")

        expected_conditional = 1.0
        if spec.uses_rf:
            expected_conditional *= cast(float, self.rf_packet_miss_probability)
        if spec.uses_vlc:
            expected_conditional *= cast(float, self.vlc_miss_probability)
        if not math.isclose(
            self.conditional_miss_probability,
            expected_conditional,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise PacketOutcomeError(
                "conditional miss probability does not match selected-link risks"
            )

        if self.delivered:
            expected_cause = FailureCause.NONE
        elif spec.duplicates:
            expected_cause = FailureCause.JOINT_FAILURE
        elif spec.uses_rf:
            expected_cause = self.rf_attempts[-1].failure_cause
        else:
            assert self.vlc_result is not None
            expected_cause = self.vlc_result.failure_cause
        if self.failure_cause is not expected_cause:
            raise PacketOutcomeError("packet failure cause does not match selected outcomes")

    @property
    def rf_attempts_used(self) -> int:
        return len(self.rf_attempts)

    def as_dict(self) -> dict[str, object]:
        """Return explicit diagnostics without flattening them into policy inputs."""

        return {
            "pair_id": self.pair_id,
            "action_index": int(self.action),
            "action_name": self.action.label,
            "reward": self.reward,
            "delivered": self.delivered,
            "sampled_miss_cost": self.sampled_miss_cost,
            "conditional_miss_probability": self.conditional_miss_probability,
            "failure_cause": self.failure_cause.value,
            "rf_delivered": self.rf_delivered,
            "vlc_delivered": self.vlc_delivered,
            "rf_attempts_used": self.rf_attempts_used,
            "rf_attempts": [row.as_dict() for row in self.rf_attempts],
            "rf_packet_miss_probability": self.rf_packet_miss_probability,
            "vlc_miss_probability": self.vlc_miss_probability,
            "rf_risk": (
                self.rf_attempt_risk.as_dict()
                if self.rf_attempt_risk is not None
                else None
            ),
            "vlc_diagnostics": (
                {
                    "received_power_w": self.vlc_result.received_power_w,
                    "electrical_snr": self.vlc_result.electrical_snr,
                    "within_field_of_view": self.vlc_result.within_field_of_view,
                    "beam_aimed": self.vlc_result.beam_aimed,
                    "occluded": self.vlc_result.occluded,
                    "bit_error_rate": self.vlc_result.bit_error_rate,
                    "decoding_failure_probability": (
                        self.vlc_result.decoding_failure_probability
                    ),
                    "total_failure_probability": (
                        self.vlc_result.total_failure_probability
                    ),
                    "success": self.vlc_result.success,
                    "failure_cause": self.vlc_result.failure_cause.value,
                }
                if self.vlc_result is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class FramePacketOutcomes:
    """Contract-shaped target arrays aligned to one action frame's pair IDs."""

    trace_id: str
    frame_index: int
    time_s: float
    pair_ids: tuple[str, ...]
    rewards: OutcomeArray
    sampled_miss_costs: OutcomeArray
    conditional_miss_probabilities: OutcomeArray
    pair_outcomes: tuple[PairPacketOutcome, ...]
    rf_pool_response: RFPoolResponse

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise PacketOutcomeError("trace_id must be a non-empty string")
        if (
            not isinstance(self.frame_index, int)
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
        ):
            raise PacketOutcomeError("frame_index must be a non-negative integer")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise PacketOutcomeError("time_s must be finite and non-negative")
        if self.pair_ids != tuple(sorted(self.pair_ids)) or len(self.pair_ids) != len(
            set(self.pair_ids)
        ):
            raise PacketOutcomeError("pair_ids must be unique and canonically ordered")
        if tuple(row.pair_id for row in self.pair_outcomes) != self.pair_ids:
            raise PacketOutcomeError("pair outcome rows do not align with pair_ids")
        if not isinstance(self.rf_pool_response, RFPoolResponse):
            raise PacketOutcomeError("frame outcomes require the current RF-pool response")

        population = len(self.pair_ids)
        expected_arrays = {
            "rewards": tuple(row.reward for row in self.pair_outcomes),
            "sampled_miss_costs": tuple(
                float(row.sampled_miss_cost) for row in self.pair_outcomes
            ),
            "conditional_miss_probabilities": tuple(
                row.conditional_miss_probability for row in self.pair_outcomes
            ),
        }
        for name, expected in expected_arrays.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
                raise PacketOutcomeError(f"{name} must be a float32 ndarray")
            if value.shape != (population,):
                raise PacketOutcomeError(f"{name} must have shape (N_t,)")
            if not bool(np.all(np.isfinite(value))):
                raise PacketOutcomeError(f"{name} must contain only finite values")
            if not np.array_equal(value, np.asarray(expected, dtype=np.float32)):
                raise PacketOutcomeError(f"{name} does not match the pair outcome rows")
            frozen = value.copy(order="C")
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)

        if not bool(
            np.all(
                (self.sampled_miss_costs == 0.0)
                | (self.sampled_miss_costs == 1.0)
            )
        ):
            raise PacketOutcomeError("sampled miss costs must be binary")
        if not bool(
            np.all(
                (self.conditional_miss_probabilities >= 0.0)
                & (self.conditional_miss_probabilities <= 1.0)
            )
        ):
            raise PacketOutcomeError("conditional miss probabilities must lie in [0, 1]")

    @property
    def population_size(self) -> int:
        return len(self.pair_ids)

    def as_step_info(self) -> OutcomeInfo:
        """Expose costs and diagnostics through ``info``, never observations."""

        return MappingProxyType(
            {
                "transition_pair_ids": self.pair_ids,
                "sampled_miss_cost": self.sampled_miss_costs,
                "conditional_miss_probability": self.conditional_miss_probabilities,
                "packet_outcomes": self.pair_outcomes,
                "rf_pool": self.rf_pool_response,
            }
        )


def _evaluate_rf_attempt(
    *,
    attempt_index: int,
    tape: MatchedPacketTape,
    risk: RFAttemptRisk,
) -> RFAttemptOutcome:
    draws = tape.rf_attempts[attempt_index]
    half_duplex_failure = draws.half_duplex_draw < risk.half_duplex_probability
    collision_failure = False
    decoding_failure = False
    if not half_duplex_failure:
        collision_failure = draws.collision_draw < risk.collision_probability
    if not half_duplex_failure and not collision_failure:
        decoding_failure = (
            draws.decoding_draw < risk.decoding_failure_probability
        )
    success = not (
        half_duplex_failure or collision_failure or decoding_failure
    )
    if half_duplex_failure or collision_failure:
        cause = FailureCause.RF_COLLISION
    elif decoding_failure:
        cause = FailureCause.RF_CHANNEL
    else:
        cause = FailureCause.NONE
    return RFAttemptOutcome(
        attempt_index=attempt_index,
        success=success,
        half_duplex_failure=half_duplex_failure,
        collision_failure=collision_failure,
        decoding_failure=decoding_failure,
        failure_cause=cause,
    )


def _validate_vlc_result(
    *,
    pair_id: str,
    result: VLCChannelResult,
    tape: MatchedPacketTape,
) -> None:
    if not isinstance(result, VLCChannelResult):
        raise PacketOutcomeError(
            "VLC results must contain VLCChannelResult rows",
            context={"pair_id": pair_id},
        )
    for name in ("decoding_failure_probability", "total_failure_probability"):
        if not _is_probability(getattr(result, name)):
            raise PacketOutcomeError(
                f"VLC {name} must be finite and lie in [0, 1]",
                context={"pair_id": pair_id},
            )
    if not math.isclose(
        result.total_failure_probability,
        result.decoding_failure_probability,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise PacketOutcomeError(
            "VLC total and decoding failure probabilities must agree",
            context={"pair_id": pair_id},
        )

    if result.is_geometric_failure:
        expected_success = False
        expected_cause = (
            FailureCause.VLC_OCCLUSION
            if result.occluded
            else FailureCause.VLC_ALIGNMENT
        )
        if result.total_failure_probability != 1.0:
            raise PacketOutcomeError(
                "a geometric VLC failure must have unit miss probability",
                context={"pair_id": pair_id},
            )
    else:
        expected_success = not (
            tape.vlc.decoding_draw < result.decoding_failure_probability
        )
        expected_cause = (
            FailureCause.NONE if expected_success else FailureCause.VLC_CHANNEL
        )
    if result.success != expected_success or result.failure_cause is not expected_cause:
        raise PacketOutcomeError(
            "VLC sampled outcome does not match the packet's tape and risk",
            context={"pair_id": pair_id},
        )


def _assemble_pair_outcome(
    *,
    accounting: PairResourceAccounting,
    tape: MatchedPacketTape,
    rf_risk: RFAttemptRisk | None,
    vlc_result: VLCChannelResult | None,
) -> PairPacketOutcome:
    spec = action_resources(accounting.action)
    rf_attempts: list[RFAttemptOutcome] = []
    rf_delivered = False
    rf_packet_probability: float | None = None
    if spec.uses_rf:
        assert rf_risk is not None
        rf_packet_probability = rf_risk.total_failure_probability ** (
            spec.reserved_rf_attempts
        )
        for attempt_index in range(spec.reserved_rf_attempts):
            attempt = _evaluate_rf_attempt(
                attempt_index=attempt_index,
                tape=tape,
                risk=rf_risk,
            )
            rf_attempts.append(attempt)
            if attempt.success:
                rf_delivered = True
                break

    vlc_delivered = False
    vlc_probability: float | None = None
    if spec.uses_vlc:
        assert vlc_result is not None
        _validate_vlc_result(
            pair_id=accounting.pair_id,
            result=vlc_result,
            tape=tape,
        )
        vlc_delivered = vlc_result.success
        vlc_probability = vlc_result.total_failure_probability

    delivered = rf_delivered or vlc_delivered
    conditional = 1.0
    if rf_packet_probability is not None:
        conditional *= rf_packet_probability
    if vlc_probability is not None:
        conditional *= vlc_probability

    if delivered:
        cause = FailureCause.NONE
    elif spec.duplicates:
        cause = FailureCause.JOINT_FAILURE
    elif spec.uses_rf:
        cause = rf_attempts[-1].failure_cause
    else:
        assert vlc_result is not None
        cause = vlc_result.failure_cause

    return PairPacketOutcome(
        pair_id=accounting.pair_id,
        action=accounting.action,
        reward=accounting.reward,
        delivered=delivered,
        sampled_miss_cost=int(not delivered),
        conditional_miss_probability=conditional,
        failure_cause=cause,
        rf_delivered=rf_delivered,
        vlc_delivered=vlc_delivered,
        rf_attempts=tuple(rf_attempts),
        rf_attempt_risk=rf_risk,
        rf_packet_miss_probability=rf_packet_probability,
        vlc_result=vlc_result,
        vlc_miss_probability=vlc_probability,
    )


def assemble_frame_outcomes(
    ledger: FrameActionLedger,
    pool_response: RFPoolResponse,
    *,
    tapes_by_pair: Mapping[str, MatchedPacketTape],
    rf_risks_by_pair: Mapping[str, RFAttemptRisk],
    vlc_results_by_pair: Mapping[str, VLCChannelResult],
) -> FramePacketOutcomes:
    """Build aligned CMDP targets from selected-action simulator evaluations."""

    if not isinstance(ledger, FrameActionLedger):
        raise PacketOutcomeError("outcome assembly requires a FrameActionLedger")
    if not isinstance(pool_response, RFPoolResponse):
        raise PacketOutcomeError("outcome assembly requires an RFPoolResponse")
    expected_demand = RFPoolDemand.from_ledger(ledger)
    if pool_response.demand != expected_demand:
        raise PacketOutcomeError(
            "RF-pool response does not belong to the action ledger's frame"
        )

    accounting = ledger.pair_accounting
    pair_ids = ledger.pair_ids
    rf_pair_ids = tuple(row.pair_id for row in accounting if row.uses_rf)
    vlc_pair_ids = tuple(row.pair_id for row in accounting if row.uses_vlc)
    _validate_exact_keys(tapes_by_pair, pair_ids, name="matched packet tapes")
    _validate_exact_keys(rf_risks_by_pair, rf_pair_ids, name="RF risks")
    _validate_exact_keys(vlc_results_by_pair, vlc_pair_ids, name="VLC results")

    outcomes: list[PairPacketOutcome] = []
    for row in accounting:
        tape = tapes_by_pair[row.pair_id]
        if not isinstance(tape, MatchedPacketTape):
            raise PacketOutcomeError(
                "matched packet tape mapping contains an invalid row",
                context={"pair_id": row.pair_id},
            )
        expected_identity = (
            ledger.trace_id,
            row.pair_id,
            row.lifecycle.episode_step,
        )
        actual_identity = (
            tape.identity.trace_id,
            tape.identity.pair_episode_id,
            tape.identity.packet_index,
        )
        if actual_identity != expected_identity:
            raise PacketOutcomeError(
                "matched packet tape identity does not match the action row",
                context={
                    "pair_id": row.pair_id,
                    "actual_identity": actual_identity,
                    "expected_identity": expected_identity,
                },
            )
        tape.view_for_action(row.action)

        rf_risk = rf_risks_by_pair.get(row.pair_id)
        if rf_risk is not None:
            if not isinstance(rf_risk, RFAttemptRisk):
                raise PacketOutcomeError(
                    "RF risks mapping contains an invalid row",
                    context={"pair_id": row.pair_id},
                )
            if rf_risk.pair_id != row.pair_id or rf_risk.pool_response != pool_response:
                raise PacketOutcomeError(
                    "RF risk does not belong to its pair and current pool response",
                    context={"pair_id": row.pair_id},
                )

        vlc_result = vlc_results_by_pair.get(row.pair_id)
        outcomes.append(
            _assemble_pair_outcome(
                accounting=row,
                tape=tape,
                rf_risk=rf_risk,
                vlc_result=vlc_result,
            )
        )

    outcome_rows = tuple(outcomes)
    return FramePacketOutcomes(
        trace_id=ledger.trace_id,
        frame_index=ledger.frame_index,
        time_s=ledger.time_s,
        pair_ids=pair_ids,
        rewards=_freeze_float32(tuple(row.reward for row in outcome_rows)),
        sampled_miss_costs=_freeze_float32(
            tuple(float(row.sampled_miss_cost) for row in outcome_rows)
        ),
        conditional_miss_probabilities=_freeze_float32(
            tuple(row.conditional_miss_probability for row in outcome_rows)
        ),
        pair_outcomes=outcome_rows,
        rf_pool_response=pool_response,
    )


__all__ = [
    "FramePacketOutcomes",
    "OutcomeArray",
    "OutcomeInfo",
    "PacketOutcomeError",
    "PairPacketOutcome",
    "RFAttemptOutcome",
    "assemble_frame_outcomes",
]
