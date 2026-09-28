"""Phase 5 reward, realized miss, and conditional-risk target boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.model import (
    RFPacketRandomness,
    RFPropagationRequest,
    RFPropagationResult,
)
from hybrid_v2x_rl.channels.vlc.model import VLCChannelResult, VLCPacketRandomness
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import FailureCause, RFPropagationState
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.action_ledger import (
    FrameActionLedger,
    PairActionLifecycle,
    PairActionReservation,
)
from hybrid_v2x_rl.mean_field.actor_observations import CausalActorFrame
from hybrid_v2x_rl.mean_field.critic_observations import CriticObservationFrame
from hybrid_v2x_rl.mean_field.environment_api import FrameObservation
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.local_rf_pipeline import (
    FrameLocalRFPhysics,
    LocalRFPhysicsModel,
)
from hybrid_v2x_rl.mean_field.local_rf_risk import PairLocalRFAttemptRisk
from hybrid_v2x_rl.mean_field.packet_outcomes import (
    PacketOutcomeError,
    assemble_frame_outcomes,
)
from hybrid_v2x_rl.mean_field.random_tape import (
    MatchedPacketTape,
    MatchedPacketTapeFactory,
    PacketRandomnessIdentity,
)
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d20-train-000"
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=0.3,
    vlc_activation_cost=2.0,
)


def _ledger(actions: tuple[PolicyAction, ...]) -> FrameActionLedger:
    lifecycle = PairActionLifecycle(
        episode_step=4,
        born=False,
        terminated=False,
        truncated=False,
        bootstrap_valid=False,
        end_reason=None,
    )
    return FrameActionLedger(
        trace_id=TRACE_ID,
        frame_index=7,
        time_s=0.7,
        resource_map=RESOURCE_MAP,
        reservations=tuple(
            PairActionReservation(
                pair_id=f"pair-{index}",
                action=action,
                lifecycle=lifecycle,
            )
            for index, action in enumerate(actions)
        ),
    )


def _frame(population: int) -> PopulationFrame:
    vehicles: list[VehicleTraceRecord] = []
    pairs: list[PopulationPair] = []
    for index in range(population):
        transmitter = VehicleTraceRecord(
            trace_id=TRACE_ID,
            time_s=0.7,
            vehicle_id=f"tx-{index}",
            x_m=float(index * 20),
            y_m=0.0,
            heading_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            length_m=4.5,
            width_m=1.8,
            height_m=1.5,
            lane_id="edge-0_0",
            edge_id="edge-0",
            route_id="route-0",
            vehicle_type="passenger",
        )
        receiver = replace(
            transmitter,
            vehicle_id=f"rx-{index}",
            x_m=float(index * 20 + 10),
        )
        vehicles.extend((transmitter, receiver))
        pairs.append(
            PopulationPair(
                pair_id=f"pair-{index}",
                episode_step=4,
                transmitter=transmitter,
                receiver=receiver,
                lifecycle=PairLifecycle(born=False),
            )
        )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="train",
            density=20.0,
            replicate=0,
        ),
        index=7,
        time_s=0.7,
        vehicles=tuple(sorted(vehicles, key=lambda row: row.vehicle_id)),
        pairs=tuple(pairs),
    )


def _propagation() -> RFPropagationResult:
    return build_rf_channel(load_headline_config(PROJECT_ROOT)).evaluate_propagation(
        RFPropagationRequest(
            distance_m=100.0,
            propagation_state=RFPropagationState.LOS,
            blockage_db=0.0,
            shadowing_normalized=0.0,
            fading_power_gain=1.0,
        )
    )


def _vlc_result(
    tape: MatchedPacketTape,
    *,
    probability: float = 0.4,
) -> VLCChannelResult:
    success = tape.vlc.decoding_draw >= probability
    return VLCChannelResult(
        received_power_w=1e-6,
        electrical_snr=10.0,
        within_field_of_view=True,
        occluded=False,
        bit_error_rate=0.01,
        decoding_failure_probability=probability,
        total_failure_probability=probability,
        success=success,
        failure_cause=FailureCause.NONE if success else FailureCause.VLC_CHANNEL,
        beam_aimed=True,
    )


def _inputs(
    actions: tuple[PolicyAction, ...],
    *,
    seed: int = 31,
    vlc_probability: float = 0.4,
) -> tuple[
    FrameActionLedger,
    FrameLocalRFPhysics,
    dict[str, MatchedPacketTape],
    dict[str, PairLocalRFAttemptRisk],
    dict[str, VLCChannelResult],
]:
    ledger = _ledger(actions)
    frame = _frame(len(actions))
    model = LocalRFPhysicsModel.from_config(load_headline_config(PROJECT_ROOT))
    tapes = {
        row.pair_id: MatchedPacketTapeFactory(seed).build(
            PacketRandomnessIdentity(
                trace_id=TRACE_ID,
                pair_episode_id=row.pair_id,
                packet_index=row.lifecycle.episode_step,
            )
        )
        for row in ledger.pair_accounting
    }
    propagation = {
        pair_id: _propagation() for pair_id in ledger.pair_ids
    }
    physics = model.evaluate(
        model.context_for(frame),
        ledger,
        propagation_by_pair=propagation,
    )
    risks = {
        row.pair_id: physics.attempt_risks.risk_for(row.pair_id)
        for row in ledger.pair_accounting
        if row.uses_rf
    }
    vlc = {
        row.pair_id: _vlc_result(
            tapes[row.pair_id],
            probability=vlc_probability,
        )
        for row in ledger.pair_accounting
        if row.uses_vlc
    }
    return ledger, physics, tapes, risks, vlc


def _fixed_tape(
    pair_id: str,
    *,
    rf_draws: tuple[tuple[float, float, float], ...],
    vlc_draw: float,
) -> MatchedPacketTape:
    padded = rf_draws + ((1.0, 1.0, 1.0),) * (4 - len(rf_draws))
    return MatchedPacketTape(
        identity=PacketRandomnessIdentity(
            trace_id=TRACE_ID,
            pair_episode_id=pair_id,
            packet_index=4,
        ),
        rf_attempts=tuple(
            RFPacketRandomness(
                collision_draw=collision,
                decoding_draw=decoding,
                half_duplex_draw=half_duplex,
            )
            for collision, decoding, half_duplex in padded
        ),
        vlc=VLCPacketRandomness(decoding_draw=vlc_draw),
    )


def test_all_nine_actions_emit_contract_shaped_reward_cost_and_risk() -> None:
    actions = tuple(PolicyAction)
    ledger, physics, tapes, risks, vlc = _inputs(actions)

    frame = assemble_frame_outcomes(
        ledger,
        physics,
        tapes_by_pair=tapes,
        vlc_results_by_pair=vlc,
    )

    assert frame.pair_ids == ledger.pair_ids
    assert frame.rewards.dtype == np.float32
    assert frame.sampled_miss_costs.dtype == np.float32
    assert frame.conditional_miss_probabilities.dtype == np.float32
    assert frame.rewards.shape == frame.sampled_miss_costs.shape == (
        len(actions),
    )
    assert not frame.rewards.flags.writeable
    assert not frame.sampled_miss_costs.flags.writeable
    assert not frame.conditional_miss_probabilities.flags.writeable
    assert frame.rewards == pytest.approx(
        [RESOURCE_MAP.reward(action) for action in actions]
    )

    for action, outcome in zip(actions, frame.pair_outcomes, strict=True):
        spec = action_resources(action)
        expected = 1.0
        if spec.uses_rf:
            expected *= risks[outcome.pair_id].total_failure_probability ** (
                spec.reserved_rf_attempts
            )
        if spec.uses_vlc:
            expected *= vlc[outcome.pair_id].total_failure_probability
        assert outcome.conditional_miss_probability == pytest.approx(expected)
        assert outcome.sampled_miss_cost == int(not outcome.delivered)


def test_sampled_mechanisms_stop_at_first_success_but_keep_committed_reward() -> None:
    actions = (PolicyAction.RF_4, PolicyAction.VLC, PolicyAction.DUP_2)
    ledger, physics, tapes, risks, vlc = _inputs(actions, vlc_probability=0.4)
    tapes["pair-0"] = _fixed_tape(
        "pair-0",
        rf_draws=((1.0, 1.0, 1.0),),
        vlc_draw=1.0,
    )
    tapes["pair-1"] = _fixed_tape(
        "pair-1",
        rf_draws=(),
        vlc_draw=0.0,
    )
    tapes["pair-2"] = _fixed_tape(
        "pair-2",
        rf_draws=((0.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        vlc_draw=0.0,
    )
    vlc["pair-1"] = _vlc_result(tapes["pair-1"], probability=0.4)
    vlc["pair-2"] = _vlc_result(tapes["pair-2"], probability=0.4)

    frame = assemble_frame_outcomes(
        ledger,
        physics,
        tapes_by_pair=tapes,
        vlc_results_by_pair=vlc,
    )
    rf_four, vlc_only, dup_two = frame.pair_outcomes

    assert frame.sampled_miss_costs.tolist() == [0.0, 1.0, 1.0]
    assert rf_four.rf_attempts_used == 1
    assert rf_four.rf_attempts[0].success
    assert rf_four.reward == pytest.approx(-4 * RESOURCE_MAP.rf_activation_cost)
    assert rf_four.rf_packet_miss_probability == pytest.approx(
        risks["pair-0"].total_failure_probability**4
    )
    assert vlc_only.failure_cause is FailureCause.VLC_CHANNEL
    assert dup_two.failure_cause is FailureCause.JOINT_FAILURE
    assert all(row.collision_failure for row in dup_two.rf_attempts)
    assert dup_two.reward == pytest.approx(
        -(2 * RESOURCE_MAP.rf_activation_cost + RESOURCE_MAP.vlc_activation_cost)
    )


def test_geometric_vlc_failure_is_certain_even_at_uniform_endpoint() -> None:
    ledger, physics, tapes, risks, vlc = _inputs((PolicyAction.VLC,))
    tapes["pair-0"] = _fixed_tape(
        "pair-0",
        rf_draws=(),
        vlc_draw=1.0,
    )
    vlc["pair-0"] = VLCChannelResult(
        received_power_w=0.0,
        electrical_snr=0.0,
        within_field_of_view=True,
        occluded=True,
        bit_error_rate=0.5,
        decoding_failure_probability=1.0,
        total_failure_probability=1.0,
        success=False,
        failure_cause=FailureCause.VLC_OCCLUSION,
        beam_aimed=True,
    )

    frame = assemble_frame_outcomes(
        ledger,
        physics,
        tapes_by_pair=tapes,
        vlc_results_by_pair=vlc,
    )

    assert frame.sampled_miss_costs.tolist() == [1.0]
    assert frame.conditional_miss_probabilities.tolist() == [1.0]
    assert frame.pair_outcomes[0].failure_cause is FailureCause.VLC_OCCLUSION


def test_step_info_retains_diagnostics_outside_actor_and_critic_tensors() -> None:
    ledger, physics, tapes, _, vlc = _inputs((PolicyAction.DUP_1,))
    frame = assemble_frame_outcomes(
        ledger,
        physics,
        tapes_by_pair=tapes,
        vlc_results_by_pair=vlc,
    )
    info = frame.as_step_info()

    assert info["transition_pair_ids"] == ledger.pair_ids
    assert info["sampled_miss_cost"] is frame.sampled_miss_costs
    assert info["conditional_miss_probability"] is frame.conditional_miss_probabilities
    assert info["packet_outcomes"] is frame.pair_outcomes
    assert info["local_rf_physics"] is physics
    with pytest.raises(TypeError):
        info["leak"] = True  # type: ignore[index]

    forbidden = {
        "sampled_miss_cost",
        "conditional_miss_probability",
        "local_rf_physics",
        "pair_outcomes",
    }
    assert forbidden.isdisjoint(FrameObservation.__dataclass_fields__)
    assert forbidden.isdisjoint(CausalActorFrame.__dataclass_fields__)
    assert forbidden.isdisjoint(CriticObservationFrame.__dataclass_fields__)


def test_empty_frame_emits_zero_length_target_arrays() -> None:
    ledger, physics, tapes, _, vlc = _inputs(())

    frame = assemble_frame_outcomes(
        ledger,
        physics,
        tapes_by_pair=tapes,
        vlc_results_by_pair=vlc,
    )

    assert frame.population_size == 0
    assert frame.pair_ids == ()
    assert frame.rewards.shape == (0,)
    assert frame.sampled_miss_costs.shape == (0,)
    assert frame.conditional_miss_probabilities.shape == (0,)


def test_outcome_assembly_rejects_incomplete_or_misaligned_inputs() -> None:
    ledger, physics, tapes, _, vlc = _inputs(
        (PolicyAction.VLC, PolicyAction.RF_1)
    )

    with pytest.raises(PacketOutcomeError, match="matched packet tapes"):
        assemble_frame_outcomes(
            ledger,
            physics,
            tapes_by_pair={"pair-0": tapes["pair-0"]},
            vlc_results_by_pair=vlc,
        )

    wrong_tapes = dict(tapes)
    wrong_tapes["pair-0"] = replace(
        tapes["pair-0"],
        identity=replace(tapes["pair-0"].identity, packet_index=5),
    )
    with pytest.raises(PacketOutcomeError, match="tape identity"):
        assemble_frame_outcomes(
            ledger,
            physics,
            tapes_by_pair=wrong_tapes,
            vlc_results_by_pair=vlc,
        )

    inconsistent_vlc = dict(vlc)
    inconsistent_vlc["pair-0"] = replace(
        vlc["pair-0"],
        success=not vlc["pair-0"].success,
        failure_cause=(
            FailureCause.VLC_CHANNEL
            if vlc["pair-0"].success
            else FailureCause.NONE
        ),
    )
    with pytest.raises(PacketOutcomeError, match="VLC sampled outcome"):
        assemble_frame_outcomes(
            ledger,
            physics,
            tapes_by_pair=tapes,
            vlc_results_by_pair=inconsistent_vlc,
        )


def test_outcome_assembly_rejects_local_physics_from_another_joint_action() -> None:
    ledger, _, tapes, _, vlc = _inputs((PolicyAction.RF_1,))
    other_ledger, other_physics, _, _, _ = _inputs((PolicyAction.RF_4,))
    assert other_ledger.pair_ids == ledger.pair_ids

    with pytest.raises(PacketOutcomeError, match="does not belong"):
        assemble_frame_outcomes(
            ledger,
            other_physics,
            tapes_by_pair=tapes,
            vlc_results_by_pair=vlc,
        )
