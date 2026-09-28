"""Certificate-aware population-joint search under pair-local RF physics."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.channels.rf.model import RFPropagationResult
from hybrid_v2x_rl.config import load_headline_config
from hybrid_v2x_rl.core.enums import RFPropagationState
from hybrid_v2x_rl.core.policy_actions import (
    ACTION_CONTRACT_VERSION,
    ActionResourceMap,
    PolicyAction,
    action_resources,
)
from hybrid_v2x_rl.env.assembly import build_rf_channel
from hybrid_v2x_rl.mean_field.action_ledger import FrameActionLedger
from hybrid_v2x_rl.mean_field.frames import (
    FrameTraceSource,
    PairLifecycle,
    PopulationFrame,
    PopulationPair,
)
from hybrid_v2x_rl.mean_field.joint_risk_oracle import (
    JointRiskOracleError,
    JointRiskOracleProblem,
    solve_pair_local_joint_risk,
)
from hybrid_v2x_rl.mean_field.local_rf_pipeline import LocalRFPhysicsModel
from hybrid_v2x_rl.mean_field.local_rf_response import LocalRFResponseModel
from hybrid_v2x_rl.mobility.trace_io import VehicleTraceRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_ID = "synthetic-d10-validation-000"
RESOURCE_MAP = ActionResourceMap(
    contract_version=ACTION_CONTRACT_VERSION,
    rf_activation_cost=1.0,
    vlc_activation_cost=1.0,
)
ALLOWED_ACTIONS = (PolicyAction.VLC, PolicyAction.RF_1, PolicyAction.DUP_1)


def _vehicle(vehicle_id: str, x_m: float) -> VehicleTraceRecord:
    return VehicleTraceRecord(
        trace_id=TRACE_ID,
        time_s=0.0,
        vehicle_id=vehicle_id,
        x_m=x_m,
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


def _frame(population: int = 3, *, separation_m: float = 40.0) -> PopulationFrame:
    vehicles: list[VehicleTraceRecord] = []
    pairs: list[PopulationPair] = []
    for index in range(population):
        transmitter = _vehicle(f"tx-{index}", index * separation_m)
        receiver = _vehicle(f"rx-{index}", index * separation_m + 10.0)
        vehicles.extend((transmitter, receiver))
        pairs.append(
            PopulationPair(
                pair_id=f"pair-{index}",
                episode_step=0,
                transmitter=transmitter,
                receiver=receiver,
                lifecycle=PairLifecycle(born=True),
            )
        )
    return PopulationFrame(
        source=FrameTraceSource(
            path=Path(TRACE_ID),
            trace_id=TRACE_ID,
            split="validation",
            density=10.0,
            replicate=0,
        ),
        index=0,
        time_s=0.0,
        vehicles=tuple(sorted(vehicles, key=lambda row: row.vehicle_id)),
        pairs=tuple(pairs),
    )


def _model() -> LocalRFPhysicsModel:
    config = load_headline_config(PROJECT_ROOT)
    rf = build_rf_channel(config, band=SensitivityBand.NOMINAL)
    return LocalRFPhysicsModel(
        response_model=LocalRFResponseModel(
            parameters=rf.collision,
            sensitivity_band=SensitivityBand.NOMINAL,
            attempt_airtime_s=float(config.rf.timing.airtime_s),
        ),
        buildings=(),
        antenna_height_m=float(config.geometry.rf_antenna_height_m),
    )


def _propagation(probability: float) -> RFPropagationResult:
    return RFPropagationResult(
        propagation_state=RFPropagationState.LOS,
        pathloss_db=80.0,
        shadowing_db=0.0,
        fading_gain_linear=1.0,
        sinr_db=20.0,
        decoding_failure_probability=probability,
    )


def _problem(
    *,
    usable: tuple[bool, ...] = (True, True, True),
    separation_m: float = 40.0,
) -> JointRiskOracleProblem:
    frame = _frame(len(usable), separation_m=separation_m)
    model = _model()
    return JointRiskOracleProblem(
        frame=frame,
        context=model.context_for(frame),
        usable_mask=usable,
        allowed_actions=ALLOWED_ACTIONS,
        fallback_action=PolicyAction.DUP_1,
        resource_map=RESOURCE_MAP,
        local_rf_model=model,
        rf_propagation_by_pair={
            pair_id: _propagation(probability)
            for pair_id, probability in zip(
                frame.active_pair_ids,
                (0.02, 0.15, 0.40)[: len(usable)],
                strict=True,
            )
        },
        vlc_failure_probabilities=np.asarray(
            (0.001, 0.35, 1.0)[: len(usable)],
            dtype=np.float64,
        ),
    )


def _candidate_rank(
    problem: JointRiskOracleProblem,
    actions: tuple[PolicyAction, ...],
) -> tuple[float, float, int, tuple[int, ...]]:
    ledger = FrameActionLedger.from_frame(
        problem.frame,
        dict(zip(problem.pair_ids, actions, strict=True)),
        resource_map=problem.resource_map,
    )
    physics = problem.local_rf_model.evaluate(
        problem.context,
        ledger,
        propagation_by_pair=problem.rf_propagation_by_pair,
    )
    risks: list[float] = []
    for index, (pair_id, action) in enumerate(
        zip(problem.pair_ids, actions, strict=True)
    ):
        resources = action_resources(action)
        risk = 1.0
        if resources.uses_rf:
            attempt_risk = physics.attempt_risks.risk_for(
                pair_id
            ).total_failure_probability
            risk *= attempt_risk**resources.reserved_rf_attempts
        if resources.uses_vlc:
            risk *= float(problem.vlc_failure_probabilities[index])
        risks.append(risk)
    return (
        math.fsum(risks),
        math.fsum(problem.resource_map.activation_cost(action) for action in actions),
        sum(action_resources(action).reserved_rf_attempts for action in actions),
        tuple(int(action) for action in actions),
    )


def _brute_force(
    problem: JointRiskOracleProblem,
) -> tuple[tuple[float, float, int, tuple[int, ...]], tuple[PolicyAction, ...]]:
    choices = tuple(
        problem.allowed_actions if usable else (problem.fallback_action,)
        for usable in problem.usable_mask
    )
    ranked = tuple(
        (_candidate_rank(problem, actions), actions)
        for actions in itertools.product(*choices)
    )
    return min(ranked, key=lambda row: row[0])


@pytest.mark.parametrize("usable", [(True, True, True), (True, False, True)])
def test_exact_solver_matches_independent_pair_local_exhaustive_search(
    usable: tuple[bool, ...],
) -> None:
    problem = _problem(usable=usable)

    solution = solve_pair_local_joint_risk(problem)
    brute_rank, brute_actions = _brute_force(problem)

    assert solution.optimality_proven
    assert solution.total_conditional_miss_risk == pytest.approx(
        brute_rank[0], abs=1e-12
    )
    assert solution.total_activation_cost == pytest.approx(brute_rank[1])
    assert solution.total_rf_attempts == brute_rank[2]
    assert solution.actions == brute_actions
    assert solution.certified_lower_bound == pytest.approx(brute_rank[0])
    assert solution.absolute_optimality_gap == 0.0
    assert solution.assignments_evaluated == solution.assignment_space_size
    assert solution.action_counts == tuple(
        solution.actions.count(action) for action in PolicyAction
    )


def test_all_unusable_population_is_exactly_forced_to_fallback() -> None:
    problem = _problem(usable=(False, False, False))

    solution = solve_pair_local_joint_risk(problem)

    assert solution.actions == (PolicyAction.DUP_1,) * 3
    assert solution.total_rf_attempts == 3
    assert solution.assignment_space_size == 1
    assert solution.assignments_evaluated == 1
    assert solution.optimality_proven
    assert solution.usable_conditional_miss_risk == 0.0
    assert solution.forced_fallback_conditional_miss_risk == pytest.approx(
        solution.total_conditional_miss_risk
    )


def test_bounded_search_returns_realizable_candidate_and_valid_certificate() -> None:
    problem = replace(_problem(), exact_assignment_cap=1)

    solution = solve_pair_local_joint_risk(problem)
    realized_rank = _candidate_rank(problem, solution.actions)

    assert not solution.optimality_proven
    assert solution.assignments_evaluated < solution.assignment_space_size
    assert solution.total_conditional_miss_risk == pytest.approx(realized_rank[0])
    assert solution.certified_lower_bound <= solution.total_conditional_miss_risk
    assert solution.absolute_optimality_gap == pytest.approx(
        solution.total_conditional_miss_risk - solution.certified_lower_bound
    )
    assert solution.search_starts >= 1
    assert solution.search_iterations >= 1


def test_spatial_reuse_keeps_a_distant_transmitter_out_of_focal_domain() -> None:
    problem = _problem(separation_m=1_000.0)
    focal = problem.pair_ids[0]

    first = problem.local_rf_model.evaluate(
        problem.context,
        FrameActionLedger.from_frame(
            problem.frame,
            {
                focal: PolicyAction.RF_1,
                problem.pair_ids[1]: PolicyAction.VLC,
                problem.pair_ids[2]: PolicyAction.VLC,
            },
            resource_map=problem.resource_map,
        ),
        propagation_by_pair=problem.rf_propagation_by_pair,
    ).attempt_risks.risk_for(focal)
    second = problem.local_rf_model.evaluate(
        problem.context,
        FrameActionLedger.from_frame(
            problem.frame,
            {
                focal: PolicyAction.RF_1,
                problem.pair_ids[1]: PolicyAction.VLC,
                problem.pair_ids[2]: PolicyAction.RF_1,
            },
            resource_map=problem.resource_map,
        ),
        propagation_by_pair=problem.rf_propagation_by_pair,
    ).attempt_risks.risk_for(focal)

    assert second.total_failure_probability == pytest.approx(
        first.total_failure_probability,
        abs=1e-15,
    )
    assert problem.context.topology.domain_for(focal).member_pair_ids == (focal,)


def test_problem_rejects_incomplete_pair_local_propagation_truth() -> None:
    base = _problem()

    with pytest.raises(JointRiskOracleError, match="cover the population exactly"):
        replace(
            base,
            rf_propagation_by_pair={
                pair_id: result
                for pair_id, result in base.rf_propagation_by_pair.items()
                if pair_id != base.pair_ids[-1]
            },
        )
