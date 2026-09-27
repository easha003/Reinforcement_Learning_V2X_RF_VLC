# Phase 8 seed-1001 residual feasibility diagnosis

Date completed: 2026-09-27

## Decision

The residual validation diagnosis is complete. The corrected seed-1001 policy
has a large action-selection error at every density, and its current
population RF load also makes the `1e-4` aggregate target unattainable by any
unilateral focal-action change at densities 20 and 30. A standard PPO
constraint-pressure recovery arm is therefore **not authorized** yet.

The next experiment must compute a population-joint action oracle, not merely
increase the PPO multiplier. It should jointly assign the nine actions while
recomputing the RF load, determine whether a self-consistent population action
profile can meet `1e-4`, and provide coordinated labels if it can.

## Evidence boundary

The v2 evaluator restores the frozen iteration-265 seed-1001 checkpoint and
replays the same nine validation windows used by the earlier regime audit. It
adds all-usable-row summaries so overlapping regime labels cannot distort the
per-density CMDP constraint. For every usable row and load profile it records:

```text
selected policy risk = minimum action risk + selected action risk regret
```

The minimum is taken across every hardware-allowed action after action
selection, using isolated simulator truth. It is a non-deployable diagnostic;
truth never enters the PPO actor. All decompositions close to numerical
precision, normalization remains frozen, and the test split remains unopened.

The four load profiles are:

- `policy_induced_load`: change the focal action while holding the exact RF
  attempts of every other policy-selected action fixed;
- `vlc_offload`: optimistic probe with zero RF attempts from every other pair;
- `rf1_pressure`: every other active pair contributes one RF attempt; and
- `rf4_pressure`: every other active pair contributes four RF attempts.

These profiles expose load sensitivity but do not prove that a stable joint
assignment exists. In particular, a passing `vlc_offload` result cannot be
simultaneously realized by every focal vehicle.

## Artifact identity

- Checkpoint: iteration 265, policy seed 1001
- Checkpoint SHA-256:
  `39186e0b79465856369a75805237b79957e889166e35efcb4b02ac4303101dbf`
- Configuration hash:
  `df5cf40513f0b08ceba1b037b58a1002b9cc3fa87033601d0f3ee9ec452800c4`
- V2 evaluation artifact:
  `artifacts/evaluations/phase8_seed1001_dual_init10_full_ppo_regime_evaluation_v2.json`
- V2 evaluation SHA-256:
  `4653422161883923d33b3b534a68d6fed6996124e06bde25ca3a5c6e261a0619`
- Residual diagnosis artifact:
  `artifacts/evaluations/phase8_seed1001_residual_feasibility.json`
- Residual diagnosis SHA-256:
  `02f78cca13db25a4ec1ba4d6fb3825377f8ec1899310ee7534bb0b7824f344e0`
- All usable validation rows: 68,135
- Test split opened: no

## Policy-load decomposition

| Density | Rows | Selected risk | Minimum-action floor | Floor / budget | Action regret | Selected risk due to regret | Floor gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 7,802 | 0.0135006 | 0.00006134 | 0.61x | 0.0134392 | 99.55% | pass |
| 20 | 22,714 | 0.0115388 | 0.00022337 | 2.23x | 0.0113155 | 98.06% | fail |
| 30 | 37,619 | 0.0089962 | 0.00044637 | 4.46x | 0.0085499 | 95.04% | fail |
| Campaign | 68,135 | 0.0103596 | 0.00032794 | 3.28x | 0.0100317 | 96.83% | fail |

This resolves two different failure mechanisms:

1. **Action-selection regret is dominant.** Holding other-pair load fixed, a
   minimum-risk action would remove 95.0%--99.5% of the deterministic policy's
   current risk. The corrected actor still selects underpowered actions.
2. **Action selection alone is insufficient at higher density.** Even the
   minimum unilateral action averages `2.23e-4` and `4.46e-4` at densities 20
   and 30 under the policy-induced load. No focal-action optimizer can close
   that residual without changing the population load.

Across all 68,135 rows, the minimum-risk action is `DUP-4` on 62,019 rows and
`RF-4` on 6,116 rows. This is a reliability floor, not a resource-optimal
oracle: assigning those actions independently to all vehicles would increase
contention and invalidate the fixed-load calculation.

## Declared load sensitivity

| Density | VLC-offload floor | RF-1 pressure floor | Policy-load floor | RF-4 pressure floor |
|---:|---:|---:|---:|---:|
| 10 | `1.17e-12` | `3.12e-5` | `6.13e-5` | `4.46e-3` |
| 20 | `5.05e-13` | `5.96e-4` | `2.23e-4` | `4.02e-2` |
| 30 | `3.13e-13` | `2.05e-3` | `4.46e-4` | `7.11e-2` |

The optimistic zero-other-RF-load floor is far below `1e-4` at every density.
The current PHY and nine-action set can therefore produce sufficiently
reliable focal transmissions when contention is removed. Conversely, the
uniform RF-4 probe is severely infeasible. The target lies between those
extremes, so the unresolved question is the existence of a coordinated joint
assignment that gives difficult links strong redundancy without causing a
population-wide RF overload.

## Scientific consequence

The earlier proposal to imitate a per-row minimum-risk oracle must be refined.
Independent `RF-4`/`DUP-4` labels would teach nearly every pair to consume four
RF attempts and could reproduce the failing `rf4_pressure` condition. Oracle
pretraining is appropriate only after labels are produced by a population-
joint optimizer that accounts for the RF attempts of all simultaneously active
pairs.

Increasing the dual multiplier alone is also not justified yet. It could shift
the actor toward the correct strong actions, addressing action regret, while
simultaneously increasing the load floor at densities 20 and 30. The next
oracle must determine the coordinated allocation boundary first.

## Next task

For each validation frame, enumerate candidate total RF-attempt loads. At each
load, compute every pair/action risk using that shared load and solve the
multiple-choice assignment problem: exactly one action per usable pair, exact
aggregate RF attempts, and minimum aggregate miss risk (then minimum resource
cost among reliability-equivalent assignments). Selecting the best candidate
load yields a population-joint lower bound and an explicit action assignment.

The result will provide the next gate:

- joint floor at or below `1e-4` for every density: design a coordinated or
  coordination-aware learner and use the joint assignments for pretraining;
- joint floor above `1e-4` at any density: revise the action/resource or
  physical system before further PPO training.
