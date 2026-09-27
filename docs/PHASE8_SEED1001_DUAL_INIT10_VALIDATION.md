# Phase 8 fresh seed-1001 dual-init-10 validation

Date completed: 2026-09-27

## Verdict

The fresh full seed-1001 run with density multipliers initialized at 10 is
engineering-complete and materially safer than the original full seed-1001
pilot. It does **not** pass the `1e-4` validation reliability gate. Seeds
1002--1005 remain blocked.

The correction is scientifically informative rather than sufficient. Across
the five frozen causal validation regimes, row-weighted feasible action mass
increased from `0.502278` in the original full run to `0.655464`, while
policy-expected conditional miss risk fell from `0.061051` to `0.013287`.
Nevertheless, every observed density/regime cell remains above `1e-4`; the
best cell is approximately 59 times the target.

## Run identity and integrity

- Policy seed: `1001`
- Configuration: headline layers plus
  `configs/training/recovery_dual_init_10.yaml`
- Configuration hash:
  `df5cf40513f0c08ceba1b037b58a1002b9cc3fa87033601d0f3ee9ec452800c4`
- Completed: 9,998,802 of 10,000,000 environment transitions
- PPO iterations: 265
- Stop reason: `insufficient_budget_for_balanced_round`
- Unused tail: 1,198 transitions
- Final checkpoint SHA-256:
  `39186e0b79465856369a75805237b79957e889166e35efcb4b02ac4303101dbf`
- Iteration-report schema:
  `hybrid-rf-vlc-rl.joint-density-training-iteration.v3`
- Constraint-pressure schema:
  `hybrid-rf-vlc-rl.constraint-pressure-diagnostics.v1`
- All persisted iteration-report numbers are finite.
- The final checkpoint file digest agrees with both the final iteration report
  and the completed session report.

The normal stop is expected. Another complete density-balanced round would
have crossed the configured 10-million-transition limit.

## Training-side result

The final 20-update rollout-weighted conditional-miss estimates improve
substantially over the original pilot:

| Density | Original full run | Dual-init-10 run | Reduction | Budget multiple after correction |
|---:|---:|---:|---:|---:|
| 10 | 0.07110 | 0.01240 | 82.6% | 124x |
| 20 | 0.07230 | 0.01717 | 76.3% | 172x |
| 30 | 0.06461 | 0.01453 | 77.5% | 145x |

The final-20 action distribution also changes materially:

| Action/group | Fraction |
|---|---:|
| VLC | 59.432% |
| RF-2 | 27.564% |
| DUP-1 through DUP-4 | 12.402% |
| RF-1 | 0.241% |
| RF-3 and RF-4 | 0.361% |

The intervention therefore corrected the original 99.58% RF-1/VLC collapse,
but it did not drive the policy to the higher redundancy required by the
strict target. Mean final-20 entropy is `0.02815`, showing a highly
concentrated policy rather than continued broad exploration.

Final constraint-to-reward advantage ratios are `0.2129`, `0.3646`, and
`0.3911` at densities 10, 20, and 30. They remain much higher than the
approximately 3--4% ratios of the bounded control, but have fallen below the
roughly balanced ratios observed after nine updates in the bounded
dual-init-10 arm. The cost-advantage signal weakened relative to reward as
training progressed even though the dual multipliers remained near 10.2.

## Frozen validation procedure

Checkpoint 265 was evaluated with the versioned PPO regime evaluator on the
nine validation windows frozen by the training-fitted state-regime audit:

- deterministic masked argmax action selection;
- training-only normalization restored from the checkpoint and frozen;
- causal actor inputs only;
- exact simulator truth consumed only after action choice by the non-deployable
  evaluator;
- all nine actions assessed under policy-induced joint RF load;
- no test trace opened;
- actor, normalization, and global RNG state verified unchanged.

Validation artifact SHA-256:
`e556d7f5aa6c47ef4458ac28b6f647984599b9ebde91868edb98cb259a48e022`.
The artifact binds checkpoint SHA-256, configuration hash, the common
policy-environment scope hash, and state-regime audit SHA-256.

## Campaign-level validation result

| Regime | Rows | Selected risk | Feasible probability mass | Selected feasible fraction | Gate |
|---|---:|---:|---:|---:|---|
| Easy state | 5,454 | 0.015422 | 0.401452 | 0.401173 | fail |
| Moderate RF conditions | 7,123 | 0.005896 | 0.884042 | 0.884038 | fail |
| Poor VLC / usable RF | 8,110 | 0.016595 | 0.530936 | 0.530826 | fail |
| Uncertain mixed state | 5,871 | 0.017253 | 0.692390 | 0.692557 | fail |
| Heavy RF contention / optical permitted | 7,174 | 0.012254 | 0.732178 | 0.732088 | fail |

The overlapping regime rows yield these like-for-like checkpoint comparisons:

| Checkpoint | Feasible mass | Policy-expected risk | Selected risk | Selected feasible fraction |
|---|---:|---:|---:|---:|
| Original full seed 1001 | 0.502278 | 0.061051 | 0.061831 | 0.500534 |
| Bounded dual-init-10 | 0.590858 | 0.026301 | 0.017124 | 0.615588 |
| Corrected full seed 1001 | **0.655464** | **0.013287** | **0.013337** | **0.655401** |

Full training therefore improved the bounded selected configuration, and the
correction reduced full-run validation risk by about 78% relative to the
original pilot. The absolute reliability requirement is still missed by two
orders of magnitude.

## Density-conditioned gate

Thirteen density/regime cells have validation observations. Density 10 has no
moderate-RF or heavy-contention rows under the frozen regime definitions; the
audit never claims those absent cells are supported.

| Density | Observed regime cells | Selected-risk range | Passing cells at `1e-4` |
|---:|---:|---:|---:|
| 10 | 3 | 0.011725--0.016663 | 0 / 3 |
| 20 | 5 | 0.005895--0.016708 | 0 / 5 |
| 30 | 5 | 0.005897--0.019994 | 0 / 5 |

Because reliability is not averaged across density constraints, failure in any
represented density blocks continuation. Here every represented cell fails;
no statistical uncertainty calculation could turn point risks 59--200 times
the budget into a pass.

## Residual policy behavior

The policy is clearly context-sensitive:

- easy states: 56.7% RF-2 and 38.4% VLC;
- moderate RF: 85.9% VLC and 10.8% RF-2;
- poor VLC / usable RF: 50.3% VLC and 41.5% RF-2;
- uncertain states: 46.0% VLC, 24.6% DUP-1, and 22.5% RF-2;
- heavy contention: 69.0% VLC and 24.7% RF-2.

However, RF-2 is never the cheapest feasible oracle action at the `1e-4`
target in these campaign-regime rows. When VLC is not feasible, the
counterfactual oracle most often requires RF-4 or DUP-4, with occasional RF-3.
The learned policy assigns almost no mass to RF-4 or DUP-4. The correction thus
moved the policy from an underpowered RF-1 solution to an improved but still
underpowered RF-2 solution.

The environment also presents a hard feasibility boundary: under the final
policy-induced load, at least one feasible action exists in only 61.6% to
93.3% of rows depending on regime. A next diagnosis must distinguish:

1. optimization failure to choose RF-4/DUP-4 where those actions are feasible;
2. population-load coupling that makes otherwise reliable actions infeasible;
3. states for which no declared action can meet `1e-4` under the current
   physical model; and
4. whether the global reliability target is attainable despite statewise
   infeasibility.

## Decision and next task

This seed is not eligible as a paper checkpoint, and seeds 1002--1005 must not
start. The next task is a bounded residual-feasibility diagnosis, not another
full training run. It should quantify the per-state oracle minimum risk under
policy-induced and declared counterfactual loads, decompose the remaining miss
risk into unavoidable versus policy-regret components, and determine whether
additional constraint scaling can help before any new hyperparameter arm is
predeclared.
