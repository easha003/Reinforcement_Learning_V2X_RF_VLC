# Phase 8 seed-1001 population-joint reliability oracle

Date completed: 2026-09-27

> **Interpretation update:** `PHASE8_RF_CONTENTION_DOMAIN_AUDIT.md` subsequently
> established that the evaluated mean-field RF pool aggregates the complete
> Manhattan frame while the project contract defines a 200 m local contention
> domain. The oracle below remains exact for that implemented global-pool
> model, but its physical-system infeasibility conclusion is superseded until
> pair-local contention and spatial reuse are implemented and reevaluated.

## Decision

The exact population-joint validation oracle fails the `1e-4` conditional
miss-risk target at every density. Its per-density floors are `5.40e-4`,
`1.48e-3`, and `1.85e-3`, or 5.40, 14.85, and 18.49 times the budget. Because
this non-deployable oracle sees current channel truth and jointly chooses every
pair's action, no PPO policy operating under the current action, RF-pool, VLC,
and fallback boundary can achieve the declared target on these windows.

Further PPO recovery training and seeds 1002--1005 remain blocked. The next
task is now the pair-local RF contention repair identified by the contention-
domain audit. A system-feasibility frontier follows only if the corrected
local-domain oracle still fails.

## Exactness boundary

For each nonempty validation frame, the solver:

1. enumerates every feasible aggregate RF-attempt load, including attempts
   forced by unusable rows' `DUP-4` fallback;
2. computes each pair/action conditional miss risk at that shared load;
3. solves the exact one-action-per-pair allocation for that load using the
   action ladder's diminishing marginal risk reductions; and
4. selects the minimum aggregate conditional miss risk, breaking
   reliability-equivalent ties by activation cost, load, and action order.

This is exact for the current per-frame model because RF access risk depends
on the population only through aggregate offered attempts and the fixed-load
risk ladder has diminishing marginal reductions. A monotone lower bound safely
prunes later loads that cannot improve the incumbent. The implementation
evaluated 69,595 of 272,657 candidate loads and pruned 203,062 (74.5%) without
approximating the optimum.

The result is a lower bound, not a deployable controller. It uses isolated
current channel truth that is unavailable to the causal actor. A failed lower
bound is therefore a hard infeasibility result within the modeled system; a
passing lower bound would only establish possibility, not learnability.

The frozen audit supplied nine `3 x 16` validation windows. Twenty-seven empty
frames contain no packet decisions, so the oracle aggregates 117 nonempty
frames and 69,626 transitions. It restores the checkpoint only to reuse its
training-fitted frozen normalization and does not use the actor. The test split
remains unopened.

## Artifact identity

- Schema: `hybrid-rf-vlc-rl.population-joint-risk-oracle-evaluation.v1`
- Artifact:
  `artifacts/evaluations/phase8_seed1001_population_joint_oracle.json`
- Artifact SHA-256:
  `a2d127dd3aedfb9e6f6cc652bc07c7eee8befbc37a4e1167e5986f448d302876`
- Checkpoint: iteration 265, policy seed 1001
- Checkpoint SHA-256:
  `39186e0b79465856369a75805237b79957e889166e35efcb4b02ac4303101dbf`
- State-regime window artifact SHA-256:
  `35b240c2fc58b46f02999d225663b6a91803509309cf098b4927e19aa6a7b2a7`
- Configuration hash:
  `df5cf40513f0c08ceba1b037b58a1002b9cc3fa87033601d0f3ee9ec452800c4`
- Policy-environment scope hash:
  `46fecd53689db6f2c9aa21b4314da610b0caa7e47444a4a9b01f5ba4a3b1cb29`
- Test split opened: no

## Reliability result

| Density | Transitions | Joint floor | Budget multiple | Usable-row floor | Fallback-row risk | Fallback share of total risk | Gate |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 7,938 | `5.395e-4` | 5.40x | `5.358e-4` | `7.524e-4` | 2.39% | fail |
| 20 | 23,204 | `1.485e-3` | 14.85x | `1.158e-3` | `1.663e-2` | 23.65% | fail |
| 30 | 38,484 | `1.849e-3` | 18.49x | `1.340e-3` | `2.401e-2` | 29.19% | fail |
| Campaign | 69,626 | `1.578e-3` | 15.78x | `1.187e-3` | `1.946e-2` | 26.41% | fail |

Only 1,491 transitions (2.14%) are forced fallbacks, but they contribute 26.4%
of campaign conditional risk. Improving missing-observation handling is
therefore important. It is not sufficient: the usable-row oracle floor alone
already exceeds `1e-4` by 5.36x, 11.58x, and 13.40x at densities 10, 20, and
30. The infeasibility is present even before fallback risk is added.

The sampled campaign miss rate is `1.537e-3`, close to the deterministic
conditional-risk mean `1.578e-3`. Sampled misses remain diagnostic only; the
gate uses the lower-variance conditional expectation.

## Exact joint action profile

| Action | Count | Campaign fraction |
|---|---:|---:|
| VLC | 50,737 | 72.87% |
| RF-4 | 6,116 | 8.78% |
| DUP-4 | 12,693 | 18.23% |
| DUP-1/2/3 combined | 80 | 0.11% |
| RF-1/2/3 combined | 0 | 0.00% |

The exact optimizer does not recover the proposed hand-written progression
through RF-1, RF-2, RF-3, and RF-4. It mainly offloads optically, reserves four
RF attempts for difficult links, and duplicates at four attempts. Intermediate
actions are almost entirely dominated under the current risk-only first
objective. The mean selected RF load is 1.083 attempts per pair; the selected
frame load ranges from 240 to 3,460 attempts as population reaches 1,096.

## Relationship to the residual-feasibility diagnosis

The earlier unilateral diagnosis held every other policy action fixed. It
found a density-10 floor of `6.13e-5` and optimistic zero-other-RF floors below
`1e-4`, but those focal optima could not be selected by all pairs
simultaneously. The joint oracle closes that gap: assigning strong RF actions
to difficult links raises shared contention, while assigning VLC avoids RF
load but retains optical failures. The best self-consistent population
assignment is consequently worse than the unilateral floor and still fails at
all densities.

This result separates two questions:

- The seed-1001 actor still has action regret relative to strong actions.
- Eliminating that regret cannot reach `1e-4` because the modeled system's
  omniscient joint optimum also fails.

## Next task: system-feasibility frontier

Before changing training again, add a versioned sensitivity evaluator around
the same exact oracle. Predeclare a small grid of physically interpretable
interventions and retain the unchanged headline point as the control:

1. RF resource-pool capacity (for example, explicitly modeled additional
   orthogonal subchannels or shorter per-attempt airtime);
2. the already declared RF sensing-reliability bands;
3. optical availability/error assumptions through named VLC configurations,
   without silently scaling outcomes; and
4. fallback handling as a separate diagnostic, while retaining the all-usable
   floor so fallback improvements are never mistaken for a complete fix.

Report the per-density joint floor, action mix, RF load, and distance to the
`1e-4` target at every point. Authorize a new learner only if one scientifically
defensible configuration passes all three densities. Otherwise revise the
research target or communication architecture explicitly rather than tuning
PPO against an impossible constraint.
