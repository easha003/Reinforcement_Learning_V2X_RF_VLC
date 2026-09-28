# Phase 6 baseline suite

> Migration note (2026-09-27): realized baseline outcomes and analytical
> allocation now use the shared pair-local RF pipeline. References below to a
> single `RFPoolDemand`/`RFPoolModel` describe the historical Phase 6
> implementation. The centralized model-based allocators now evaluate complete
> pair-local joint ledgers; see `PAIR_LOCAL_ROLLOUT_MIGRATION_V1.md`.

## Purpose

This suite establishes the comparison boundary that PPO must beat. A baseline
only chooses the joint action. After that choice, every baseline uses the same
action mask, missing-observation fallback, action ledger, RF-pool model,
identity-addressed random tapes, physical channels, packet-outcome assembly,
feedback path, and pair lifecycle as the Phase 5 environment.

The shared runner is `run_policy_rollout`. `run_deterministic_rollout` adapts
the Phase 5 random/cycle checks to it, and `run_baseline_rollout` sends Phase 6
policies to the same function. There is no baseline-specific scoring path.

## Information boundary

`PopulationPolicyFrame` contains only the current population frame, raw causal
actor rows, normalized actor tensor, hardware action mask, configured resource
costs, analytical RF-pool model, and service miss budget. Physical channel
truth is a separate mapping. The runner passes that mapping only when a policy
declares `requires_oracle_truth=True`.

All deployable baselines reject a non-null truth mapping. The truth-risk oracle
requires one and is therefore explicitly non-deployable. Channel truth is
materialized before the action because it is action-independent, but its
availability inside the environment does not make it an actor observation.

Rows without a usable causal observation never reach a policy decision. They
take the contract fallback `DUP-4`, remain in resource and outcome accounting,
and remain excluded from policy learning exactly as in Phase 5.

## Canonical baselines

| Artifact name | Decision rule | Information used |
|---|---|---|
| `always-rf-1` through `always-rf-4` | Fixed RF reservation level | None |
| `always-vlc` | VLC only | None |
| `duplicate-all` | `DUP-4` | None |
| `geometry-threshold` | VLC when tracked range is below 9.5 m and observable FOV margin exceeds 0.90 rad; otherwise `RF-4` | Current noisy/predicted geometry |
| `contextual-no-history` | Cheapest per-packet action estimated to meet the miss budget; otherwise minimum estimated risk | Current CBR, geometry, traffic context, and causal blockage forecast |
| `supervised-risk-allocation` | Training-fitted optical risk plus an action-coupled analytical load allocation | Current causal context and analytical RF access risk |
| `truth-risk-oracle` | The same load allocation with exact current VLC risk and RF decoding risk | Privileged simulator truth |

The contextual policy cannot read RF/VLC quality, quality ages, quality
histories, previous action, last outcome, consecutive misses, or delayed
mean-field history. It is the stateless comparison needed to test whether the
full sequential observation provides additional value.

## Supervised estimator

`SupervisedOpticalRiskEstimator.fit` accepts raw training-split actor rows and
their simulator conditional VLC miss probabilities. It fits ridge regression
on log-odds after standardizing only these current-context features:

- neighbour count;
- pair distance and bearing;
- relative speed and heading difference;
- optical FOV margin;
- junction distance and path-spanning flag;
- predicted blockage probability and confidence; and
- track age.

The fit API deliberately requires a fitted estimator when constructing
`supervised-risk-allocation`; there is no default model trained on validation
or test data. Phase 6's matched-split task will own collection, persistence,
and train-only fitting.

The deployable allocator uses the inherited optimistic RF assumption: current
RF decoding error is zero and action-coupled access risk comes from the common
pool model. This is stronger than inventing a noisy current SINR and avoids
leaking simulator propagation truth. Final packet scoring still uses the real
RF propagation result.

## Analytical joint allocation

For a provisional joint action, the allocator builds a real
`FrameActionLedger`, derives `RFPoolDemand`, and evaluates the real
`RFPoolModel`. It then calculates the risk of every permitted action from the
resulting access risk and the policy's optical belief. Starting from several RF
load states, it repeatedly applies the best marginal risk reduction per added
activation cost until the population mean risk meets the service budget or no
further reduction exists. Iteration stops on a fixed point or a detected
cycle; the lowest-cost feasible candidate is selected, otherwise the
lowest-risk candidate is retained.

This is an analytical fixed-point comparator over the realized nine-action
boundary, not a proof of the globally optimal mixed-integer allocation. The
truth-risk version is an information oracle: it measures the headroom created
by perfect current channel knowledge under the same allocator and accounting.
Any paper claim must call it a privileged-information oracle reference unless
a later exact-optimization check proves the stronger mathematical bound.

Zero RF demand is a first-class allocation state. Its collision and
half-duplex access risk are both zero; the implementation handles it without
constructing an invalid zero-airtime legacy collision parameter.

## Current verification

Unit and integrated tests establish that:

- every fixed baseline's RF attempts and VLC activations reconcile with usable
  rows plus the common fallback rows;
- geometry, contextual, supervised, and truth-oracle policies complete the
  population rollout path;
- causal policies are never handed the oracle mapping;
- contextual/supervised feature lists exclude temporal link history; and
- the analytical access-risk helper agrees with realized RF attempt accounting
  and handles an all-VLC population.

Matched trace splits, matched-random-number campaign execution, expected
ordering checks, per-density metrics, and the measured deployable-to-oracle gap
are the following Phase 6 tasks; they are not claimed by this implementation
step.
