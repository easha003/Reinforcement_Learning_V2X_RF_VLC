# RL extension: initial problem contract

This initial scope is retained as the research rationale. The normative,
implementation-level contract is now
[`RL_ENVIRONMENT_CONTRACT_V1.md`](RL_ENVIRONMENT_CONTRACT_V1.md), version
`1.0.0`.

## Research question

Can causal decentralized RF/VLC decisions reduce activation and shared RF-pool
use while maintaining the packet deadline-miss constraint, when the population's
joint behavior changes the contention process experienced by every vehicle?

This is a constrained mean-field or multi-agent control problem. It is not the
former per-packet link-selection CMDP, where RF risk was fixed before the agent
acted.

## Reused validated foundation

- Microscopic mobility traces and disjoint train/validation/test splits.
- Vehicle and building geometry, tagged pairs, and optical occlusion.
- NR sidelink and VLC channel models.
- Causal noisy observations and action-dependent link feedback.
- Matched packet randomness and counterfactual single-link outcomes.
- Cluster-aware reliability statistics.
- The analytical load-allocation fixed point as an oracle bound.

The completed paper, presentation material, generated results, and historical
planning documents are not part of this repository.

## Environment contract

At decision epoch `t`:

1. Each active transmitter receives a local causal observation.
2. Each transmitter chooses a medium and RF reservation level. The initial
   action design will cover VLC-only and RF use with zero through the service
   profile's maximum RF attempts; duplication is the joint VLC-plus-RF case.
3. The environment aggregates population actions into offered RF load.
4. Pool demand and collision probabilities are recomputed from that load.
5. Packet outcomes and action-dependent feedback are produced.
6. The next observation contains only locally available measurements and an
   explicitly delayed or measured congestion signal.

Mobility remains exogenous in the first extension: communication decisions do
not change vehicle motion. This lets the existing traces be replayed without
claiming that they are an offline behavior-policy dataset.

## Objective and constraints

- Reward: negative activation/resource cost.
- Primary constraint: deadline-miss probability at each traffic density.
- Resource accounting: RF attempts consume the common Mode-2 pool; VLC-only
  packets return their RF reservation.
- Statistical decision rule: a policy is feasible only when the declared
  one-sided, cluster-aware upper confidence bound is below the miss budget.

Binary misses at `1e-5` are too sparse for a stable training signal. Training
may use simulator conditional risk, but final evaluation must also report
realized miss counts and confidence bounds.

## Baselines

1. RF-only, VLC-only, and duplicate-all.
2. Observable geometric threshold selector.
3. Supervised optical-risk estimator plus analytical fixed-point allocation.
4. Constrained contextual policy with no history.
5. Analytical truth-risk equilibrium as a non-deployable upper bound.

RL is retained only if it improves cost or pool use over deployable baselines at
matched statistical feasibility.

## Required ablations

- Local policy with versus without link-history features.
- Fixed RF risk versus action-coupled mean-field risk.
- Centralized training information removed at decentralized execution.
- Density and channel-model shifts outside the training support.
- Multiple policy seeds and independent mobility replicates.

## Implementation order

1. Preserve the inherited simulator tests under the new package name.
2. Implement and test a population action ledger and load-conditioned RF risk.
3. Build a vectorized Gymnasium-compatible mean-field environment.
4. Add deterministic and supervised deployable baselines.
5. Add the simplest constrained learner that can test the hypothesis.
6. Evaluate on untouched traces and sensitivity conditions.

Algorithm choice is intentionally deferred until the environment and baselines
make the learning signal explicit. PPO is one candidate, not a requirement.
