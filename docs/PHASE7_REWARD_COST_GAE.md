# Phase 7 reward and cost generalized advantage estimation

Date: 2026-09-23

## Numerical contract

The estimator computes independent generalized advantage estimates for the
activation reward and the configured reliability-cost signal. For either
scalar signal `x`, its temporal-difference residual and reverse recursion are

```text
delta_t = x_t + gamma * bootstrap_t * V_next_t - V_t
A_t = delta_t + gamma * lambda * continuation_t * A_(t+1)
target_t = A_t + V_t
```

Reward and cost use separate value predictions and produce separate
advantages and critic targets. Neither stream is combined with the other in
this module. A later primal-dual PPO task will apply the learned multiplier at
the actor-objective boundary.

## Two-mask boundary

The estimator deliberately requires both masks defined by the Phase 5 return
contract:

- `value_bootstrap_mask` controls whether `V_next_t` enters the one-step
  residual; and
- `gae_continuation_mask` controls whether the residual recursion crosses into
  the following rollout index.

They are not interchangeable. In particular, a valid final-observation
bootstrap can be used while recursive GAE still stops at a reset. The caller
also supplies `next_values` explicitly so a boundary value never has to be
reconstructed from a post-reset observation.

This task implements and tests the numerical two-mask kernel. The companion
`PHASE7_LIFECYCLE_BOOTSTRAP.md` adapter now binds termination, truncation,
bootstrap validity, stable pair IDs, and separately evaluated final-observation
critic values to these inputs and tests GAE across all concrete lifecycle
cases.

## Variable populations

The time dimension is always explicit. This supports both a simple `(T,N)`
segment and the environment contract's padded `(B,T,N_max)` rollout layout.
`active_mask` is authoritative for padding: inactive estimates and targets are
zero, bootstrap and continuation masks cannot enable padded rows, and a
continuation cannot enter an inactive next-time slot. Empty time dimensions
are shape-preserving.

All rollout tensors must share shape, floating dtype, and device; masks must be
boolean on that same device. Results preserve signal dtype and device and are
detached from rollout-time computation graphs before PPO minibatch reuse.

## Verification

The unit suite checks a hand-derived three-step trajectory, the distinction
between bootstrapping and recursive continuation, independent reward/cost
signals and critics, non-leading time dimensions, padding isolation, detached
outputs, empty rollouts, and fail-closed shape, dtype, mask, parameter, and
continuity validation.
