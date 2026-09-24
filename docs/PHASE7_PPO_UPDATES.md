# Phase 7 clipped PPO and critic updates

Date: 2026-09-23

## Optimization boundary

`agents.ppo` implements one flattened PPO minibatch update over rows already
selected by `active_mask & learn_mask`. The actor consumes the stored
37-column normalized observation and its original nine-action hardware mask.
The reward and cost critics are separate two-by-64 tanh networks over the
78-column centralized training observation.

The rollout batch retains the action and log probability produced by the old
policy. Re-evaluation always uses the stored action mask, so PPO cannot assign
probability mass to a hardware-impossible action during an update.

## Actor objective

The constrained actor advantage is assembled as

```text
A_actor = A_reward - weight_cost * A_cost
ratio   = exp(log pi_new(a|s) - log pi_old(a|s))
L_clip  = mean(min(ratio * A_actor,
                   clip(ratio, 1-epsilon, 1+epsilon) * A_actor))
loss_actor = -L_clip - entropy_coefficient * mean(entropy)
```

`A_cost` is computed from the configured selected-action conditional miss
probability; sampled binary misses are retained for final feasibility rather
than substituted into the primary PPO estimator. `weight_cost` is an
externally supplied, nonnegative, detached value for each row.
`agents.dual_ascent.PerDensityDualAscent` owns its exact
density-to-dual assignment and projected ascent update. Keeping that state
outside this module separates the once-per-rollout constraint estimate from
the epoch/minibatch PPO updates and keeps clipping behavior independently
testable.

Advantages are not silently normalized. Reward and reliability cost have
declared physical meanings, and an unversioned normalization choice would
change the effective scale of the Lagrange multiplier.

## Critic updates

Reward and cost critics have independent parameters and Adam optimizers. Each
minimizes mean squared error against its own detached GAE value target. The
PPO clip ratio applies to the policy surrogate only; value clipping is not
introduced because the current training configuration defines no value-clip
hyperparameter.

All three losses are evaluated before mutation, all gradients are checked for
finiteness, and only then are the three optimizer steps applied. Returned
metrics are pre-update scalars. Persistent metric logging, explained variance,
checkpoint state, epoch/minibatch scheduling, and early stopping remain later
tasks.

## Verification

Tests prove the sign-dependent PPO clipping rule and zero gradients in clipped
directions, identity behavior at ratio one, reward-minus-cost advantage
composition, entropy regularization, independent critic targets and gradients,
the configured actor/critic architectures, three-network parameter updates,
masked-action rejection, and fail-closed minibatch contracts.
