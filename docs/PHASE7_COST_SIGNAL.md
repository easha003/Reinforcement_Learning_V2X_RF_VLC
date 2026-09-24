# Phase 7 reliability cost-signal decision

Date: 2026-09-23

## Decision

The primary learner uses `conditional_miss_probability` as its only
reliability training signal.  The selected action's conditional risk feeds all
three training consumers:

1. the cost critic target and cost GAE;
2. the cost advantage subtracted in the primal PPO objective; and
3. the undiscounted per-density estimate used by projected dual ascent.

The actor does not receive the probability as an observation.  It affects the
actor only after action selection through the cost advantage and current
density multiplier.  This preserves decentralized execution and the existing
causal observation contract.

There is no sampled-cost training stage and no staged mixture in the primary
experiment.  `training.cost_signal` therefore accepts only
`conditional_miss_probability`, and the runtime selector rejects any other
value rather than silently changing estimators.

## Why conditional risk

The headline constraint is a miss probability of `1e-4`.  A binary signal at
that rate is almost always zero in an ordinary PPO rollout, producing a sparse,
high-variance cost advantage and unstable density multipliers.  Conditional
risk uses the already validated probability of the action selected under the
current geometry, propagation state, and joint-action load.  It supplies a
dense Rao-Blackwellized target: averaging the conditional probability estimates
the same model-implied miss probability while integrating out the final packet
mechanism draws.

A staged switch was rejected for the primary method because it introduces an
additional schedule and a nonstationary cost target without improving the
scientific estimand.  Sampled-cost or staged training may be studied later as
an explicitly configured ablation, but it is not an undocumented option in the
headline algorithm.

## Evaluation boundary

Conditional risk is a training estimator and supporting diagnostic, not proof
that the reliability constraint was met.  Every sampled binary miss remains in
the rollout/evaluation record.  Validation, checkpoint feasibility ranking,
final test reporting, and paper claims use realized sampled misses with the
configured cluster-aware one-sided confidence bound.  Conditional-risk results
may be reported alongside them but cannot replace that verdict.

`agents.cost_signal.ReliabilityCostBatch` makes this separation explicit:
`training_costs` equals the conditional probabilities, while
`sampled_miss_costs` is retained separately.  The selector validates aligned
shape, dtype, device, finite probabilities, binary sampled outcomes, and
detached rollout data before returning independent copies.

## Interpretation limit

The conditional probability is only as valid as the synthetic RF/VLC channel
and dependence model that produced it.  Using it reduces Monte Carlo variance;
it does not turn synthetic calibration into real-world evidence.  Final claims
must retain that scope, and model-sensitivity experiments remain necessary.
