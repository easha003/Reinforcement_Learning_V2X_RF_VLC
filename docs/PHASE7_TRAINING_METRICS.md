# Phase 7 training metrics and JSONL logging

Date: 2026-09-24

## Record boundary

`agents.training_metrics` emits one versioned record after each complete
rollout and PPO optimization iteration.  A record combines three scopes that
must not be averaged together blindly:

- PPO optimizer diagnostics are row-weighted over every minibatch in every
  update epoch;
- reward and cost explained variance are computed once over the unique
  learning-eligible rollout rows using pre-update critic predictions and GAE
  targets; and
- constraint estimates and dual values are recorded once per configured
  density from the atomic projected-ascent report and its resulting snapshot.

The JSONL schema is `hybrid-rf-vlc-rl.training-metrics.v1`.  Each row carries
the configuration hash, policy seed, zero-based iteration, cumulative
environment transitions, current rollout transitions, unique learning-row
count, optimizer step/row counts, and the configured
`conditional_miss_probability` signal name.  Density sample counts must
partition the collected rollout transitions; learning rows may be fewer when
fallback packets remain part of the deployment constraint but not PPO.

## PPO diagnostics

Every `PPOUpdateMetrics` now includes its actual minibatch size.  The logger
uses that size to compute row-weighted means for actor loss, clipped policy
loss, reward-value loss, cost-value loss, entropy, approximate KL divergence,
clip fraction, and mean probability ratio.  This remains correct when the last
minibatch is smaller than the others.  `optimizer_rows` counts repeated rows
across update epochs; it is not mislabeled as newly collected transitions.

## Critic explained variance

For each critic the logged diagnostic is

```text
1 - Var(target - prediction) / Var(target)
```

using population variance in float64 over the flattened, learning-eligible
rollout exactly once.  Perfect prediction gives one; zero means no improvement
over a constant prediction; negative values are retained because they diagnose
a critic worse than that reference.  A constant target has no explainable
variance, so the log uses the explicit finite sentinel zero instead of NaN.

## Density constraints and duals

Each density row contains its sample count, conditional-miss estimate, active
curriculum budget, violation, multiplier before/after projection, learning
rate, cap, and cumulative density update count.  An absent density is explicit:
its count is zero, estimate and violation are JSON `null`, multiplier is
unchanged, and update count does not advance.  The builder cross-checks the
post-update multiplier against the supplied immutable dual snapshot.

These are training diagnostics based on conditional risk.  They do not replace
sampled-miss validation/test metrics or their cluster-aware confidence bounds.

## Persistence and resume safety

`TrainingMetricsJSONL` is a single-writer append-only sink.  Records use sorted,
compact JSON with non-finite values forbidden and are flushed and `fsync`ed on
append.  When reopening an existing log, every row's schema and monotonic
iteration/environment-transition counters are checked.  A partial final line,
wrong schema, duplicate iteration, counter rollback, directory target, or
symlink target fails closed.  Configuration hash and policy seed are fixed for
the lifetime of one log, including after reopening, so two runs cannot be
silently concatenated.

## Verification

Tests cover known explained-variance cases, the constant-target convention,
row-weighted minibatch aggregation, complete density/dual serialization,
absent-density representation, deterministic JSONL append/resume behavior,
and invalid scalar, tensor, schema, and partial-record rejection.
