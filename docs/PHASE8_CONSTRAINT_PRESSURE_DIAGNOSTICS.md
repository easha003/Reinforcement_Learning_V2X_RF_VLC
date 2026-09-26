# Phase 8 constraint-pressure diagnostics

Date: 2026-09-26

## Purpose

The first full seed-1001 pilot converged numerically while assigning 99.58% of
its final-window actions to `RF-1` or `VLC`. Its conditional miss estimates
remained hundreds of times above the `1e-4` target. Aggregate PPO losses and
dual values were insufficient to determine whether reward advantages,
reliability advantages, or their Lagrangian scaling caused that behavior.

This task adds a read-only pre-update diagnostic boundary. It records the
quantities actually entering the constrained actor objective before any PPO or
dual mutation:

```text
A_actor = A_reward - lambda_density * A_cost
```

The diagnostics are evidence for designing a bounded recovery experiment.
They do not change the algorithm or claim that dual scaling is the sole cause
of the failed pilot.

## Versioned artifact boundary

Every default multi-iteration training report now contains:

```text
constraint_pressure.schema =
  hybrid-rf-vlc-rl.constraint-pressure-diagnostics.v1
```

The containing joint-density iteration report advances from schema v2 to v3:

```text
hybrid-rf-vlc-rl.joint-density-training-iteration.v3
```

Resume validation requires both the v3 iteration schema and the nested
constraint-pressure schema. Historical v2 reports, including the completed
seed-1001 engineering pilot, are intentionally not resumable with this trainer.
That pilot remains immutable audit evidence and is already ineligible for paper
checkpoint selection; the corrected scientific seed must begin at transition
zero.

## Measurement point

The diagnostic is constructed after all density-balanced rollout segments have
been merged and before:

1. minibatch permutations are drawn;
2. any PPO optimizer step occurs; or
3. the per-density dual ascent update occurs.

The actor is re-evaluated under `torch.no_grad()` with the stored rollout
observations, masks, and selected actions. Its evaluated log probabilities must
match the stored old-policy log probabilities within `1e-6`; otherwise the
iteration fails rather than logging statistics from the wrong policy.

The actor architecture contains only linear and tanh layers, so evaluation
draws no randomness and owns no mutable running statistics. The code neither
samples an action nor touches the explicit action or minibatch generators.

## Per-density fields

Diagnostics partition the PPO learning rows by their exact configured density.
Each density row records:

- learning-row count;
- dual multiplier used by those rollout rows before the dual update;
- signed mean, population standard deviation, mean absolute magnitude, minimum,
  and maximum for reward advantage;
- the same distribution summary for cost advantage;
- the same summary for `lambda * A_cost`;
- the same summary for the combined actor advantage;
- the ratio
  `mean(abs(lambda * A_cost)) / mean(abs(A_reward))`, or `null` when the reward
  magnitude is exactly zero;
- fraction of combined advantages that are positive;
- categorical entropy distribution;
- mean probability of every canonical action;
- availability fraction of every action under the stored hardware masks; and
- selected-action counts over learning-eligible rows.

The iteration report's existing `action_counts` still covers every acted
rollout row, including declared no-observation fallback rows. The new selected
counts deliberately cover only rows entering PPO, so the two have different
but explicit denominators.

## Invariants

The diagnostic builder rejects:

- non-finite, nonpositive, misaligned, or gradient-owning density labels;
- an actor whose current log probabilities do not match the rollout policy;
- density rows with inconsistent dual weights;
- missing or duplicate density summaries;
- probability vectors that do not sum to one;
- nonzero probability or selections for an action unavailable on every row;
- selected-action counts that do not partition the density's PPO rows; and
- non-finite advantage, entropy, ratio, or fraction values.

All report serialization remains strict JSON with no NaN or infinity values.

## Determinism evidence

Focused unit and integration coverage proves:

- known reward, cost, weighted-cost, and combined-advantage summaries;
- exact per-density partitioning;
- correct probability and availability summaries under partial action masks;
- preservation of the actor's training mode and bitwise-unchanged parameters;
- unchanged PyTorch global RNG state;
- rejection when the actor differs from the rollout policy;
- schema and learning-row persistence in iteration reports; and
- byte-identical checkpoints and metrics for otherwise identical one-iteration
  runs with diagnostics enabled versus internally disabled.

The broader constrained-PPO regression set passes 134 tests spanning action
sampling, GAE, PPO, dual ascent, metrics, checkpointing, trace preparation,
joint-density training, window scheduling, and split/resumed execution.
The complete repository suite passes 1,412 tests with two expected skips for
optional training-cache and evaluation artifacts absent from this checkout.

## Interpretation for the recovery experiment

The most direct scale indicator is
`mean_absolute_constraint_to_reward_ratio`:

- a value far below one while miss risk violates the budget indicates weak
  reliability pressure relative to the resource objective;
- a value near or above one means scale alone may not explain action selection,
  so signs, state conditioning, masks, entropy, and RF-load feedback must be
  examined; and
- a large value paired with continued cheap-action selection points away from
  simple dual-rate scaling and toward cost-advantage quality or observability.

The next task is a bounded, versioned experiment that first reproduces the
current configuration with these diagnostics and then changes one declared
constraint-enforcement parameter at a time. No further full seed should begin
until that experiment establishes a credible reliability response.
