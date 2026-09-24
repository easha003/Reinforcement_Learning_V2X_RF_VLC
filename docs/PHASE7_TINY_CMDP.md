# Phase 7 Tiny-CMDP Convergence Test

## Purpose

This test closes the Phase 7 learning gate by exercising the complete
primal-dual PPO path on a problem with a known constrained solution. It is an
algorithmic convergence test, not evidence about physical RF/VLC performance.

## Problem definition

The environment has one observation and the repository's nine-action
categorical interface. Only two abstract actions are valid; the other seven
are masked.

| Abstract action | Reward | Binary miss cost | Feasible as a deterministic policy? |
|---|---:|---:|---|
| Risky | 1.0 | 1.0 | No |
| Safe | 0.8 | 0.0 | Yes |

The miss budget is 0.1. Therefore:

- unconstrained reward maximization selects the risky action;
- the optimal deterministic feasible policy selects the safe action; and
- a reliability penalty must change the learned solution for the constrained
  learner to pass.

The action names are deliberately abstract. Their indices reuse the production
action width only to test the same categorical masking and sampling machinery.

## Learning path exercised

The test uses production implementations for:

- masked categorical sampling and deterministic action selection;
- the shared categorical actor and separate reward and cost critics;
- reward/cost generalized advantage estimation;
- clipped PPO actor and critic updates; and
- per-density projected dual ascent.

Each sampled action ends a one-step episode, so neither value stream
bootstraps. Fifty batches of 128 episodes are trained with four PPO epochs per
batch. The constrained and reward-only learners use the same fixed seed and
initial parameters; only the cost penalty and dual update differ.

## Acceptance criteria

The constrained learner must assign more than 0.97 probability to the safe
action, choose it deterministically, keep the recent sampled miss rate below
0.05, and learn reward and cost values near 0.8 and 0.0. Its dual trajectory
must react to early violations and remain meaningfully nonzero. Invalid masked
actions must retain exactly zero probability.

The reward-only control must assign more than 0.97 probability to the risky
action, choose it deterministically, and retain a recent miss rate above 0.95.
These thresholds test broad convergence margins rather than exact floating-
point trajectories.

## Reproducibility scope

The test fixes model initialization and action-sampling seeds. It is intended
as a fast deterministic regression test for the local training implementation.
It does not replace Phase 8 trace-based smoke training, throughput profiling,
multi-seed training, or held-out reliability evaluation.
