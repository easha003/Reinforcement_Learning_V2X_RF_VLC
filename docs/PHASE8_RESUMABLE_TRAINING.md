# Phase 8 Resumable Joint-Density Training

## Outcome

The repository now has a production training driver that can start a fresh
joint-density primal-dual PPO run, stop at an explicit invocation boundary, and
resume from its latest immutable checkpoint without resetting learner or random
state. The driver advances cumulative counters and the declared reliability
curriculum while treating the configured per-seed transition budget as a hard
upper bound.

The historical `training joint-iteration` command remains available as a
one-update vertical slice. The production continuation boundary is exposed as:

```bash
hybrid-v2x-rl training joint-train \
  --output artifacts/logs/phase8-joint-seed1001
```

An optional `--max-iterations` limit stops one invocation without changing the
configured experiment budget. It is intended for bounded pilots, operational
batching, and resume verification. Omitting it runs until the configured budget
is reached or the unused budget cannot hold one complete balanced round.

## Complete resume state

Every iteration checkpoint retains and restores:

- the shared actor, reward critic, and cost critic;
- all three optimizer states;
- the density-specific dual multipliers and update counts;
- training-only observation-normalization statistics;
- cumulative iteration, environment-transition, learning-row, episode, and
  optimizer-step counters;
- Python, NumPy, PyTorch CPU, and supported accelerator global random states;
  and
- the named training, policy-action, and PPO-minibatch generators.

The next source replicate is selected by an iteration-offset round-robin rule:
`(completed iteration + local balanced round) mod available replicates`. This
keeps source scheduling deterministic across process boundaries. Environment
seeds, policy samples, and minibatch permutations continue from the restored
named generators rather than being reseeded.

A resume is accepted only from the latest checkpoint in the requested output
root. Its filename, checkpoint counters, complete checkpoint/report sequence,
and append-only metric history must agree. The optional
`--expected-checkpoint-sha256` adds an independent artifact-digest assertion.
Older checkpoints cannot silently fork an existing run.

## Hard transition budget

The configured `training.total_transitions_per_seed` counts acted pair
transitions and is a hard ceiling. A density-balanced round is the smallest
collection unit. Before a round starts, the driver computes its exact acted
transition count from the immutable pair episode schedules and requested frame
limit. The driver does not draw seeds, run environments, or mutate normalization
if that round would cross the budget.

If at least one full round has already been collected for the current update,
the driver may optimize that smaller final batch. If the remaining budget
cannot hold even one full round, it stops with
`insufficient_budget_for_balanced_round` and reports the unused tail. This is
preferable to truncating a simultaneous frame, omitting a density, or publishing
an over-budget checkpoint.

The rollout packet count remains an update-size target. Collection can exceed
that target only by finishing its current balanced round. Near the total budget,
the last update can instead finish below the target when another full round
would exceed the hard environment-transition ceiling.

## Reliability curriculum

Curriculum stages are resolved from cumulative acted transitions at the start
of every update. For the headline 10-million-transition configuration, the
boundaries are:

| Stage | Cumulative transition interval | Miss budget |
|---:|---:|---:|
| 0 | `[0, 1,000,000)` | `1e-2` |
| 1 | `[1,000,000, 3,000,000)` | `1e-3` |
| 2 | `[3,000,000, 10,000,000)` | `1e-4` |

The selected miss budget is used by every density-specific dual update in that
PPO iteration. Because a balanced round is indivisible, a stage boundary can be
crossed by the final round of an update. The iteration report records whether
that happened and the exact boundary overshoot; the next iteration uses the new
stage. No samples within one PPO batch are assigned conflicting dual budgets.

## Immutable artifact layout

A fresh run creates:

```text
OUTPUT/
├── metrics.jsonl
├── checkpoints/
│   ├── checkpoint-iteration-000001.pt
│   └── checkpoint-iteration-000002.pt
├── iterations/
│   ├── iteration-000001.json
│   └── iteration-000002.json
└── sessions/
    ├── session-000000-000001.json
    └── session-000001-000002.json
```

Checkpoints and iteration reports are never overwritten. `metrics.jsonl` is
append-only and rejects repeated or nonmonotonic iteration/transition counters.
Each invocation publishes a new session report identifying its start and end
counters, stop reason, unused budget, latest checkpoint, and newly created
artifacts.

Resume example:

```bash
hybrid-v2x-rl training joint-train \
  --output artifacts/logs/phase8-joint-seed1001 \
  --resume-checkpoint \
    artifacts/logs/phase8-joint-seed1001/checkpoints/checkpoint-iteration-000001.pt \
  --expected-checkpoint-sha256 <sha256>
```

## Verification

The integration tests establish that:

- two uninterrupted updates and one update followed by restore and another
  update produce identical JSONL metrics;
- actor and critic tensors, dual state, normalization state, cumulative
  counters, and named random-generator states match exactly;
- resuming appends new evidence without changing the prior checkpoint or
  iteration report;
- the curriculum moves from `1e-2` to `1e-3` at the declared cumulative
  boundary; and
- a final balanced round that would cross the total transition budget is not
  started.

Focused lint, formatting, and static typing pass. The complete repository test
suite passes 1,382 tests with 2 expected artifact-dependent skips.

The next operational task is a bounded multi-iteration pilot on the configured
training traces, including a real checkpoint/resume cycle and artifact audit,
before committing to the full 10-million-transition run for each policy seed.
