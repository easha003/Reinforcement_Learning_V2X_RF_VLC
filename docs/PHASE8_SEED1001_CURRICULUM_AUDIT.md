# Phase 8 seed-1001 curriculum and stability audit

Date: 2026-09-25

## Verdict

The first configured 10-million-transition joint-density run is operationally
complete. Artifact continuity, curriculum advancement, trace scheduling,
checkpoint restoration, and PPO numerical diagnostics pass their engineering
checks.

The run does **not** pass the scientific continuation gate. In the final 20
updates, the rollout-weighted conditional miss estimates were approximately
`7.11e-2`, `7.23e-2`, and `6.46e-2` at densities 10, 20, and 30 vehicles per
lane-kilometer. These are 646 to 723 times the active `1e-4` budget. Over the
same window, 99.58% of rollout actions were either `RF-1` or `VLC`; the policy
had largely abandoned the higher-redundancy actions required to trade resource
cost for reliability.

Therefore this seed is an engineering/curriculum pilot, not a selectable paper
checkpoint. Seeds 1002--1005 must not be launched with the unchanged
hyperparameters. The next task is a bounded constraint-scaling diagnosis and
recovery experiment, followed by a fresh seed-1001 run only after the learner
demonstrates reliability movement at all three densities.

## Run identity

- Local ignored artifact root:
  `artifacts/logs/phase8-joint-seed1001`
- Policy seed: `1001`
- Device: CPU on the MacBook Pro M2
- Configuration hash:
  `69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`
- Training sources: all nine configured training traces, with three independent
  replicates at each of densities 10, 20, and 30 vehicles/lane-kilometer
- Requested budget: 10,000,000 environment transitions
- Completed: 9,998,802 environment transitions in 265 PPO iterations
- Stop reason: `insufficient_budget_for_balanced_round`
- Unused tail: 1,198 transitions, or 0.01198% of the configured budget
- Final checkpoint:
  `checkpoints/checkpoint-iteration-000265.pt`
- Final checkpoint SHA-256:
  `ff4a76177596dfabc3bd0c2f3001555e21b299bed07f84f02a20a6ec5f86a612`

The 1,198-transition tail is expected. Starting another complete balanced round
would have exceeded the hard per-seed budget, so the scheduler stopped before
that round rather than biasing the final update toward one density.

## Recovery history and code provenance

The campaign exposed two previously unobserved truncation boundaries. Both
failures occurred before the affected iteration was published, so the last
immutable checkpoint remained a safe resume boundary.

| Boundary | Last safe state | Resolution |
|---|---:|---|
| Internal truncation followed by an ordinary empty population | checkpoint 86, 3,271,262 transitions | Commit `91a8c81` constructs the centralized bootstrap suffix explicitly for an empty next population. |
| Max-duration truncation whose endpoint disappears from the next physical frame | checkpoint 137, 5,215,074 transitions | Commit `b3bd15b` uses one-frame endpoint lookahead and disables bootstrap when the next pair observation cannot exist. |
| Conversation/monitor interruption | checkpoint 237, 8,994,674 transitions | No trainer process remained; the matching checkpoint/report/metric boundary was verified and resumed by digest. |

The campaign began after trace-window commit `de60f29` and completed at
`b3bd15b`. Completed iterations before each failure had not encountered the
new boundary: the old behavior failed on the first such attempted iteration,
and no partial iteration artifact was published. Nevertheless, because code
changed during this engineering run, its checkpoints must not be presented as
final paper artifacts. A final scientific run must start from one frozen
commit after the constraint-learning issue below is corrected.

## Artifact and resume integrity

The completed tree occupies approximately 97 MiB and contains 534 regular,
non-symbolic-link files:

| Artifact class | Count |
|---|---:|
| Immutable checkpoints | 265 |
| Immutable iteration reports | 265 |
| Append-only metric rows | 265 |
| Completed invocation/session reports | 3 |

The small session-report count is explained by the two exception exits and the
lost monitor process; session reports are written only on normal invocation
completion. It does not create a gap in training state. The checkpoint,
iteration-report, and metric histories are contiguous from iterations 0 through
264. Environment-transition counters are strictly increasing, the JSONL file
is newline-terminated, and every iteration report's embedded checkpoint path,
size, SHA-256, and cumulative counters match the corresponding file.

All 1,671 segment reports contain complete rollout and matched-tape
fingerprints, and all 1,671 environment seeds are distinct. The final counters
reconcile exactly with the sum of the immutable iteration reports:

| Counter | Final value |
|---|---:|
| Completed iterations | 265 |
| Environment transitions | 9,998,802 |
| Rollout transitions | 8,902,351 |
| Learning transitions | 8,900,869 |
| Completed pair episodes | 31,604 |
| PPO optimizer steps | 88,200 |
| Internal max-duration truncations | 8,580 |

The 1,482-row difference between rollout and learning transitions occurs only
in iteration 0 and is the declared no-observation fallback mask, not lost
training state.

## Curriculum accounting

The curriculum advanced from cumulative acted transitions and crossed both
declared boundaries exactly once. Complete balanced rounds may pass a
curriculum boundary; the overshoot is recorded, and the following iteration
uses the next budget.

| Stage | Active miss budget | Iterations | Iteration range | Transition interval | Boundary overshoot |
|---:|---:|---:|---:|---:|---:|
| 0 | `1e-2` | 26 | 0--25 | 0--1,004,293 | 4,293 |
| 1 | `1e-3` | 53 | 26--78 | 1,004,293--3,005,673 | 5,673 |
| 2 | `1e-4` | 186 | 79--264 | 3,005,673--9,998,802 | 0 |

Every density received one dual update in every PPO iteration. The final dual
multipliers were `(1.59509, 1.57200, 1.49899)`, with update counts
`(265, 265, 265)`. All three multipliers increased monotonically because every
reported density estimate remained above its active budget.

## Density and temporal coverage

The scheduler executed 557 complete density-balanced rounds and emitted 1,671
segments. It used all nine training traces and advanced through schedule cycles
0 through 29. Source use remained close across replicates: 181, 186, and 190
windows per trace at each density.

Because the three density traces with the same replicate suffix share the
window schedule, temporal coverage is summarized once per suffix:

| Replicate | Windows per density | Unique frames per trace | Trace coverage | Observed time-axis span | Wraps |
|---:|---:|---:|---:|---:|---:|
| 000 | 186 | 1,237 / 9,000 | 13.74% | frame 0--8,991 | 181 |
| 001 | 190 | 1,241 / 9,000 | 13.79% | frame 272--7,106 | 185 |
| 002 | 181 | 1,181 / 9,000 | 13.12% | frame 1,269--8,175 | 176 |

This is broad, repeatable sampling over the trace time axis rather than the old
frame-zero prefix replay. It is not exhaustive visitation of every physical
frame, which is neither promised nor required by the sampled-window contract.

The density totals also reconcile with the global counters:

| Density | Environment transitions | Learning rows |
|---:|---:|---:|
| 10 | 1,246,971 | 1,109,981 |
| 20 | 3,352,018 | 2,983,512 |
| 30 | 5,399,813 | 4,807,376 |

The larger packet populations at higher densities naturally contribute more
rows even though every round includes all three densities.

## PPO numerical stability

Every persisted numeric metric is finite. No model, optimizer, counter, or
random-state corruption was detected.

| Diagnostic | First | Final | Last-20 mean | Observed range |
|---|---:|---:|---:|---:|
| Entropy | 2.15736 | 0.18927 | 0.08290 | 0.05827--2.15736 |
| Approximate KL | 0.03363 | 0.00473 | 0.00161 | 0.000664--0.03363 |
| Clip fraction | 0.19844 | 0.04171 | 0.01546 | 0.00841--0.38063 |
| Ratio mean | 0.99375 | 0.99811 | 0.99976 | 0.99375--1.00213 |
| Reward explained variance | -0.00865 | 0.94817 | 0.91063 | -0.00865--0.96833 |
| Cost explained variance | 0.01082 | 0.93867 | 0.93763 | 0.01082--0.97470 |

KL, clipping, ratios, critics, and losses show a numerically controlled run.
The entropy trajectory, however, is a visible policy-concentration warning
rather than evidence of a useful constrained solution.

## Constraint behavior and action concentration

The table below weights the conditional miss estimate by each density's rollout
sample count. Action fractions use all rollout decisions, including the
declared iteration-0 fallback rows.

| Stage | Budget | Miss at density 10 | Miss at density 20 | Miss at density 30 | `RF-1` | `VLC` | Higher RF | DUP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `1e-2` | 0.23664 | 0.20653 | 0.19714 | 38.64% | 43.95% | 7.36% | 10.06% |
| 1 | `1e-3` | 0.13802 | 0.12082 | 0.09886 | 43.63% | 56.21% | 0.10% | 0.07% |
| 2 | `1e-4` | 0.10110 | 0.10733 | 0.10690 | 53.50% | 46.27% | 0.05% | 0.18% |

The final 20-update window improves to `(0.07110, 0.07230, 0.06461)`, but it
remains orders of magnitude infeasible. Its action mix is 48.81% `RF-1`,
50.77% `VLC`, and only 0.42% across all seven higher-redundancy actions.

This is not merely sampling noise around the target. The configured dual step
is `0.05`, so 265 positive updates raise the multipliers only to about 1.5.
With resource rewards separated by whole activation-cost units, the observed
multipliers did not make the reliability advantage of additional RF attempts
or duplication competitive soon enough. This is an evidence-based scaling
hypothesis, not yet a proven sole cause; action masking, advantage magnitudes,
and joint RF-load feedback must be measured in the recovery experiment.

## Final-checkpoint restoration audit

Checkpoint 265 restores successfully against the headline configuration and
its independently verified SHA-256. Two independent restores produced:

- bitwise-equal actor, reward-critic, and cost-critic tensors;
- bitwise-equal deterministic actions and log probabilities on a fixed probe;
- finite tensors in all three models;
- six populated Adam state entries for each of the three optimizers;
- dual multipliers and update counts matching the final metric row;
- observation-normalization counts of 9,997,320 for every standardized actor
  column;
- NumPy generator `training_streams`; and
- PyTorch generators `policy_actions` and `ppo_minibatches`.

This proves the final artifact is restorable and resume-complete. It does not
make the policy reliable.

## Decision and required next task

The engineering run closes the Phase 8 task to execute and analyze the
three-stage curriculum. It does not close joint five-seed training, validation
selection, or the Phase 8 completion gate.

Before spending another four full-seed CPU runs:

1. instrument per-update reward advantages, cost advantages, dual-weighted cost
   advantages, and action probabilities by density;
2. run a bounded hyperparameter experiment that changes only declared
   constraint enforcement, beginning with dual learning-rate/initialization
   scaling and an entropy safeguard;
3. require the bounded run to retain material probability on reliable actions
   and show conditional miss risk moving toward the active budget at all three
   densities;
4. freeze the corrected configuration and code at one commit;
5. discard this pilot from paper checkpoint selection and restart seed 1001
   from transition zero; and
6. launch seeds 1002--1005 only after the fresh seed-1001 validation gate
   passes reliability first and resource cost second.

No `1e-5` experiment is justified by this result.
