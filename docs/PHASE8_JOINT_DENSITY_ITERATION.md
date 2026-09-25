# Phase 8 Joint-Density Training Iteration

## Outcome

The repository now has a production-shaped, versioned training entry point that
collects training traces from densities 10, 20, and 30 vehicles per
lane-kilometer into one shared primal-dual PPO update. This replaces the
single-density limitation of the smoke runner while retaining the same
authoritative environment, causal observations, lifecycle bootstraps, and
checkpoint contracts.

The implementation deliberately runs one iteration. It establishes the joint
rollout and update boundary before adding multi-iteration resume, curriculum
progression, validation selection, and five-seed orchestration.

## Density-balanced accumulation

Training sources are grouped by the exact configured density labels. A
collection round selects one training trace per density in ascending density
order. Replicates cycle deterministically by round, using `(round mod available
replicates)` within each density.

The configured `training.rollout_packets` value is an optimized-transition
target, not permission to cut a simultaneous frame or omit a density. Automatic
sizing starts with a three-frame calibration round, estimates aggregate packets
per joint density-frame, and chooses each later round between 3 and 20 frames
from the remaining target. When the target is crossed partway through a round,
the trainer finishes that round. The report records:

- target packet transitions;
- actual accumulated transitions;
- nonnegative round-boundary overshoot;
- per-density transition counts;
- every selected trace, environment seed, frame count, and fingerprint; and
- learning rows, action counts, lifecycle completions, and normalization count.

This guarantees that every dual update has samples from every declared density.
Reliability costs remain partitioned by their exact density label; they are not
pooled into one constraint estimate. Here, “density-balanced” means one selected
trace segment per density in every round; it does not mean equal packet counts.
The physical populations differ by density, and the report preserves those
sample-count differences rather than resampling them away.

## Shared mutable state

All trace segments in the iteration share:

- one decentralized actor;
- one reward critic and one cost critic;
- one training-only observation-normalization stream;
- one set of density-specific dual multipliers;
- one policy-action generator; and
- one PPO-minibatch generator.

GAE is estimated independently inside each trace segment, preventing stable
pair IDs or recursion from crossing trace resets. Prepared segment tensors are
then concatenated into one PPO batch. Cost-penalty weights are evaluated from
the same pre-update dual snapshot, and each represented density is updated once
after optimization.

## Reproducibility and artifacts

`hybrid-v2x-rl training joint-iteration` exposes the driver. By default it uses
the configured 32,768-packet target and adaptive 3-to-20-frame trace segments.
The optional `--rollout-packets` flag provides a bounded packet target;
`--max-frames-per-trace` disables adaptive sizing and fixes every segment to the
declared length. Both choices and every per-segment frame limit are recorded in
the report.

The output directory must be new or empty. A successful iteration publishes:

- `report.json` using schema
  `hybrid-rf-vlc-rl.joint-density-training.v1`;
- `metrics.jsonl` with one versioned training record; and
- `checkpoint-iteration-000001.pt` containing models, optimizers, density
  duals, normalization, counters, global random state, and named generators.

The scope string explicitly states that one iteration is not convergence
evidence.

## Fail-closed conditions

The driver rejects:

- undeclared policy seeds;
- duplicate, non-training, or unconfigured trace IDs;
- missing or unexpected density groups;
- invalid packet targets or frame bounds;
- trace segments with fewer than two acted frames;
- inconsistent merged transition/action counts;
- an iteration that exceeds the configured per-seed transition budget; and
- occupied or symbolic-link output destinations.

## Verification and next boundary

A compact three-trace integration fixture verifies one shared update across all
configured densities, balanced-round overshoot accounting, per-density sample
counts, JSONL publication, complete checkpoint restoration, named RNG streams,
missing-density rejection before publication, and preservation of occupied
output directories.

A bounded run over the repository's configured training traces also completed
with the headline 32,768-packet target. Adaptive limits were `3 → 20 → 3`
frames per selected trace, producing 35,063 rollout transitions, 30,586 PPO
learning rows, and a 2,295-packet (7.0%) round-boundary overshoot. Per-density
constraint samples were 3,582, 11,272, and 20,209 for densities 10, 20, and 30.
The published checkpoint restored with one completed iteration, 39,598 acted
environment transitions, 36 completed pair episodes, and 300 optimizer steps.
Focused lint and static typing pass, and the complete repository regression
suite passes 1,379 tests with 2 expected skips.

The next task is a resumable multi-iteration driver. It must restore the
checkpoint, continue deterministic trace scheduling and random streams, enforce
the configured total transition budget, advance the reliability curriculum at
declared boundaries, and append metrics without rewriting prior evidence.
