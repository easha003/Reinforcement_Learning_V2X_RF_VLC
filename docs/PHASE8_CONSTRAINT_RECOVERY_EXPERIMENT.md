# Phase 8 bounded constraint-pressure recovery experiment

Date predeclared: 2026-09-26

## Question and boundary

This experiment asks whether a bounded change in constraint pressure causes a
fresh PPO policy to place materially more probability on actions that satisfy
the active reliability budget. It is a diagnostic hyperparameter experiment,
not a paper result and not a continuation of the failed 10-million-transition
seed-1001 checkpoint. Every arm starts from the same policy seed and transition
zero, uses only training traces for learning, and uses the already declared
bounded validation windows for comparison. The test split remains closed.

No full seed is authorized by this protocol. A full fresh seed-1001 run remains
blocked unless one recovery arm passes the gate below.

## Frozen common protocol

- Policy seed: `1001`
- Training sources: all nine configured training traces at densities 10, 20,
  and 30 vehicles/lane-kilometer
- Transition budget per arm: `300,000`
- Rollout target: `32,768`
- Curriculum: unchanged headline `1e-2 -> 1e-3 -> 1e-4`, with fractions
  `0.10`, `0.20`, and `0.70`
- Actor, critics, PPO optimizer, causal observations, physical environment,
  scheduling, masks, action costs, and all random seeds: unchanged
- Evaluation: deterministic masked argmax on the validation windows frozen by
  the state-regime audit; normalization remains training-only and frozen
- Constraint-pressure evidence: the versioned pre-update v1 diagnostic in
  every iteration report

The common layer is
`configs/training/recovery_bounded_300k.yaml`. A 300,000-transition arm is long
enough to enter the strict `1e-4` curriculum stage and emit several complete
density-balanced PPO updates, while remaining small compared with a
10-million-transition seed.

## Predeclared arms

| Arm | Only change from bounded control | Reason |
|---|---|---|
| `control` | none | Measures the bounded behavior of the failed headline settings with the new diagnostics. |
| `dual_lr_5` | density dual learning rate `0.05 -> 5.0` | Tests the diagnosis that projected ascent was two orders of magnitude too weak over early learning. |
| `dual_init_10` | density dual initial value `0 -> 10` | Tests whether immediate reliability pressure changes the policy before cheap actions dominate. |
| `entropy_005` | entropy coefficient `0.01 -> 0.05` | Tests whether delayed policy concentration is sufficient without changing dual ascent. |

The three treatment arms change one parameter family at a time. They must not
be combined or retuned after inspecting results in this experiment.

## Primary recovery gate

For each of the five causal validation regimes, let `F` be the evaluator's
`policy_induced_load.mean_feasible_action_probability_mass`, and let `R` be its
`mean_policy_expected_conditional_miss_risk`. Campaign aggregates weight these
values by regime row count; overlapping regime membership remains explicit and
is treated identically in every arm.

A treatment arm passes only when all of the following hold relative to the
bounded control:

1. weighted mean feasible action mass increases by at least `0.10` absolute;
2. at least three of five regimes increase feasible action mass by at least
   `0.05` absolute;
3. no regime loses more than `0.02` absolute feasible action mass;
4. weighted policy-expected conditional miss risk is no more than `1.05` times
   control; and
5. training, constraint-pressure diagnostics, checkpoints, and evaluation are
   finite and structurally valid at all three densities.

The first four thresholds are fixed before any arm is run. If multiple arms
pass, choose the largest weighted feasible-mass gain, breaking a tie by lower
weighted expected risk. Deterministic selected-action feasibility and sampled
misses are secondary diagnostics; they do not replace the probability-mass
gate because a bounded argmax can conceal distributional learning.

## Trace and audit compatibility

The immutable traces were originally stamped with the complete headline
configuration. Optimizer-only recovery layers intentionally change that run
hash without changing a mobility byte. Replay therefore checks a fail-closed
`mobility_trace` scope reconstructed from each trace's integrity-protected
archived configuration. The scope retains mobility, geometry, and
`training.root_seed`, while excluding current optimizer fields. New traces
persist this scope directly; existing traces are supported without mutation.

The state-regime audit similarly publishes a `policy_environment` scope. It
retains policy-visible observations and all physical/environment definitions,
but excludes optimizer choices. Recovery arms may reuse the frozen thresholds
and windows only when this scope agrees exactly.

## Interpretation

- A passing dual arm supports insufficient constraint pressure as a causal
  engineering diagnosis and authorizes a fresh full seed-1001 run using that
  single declared change.
- Only the entropy arm passing points instead to premature concentration; the
  full-run candidate is the entropy setting alone.
- No arm passing means the current formulation has not demonstrated recovery.
  Do not combine arms post hoc or restart a full seed; inspect cost-advantage
  signs, observability, and reward/cost scaling and predeclare another bounded
  experiment.
- Passing this gate is evidence of learning movement, not proof that the final
  `1e-4` constrained optimum will be reached. The full run must still pass the
  existing density-specific validation reliability gate.

## Status

Completed on 2026-09-27 under frozen commit `4c8ba75`. Both dual-pressure arms
passed, the entropy-only arm failed, and the predeclared selection rule chose
`dual_init_10`. See `PHASE8_CONSTRAINT_RECOVERY_RESULTS.md` for the immutable
artifact identities, gate calculation, mechanism diagnostics, limitations, and
the next-run decision.
