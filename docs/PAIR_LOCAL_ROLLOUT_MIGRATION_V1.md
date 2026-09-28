# Pair-local RF rollout migration v1

Status: implemented and validated

Date completed: 2026-09-27

## Purpose

This migration removes the frame-global RF-pool response from every live
packet-risk consumer. One authoritative evaluator now maps a population frame,
its selected joint action, and action-independent propagation rows to complete
pair-local RF physics. Packet outcomes, delayed feedback, analytical
counterfactuals, and feasibility evaluation no longer construct competing RF
responses.

The legacy `mean_field/rf_pool.py` contract remains available for historical
tests and archived Phase 4/Phase 6 analyses. It is not imported by the live
rollout, packet-outcome, actor-feedback, baseline, state-regime, or joint-risk
paths.

## Authoritative frame path

For each chronological population frame, the environment performs exactly
this sequence:

```text
PopulationFrame
  -> FrameLocalRFContext(topology, geometric sensing)
  -> selected FrameActionLedger
  -> FrameLocalRFLoads
  -> FrameLocalRFSensedLoads
  -> FrameLocalRFResponses
  -> FrameEndpointRFSchedule
  -> FrameLocalRFAttemptRisks
  -> FramePacketOutcomes
```

`LocalRFPhysicsModel.context_for` constructs the action-independent context
once. `LocalRFPhysicsModel.evaluate` then projects a realized or counterfactual
ledger through every action-dependent boundary and returns one
`FrameLocalRFPhysics`. Every component must identify the same trace, frame
index, frame time, and canonical pair population. Propagation truth must cover
the frame population exactly; the attempt-risk subset is derived from the
selected RF-using rows rather than supplied separately.

The model is configuration-bound to:

- the declared collision parameters and sensitivity band;
- attempt airtime and generation period;
- the 200 m contention radius;
- the canonical Manhattan building geometry; and
- the RF antenna height.

Realized rollout and counterfactual evaluators receive the same model and
frame context. A distant reservation therefore cannot affect a focal response
in one consumer while affecting it in another.

## Outcome and randomness boundary

Packet-outcome assembly receives the selected ledger and its retained
`FrameLocalRFPhysics`. It obtains collision, receiver half-duplex, and
propagation risk directly from the pair-local attempt-risk row. It cannot
accept a second response or caller-computed risk mapping.

The matched-tape schema remains `hybrid-rf-vlc-rl.matched-packet-tape.v1`.
Analytical assembly consumes no random draw. Outcome sampling still uses the
four pre-addressed RF draws and one VLC draw, preserving the RF-n/DUP-n prefix
rule and matched comparison identity.

## Delayed actor feedback

The actor schema remains 37 columns. Current pair-local CBR, collision,
topology, and endpoint activity remain simulator diagnostics and do not enter
the actor tensor.

The existing one-frame-delayed aggregate is retained for compatibility:

```text
mean_rf_attempt_fraction_t
  = sum_i focal_reserved_attempts_i,t / (4 * active_pairs_t)
```

It is now derived from the complete pair-local response set, whose focal
reservation counts reconcile with the selected ledger. Missing history and a
measured zero-load frame remain distinct through the validity flag. Empty
frames produce a valid zero for the following frame.

## Baselines and audit counterfactuals

Fixed and actor-context baselines continue to select actions at the common
policy boundary; their realized packets automatically use the pair-local
physics. Analytical allocation and truth-risk comparators now evaluate every
provisional joint ledger through `LocalRFPhysicsModel` rather than a global
offered-load scalar. State-regime counterfactuals hold the exact other-pair
actions fixed, vary the focal action, and evaluate the same pair-local path.

The analytical allocation comparator is a centralized model-based comparator:
it uses the model-side current population topology and declared building map,
but not current RF/VLC propagation truth unless it is the explicitly
non-deployable truth-risk oracle. This access does not change the decentralized
actor tensor. Any deployment claim for that comparator must state the assumed
availability of current transmitter positions and the map.

## Certificate-aware joint feasibility

Overlapping local domains remove the scalar-load separability used by the
former exact global-pool oracle. The replacement evaluates complete joint
assignments through the authoritative pair-local pipeline:

- if the full allowed assignment space is no larger than the declared cap,
  every assignment is enumerated and optimality is proven;
- otherwise deterministic multi-start simultaneous best-response search
  returns a realizable candidate; and
- an optimistic zero-contention, zero-half-duplex relaxation supplies a valid
  certified lower bound.

The persisted result reports assignment-space size, assignments evaluated,
search starts and iterations, exactness, the lower bound, and the absolute
optimality gap. Its scientific interpretations are deliberately asymmetric:

| Evidence | Valid conclusion |
|---|---|
| Realizable candidate mean risk is at or below `1e-4` | Feasibility is proven for that evaluated configuration |
| Certified lower bound exceeds `1e-4` | Infeasibility is proven |
| Exhaustive exact optimum exceeds `1e-4` | Infeasibility is proven |
| Candidate fails, lower bound passes, and search is not exact | Inconclusive optimality gap |

A failed non-exact candidate is never reported as proof of infeasibility.

## Metrics and compatibility

The rollout report retains the historical field names
`pool_utilization_sum` and `max_pool_utilization` to avoid an unrelated
artifact-schema migration. Their v1 pair-local meaning is now:

- the sum across frames of mean focal-domain utilization; and
- the maximum focal-domain utilization observed across all frames.

They are not global-pool utilization. New evidence and paper text must use the
pair-local interpretation.

## Validation evidence

The migration is covered by:

- direct pipeline identity, conservation, empty-frame, spatial-reuse, and
  fail-closed tests;
- packet sampling and conditional-risk reconciliation tests;
- one-frame-delay, reset, and trace/frame alignment tests;
- actor lifecycle and final-observation tests;
- pair-local exact-search comparison against independent exhaustive
  enumeration;
- bounded-search candidate and lower-bound certificate tests; and
- evaluation-artifact tests for feasible, infeasible, and inconclusive
  verdicts.

The next task is the predeclared bounded system-feasibility frontier. It must
vary only named physical-system factors, retain validation/test separation,
and preserve the candidate/lower-bound gap whenever exhaustive optimality is
not available. PPO training remains paused until that feasibility gate closes.

Pre-migration PPO checkpoints must not be resumed for optimization because
their transition dynamics were generated by the superseded global-pool
environment. The bounded feasibility evaluator may reuse a frozen checkpoint's
actor-normalization statistics only where the actor itself is explicitly not
used; that limited reuse must be recorded in the evaluation artifact. Any
post-gate PPO campaign starts from fresh model, optimizer, dual, and rollout
state under this pair-local pipeline.

Final repository validation for this migration passed Ruff, strict mypy across
125 source files, 103 directly affected/integration tests, and the complete
unit suite with 1,534 passes plus 2 expected artifact-dependent skips.
