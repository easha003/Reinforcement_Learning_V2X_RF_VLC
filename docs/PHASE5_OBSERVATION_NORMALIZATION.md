# Phase 5 observation normalization

Date: 2026-09-21

## Contract implemented

`ObservationNormalizer` owns the running state used by the 37-column actor
input. It implements the environment-contract sequence at the decision-frame
boundary:

1. freeze the historical count, mean, and second central moment;
2. transform every valid causal row in the frame with that same snapshot;
3. expose the normalized immutable `FrameObservation` for joint action
   selection;
4. validate the selected action vector and the fallback action on unavailable
   rows;
5. batch-update Welford state once, using every and only valid raw row from a
   training frame.

This ordering prevents an earlier row in the same population from changing the
normalization applied to a later actor. It also prevents current population
statistics from entering any current action.

For a column with at least two historical training rows, the transformation is

```text
(x - historical_mean) / sqrt(historical_sample_variance + epsilon)
```

and is clipped to the configured symmetric bound. With fewer than two
historical rows, mean zero and variance one are used. Sample variance is the
Welford second central moment divided by `count - 1`.

## Standardized and pass-through columns

The standardizer derives the exact actor column order from the resolved
configuration. It passes these encoded columns through unchanged:

- `path_spans_junction`
- `previous_action`
- `last_delivery_outcome`
- `mean_field_valid`

All other actor columns are standardized. Finite missing sentinels such as the
`-1` unmeasured age and zero-padded quality histories are included as ordinary
training samples; they are not filtered or imputed.

An unavailable causal row does not have raw features to normalize. It receives
an internal zero placeholder solely to preserve the rectangular frame API,
retains `learn_mask = false`, must select the configured `DUP-4` fallback, and
does not update Welford state.

## Split isolation and lifecycle

New state begins in training mode:

- only a trace whose immutable `FrameTraceSource.split` is `train` may be
  processed;
- validation or test input before `freeze()` is rejected;
- `freeze()` is allowed only between decision frames and is irreversible for
  that instance;
- a frozen instance transforms train, validation, or test rows but never
  updates its statistics.

The split is therefore taken from trace identity rather than a caller-provided
boolean. Validation and test cannot silently mutate training state.

## Checkpoint state

`state_dict()` persists a versioned JSON-safe payload containing:

- contract version and the exact 37 column names;
- the standardized/pass-through mask;
- configured epsilon and clipping bound;
- frozen/training mode;
- per-column count, mean, and Welford second central moment.

`from_state_dict()` fails closed if schema, columns, mask, contract version, or
configuration constants drift. Pass-through columns use canonical zero
statistics. A checkpoint cannot be emitted or frozen while a frame is waiting
for action selection.

The normalizer state is independent of PPO epochs. The environment stores the
already normalized actor tensor used for action selection; replaying a PPO
minibatch must not invoke the normalizer or update it again.

## Integration

The deterministic Phase 5 rollout now routes every causal actor frame through
this workflow. Its content fingerprint includes normalized actor rows, the
pre-frame counts, and the final checkpoint state. For a training-trace run,
`normalization_training_rows` must exactly equal `usable_transitions`.

The existing `CentralizedCriticBuilder` consumes the resulting normalized
actor-facing frame. Consequently, its 37-column actor prefix and population
mean are computed from the exact frozen normalized rows used to select actions;
raw observations never enter the critic tensor.

## Mechanical checks

`tests/unit/test_observation_normalization.py` verifies:

- the exact four pass-through columns;
- cold-start mean/variance behavior;
- one frozen historical snapshot for an entire population frame;
- batch Welford count, mean, and second-moment reconciliation;
- inclusion of missing sentinels and history padding;
- fallback-only closure and exclusion of unavailable rows from statistics;
- rejection of unfrozen validation/test input;
- immutable statistics throughout frozen validation and test frames;
- JSON checkpoint round-trip and fail-closed metadata/corruption checks;
- no checkpoint/freeze operation while a frame is open;
- normalized actor rows feeding the existing 78-column centralized critic;
- empty-frame shapes without fabricated statistics.

The real-campaign replay results and fingerprints are recorded in
`PHASE5_DETERMINISTIC_ROLLOUTS.md`.
