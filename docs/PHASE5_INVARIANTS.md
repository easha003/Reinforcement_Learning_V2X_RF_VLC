# Phase 5 runtime invariant checks

## Purpose

The environment fails at the boundary where invalid data first becomes
observable. It does not repair, clamp, wrap, or silently discard invalid
policy outputs or simulator values. This keeps training failures attributable
and prevents corrupted transitions from entering a rollout buffer.

## Finite values and probabilities

The typed frame boundaries enforce the following checks:

| Boundary | Required invariant |
|---|---|
| `FrameObservation` | Actor matrix is `float32`, finite, rank two, and pair-aligned |
| `CriticObservationFrame` | Critic matrix and global summary are finite and shape-aligned |
| `FramePacketOutcomes` | Rewards, sampled costs, and conditional risks are finite and pair-aligned |
| `FrameStepOutput` | Rewards are finite; standard cost/risk info arrays are finite, `float32`, and pair-aligned |
| RF/VLC risk boundaries | Every mechanism and total failure probability lies in `[0, 1]` |

`sampled_miss_cost` must contain only zero or one.
`conditional_miss_probability` must lie in `[0, 1]`. Both arrays are copied
into read-only storage by `FrameStepOutput`, so callers cannot mutate an
already validated transition.

## Selected actions

`FrameAPISchema.validate_actions(observation, actions)` is the mandatory
pre-step action boundary. It requires:

- one-dimensional `int64` actions with exactly one entry per current pair;
- every index in the frozen range `[0, 8]`; and
- every selected index enabled by that pair's action-mask row.

Range validation precedes NumPy mask indexing. Negative indices therefore
cannot wrap around to a valid column, and oversized indices cannot escape as
implementation-level indexing errors. The validated action vector is a
read-only copy.

## Cross-frame lifecycle state machine

`PopulationLifecycleTracker` adds sequence checks that a single frame cannot
establish:

- frame indices advance by exactly one and timestamps increase strictly;
- a live, non-final pair cannot disappear;
- a continuing pair advances by exactly one episode step and keeps the same
  ordered transmitter/receiver identities;
- a pair first observed after the reset frame is marked as a birth;
- a final pair cannot remain active in the next frame; and
- a completed or otherwise departed pair ID cannot reappear in the trace.

The first observed frame is a reset boundary and may contain continuing pairs
when replay begins inside an existing episode. Trace changes require an
explicit tracker reset. Validation is transactional: a rejected frame does not
advance the tracker's accepted state.

`PopulationFrameReader.iter_frames()` runs this tracker before yielding each
frame, while the future vectorized environment can use the same tracker at its
reset/step boundary.

## Failure policy

Violations raise typed environment errors with trace, frame, pair, row, or
action context where applicable. No invariant path clips a probability,
substitutes an action, fabricates a lifecycle flag, or mutates the last valid
state.
