# Phase 8 Internal-Truncation Bootstrap

## Outcome

The authoritative trace rollout and PPO smoke trainer now support
bootstrap-valid `max_duration` boundaries. The implementation supplies a real
next-physical observation under the ending episode's stable pair ID; it does
not substitute the reset population, reuse an array position, or force the
next value to zero.

This closes the lifecycle prerequisite for the joint-density trainer. It does
not itself implement multi-trace rollout accumulation or long training.

## Causal sequence

For a pair whose current packet ends with `truncated=True` and
`bootstrap_valid=True`, one frame closes in this order:

1. The current joint action is evaluated and the packet outcome is fixed.
2. Action-dependent receiver feedback is recorded in the ending pair's local
   history.
3. The audited current RF demand is queued as the one-frame-delayed congestion
   signal.
4. The next physical mobility frame is observed using the ending pair ID,
   endpoint records from that next frame, and the still-live old history.
5. That critic-only actor row is normalized with the state that already
   includes the current acted frame. The transform is read-only and adds no
   normalization sample.
6. The old pair history is released.
7. The ordinary next population begins normally. New pair IDs receive fresh
   local history, while every actor sees the same delayed congestion signal.

The rollout uses one-frame lookahead so a diagnostic `max_frames` cutoff does
not prevent materialization when the required physical frame exists in the
trace. The lookahead frame is not acted or counted unless it is inside the
requested rollout.

## Critic and return semantics

Each final observation is a one-row `FrameObservation` keyed by its old stable
pair ID. During rollout preparation, its 37 normalized actor columns are
combined with the 41-column global suffix from the ordinary next population.
Both reward and cost critics evaluate that 78-column row.

The lifecycle adapter then applies the two distinct masks already defined by
the environment contract:

- one-step value bootstrap: enabled for the bootstrap-valid truncation;
- recursive GAE continuation: disabled because the episode resets.

Natural terminations and physical `trace_end` truncations still receive exact
zero next values. Ordinary continuing pairs still resolve their next critic
values by stable ID from the next population.

## Fail-closed conditions

The rollout rejects the boundary when any of the following is true:

- the immediate next physical frame is absent or has the wrong identity;
- an ending pair endpoint is absent from that frame;
- the final causal observation is unavailable;
- final-observation keys differ from bootstrap-valid pair IDs;
- the final row and ordinary next population do not describe the same physical
  frame;
- a centralized next-population summary cannot be formed.

These checks prevent reset rows, newborn identities, stale feedback, or zero
placeholders from silently entering critic targets.

## Verification

Focused verification covers:

- non-consuming preview of queued next-frame congestion feedback;
- feedback-before-observation and observation-before-release ordering;
- immediate-frame and endpoint identity failures;
- read-only final-row normalization;
- numeric final observations at the shared rollout observer boundary;
- an end-to-end PPO update containing one internal truncation;
- checkpoint restoration after that update;
- lint and static typing for every changed source module.

The focused regression suite passes 64 tests. The complete repository suite
passes 1,376 tests with 2 expected skips. The next Phase 8 task is the
versioned joint-density trainer and rollout accumulator for densities 10, 20,
and 30 vehicles per lane-kilometer.
