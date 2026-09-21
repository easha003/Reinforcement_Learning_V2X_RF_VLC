# Phase 5 return-estimation boundaries

## Pair-aligned semantics

The final packet remains a normal transition: its selected action is charged,
its reward and reliability cost are recorded, and its causal outcome feedback
is processed. Lifecycle flags change only what happens after that packet.

| Row type | `terminated` | `truncated` | `bootstrap_valid` | value bootstrap | recursive GAE continuation |
|---|---:|---:|---:|---:|---:|
| Continuing pair | 0 | 0 | 0 | next regular observation | 1 |
| Natural pair end | 1 | 0 | 0 | zero | 0 |
| Internal 60 s boundary with a next trace row | 0 | 1 | 1 | `final_observation` | 0 |
| Trace end, or internal boundary with no next trace row | 0 | 1 | 0 | zero | 0 |

`FrameReturnBoundary` materializes two different masks because a valid
one-step value bootstrap does not permit the advantage recursion to cross a
reset:

```text
value_bootstrap_mask = (~terminated & ~truncated) | bootstrap_valid
gae_continuation_mask = ~(terminated | truncated)
```

Phase 7 will apply these masks in the actual GAE calculation. Phase 5 owns the
lifecycle truth and prevents the later trainer from reconstructing it from an
ambiguous combined `done` value.

## Final observations

An internal truncation with `bootstrap_valid = true` requires one
`final_observation` keyed by its stable pair ID. It is the next physical trace
observation used only to evaluate the boundary value; it is not returned as a
normal next transition and is never allowed to connect the rollout to the
freshly reset episode. Natural endings and physical trace endings reject a
final observation because their value target is exactly zero.

The return boundary and `FrameStepOutput` both fail closed unless the final
observation mapping covers the bootstrap-valid IDs exactly. This preserves the
distinction between “the trace has a next row” and “the next sampled episode
starts after reset.”

## Learning and state release

`learn_mask` is copied from causal actor-row usability. An unusable current
row still executes the configured fallback and produces outcomes, but it does
not enter the policy loss. This mask is independent of lifecycle: a usable
final packet remains learnable.

Pair-local perception history is released only in `close_frame()`, after every
active pair has supplied outcome feedback and the shared RF response has been
closed. A finalized stable ID is then forbidden from appearing in the next
frame. A later physical episode must receive a new stable episode identity.
Reset clears this guard along with pair-local and delayed population state.

## Validation coverage

The unit suite verifies:

- identity alignment between actor rows and action-ledger lifecycle rows;
- all four lifecycle cases in the table above;
- exact final-observation coverage for bootstrap-valid truncations;
- immutable lifecycle and learning masks;
- feedback-before-release ordering; and
- rejection of a finalized ID that remains active after its last packet.
