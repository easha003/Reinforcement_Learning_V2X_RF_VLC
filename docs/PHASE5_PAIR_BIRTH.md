# Phase 5 pair-birth semantics

## Boundary

A pair episode enters the environment in one of two ways:

1. **Physical birth during a sampled rollout.** The pair must appear with
   `episode_step == 0` and `lifecycle.born == true`.
2. **First frame after environment reset.** A sampled segment may begin inside
   an already-running physical pair episode. Every pair on that first frame is
   new to the rollout and receives fresh local history even when its physical
   `born` flag is false and its episode step is greater than zero.

After the first sampled frame, the exact set difference between current and
previous stable pair IDs must equal the frame's declared births. An entry
without a birth flag and a repeated birth flag on a continuing pair both fail
before an actor observation is emitted. A retired ID may not reappear inside
the same sampled episode; the trace layer must provide a new pair-episode ID.

## Fresh pair-local state

`Perception.initialize_pair_history()` creates one new `LinkStateTracker` for
each entering pair. It rejects an ID that already owns link history, making
accidental state carry-over an explicit lifecycle error. The first usable raw
actor row therefore contains:

| Field | Birth value |
|---|---:|
| RF quality | `0` (unmeasured sentinel) |
| RF quality age | `-1` |
| RF quality history | eight zeros |
| VLC quality | `0` (unmeasured sentinel) |
| VLC quality age | `-1` |
| VLC quality history | eight zeros |
| Previous action | `-1` |
| Last delivery outcome | `-1` |
| Consecutive miss count | `0` |

These values come from a genuinely empty tracker; they are not copied from a
continuing pair or reconstructed from simulator truth. If either endpoint has
no live causal track, the complete row remains `None`, exactly as for any other
unusable observation. The environment will use the configured fallback and a
zero learning mask rather than fabricate geometry for the newborn.

## State that is not reset

A physical birth resets only state owned by that pair. It does not reset:

- causal vehicle tracks shared by the population;
- link histories of continuing pairs; or
- the delayed mean-field signal produced by the preceding closed frame.

Consequently, a newborn with a usable causal track receives the same valid
two-column delayed population suffix as every continuing actor in that frame.
Resetting that suffix at each birth would discard lawful shared history and
make observations depend on unrelated population churn.

Pair termination/truncation flags, value-bootstrap masks, and post-outcome
state release are handled by the next Phase 5 lifecycle task.
