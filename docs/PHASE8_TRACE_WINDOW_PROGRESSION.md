# Phase 8 Deterministic Trace-Window Progression

## Outcome

The frame-zero replay defect found by the first configured resume pilot is
closed. Joint-density PPO now selects deterministic bounded windows throughout
each immutable mobility trace, reconstructs the same selections after a
checkpoint restore, and records every selection in the iteration artifact.

The implementation passed a split-versus-uninterrupted configured-trace audit:
two updates run as separate processes produced byte-identical metrics and
byte-identical checkpoints to the same two updates run in one process. The full
10-million-transition seed-1001 campaign is no longer blocked by temporal
window progression. The pilot remains engineering evidence, not convergence or
policy-quality evidence.

## Versioned schedule

The schedule schema is
`hybrid-rf-vlc-rl.trace-window-schedule.v1`. For completed iteration `i` and
local density-balanced round `r`, it computes:

```text
position = 997 i + 4099 r
window_frames = min(requested_frames, available_frames)
span = available_frames - window_frames + 1
start = position mod span
end = start + window_frames - 1
cycle = floor(position / span)
```

The iteration counter is already checkpointed, and the round index is local to
an atomic update. The next window therefore requires no independent mutable
cursor. The fixed iteration stride is coprime with the valid start spans for
the headline 9,000-frame traces and 3- and 20-frame windows. Even after the
three-replicate source rotation is included, the round-zero subsequence for
each replicate is coprime with those spans and visits every valid start before
repeating.

Each segment report stores the schema, available and requested frame counts,
bounded window size, physical start/end frame indices, unbounded schedule
position, cycle, and wrap flag. Joint-density iteration reports advance from
schema v1 to v2. A resume intentionally rejects v1 reports because their frame
position cannot be reconstructed safely under the new contract.

## Reset and lifecycle semantics

A nonzero window start is a new sampled environment episode, not a rewrite of
the underlying physical pair episode:

- pair `episode_step`, `born`, and physical termination/truncation metadata
  retain their trace meaning;
- all pairs present on the first sampled frame receive fresh causal observation
  histories, even when their physical `born` flag is false;
- delayed mean-field state and normalization flow begin at the sampled reset,
  so no observation from before the window is exposed;
- matched random tapes remain addressed by physical trace, pair, and episode
  step; and
- one look-ahead frame is still materialized when needed for a valid internal
  truncation bootstrap, without converting a window cutoff into a physical
  lifecycle event.

Parquet vehicle replay now accepts inclusive time bounds and uses row-group
statistics plus Arrow filtering, so a late window does not decode every earlier
vehicle row. Population replay initializes the physically active pair set at
the requested frame and emits only the bounded physical-frame interval.

## Budget accounting

Before a balanced round starts, the trainer intersects every immutable pair
episode with that round's selected `[start, end]` interval. The resulting acted
transition total is the exact budget projection for the window. Seeds are not
drawn and environments are not mutated when the projected round would cross
the configured per-seed ceiling. The collected total is checked against the
projection after every round.

## Configured pilot

The audit used all nine configured training traces, policy seed `1001`, the
headline rollout target of 32,768 packet decisions, and one fresh update
followed by one resumed update. The split run is stored locally under:

```text
artifacts/logs/phase8-window-resume-pilot-seed1001
```

The matching uninterrupted reference is stored under:

```text
artifacts/logs/phase8-window-uninterrupted-pilot-seed1001
```

Both roots are ignored operational artifacts.

| Quantity | Iteration 0 | Iteration 1 | Cumulative |
|---|---:|---:|---:|
| Environment transitions | 42,668 | 37,833 | 80,501 |
| Rollout transitions | 39,272 | 33,862 | 73,134 |
| Learning rows | 37,790 | 33,862 | 71,652 |
| Completed pair episodes | 105 | 66 | 171 |
| Optimizer steps | 370 | 340 | 710 |
| Balanced rounds | 2 | 2 | 4 |

The first update selected frames `0–2` and `4099–4118`. After checkpoint
restore, the second selected `997–999` and `5096–5111`; its adaptive second
window requested 16 frames after observing the first round's packet density.
The selected corpus windows therefore reached frame 5,111, or 56.8% through
the 9,000-frame time axis, in two updates instead of repeatedly visiting only
frames 0–19. Schedule tests prove complete start-position traversal before
repetition for each rotating replicate.

## Resume-equivalence and artifact audit

The split and uninterrupted runs produced:

- byte-identical `metrics.jsonl` files;
- identical iteration payloads after removing only the output-root-specific
  checkpoint path;
- identical first-checkpoint SHA-256
  `81e1a2aec0562adfb023d6fa20c4ee7857594f25a94d4344af590e8860d1a9b4`;
- identical final-checkpoint SHA-256
  `5fe248a0473240798e0b397242627587ace76f27dbd732b9857fa41676b96fc1`;
  and
- unchanged checkpoint-1, iteration-1, and session-1 hashes after the resumed
  process appended its second update.

The resumed run's iteration-report digests are:

| Artifact | SHA-256 |
|---|---|
| Iteration 1 | `1c4a837a4a37b42f9074df3316df3711739920fe7065ba6cf8c11bd5ab6ca06c` |
| Iteration 2 | `92512091be5d7516c5b0789f54c60933e8e023d8a3594354a6c7bff1f503cf49` |
| Two-row metric log | `b51bbfefa67ef026038d3fa668ac7bd51921ee32db7b8bddebad825555488067` |

## Verification boundary

Tests cover inclusive Parquet time filtering, nonzero population-frame starts,
continuing physical pairs at a fresh sampled reset, deterministic rollout
fingerprints and physical coordinates, schedule bounds/wraps/full traversal,
window-aware budget accounting, v1 resume rejection, and exact
split-versus-uninterrupted training equivalence. Static typing and focused lint
also pass.

The next Phase 8 task is to begin the full joint-density seed-1001 campaign,
monitor stability through the reliability curriculum, and then apply the same
frozen procedure to the remaining policy seeds.
