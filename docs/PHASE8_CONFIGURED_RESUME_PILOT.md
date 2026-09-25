# Phase 8 Configured-Trace Resume Pilot

## Resolution (2026-09-25)

The temporal-coverage defect documented below is now closed. Deterministic,
checkpoint-resumable within-trace window progression passed unit, integration,
and configured nine-trace audits, including byte-identical split and
uninterrupted checkpoints. See `PHASE8_TRACE_WINDOW_PROGRESSION.md` for the
versioned schedule, reset semantics, coverage evidence, and artifact hashes.

The original v1 pilot remains a historical record and cannot be resumed by the
v2 trainer because it lacks trace-window metadata. Its findings below are
retained to show why the new boundary was required.

## Verdict

The configured-trace checkpoint/resume mechanism passed its operational audit,
but the full 10-million-transition campaign is **not ready to launch**. The
trainer restores and advances learning state correctly; however, every bounded
segment currently starts at frame zero. The pilot therefore exposed a temporal
coverage defect that would repeatedly train on only the first 20 of each
9,000-frame mobility trace.

The next implementation boundary is deterministic, checkpoint-resumable
within-trace window progression. This must be complete before any pilot output
is treated as learning evidence or a full policy-seed campaign is started.

## Pilot identity

- Date: 2026-09-24
- Repository commit: `722f30312b9b62af382259b9a24ef962e82bc786`
- Configuration hash:
  `69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`
- Policy seed: `1001`
- Training sources: all nine configured training traces, comprising three
  replicates at each of densities 10, 20, and 30 vehicles/lane-kilometer
- Rollout target: 32,768 packet transitions per update
- Frame scheduling: adaptive `3 → 20 → 3` balanced rounds
- Invocation limit: one update in the fresh process and one update in the
  resumed process
- Local ignored artifact root:
  `artifacts/logs/phase8-resume-pilot-seed1001`

Fresh invocation:

```bash
hybrid-v2x-rl training joint-train \
  --output artifacts/logs/phase8-resume-pilot-seed1001 \
  --policy-seed 1001 \
  --max-iterations 1
```

Resume invocation:

```bash
hybrid-v2x-rl training joint-train \
  --output artifacts/logs/phase8-resume-pilot-seed1001 \
  --policy-seed 1001 \
  --resume-checkpoint \
    artifacts/logs/phase8-resume-pilot-seed1001/checkpoints/checkpoint-iteration-000001.pt \
  --expected-checkpoint-sha256 \
    5815ccfcf900028459c131fa9a484db2bfcc38475e96967a0741a88f20a2a8aa \
  --max-iterations 1
```

## Training results

| Quantity | Iteration 0 | Iteration 1 | Cumulative after resume |
|---|---:|---:|---:|
| Environment transitions | 39,598 | 39,498 | 79,096 |
| Rollout transitions | 35,063 | 34,955 | 70,018 |
| Learning rows | 30,586 | 30,478 | 61,064 |
| Completed pair episodes | 36 | 43 | 79 |
| Optimizer steps | 300 | 300 | 600 |
| Balanced rounds | 3 | 3 | 6 |
| Rollout-target overshoot | 2,295 | 2,187 | 4,482 |

Both updates remained in curriculum stage 0, whose interval is
`[0, 1,000,000)` acted transitions and whose miss budget is `1e-2`. Neither
update was limited by the total transition budget.

Per-density rollout samples were:

| Density | Iteration 0 | Iteration 1 |
|---:|---:|---:|
| 10 | 3,582 | 3,657 |
| 20 | 11,272 | 11,294 |
| 30 | 20,209 | 20,004 |

The density duals continued rather than restarting:

| Density | After iteration 0 | Before iteration 1 | After iteration 1 | Update count |
|---:|---:|---:|---:|---:|
| 10 | 0.0024073602 | 0.0024073602 | 0.0052499256 | 2 |
| 20 | 0.0037326663 | 0.0037326663 | 0.0073260931 | 2 |
| 30 | 0.0058761529 | 0.0058761529 | 0.0103392245 | 2 |

## Resume and artifact audit

The resume invocation accepted the independently supplied first-checkpoint
digest. Acceptance verifies exact configuration identity, policy seed,
checkpoint filename/counter agreement, contiguous prior metrics, complete prior
checkpoint/report sequences, and the required named random streams.

After iteration 1, the second checkpoint restored successfully with:

- two completed iterations;
- 79,096 environment transitions and 61,064 learning rows;
- 79 completed episodes and 600 optimizer steps;
- 70,142 observation-normalization training rows;
- dual update counts `(2, 2, 2)`;
- populated actor, reward-critic, and cost-critic optimizer states;
- NumPy generator `training_streams`; and
- PyTorch generators `policy_actions` and `ppo_minibatches`.

The first checkpoint, first iteration report, and first session report retained
their exact pre-resume hashes. The first JSONL record after resume also retained
the original whole-file digest from the one-record log, proving append-only
publication rather than record rewriting.

| Artifact | SHA-256 |
|---|---|
| Checkpoint 1 | `5815ccfcf900028459c131fa9a484db2bfcc38475e96967a0741a88f20a2a8aa` |
| Iteration report 1 | `11a2a12fe5fe0f57037f693f25160901cf626de818eaf72f9a8bfa3245e5c112` |
| Session report 1 | `861ba37273fd619a688e801f7704941748d883887e3b3f04894867ad46f9d049` |
| Original metric row | `8fd1c1edb0cc4ae179fea9450b91b72819edb1080580fd5774f34ad74af06881` |
| Checkpoint 2 | `20f076c34c29987da6ae504ba27abc8317546b76a009452cd68fa156b38fdd22` |
| Iteration report 2 | `b9bd5e32c32a372e038100961dc64ef576c1622e83e9d0bfc5a8461d59188390` |
| Session report 2 | `b0d91e56262b235f4c1d299ad4392ffb41c80dedae874aaf0076e7bae1456124` |
| Two-record metric log | `33dabca7809a4cdff27aa55d9efb06a7c433a49fd841989f0908ee9e91fb720f` |

The final artifact tree contains seven regular files, no symbolic links, a
newline-terminated two-record metric log, and occupies approximately 756 KiB.
Metric iterations are exactly `(0, 1)`, cumulative environment transitions are
strictly increasing `(39,598, 79,096)`, and both rows retain one configuration
hash and policy seed.

## Trace scheduling evidence

Replicate scheduling continued deterministically across the checkpoint. The
three density-balanced rounds used these replicate orders:

```text
iteration 0: 000 → 001 → 002
iteration 1: 001 → 002 → 000
```

The nine restored environment seeds differ from the first invocation and match
the continued named NumPy stream. All segment and matched-tape fingerprints are
present in the immutable iteration reports.

## Blocking temporal-coverage finding

Every configured training trace contains 9,000 decision frames and between
9,808 and 25,824 decision-pair episodes. The rollout API currently accepts only
a maximum frame count, not a starting frame or a persisted segment cursor.
Consequently, every selected segment begins at frame 0. The pilot's largest
segment reached frame 19:

```text
maximum distinct temporal prefix = 20 / 9,000 frames = 0.2222%
```

Rotating among mobility replicates does not solve this problem: after every
replicate has occupied the 20-frame round once, subsequent iterations revisit
the same prefix with new channel/action randomness. A 10-million-transition run
would therefore accumulate many samples without covering the remaining 99.78%
of each trace's time axis. This is unsuitable for a paper-quality training
campaign and contradicts the environment contract's provision for resets that
sample later segments of the same trace.

## Required next task

Implement a versioned trace-window schedule that:

1. selects later decision-frame windows without leaking pre-window observations
   into actor state;
2. defines fresh pair histories and mean-field state correctly for pairs active
   at a sampled window boundary;
3. persists or deterministically reconstructs every per-trace window position
   across checkpoints;
4. makes exact pre-round budget projection window-aware;
5. records start/end frame indices and wrap events in iteration reports; and
6. proves split-versus-uninterrupted equivalence plus broad 9,000-frame coverage
   before the full seed-1001 run begins.

The two pilot checkpoints remain useful engineering evidence for resume
correctness, but they are not convergence, policy-quality, or trace-coverage
evidence.
