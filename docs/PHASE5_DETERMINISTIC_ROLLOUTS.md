# Phase 5 deterministic rollout validation

Date: 2026-09-21

## Purpose

The deterministic rollout harness composes the implemented Phase 5 boundaries
before a trainable PPO policy is introduced. It is a validation runner, not a
second simulator and not yet the final training environment.

For every global decision frame it executes this order:

1. stream the immutable population frame and lifecycle metadata;
2. freeze causal raw observations from current trace state and past feedback;
3. normalize every valid row with the pre-frame training-state snapshot;
4. select and validate a complete masked joint action;
5. batch-update Welford state once from the valid training rows;
6. audit per-pair and population resource accounting;
7. calculate shared RF-pool demand and collision risk;
8. advance policy-independent RF/VLC physical state once per pair;
9. apply identity-addressed matched random tapes to selected actions;
10. assemble reward, sampled miss cost, and conditional miss risk;
11. validate termination, truncation, bootstrap, and learning masks;
12. record bounded action-dependent feedback, close the frame, and release
    final pair state.

The physical path uses the same empty building set as the inherited synthetic
campaign evaluation. Vehicle occlusion and all other configured RF/VLC channel
effects remain active. Adding trace-backed building geometry is outside this
validation task.

## Policies

- `random`: selects uniformly from hardware-allowed actions with a generator
  addressed by policy seed, trace ID, stable pair-episode ID, and packet index.
- `cycle`: selects the allowed action at `episode_step mod action_count`.
- A fixed contract action such as `VLC`, `RF-2`, or `DUP-4` is also accepted.

Policy randomness is separate from environment randomness. Unavailable causal
observations are never shown to a policy: the configured `DUP-4` fallback is
applied, and the transition's learning mask is false. The normalizer uses an
internal finite zero placeholder only to preserve the rectangular
`FrameObservation`; that placeholder is not policy input, does not update
statistics, and is not training data. Every valid row is normalized before
action selection with statistics frozen at the start of its frame.

An internal max-duration truncation receives an immutable bootstrap-observation
reference so exact final-observation coverage is checked. This harness does not
claim that reference is the normalized numeric tensor required by a value
network; the trainable environment must materialize that tensor before value
inference.

## Reproducibility check

Run from the repository root:

```bash
.venv/bin/python scripts/run_deterministic_rollouts.py \
  --trace synthetic-d10-train-000 \
  --frames 100 \
  --out artifacts/logs/phase5_deterministic_rollouts.json
```

The default policy seed is `7001`; the environment seed defaults to the frozen
headline configuration. The script runs each requested policy twice unless
`--no-replay-check` is supplied. It compares the complete immutable report,
including a SHA-256 digest streamed from actor rows, actions, shared-pool state,
outcomes, lifecycle masks, pre-frame normalization counts, and the final
normalization checkpoint state. `--frames` is a diagnostic processing cutoff
and does not manufacture truncation flags for pairs still active at the
cutoff. Use `--frames 0` to replay the complete trace.

The generated JSON report is reproducible evidence under `artifacts/logs/` and
is intentionally ignored by Git.

## Real-campaign result

The command above evaluated the first 100 frames (10 simulated seconds) of
`synthetic-d10-train-000`. Each policy processed 18,810 pair transitions and
was then repeated from a fresh state, for 75,240 transition evaluations across
the four passes.

| Metric | Random | Cycle |
|---|---:|---:|
| Frames | 100 | 100 |
| Pair transitions | 18,810 | 18,810 |
| Learning-usable transitions | 18,669 | 18,669 |
| Missing-observation fallbacks | 141 | 141 |
| Rows incorporated into normalization | 18,669 | 18,669 |
| Pair births | 283 | 283 |
| Natural terminations | 60 | 60 |
| Sampled misses | 1,385 | 1,415 |
| Reserved RF attempts | 42,173 | 41,715 |
| VLC activations | 10,561 | 10,381 |
| Maximum active population | 232 | 232 |
| Maximum RF-pool utilization | 1.410 | 1.715 |
| Replay verified | yes | yes |
| SHA-256 fingerprint | `d8967d35cd6e55614c9785a3f5332653b769657150242e2878c0637481fe10f1` | `faa2512c18d82ce2594e3bf1d46d75aa9c8f94c48c778c31c7443a00b8572d5d` |

No future-leakage boundary, invalid/masked action, non-finite value, invalid
probability, stable-identity alignment, feedback timing, accounting,
random-tape, or lifecycle invariant failed. The different policy fingerprints,
resource loads, and outcomes also confirm that the harness is exercising the
action-coupled system rather than replaying a fixed outcome table.

## Automated checks

`tests/unit/test_deterministic_rollout.py` builds a small immutable trace and
checks:

- exact report equality under identical environment and policy seeds;
- fingerprint separation when the random-policy seed changes;
- one normalization update for every and only learning-usable transition;
- full-trace natural, internal-truncation, and trace-end lifecycle accounting;
- cutoff semantics without fabricated truncations;
- random, cycle, and fixed-action policy-name validation.

The existing legacy rollout tests remain unchanged in behavior. The extracted
action-independent channel evaluation shares one correlated-state advance with
the legacy action evaluator, which continues to use its complete hopped fading
tape.

The dedicated normalization suite additionally verifies frame-frozen Welford
updates, encoded-column pass-through, sentinel/history-padding inclusion,
checkpoint round-trip, and immutable validation/test state. See
`PHASE5_OBSERVATION_NORMALIZATION.md`.

## Phase 5 completion

The long-rollout completion gate and the stateful, checkpointable,
split-isolated normalization deliverable have both passed. Phase 5 is complete;
the next work-plan gate is the Phase 6 baseline and oracle study.
