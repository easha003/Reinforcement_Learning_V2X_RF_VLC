# Phase 5 deterministic rollout validation

Date: 2026-09-21

## Purpose

The deterministic rollout harness composes the implemented Phase 5 boundaries
before a trainable PPO policy is introduced. It is a validation runner, not a
second simulator and not yet the final training environment.

For every global decision frame it executes this order:

1. stream the immutable population frame and lifecycle metadata;
2. freeze causal actor observations from current trace state and past feedback;
3. select a complete masked joint action;
4. audit per-pair and population resource accounting;
5. calculate shared RF-pool demand and collision risk;
6. advance policy-independent RF/VLC physical state once per pair;
7. apply identity-addressed matched random tapes to selected actions;
8. assemble reward, sampled miss cost, and conditional miss risk;
9. validate termination, truncation, bootstrap, and learning masks;
10. record bounded action-dependent feedback, close the frame, and release
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
applied, and the transition's learning mask is false. A rectangular
`FrameObservation` uses an internal finite zero sentinel only to exercise API
shape/action validation for those rows; that sentinel is not policy input or
training data.

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
outcomes, and lifecycle masks. `--frames` is a diagnostic processing cutoff and
does not manufacture truncation flags for pairs still active at the cutoff.
Use `--frames 0` to replay the complete trace.

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
| Pair births | 283 | 283 |
| Natural terminations | 60 | 60 |
| Sampled misses | 1,385 | 1,415 |
| Reserved RF attempts | 42,173 | 41,715 |
| VLC activations | 10,561 | 10,381 |
| Maximum active population | 232 | 232 |
| Maximum RF-pool utilization | 1.410 | 1.715 |
| Replay verified | yes | yes |
| SHA-256 fingerprint | `5435a14a94df9eec84820d77a0060a22264e005f71e7bc811693e4432b8f1ee8` | `481ab31ed03a75f16867e843c089b883821170ba1d29ed0eabf3fb57b715ca5d` |

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
- full-trace natural, internal-truncation, and trace-end lifecycle accounting;
- cutoff semantics without fabricated truncations;
- random, cycle, and fixed-action policy-name validation.

The existing legacy rollout tests remain unchanged in behavior. The extracted
action-independent channel evaluation shares one correlated-state advance with
the legacy action evaluator, which continues to use its complete hopped fading
tape.

## Remaining Phase 5 deliverable

This validation completes the final item in the Phase 5 task checklist and its
long-rollout completion gate. The separate work-plan deliverable
"Observation-normalization workflow fitted on training data only" is not
implemented by this harness. Phase 5 should not be declared fully complete
until that stateful, checkpointable, split-isolated workflow and its leakage
tests exist.
