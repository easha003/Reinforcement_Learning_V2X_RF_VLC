# Phase 6 matched policy comparisons

Date: 2026-09-22

## Contract implemented

`run_matched_policy_campaign` compares policies only when they replay the exact
train, validation, and test membership declared by the resolved configuration.
It rejects a missing, added, reordered, or relabeled trace. The current
headline catalog contains:

- 9 training traces: three replicates at each of densities 10, 20, and 30;
- 3 validation traces: one replicate at each density; and
- 9 test traces: three replicates at each density.

Every policy receives the same active environment root seed, trace order, and
per-trace frame cutoff. Each `DeterministicRolloutReport` now records both the
full policy-dependent rollout fingerprint and a separate
`matched_tape_fingerprint`.

## Random-number evidence

The tape fingerprint hashes the complete pre-action randomness for every
packet processed in a trace:

- stable trace, pair-episode, and packet identity;
- all four RF half-duplex, collision, and decoding draws; and
- the VLC decoding draw.

The digest also binds the matched-tape schema, active environment seed, trace
ID, and frame cutoff. A `MatchedTraceComparison` is invalid unless every policy
has the same 64-character digest. This proves that action differences select
different views of the same packet tapes rather than advancing policy-specific
random streams.

The environment seed also addresses sensing noise, correlated RF shadowing and
fading, and feedback measurement noise. Mobility itself is frozen in immutable
trace artifacts. Policy action-sampling seeds remain a separate concern and do
not advance environment randomness.

## Structural matching

For every trace, the campaign also requires all policies to agree on:

- frame count and source-exhaustion status;
- transition, usable-row, and fallback-row counts;
- births, natural terminations, internal truncations, and trace-end
  truncations; and
- maximum active population.

Rewards, misses, conditional risks, RF attempts, VLC activations, pool load,
and the full rollout fingerprint are intentionally allowed to differ because
those are consequences of policy actions.

## Train-only normalization across splits

Each policy owns one normalization state for the campaign. Its state is carried
in configured order across all nine training traces. After the final training
trace, it is frozen and checkpointed. The frozen state is then restored for
all validation and test traces, where the report requires zero normalization
updates.

Normalization values can be policy-dependent because link-quality feedback is
action-dependent. They therefore are not forced equal across policies. Counts
and split timing are matched, while each policy retains its honest training
trajectory. All current Phase 6 baselines make decisions from raw causal
features; normalized tensors are still produced and audited through the common
environment path for later PPO compatibility.

## Supervised estimator artifact

`SupervisedOpticalRiskEstimator` now has a versioned JSON round trip. A matched
campaign can include `supervised-risk-allocation` only when supplied an
estimator fitted previously on training data. The command-line runner never
fits a model from validation or test rows.

## Reproduction

Run a diagnostic campaign from the repository root:

```bash
.venv/bin/python scripts/run_matched_baselines.py \
  --policy always-vlc \
  --policy always-rf-1 \
  --frames 10 \
  --out artifacts/evaluations/phase6_matched_baselines.json
```

Use `--frames 0` for complete traces. The default policy set contains all fixed
RF levels, VLC-only, duplicate-all, geometry-threshold, contextual-no-history,
and the truth oracle. Add `supervised-risk-allocation` with
`--estimator PATH` after its train-only artifact has been produced.

## Real-catalog smoke result

A one-frame campaign with environment seed 81 compared `always-vlc` and
`always-rf-1` over all 21 configured traces. It passed exact membership,
structural matching, frozen split handling, and all 21 tape-fingerprint checks.
The report was written outside the repository because one frame per trace is a
mechanical smoke check, not a scientific evaluation result.

## Automated checks

`tests/unit/test_matched_policy_campaign.py` verifies:

- exact configured train/validation/test membership;
- identical per-trace tape fingerprints across policies;
- matched lifecycle and transition structure;
- policy-specific training normalization carried across traces;
- freezing at the train/validation boundary and zero later updates;
- deterministic campaign replay;
- a different environment seed producing a different tape fingerprint;
- rejection of an incomplete catalog; and
- atomic JSON report persistence.

The matched campaign creates the fair execution substrate only. Expected
baseline ordering, per-density reliability/resource summaries, and
deployable-to-oracle gaps remain separate Phase 6 tasks.
