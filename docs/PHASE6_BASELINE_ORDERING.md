# Phase 6 baseline ordering checks

Date: 2026-09-22

## Purpose

This task checks the baseline relations that must hold before aggregate
performance comparisons are interpreted. It separates mathematical
invariants from empirical rankings. Only the former are called "expected
ordering" here.

There is no valid universal ordering of geometry-threshold, contextual,
supervised, and truth-risk policies. They choose different actions as channel
state and action-coupled RF load change. Their reliability and resource use
must therefore be measured by density in the next Phase 6 tasks.

## Fixed-load limiting cases

For RF per-attempt miss risk `r`, VLC miss risk `v`, and a shared RF-pool
response held fixed, the action risks are:

- `RF-n`: `r^n`;
- `VLC`: `v`; and
- `DUP-n`: `r^n v`.

The executable audit checks the following boundaries:

- RF retry risk is non-increasing from RF-1 through RF-4 for `0 <= r <= 1`;
- duplication is no worse than either component leg;
- perfect VLC makes VLC and duplication risk zero;
- certain VLC failure reduces duplicate risk to its RF leg;
- a perfect RF attempt makes RF risk zero; and
- certain RF failure reduces duplicate risk to its VLC leg.

These are fixed-load statements. A cross-policy RF-1/RF-4 rollout does not
hold load fixed because reserving more retries also increases current pool
demand. The audit intentionally does not claim that RF-4 must outperform RF-1
under every population load.

## Matched-rollout ordering

The audit requires `always-vlc`, all four fixed RF levels, and
`duplicate-all` in one `MatchedPolicyCampaignReport`. For every trace it
checks:

1. RF attempts, VLC activations, and resource reward exactly match the chosen
   fixed action plus the common no-observation fallback.
2. Lower configured activation cost produces no lower resource reward. Equal
   configured costs produce equal reward.
3. `duplicate-all` has no greater conditional-risk sum or sampled miss count
   than `always-vlc`.
4. `duplicate-all` has no greater conditional-risk sum or sampled miss count
   than `always-rf-4`.

The duplicate comparisons are valid pathwise. Duplicate-all and RF-4 reserve
the same RF attempts and therefore see the same RF-pool response; duplicate-all
and VLC see the same action-independent optical channel. Matched packet tapes
then make duplicate delivery the logical union of the two component outcomes.

The audit rejects a campaign with no usable policy transitions so a one-frame
cold-start run cannot pass through identical fallback actions alone.

## Reproduction

Run the fixed comparator set over the configured traces and emit both
artifacts:

```bash
.venv/bin/python scripts/run_matched_baselines.py \
  --policy always-vlc \
  --policy always-rf-1 \
  --policy always-rf-2 \
  --policy always-rf-3 \
  --policy always-rf-4 \
  --policy duplicate-all \
  --frames 10 \
  --out artifacts/evaluations/phase6_matched_fixed.json \
  --ordering-out artifacts/evaluations/phase6_baseline_ordering.json
```

Use `--frames 0` for the later full scientific evaluation. The ordering JSON
contains every relation, both operands, its scenario, and its verdict. The
command exits nonzero if any guaranteed relation fails.

## Real-catalog smoke result

The command above was run with environment seed 81 and a two-frame cutoff over
the exact configured 9/3/9 train-validation-test catalog. The common trace
structure contained 10,430 usable transitions per policy across 21 traces, and
all six fixed policies replayed that structure. All 787 limiting-case,
resource-accounting, cost-ordering, conditional-risk, and pathwise sampled-miss
checks passed; zero checks failed. The JSON artifacts were written outside the
repository because this bounded smoke run is validation evidence, not the
full per-density scientific result.

## Scope boundary

Passing these checks establishes internal consistency, not the RL opportunity.
Per-density reliability/resource estimates, deployable-to-oracle gaps, and the
sequential-information advantage remain open Phase 6 tasks.
