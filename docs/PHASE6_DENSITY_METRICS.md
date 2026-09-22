# Phase 6 reliability and resource metrics by density

Date: 2026-09-22

## Reporting contract

`build_density_metrics_report` groups a matched policy campaign by traffic
density and policy. The default reporting split is held-out `test`; training
and validation traces remain part of campaign execution only so observation
normalization is learned on training data and frozen before evaluation.

Every policy/density row reports raw evidence plus these metrics:

- sampled misses, sampled miss rate, and its one-sided cluster-bootstrap upper
  bound;
- conditional miss rate and its one-sided cluster-bootstrap upper bound;
- mean activation cost and reserved RF attempts per packet;
- RF-use, VLC-use, duplication, and no-observation-fallback fractions;
- mean population, mean RF-pool utilization, and maximum utilization; and
- the count of every one of the nine contract actions.

The conditional-risk upper bound carries the reliability decision because it
remains informative when no sampled misses occur. The sampled bound is retained
as a calibration check and explicitly marked uninformative when its miss count
is zero.

## Correlation and matched uncertainty

Packets within one trajectory/pair episode share geometry and channel state,
so they are not treated as independent Bernoulli trials. Each rollout now
preserves one compact `EpisodeClusterTally` per pair episode. The report pools
clusters across the three held-out replicates at a density and resamples whole
clusters, recomputing each rate as a ratio of sums.

Cluster IDs and packet counts are checked for exact equality across policies.
Every policy uses the same canonical cluster order and bootstrap seed, so its
confidence bound is based on matched resamples as well as matched packet tapes.

## Evidence gate

A row receives a boolean `meets_miss_budget` verdict only when all three
conditions hold:

1. every contributing trace was consumed completely;
2. packets meet `evaluation.min_packets_per_policy_density`; and
3. pair episodes meet `evaluation.min_trajectory_pair_clusters`.

Otherwise the JSON value is `null` and `evaluation_ready` is false. This makes
a diagnostic frame cutoff useful for integration testing without allowing it
to masquerade as a paper result. Under the headline profile the thresholds are
1,000,000 packets and 200 pair-episode clusters per policy/density.

## Reproduction

The following command runs the default baseline set and writes both the matched
campaign and its test-density report:

```bash
.venv/bin/python scripts/run_matched_baselines.py \
  --frames 0 \
  --environment-seed 81 \
  --bootstrap-seed 17 \
  --out artifacts/evaluations/phase6_matched_baselines.json \
  --density-out artifacts/evaluations/phase6_density_metrics.json
```

The supervised comparator is intentionally not fitted by this command. Include
`--policy supervised-risk-allocation --estimator PATH` with an estimator fitted
only on training traces. A scientific all-baseline run must name that policy
and artifact explicitly; validation or test fitting remains prohibited.

## Real-catalog diagnostic

A two-frame integration run used environment seed 81, 1,000 bootstrap
replicates, and the two fixed comparators `always-vlc` and `always-rf-1` over
all 21 configured traces. The density report correctly selected the nine test
traces and produced one row per policy at each density:

| Density | Policy | Packets | Pair clusters | Conditional miss | Sampled miss | Mean cost | RF attempts/packet | VLC use | Mean pool utilization |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | always-vlc | 832 | 416 | 0.233982 | 0.233173 | 3.0 | 2.0 | 1.0 | 0.6933 |
| 10 | always-rf-1 | 832 | 416 | 0.027984 | 0.021635 | 3.0 | 2.5 | 0.5 | 0.8667 |
| 20 | always-vlc | 2,924 | 1,462 | 0.105675 | 0.110807 | 3.0 | 2.0 | 1.0 | 2.4367 |
| 20 | always-rf-1 | 2,924 | 1,462 | 0.093170 | 0.097127 | 3.0 | 2.5 | 0.5 | 3.0458 |
| 30 | always-vlc | 5,168 | 2,584 | 0.065247 | 0.065015 | 3.0 | 2.0 | 1.0 | 4.3067 |
| 30 | always-rf-1 | 5,168 | 2,584 | 0.154451 | 0.152283 | 3.0 | 2.5 | 0.5 | 5.3833 |

These values are pipeline diagnostics, not scientific findings. With only two
frames, half of every pair's actions are the conservative cold-start `DUP-4`
fallback, no trace is exhausted, and the million-packet requirement is unmet.
Accordingly all six budget verdicts are `null` and the report-level
`evaluation_ready` flag is false.

## Scope boundary

This task provides the reproducible density table and prevents undersized runs
from making reliability claims. Selecting the strongest deployable comparator,
quantifying its matched gap to the truth-risk oracle, and making the PPO
go/no-go decision remain separate Phase 6 tasks.
