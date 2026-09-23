# Phase 6 deployable-to-oracle gap

Date: 2026-09-23

## Purpose

`build_oracle_gap_report` measures the remaining resource-efficiency headroom
between the strongest reliable deployable baseline and the non-deployable
truth-risk oracle. It consumes the matched campaign and the density report,
rather than replaying policies through a separate evaluation path.

The report refuses to call a policy "best" unless all required comparators are
present:

- always RF at each of the four RF-attempt levels;
- always VLC;
- duplicate-all;
- observable geometry-threshold selection;
- contextual selection without link history;
- supervised optical-risk estimation plus analytical allocation; and
- the truth-risk oracle.

The supervised comparator requires a serialized estimator fitted only on the
training traces. A report built from an incomplete comparator set raises an
error instead of silently overstating the available baseline.

## Reliability-first selection

Selection is performed independently at each held-out test density. A
deployable policy is eligible only when its one-sided cluster-bootstrap upper
bound for conditional deadline-miss risk is at or below the configured budget.
Among eligible policies, the report selects the lowest mean activation cost,
then breaks ties by conditional-risk upper bound, reserved RF attempts, and
policy name.

The density status makes non-comparable cases explicit:

- `diagnostic`: the complete-source, packet-count, or cluster-count evidence
  gate has not passed;
- `comparable`: both the selected deployable and oracle are feasible;
- `oracle-only-feasible`: the oracle is feasible but no deployable is;
- `oracle-infeasible`: a deployable passes while the oracle does not; and
- `no-feasible-policy`: neither side passes the reliability rule.

Only `comparable` rows contain numerical gap estimates. A cutoff diagnostic
therefore cannot produce a best-policy or efficiency claim.

## Paired gap estimates

Every reported gap is deployable minus oracle. Positive activation-cost or RF
use gaps indicate potential headroom for a learned policy. The report includes:

- mean activation cost;
- mean reserved RF attempts;
- RF-use, VLC-use, and duplication fractions;
- conditional miss rate; and
- sampled miss rate.

The estimator verifies exact trace, pair-episode, and packet-count alignment,
then applies the same episode-cluster bootstrap draws to both policies and all
metrics. Its two-sided percentile interval therefore preserves the matched
experimental design and within-trajectory dependence. Configuration hash,
environment seed, policy set, trace membership, packet counts, and source
completion are checked across the input artifacts before comparison.

## Reproduction

A scientific run supplies a real training-fitted estimator, consumes every
configured frame, and writes the matched campaign, density metrics, and gap
report together:

```bash
.venv/bin/python scripts/run_matched_baselines.py \
  --estimator artifacts/models/supervised_optical_risk.json \
  --frames 0 \
  --environment-seed 81 \
  --bootstrap-seed 17 \
  --out artifacts/evaluations/phase6_matched_baselines.json \
  --density-out artifacts/evaluations/phase6_density_metrics.json \
  --oracle-gap-out artifacts/evaluations/phase6_oracle_gap.json
```

When `--estimator` is supplied without explicit `--policy` arguments, the CLI
runs the complete ten-policy comparison automatically. `--frames 0` means no
frame cutoff.

## Validation evidence

Unit coverage exercises two complementary boundaries. Complete small synthetic
sources with relaxed evidence thresholds produce comparable density rows,
select a feasible deployable policy, and emit all seven paired estimates.
Cutoff sources produce diagnostic rows with no winner and no numerical gap. A
separate test proves that an incomplete comparator set is rejected.

A one-frame real-catalog integration diagnostic also ran all ten policies on
all 21 configured traces. The matched campaign passed, and all three held-out
density rows correctly remained `diagnostic`: no source was exhausted, no best
deployable was selected, and no estimates were emitted. The estimator used for
that plumbing check was temporary synthetic data, not a training-trace model;
the run is not scientific evidence and no numerical gap from it may be quoted.

## Claim boundary

This task establishes the reproducible and statistically guarded gap
measurement. The paper's numerical gap is still pending a real estimator fitted
only on training traces and the uncapped, full-evidence campaign. The result of
that campaign, together with the next temporal/population-coupling test, will
support the Phase 6 PPO go/no-go decision; this task alone does not make that
decision.
