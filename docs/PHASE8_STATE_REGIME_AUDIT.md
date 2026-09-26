# Phase 8 causal state-regime and action-feasibility audit

## Decision

The audit path and the bounded `3 x 16` evidence run are complete. All five
proposed causal contexts occur in both training and validation when aggregated
across density, so the synthetic campaign does expose every conceptual state.
Coverage is not uniform: moderate-RF and heavy-contention/optical-permitted
contexts are effectively absent at density 10, and only 4 of the 30 strict
split/density/regime cells reached 10,000 rows plus 200 pair episodes. The
campaign-level training rows meet those minima for all five regimes, while the
bounded validation sample does not. A full PPO recovery run remains blocked
until the intended coverage claim is frozen and the bounded
constraint-pressure recovery experiment passes.

This audit does not prescribe a supervised action label to PPO.  It asks
whether the causal observations contain the proposed contexts and separately
uses hidden simulator truth to identify feasible and resource-minimal actions.
Simulator truth remains audit-only and never enters the actor observation.

## Operational regime definitions

Thresholds are fitted from sampled **training** actor rows and frozen before
validation.  The lower and upper thresholds are the 25th and 75th percentiles.
The optical FOV boundary remains the physical zero-radian boundary.  Labels may
overlap.

| Regime | Causal definition |
|---|---|
| `easy_state` | favorable optical geometry/forecast, light RF context, and no uncertainty flag |
| `moderate_rf_conditions` | moderate RF context, optically ambiguous, and no uncertainty flag |
| `poor_vlc_usable_rf` | impaired optical context while RF is not in the heavy tail |
| `uncertain_mixed_state` | predictor confidence below its training lower quartile or track age above its upper quartile |
| `heavy_rf_contention_optical_permitted` | heavy RF context while optical conditions remain favorable |

RF context uses only actor-visible local CBR and neighbor count.  Optical
context uses actor-visible FOV margin and predicted blockage probability.
Uncertainty uses actor-visible predictor confidence and track age.  Link
quality histories, true channel state, current actions, outcomes, and density
labels are not used to define a regime.

Strict inequalities are used for the uncertainty tails.  This prevents a
quantile tie, such as every causal track being exactly 50 ms old, from labeling
the entire dataset uncertain.

## Counterfactual action audit

Every usable row is evaluated against all nine allowed actions under three
declared RF population loads:

| Profile | Other active pairs reserve |
|---|---:|
| `vlc_offload` | 0 RF attempts |
| `rf1_pressure` | 1 RF attempt each |
| `rf4_pressure` | 4 RF attempts each |

The focal candidate replaces only its own RF-attempt count.  The shared-pool
collision and half-duplex risks are recomputed from aggregate load, then
combined with the focal pair's exact RF decoding risk.  VLC and RF/VLC joint
risks use the exact same physical truth as the authoritative rollout.  The
report records every feasible action and selects the lowest-cost feasible
action; when none is feasible, it records the minimum-risk action instead.

These profiles are sensitivity probes, not claims that the population truly
chooses one common action.  They expose whether the desired action preference
survives low, moderate, and severe endogenous RF pressure.

## Leakage and reproducibility controls

- Only `train` is used to fit thresholds and observation normalization.
- Normalization is frozen once before any validation window is opened.
- Only `train` and `validation` sources can enter the campaign builder.
- Every report persists `test_split_opened: false`.
- Window locations, seeds, thresholds, per-regime trace/cluster counts, all
  action feasibility counts, and selected-action counts are persisted.
- The report is versioned as
  `hybrid-rf-vlc-rl.state-regime-audit.v2`. Version 2 freezes the scientific
  claim as campaign-level coverage with density-conditioned support and
  explicitly disclaims uniform regime support at every density.

## Preliminary integration run

The integration run sampled one frame at the beginning, middle, and end of
every configured training and validation trace.  It used all three training
replicates and the one configured validation replicate at each density.  The
coverage requirement was intentionally left at 1,000 rows and 200 independent
pair episodes per split/density/regime, so a three-frame diagnostic could not
silently pass as final evidence.

Aggregate observed rows were:

| Regime | Training rows | Validation rows | Densities with nonzero train / validation support |
|---|---:|---:|---|
| Easy | 564 | 197 | 3 / 3 |
| Moderate RF | 1,030 | 237 | 2 / 2 |
| Poor VLC / usable RF | 964 | 255 | 3 / 3 |
| Uncertain mixed | 0 | 0 | 0 / 0 |
| Heavy RF / optical permitted | 517 | 212 | 2 / 2 |

The formal result was `0/30` supported split/density/regime cells because no
cell met both declared evidence minima.  This is the intended fail-closed
result for a sparse diagnostic.

The preliminary fitted thresholds initially suggested uncertainty needed
attention:
the predictor-confidence lower quartile was approximately `0.9950`, and the
track-age upper quartile was approximately `0.05 s`.  No audited row crossed
the strict tail boundary in the one-frame windows. The evidence run below
shows that this zero was a sparse-window artifact, not an absent scenario.

## Evidence-scale `3 x 16` result

The bounded evidence run processed three 16-frame causal windows per trace:
beginning, middle, and end. It retained 205,225 training rows for threshold
fitting and audited 36 windows across all training and validation traces. The
test split remained unopened.

| Regime | Training rows / clusters | Validation rows / clusters | Strict supported density cells |
|---|---:|---:|---:|
| Easy | 16,248 / 2,864 | 5,454 / 1,002 | 0 |
| Moderate RF | 30,011 / 4,476 | 7,123 / 1,009 | 2 |
| Poor VLC / usable RF | 30,394 / 7,495 | 8,110 / 1,995 | 1 |
| Uncertain mixed | 17,580 / 5,860 | 5,871 / 1,957 | 0 |
| Heavy RF / optical permitted | 18,090 / 4,918 | 7,174 / 2,007 | 1 |

Therefore:

- all five regimes are observed in training and validation at campaign level;
- all five training aggregates exceed 10,000 rows and 200 clusters;
- all validation aggregates exceed 200 clusters but not 10,000 rows in this
  bounded sample;
- the strict cell gate is `4/30`, so `all_split_density_regimes_supported` is
  false; and
- density 10 contains only 5 training and 0 validation moderate-RF rows, and no
  heavy-contention/optical-permitted rows. This is consistent with the light
  traffic context but prevents an every-regime-at-every-density claim.

## Preliminary action evidence

The evidence-scale action audit does not support the proposed mapping as a
literal oracle table. Aggregated across split and density:

- Under VLC offload, RF-1 and VLC dominate every regime. For example, the easy
  regime is 56% RF-1 and 44% VLC.
- Under RF-4 population pressure, the easy regime is 45% DUP-4, 44% VLC, and
  11% RF-4; this is not an unconditional VLC/RF-1 state.
- Moderate RF is 89% VLC under every declared load profile, not predominantly
  RF-2/RF-3.
- Poor-VLC/usable-RF is still 56% VLC under truth and rises from 10% DUP-4 under
  RF-1 pressure to 16% under RF-4 pressure.
- Uncertain mixed state rises from 12% DUP-4 under RF-1 pressure to 17% under
  RF-4 pressure, but remains 70% VLC.
- Heavy RF with optical permission is 74% VLC under pressure, which supports
  that qualitative part of the hypothesis.
- RF-2 and RF-3 occur only exceptionally.
- Rows labeled poor-VLC from causal forecasts sometimes have VLC as the
  truth-oracle selection.  This is direct evidence of partial observability,
  not a report inconsistency.

Consequently, successful learning should be judged by feasible action mass,
resource regret, constraint satisfaction, and physically consistent
sensitivity—not by forcing a hand-written action label in each regime.

## Commands

Focused verification:

```bash
.venv/bin/ruff check \
  src/hybrid_v2x_rl/mean_field/state_regime_audit.py \
  src/hybrid_v2x_rl/mean_field/rf_pool.py \
  scripts/run_state_regime_audit.py \
  tests/unit/test_state_regime_audit.py
.venv/bin/mypy \
  src/hybrid_v2x_rl/mean_field/state_regime_audit.py \
  src/hybrid_v2x_rl/mean_field/rf_pool.py
.venv/bin/pytest -q \
  tests/unit/test_state_regime_audit.py \
  tests/unit/test_rf_pool.py \
  tests/unit/test_rf_pool_regimes.py
```

Bounded evidence run:

```bash
.venv/bin/python scripts/run_state_regime_audit.py \
  --windows-per-trace 3 \
  --frames-per-window 16 \
  --minimum-rows 10000 \
  --minimum-clusters 200
```

The command above produced the evidence-scale result recorded in this
document. Its generated JSON remains a local, ignored evaluation artifact.

Repository verification after integration completed with:

- Ruff: all source, test, and script checks passed.
- Strict mypy: 113 source files passed.
- Pytest: 1,421 passed and 2 expected skips.

## Next gate

1. [x] Freeze the scientific interpretation as campaign-level state coverage
   plus density-conditioned support; do not claim every regime exists at every
   density.
2. [x] Add the same regime labels to bounded PPO evaluation so learned action
   probabilities, feasible-action mass, miss risk, and resource regret are
   reported per regime. The completed seed-1001 engineering-checkpoint result
   is recorded in `PHASE8_PPO_REGIME_EVALUATION.md`.
3. [ ] Run the bounded constraint-pressure recovery experiment. Do not require PPO
   to reproduce a hand-written action table that the counterfactual oracle
   itself does not reproduce.

When a full PPO training campaign eventually begins, Codex monitoring uses a
one-hour check interval unless the user changes that instruction.  This audit
does not launch or authorize that full campaign.
