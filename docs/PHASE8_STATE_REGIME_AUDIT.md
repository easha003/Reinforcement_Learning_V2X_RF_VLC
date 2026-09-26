# Phase 8 causal state-regime and action-feasibility audit

## Decision

The audit path is implemented and passes its integration smoke test, but the
coverage gate is not yet satisfied.  The first three-point temporal diagnostic
found four of the five proposed causal contexts somewhere in training and
validation.  It found no exogenous uncertainty context under the frozen
predictor-confidence and track-age definition, and several contexts were absent
at density 10.  A full PPO recovery run must therefore remain blocked until an
evidence-scale audit either demonstrates adequate support or motivates a
declared scenario/observation change.

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
  `hybrid-rf-vlc-rl.state-regime-audit.v1`.

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

The fitted training thresholds also expose why uncertainty needs attention:
the predictor-confidence lower quartile was approximately `0.9950`, and the
track-age upper quartile was approximately `0.05 s`.  No audited row crossed
the strict tail boundary.  The current synthetic campaign therefore does not
yet support a claim that PPO learned an exogenous uncertainty-to-duplication
rule.

## Preliminary action evidence

The action audit does not support the proposed mapping as a literal oracle
table:

- Under VLC offload, RF-1 and VLC dominate the cheapest feasible selections.
- Under RF-1 and RF-4 population pressure, selections often move directly to
  RF-4, DUP-4, or VLC.
- RF-2 and RF-3 occur only exceptionally in this sparse sample.
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

The `3 x 16` run is intentionally larger than the completed three-point
diagnostic.  Its result must be reviewed before changing scenario generation,
regime definitions, constraint scaling, or PPO hyperparameters.

Repository verification after integration completed with:

- Ruff: all source, test, and script checks passed.
- Strict mypy: 113 source files passed.
- Pytest: 1,421 passed and 2 expected skips.

## Next gate

1. Run and inspect the bounded `3 x 16` audit.
2. Determine whether the zero uncertainty support persists across temporal
   windows with causal history.
3. If it persists, explicitly choose between adding a synthetic uncertainty
   scenario and removing the uncertainty-to-duplication claim.
4. Only after the coverage decision, run the bounded constraint-pressure
   recovery experiment.

When a full PPO training campaign eventually begins, Codex monitoring uses a
one-hour check interval unless the user changes that instruction.  This audit
does not launch or authorize that full campaign.
