# Phase 8 regime-conditioned PPO evaluation

## Decision

The bounded validation evaluation is complete for the first seed-1001
engineering checkpoint. It confirms that the policy responds to density and
causal context, but it also confirms severe reliability failure and
near-collapse to `RF-1`/`VLC`. This checkpoint remains ineligible for paper
selection. The next task is the bounded constraint-pressure recovery
experiment; seeds 1002--1005 and a fresh full seed-1001 run remain blocked.

The scientific coverage claim is now frozen as **campaign-level regime
coverage with density-conditioned support**. Every declared regime occurs in
both training and validation across the campaign, but the project does not
claim that every regime occurs at every density. In particular, density 10 has
no moderate-RF or heavy-contention/optical-permitted validation rows in the
bounded sample.

## Evaluation contract

The evaluator restores a checkpoint with global-RNG restoration disabled,
freezes the checkpoint's training-fitted normalization state, and replays only
the nine validation windows inherited from the `3 x 16` causal coverage audit.
It never opens the test split.

For each usable actor row:

1. the PPO actor receives only its normalized causal observation and hardware
   action mask;
2. the evaluator records all nine masked action probabilities and selects the
   deterministic masked argmax;
3. raw actor-visible values receive the frozen training-fitted regime labels;
4. isolated simulator truth then scores the already-produced policy
   distribution under the exact policy-induced RF load and the three declared
   load probes; and
5. the ordinary rollout path produces the actual selected-action risk and
   sampled outcome.

Simulator truth therefore scores behavior but cannot affect PPO probabilities
or action selection. The evaluator also proves that the actor parameters,
frozen normalization, and Python, NumPy, and Torch global RNG states do not
change.

The versioned report schema is
`hybrid-rf-vlc-rl.ppo-regime-evaluation.v1`. The source regime-audit schema is
`hybrid-rf-vlc-rl.state-regime-audit.v2`, which persists the coverage claim,
density-conditioned cells, training-only threshold fit, validation windows,
and `test_split_opened: false`.

## Seed-1001 bounded result

- Checkpoint: iteration 265, 9,998,802 environment transitions
- Policy seed: 1001
- Validation windows: 9 (three per validation trace and density)
- Reliability budget: `1e-4`
- Test split opened: no

| Causal regime | Rows | Mean VLC probability | Mean RF-1 probability | Actual conditional miss risk | Selected feasible fraction | Feasible policy mass | Feasible-conditioned resource regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 5,454 | 0.206 | 0.785 | 0.0741 | 0.203 | 0.203 | 0.458 |
| Moderate RF | 7,123 | 0.791 | 0.197 | 0.0239 | 0.801 | 0.797 | 0.098 |
| Poor VLC / usable RF | 8,110 | 0.370 | 0.615 | 0.0687 | 0.373 | 0.371 | 0.279 |
| Uncertain mixed | 5,871 | 0.433 | 0.564 | 0.1009 | 0.417 | 0.432 | 0.114 |
| Heavy RF / optical permitted | 7,174 | 0.638 | 0.340 | 0.0505 | 0.641 | 0.643 | 0.184 |

The feasible-policy-mass column is the probability assigned to actions whose
conditional miss risk is at most `1e-4` under the exact other-pair RF load
induced by the deterministic PPO joint action. Resource regret is computed
only after renormalizing over feasible policy mass. Infeasible probability is
reported separately in the JSON, so an unsafe low-cost action cannot appear
resource-optimal merely because it is cheap.

## Interpretation

The actor is context-sensitive: mean VLC probability rises from 20.6% in easy
states to 79.1% in moderate-RF states and 63.8% under heavy RF contention when
optical conditions permit. That direction is physically plausible. It is not,
however, sufficient constrained learning:

- actual conditional miss risk remains orders of magnitude above `1e-4` in
  every regime;
- only 20.3% of deterministic easy-state selections and 41.7% of uncertain
  selections are feasible;
- the distribution places about 79.7%, 62.9%, and 56.8% infeasible mass in the
  easy, poor-VLC, and uncertain regimes, respectively;
- RF-2, RF-3, RF-4, and duplication actions receive negligible probability,
  despite counterfactual evidence that stronger RF or duplication is often
  required for feasibility; and
- deterministic selected-action feasibility closely follows feasible policy
  mass, confirming that this is a distributional collapse rather than an
  argmax-only artifact.

The hand-written action table remains inappropriate as a supervised target.
The recovery decision should instead use constraint satisfaction, feasible
probability mass, conditional risk, resource regret, and load sensitivity.

## Density-conditioned support

The regime-conditioned report includes all 15 validation density/regime cells,
including zero-row cells. Density 10 contains zero moderate-RF and zero heavy
RF/optical-permitted rows. Nonzero examples also show strong density effects:
easy-state VLC probability is approximately 0.113, 0.320, and 0.216 at
densities 10, 20, and 30, while heavy-RF/optical-permitted VLC probability is
approximately 0.491 at density 20 and 0.671 at density 30.

These are conditional diagnostics, not estimates for unsupported cells. The
campaign aggregate demonstrates context coverage; density rows define where
each conditional claim has empirical support.

## Commands

Refresh the frozen coverage artifact:

```bash
.venv/bin/python scripts/run_state_regime_audit.py \
  --windows-per-trace 3 \
  --frames-per-window 16 \
  --minimum-rows 10000 \
  --minimum-clusters 200
```

Evaluate the engineering checkpoint:

```bash
.venv/bin/python scripts/run_ppo_regime_evaluation.py \
  --checkpoint \
    artifacts/logs/phase8-joint-seed1001/checkpoints/checkpoint-iteration-000265.pt
```

The generated JSON artifacts remain local and ignored:

- `artifacts/evaluations/phase8_state_regime_audit.json`
- `artifacts/evaluations/phase8_seed1001_ppo_regime_evaluation.json`

## Verification

- Ruff passed across all source, test, and script files.
- Strict mypy passed across 114 source files.
- Pytest passed 1,425 tests with 2 expected skips.
- The evidence runner verified an unchanged actor, frozen normalization, and
  unchanged Python, NumPy, and Torch global RNG states.
- Both evidence artifacts record `test_split_opened: false`; the PPO report
  binds the checkpoint and threshold source by SHA-256.

## Next gate

Run a bounded constraint-pressure recovery experiment with the versioned
diagnostics enabled. Compare it with the failed pilot using aggregate,
density-conditioned, and regime-conditioned reliability. Do not launch a full
campaign until the bounded run shows that the constraint signal materially
increases feasible action mass without numerical instability.
