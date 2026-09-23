# Phase 6 sequential-control opportunity

Date: 2026-09-23

## Question and claim boundary

This task asks whether the environment contains information or interaction that
a stateless, independently acting contextual selector cannot exploit. It uses
two necessary-condition probes:

1. Does action-dependent causal link history improve held-out prediction of
   current RF or VLC failure risk beyond the contextual feature set?
2. Holding a focal RF action and its propagation state fixed, can changing only
   the other agents' simultaneous RF reservations move that focal action across
   the reliability budget?

These probes establish whether sequential/population-aware control has a real
opportunity. They do not establish that PPO can learn it or outperform the
strongest feasible deployable baseline. That claim remains subject to Phases 7
through 9.

## Link-history probe

The history comparison reconstructs the configured causal observation from
policy-independent transition caches. Within each episode it follows a fixed
RF, VLC, DUP cycle that depends only on the episode step. The chosen action
therefore decides which noisy post-transmission quality is refreshed, while the
unused leg ages exactly as it will during training.

Two equal-form ridge-logit models are fitted only on training caches:

- contextual: the 12 features used by `contextual-no-history`; and
- history-aware: the same 12 features plus RF/VLC quality, quality ages,
  eight-reading histories, previous action, last outcome, and consecutive
  misses.

Both are scored against untouched per-packet RF/VLC conditional risks from test
caches. The metric is Brier score, and uncertainty resamples complete test
pair episodes. A history opportunity is declared only when the configured row
and pair-episode evidence minima pass, every configured train/test source is
present, and at least one paired Brier-gain interval is strictly above zero.

## Population-coupling probe

The population probe uses the complete verified test frame caches. For each
observed population and each RF reservation level from one through four, it
compares two counterfactuals through the authoritative `RFPoolModel`:

- low demand: the focal pair selects RF-n and every other pair selects VLC; and
- high demand: every pair selects RF-n.

Population, focal action, sensing band, and optimistic zero RF decoding risk
are fixed. Thus the only changing mechanism is access risk created by the
other agents' current reservations. A feasibility flip occurs when low-demand
packet risk is within the `1e-4` budget and high-demand risk exceeds it.

This is deliberately stronger than showing that pool utilization changes. A
budget crossing means a reliable controller must change its action or accept a
constraint violation; a per-agent selector that treats current joint demand as
fixed cannot represent that response.

## Real-artifact diagnostic

The report verified all nine held-out population-frame caches: 26,973 nonempty
frames at each density. Population support and the configured RF-3 probe were:

| Density | Population min/median/max | First RF-3 flip | Flip-frame share | Low-demand risk at median | All-RF risk at median |
|---:|---:|---:|---:|---:|---:|
| 10 | 137 / 245 / 306 | 30 | 100% | 5.35e-10 | 1.61e-2 |
| 20 | 484 / 666 / 827 | 30 | 100% | 4.62e-10 | 1.53e-1 |
| 30 | 858 / 1,061 / 1,509 | 30 | 100% | 4.47e-10 | 3.46e-1 |

All four RF reservation levels had a feasibility flip on every observed test
frame. This is complete held-out structural evidence that current population
actions are decision-relevant under the configured RF-pool model. It does not
say that the actual contextual policy chooses the all-RF counterfactual.

The bounded history diagnostic used 10,008 source packets from one training
trace and 10,343 from one test trace. After excluding the first packet of each
episode it contained 9,274 training rows, 9,631 test rows, and 712 test pair
episodes. VLC Brier score fell from 0.08059 to 0.05666, a paired gain of
0.02393 with a 95% interval of [0.01229, 0.03568]. RF history showed no gain.
This VLC result is encouraging but is not a claim: eight of nine train and test
sources are absent and the one-million-row threshold is unmet, so the report
status is `diagnostic` and its history verdict is `null`.

## PPO gate decision

The generated report returns `gate_decision: go`, based only on the complete
population-coupling evidence. This satisfies the Phase 6 rule that at least one
concrete opportunity must remain before implementing PPO. The history result
does not contribute to that decision until its evidence gate passes.

The GO decision means “proceed to implement and test the learner.” It is not a
prediction that PPO will beat the supervised analytical allocator, which is
already population-coupled and remains the strongest deployable comparison.

## Reproduction

Build transition caches independently for every train/test trace. The example
below uses the cache builder's default packet target:

```bash
for split in train test; do
  for replicate in 0 1 2; do
    .venv/bin/python scripts/build_training_cache.py \
      --split "$split" \
      --densities 10 20 30 \
      --replicate "$replicate"
  done
done
```

Then create the versioned opportunity report:

```bash
.venv/bin/python scripts/run_sequential_opportunity.py \
  --transition-cache-root artifacts/caches \
  --frame-cache-root artifacts/frame_caches \
  --bootstrap-seed 17 \
  --out artifacts/evaluations/phase6_sequential_opportunity.json
```

The CLI discovers train/test transition caches, verifies their configuration
and split identities, verifies exact held-out frame-cache membership, and
writes no claim when either evidence gate is incomplete.
