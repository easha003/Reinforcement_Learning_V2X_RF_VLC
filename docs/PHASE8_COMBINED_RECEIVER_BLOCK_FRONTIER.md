# Phase 8 combined receiver/block frontier

Status: declared and implemented; validation execution pending

## Purpose

Neither the headline two-branch MRC receiver at the original blocklength nor a
longer QPSK block with the headline receiver reached the 300 B / 10 ms /
`1e-4` target at every density. This experiment tests the smallest untried
combination of those existing physical interventions: the grid-realizable 2.0
ms QPSK block plus integrated zero-loss two-branch MRC.

It is a propagation-only necessary-condition screen. RF collision and
receiver half-duplex loss remain removed, so passing does not itself establish
system feasibility or authorize PPO.

## Frozen physical grid

The RF candidate is `qpsk-2p0ms-sensitivity`: 9,676 finite-blocklength channel
uses, four current 0.5 ms slots per attempt, and 8 ms for `RF-4`. The action
space, four-attempt limit, payload, deadline, trace windows, environment seed,
and optical profiles remain unchanged.

All three already-declared integrated zero-loss MRC correlation profiles are
retained:

| Receive profile | Correlation | Role |
|---|---:|---|
| Independent ideal | 0.0 | Optimistic sensitivity |
| Low-correlation hardware bound | 0.173205 | Hardware primary |
| Correlated stress | 0.7 | Correlated sensitivity |

Both `wide-60deg` and `concentrated-30deg` optical configurations and all
densities 10, 20, and 30 are evaluated. No profile or density may be added or
removed after results are observed.

## Exact and exploratory decisions

The scientific target remains `1e-4`. A profile/optical pair is exact-feasible
only when its optimistic mean lower bound is at or below `1e-4` at every
density.

Before execution, an exploratory near-feasible boundary is also frozen at
`1.1e-4`, exactly 10% above the target. If no exact pair exists, a pair may
continue only when every density is at or below this near boundary. Candidates
are ranked by their worst-density mean, then their average mean, profile order,
and optical order.

A near-feasible selection does not satisfy the reliability target and must be
reported as such. Either an exact or near selection proceeds first to the
pair-local joint contention and half-duplex frontier. PPO remains blocked
until that full-system behavior is characterized. The test split remains
closed.

## Reproduction commands

Structural validation, with zero evaluated frames:

```bash
.venv/bin/python scripts/run_combined_receiver_block_frontier.py
```

Execute or resume the validation-only propagation screen:

```bash
.venv/bin/python scripts/run_combined_receiver_block_frontier.py --execute
```

The executor checkpoints after every completed receive profile and writes the
final result to
`artifacts/evaluations/phase8_combined_receiver_block_frontier.json`.
