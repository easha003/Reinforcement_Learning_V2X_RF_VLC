# Phase 8 combined receiver/block frontier

Status: completed; no profile reaches the exact target or frozen 10%-over
exploratory gate, so the joint frontier and PPO remain blocked pending an
explicit exploratory override

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

## Completed validation result

The declaration SHA-256 is
`566fef2e20eeb628efe3565160349913300dc57b04800edbc9e9faf74be0667c`.
The completed result SHA-256 is
`3dfbc332123ff2c202af36e02e69d6be38ae4231364a127638564c8e0c2445f3`.
The screen evaluated all 18 declared profile/optical/density rows. Both optical
configurations have the same material density-20 result:

| Integrated zero-loss MRC profile | Density 10 | Density 20 | Density 30 | Exact | Within 10% |
|---|---:|---:|---:|---:|---:|
| Independent ideal | `5.24e-61` | `1.182919e-4` | `1.80e-61` | no | no |
| Low-correlation hardware bound | `5.24e-61` | `1.292880e-4` | `1.80e-61` | no | no |
| Correlated stress | `5.24e-61` | `2.154571e-4` | `1.80e-61` | no | no |

The independent-ideal result is the best observed pair. It improves the prior
2.0 ms headline-receiver density-20 lower bound by approximately 31.4%, but it
remains 18.29% above the exact target and 7.54% above the frozen `1.1e-4`
exploratory boundary. The hardware-primary profile improves the prior result
by 25.0% but remains 29.29% above target. Correlation sensitivity is material:
the correlated-stress result is more than twice the target.

The best-observed pair is the independent-ideal zero-loss receiver with the
wide optical configuration under the frozen tie-break, although the wide and
concentrated density-20 values are identical. Because the profile is an
optimistic sensitivity assumption rather than the hardware-primary model, any
continuation with it requires especially explicit claim boundaries.

## Decision

No exact or predeclared near-feasible pair exists. The result therefore does
not automatically authorize the pair-local joint contention frontier or PPO.
Widening the 10% margin after observing the result would be a post-hoc change,
so the frozen result is retained unchanged.

If work continues with the best observed pair, it must be recorded as a
user-directed exploratory override. The next computation would be the
pair-local joint contention and half-duplex frontier for that single frozen
pair—not PPO immediately—because the propagation-only value is optimistic and
the joint losses can only worsen it. Training after that characterization may
study policy behavior, but it cannot be described as meeting the `1e-4`
reliability constraint.
