# Phase 8 longer-block RF frontier

Status: completed; no candidate meets the propagation-only necessary condition
at every density, so the joint frontier and PPO remain blocked

## Purpose

The propagation-tail decomposition found that effectively all residual
lower-bound risk comes from a handful of actor-usable rows in which RF is NLOS,
VLC is geometrically unavailable, and the optimistic oracle already selects
`RF-4`. This frontier tests the smallest physical change supported by the
proposed 10 ms service deadline: give each of the existing four RF attempts a
longer finite blocklength without changing the nine-action policy contract.

This is a necessary-condition experiment. It removes RF collision and receiver
half-duplex losses from the scored risk. A passing candidate must still undergo
the pair-local joint contention frontier before PPO can be authorized.

## Frozen candidates

The payload remains 300 B, the miss budget remains `1e-4`, the receiver remains
the headline two-branch low-correlation MRC profile with 3.5 dB secondary-chain
loss, and both frozen optical configurations are retained.

| Candidate | Modulation | Airtime per attempt | Channel uses | RF-4 airtime | Role |
|---|---|---:|---:|---:|---|
| `16qam-0p5ms-control` | 16QAM | 0.5 ms | 2,419 | 2 ms | Reuse the hash-verified current control |
| `qpsk-1p0ms-primary` | QPSK | 1.0 ms | 4,838 | 4 ms | Primary longer-block intervention |
| `qpsk-1p5ms-sensitivity` | QPSK | 1.5 ms | 7,257 | 6 ms | Sensitivity |
| `qpsk-2p0ms-sensitivity` | QPSK | 2.0 ms | 9,676 | 8 ms | Strongest bounded sensitivity |

The service deadline leaves 9.9 ms after the existing 0.1 ms predecision lead,
so all four RF attempts fit for every candidate. The 2.4 ms VLC leg runs
concurrently and therefore does not add to RF airtime.

The QPSK rows retain the configured nominal rate-1/3 coded-block check. The
resource-grid validator independently proves that the coded 348-byte block
fits the declared 24-RB grid at every duration. The finite-blocklength decoder
uses the complete declared channel-use count, so increasing airtime lowers the
information rate from the control row through the 2.0 ms row.

## Frozen selection and safety rules

The propagation screen replays only the nine existing validation windows at
densities 10, 20, and 30. For usable rows, a non-deployable oracle chooses the
lowest-risk allowed action; unusable rows retain the contract `DUP-4` fallback.
The 0.5 ms control is reused from the hash-verified receive-diversity result
because changing the deadline alone cannot affect propagation or decoding.

A candidate survives only when at least one single optical configuration meets
`1e-4` at all three densities. The shortest survivor is selected. No result may
add another airtime, modulation, retry count, receive profile, or optical
configuration.

The runner checkpoints after every completed candidate. It never loads a PPO
actor or checkpoint, never trains, and never opens the test split.

## Reproduction commands

Structural validation, with zero evaluated frames:

```bash
.venv/bin/python scripts/run_longer_block_frontier.py
```

Execute or resume the validation-only screen:

```bash
.venv/bin/python scripts/run_longer_block_frontier.py --execute
```

## Completed validation result

The declaration SHA-256 is
`3db9bceffd82a7251cff813499a94e5513c40c4edfd1dd0be2e9f197904239e6`.
The completed result is
`artifacts/evaluations/phase8_longer_block_rf_frontier.json`, SHA-256
`80433fc1733bfc5c25be183b0e853d15cb594f03570326c25be3ae5c302052b4`.

Both optical configurations produce the same material density-20 and
density-30 means:

| Candidate | Density 10 | Density 20 | Density 30 | Survives |
|---|---:|---:|---:|---|
| 0.5 ms 16QAM control | `5.24e-61` | `6.464405e-4` | `2.895417e-4` | no |
| 1.0 ms QPSK | `5.24e-61` | `4.340970e-4` | `1.192608e-4` | no |
| 1.5 ms QPSK | `5.24e-61` | `3.017842e-4` | `2.594365e-5` | no |
| 2.0 ms QPSK | `5.24e-61` | `1.723841e-4` | `5.51e-49` | no |

The concentrated receiver changes only the already negligible density-10
value. At density 30, 1.5 ms is sufficient and 2.0 ms drives the selected
finite-blocklength risk to the numerical floor. Density 20 remains limiting:
2.0 ms reduces its control risk by approximately 73.3%, but the resulting mean
is still 1.724 times the `1e-4` budget.

The frontier therefore has zero survivors. No pair-local contention cell runs,
because contention and half-duplex loss can only worsen this optimistic bound.
Training is not authorized and the test split remains unopened.

## Consequence

The declared 0.5--2.0 ms longer-block intervention is beneficial but
insufficient. The next bounded task should diagnose only the remaining
density-20 tail under the 2.0 ms QPSK candidate and determine whether any
deadline-edge blocklength up to 2.475 ms per attempt could close the remaining
1.724-fold gap. That diagnosis must be frozen before adding a candidate; the
completed grid must not be expanded after observing its result.
