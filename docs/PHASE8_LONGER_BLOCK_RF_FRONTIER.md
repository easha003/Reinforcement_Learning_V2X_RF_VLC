# Phase 8 longer-block RF frontier

Status: frozen declaration and resumable propagation-only executor implemented;
validation execution pending

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

The final result will be written to
`artifacts/evaluations/phase8_longer_block_rf_frontier.json`. Until that result
exists, no longer-block candidate has passed and neither the joint frontier nor
PPO is authorized.
