# Phase 8 Single-Trace Smoke Training

Date: 2026-09-24

## Purpose

The first Phase 8 trace-backed run verifies that the production environment and
the Phase 7 learner compose into one executable training iteration. It is an
integration result, not evidence of convergence, policy superiority, or
reliability feasibility.

## Command

```bash
.venv/bin/hybrid-v2x-rl training smoke \
  --project-root . \
  --output artifacts/logs/phase8-smoke-d10-seed1001 \
  --trace-id synthetic-d10-train-000 \
  --max-frames 5 \
  --policy-seed 1001
```

The output directory is generated evidence and remains outside Git. It
contains `report.json`, append-only `metrics.jsonl`, and the complete
`checkpoint.pt` training state.

## Path exercised

The smoke runner uses the existing authoritative population rollout rather
than a simplified training simulator. Every joint decision passes through:

- causal actor observation assembly and train-only running normalization;
- the 37-column decentralized actor and separate 78-column centralized critic
  inputs;
- hardware action masks and the configured no-observation fallback;
- simultaneous joint-action accounting and the shared RF pool;
- RF/VLC physical evaluation, matched packet tapes, sampled packet outcomes,
  and conditional miss risk;
- stable pair lifecycle masks and next-frame critic bootstrapping;
- reward/cost GAE, minibatched clipped PPO, density-specific dual ascent,
  metrics logging, and complete checkpoint publication.

The first four frames form the reported rollout and the fifth supplies the
next values for the fourth. The fifth frame is processed to leave every
environment owner at a complete frame boundary, but it is excluded from PPO
and dual samples. The first frame has no causal track, so its 141 fallback
packets remain in the density constraint estimate while being excluded from
the policy loss, as required by the contract.

## Result

The canonical run used configuration hash
`69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`.

| Quantity | Result |
|---|---:|
| Training trace | `synthetic-d10-train-000` |
| Density | 10 veh/lane-km |
| Frames processed | 5 |
| Environment transitions | 705 |
| Rollout transitions | 564 |
| PPO learning rows | 423 |
| Fallback rows in the rollout | 141 |
| PPO optimizer steps | 10 |
| Optimizer row exposures | 4,230 |
| Mean resource reward | -3.386525 |
| Mean conditional miss risk | 0.049201 |
| Sampled miss rate | 0.056738 |
| Initial curriculum budget | 0.01 |
| Density-10 dual, before → after | 0 → 0.00196005 |
| Approximate KL | 0.003125 |
| Clip fraction | 0.028605 |
| Entropy | 2.166247 |

The dual increase is directionally correct because the rollout's conditional
risk exceeded the initial curriculum budget. Density-20 and density-30 duals
remained exactly unchanged because those densities had no samples. All nine
actions were selected in the optimization rollout, and the 141 initial
fallback rows used `DUP-4`.

The initial reward and cost explained variances were `0.00669` and `-0.07854`.
Those values are diagnostics from an untrained network after one short
rollout; they are not a failure criterion for this integration test. They are
also why this result must not be described as learned performance.

The published checkpoint is 365,116 bytes with SHA-256
`bd85424d618f5a2e5e5b80332746e4403d1cb55b8b859976bae74d3ac529ad2f`.
It contains the actor, both critics, all optimizers, density duals,
normalization state, counters, global random states, the named PCG64 training
stream, and the named policy/minibatch Torch generators.

## Verification and current boundary

Automated tests run the same training path on a compact immutable trace,
restore the resulting checkpoint, verify exact named generator coverage,
repeat the run to compare metrics and environment fingerprints, and reject
reuse of a nonempty evidence directory.

The smoke runner currently rejects a rollout containing an internal
`max_duration` truncation. Such a row needs a separately materialized final
physical observation for value inference; using the reset population or zero
would violate the established lifecycle contract. The five-frame canonical
run contains no such boundary. Full-length training must implement that final
observation path before it is allowed to cross an internal truncation.

## Next task

Profile environment throughput, optimizer time, memory, and dominant
bottlenecks on the MacBook Pro M2 before choosing full-run rollout sizes or
external compute.
