# Phase 8 MacBook Pro M2 Training Profile

Date: 2026-09-24

## Purpose

This profile measures whether the current trace-backed primal-dual PPO path is
practical on the development Mac before choosing full-run hardware. It reports
performance only. It is not evidence of convergence, reliability feasibility,
or policy quality.

## Host and workload

The measurements ran on a MacBook Pro `Mac14,7` with an Apple M2, four
performance plus four efficiency cores, and 8 GB of unified memory. The host
used macOS 26.4.1, Python 3.12.3, PyTorch 2.14.0, CPU execution, four PyTorch
intra-op threads, and eight inter-op threads.

Every repeat used configuration hash
`69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`,
policy seed `1001`, and training trace `synthetic-d10-train-000`. Each measured
iteration processed 20 consecutive frames:

| Workload quantity | Value |
|---|---:|
| Environment transitions | 2,961 |
| Rollout transitions | 2,807 |
| PPO learning rows | 2,666 |
| PPO epochs | 10 |
| Optimizer minibatch updates | 30 |
| Optimizer row exposures | 26,660 |

The three independent processes produced the identical checkpoint SHA-256
`5688d4c9c9da1907982abcaac19e5605f0337617f5ce6c0484750879bf865693`,
so timing variation did not alter the sampled trajectory or update result.

## Reproduction

The canonical command is:

```bash
.venv/bin/hybrid-v2x-rl training profile \
  --project-root . \
  --output artifacts/logs/phase8-profile-d10-seed1001 \
  --trace-id synthetic-d10-train-000 \
  --max-frames 20 \
  --policy-seed 1001
```

The generated directory contains the underlying training `report.json`,
`metrics.jsonl`, complete `checkpoint.pt`, and performance `profile.json`.
Generated measurements remain outside Git.

## Repeated result

| Repeat | Total time | Environment throughput | End-to-end throughput | Estimated core hours/seed | Peak RSS |
|---:|---:|---:|---:|---:|---:|
| 1 (canonical) | 8.70 s | 450.5 transitions/s | 340.4 transitions/s | 6.90 h | 421.0 MiB |
| 2 | 12.07 s | 305.9 transitions/s | 245.4 transitions/s | 10.23 h | 313.5 MiB |
| 3 | 8.12 s | 483.2 transitions/s | 364.5 transitions/s | 6.43 h | 410.0 MiB |
| Median | 8.70 s | 450.5 transitions/s | 340.4 transitions/s | 6.90 h | — |

The observed CPU-speed range is material, so the wall-clock result must be
treated as an estimate rather than a deadline. The conservative maximum
uninstrumented peak RSS across the repeats was 421.0 MiB. RSS is the process
high-water mark and includes Python, PyArrow, and native PyTorch allocations;
its increment can be understated when process startup established an earlier
peak.

## Stage breakdown

The canonical repeat attributed 8.697 of its 8.699 seconds to explicit stages:

| Stage | Time | Fraction of total |
|---|---:|---:|
| Authoritative environment rollout | 6.573 s | 75.56% |
| Model/setup initialization | 1.311 s | 15.07% |
| Lifecycle alignment, GAE, and batch preparation | 0.403 s | 4.63% |
| PPO optimization | 0.375 s | 4.31% |
| Checkpoint and report publication | 0.030 s | 0.35% |
| Dual update and metric aggregation | 0.005 s | 0.06% |

The measured PPO path sustained 71,163 optimizer row exposures/s. The
environment sustained 450.5 transitions/s and is therefore the dominant
bottleneck. Moving only the small feed-forward actor and critics to a GPU would
not remove most of the present wall time.

## Call-level bottlenecks

A separate `cProfile` diagnostic repeated the same 20-frame workload. Profiling
overhead reduced its apparent environment throughput to 240.2 transitions/s,
so that run is used only to rank functions, not for wall-clock extrapolation.
Its main cumulative environment costs were:

1. Channel evaluation and per-pair state advancement, including matched random
   tape construction. Random-stream construction made 81,695 generator calls.
2. Joint-action ledger accounting and conservation audits. The profile made
   59,220 per-action accounting calls.
3. Causal actor-observation assembly and repeated spatial-neighbour queries.
4. Population-frame replay and Parquet vehicle iteration.

These are cumulative call relationships and overlap; their times must not be
summed. The actionable optimization order is to reduce repeated random-generator
construction, then remove avoidable repeated ledger/resource scans, then batch
spatial and observation work. Every optimization must preserve deterministic
fingerprints, matched tapes, and conservation checks.

## Full-run estimate and compute decision

The configured budget is 10 million environment transitions per policy seed,
with 32,768-transition rollouts, 1,024-row minibatches, 10 PPO epochs, and five
seeds. Using the canonical stage rates, which yield the median repeated core
estimate, gives:

| Estimate | Result |
|---|---:|
| Updates per seed | 306 |
| Environment time per seed | 6.17 h |
| Rollout preparation per seed | 0.38 h |
| PPO optimization per seed | 0.35 h |
| Core compute per seed | 6.90 h |
| Core compute for five seeds, serial | 34.48 h |
| Core time per configured update | 81.1 s |

The slower repeat raises the core estimate to 10.23 hours per seed and 51.13
hours for five serial seeds. These estimates exclude validation, evaluation,
queueing, and future full-trainer bookkeeping, and they assume the short-run
rates survive longer multi-density execution and thermal load.

The decision is to keep development and the first end-to-end joint-density run
on the M2 CPU. A GPU is not justified by this profile because PPO is only about
4% of canonical wall time. External compute becomes useful for parallel seed
turnaround, but CPU capacity and environment parallelism matter more than GPU
capacity at the current architecture. Before purchasing or scheduling GPU
time, profile again after environment optimization and after the full trainer
crosses trace and episode boundaries.

## Current boundary and next task

The 20-frame workload deliberately remains below the first internal
`max_duration` truncation. The smoke/profile runner still refuses to approximate
that bootstrap with reset state or zero. The next task is therefore to build the
versioned joint-density trainer while first materializing the final physical
observation required at internal truncation; only then should training proceed
across densities 10, 20, and 30.
