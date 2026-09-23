# Phase 7 lifecycle-aware critic bootstrapping

Date: 2026-09-23

## Purpose

The Phase 5 environment already preserves natural termination, internal
time-limit truncation, physical trace-end truncation, and bootstrap validity as
separate pair-aligned fields. `agents.lifecycle_bootstrap` now converts that
contract into the current and next reward/cost critic values consumed by GAE.
It never reconstructs lifecycle from one combined `done` flag.

## Stable-identity selection

Critic predictions are carried in `PairedCriticValues`, which binds reward and
cost values to canonical stable pair IDs. For each transition:

| Lifecycle | Next-value source | Value bootstrap | GAE continuation |
|---|---|---:|---:|
| Continuing pair | Same stable ID in normal next population | 1 | 1 |
| Natural termination | Exact zero | 0 | 0 |
| Internal truncation with next physical row | Separately evaluated `final_observation` | 1 | 0 |
| Trace-end/no-next-row truncation | Exact zero | 0 | 0 |

The normal next population may contain newborn IDs, but their values cannot be
used for any prior transition. Every continuing ID must remain present, every
finalized ID must be absent, and final-observation critic values must cover
exactly the bootstrap-valid IDs. These checks prevent a reset observation or a
reused array position from contaminating the prior episode's target.

## Torch boundary

The adapter converts the environment's immutable NumPy lifecycle arrays to
boolean Torch tensors on the critic-value device. Current and selected next
reward/cost values are detached and copied into a `LifecycleBootstrapBatch`.
All value tensors share shape, dtype, and device; zero-bootstrap rows are
validated to contain exact zeros.

The adapter accepts already evaluated final-observation critic values from the
separate reward and cost networks implemented in `PHASE7_PPO_UPDATES.md`. This
keeps value-source selection independently testable: the later rollout
collector must evaluate `info["final_observation"]` before reset and submit
predictions under the same stable IDs.

## Verification

Tests exercise all four lifecycle cases together, prove that newborn next rows
are ignored, verify that internal-truncation GAE uses a final value without
recursing into the reset episode, and reject missing continuing IDs, repeated
final IDs, incomplete final values, identity drift, malformed critic batches,
and incompatible tensor contracts. Empty population frames remain valid.
