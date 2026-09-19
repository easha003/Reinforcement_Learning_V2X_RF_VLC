# Phase 0 Repository Baseline

## Status

This document records the reproducible starting point for the Hybrid RF/VLC
reinforcement-learning extension. The commit containing this document is the
initial source baseline; generated traces, caches, checkpoints, evaluations,
logs, and figures are deliberately excluded.

- Date: 2026-09-18
- Branch: `main`
- Remote: `https://github.com/easha003/Reinforcement_Learning_V2X_RF_VLC.git`
- Platform: macOS 26.4.1, Apple `arm64`
- Python: 3.12.3 in repository-local `.venv`

## Installed development environment

| Package | Version |
|---|---:|
| NumPy | 2.4.6 |
| PyArrow | 19.0.1 |
| Shapely | 2.1.2 |
| Gymnasium | 1.3.0 |
| PyTorch | 2.14.0 |
| Pydantic | 2.13.5 |
| pytest | 9.1.1 |
| Ruff | 0.16.8 |
| mypy | 1.20.2 |

NumPy is constrained to `<2.5` because NumPy 2.5 type stubs require a Python
3.12 type-checking target, while this project intentionally supports Python
3.11 and 3.12 and runs mypy with `python_version = "3.11"`.

PyTorch reports that MPS support is built, but MPS is unavailable in the
current Codex execution environment. CPU execution is therefore the verified
baseline. This does not block environment development or correctness testing.

## Validation results

Commands were run from the repository root using the local environment.

| Check | Result |
|---|---|
| `hybrid-v2x-rl doctor --project-root . --json` | Pass |
| `python -m pip check` | Pass; no broken requirements |
| `ruff check src tests scripts` | Pass |
| `pytest -q -p no:cacheprovider` | 813 passed, 27 skipped in 504.07 seconds |
| `mypy src` | Known debt: 32 errors in 11 files |

The 27 skips are expected at this point:

- Tests that require the generated mobility campaign
- Tests that require training caches or evaluation artifacts
- One provenance test that requires the initial Git commit

No inherited unit test failed.

## Known static-typing debt

Strict mypy currently reports 32 inherited errors across these areas:

- RF/VLC scalar return typing
- PyArrow's missing inline type information
- Episode/cache collection annotations
- Mobility control-flow narrowing
- RF fading optional-state narrowing
- Rollout, assembly, perception, and campaign annotations

These errors are recorded rather than silently ignored. They do not represent
test failures, but they should be cleared before the new mean-field environment
and agent APIs make the affected interfaces larger.

## Trace and artifact policy

The canonical local artifact paths remain:

```text
artifacts/traces/
artifacts/caches/
artifacts/checkpoints/
artifacts/evaluations/
artifacts/policies/
artifacts/logs/
artifacts/figures/
```

These paths are ignored by Git. The small, versioned VLC calibration manifests
under `artifacts/calibration/` are tracked because they are required simulation
inputs rather than experiment outputs.

The final raw mobility campaign is not present in this checkout. The completed
paper repository retains episode-oriented caches, but those caches must not be
used as the new RL environment because they do not preserve the required
simultaneous population frames and action-dependent RF contention.

After the baseline commit is clean, the intended full campaign command is:

```bash
.venv/bin/hybrid-v2x-rl mobility generate-traces \
  --output artifacts \
  --train 3 \
  --validation 1 \
  --test 3
```

This produces 21 traces across densities 10, 20, and 30 vehicles per
lane-kilometer using disjoint train, validation, and test replicates. The
campaign is expected to require approximately 6 GiB and must remain untracked.

## Post-commit verification

The committed baseline was verified before any full trace campaign was
started:

- Baseline commit: `9d33b1d`
- Remote branch: `origin/main`
- Provenance tests: 6 passed
- Trace-generation smoke test: Gate 1 passed for train, validation, and test
- Smoke density: 10 vehicles per lane-kilometer
- Smoke duration: 60 seconds per trace after a 5-second warm-up
- Smoke configuration hash:
  `b852419b64d5f08273399abdb422a7f622ba24aa74c41e01582be563195caad0`

| Split | Pair episodes | Usable episodes | Realized density |
|---|---:|---:|---:|
| Train | 805 | 767 | 9.9914 |
| Validation | 803 | 767 | 9.9914 |
| Test | 804 | 764 | 9.9914 |

All 27 Gate-1 checks passed across the three smoke traces. The temporary
31 MiB smoke campaign was removed after verification; it is not a training
dataset and was never added to Git.

## Current modeling boundary

The inherited code still exposes the original three actions: `RF`, `VLC`, and
`DUP`. The planned nine-action interface, including RF attempt levels 1–4, is
not part of this baseline. It will be frozen in the Phase 1 environment
contract and implemented through the population action model in Phase 3.

## Next gate

1. Freeze the Phase 1 environment contract while the full campaign is being
   prepared.
2. Generate and verify all 21 raw mobility traces from a clean commit.
3. Clear the inherited mypy debt before adding new environment interfaces.
4. Begin Phase 2 only after trace counts, splits, and lifecycle semantics are
   confirmed against the generated artifacts.
