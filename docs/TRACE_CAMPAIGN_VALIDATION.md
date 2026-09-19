# Synthetic Mobility Trace Campaign Validation

## Status

The full synthetic Manhattan mobility campaign was generated and independently
verified on 2026-09-19. The trace-count, split, integrity, provenance, and
lifecycle prerequisites for Phase 2 are satisfied.

- Result: **PASS — Phase 2 is unblocked**
- Source commit: `3a35fba7ba9a98494a9b1b7bdd7f7e9625e5d6ec`
- Package version: `0.1.0`
- Project configuration hash:
  `69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`
- Mobility-scope configuration hash:
  `0f746ca381e25a73327540b37945c63a89e21c9139c5520abaa726c96132881b`
- Trace schema: `1.0.0`
- Environment contract: `1.0.0`

The campaign was generated from a clean `main` checkout with:

```bash
.venv/bin/hybrid-v2x-rl mobility generate-traces \
  --output artifacts \
  --train 3 \
  --validation 1 \
  --test 3 \
  --json
```

The generated corpus and `artifacts/campaign_manifest.json` remain ignored by
Git. The local corpus occupies approximately 5.7 GiB.

## Campaign matrix

| Split | Density 10 | Density 20 | Density 30 | Total |
|---|---:|---:|---:|---:|
| Train | 3 | 3 | 3 | 9 |
| Validation | 1 | 1 | 1 | 3 |
| Test | 3 | 3 | 3 | 9 |
| **Total** | **7** | **7** | **7** | **21** |

The generated trace IDs match the 21 immutable split assignments in
`configs/project/default.yaml` exactly. There are no missing, unexpected, or
cross-split trace IDs. All 21 mobility seeds are distinct.

## Gate 1 results

- Campaign result: `gate1_passed = true`
- Trace results: 21 of 21 passed
- Individual Gate 1 checks: 189 of 189 passed
- Insertion failures: 0
- Teleports: 0
- Unexplained teleports: 0
- Signals with state variation: 40 of 40 in every trace
- Signal transitions: 67,200 across the campaign
- Maximum density relative error: 0.0858%, below the 5% limit

| Target density (veh/lane-km) | Realized density (veh/lane-km) | Relative error |
|---:|---:|---:|
| 10 | 9.991428 | 0.0857% |
| 20 | 20.009643 | 0.0482% |
| 30 | 30.001071 | 0.0036% |

## Independent artifact-integrity audit

Every trace was opened with the repository's canonical
`MobilityTraceReader`. This independently verifies the artifact-manifest
schema, exact file inventory, file sizes, every SHA-256 digest, parquet
schemas, trace identity columns, finite timestamps, and persisted row counts.

| Verified quantity | Result |
|---|---:|
| Immutable trace artifacts | 21 |
| Vehicle-state rows | 282,240,000 |
| Signal-state rows | 756,000 |
| Vehicle parquet parts | 1,134 |
| First saved mobility time | 0.00 s |
| Last saved mobility time | 899.95 s |
| Temporary artifact directories left behind | 0 |

Every artifact records the same clean source commit, project configuration
hash, mobility-scope hash, package version, and trace schema. The configured
split membership and campaign-manifest membership reconcile exactly.

## Pair and lifecycle audit

The 21 `pairs.parquet` files contain 393,192 unique `(trace_id, pair_id)`
episodes. Of these, 385,928 have positive duration and can create active agents.
The remaining 7,264 episodes (1.847%) have zero duration and must be reconciled
as source records but must not emit an active-agent transition.

| Split | Persisted episodes | Usable episodes | Zero-duration episodes |
|---|---:|---:|---:|
| Train | 167,781 | 164,673 | 3,108 |
| Validation | 56,209 | 55,235 | 974 |
| Test | 169,202 | 166,020 | 3,182 |
| **Total** | **393,192** | **385,928** | **7,264** |

The positive-duration episodes reconcile with the lifecycle contract as
follows:

| Lifecycle outcome | Source end reasons | Count | Share of usable episodes |
|---|---|---:|---:|
| Natural termination | `outside_range_1s`, `route_diverged`, `vehicle_missing` | 266,767 | 69.124% |
| Internal 60 s truncation | `max_duration` | 105,787 | 27.411% |
| Physical trace-end truncation | `trace_end` | 13,374 | 3.465% |
| **Total** |  | **385,928** | **100.000%** |

The zero-duration records comprise 6,894 `vehicle_missing` endings and 370
`trace_end` endings. Phase 2 must retain these counts in source-to-cache
reconciliation while excluding them from policy transitions.

This audit confirms that the raw data contains explicit pair births (episode
starts), natural endings, internal episode truncations, and trace-end
truncations. Gap/reappearance behavior is represented by distinct pair episode
IDs and must receive fresh state in the frame layer, as required by the
environment contract.

Endpoint-sharing counts are intentionally not asserted here. They depend on
the exact decision-frame interval convention implemented by Phase 2. The
Phase 2 source-to-cache validation report must measure frames and agent
instances in which a physical vehicle is an endpoint of more than one active
pair; it must not silently replace the population with a disjoint matching.

## Verification commands

The following checks completed successfully:

```text
generator campaign summary                         PASS
campaign manifest split/density/lifecycle audit    PASS
MobilityTraceReader full 21-artifact audit         PASS
pytest test_episodes.py + test_mobility_trace_io.py
                                                    17 passed in 50.12 s
git status after generation                        clean
```

## Phase 2 handoff

The next implementation task is the chronological population-frame layer in
the Phase 2 work plan. It should start with the frame and lifecycle data
structures plus small asynchronous-birth fixtures, then add deterministic raw
trace replay and a source-to-cache reconciliation report. Phase 2 is complete
only when repeated replay produces identical frames and reconciles records,
decision frames, usable births, natural terminations, both truncation types,
densities, and endpoint-sharing counts.
