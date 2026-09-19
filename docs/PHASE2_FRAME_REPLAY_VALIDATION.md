# Phase 2 Population-Frame Replay Validation

## Status

Phase 2 completed on 2026-09-19. Every configured synthetic trace passed the
population-frame integrity gate, and compact replay-cache format `1.0.0` is
frozen.

- Result: **PASS — Phase 3 is unblocked**
- Trace campaign: 21 of 21 configured traces
- Environment contract: `1.0.0`
- Frame-cache format: `1.0.0`
- Package version: `0.1.0`
- Project configuration hash:
  `69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`
- Decision period: 0.1 s

The machine-readable local report is
`artifacts/frame_campaign_validation.json`. It and the generated cache corpus
are reproducible artifacts and remain outside Git.

## Campaign coverage

The validator used the immutable split assignments in
`configs/project/default.yaml`; it did not discover or relabel traces from the
filesystem.

| Split | Density 10 | Density 20 | Density 30 | Total |
|---|---:|---:|---:|---:|
| Train | 3 | 3 | 3 | 9 |
| Validation | 1 | 1 | 1 | 3 |
| Test | 3 | 3 | 3 | 9 |
| **Total** | **7** | **7** | **7** | **21** |

All artifacts passed manifest inventory, file-size, SHA-256, Parquet-schema,
trace-identity, finite-time, configuration-hash, and exact input-reference
checks before their frame counts were accepted.

## Source-to-frame reconciliation

| Quantity | Campaign total |
|---|---:|
| Source vehicle rows | 282,240,000 |
| Source signal rows | 756,000 |
| Source pair rows | 393,192 |
| Zero-duration source pairs | 7,264 |
| Positive-duration decision episodes | 385,928 |
| Positive-duration pairs with no decision point | 0 |
| Decision frames | 189,000 |
| Nonempty decision frames | 188,811 |
| Pair-frame instances | 124,226,108 |
| Births | 385,928 |
| Continuing pair instances | 123,840,180 |
| Natural terminations | 266,767 |
| Internal `max_duration` truncations | 105,787 |
| Physical trace-end truncations | 13,374 |

The lifecycle identities reconcile exactly:

```text
births = decision episodes = 385,928
natural terminations + internal truncations + trace-end truncations
  = 266,767 + 105,787 + 13,374
  = 385,928
pair-frame instances = births + continuing instances
  = 385,928 + 123,840,180
  = 124,226,108
```

Zero-duration rows account for 1.847% of source pair rows. They remain in the
source audit but correctly produce no policy transition. No positive-duration
episode fell between decision grid points in this campaign.

| Density (veh/lane-km) | Traces | Source pairs | Decision episodes | Pair-frame instances | Natural ends | Internal truncations | Trace-end truncations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7 | 71,182 | 69,499 | 15,433,339 | 62,420 | 5,386 | 1,693 |
| 20 | 7 | 138,628 | 135,987 | 41,879,167 | 100,325 | 31,188 | 4,474 |
| 30 | 7 | 183,382 | 180,442 | 66,913,602 | 104,022 | 69,213 | 7,207 |
| **Total** | **21** | **393,192** | **385,928** | **124,226,108** | **266,767** | **105,787** | **13,374** |

## Simultaneous-population evidence

Endpoint sharing is substantial and therefore cannot be represented by a
one-pair-at-a-time environment or a disjoint matching:

- 188,811 frames contained active pairs and endpoint overlap.
- 120,024,774 pair-frame instances (96.618%) involved a pair whose transmitter
  or receiver simultaneously served another pair.
- 214,456,688 endpoint assignments participated in multiplicity above one.
- Maximum simultaneous endpoint multiplicity was 8.

This result validates the Phase 2 design choice: the environment must retain
the full concurrent population and account for shared resources only after the
joint action is selected.

## Frozen compact-cache contract

Each trace produces one immutable `population_frame_cache` artifact under
`artifacts/frame_caches/<trace_id>/`:

| File | Granularity | Purpose |
|---|---|---|
| `episodes.parquet` | One row per usable pair episode | Pair endpoints, source interval, first/last decision frame, end reason, termination kind, and bootstrap validity |
| `frames.parquet` | One row per decision frame | Population, lifecycle, and endpoint-overlap aggregate counts |
| `summary.json` | One row per trace | Format/version declaration and complete source-to-frame report |
| `manifest.json` | One row per artifact | SHA-256 inventory, configuration/code provenance, and exact source-trace manifest reference |

The 21 caches contain about 15 MiB (15,887,152 manifest-counted data bytes),
compared with approximately 5.7 GiB for the raw trace corpus. Vehicle states
remain in the immutable source traces and are streamed when the environment
needs geometry.

The cache contains only policy-independent structure. It explicitly excludes:

- policy actions;
- RF demand, airtime contention, and collisions;
- packet outcomes, rewards, and costs; and
- mean-field observations.

Those quantities must be recomputed after the simultaneous actions are known.
This prevents Phase 2 from freezing the population coupling that the RL policy
is intended to learn.

## Validation implementation

`PopulationFrameReader.validate_with_aggregates()` performs a column-selective
Arrow scan of every vehicle timestamp and ID at the decision clock. For every
frame it verifies chronological coverage, unique vehicle IDs, presence of both
endpoints of every active pair, exact births/finals, and endpoint multiplicity.
The ordinary training replay API remains full fidelity and constructs complete
vehicle records and spatial indexes on demand.

The cache reader independently verifies exact schemas and metadata, artifact
digests, source references, row identities, chronological frame indices, and
all aggregate sums. Campaign execution is resumable: an existing cache is
reused only after that full verification succeeds.

## Reproduction

```bash
.venv/bin/hybrid-v2x-rl frames validate-campaign \
  --project-root /path/to/Hybrid_RF_VLC_RL
```

The command exits nonzero on the first integrity or reconciliation failure and
writes `artifacts/frame_campaign_validation.json` atomically only after the
full configured catalog passes.

Focused implementation checks at the validation point:

```text
test_population_frames.py                           14 passed
Ruff (CLI, mean_field package, Phase 2 tests)       passed
mypy --strict (CLI and mean_field package)          passed
21-trace source-to-cache campaign                   passed
full repository suite                               853 passed, 2 expected skips
```

## Phase 3 handoff

Phase 3 can consume the simultaneous pair population and must implement one
authoritative mapping from each of the nine frozen actions to VLC activation,
RF attempts, duplication, and activation cost. The first Phase 3 gate is exact
per-agent-to-population accounting, including zero RF demand for VLC-only
actions.
