# VLC Diversity Frontier Side-Project Plan

## Project identity

- Repository: `Reinforcement_Learning_V2X_RF_VLC`
- Development branch: `vlc-diversity-frontier`
- Branch point: `0b73220` (`Fix A/B structural trace precondition`)
- Status at branch creation: planning only; no VLC physics or experiment result
  has been changed.
- Main-line relationship: this is a side project. GitHub `main` remains the
  corrected pair-local PPO path and is not advanced by side-project work until
  a reviewed merge decision is made.

## Research objective

Determine whether a minimal, physically defensible VLC diversity design can
make VLC available and reliable in the RF-hard states that currently determine
the 300 B / 10 ms / `1e-4` packet-miss failure, while retaining the corrected
200 m pair-local RF contention and endpoint half-duplex model.

The target is a packet miss probability no greater than `1e-4` at each of
densities 10, 20, and 30 vehicles per lane-kilometer on the frozen validation
windows. This corresponds to at least 99.99% modeled packet delivery
reliability. A campaign average alone is not sufficient.

## Baseline evidence

The side project starts from these completed results and does not rerun them
unless a dependency changes:

1. The global-versus-local accounting A/B shows that pair-local accounting
   lowers mean conditional risk by 15.72%--31.50% at 3 ms and
   15.85%--29.35% at 10 ms across RF-involving fixed-action cells. The local
   model is the only live environment for this project.
2. In the 10 ms pair-local A/B, adding VLC to RF-4 lowers fixed-action miss
   risk by 41.1%, 72.9%, and 81.7% at densities 10, 20, and 30. VLC is already
   complementary, especially where RF contention is highest.
3. The remaining propagation tail is concentrated in RF-NLOS rows where the
   single VLC path is occluded, outside the receiver FOV, or not illuminated by
   the transmitter beam. A single 30-degree receiver does not change the
   limiting tail.
4. The selected 10 ms 2.0 ms-QPSK plus independent-ideal zero-loss two-branch
   RF-MRC propagation screen reaches a worst-density mean of `1.182919e-4`,
   but its nominal pair-local joint candidate reaches `1.303173e-2`. Optical
   availability and RF-load offload must therefore be evaluated together.

Frozen source artifacts include:

- `docs/PHASE8_GLOBAL_LOCAL_ACCOUNTING_AB.md`
- `docs/PHASE8_PROPAGATION_TAIL_DECOMPOSITION.md`
- `docs/PHASE8_COMBINED_RECEIVER_BLOCK_FRONTIER.md`
- `docs/PHASE8_EXPLORATORY_JOINT_OVERRIDE.md`
- `artifacts/evaluations/phase8_global_local_accounting_ab.json`, SHA-256
  `554f5a71966a00124c8ffa929b176f8912b4615cfae3d578fd3fb63decec107a`

## Claim boundary

This remains a synthetic-trace, analytical-channel study. No configuration is
called standards-compliant, calibrated, or real-world feasible unless the
supporting source or measurement exists. In particular:

- no uncalibrated residual optical floor may turn blockage into success;
- optical branches sharing the same geometry may not be assumed independent;
- a wider coverage cone may not retain a narrow cone's concentrator gain;
- receiver diversity may not create extra transmitter power;
- repeated VLC transmissions may not redraw persistent blockage inside 10 ms;
- clear-night success may not be generalized to day, weather, road grade,
  pitch, lane changes, or hardware without explicit sensitivity evidence;
- PPO cannot authorize a physical `1e-4` claim.

The validation split is used for bounded design selection. The test split
remains closed until the complete system and policy are frozen.

## Frozen baseline profile

Every side-project candidate is compared with this unchanged control:

| Component | Control |
|---|---|
| Service | 300 B / 10 ms / miss budget `1e-4` |
| RF attempt | 2.0 ms QPSK, at most four attempts |
| RF receiver | Two-branch independent-ideal zero-loss MRC sensitivity |
| RF pool | Four full-carrier resources per slot, 800 candidates |
| RF interaction | 200 m pair-local contention and endpoint half-duplex |
| Sensing | Nominal |
| VLC | One direct headlamp-to-photodiode path, 60-degree semi-angle |
| VLC PHY | OOK, 4.5 Mbit/s, 2.4 ms, rate 1/4, clear night |
| Blockage | Complete direct-path blockage; no residual floor |
| Fallback | `DUP-4` |
| Windows | Existing nine frozen validation windows |
| Environment seed | `20260728` |

## Candidate hierarchy

Candidate values must be frozen before their outcomes are observed. The first
frontier should be staged rather than a full combinatorial product.

### Stage A — Optical coding duration

Evaluate a small bounded duration grid using the same optical hardware:

- 2.4 ms control;
- approximately 4.8 ms;
- approximately 7.2 ms;
- no more than 9.6 ms.

The implementation must specify whether additional time represents one longer
codeword, repetition combining, or another coded construction. Independent
decoding draws cannot be invented for a path whose error mechanism is shared.
This stage can improve clear-path decoding but cannot repair geometric outage.

### Stage B — Angular receiver diversity

Evaluate a small bank of narrow-FOV photodiodes with predeclared boresight
orientations and selection or physically defined electrical combining. The
union of their acceptance regions may increase coverage, but each branch must
use its own FOV-consistent concentrator gain. This is preferred to giving one
wide receiver the gain of a narrow receiver.

### Stage C — Spatial receiver diversity

Evaluate two physically named receiver placements, such as a rear-facing
control branch and a higher or laterally separated branch. Geometry must decide
branch-specific FOV, beam coverage, and blockage. Report at least shared- and
partially correlated blockage sensitivities; use an independent-blockage case
only as an explicitly optimistic bound.

### Stage D — Transmitter coverage sensitivity

Only if receiver-side candidates leave beam-not-aimed rows material, evaluate
a separately costed dual-headlamp, auxiliary beam, or steering profile. Beam
coverage and radiant intensity must come from a declared or calibrated pattern.
This stage may require a new action-cost interpretation and is not silently
folded into the existing one-VLC-activation cost.

### Stage E — Minimal combined candidate

Combine only the smallest survivors from Stages A--D. Do not search arbitrary
combinations after observing results. Any combined candidate must state which
failure mechanisms are genuinely independent and which remain shared.

## Work phases

### VD0 — Branch and provenance freeze

- [x] Create the `vlc-diversity-frontier` branch from corrected pair-local
  commit `0b73220`.
- [x] Add this side-project plan and the PI progress summary.
- [ ] Freeze a machine-readable declaration containing source paths, SHA-256
  values, profile IDs, candidate order, validation windows, seed, gates, and
  output paths.
- [ ] Add a dry-run command that validates the declaration without evaluating
  a channel frame.

Deliverable: `configs/evaluation/vlc_diversity_frontier.yaml` and a structural
validation report.

### VD1 — VLC failure and complementarity audit

- [ ] Reproduce the current 10 ms VLC, RF-n, and DUP-n risks exactly.
- [ ] Partition VLC misses by complete occlusion, receiver FOV, transmitter
  beam coverage, and clear-path decoding.
- [ ] Report every partition by density, actor usability, RF propagation class,
  and local-load regime.
- [ ] Compute RF-risk-weighted VLC miss probability, because VLC matters most
  on rows where RF is likely to fail.
- [ ] Measure the current complementarity gain
  `1 - risk(DUP-n) / risk(RF-n)` for every retry count and density.
- [ ] Identify the minimum set of material rows each candidate mechanism must
  rescue to move below `1e-4`.

Gate: the audit must reconcile exactly with the existing A/B and propagation
tail artifacts before a new physical candidate is evaluated.

Deliverable: `docs/VLC_DIVERSITY_BASELINE_AUDIT.md` and a versioned JSON
artifact.

### VD2 — VLC diversity contract

- [ ] Define branch identity, placement, orientation, FOV, concentrator gain,
  timing, activation cost, and combining rule.
- [ ] Separate shared geometry, branch-specific geometry, common optical power,
  and branch-specific receiver noise.
- [ ] Define deterministic, identity-addressed random streams without changing
  existing RF draws.
- [ ] Define blockage-correlation profiles and prohibit undeclared independence.
- [ ] Decide whether receiver diversity remains one VLC activation. Treat any
  additional transmitter as a separately accounted resource.
- [ ] Preserve the nine-action contract for receiver-only diversity if it can
  be done without changing action meaning; otherwise version the action
  contract explicitly.

Deliverable: `docs/VLC_DIVERSITY_CONTRACT_V1.md`.

### VD3 — Physical implementation and invariant tests

- [ ] Implement branch-resolved geometry and optical channel evaluation.
- [ ] Implement the declared combining rule.
- [ ] Connect the selected VLC profile to rollout, packet outcomes,
  counterfactuals, and feasibility evaluation through one authoritative path.
- [ ] Prove single-branch equivalence with the current implementation.
- [ ] Test zero/fully correlated and declared partial-correlation limits.
- [ ] Test FOV union, no double-counted gain, complete blockage, beam coverage,
  coding duration, deadline accounting, and action-cost conservation.
- [ ] Prove that actor observations contain no current channel truth.

Gate: unit, integration, deterministic replay, and matched-tape tests pass.

### VD4 — Staged propagation/availability screen

- [ ] Execute Stage A controls and coding-duration candidates.
- [ ] Execute Stage B angular-diversity candidates.
- [ ] Execute Stage C spatial/blockage-diversity candidates.
- [ ] Execute Stage D only if transmitter misalignment remains material.
- [ ] Record geometric availability, clear-path PER, RF-risk-weighted optical
  miss, hybrid propagation lower bound, and exact budget multiple by density.
- [ ] Select the smallest candidate that reaches the frozen propagation gate at
  every density; retain negative results without expanding the grid.

Gate: a candidate proceeds only if its hybrid propagation-only mean is no
greater than `1e-4` at every density. An explicit exploratory override may
continue a near candidate, but it cannot relabel it feasible.

### VD5 — Pair-local joint feasibility frontier

- [ ] Evaluate only Stage-VD4 survivors with the current pair-local RF model.
- [ ] Preserve nominal sensing as the headline and declared sensing
  sensitivities as stress cases.
- [ ] Report realizable candidates, certified lower bounds, optimality gaps,
  local utilization, collision, endpoint half-duplex, and optical offload.
- [ ] Quantify whether improved VLC reduces RF load as well as direct packet
  risk.
- [ ] Apply the per-density exact-target gate without averaging away a failed
  density.

Gate: exact feasibility requires a realizable candidate at or below `1e-4` for
all three densities. A lower bound below target is not itself a pass.

### VD6 — Bounded PPO smoke campaign

- [ ] Begin only after VD5 selects a profile or the user records a separate
  exploratory override.
- [ ] Freeze the final config and start a fresh PPO state; no pre-migration or
  different-physics checkpoint may be resumed.
- [ ] Verify stable optimization, checkpoint/resume, action availability,
  finite duals, and causal observations.
- [ ] Report actions conditioned on local RF load, VLC availability, RF state,
  endpoint half-duplex exposure, density, and state regime.
- [ ] Compare PPO with the selected pair-local joint oracle and measure action
  regret, feasible action mass, resource cost, and optical offload.
- [ ] Fail the behavioral gate if one action dominates unrelated regimes or if
  `DUP-4` becomes a universal response.

Gate: the smoke run establishes learnability and behavior only. It does not
upgrade the physical feasibility verdict.

### VD7 — Final evidence and PI/paper package

- [ ] Freeze the selected or rejected design decision.
- [ ] Produce per-density tables, confidence/statistical limitations,
  mechanism decomposition, resource costs, and sensitivity results.
- [ ] State separately whether optical propagation, complete joint feasibility,
  and PPO behavior pass their respective gates.
- [ ] Update the PI summary with the final side-project result.
- [ ] Decide whether to merge into `main`, retain the branch as a negative
  result, or continue as a separate paper extension.

## Required result metrics

Every candidate report must include:

- standalone VLC conditional miss probability;
- VLC geometric-outage fraction and category counts;
- clear-path decoding miss probability;
- RF-risk-weighted VLC miss probability;
- `RF-n` versus `DUP-n` complementarity gain;
- pair-local RF attempts, utilization, collision, and endpoint half-duplex;
- realizable hybrid risk and certified lower bound by density;
- activation/resource cost;
- branch-correlation sensitivity;
- actor/checkpoint/training/test-split usage declarations.

## Stop rules

Stop the frontier without adding candidates when any of the following holds:

1. No frozen candidate passes the propagation gate and no new mechanism is
   justified by the measured failure decomposition.
2. A candidate improves only already-successful clear paths while geometric
   outage continues to carry the target failure.
3. A candidate requires an uncalibrated residual floor, false branch
   independence, inconsistent FOV gain, or an unaccounted transmitter cost.
4. The joint certified lower bound remains above `1e-4` after the declared
   optical candidates.
5. Reaching the target would require adapting the grid after observing results
   rather than freezing a new, scientifically motivated experiment.

Negative results remain publishable evidence about the boundary of direct-path
V-VLC complementarity.

## Execution and monitoring policy

- Structural checks and unit tests run before simulations.
- Every heavy screen is resumable and checkpoints after each declared cell.
- CPU-heavy simulations are monitored at one-hour intervals unless the user
  explicitly changes that interval.
- Intermediate checks do not open the test split or start PPO.
- Generated artifacts remain local and are summarized in version-controlled
  documentation with content hashes.

## Immediate next task

Implement VD0's machine-readable declaration and dry-run validator, then begin
VD1 by reproducing and decomposing the current 10 ms VLC contribution. No new
VLC candidate should be implemented before that baseline audit is frozen.
