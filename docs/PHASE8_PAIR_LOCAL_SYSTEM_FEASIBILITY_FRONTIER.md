# Phase 8 pair-local system-feasibility frontier

Status: complete; all 36 cells certified infeasible; training not authorized

Date frozen: 2026-09-27

Executor completed: 2026-09-28

Frontier executed: 2026-09-28

## Decision this protocol will support

The next experiment asks whether any bounded, physically interpretable system
configuration admits a realizable pair-local joint action whose mean
conditional miss risk is at most `1e-4` at densities 10, 20, and 30
vehicles/lane-km. It is a validation-set screening gate, not final statistical
evidence that the operational miss probability is `1e-4`.

PPO training remains blocked while this gate is open. The frontier does not
load an actor, does not resume the pre-migration seed-1001 checkpoint, and does
not open the test split. If the gate eventually authorizes training, the
learner must start with fresh model, optimizer, dual, normalization, and
rollout state under the pair-local RF environment.

## Frozen identity and evidence boundary

- Declaration schema:
  `hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-declaration.v1`
- Declaration:
  `configs/evaluation/pair_local_system_feasibility_frontier.yaml`
- Declaration SHA-256:
  `c6499afeb4d7ecf15c09bbe3006dd08183a4d3ce97c1e524d0e8ba93356c3cf2`
- Frozen window source:
  `artifacts/evaluations/phase8_state_regime_audit.json`
- Window-source SHA-256:
  `35b240c2fc58b46f02999d225663b6a91803509309cf098b4927e19aa6a7b2a7`
- Environment seed: `20260728`
- Policy-environment scope hash:
  `46fecd53689db6f2c9aa21b4314da610b0caa7e47444a4a9b01f5ba4a3b1cb29`
- Evidence: three validation windows of 16 frames at each density
- Test split opened: no
- Planned result:
  `artifacts/evaluations/phase8_pair_local_system_feasibility_frontier.json`

The declaration validator verifies the source artifact's digest, schema,
environment seed, scope hash, split, trace identity, window lengths, and exact
per-density coverage before execution is allowed. The runner uses
a fresh frozen identity normalizer. The oracle actor is absent; a historical
actor normalization state is neither needed nor permitted.

## Predeclared grid

The physical grid has 18 points:

```text
3 RF capacities x 3 sensing bands x 2 optical configurations
```

Each point has two fallback views, producing 36 evaluation cells. All cells
must be evaluated; results cannot expand, shrink, or redirect the grid.

| Axis | Frozen values | Interpretation |
|---|---|---|
| RF resource-pool capacity | 2, 4, 8 orthogonal subchannels over 200 slots, giving 400, 800, 1,600 candidate resources | 1x, 2x, and 4x shared resource-pool capacity; equivalent 10, 20, and 40 MHz system pools |
| RF sensing uncertainty | pessimistic `0.70`, nominal `0.85`, optimistic `0.95` | Declared uncertainty in reservation sensing, not a selectable system improvement |
| Optical receiver | named wide `60 deg` control and concentrated `30 deg` sensitivity | The concentrated layer changes receiver field of view only |
| Fallback view | actual `DUP-4` contract and all-rows-oracle-controlled diagnostic | Only the actual contract can authorize training |

The unchanged control is 2 subchannels, nominal sensing, the wide 60-degree
receiver, and the actual `DUP-4` fallback contract.

The RF-capacity axis adds orthogonal resource-pool capacity while holding the
per-link RF propagation and attempt PHY fixed. Its MHz values are equivalent
system-pool labels obtained by scaling the 10 MHz control; they are not a claim
that the existing RF link-budget configuration has already been changed to a
different waveform. The executor passes the declared capacity into
the collision and local-contention model explicitly and persists the effective
parameters.

The concentrated optical configuration is optimistic on alignment because
the synthetic Manhattan traces omit road curvature, vehicle pitch, and lane
changes. It can quantify link-budget sensitivity in this model, but a passing
30-degree cell cannot by itself establish real-world optical availability.

## Search and certificate boundary

Every nonempty frame uses the certificate-aware pair-local joint evaluator.
The complete joint assignment is enumerated only when its assignment space is
at most 100,000. Larger spaces use the deterministic multi-start search for at
most 16 iterations and retain both:

- a realizable candidate, which is an upper bound on minimum risk; and
- the certified optimistic zero-contention, zero-half-duplex lower bound.

For each density, the verdict is:

| Evidence | Density verdict |
|---|---|
| Realizable candidate mean risk `<= 1e-4` | feasible |
| Certified lower bound `> 1e-4` | infeasible |
| Exact optimum `> 1e-4` | infeasible |
| Candidate fails, lower bound passes, and search is not exact | inconclusive |

A cell is feasible only when every density is feasible. It is infeasible when
at least one density has proven infeasibility. Otherwise it is inconclusive.
The evaluator must never turn a failed non-exact candidate into an
infeasibility claim.

## Training-authorization rule

A physical configuration authorizes a fresh learner only if its actual
`DUP-4` contract cell is feasible at all three densities under every sensing
band. Nominal-only or optimistic-only success is insufficient. The
all-rows-oracle-controlled view is diagnostic and can never authorize
training, even if it passes.

If several physical configurations meet this robust gate, report every
Pareto-minimal point over two declared changes: added RF subchannels and the
optical profile. Do not invent a scalar hardware cost to force a single
winner. A result outside this bounded grid requires a new predeclared
experiment.

## Executor and artifact contract

`scripts/run_system_feasibility_frontier.py` is safe by default: without an
execution flag it performs only the structural dry run and writes no result.

```bash
PYTHONPATH=src .venv/bin/python scripts/run_system_feasibility_frontier.py
```

The explicit `--execute` flag is required to evaluate channel frames and write
the declared result. Execution injects the selected sensing band and RF
subchannel count into both the physical rollout façade and the authoritative
pair-local response model. These are opt-in evaluation parameters; existing
training, baseline, and rollout callers retain their unchanged nominal
defaults.

During execution, every completed cell is atomically persisted beside the
final output as `phase8_pair_local_system_feasibility_frontier.progress.json`.
A repeated `--execute` command resumes only when that artifact is a validated
ordered prefix of the frozen grid and its declaration, evidence, verdicts, and
effective physical parameters still match. `--restart` explicitly replaces
the prefix. A partial progress file is never accepted as the final result.

The `all-rows-oracle-controlled` view is also opt-in. Only a policy declaring
non-deployable oracle truth may control causally unusable rows, and only with a
frozen normalizer. The actual contract view continues to force `DUP-4` on
those rows.

The result schema is
`hybrid-rf-vlc-rl.pair-local-system-feasibility-frontier-result.v1`. A final
artifact cannot be constructed from a partial or reordered grid, missing
density, duplicate window, changed window length, or inconsistent candidate
budget flag. It records every density verdict, cell verdict, robust design
verdict, current-system nominal and robust conclusions, every robust-feasible
design, and the Pareto-minimal subset. Diagnostic fallback success is hard
coded not to authorize training.

## Structural dry-run evidence

The artifact-backed dry run passed on 2026-09-28. It verified:

- the declaration and frozen window-source digests;
- all nine validation windows and their trace-manifest compatibility;
- 18 physical points and exactly 36 ordered evaluation cells;
- effective resource pools of 400, 800, and 1,600 candidates;
- effective sensing reliabilities of `0.70`, `0.85`, and `0.95`;
- both 60-degree and 30-degree optical configurations and calibration assets;
- the actual `DUP-4` fallback contract;
- a fresh, frozen, zero-statistic normalizer; and
- actor absent, checkpoint absent, test split closed, and frontier not
  executed.

This dry run evaluated no channel frame and therefore produced no feasibility
answer.

## Full execution evidence

The opt-in CPU execution completed all 36 cells and wrote:

`artifacts/evaluations/phase8_pair_local_system_feasibility_frontier.json`

Result SHA-256:
`b5d44445b5a7c61cdb9d178793f26fc6a39ca75e84e7d4406f2963f9bf138944`

The final artifact validates the frozen declaration and source hashes, marks
the frontier complete, and records the following fail-closed outcomes:

| Evidence level | Infeasible | Feasible | Inconclusive |
|---|---:|---:|---:|
| Density evaluations | 108 | 0 | 0 |
| Evaluation cells | 36 | 0 | 0 |
| Actual `DUP-4` contract cells | 18 | 0 | 0 |
| All-rows diagnostic cells | 18 | 0 | 0 |
| Physical designs | 6 | 0 | 0 |

The current-system nominal and robust verdicts are both `infeasible`. The
robust-feasible and Pareto-minimal design lists are empty. Training is not
authorized, and the test split was not opened.

### Current-system control

For the unchanged 2-subchannel, wide-60-degree, nominal-sensing,
actual-`DUP-4` system:

| Density (vehicles/lane-km) | Realizable candidate risk | Certified lower bound | Lower-bound / budget |
|---:|---:|---:|---:|
| 10 | `2.901441e-3` | `2.403681e-4` | `2.40x` |
| 20 | `2.773961e-2` | `9.552398e-4` | `9.55x` |
| 30 | `3.827212e-2` | `7.013525e-4` | `7.01x` |

The failed control is therefore not an artifact of PPO behavior: even its
optimistic certified floor exceeds `1e-4` at every density.

### Effect of the bounded physical changes

Added RF capacity and the concentrated optical receiver substantially reduce
the best realizable risk, but they do not cross the target. The nominal
actual-contract candidates are:

| RF pool | Optical receiver | d10 candidate | d20 candidate | d30 candidate | Verdict |
|---|---|---:|---:|---:|---|
| 2 subchannels | wide 60 deg | `2.901441e-3` | `2.773961e-2` | `3.827212e-2` | infeasible |
| 2 subchannels | concentrated 30 deg | `2.322190e-3` | `2.258730e-2` | `2.876072e-2` | infeasible |
| 4 subchannels | wide 60 deg | `5.236172e-4` | `5.994590e-3` | `1.091183e-2` | infeasible |
| 4 subchannels | concentrated 30 deg | `4.629647e-4` | `5.094968e-3` | `8.234037e-3` | infeasible |
| 8 subchannels | wide 60 deg | `2.619144e-4` | `1.373642e-3` | `1.889953e-3` | infeasible |
| 8 subchannels | concentrated 30 deg | `2.553347e-4` | `1.260855e-3` | `1.502506e-3` | infeasible |

The best actual-contract candidates anywhere in the grid occur at 8
subchannels, the concentrated receiver, and optimistic sensing. They are
`2.536417e-4`, `1.229059e-3`, and `1.430628e-3` at densities 10, 20, and 30.
Even this non-robust best case misses the target by `2.54x`, `12.29x`, and
`14.31x`, respectively.

### Why the result is a feasibility conclusion

The diagnostic view removes forced fallback and permits oracle control of all
rows. Its certified zero-contention, zero-half-duplex lower bounds are
invariant across all 18 physical points:

| Density (vehicles/lane-km) | Diagnostic certified lower bound | Budget multiple |
|---:|---:|---:|
| 10 | `2.271923e-4` | `2.27x` |
| 20 | `6.895343e-4` | `6.90x` |
| 30 | `3.897709e-4` | `3.90x` |

These bounds exceed the target before RF contention, receiver half duplex, or
forced fallback can add risk. Consequently, incomplete enumeration of large
joint action spaces cannot change the verdict: every density is certified
infeasible, and no open candidate/lower-bound gap remains.

## What has and has not been learned

The declared bounded system is infeasible at `1e-4` on the frozen validation
windows. Increasing the shared RF pool from 2 to 8 subchannels, varying the
declared sensing band, concentrating the optical receiver from 60 to 30
degrees, and removing the forced-fallback limitation are insufficient. More
PPO training, different PPO hyperparameters, or additional policy seeds cannot
overcome the certified floor under these physical assumptions.

This is not a claim that hybrid RF/VLC reliability at `1e-4` is universally
impossible. It is a scoped result for the current trace evidence, propagation
models, action contract, and frozen intervention grid. It is also not final
test-set statistical evidence; the held-out test split remains unopened.

The immediate next task is to quantify the per-link reliability improvement
needed to move the diagnostic floor below `1e-4`, then predeclare a bounded
architecture-level frontier containing mechanisms that can produce that
improvement. Candidate mechanisms must alter RF link reliability or add
genuinely independent optical/RF diversity; capacity, sensing, or PPO-only
changes are not sufficient. No PPO training begins unless an actual-contract
design passes the robust validation gate.
