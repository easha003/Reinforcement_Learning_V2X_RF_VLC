# Phase 8 pair-local system-feasibility frontier

Status: protocol predeclared and validated; evaluator and results not yet
implemented

Date frozen: 2026-09-27

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
per-density coverage before execution is allowed. The planned runner must use
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
different waveform. The future evaluator must pass the declared capacity into
the collision and local-contention model explicitly and persist the effective
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

## What has and has not been learned

This document contains no frontier result. It does not show that `1e-4` is
feasible, infeasible, or learnable. It freezes the hypotheses, evidence,
interventions, exactness rules, and decision rule before seeing the answers.

The immediate next task is to implement the versioned frontier executor and
artifact contract, then run a cheap structural dry run. Only after those pass
should the full CPU-heavy frontier begin. During that heavy run, Codex checks
progress at the project's requested one-hour interval.
