# Phase 8 full-carrier RF allocation contract

Status: allocation correction complete; corrected frontier structurally frozen;
execution and training remain blocked

Date frozen: 2026-09-28

## Decision

One RF transmission attempt occupies the complete configured 10 MHz, 24-RB
carrier for one 0.5-ms slot. The collision model's existing `subchannels` field
is retained for schema compatibility, but its physical unit is now one complete
carrier allocation rather than one 12-RB half-carrier.

The corrected capacity mapping is:

| System bandwidth | Full-carrier resources per slot | Selection-window slots | Candidate resources |
|---:|---:|---:|---:|
| 10 MHz | 1 | 200 | 200 |
| 20 MHz | 2 | 200 | 400 |
| 40 MHz | 4 | 200 | 800 |

This makes collision accounting agree with the active link calculation: every
attempt uses 24 RB, provides 9,676.8 coded bits after overhead, and supplies
2,419 finite-blocklength channel uses for the 2,784-bit information block.

## Fading consequence

The previous implementation also spread repeated attempts across frequency
points inside the same 10 MHz carrier. That would retain the same hidden
half-carrier assumption after correcting only the collision count.

The rollout now derives attempt centres from the number of complete carriers:

- the 10-MHz baseline repeats every attempt at the same carrier centre;
- a 20-MHz pool alternates between two adjacent 10-MHz carrier centres; and
- a 40-MHz pool can use four adjacent 10-MHz carrier centres.

Therefore, the baseline receives no invented intra-carrier hopping diversity.
Frequency diversity becomes an explicit benefit of additional spectrum. The
model still draws fresh access and decoding uniforms per attempt; correlated
collisions remain an optimistic-model limitation that the feasibility report
must disclose.

## Consequences for the headline pool

At the campaign's pinned median neighbour counts, three committed 0.5-ms RF
attempts produce the following full-carrier resource demand:

| Density (vehicles/lane-km) | Median neighbours | Resource demand | Deliverable? |
|---:|---:|---:|---:|
| 10 | 47 | `0.705` | yes |
| 20 | 108 | `1.620` | no |
| 30 | 162 | `2.430` | no |

The nominal analytical access-failure probabilities are now approximately
4.92%, 9.18%, and 12.80% at densities 10, 20, and 30. This is worse than the
superseded half-carrier accounting, as expected. It is a correction of the
physical baseline rather than a reliability intervention.

The earlier equilibrium regression is consequently not deliverable at the
10-MHz headline point: after its own allocation is fed back into the pool, its
resource demand remains above one. Its numerical miss estimate cannot
authorize training.

## Frozen declarations and evidence

The completed 2/4/8-resource frontier declaration remains unchanged at
`configs/evaluation/pair_local_system_feasibility_frontier.yaml` so its
historical result and the abstract RF-decoding scaling diagnostic retain their
exact provenance. Its former 10/20/40-MHz interpretation is superseded.

The corrected declaration is
`configs/evaluation/full_carrier_system_feasibility_frontier.yaml`, SHA-256:

```text
60718bcdead081ad77b794ef64daadb054fb8c0fa2f052fbe4334f67a82d99c7
```

Its structural dry run passed with nine validation windows, 18 physical
points, and 36 evaluation cells. It evaluated no channel frames, used no actor
or checkpoint, opened no test data, and cannot authorize training.

The updated machine-readable allocation audit is
`artifacts/evaluations/phase8_rf_full_carrier_profile_audit.json`, SHA-256:

```text
a2a298fd0333617f7c32a2e1e140032d719404ff63a97940cc529b6bd8048f48
```

It reports `allocation_contract_consistent: true`. The complete physical
profile remains unfreezable only because the declared RF calibration artifact
is absent; that missing evidence now belongs to the receive-diversity design,
not to an unresolved allocation unit.

## Scope and next task

This task changes collision capacity and attempt-frequency correlation. It
does not regenerate mobility traces, alter the RF link budget, execute the
corrected frontier, train PPO, or open the test split.

The receive-diversity experiment is now frozen in
`PHASE8_RECEIVE_DIVERSITY_FRONTIER_DECLARATION.md`. It explicitly declares the
antenna count, combining and channel-state-information assumptions, branch
correlation, secondary-chain implementation loss, and sensitivity bands over
this corrected 1/2/4 full-carrier resource axis. That physical branch/combiner
model is now implemented and verified in
`PHASE8_RECEIVE_DIVERSITY_PHYSICAL_MODEL.md`; the next task is the frozen
screening and certificate-aware execution sequence.
