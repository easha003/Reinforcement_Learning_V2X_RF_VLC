# Phase 8 propagation-tail decomposition

Status: completed validation-only diagnosis; training remains unauthorized and
the test split remains unopened.

## Question

The receive-diversity frontier proved that its optimistic propagation-only
lower bound still exceeds the `1e-4` miss budget at densities 20 and 30. This
diagnosis asks which rows produce that lower bound before another physical
intervention is declared. It partitions the frozen headline receive profile by:

- lower-bound action;
- actor usability versus contract fallback;
- RF propagation class; and
- VLC geometric availability.

The diagnostic declaration is
`configs/evaluation/propagation_tail_decomposition.yaml`, SHA-256
`5584323a53d68ceb572c843bbe7b23f2445f1d5a79f9ad8a03ac73deba6f609a`.
It fixes the two failed densities, both previously declared optical
configurations, the existing validation windows, and the headline
`rx2-mrc__low-correlation-hardware-bound__short-cable-loss` profile. It does
not use a trained actor or checkpoint.

## Reproduction

The bounded replay exactly reproduces the four source-screen rows:

| Optical configuration | Density | Transitions | Mean lower bound | Budget multiple |
|---|---:|---:|---:|---:|
| wide 60 degrees | 20 | 23,204 | `6.464405e-4` | 6.464405 |
| wide 60 degrees | 30 | 38,484 | `2.895417e-4` | 2.895417 |
| concentrated 30 degrees | 20 | 23,204 | `6.464405e-4` | 6.464405 |
| concentrated 30 degrees | 30 | 38,484 | `2.895417e-4` | 2.895417 |

Every reproduction difference is zero at the persisted precision. The result
is stored in
`artifacts/evaluations/phase8_propagation_tail_decomposition.json`, SHA-256
`fc7141c10877f841d24e16a18adbf306a7e273ceb525c9ffed21fdd7ac087e0f`.

## Density 20

Only 16 of 23,204 transitions have selected propagation-only risk above
`1e-8`; they carry effectively 100% of the total risk. Fifteen exceed the
headline `1e-4` threshold. All 16 share one decomposition cell:

| Action | Actor status | RF state | VLC state | Rows | Mean selected risk | Risk share |
|---|---|---|---|---:|---:|---:|
| `RF-4` | actor usable | NLOS | occluded and outside FOV | 16 | `9.375004e-1` | 100% |

The corresponding mean per-attempt RF decoding failure is `9.406220e-1`.
The failure is therefore not caused by the contract fallback or by the learned
policy being unable to observe the row. VLC is geometrically absent and the
best optimistic action exhausts the existing four RF repetitions.

## Density 30

Only 12 of 38,484 transitions exceed `1e-8`, and those 12 carry effectively
100% of the total risk. The complete material risk lies in two NLOS,
VLC-unavailable cells:

| Action | Actor status | RF state | VLC state | Cell rows | Mean selected risk | Risk share |
|---|---|---|---|---:|---:|---:|
| `RF-4` | actor usable | NLOS | beam not aimed | 31 | `2.305295e-1` | 64.14% |
| `RF-4` | actor usable | NLOS | occluded and beam not aimed | 34 | `1.175384e-1` | 35.86% |

The cells contain additional negligible-risk rows; only 12 rows form the
material tail. Actor fallback, LOS, NLOSv, and VLC-available rows contribute
only numerical-floor risk.

## Interpretation

The failure is a rare, sharply concentrated shared-geometry tail:

1. RF is NLOS.
2. VLC is geometrically unavailable.
3. The actor observation is usable.
4. The optimistic oracle already selects `RF-4`.

Changing the optical configuration from the wide to the concentrated profile
does not change the limiting rows or either density mean. Better sensing or
additional RF pool capacity also cannot change this propagation-only result.
Another PPO run cannot repair it because the lower-bound oracle already uses
the strongest action in the current contract.

## Next physical intervention

The next experiment should use the proposed 10 ms service deadline to improve
the decoding strength of each of the existing four RF attempts. It should
predeclare a bounded per-attempt airtime/blocklength frontier while retaining
the nine-action policy interface:

| Candidate | Airtime per RF attempt | RF-4 airtime | Purpose |
|---|---:|---:|---|
| control | 0.5 ms | 2.0 ms | reproduce the current lower bound |
| primary | 1.0 ms | 4.0 ms | first QPSK longer-block candidate |
| sensitivity | 1.5 ms | 6.0 ms | stronger bounded candidate |
| sensitivity | 2.0 ms | 8.0 ms | strongest candidate within 10 ms |

The propagation-only screen must select the shortest candidate that meets
`1e-4` at every density. Only surviving candidates may enter the pair-local
contention and half-duplex frontier. Longer airtime increases RF-pool demand,
so a propagation pass is necessary but not sufficient.

Increasing the action contract beyond four retransmissions is not selected at
this stage. It would enlarge the PPO output and matched-tape schemas while
adding contention. Time-spaced attempts remain a secondary option if longer
coding passes propagation but the resource-coupled frontier fails.

No training is authorized by this diagnosis.
