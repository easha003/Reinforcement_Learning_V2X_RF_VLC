# Phase 8 receive-diversity execution

Status: complete; all receive profiles fail the propagation-only necessary
condition, so no joint cell or PPO run is authorized

Date completed: 2026-09-28

## Stage 1: propagation-only necessary condition

For every frozen receive profile and both frozen optical configurations, the
runner replays the nine validation windows with RF collision and receiver
half-duplex loss removed from the risk calculation. On a usable row it chooses
the lowest-risk allowed action. On an unusable row it preserves the deployed
contract and forces `DUP-4`.

For RF decoding failure `p_rf`, VLC failure `p_vlc`, and an action reserving
`n` RF attempts, the optimistic action risk is

```text
p_rf^n                         RF-only
p_vlc                          VLC-only
p_rf^n * p_vlc                 duplicated RF/VLC
```

A receive profile survives only if at least one *single* optical
configuration meets `1e-4` at every required density. Therefore, a screened
profile cannot be rescued by RF capacity or sensing changes: those mechanisms
can reduce contention but cannot improve on the zero-contention bound already
used by the screen. Passing is only a necessary condition and is not a
feasibility verdict.

## Stage 2: certificate-aware joint frontier

Every surviving profile is evaluated on all 36 source cells:

- three full-carrier RF-capacity levels;
- three sensing bands;
- two optical configurations; and
- the contract and diagnostic fallback views.

The runner invokes the existing pair-local joint optimizer and its asymmetric
certificate rule. It does not introduce a second optimizer. A realizable
candidate at or below the budget is feasible; a certified lower bound above
the budget or an exact failing optimum is infeasible; an unresolved gap is
inconclusive.

Only the predeclared low-correlation, 3.5-dB-loss headline receive profile can
authorize later PPO training, and only when one RF-capacity/optical design is
feasible under every sensing band in the contract view. SISO, ideal, and stress
profiles remain controls or sensitivity evidence.

## Reproducibility and safety

Run the structural check without evaluating a frame:

```bash
.venv/bin/python scripts/run_receive_diversity_frontier.py
```

Run or resume the bounded validation execution:

```bash
.venv/bin/python scripts/run_receive_diversity_frontier.py --execute
```

The runner writes an atomic `.progress.json` artifact after the propagation
screen and after every joint cell. A restart accepts only the exact declaration
hash, source-frontier hash, window-source hash, complete frozen screen, and an
ordered prefix of the screened joint grid. `--restart` discards reuse of an
existing matching progress artifact. The held-out test split, learned actor,
checkpoint, and PPO trainer are not used.

The structural dry-run resolves ten receive profiles, nine validation windows,
180 physical profile/point instances, and 360 pre-screen cells while evaluating
zero frames. Heavy execution was monitored at the project-wide one-hour
interval.

## Completed result

The validation run completed in approximately 2 hours 50 minutes. It screened
all ten receive profiles and found zero survivors. The final artifact is:

```text
artifacts/evaluations/phase8_receive_diversity_system_feasibility_frontier.json
SHA-256 eb5de3183f3022389d7a564976a502d2cbb853cbb9d9230ddcec37492dbd8879
```

Every two-branch profile meets the necessary condition at density 10 and fails
it at densities 20 and 30. The headline low-correlation, 3.5-dB-loss profile
produces:

| Density | Propagation-only mean lower bound | Budget multiple | Verdict |
|---:|---:|---:|---|
| 10 | `5.241469e-61` | `5.241469e-57` | passes screen |
| 20 | `6.464405e-4` | `6.464405` | fails screen |
| 30 | `2.895417e-4` | `2.895417` | fails screen |

The concentrated optical configuration produces the same density-20 and
density-30 headline bounds; its density-10 difference is immaterial because
both values are far below budget. Across the complete screen, the optimistic
oracle selects only `RF-4` and `DUP-4`.

The best declared sensitivity result is still insufficient: the minimum
density-20 bound is `5.602346e-4` (correlated-stress, zero-loss) and the minimum
density-30 bound is `1.547163e-4` (independent, zero-loss). The finite frozen
sample must not be interpreted as evidence that high branch correlation is
generally beneficial; these non-headline profiles bound model sensitivity.

Because no profile survives, the second-stage grid is empty by construction:
zero of the 360 pre-screen cells are executed. This is not an inconclusive
joint-search result. It is a certified necessary-condition failure before
contention, sensing, RF capacity, or joint optimization can help. Training is
not authorized, and the test split remains unopened.

## Consequence

The declared two-branch receive-diversity intervention cannot establish
`1e-4` feasibility for the current system contract. Increasing capacity or
improving sensing alone cannot repair a lower bound that already removes RF
access losses. The next analysis should decompose the density-20 and
density-30 propagation tails by action, usability/fallback status, RF
propagation class, and VLC availability before freezing another physical
intervention. PPO remains blocked.
