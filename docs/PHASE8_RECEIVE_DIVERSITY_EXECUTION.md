# Phase 8 receive-diversity execution

Status: two-stage executor implemented and structurally validated; bounded
validation execution in progress

Date started: 2026-09-28

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
zero frames. Heavy execution is monitored at the project-wide one-hour
interval.
