# Phase 8 RF-decoding reliability-scaling diagnostic

Status: complete; `2x` is the first declared passing factor; training remains blocked

Date frozen: 2026-09-28

> **Scope note:** the source frontier's former bandwidth labels were
> superseded by `PHASE8_FULL_CARRIER_ALLOCATION_CONTRACT.md`. This diagnostic's
> zero-contention propagation calculation remains an abstract necessary
> condition because it changes only action-independent RF decoding failure; it
> is not evidence that the historical collision-resource mapping is physical.

## Purpose

The completed system-feasibility frontier proved that the current bounded
architecture cannot meet the `1e-4` validation target. This diagnostic asks a
narrower question before any new physical architecture is chosen:

> By what factor would the action-independent RF decoding-failure probability
> need to improve before the optimistic propagation lower bound crosses
> `1e-4` at densities 10, 20, and 30 vehicles/lane-km?

This is a necessary-condition sensitivity analysis. It is not a deployable
channel model, does not run PPO, does not search joint actions, and cannot
authorize training.

## Frozen evidence and intervention

The declaration is
`configs/evaluation/rf_decoding_reliability_scaling.yaml`. It reuses the nine
frozen validation windows and environment seed from the completed frontier and
anchors factor `1x` to its 8-subchannel, concentrated-30-degree, nominal-sensing
all-rows diagnostic cell. The source result SHA-256 is
`b5d44445b5a7c61cdb9d178793f26fc6a39ca75e84e7d4406f2963f9bf138944`.
The frozen scaling declaration SHA-256 is
`6cad049867a09da50381defe9921bce15099591266f65c707a0a0e8b0d265fc0`.

The predeclared RF decoding-failure divisors are:

```text
1x, 2x, 4x, 8x, 16x, 32x
```

For each pair and factor, the diagnostic divides only the RF propagation
decoding-failure probability by that factor. VLC risk, the nine-action set,
fallback behavior, mobility traces, validation windows, and random tapes stay
unchanged. RF contention and receiver half-duplex risk are set to their
optimistic zero-risk limits when the analytical lower bound is calculated.

Both fallback views are reported:

- the actual contract, which forces `DUP-4` on causally unusable rows; and
- the all-rows-oracle-controlled diagnostic.

## Decision rule

For each view, report the smallest declared factor whose mean optimistic
conditional miss lower bound is at most `1e-4` at all three densities. The
preceding failed factor and first passing factor form the interval for the next
physical architecture design.

A passing factor is necessary but not sufficient. Contention, half duplex,
forced fallback, implementation constraints, and model uncertainty can raise a
realizable design above the bound. Therefore:

- training authorization is always false;
- the held-out test split remains closed; and
- a physical, predeclared architecture frontier must follow.

## Execution

The command is dry-run only unless `--execute` is supplied:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_reliability_scaling_diagnostic.py
```

The eventual result path is
`artifacts/evaluations/phase8_rf_decoding_reliability_scaling.json`.

The structural dry run passed on 2026-09-28. It resolved all nine validation
windows, six ordered improvement factors, both fallback views, and 36 expected
density rows without evaluating a channel frame. It also verified the source
frontier declaration and result digests, factor-one anchor identity, closed
test split, absent actor and checkpoint, disabled joint-action search, and
disabled training authorization.

## Result

The diagnostic completed on 2026-09-28. The result SHA-256 is
`30dbf82aa5a5d056f180fcb18345560238f5eba3c0a583b2e245418a193521ff`.
Factor `1x` reproduced the completed frontier's all-rows lower bound exactly,
so the source baseline is reconciled before interpreting any scaled result.

The actual fallback contract and the all-rows diagnostic produced the same
lower bounds throughout this experiment:

| RF decoding-failure divisor | d10 lower bound | d20 lower bound | d30 lower bound | All densities pass? |
|---:|---:|---:|---:|---|
| `1x` | `2.271923e-4` | `6.895343e-4` | `3.897709e-4` | no |
| `2x` | `1.419952e-5` | `4.309589e-5` | `2.436068e-5` | yes |
| `4x` | `8.874701e-7` | `2.693493e-6` | `1.522543e-6` | yes |
| `8x` | `5.546688e-8` | `1.683433e-7` | `9.515891e-8` | yes |
| `16x` | `3.466680e-9` | `1.052146e-8` | `5.947432e-9` | yes |
| `32x` | `2.166675e-10` | `6.575911e-10` | `3.717145e-10` | yes |

The views agree because every bound-minimizing action uses four RF attempts.
Usable rows choose either `RF-4` or `DUP-4`, and the actual fallback already
forces `DUP-4` on unusable rows. Scaling RF decoding failure by a divisor
`g` therefore scales every selected packet-level propagation term by
`1 / g^4`; the intervention does not change the minimizing actions.

The continuous factor at which each density's unchanged four-attempt bound
would exactly equal the budget is:

| Density (vehicles/lane-km) | Exact analytical divisor |
|---:|---:|
| 10 | `1.227717` |
| 20 | `1.620462` |
| 30 | `1.405084` |

Density 20 is limiting. Thus `1.620462x` is the optimistic mathematical
threshold for this diagnostic and `2x` is the first predeclared passing point.
At `2x`, the largest bound is `4.309589e-5`, or `0.431x` the budget.

## Interpretation and next task

This result corrects the earlier coarse intuition that the per-attempt RF
failure probability itself might need an order-of-magnitude improvement. Four
attempts compound the improvement, so the optimistic propagation floor crosses
the target between `1x` and `2x`.

It does not show that a realizable system passes. The completed frontier's
candidate risks also include contention and receiver half-duplex exposure,
which this diagnostic deliberately removes. The `2x` divisor is an abstract
sensitivity parameter, not yet a claim about transmit power, coding, MCS, or a
specific radio implementation.

The immediate next task is to map a decoding-failure reduction of at least
`1.620462x`, with engineering margin represented by the `2x` point, to one or
more physically interpretable RF configurations. Those configurations must be
frozen before a new certificate-aware joint feasibility frontier is run. PPO
training and the held-out test split remain blocked.
