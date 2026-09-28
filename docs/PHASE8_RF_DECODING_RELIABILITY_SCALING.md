# Phase 8 RF-decoding reliability-scaling diagnostic

Status: protocol and executor implemented; result not yet executed

Date frozen: 2026-09-28

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
