# Phase 8 deadline-edge blocklength threshold

Status: declared and implemented; validation execution pending

## Purpose

The 2.0 ms QPSK candidate reduced the density-20 optimistic
propagation-only lower bound to `1.723841e-4`, but it did not reach the
`1e-4` miss budget. This experiment diagnoses that residual tail and asks a
narrow necessary-condition question: with the same frozen mobility, fading,
VLC geometry, MRC receiver, action space, and four-attempt contract, does any
continuous QPSK blocklength up to the deadline edge reach the budget?

This is not a system-feasibility result. Collision and receiver half-duplex
loss remain removed, and the runner opens validation data only. No PPO actor,
checkpoint, training run, or test split is used.

## Frozen threshold

The 10 ms deadline retains the existing 0.1 ms predecision lead. Four RF
attempts therefore have at most 9.9 ms, or 2.475 ms per attempt. The screen is
frozen from 9,676 channel uses at 2.0 ms through 11,975 channel uses at 2.475
ms. It reports five declared anchors and uses exact integer-channel-use
bisection to find the first passing value, if one exists.

Each density-20 channel row is replayed once per frozen optical profile. The
runner stores its instantaneous MRC SNR, VLC failure probability, propagation
class, actor usability, available actions, and action costs. Every anchor and
bisection point is then rescored from those same rows. This paired design
prevents mobility or fading differences from being mistaken for a blocklength
effect. The 2.0 ms score must reproduce the hash-verified source result before
the threshold is accepted.

## Full-slot realizability boundary

The current RF slot is 0.5 ms. The source candidate occupies four full slots
per attempt. The next full-slot candidate would occupy five slots, or 2.5 ms
per attempt, and `RF-4` would require 10 ms of RF airtime. That exceeds the
9.9 ms available after the predecision lead.

Consequently, a theoretical threshold between 2.0 and 2.475 ms would show
that the continuous finite-blocklength model can reach the target, but it
would not authorize a candidate on the current full-slot grid. Such a result
requires a separately predeclared timing, numerology, or minislot intervention
before any joint contention frontier. If even 2.475 ms fails, the next
intervention must change some other physical reliability mechanism.

## Reproduction commands

Structural validation, with zero evaluated frames:

```bash
.venv/bin/python scripts/run_deadline_edge_threshold.py
```

Execute or resume the validation-only replay:

```bash
.venv/bin/python scripts/run_deadline_edge_threshold.py --execute
```

The executor writes a checkpoint after each optical profile. The completed
result is declared at
`artifacts/evaluations/phase8_deadline_edge_blocklength_threshold.json`.

## Frozen decision rule

- If the 2.475 ms boundary fails, select a different physical reliability
  intervention.
- If the continuous threshold passes but its full-slot rounding misses the
  deadline, predeclare a timing/numerology intervention.
- Only a threshold that also maps to the current full-slot grid can authorize
  the pair-local joint contention frontier.
- This experiment never authorizes PPO directly and never opens the test
  split.
