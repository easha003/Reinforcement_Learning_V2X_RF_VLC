# Phase 8 deadline-edge blocklength threshold

Status: completed; the continuous deadline boundary fails, so the current
full-slot grid, joint frontier, and PPO remain blocked

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

## Completed validation result

The declaration SHA-256 is
`c0d257353d5582ec9958727e2806a44b6fcd97e4068a0cf5687253b402f3607e`.
The completed result SHA-256 is
`6527d511a62b73da7f1a70617b471365ad03f131522409a00e2d96df60913a62`.
Both optical profiles exactly reproduce the 2.0 ms source mean and produce
identical threshold results over 23,204 density-20 transitions:

| Airtime per attempt | Channel uses | Mean optimistic lower bound | Budget multiple |
|---:|---:|---:|---:|
| 2.000 ms | 9,676 | `1.723841e-4` | 1.724 |
| 2.125 ms | 10,281 | `1.723833e-4` | 1.724 |
| 2.250 ms | 10,886 | `1.713567e-4` | 1.714 |
| 2.375 ms | 11,491 | `1.433501e-4` | 1.434 |
| 2.475 ms | 11,975 | `1.294405e-4` | 1.294 |

The 2.475 ms edge lowers the 2.0 ms mean by approximately 24.9%, but it still
exceeds the miss budget by approximately 29.4%. There is therefore no passing
integer channel-use threshold inside the continuous deadline interval.

The control decomposition also sharpens the diagnosis. Four of 23,204 rows
(`0.0172384%`) contribute more than 99.999999995% of the complete risk sum.
They lie in the actor-usable, `RF-4`, RF-NLOS group for which VLC is
geometrically unavailable because it is occluded and outside the receiver
field of view. The 2.0 ms aggregate risk sum is effectively 4.0; extending to
the deadline edge reduces it only to 3.00354. Optical-profile width cannot
repair these rows because VLC is unavailable in both configurations.

## Consequence and next bounded step

The failed continuous boundary rules out slot granularity alone as the fix:
even a hypothetical fractional-slot implementation at 2.475 ms would miss the
target. A fifth current slot and extra full-length retransmission also do not
fit the deadline. The next minimal experiment should therefore combine the
grid-realizable 2.0 ms QPSK candidate with the already-declared integrated
zero-loss two-branch MRC receive profiles. It should retain all three frozen
branch-correlation levels rather than selecting the empirically best prior
profile after observing its result.

That combined receiver-loss/blocklength experiment remains a
propagation-only necessary-condition screen. It must be predeclared, and no
joint contention frontier or PPO training may begin unless a frozen profile
passes the `1e-4` target.
