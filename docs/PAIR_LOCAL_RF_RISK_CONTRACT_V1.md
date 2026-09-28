# Pair-local RF attempt-risk contract v1

Status: frozen, implemented, tested, and live in the atomic rollout path

Date frozen: 2026-09-27

## Purpose

This contract composes the three failure mechanisms affecting one selected RF
attempt without sampling its outcome:

- pair-local collision probability from the focal transmitter domain;
- half-duplex exposure from the focal receiver's physical transmit activity;
  and
- policy-independent RF propagation and decoding failure.

The result replaces the legacy combination of global collision risk and
population-mean half-duplex exposure. Packet outcomes, delayed feedback,
baselines, and the feasibility evaluator now consume it through the shared
atomic assembler.

## Identity and input agreement

For each RF-using pair `i`, the response and endpoint-schedule contracts must
refer to the same trace, frame index, frame time, canonical pair population,
and selected reservation count `n_i`. The propagation mapping must cover the
RF-using pair subset exactly: missing, extra, or invalid rows fail closed.

VLC-only pairs remain in the frame identity and have zero RF reservations, but
they do not receive a fabricated RF propagation result or attempt-risk row.
Empty frames and frames containing only VLC selections therefore contain no RF
risks.

The propagation row retains the simulator-truth state, pathloss, shadowing,
fading gain, SINR, and decoding-failure probability for diagnostics. These
values are not added to the causal actor observation.

## Per-attempt composition

Let:

```text
p_col_i = local per-attempt collision probability
p_hd_i  = transmit duty cycle of the focal physical receiver
p_dec_i = propagation-conditioned decoding-failure probability
```

The access and total per-attempt failure probabilities are:

```text
p_access_i = 1 - (1 - p_col_i) * (1 - p_hd_i)
p_rf_i     = 1 - (1 - p_access_i) * (1 - p_dec_i)
```

This preserves the existing analytical assumption that collision,
half-duplex exposure, and decoding failure are independent mechanisms for an
attempt. Every component and both derived probabilities remain separately
auditable. All probabilities must be finite and lie in `[0, 1]`; propagation
diagnostics and the derived equations are validated rather than trusted.

The selected reservation count does not change `p_rf_i` itself. The later
packet-outcome boundary applies the existing retry model: for `n_i` reserved
RF attempts with identical analytical risk and independent matched-tape draws,
the conditional all-RF-fail probability is `p_rf_i ^ n_i`. This task does not
change that retry assumption.

## Randomness boundary

Risk composition is analytical. It does not accept, inspect, create, or
advance a matched packet tape, and it does not produce a sampled success or
failure. Consequently the identity-addressed
`hybrid-rf-vlc-rl.matched-packet-tape.v1` schema and its four RF draws per
packet remain unchanged.

Outcome sampling remains downstream of analytical composition and consumes the
pre-addressed tape only in the packet-outcome boundary. Thus local risk and
sampled outcomes share one selected ledger without moving the randomness
boundary.

## Implemented boundary and evidence

`mean_field/local_rf_risk.py` provides:

- `PairLocalRFAttemptRisk`, retaining all three mechanism inputs and their
  composed access and total failure probabilities; and
- `FrameLocalRFAttemptRisks`, binding the RF-using subset to one population
  frame, pair-local response set, endpoint schedule, and exact propagation
  mapping.

Unit tests establish independent closed-form composition, distinct risks for
distinct local domains, receiver-specific rather than population-mean
half-duplex exposure, all nine action reservation counts, zero and unit
decoding limits, VLC-only and empty frames, propagation-map exactness,
ordering invariance, frame/count/timing validation, propagation and derived
field validation, and preservation of the matched-tape schema without tape
consumption.

`PAIR_LOCAL_ROLLOUT_MIGRATION_V1.md` records the completed atomic migration.
The next task is the corrected, certificate-aware system-feasibility frontier.
No PPO training is authorized before that gate is evaluated.
