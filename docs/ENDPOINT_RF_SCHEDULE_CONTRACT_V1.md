# Endpoint RF schedule and half-duplex contract v1

Status: frozen, implemented, tested, and live in the atomic rollout path

Date frozen: 2026-09-27

## Purpose

The legacy RF risk uses population-mean committed airtime as every receiver's
half-duplex probability. That assigns transmit activity to vehicles that are
only receiving and hides concentrated activity when one physical transmitter
serves multiple flows.

This contract assigns every selected service-flow reservation to its physical
transmitter exactly once, serializes reservations without merging packet
identities, and derives half-duplex exposure from the focal receiver's own RF
activity. It is now integrated with packet outcomes and the rest of the local
rollout through one atomic assembler.

## Service flows and physical radios

For active flow `j`, the population frame provides physical transmitter
`tx(j)` and receiver `rx(j)`, while the action ledger provides selected RF
attempts `n_j`.

For physical endpoint `v`, its outgoing flow set and offered attempts are:

```text
T_v = { j : tx(j) = v }
A_v = sum(j in T_v) n_j
```

Every flow remains a separate reservation row, including a VLC-only row with
zero RF attempts. Within an endpoint, rows follow canonical pair-ID order and
attempts follow their action index order. This defines a logical serialized
sequence:

```text
pair-a attempt 0, ..., pair-a attempt n_a-1,
pair-b attempt 0, ..., pair-b attempt n_b-1, ...
```

The sequence is an accounting and serialization boundary, not a claim that
these offsets are realized NR Mode-2 slot selections. The analytical collision
model still represents resource selection statistically. Logical offsets make
it impossible for two service flows from one radio to be misclassified as
independent simultaneous hidden contenders.

The conservation identity is exact:

```text
sum(v) A_v = sum(j) n_j = frame ledger RF attempts
```

Thus a shared transmitter does not merge service packets, double-count them,
or allow their airtime to overlap on one radio.

## Endpoint activity and exposure

Let `tau` be per-attempt airtime and `T` the packet-generation period. Endpoint
offered airtime, utilization, and bounded transmit duty cycle are:

```text
L_v = A_v * tau
u_v = L_v / T
d_v = min(1, u_v)
```

Unbounded `u_v` retains endpoint oversubscription while `d_v` remains a valid
probability. An oversubscribed schedule is reported explicitly; later physical
assembly must not silently describe its excess reservations as deliverable.

For focal flow `i`, half-duplex exposure is:

```text
p_hd_i = d_rx(i)
```

Only flows transmitted by the focal receiver contribute. Packets arriving at
that receiver do not make it transmit, and activity at unrelated endpoints
does not enter the calculation. Multiple outgoing service flows from the same
receiver all contribute and remain individually auditable.

The formula retains the existing analytical assumption that transmission phase
relative to an arriving attempt is independent over the generation period. It
changes the source of the duty cycle from a population mean to exact current
endpoint activity; it does not claim a realized slot-level half-duplex trace.

Under the headline profile, `tau = 0.5 ms` and `T = 100 ms`. A receiver with
four selected attempts therefore has `p_hd = 0.02`, while a receiver with no
outgoing RF reservation has `p_hd = 0`, regardless of frame-wide load.

## Implemented boundary and evidence

`mean_field/endpoint_rf_schedule.py` provides:

- `PairRFEndpointReservation`, binding each flow to physical endpoints;
- `EndpointRFSerializedAttempt`, preserving canonical per-radio attempt order;
- `EndpointRFTransmitSchedule`, exposing offered airtime, utilization, duty
  cycle, and endpoint overload;
- `PairHalfDuplexExposure`, binding a focal flow to its receiver schedule; and
- `FrameEndpointRFSchedule`, enforcing complete frame and ledger conservation.

Unit tests establish shared-transmitter serialization, shared-receiver
exposure, exclusion of inbound traffic, divergence from the legacy population
mean, exact and overloaded endpoint-capacity limits, all nine action counts,
VLC release, empty-frame behavior, ordering invariance, frame binding, timing
validation, and fail-closed derived-field reconciliation.

This simulator-truth activity is not added to the actor observation. The
companion `PAIR_LOCAL_RF_RISK_CONTRACT_V1.md` composes pair-local
collision, endpoint half-duplex, and RF propagation into pair-specific attempt
risk without changing the matched-tape boundary. Both are now live through the
atomic assembler documented in `PAIR_LOCAL_ROLLOUT_MIGRATION_V1.md`.
