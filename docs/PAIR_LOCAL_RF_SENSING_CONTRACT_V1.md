# Pair-local RF sensing contract v1

Status: frozen migration boundary; implemented and tested; not yet wired into
rollout

Date frozen: 2026-09-27

## Purpose

The RF collision model distinguishes reservations that carrier sensing can
decode from hidden reservations. A single frame-global sensed fraction cannot
represent Manhattan spatial reuse: the same transmitter can be visible to one
focal domain and hidden behind a building from another.

This contract converts the pair-local topology into deterministic geometric
visibility and then weights that visibility by selected RF attempts. The
companion `PAIR_LOCAL_RF_RESPONSE_CONTRACT_V1.md` now consumes this boundary,
but the live rollout is unchanged until the full pair-specific physical path
can migrate atomically.

## Geometry and information boundary

For each focal flow `i`, the topology contract supplies the local member set
`M_i,t`. Let `tx(j)` be member flow `j`'s physical transmitter. For every
external transmitter edge from `tx(i)` to `tx(j)`:

```text
g_i,j,t = 0  if a configured building intersects the antenna path
g_i,j,t = 1  otherwise
```

The building rectangles are deterministically reconstructed from the same
Manhattan grid configuration used for mobility. Their canonical coordinates
are recorded by SHA-256 in each frame sensing report. Building obstruction is
computed once per undirected physical transmitter edge and reused in both
directions.

This rule intentionally treats vehicle-blocked but building-clear edges as
geometrically decodable. Residual reservation-decoding uncertainty remains the
responsibility of the collision model's declared sensing-reliability band; it
is not duplicated as a second geometry penalty.

Flows sharing the focal physical transmitter are locally known and cannot be
hidden behind a building from that transmitter. They remain distinct service
reservations, however, because co-location does not merge packet identities.

All exact geometry is simulator truth. It is used to produce physical outcomes
and diagnostics only. It is not added to the actor observation, which retains
causal noisy neighbor estimates and delayed feedback.

## Attempt-weighted sensed fraction

For focal flow `i`, selected reservations are partitioned into four disjoint
roles:

1. the focal flow;
2. other flows using the focal physical transmitter;
3. geometrically sensed external transmitters; and
4. geometrically hidden external transmitters.

Let `n_rf(a_j,t)` be flow `j`'s reserved RF-attempt count. External offered
attempts, geometrically sensed external attempts, and the sensed fraction are:

```text
E_i,t = sum(j in M_i,t, tx(j) != tx(i)) n_rf(a_j,t)
S_i,t = sum(j in M_i,t, tx(j) != tx(i)) g_i,j,t * n_rf(a_j,t)

s_i,t = S_i,t / E_i,t  if E_i,t > 0
s_i,t = 1              if E_i,t = 0
```

The fraction is weighted by reservation attempts, not by the number of flows.
A hidden RF-4 reservation therefore contributes four hidden attempts while a
visible RF-1 reservation contributes one sensed attempt. VLC-only rows remain
in the audit list with zero RF attempts and cannot distort the fraction.

The neutral value `s_i,t = 1` when there is no external RF demand avoids
inventing hidden load in an empty denominator. It does not assert that an
unobserved transmitter exists.

The local demand conservation identity is:

```text
D_i,t = F_i,t + C_i,t + S_i,t + H_i,t
E_i,t = S_i,t + H_i,t
```

where `F` is focal attempts, `C` is co-located other-flow attempts, and `H` is
geometrically hidden external attempts.

## Collision handoff

The response boundary consumes external attempts rather than treating the
focal flow's own retries or co-located service flows as independent random
contenders. If `r_sense` is the declared decoding reliability for the active
sensitivity band, the expected sensed share passed into that model is
`r_sense * s_i,t`; the complementary external share is hidden. Its
monotonicity and limiting cases are tested before rollout integration.

This document does not define endpoint serialization. Co-located attempts are
kept separate so `ENDPOINT_RF_SCHEDULE_CONTRACT_V1.md` can serialize and
account for them without losing service-flow identity.

## Implemented boundary and evidence

`mean_field/local_rf_sensing.py` provides:

- deterministic building construction from `ProjectConfig`;
- `FrameLocalRFSensing`, with pair-aligned visibility and a building digest;
- `FrameLocalRFSensedLoads`, with action-weighted role partitions; and
- strict frame binding, ordering, type, and conservation checks.

Unit tests cover the complete 55-block headline layout, deterministic geometry
identity, building-hidden and building-clear edges, shared transmitters,
attempt rather than flow weighting, zero-external-load semantics, the
no-building limit, empty frames, and mismatched-frame rejection.
