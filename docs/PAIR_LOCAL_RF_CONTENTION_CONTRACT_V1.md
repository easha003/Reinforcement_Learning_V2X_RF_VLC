# Pair-local RF contention contract v1

Status: frozen migration boundary; not yet wired into rollout

Date frozen: 2026-09-27

## Purpose

This contract replaces the frame-global RF demand assumption identified by
`PHASE8_RF_CONTENTION_DOMAIN_AUDIT.md`. It freezes the action-independent
spatial topology and action-coupled local demand before collision,
half-duplex, feedback, or packet outcomes are migrated. Pair-local sensing is
now frozen in the companion `PAIR_LOCAL_RF_SENSING_CONTRACT_V1.md`.

The implementation is intentionally not connected to the live rollout yet. A
partially migrated environment could calculate local feedback while retaining
global packet risk, creating a harder observation/physics mismatch than the
one being repaired.

## Identities and local membership

At decision frame `t`, let `P_t` be the canonically ordered set of active
service-pair flows. Each flow `i` has a physical transmitter position
`x_tx(i,t)`. The frozen contention radius is:

```text
R = 200 m
```

The pair-local membership set is:

```text
M_i,t = { j in P_t : ||x_tx(j,t) - x_tx(i,t)||_2 <= R }
```

Properties:

- Membership is centered on the focal transmitter, matching the existing
  physical and causal-observation contracts.
- The 200 m boundary is inclusive.
- Membership is computed from current simulator geometry before actions and is
  therefore action-independent.
- The focal flow is always a member of its own domain.
- Euclidean transmitter-distance membership is reciprocal: if `j` belongs to
  `M_i,t`, then `i` belongs to `M_j,t`.
- Member flows follow canonical pair-ID order. Population iteration order
  cannot change membership or local demand.
- An empty population produces an empty topology rather than a synthetic
  domain.

This topology is simulator truth. It must not be exposed directly to the
actor. The actor retains its noisy, forecast-based neighbor count and delayed
mean-field information unless a later versioned observation change is
explicitly approved.

## Action-coupled local demand

Let `n_rf(a_j,t)` be the authoritative RF-attempt reservation for flow `j`'s
selected action. For focal flow `i`:

```text
D_i,t = sum(j in M_i,t) n_rf(a_j,t)
U_i,t = sum(j in M_i,t) 1[n_rf(a_j,t) > 0]
```

`D_i,t` is reserved RF attempts per generation frame in the focal domain;
`U_i,t` is the number of RF-using service flows in that domain.

Every member is retained in the auditable reservation list, including VLC-only
members with zero RF attempts. The local sum and RF-user count must reconcile
exactly with those rows. A flow outside `M_i,t` cannot change `D_i,t`, which is
the operational spatial-reuse invariant.

Local demands overlap and are not additive. One physical RF reservation can
appear in many focal-domain views because it can affect many nearby links. It
is still one reservation in the authoritative action ledger. Summing all
`D_i,t` values would double-count shared influence and is forbidden.

## Multiple flows and shared physical transmitters

The policy population is a set of service-pair flows, not a deduplicated set of
vehicles. If two active flows share one physical transmitter, they remain two
action rows and two packet reservations. Each selected reservation appears
exactly once in a focal domain when its transmitter is a member:

```text
same transmitter + two service packets != one merged policy action
```

This contract does not assume that those attempts occur at the same physical
instant. Endpoint scheduling, serialization, and half-duplex exposure are the
next repair boundary. They must use physical endpoint identity without erasing
distinct service packets.

## Frame binding and failure behavior

The topology and action ledger must have identical:

- trace ID;
- frame index and time;
- canonical active pair IDs; and
- lifecycle population.

A mismatch fails closed before any local load is produced. Invalid radii,
noncanonical membership, missing focal flows, repeated member flows,
nonreciprocal membership, and nonconserving reservation totals are rejected.

## Implemented boundary

`mean_field/local_rf_domain.py` now provides:

- `PairLocalRFDomain`: one focal membership set;
- `FrameLocalRFTopology`: the complete action-independent frame graph;
- `PairLocalRFLoad`: one focal action-coupled reservation sum; and
- `FrameLocalRFLoads`: pair-aligned local demands for the frame.

Unit tests establish inclusive-radius geometry, reciprocal membership,
canonical ordering, empty-frame behavior, exact per-domain conservation,
shared-transmitter flow identity, frame binding, and the spatial-reuse
invariant that a distant action change cannot alter focal demand.

## Explicitly deferred boundaries

The live environment still uses the old global pool until all migration
boundaries are ready. Pair-local sensing, collision/CBR responses, and endpoint
half-duplex exposure are implemented and tested but, like this topology
boundary, are not yet connected to rollout. The remaining changes are:

1. Pair-specific packet-risk and outcome assembly.
2. Delayed local/aggregate congestion feedback semantics.
3. Baseline and oracle migration to overlapping local domains.
4. Removal or archival of the old global `D_t` physical-risk path.

The rollout migration must occur atomically after these components agree on
identity, units, and timing.
