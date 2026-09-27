# Phase 8 RF contention-domain audit

Date completed: 2026-09-27

## Decision

The mean-field environment's RF contention boundary is inconsistent with the
project's declared 200 m local contention domain. The current rollout sums RF
attempts from every active pair in the Manhattan frame, then gives every RF
attempt the same collision probability. Across the frozen validation windows,
84.3%--85.4% of those globally pooled potential pair flows lie outside the
focal transmitter's 200 m domain. No evaluated pair has a local domain equal
to the complete frame.

This is a blocking environment-model defect. The previous population-joint
oracle remains an exact result for the implemented global-pool model, but it
must not be interpreted as evidence that the intended local-contention hybrid
system is infeasible. PPO training, seeds 1002--1005, and the proposed system-
capacity frontier remain blocked until the RF contention boundary is repaired
and the feasibility oracle is rerun.

## Contract evidence

Three pre-existing boundaries describe local contention:

- `env/rollout.py` defines true RF contenders as vehicles within 200 m of the
  pair transmitter;
- `env/perception.py` uses the same 200 m radius for the actor's noisy,
  forecast-based neighbor count; and
- `channels/rf/collision.py` describes its input as nearby contenders and
  validates headline behavior against measured within-200-m counts.

The mean-field path instead does the following:

1. `FrameActionLedger` preserves an action and RF reservation for every active
   pair in the complete frame.
2. `RFPoolDemand.from_ledger` sums all of those reservations into one `D_t`.
3. `RFPoolModel.evaluate` sets the focal contender count to `D_t - 1`.
4. `deterministic_rollout.py` applies the resulting single pool response to
   every RF-using pair, with `sensed_fraction=1.0`.

Consequently, a transmitter on the far side of the Manhattan map contributes
collision pressure to the focal link exactly like a nearby transmitter. This
forbids spatial reuse and changes the collision model's input from a local
neighbor count to a network-wide flow-attempt count.

## Measurement boundary

The audit replays the same nine frozen validation windows as the regime and
joint-oracle evaluations: 144 sampled frames, 117 nonempty frames, and 69,626
active pair rows. It opens no test trace and chooses no actions.

For each focal pair it counts:

- exact vehicles within 200 m of its transmitter, excluding the transmitter;
  and
- active pair flows whose transmitters lie within 200 m, including the focal
  flow because this is the local analogue of the frame-wide demand population.

Pair flows are retained rather than merged by physical transmitter because the
current action ledger deliberately represents each service pair separately.
The audit is therefore a scope comparison, not yet a corrected resource
scheduler. Building obstruction, sensing success, selected actions, and exact
local RF attempts are intentionally deferred to the repair.

## Artifact identity

- Schema: `hybrid-rf-vlc-rl.rf-contention-domain-audit.v1`
- Artifact: `artifacts/evaluations/phase8_rf_contention_domain_audit.json`
- Artifact SHA-256:
  `9e2c3752f37781b6962cad5e7092c0067b37219ec36a470f28cf0a3fde399c70`
- State-regime window artifact SHA-256:
  `35b240c2fc58b46f02999d225663b6a91803509309cf098b4927e19aa6a7b2a7`
- Configuration hash:
  `df5cf40513f0c08ceba1b037b58a1002b9cc3fa87033601d0f3ee9ec452800c4`
- Policy-environment scope hash:
  `46fecd53689db6f2c9aa21b4314da610b0caa7e47444a4a9b01f5ba4a3b1cb29`
- Test split opened: no

## Results

| Density | Pair rows | Mean global active pairs/frame | Mean local pair flows | Median global/local ratio | Mean local vehicles | Global flows outside local domain | Exact global rows |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 7,938 | 203.5 | 31.5 | 6.83x | 47.8 | 85.38% | 0 |
| 20 | 23,204 | 595.0 | 95.6 | 6.72x | 108.7 | 84.31% | 0 |
| 30 | 38,484 | 986.8 | 150.7 | 6.43x | 162.0 | 84.88% | 0 |
| Campaign | 69,626 | 595.1 | 118.7 | 6.55x | 131.2 | 84.74% | 0 |

The exact local vehicle means reproduce the repository's independently pinned
campaign scale of approximately 47, 108, and 162 neighbors. This confirms that
the audit is measuring the intended local quantity. In contrast, nonempty
frames contain as many as 256, 689, and 1,096 simultaneously active pair flows
at densities 10, 20, and 30.

The mismatch is not confined to edge cases. The 25th--75th percentile local
pair-flow domains are 22--41, 67--116, and 120--183, while the corresponding
median complete-frame populations are 238, 643, and 1,058. All 69,626 pair rows
exclude at least one global flow from their physical 200 m domain.

## Effect on the feasibility conclusion

The global-pool oracle correctly answered:

> Can `1e-4` be achieved if every RF attempt in the complete Manhattan frame
> contends with every other RF attempt?

Its answer was no. The intended research question is different:

> Can `1e-4` be achieved when resources may be reused outside a pair's local
> RF interference domain?

The existing oracle cannot answer the second question. Pair-local load makes
collision risk pair-specific, and each action changes the load of several
overlapping neighborhoods rather than one scalar global pool. The earlier
aggregate-load separability proof therefore no longer applies after the model
is repaired.

This audit does not prove that a local-domain system meets `1e-4`; it proves
that the failed global-domain gate cannot settle that question.

## Required repair

Before further learning or capacity sensitivity experiments, define and test a
new RF contention contract with these properties:

1. Each active RF flow contributes attempts only to focal domains in which its
   transmitter is a contender; distant flows may reuse the same resources.
2. Pool responses and RF access risks are pair-specific rather than one shared
   scalar response for the frame.
3. The physical 200 m membership calculation is distinct from the actor's
   noisy causal estimate; simulator truth must not enter the actor.
4. Geometric sensing visibility and the declared sensing-reliability band are
   represented explicitly instead of silently retaining a global
   `sensed_fraction=1.0`.
5. Half-duplex exposure is tied to physical endpoint activity rather than the
   complete population's mean RF-attempt count.
6. Shared endpoints and multiple service flows from one transmitter conserve
   resources without double-counting simultaneous physical transmissions.
7. Empty frames, pair lifecycle, stable ordering, and matched randomness retain
   their existing deterministic contracts.

After implementation, rerun the RF-pool monotonicity and limiting-case tests,
all baselines, the frozen seed-1001 validation evaluation, and a new feasibility
oracle appropriate for overlapping local domains. Only then decide whether a
system-capacity frontier or another RL recovery arm is necessary.
