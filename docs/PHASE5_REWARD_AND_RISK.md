# Phase 5 reward and reliability targets

## Scope

`mean_field.packet_outcomes` is the single boundary that converts a complete
current-frame action ledger and selected-link simulator evaluations into the
three pair-aligned target vectors required by environment contract 1.0.0:

| Target | Dtype and shape | Meaning |
|---|---|---|
| `rewards` | `float32 (N_t,)` | Negative committed activation cost |
| `sampled_miss_costs` | `float32 (N_t,)` | Realized binary deadline miss |
| `conditional_miss_probabilities` | `float32 (N_t,)` | Selected-action miss risk after current load and physical state are known |

Rows follow the action ledger's canonical stable pair-ID order. Arrays are
immutable. Empty frames produce three `(0,)` arrays and still retain their
validated zero-demand RF-pool response.

## Reward

Reward is copied from the Phase 3 resource ledger rather than inferred from
the sampled result:

```text
r(a) = -[cost.rf_activation * n_rf(a)
         + cost.vlc_activation * v(a)]
```

All reserved RF retries and the VLC activation are charged. An RF packet that
succeeds on its first retry under `RF-4` therefore receives the `RF-4` reward,
not the `RF-1` reward. A successful RF leg does not refund the VLC activation
of a DUP action.

## Sampled binary miss

Every packet owns the Phase 4 identity-addressed tape containing four RF
mechanism entries and one VLC entry. `RF-n` and `DUP-n` consume the same first
`n` RF entries. An RF retry checks half duplex, then collision, then decoding,
matching the channel model's causal mechanism order, and stops after the first
success. The already committed but unevaluated retry reservations remain in
load and reward accounting.

The selected packet is delivered if either selected leg succeeds. Its sampled
CMDP cost is exactly:

```text
c = 0 if delivered else 1
```

The output retains the attempted RF prefix, terminal mechanisms, link delivery
flags, and packet failure cause as simulator diagnostics. The assembler checks
that a supplied VLC result agrees with the packet's VLC tape entry and rejects
identity, population, or result drift.

## Conditional miss probability

Let `p_rf` be the Phase 4 total failure probability of one RF attempt under
the current joint-action pool response, and let `p_vlc` be the evaluated VLC
failure probability for the same current packet state. Independent matched
draws across retries give:

```text
P_miss(VLC)   = p_vlc
P_miss(RF-n)  = p_rf ** n
P_miss(DUP-n) = (p_rf ** n) * p_vlc
```

These are conditional simulator risks: current geometry, RF propagation,
optical outage, and population-coupled contention have already been resolved.
The remaining mechanism draws are independent by the matched-tape contract.
The product is not a claim that unconditional RF and VLC failures are
independent across mobility states; their shared geometric dependence remains
present because both conditional inputs came from the same packet state.

## Leakage boundary

`FramePacketOutcomes.as_step_info()` exposes sampled costs, conditional risks,
pair diagnostics, and the RF-pool response for training and audit code. None of
these fields is added to `FrameObservation`, `CausalActorFrame`, or
`CriticObservationFrame`. Both actor and centralized critic inputs are frozen
before current actions and outcomes, so the selected action's hidden truth
cannot leak into either policy input.

Final feasibility analysis must use `sampled_miss_costs`; the conditional
probability is the lower-variance signal configured for the PPO cost critic and
cost advantage.
