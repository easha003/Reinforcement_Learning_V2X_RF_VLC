# Pair-local RF response contract v1

Status: frozen migration boundary; implemented and tested; not yet wired into
rollout

Date frozen: 2026-09-27

## Purpose

This contract maps every focal flow's local, action-coupled RF load to its own
resource utilization, channel-busy ratio (CBR), and collision probability. It
replaces the invalid assumption that all active Manhattan-frame flows share one
global response, while retaining the declared analytical collision model with
sensitivity bands.

The implementation is deliberately isolated. The companion
`ENDPOINT_RF_SCHEDULE_CONTRACT_V1.md` now supplies receiver-specific
half-duplex exposure, but the live rollout continues to use the global
response until packet outcomes, delayed feedback, baselines, and the
feasibility oracle can migrate atomically.

## Occupancy response

For focal flow `i`, the preceding contracts provide:

- `D_i,t`: every selected RF attempt in the 200 m local domain;
- `F_i,t`: the focal flow's attempts;
- `C_i,t`: other service-flow attempts from the focal physical transmitter;
- `S_i,t`: geometrically sensed external attempts; and
- `H_i,t`: geometrically hidden external attempts.

They conserve as:

```text
D_i,t = F_i,t + C_i,t + S_i,t + H_i,t
E_i,t = S_i,t + H_i,t
```

Let `tau` be airtime per already-counted attempt, `K` the number of
subchannels, and `T` the generation period. Local utilization and bounded CBR
are:

```text
Q       = K * T
u_i,t   = D_i,t * tau / Q
CBR_i,t = min(1, u_i,t)
```

Every local reservation contributes airtime, including the focal and
co-located flows. Utilization remains unbounded so oversubscription stays
visible after CBR reaches one. Under the headline profile, `tau = 0.5 ms`,
`K = 2`, `T = 100 ms`, and exact saturation is 400 attempts.

## External-collision response

Focal retries cannot collide randomly with themselves. Likewise, service flows
originating at the same physical transmitter require serialization rather than
being modeled as independent hidden terminals. Therefore only external
attempts `E_i,t` enter the collision calculation.

For sensing reliability `r` in the declared band, effective sensed and hidden
external loads are:

```text
S_eff_i,t = r * S_i,t
H_eff_i,t = H_i,t + (1 - r) * S_i,t
          = E_i,t * (1 - r * s_i,t)
```

With `M` candidate resources, the per-attempt birthday-model collision
probability is:

```text
p_col_i,t = 1 - (1 - 1/M) ^ H_eff_i,t
```

The headline resource pool has `M = 400`. The sensing reliability remains the
predeclared uncertainty band rather than a fitted measurement:

| Band | Reliability `r` |
|---|---:|
| Optimistic | 0.95 |
| Nominal | 0.85 |
| Pessimistic | 0.70 |

This boundary does not itself add half-duplex risk. The endpoint schedule now
conserves every service-flow reservation and supplies a separate exposure for
later packet-risk composition.

## Proven limits and monotonicity

The test contract establishes, independently in every declared band:

- no external attempts implies zero hidden load and zero collision risk;
- all geometrically hidden attempts make collision risk band-independent;
- all geometrically sensed attempts retain only the band's decoding failures;
- for sensed traffic, optimistic risk is below nominal risk, which is below
  pessimistic risk;
- collision risk strictly increases with every additional external attempt;
- collision risk strictly decreases as a fixed external load becomes more
  geometrically sensed;
- CBR and utilization increase together below 400 attempts;
- at 400 attempts, utilization and CBR both equal one; and
- above 400 attempts, CBR remains one while utilization and collision risk
  continue increasing.

Additional tests prove that changing only focal or co-located attempts changes
occupancy but not external-collision probability, response fields reconcile
with independent closed-form equations, empty frames are valid, pair ordering
is canonical, and inconsistent models or derived fields fail closed.

## Implemented boundary

`mean_field/local_rf_response.py` provides:

- `LocalRFResponseModel`, which normalizes packet-level channel parameters to
  one already-counted RF attempt;
- `PairLocalRFResponse`, containing one focal domain's utilization, CBR,
  effective sensing loads, and collision probability; and
- `FrameLocalRFResponses`, a pair-aligned response set bound to the trace,
  frame, sensing contract, and sensitivity band.

The actor observation remains unchanged. These are simulator-truth physical
responses and diagnostics until the later feedback migration defines what
delayed aggregate is causally observable.
