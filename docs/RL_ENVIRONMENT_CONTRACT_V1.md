# Hybrid RF/VLC RL environment contract

## Status and authority

- Document revision: `1.1.0`
- Action/tensor artifact contract: `environment.contract_version = 1.0.0`
- Pair-local RF pipeline contract: `1.0.0`
- Project configuration schema: `1.1`
- Status: frozen through the pair-local RF rollout migration
- Scope: the synthetic Manhattan trace experiment at densities 10, 20, and
  30 vehicles per lane-kilometer
- Freeze configuration SHA-256:
  `69254a26b691629163f9a404777d0e0d0188867caff2f964c52b647f430e8531`
- Version `1.0.0` freeze validation: 815 tests passed, 26 expected
  artifact-dependent tests skipped, Ruff passed, and strict mypy passed across
  73 source files. Version `1.1.0` migration evidence is recorded in
  `PAIR_LOCAL_ROLLOUT_MIGRATION_V1.md`.

This document is the normative environment contract for the population-coupled RL
environment. The layered YAML configuration supplies numerical values; this
document fixes their meaning, ordering, timing, shapes, and causal boundary. A
change to an action index, tensor column, reward, constraint, lifecycle rule,
or information source requires a new document revision. The action/tensor
artifact version remains `1.0.0` because action indices and tensor columns did
not change; the pair-local physics boundary has its own independently checked
contract version.

The inherited `RF`/`VLC`/`DUP` packet lifecycle is validated foundation code,
not the new environment interface. In particular,
`service.rf_attempts_per_packet: 3` remains the legacy fixed-policy setting
until Phase 3 replaces it. The new environment reads its nine actions and
four-attempt maximum from `environment` in the schema-1.1 project
configuration.

## 1. Decision process

One agent is one active transmitter–receiver pair. All active pairs share one
policy and act synchronously at packet-generation frames. Mobility is
exogenous: an action changes communication resources, channel feedback, and
future information freshness, but never a vehicle trajectory.

The controlled service population is exactly the set of active tagged-pair
flows. Vehicles not serving as an endpoint of an active pair still affect
occlusion, tracking, and the causal neighbor-count proxy, but they generate no
packet in this service pool. If one physical vehicle is the receiver of one
active pair and the transmitter of another, the two pair flows remain separate
agents and both reservations are counted. RF half-duplex uses the exact
current transmit duty cycle implied by the selected reservations at the focal
physical receiver. Attempt timing relative to that duty cycle remains the
declared analytical statistical abstraction rather than a realized NR slot
schedule. Changing to a disjoint matching would define a different experiment.

For density group \(\rho\in\{10,20,30\}\), the reporting objective is

```text
minimize    J_g,rho(pi) = E[g(a) | rho]
subject to  J_c,rho(pi) = E[c | rho] <= epsilon_rho
```

where `epsilon_rho = service.miss_budget = 1e-4` for every headline density.
The shared policy is optimized for the equally weighted mean of the three
density-specific resource objectives. Reliability is not averaged across
densities: it has one constraint and one nonnegative dual variable per density.

PPO may use `gamma = 0.99` and GAE to estimate advantages, but reported cost
and reliability are undiscounted per-packet means. Discounting is an estimator
choice, not a redefinition of the scientific objective.

## 2. Decision-frame clock and ordering

The mobility trace step is 50 ms and packets are generated every 100 ms. A
decision frame occurs on the global trace-relative grid

```text
t_k = trace_start + k * service.generation_period_s
```

Only pairs active at `t_k` appear in frame `k`. A pair that becomes active
between two decision frames first acts at the next frame. Rows within a frame
are ordered lexicographically by stable pair ID; policy batching must not
change the identity-to-row mapping.

Each frame is processed in this exact order:

1. Load the active pairs and lifecycle events at `t_k`.
2. Ingest only measurements whose availability time is at or before `t_k`.
3. Construct local observations and append the one-frame-delayed mean-field
   signal.
4. Produce every active policy action from the same pre-action frame state.
5. Validate action masks and substitute the declared fallback only for pairs
   without a usable causal observation.
6. Project all selected RF reservations into every 200 m pair-local domain.
7. Recompute pair-local utilization, geometric sensing, collision risk, and
   endpoint half-duplex exposure.
8. Evaluate RF/VLC outcomes with matched packet randomness.
9. Assign resource reward, binary miss cost, and conditional-risk diagnostic.
10. Update per-link quality feedback and the delayed population signal.
11. Emit pair termination/truncation flags and advance to `t_(k+1)`.

No action in step 4 may depend on another current-frame action. Current joint
actions become visible only through outcomes and the delayed signal in a later
frame.

The observation and action occur at packet generation time. The configured
0.1 ms `predecision_lead_s` is processing time inside the 3 ms packet deadline;
it is not a 200 ms control delay. Current pair-geometry features describe the
best causal estimate at `t_k`. The 200 ms blockage forecast is an auxiliary
future-risk feature used for sequential planning, not the state on which the
current packet's channel is evaluated.

## 3. Action schema and resource accounting

Action indices are persistent artifact data. They must never be reordered in
place.

| Index | Name | VLC activations | Reserved RF attempts | Cost with headline coefficients |
|---:|---|---:|---:|---:|
| 0 | `VLC` | 1 | 0 | 1 |
| 1 | `RF-1` | 0 | 1 | 1 |
| 2 | `RF-2` | 0 | 2 | 2 |
| 3 | `RF-3` | 0 | 3 | 3 |
| 4 | `RF-4` | 0 | 4 | 4 |
| 5 | `DUP-1` | 1 | 1 | 2 |
| 6 | `DUP-2` | 1 | 2 | 3 |
| 7 | `DUP-3` | 1 | 3 | 4 |
| 8 | `DUP-4` | 1 | 4 | 5 |

For action `a`, let `n_rf(a)` be its reserved-attempt count and `v(a)` its VLC
activation indicator. Its resource cost and reward are

```text
g(a) = cost.rf_activation * n_rf(a)
     + cost.vlc_activation * v(a)
r(a) = -g(a)
```

Both headline coefficients equal 1 normalized activation unit. Sensitivity
runs may change their ratio only through configuration. `VLC` returns the RF
reservation and contributes zero RF demand. A DUP action always pays for both
resources, even if one copy succeeds first.

RF reservations are committed. All `n_rf(a)` attempts count toward population
demand and reward even if decoding succeeds before the last reserved attempt.
An implementation may stop simulating later attempts after success, but it may
not return those already reserved resources to the same frame.

For `N_t` active agents and joint action vector `a_t`, the aggregate committed
attempt count retained for resource accounting and delayed feedback is

```text
D_t = sum(i=1..N_t) n_rf(a_t[i])       # unit: reserved RF attempts/frame
```

`D_t` is not a physical global collision pool. For focal pair `i`, let `M_i,t`
be the active service flows whose physical transmitters lie within the
inclusive 200 m transmitter-centered domain. Physical local demand is

```text
D_i,t = sum(j in M_i,t) n_rf(a_t[j])
```

Every selected reservation is retained once in the authoritative action
ledger and can appear in multiple overlapping domain views. Those views are
not additive. The local response exposes unclipped focal-domain utilization,
clipped CBR, attempt-weighted geometric sensing, and external per-attempt
collision probability. Only external physical transmitters contribute random
collision contenders; co-located service flows are serialized by endpoint.
Under fixed membership and sensing conditions, increasing external local
demand must never reduce focal collision probability. The analytical model and
its uncertainty bands remain identified as such; they are not to be described
as a calibrated NR Mode-2 simulator.

The headline mask contains all nine actions. An action is masked only when a
medium is administratively or physically absent from the hardware profile. A
predicted blockage, poor channel, exact occlusion flag, or sampled failure is
never a reason to mask an action. Masking on those quantities would leak the
answer instead of letting the policy choose under risk.

## 4. Packet deadline and outcome

The headline packet is 300 bytes, generated every 100 ms, with a 3 ms
end-to-end deadline. After the 0.1 ms processing lead, 2.9 ms remains for
transmission. One RF attempt occupies 0.5 ms; the VLC transmission occupies
2.4 ms. RF attempts are sequential and RF/VLC legs run concurrently:

```text
RF-n completion after decision = 0.1 ms + n * 0.5 ms
VLC completion after decision = 0.1 ms + 2.4 ms
```

Thus RF-4 completes by 2.1 ms and every headline action fits the 3 ms
deadline. Configuration validation must reject a future action set whose
longest committed leg does not fit.

The packet is delivered when the first successful selected copy completes by
the deadline. The binary constraint cost is

```text
c_t[i] = 0  if at least one selected copy is delivered by t_k + 3 ms
c_t[i] = 1  otherwise
```

There is no queue across deadlines. A miss does not delay the next packet.
Alongside the sampled binary cost, the simulator emits
`conditional_miss_probability` in `[0, 1]` for the selected action after the
current population load is known. The configured PPO cost critic and cost
advantage use this conditional risk as a lower-variance training signal.
Sampled misses are always retained, and final feasibility uses realized misses
and the configured cluster-aware one-sided confidence bound.

## 5. Mean-field signal

The local observation already contains a causal neighbor-count-based RF busy
proxy. The explicit action-dependent mean-field signal is separate and has two
columns:

```text
u_t = D_t / (4 * N_t)  if N_t > 0 else 0
m_(t+1) = [u_t, 1]
```

`u_t` is the mean fraction of the maximum four RF attempts reserved by the
population, so its range is `[0, 1]`. It is computed after all frame-`t`
actions and is therefore available only at frame `t+1`. The second column is a
validity flag.

At environment reset, `m_0 = [0, 0]`; zero load is not confused with missing
history. After an observed empty frame, the next signal is `[0, 1]` because an
empty response set is a valid measurement. A pair born after reset receives
the same valid delayed population signal as every other agent in that frame.
The signal is reset at a trace or sampled episode boundary because the
preceding actions outside that rollout are unknown.

The actor receives no current action histogram, current `D_t`, future load, or
same-frame CBR. The two-column signal is the only population aggregate exposed
at decentralized execution.

For this experiment, the signal is a noiseless scalar broadcast by the common
RF resource manager after a frame closes and received before the next decision
frame. This one-frame broadcast is an explicit execution assumption, not a
quantity an individual pair can infer from simulator truth. Removing it is a
declared ablation; exposing it without the one-frame delay is forbidden.

## 6. Local observation schema

The local raw vector follows `observation.features` and has 35 float columns.
Both eight-element histories are expanded in place, oldest to newest, and
left-padded. The table below fixes semantics before normalization.

| Order | Feature | Width | Unit / raw range | Causal source and missing behavior |
|---:|---|---:|---|---|
| 1 | `rf_quality` | 1 | normalized, `[0,1]` | Latest noisy, quantized RF report; `0` if never measured |
| 2 | `rf_quality_age` | 1 | s, `{-1} U [0,60]` | Age at `t_k`; `-1` if never measured |
| 3 | `rf_quality_history` | 8 | normalized, `[0,1]` | Oldest-to-newest RF reports; left-pad with `0` |
| 4 | `vlc_quality` | 1 | normalized, `[0,1]` | Latest noisy, quantized VLC report; `0` if never measured |
| 5 | `vlc_quality_age` | 1 | s, `{-1} U [0,60]` | Age at `t_k`; `-1` if never measured |
| 6 | `vlc_quality_history` | 8 | normalized, `[0,1]` | Oldest-to-newest VLC reports; left-pad with `0` |
| 7 | `rf_channel_busy_ratio` | 1 | fraction, `[0,1]` | Local causal proxy from tracked contenders; excludes current-frame actions |
| 8 | `neighbor_count` | 1 | vehicles, nonnegative integer-valued float | Tracks within 200 m; excludes self |
| 9 | `pair_distance` | 1 | m, `[0,+inf)` | Noisy tracks propagated to current decision time |
| 10 | `pair_bearing` | 1 | rad, `[-pi,pi]` | Receiver bearing relative to transmitter heading |
| 11 | `relative_speed` | 1 | m/s, finite real | Receiver speed minus transmitter speed |
| 12 | `heading_difference` | 1 | rad, `[-pi,pi]` | Wrapped receiver minus transmitter heading |
| 13 | `optical_fov_margin` | 1 | rad, `[-2pi/3,pi/3]` headline | Positive inside the 60-degree receiver half-angle |
| 14 | `distance_to_junction` | 1 | m, `[0,244]` headline | Nearest endpoint's causal map distance |
| 15 | `path_spans_junction` | 1 | binary `{0,1}` | From noisy current pair pose and static map |
| 16 | `predicted_blockage_probability` | 1 | probability `[0,1]` | Causal track forecast at `t_k + 0.2 s`; never simulator future |
| 17 | `predictor_confidence` | 1 | fraction `[0,1]` | Forecast uncertainty, not prediction correctness |
| 18 | `track_age` | 1 | s, `[0,1]` | Older tagged-end track; pair is unusable if either track is absent |
| 19 | `previous_action` | 1 | `-1` or integer `[0,8]` | Contract action index; `-1` at pair birth |
| 20 | `last_delivery_outcome` | 1 | `{-1,0,1}` | Unknown, missed, or delivered |
| 21 | `consecutive_miss_count` | 1 | packets, integer `[0,600]` | Reset to zero after a delivery and at pair birth |

The two mean-field columns are appended after these 35 columns:

| Actor column | Feature | Range |
|---:|---|---|
| 35 | `delayed_mean_rf_attempt_fraction` | `[0,1]` |
| 36 | `mean_field_valid` | `{0,1}` |

Zero-based indexing is used above, so the actor's raw observation shape is
`(37,)`. No exact blocker flag, exact vehicle footprint, exact propagation
state, current joint action, random tape, counterfactual outcome, future trace
record, or density label may enter this tensor.

RF quality is reported SINR clipped to `[-10,40]` dB, corrupted by 1 dB
Gaussian estimation error, quantized to 1 dB, then mapped to `[0,1]`. VLC
quality follows the same process over `[0,60]` dB. A link is refreshed only by
an action that uses it. Per-link measurement time is the completion time of its
last attempted transmission; unused-link age continues to increase.

If either tagged endpoint lacks a live causal track, no plausible observation
is fabricated. The environment applies configured fallback `DUP-4`, sets
`learn_mask = 0` for that row, and still includes its reward and miss in
deployment metrics. These packets also contribute to the density-level dual
feasibility estimate because the deployed system must handle them.

## 7. Observation normalization

Normalization state is training state, not dataset metadata.

- Maintain per-column count, mean, and second central moment with Welford's
  algorithm.
- Transform every row in a decision frame with the statistics frozen at the
  start of that frame. If fewer than two historical rows exist, use mean zero
  and variance one.
- After the frame's actions have been sampled, batch-update the statistics once
  with every valid, policy-controlled raw observation from that training frame.
  Current population observations therefore cannot leak through the
  normalizer into another actor's current input.
- Transform as `(x - mean) / sqrt(variance + 1e-8)` and clip to `[-10,10]`.
- Do not standardize `path_spans_junction`, `previous_action`,
  `last_delivery_outcome`, or `mean_field_valid`; pass those encoded values
  unchanged.
- Include finite missing sentinels and history padding in the statistics for
  every standardized column.
- Store the already normalized observation used to sample an action in the
  rollout buffer; PPO epochs must not update the normalizer again.
- Save count, mean, and second moment in every checkpoint.
- Freeze the saved statistics for validation, test, baselines that consume
  normalized inputs, and restored-policy evaluation. Validation/test rows must
  never update them.

The transform preserves shape, so actor input remains `(37,)`.

## 8. Centralized training and decentralized execution

The actor receives only the normalized 37-column tensor and its nine-entry
action mask. Density, exact population size, and population summaries are
critic-only.

For a nonempty frame, define

```text
x_bar_t = mean of normalized actor observations over active agents   # 37
density_one_hot = one of [1,0,0], [0,1,0], [0,0,1]                  # 3
g_t = concat(x_bar_t, density_one_hot, log1p(N_t))                   # 41
critic_input_t[i] = concat(actor_input_t[i], g_t)                    # 78
```

Reward and cost critics have separate parameters but the same `(78,)` input
shape. The actor never receives `g_t`, a density ID, dual variables, exact
channel truth, other agents' current actions, or counterfactual outcomes.
Removing the centralized suffix at execution therefore requires no imputation
or shape change to the actor.

The three density multipliers follow

```text
lambda_rho <- clip(
    lambda_rho + alpha_rho * (estimated_miss_rho - epsilon_rho),
    0,
    configured_maximum
)
```

Increasing violation must never decrease a multiplier. Training batches are
balanced across the three densities for the primal objective; each multiplier
is updated only from its own density samples.

## 9. Variable-population and batch tensors

For one frame with `N_t` agents:

| Field | Dtype | Shape |
|---|---|---|
| actor observation | `float32` | `(N_t, 37)` |
| critic observation | `float32` | `(N_t, 78)` |
| action mask | `bool` | `(N_t, 9)` |
| selected action | `int64` | `(N_t,)` |
| reward | `float32` | `(N_t,)` |
| sampled miss cost | `float32` | `(N_t,)` |
| conditional miss risk | `float32` | `(N_t,)` |
| terminated | `bool` | `(N_t,)` |
| truncated | `bool` | `(N_t,)` |
| bootstrap valid | `bool` | `(N_t,)` |
| learn mask | `bool` | `(N_t,)` |

For a rollout batch with `B` trace segments, `T` decision frames, and `N_max`
equal to the largest population in that batch, prepend `(B,T,N_max)` and pad
the agent dimension with zeros. `active_mask` has shape `(B,T,N_max)` and is
the only authority for whether a padded row participates in losses or
statistics. Pair IDs and trace IDs are retained as aligned metadata; they are
not numeric policy features.

An empty frame advances time and produces the next valid zero-load mean-field
measurement, but contributes no transition rows. PPO flattens only rows where
both `active_mask` and `learn_mask` are true.

The environment intentionally uses a multi-agent frame API rather than a
single-agent Gymnasium scalar step:

```text
reset() -> (FrameObservation, info)
step(actions[N_t]) -> (
    next_FrameObservation,
    rewards[N_t],
    terminated[N_t],
    truncated[N_t],
    info
)
```

`info` carries sampled miss cost, conditional risk, IDs, masks, population
accounting, failure causes, and raw diagnostics. The API preserves Gymnasium's
separate termination/truncation semantics while vectorizing the active
population.

## 10. Pair and trace lifecycle

- **Birth:** create fresh link histories; quality and history are zero-padded,
  both quality ages are `-1`, previous action and last outcome are `-1`, and
  consecutive misses are zero. Global delayed mean-field state is not reset by
  a mid-episode pair birth.
- **Normal transition:** apply one action and one packet outcome, update only
  the links used, and carry state to the next frame in which the pair remains
  active.
- **Natural pair end:** emit `terminated = true`, `truncated = false`, release
  pair state, and use zero value bootstrap.
- **Internal 60 s episode boundary:** emit `terminated = false`,
  `truncated = true`. If the next trace observation exists, expose it only as
  `final_observation` and set `bootstrap_valid = true`; no transition crosses
  into the next sampled episode. The next reset starts fresh pair histories
  and mean-field state even when it samples a later segment of the same trace.
- **Trace boundary:** emit `terminated = false`, `truncated = true`. At physical
  end-of-trace, no next observation exists, so `bootstrap_valid = false` and
  the value target uses zero. This is truncation, not a claim that the physical
  pair entered an absorbing state.
- **Gap and reappearance:** a pair that disappears and later reappears receives
  a new episode instance and fresh history. State is never carried across a
  gap.

Return estimation uses two separate pair-aligned masks:

```text
value_bootstrap_mask = (~terminated & ~truncated) | bootstrap_valid
gae_continuation_mask = ~(terminated | truncated)
```

Thus an internal truncation may use the value of its separately exposed
`final_observation`, but recursive GAE never crosses the reset. Natural endings
and truncations without a next physical trace observation bootstrap from zero.
`final_observation` is an ID-keyed mapping that must cover exactly the rows
where `bootstrap_valid` is true. The final packet itself remains eligible for
learning when its causal actor row is usable. Pair-local history is released
only after that packet's outcome feedback has been processed.

No transition, normalization statistic, recurrent state, or delayed
mean-field value crosses a trace split. Full trace IDs in train, validation,
and test remain immutable and disjoint.

## 11. Randomness and counterfactual consistency

Every stochastic component is derived from the root seed and stable identity,
including trace ID, pair episode ID, packet index, link, and attempt index as
applicable. Results must not depend on iteration order or variable population
size.

Each packet owns a four-attempt RF tape and one VLC tape. RF-n consumes the
first `n` RF entries; DUP-n consumes the same RF prefix plus the same VLC entry.
This prefix rule makes actions comparable on matched randomness. Sensor and
feedback noise use separate named streams. Counterfactual outcomes may be
logged for oracle analysis but may never enter actor or critic inputs used by a
deployable policy.

At environment reset, an explicit `seed` overrides `training.root_seed` for
that reset; `seed = None` uses the configured root. Reset metadata records the
seed schema, configured root, active root, trace ID, and runtime stochastic
component list. All roots are unsigned 64-bit integers. Stateful shadowing and
fading own persistent generators keyed by trace and pair-episode identity, so
their temporal/spatial correlation is preserved without making initial state
depend on pair iteration order. The generator and correlated state are
released together at the pair boundary. A live channel process may not cross
to another trace.

Mobility is not redrawn at environment reset: it is fixed in the selected
immutable trace artifact and its generation seed remains in that artifact's
manifest. Policy-action sampling and analysis bootstrap resampling use their
own named roots outside the environment and are never advanced by `step()`.

## 12. Required invariants for implementation phases

Phases 2–5 must enforce and test all of the following before training:

1. Exactly one stable row per active pair and no duplicate pair ID per frame.
2. All actor/critic values finite after normalization.
3. Every probability in `[0,1]`; every selected action unmasked and in `[0,8]`.
4. Sum of per-agent reserved attempts equals `D_t`; every local-domain load
   exactly conserves the ledger rows in its membership; VLC contributes zero.
5. Population cost equals the sum of the action-table costs.
6. Current focal collision risk is nondecreasing in external local demand under
   fixed domain membership, geometry, sensing band, and channel state.
7. Current actions cannot appear in their own observations or mean-field input.
8. No observation imports exact future trace state or exact occlusion truth.
9. Link feedback refreshes only media used by the selected action.
10. Birth, termination, truncation, and bootstrap masks follow Section 10.
11. Reset with the same trace, configuration, and seed produces identical
    frames, actions under a deterministic policy, and outcomes.
12. Train-only normalization and split isolation are mechanically tested.

The Phase 5 runtime enforcement for items 1--3 and 10 is documented in
`PHASE5_INVARIANTS.md`. In particular, policy action arrays are checked against
the current row masks before any resource accounting, and pair lifecycle is
checked across consecutive frames rather than only row by row.

## 13. Configuration sources

| Contract quantity | Configuration source |
|---|---|
| Action/tensor version and order | `environment.contract_version`, `environment.actions` |
| Pair-local RF physics version | `LOCAL_RF_PIPELINE_CONTRACT_VERSION` |
| Maximum RF attempts | `environment.max_rf_attempts` |
| Missing-observation fallback | `environment.no_observation_fallback_action` |
| Mean-field delay/reset semantics | `environment.mean_field` |
| Normalization algorithm constants | `environment.normalization` |
| RF-attempt and VLC-activation costs | `cost.rf_activation`, `cost.vlc_activation` |
| Deadline, generation period, processing lead | `service` |
| RF/VLC airtime | `rf.timing.airtime_s`, `vlc.timing.airtime_s` |
| Reliability budget | `service.miss_budget` |
| Observation order/history/noise/latency | `observation` |
| PPO reliability training signal | `training.cost_signal` |
| Per-density dual learning rates and caps | `training.density_multipliers` |
| Train/validation/test membership | `environment.splits` |

No implementation may silently replace these values with module constants.
Derived constants such as actor width 37, critic width 78, and nine logits must
be asserted against the loaded schema at construction.

## 14. Phase 1 closure and implementation boundary

This contract freezes what later phases must build; it does not claim that the
inherited three-action code already implements it. The intentional migration
boundary is:

- Phase 2 supplies chronological population frames and lifecycle metadata.
- Phase 3 supplies the authoritative nine-action type, resource ledger, and
  legacy-action replacement.
- Phase 4 supplies action-coupled pair-local topology, sensing, response,
  endpoint schedule, and attempt-risk contracts consuming each `D_i,t`.
- Phase 5 supplies the frame API, actor/critic tensors, normalization, delayed
  mean-field state, and lifecycle masks.

The atomic integration of those gates is documented in
`PAIR_LOCAL_ROLLOUT_MIGRATION_V1.md`. A rollout is document revision `1.1.0`
compliant only when packet outcomes and delayed feedback retain the same
selected ledger and `FrameLocalRFPhysics`.
