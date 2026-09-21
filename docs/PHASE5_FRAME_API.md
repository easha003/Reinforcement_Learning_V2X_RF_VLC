# Phase 5 multi-agent frame API

Status: frozen for environment contract `1.0.0` on 2026-09-21.

## Decision

The RL extension uses a Gymnasium-shaped multi-agent frame API, but it does
not inherit from `gymnasium.Env` and must not be passed to Gymnasium's
single-agent environment checker.

The deviation is intentional and required. A scalar Gymnasium step describes
one agent and requires one scalar reward, one `terminated` boolean, and one
`truncated` boolean. A decision frame in this project contains `N_t` active
pairs. It therefore returns reward and lifecycle vectors of length `N_t`, and
the next frame can contain `N_(t+1) != N_t` pairs because agents may be born or
leave between frames. Collapsing those vectors to one scalar flag would lose
the pair-level bootstrap semantics required by the environment contract.

The core package does not import the optional `gymnasium` dependency merely to
name this interface. `MultiAgentFrameEnv` is a runtime-checkable structural
protocol, so the eventual environment can implement it without inheriting
from a third-party base class.

## Preserved Gymnasium conventions

```python
reset(*, seed=None, options=None) -> (FrameObservation, info)

step(actions) -> (
    next_frame_observation,
    rewards,
    terminated,
    truncated,
    info,
)
```

The API preserves:

- `reset` returning `(observation, info)`;
- keyword-only `seed` and `options` arguments;
- the five-part step result;
- distinct termination and truncation signals; and
- a `close()` lifecycle hook.

## Vector semantics

For the frame in which actions are selected:

| Value | Type and shape | Alignment |
|---|---|---|
| `actions` | `int64`, `(N_t,)` | current `FrameObservation.pair_ids` |
| `rewards` | `float32`, `(N_t,)` | `info["transition_pair_ids"]` |
| `terminated` | `bool`, `(N_t,)` | `info["transition_pair_ids"]` |
| `truncated` | `bool`, `(N_t,)` | `info["transition_pair_ids"]` |
| next actor observations | `float32`, `(N_(t+1), 37)` | next `pair_ids` |
| next action masks | `bool`, `(N_(t+1), 9)` | next `pair_ids` |

`transition_pair_ids` is the canonical stable-ID order of the actors that just
acted. `next_frame_observation.pair_ids` independently identifies the next
population. Code must never align transitions by row position across frames.

An empty population is represented by arrays of shape `(0, 37)` and `(0, 9)`,
not by `None` or by dropping the frame. It still advances the environment and
can produce delayed zero-load congestion for the following frame.

## Variable-population binding

`VariablePopulationBinding` joins three existing authorities:

- Phase 2 `PopulationFrame.active_pair_ids` defines active episode identities
  and their canonical row order;
- keyed actor rows supply one 37-column vector for each exact active ID; and
- the Phase 3 `ActionMask` supplies the profile-wide nine-action hardware
  mask, repeated once per active pair.

The caller's mapping insertion order has no meaning. Missing, unexpected, or
blank actor-row keys fail before an observation is emitted. The binding reports
three disjoint stable-ID sets with each observation: entered, continuing, and
exited pairs. On the first frame after reset, every active ID is entered.

Frames must belong to the reset trace and arrive at consecutive indices,
including empty frames. Once an ID exits, it is retired for that sampled
episode and cannot reappear after a gap; the source must assign a new episode
identity. Reset clears both current and retired identity state, so a later
sampled segment starts cleanly even when it belongs to the same source trace.

The current profile has one hardware configuration for every pair, so the mask
is deliberately population-wide. Agent-specific hardware and per-agent action
spaces remain deferred scope.

## Causal actor-observation assembly

`CausalActorObservationAssembler` is the pre-action boundary between a Phase 2
`PopulationFrame` and the actor rows accepted by the population binding. For
each canonical active pair it constructs the inherited `PairInstant` view over
the current trace frame, then delegates the local row to `Perception`. That
path exposes only noisy, delayed tracks, causal map context, a probabilistic
blockage forecast, a sensed neighbour-load proxy, and link reports retained
from completed packets. It does not import exact channel outcomes into the
row-building path.

After all 35-column local rows have been materialized, the assembler appends
the same two-column `MeanFieldSignal` frozen by `begin_frame`. The signal is
either reset encoding `[0, 0]` or the audited RF-attempt fraction and validity
bit queued when the preceding frame closed. The current frame's actions,
demand, CBR, collision probability, and outcomes are not arguments to
`begin_frame` and cannot affect its returned immutable tuples.

An active pair whose transmitter or receiver has no live causal track is
represented as a stable-ID `CausalActorRow(values=None)`. No zero or
plausible-looking feature vector is fabricated. `usable_mask` and
`unusable_pair_ids` make that state explicit; only `usable_actor_rows` may be
sent to the policy. The complete environment will apply the configured
`DUP-4` fallback and `learn_mask = 0` when it assembles actions for such a row.
At physical trace time zero, sensor latency means no awareness report has yet
arrived; the first frame therefore follows this explicit unusable path instead
of creating a negative measurement tick or pretending current truth was sensed.

Once observations exist, exactly one feedback record is required per active
pair before the frame closes. The record carries the persistent nine-action
`PolicyAction`, the sampled delivery bit, and only already degraded,
quantized link reports in `[0, 1]`. The resource map rejects reports for a
medium the action did not use. Availability must fall between the decision
time and the packet deadline, so a future report cannot be inserted into
history. Closing then queues the audited population response for the next
frame. Reset clears perception tracks, pair histories, and delayed congestion
together.

This also repairs an inherited vocabulary mismatch: the feasibility simulator
stored only three RF/VLC/DUP action identities. Phase 5 link state now has a
dedicated nine-action recording path, so `previous_action` retains the exact
contract index `0..8`; RF-1 through RF-4 and DUP-1 through DUP-4 are not
collapsed merely because they refresh the same physical media.

## Observation boundary

`FrameObservation` contains only:

- trace and frame identity;
- canonical stable pair IDs;
- the decentralized actor matrix; and
- legal-action masks.

It deliberately contains no critic tensor, reward, cost, current joint action,
pool response, channel truth, or counterfactual. Those values belong to
training-only structures or `info` as later Phase 5 tasks define them.

The versioned `FrameAPISchema` derives the actor width from the configured
35-column local observation plus the two delayed mean-field columns and derives
the action count from the configured canonical action order. Construction
fails unless contract `1.0.0` resolves to exactly 37 actor columns and nine
actions.

## Validation and ownership

API objects fail closed on:

- empty, duplicate, or noncanonical pair IDs;
- row-count drift between IDs, observations, and masks;
- incorrect dtypes or ranks;
- non-finite actor observations or rewards;
- an active actor with no legal action;
- reward/lifecycle arrays not aligned to `transition_pair_ids`;
- a transition marked both terminated and truncated; and
- disagreement between explicit IDs and IDs supplied through `info`.

Arrays are copied into contiguous read-only storage, and step `info` is exposed
through a read-only mapping. This prevents callers from mutating an already
validated transition after it enters a rollout buffer.

## Compatibility boundary

Libraries that understand parallel multi-agent environments may receive a
thin adapter later. A single-agent Gymnasium adapter would require an explicit
research decision about aggregation and is not part of contract `1.0.0`.
Neither adapter may change pair-level termination, truncation, or identity
alignment.
