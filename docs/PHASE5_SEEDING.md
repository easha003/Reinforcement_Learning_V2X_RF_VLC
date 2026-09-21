# Phase 5 deterministic seeding

## Reset authority

`EnvironmentSeedState` resolves exactly one active environment root at reset:

- `reset(seed=None)` uses `training.root_seed`;
- `reset(seed=n)` uses `n` for that reset and records that the override was
  explicit; and
- both configured and explicit roots must be unsigned 64-bit integers.

`TraceRandomness` binds the active root to one immutable trace ID and is the
only Phase 5 factory for causal actor sensing, matched population packet tapes,
and noisy link feedback. Its read-only reset metadata records the seed schema,
configured root, active root, trace ID, and complete runtime component list.

## Identity-addressed streams

Runtime randomness is addressed by the smallest stable identity appropriate to
the mechanism:

| Component | Stable address |
|---|---|
| Sensor noise | root, trace ID, vehicle ID, measurement tick |
| RF shadowing | root, trace ID, pair-episode ID |
| RF blockage residual | root, trace ID, pair-episode ID, blockage namespace |
| RF fading | root, trace ID, pair-episode ID |
| Half-duplex/collision/decoding tape | root, trace ID, pair-episode ID, packet index, link, mechanism, attempt |
| Receiver feedback noise | root, trace ID, pair-episode ID, packet index, link |

Every packet tape is generated before action selection. RF-n and DUP-n still
share the same RF prefix, and VLC/DUP still share the optical draw. No stream
is advanced merely because another pair was evaluated first or because a
different population size was present.

## Stateful RF correction

Shadowing and fading must remain correlated along one pair trajectory, so they
cannot be regenerated independently per packet. Previously, their new-link
states came from one shared generator. That made a pair's initial channel draw
depend on how many other pairs had already entered the population.

Both channel processes now support a keyed generator factory. The rollout
creates one persistent generator per `(root, trace, pair-state namespace)` and
releases it with the corresponding correlated state. A rollout refuses to
switch traces while any pair state remains live. The campaign passes the same
experiment root to every trace; trace IDs, rather than source-list positions,
separate their streams.

## Scope boundaries

Mobility randomness is resolved when immutable trace artifacts are generated
and remains recorded in their manifests. Replaying a trace never redraws
mobility. Policy action sampling belongs to Phase 7 and uses the declared
policy seed; statistical bootstrap resampling belongs to analysis. Neither
stream is consumed by an environment step.

## Validation coverage

The tests establish that:

- identical reset seed and trace reproduce sensor rows, packet tapes, and
  feedback reports;
- an explicit reset seed changes each runtime stochastic path;
- global NumPy RNG mutations do not affect environment draws;
- pair iteration order cannot change correlated shadowing or fading;
- trace identity changes channel streams and live state cannot cross traces;
- feedback media have separate namespaces; and
- invalid or oversized seeds fail before a rollout begins.
