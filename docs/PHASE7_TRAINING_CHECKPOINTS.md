# Phase 7 training checkpoints

## Contract

`agents.checkpointing.save_training_checkpoint` writes one versioned binary
artifact after a complete rollout/update boundary.  A checkpoint contains all
mutable training state, not merely actor weights:

- actor, reward-critic, and cost-critic parameter and buffer state;
- the three independent Adam optimizer states;
- every per-density dual multiplier and its update count;
- the complete observation-normalization snapshot;
- the canonical resolved project configuration and its SHA-256 configuration
  hash;
- the selected policy seed;
- cumulative training counters; and
- Python, NumPy, and PyTorch global RNG states plus every explicitly supplied
  NumPy and PyTorch generator.

The schema identifier is
`hybrid-rf-vlc-rl.training-checkpoint.v1`.  The payload uses only primitive
containers and tensors accepted by `torch.load(..., weights_only=True)`.
Arbitrary Python training objects are never pickled into the artifact.

## Counters and save boundary

`TrainingCounters` records completed PPO iterations, total environment
transitions, learning-eligible transitions, completed episodes, and optimizer
steps.  Learning transitions may not exceed physical environment transitions,
and the latter may not exceed the configured per-seed training budget.  Every
Adam parameter step must equal the saved optimizer-step counter, and a density
dual cannot have more updates than completed training iterations.

Saving is allowed only between decision frames.  The normalizer already owns
that lifecycle invariant and rejects snapshots while a frame is open.  This
prevents a checkpoint from representing model state after an action while its
normalization state still represents the period before that action.

## Configuration binding

The saver fails closed if the selected seed is absent from `policy_seeds`, if
actor width differs from the normalization contract, if actor/critic hidden
layers or PPO hyperparameters differ from the resolved training configuration,
if critic widths disagree, or if density labels and projection limits disagree
with the dual controller.  Actor and critic input widths are stored explicitly
instead of being inferred later from weight shapes.  The full canonical
configuration is embedded for auditability; its hash binds later restoration
to the intended run.

Restoration compares both.  The digest catches ordinary physics/training
drift, while exact canonical comparison also catches fields intentionally
excluded from the artifact hash, such as trace-split membership.  A checkpoint
therefore cannot silently resume against a different training, validation, or
test partition even when the run digest is unchanged.

## Random state

The training driver must pass every live explicit NumPy and PyTorch generator.
Empty generator maps are rejected so a caller cannot silently produce a
nominally resumable checkpoint without policy/minibatch or environment stream
positions.  Generator names are stored in sorted order.  Global Python, legacy
NumPy, PyTorch CPU, and available accelerator RNG states are also captured.
Named NumPy streams must use the project's declared PCG64 bit generator, so
every checkpoint accepted by the saver is also supported by the loader.

The project creates NumPy streams from stable experiment identities, but their
current bit-generator positions still matter when saving inside a continuing
episode.  Capturing both identities (configuration and policy seed) and states
supports either episode-boundary reconstruction or exact mid-run continuation.

## Portability and publication

All model, optimizer, and RNG tensors are detached, cloned, and moved to CPU
before writing.  A checkpoint created on CPU, MPS, or CUDA can therefore be
inspected on a CPU-only host and later mapped onto the selected training
device.

Writes use a temporary file in the destination directory, flush and `fsync`
the bytes, verify that the temporary artifact can be read in weights-only mode,
and publish it atomically as a hard link.  Existing paths and symlinks are
rejected: iteration checkpoints are immutable evidence, not mutable “latest”
files.  The returned summary reports byte length and an artifact SHA-256.

## Restoration

`agents.checkpointing.restore_training_checkpoint` reads only regular,
non-symlink files through `torch.load(..., weights_only=True)`.  Callers may
supply the SHA-256 returned by the saver; any byte-level mismatch is rejected
before state construction.  The loader validates exact top-level and nested
schemas, configuration and seed identity, network dimensions, model tensor
keys and shapes, optimizer parameter groups and steps, dual projection bounds,
normalization metadata, counters, and RNG representations.

Only after all primitive state has been parsed does the loader construct new
training owners.  It rebuilds the PPO updater from the resolved configuration,
loads all three networks and Adam optimizers, restores the density controller
through its public atomic restore boundary, and rebuilds the normalizer and
named generators.  The resulting `RestoredTrainingState` exposes those owners,
the counters and policy seed, checkpoint/software identity, and read-only maps
of mutable named RNG objects.

The destination device is explicit and defaults to CPU.  CPU, CUDA, and MPS
are accepted only when available; named generator states retain their original
device because random algorithms are not assumed portable across backends.
Unavailable saved global accelerator states are retained as artifact evidence
but are installed only when that backend exists.

Global RNG installation is transactional.  Model reconstruction necessarily
uses PyTorch's CPU generator, so the loader snapshots the caller's global
Python, NumPy, CPU, and available accelerator states first.  On any failure it
restores those entry states.  A successful call installs checkpoint globals by
default; `restore_global_rng=False` reconstructs the training objects while
leaving the caller's globals unchanged.

## Deterministic parity

The round-trip test evaluates a mixed-mask actor probe before saving and after
restoring on CPU.  Logits, selected action indices, selected log probabilities,
and entropies are required to be bitwise equal.  It also compares complete
optimizer state, reproduces the next named NumPy and PyTorch draws, applies the
same next PPO minibatch to the original and restored learners, and requires
identical metrics and resulting parameters.

## Tests

Unit tests create non-empty Adam state, advance one density dual and named RNG
streams, save a checkpoint, and verify every required section.  They also prove
CPU portability, weights-only readability, immutable destination behavior,
configuration/seed drift rejection, explicit-generator requirements, and
counter/optimizer invariants.  Restore tests add exact deterministic-action and
next-update parity, named and global RNG continuation, exact split binding,
artifact-digest rejection, malformed-model rejection, and proof that a failed
restore leaves global RNG state unchanged.
