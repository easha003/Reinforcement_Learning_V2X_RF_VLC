# Hybrid RF/VLC Reinforcement Learning

Research code for constrained mean-field reinforcement learning over hybrid
NR sidelink and vehicular visible-light communication.

The repository begins from the validated mobility, geometry, RF, VLC,
observation, packet-lifecycle, and statistical components of the completed
RF/VLC feasibility study. The extension changes the decision problem: actions
must alter the shared RF load seen by the population, rather than selecting a
link against a fixed per-packet channel.

## Initial research objective

Learn a causal decentralized policy that minimizes activation and shared-pool
use while satisfying a per-density packet deadline-miss constraint. Training
may use centralized population information, but evaluation must expose each
vehicle only to declared causal observations.

The analytical fixed-point allocation is retained as an upper bound and
diagnostic baseline. It is not treated as a deployable policy because it can
use true optical risk.

See [`docs/RL_EXTENSION_SCOPE.md`](docs/RL_EXTENSION_SCOPE.md) for the initial
problem contract and acceptance gates.

## Repository layout

```text
src/hybrid_v2x_rl/
  mobility/       reproducible Manhattan-grid mobility traces
  geometry/       pair geometry and vehicle/building occlusion
  channels/       RF and VLC outcome models
  observation/    causal noisy observations and action-dependent feedback
  env/            packet lifecycle, replay cache, statistics, oracle allocation
  mean_field/     population-coupled environment (next implementation step)
  agents/         constrained learning algorithms (added after environment tests)
configs/          layered experiment configurations
scripts/          trace, cache, and baseline evaluation entry points
tests/            inherited simulator tests and new RL-environment tests
```

## Development setup

Python 3.11 or 3.12:

```bash
python -m pip install -e '.[dev,mobility,training]'
hybrid-v2x-rl doctor
pytest
```

Generated traces, caches, checkpoints, and evaluation outputs are intentionally
not committed.
