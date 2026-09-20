# Hybrid RF/VLC Reinforcement-Learning Extension — Work Plan

## Document purpose

This document defines the Phase 0–9 execution plan for the reinforcement-learning extension of the hybrid RF/VLC V2X system.

The central research question is:

> Can causal, decentralized RF/VLC decisions reduce activation cost and shared RF-pool use while satisfying a packet deadline-miss constraint, when the joint actions of the vehicle population change the RF contention experienced by every vehicle?

The plan deliberately builds an algorithm-independent, population-coupled environment and strong baselines before committing substantial compute to reinforcement learning. Primal-dual PPO is the primary learner to test, but PPO itself is not the proposed scientific contribution.

## Repository and data boundaries

- Implementation repository: `Hybrid_RF_VLC_RL`
- GitHub repository: `https://github.com/easha003/Reinforcement_Learning_V2X_RF_VLC`
- Completed paper repository: `CMDP_PPO` — read-only reference material for this extension
- Initial dataset: existing synthetic microscopic mobility traces
- Mobility role: exogenous environment input, not an offline behavior-policy dataset
- Generated traces, caches, checkpoints, and result files must remain outside Git unless they are intentionally small test fixtures
- Train, validation, and test trace splits must remain disjoint

## Fixed initial decisions

| Item | Initial decision |
|---|---|
| Agent | One active transmitter–receiver pair |
| Policy sharing | One policy shared by all agents |
| Decision timing | All active agents act synchronously at packet-generation frames |
| Mobility | Exogenous; communication actions do not change vehicle motion |
| Observation | Causal local information plus explicitly delayed/measured congestion feedback |
| Action space | Nine categorical RF/VLC actions |
| Reward | Negative RF-attempt and VLC-activation cost |
| Constraint | Deadline-miss probability, enforced separately at each traffic density |
| Primary learner | Primal-dual PPO, subject to the baseline go/no-go gate in Phase 6 |
| Final reliability evidence | Realized misses plus cluster-aware one-sided confidence bounds |

### Initial action space

| Action | VLC active | RF attempts |
|---|---:|---:|
| `VLC` | Yes | 0 |
| `RF-1` | No | 1 |
| `RF-2` | No | 2 |
| `RF-3` | No | 3 |
| `RF-4` | No | 4 |
| `DUP-1` | Yes | 1 |
| `DUP-2` | Yes | 2 |
| `DUP-3` | Yes | 3 |
| `DUP-4` | Yes | 4 |

The action set makes both medium selection and RF reservation intensity learnable. VLC-only traffic returns its RF reservation and therefore contributes no RF attempts to the shared pool.

## Definition of success

The extension succeeds only if the learned policy:

1. Is statistically feasible at every declared traffic density.
2. Uses causal information available at execution time.
3. Reduces activation cost, RF-pool use, or both relative to the strongest feasible deployable baseline.
4. Remains credible across policy seeds, independent mobility replicates, and declared sensitivity tests.
5. Preserves performance on an untouched test split after all modeling and checkpoint-selection choices are frozen.

An improvement in average reward alone is not sufficient. Reliability feasibility is checked first; resource efficiency is compared only among feasible policies.

## Phase overview

| Phase | Work | Completion gate |
|---:|---|---|
| 0 | Repository baseline | Inherited tests pass, trace locations are configured, and no generated data are committed |
| 1 | Formal environment specification | Every observation, action, reward, constraint, timing rule, tensor shape, unit, range, and causal restriction is documented |
| 2 | Frame-oriented trace layer | Deterministic chronological replay reproduces source counts, frames, agent births, and terminations |
| 3 | Population action model | RF and VLC accounting is exact for every action and population composition |
| 4 | Shared RF-pool dynamics | RF risk responds correctly and monotonically to simultaneous population demand |
| 5 | Mean-field environment | Long deterministic rollouts have no leakage, invalid actions, NaNs, or lifecycle errors |
| 6 | Baselines and oracle checks | Baselines are reproducible and an explainable sequential-control opportunity for RL exists |
| 7 | Primal-dual PPO | Unit tests verify returns, constraint estimation, dual direction, clipping, masking, and checkpoint restoration |
| 8 | Training | Learning is stable and at least one validation-feasible checkpoint exists per density |
| 9 | Final evaluation | The frozen policy meets the reliability rule and improves resource use over the strongest feasible deployable baseline |

---

## Phase 0 — Repository baseline

### Objective

Create a clean, reproducible starting point in the extension repository without importing completed-paper artifacts or generated datasets.

### Tasks

- [x] Confirm that `origin` points to `Reinforcement_Learning_V2X_RF_VLC`.
- [x] Review the extension repository structure and package name.
- [x] Define the supported Python version and install the project in an isolated environment.
- [x] Run all inherited unit and integration tests before changing behavior.
- [x] Record the baseline test result and dependency versions.
- [x] Configure paths for raw traces, caches, checkpoints, and results outside Git tracking.
- [x] Verify `.gitignore` coverage for generated and machine-specific files.
- [x] Create the first clean baseline commit when the repository contents are verified.

### Deliverables

- Reproducible local environment
- Passing inherited test suite
- External-data path convention
- Clean baseline commit

### Completion gate

All inherited tests pass, the working tree contains only intended source files, and no raw/generated dataset or experiment artifact is tracked.

---

## Phase 1 — Formal environment specification

### Objective

Freeze the mathematical and software contract before implementing the RL environment.

Status: complete in
[`RL_ENVIRONMENT_CONTRACT_V1.md`](RL_ENVIRONMENT_CONTRACT_V1.md), contract
version `1.0.0` and project configuration schema `1.1`.

### Tasks

- [x] Define one decision epoch and the ordering of observation, action, population aggregation, outcome generation, reward/cost assignment, feedback update, and time advance.
- [x] Define the local observation vector, feature units, normalization, valid ranges, missing-data behavior, and causal availability.
- [x] Define any mean-field input, such as delayed CBR, delayed RF load, or a population-action histogram.
- [x] Define the nine actions and their RF/VLC resource consequences.
- [x] Define reward as negative activation/resource cost, with separate coefficients for VLC activation and each RF attempt.
- [x] Define the deadline event and per-packet constraint cost.
- [x] Define the per-density constrained objective and dual variables.
- [x] Define pair birth, normal transition, trace-boundary truncation, and pair termination semantics.
- [x] Separate decentralized actor inputs from any centralized critic-only inputs.
- [x] Document all tensor shapes for variable population sizes and batching.

### Core formulation

For active pair `i` at frame `t`, the shared policy samples an action

```text
a_t^i ~ pi_theta(. | o_t^i, m_t)
```

where `o_t^i` is causal local information and `m_t` is an explicitly permitted delayed or measured population signal.

Population RF demand is action-dependent:

```text
D_t = sum_i RF_attempts(a_t^i)
```

A generic activation cost is

```text
g(a_t^i) = c_vlc * I[VLC active] + c_rf * RF_attempts(a_t^i)
```

with reward `r_t^i = -g(a_t^i)`. The primary constraint is the expected deadline-miss cost at each density `rho`:

```text
J_cost,rho(pi) <= epsilon_rho
```

The numerical cost coefficients, deadline definition, and reliability budgets must be read from versioned configuration rather than silently embedded in code.

### Deliverables

- [Versioned environment contract](RL_ENVIRONMENT_CONTRACT_V1.md)
- Observation and action schema in contract Sections 3 and 6
- Reward/constraint specification in contract Sections 1 and 4
- Timing and lifecycle specification in contract Sections 2 and 10
- Centralized-training/decentralized-execution boundary in contract Section 8

### Completion gate

Another researcher can implement a compatible environment from the written contract without guessing a unit, tensor shape, timing rule, or source of information.

---

## Phase 2 — Frame-oriented trace layer

Prerequisite status (2026-09-19): **passed**. The complete 21-trace campaign
passed Gate 1, full artifact-integrity verification, split reconciliation, and
pair-lifecycle reconciliation. See
[`TRACE_CAMPAIGN_VALIDATION.md`](TRACE_CAMPAIGN_VALIDATION.md). Phase 2
implementation began on 2026-09-19 with the verified chronological reader and
lifecycle types in `hybrid_v2x_rl.mean_field.frames`. It completed on
2026-09-19 after all 21 configured traces passed source-to-frame reconciliation
and immutable compact caches were published. See
[`PHASE2_FRAME_REPLAY_VALIDATION.md`](PHASE2_FRAME_REPLAY_VALIDATION.md).

### Objective

Convert the existing trace replay into a chronological population view that preserves simultaneous decisions.

### Why this is required

An episode-grouped, one-pair-at-a-time cache loses simultaneous population structure. It can also freeze RF collision risk before actions are selected, which prevents an agent from learning the system-level effect of moving traffic away from the RF pool.

### Tasks

- [x] Read raw trace records in chronological order.
- [x] Group all active pairs that share a decision time into one population frame.
- [x] Preserve trace ID, pair ID, timestamp, density, episode membership, and neighbor relationships.
- [x] Represent pair births, continuing pairs, natural terminations, and trace truncations explicitly.
- [x] Keep train, validation, and test trace membership immutable.
- [x] Make replay deterministic under a fixed seed and configuration.
- [x] Validate source counts: records, frames, unique pairs, births, terminations, and densities.
- [x] Add small synthetic fixtures that test asynchronous births and variable population sizes.

### Safe precomputation

The cache may precompute policy-independent quantities:

- Pair and neighbor geometry
- Noisy tracking inputs
- VLC optical geometry and occlusion inputs
- RF propagation state before population contention
- Episode and split membership

It must not precompute action-dependent quantities:

- Final RF collision probability
- Population CBR or RF demand
- RF outcomes that depend on selected reservation levels
- Mean-field state derived from current joint actions

### Deliverables

- Chronological frame reader/cache
- Frame and pair lifecycle data structures
- Deterministic replay tests
- Source-to-cache validation report

### Completion gate

Repeated replay produces identical frames; all counts reconcile with the source traces; and simultaneous active pairs, births, and terminations are preserved exactly.

---

## Phase 3 — Population action model

Status (2026-09-20): **complete**. The authoritative contract-`1.0.0`
nine-action indices and resource mapping are implemented in
`hybrid_v2x_rl.core.policy_actions`. Configuration order, RF-attempt counts,
VLC activations, costs, and rewards now derive from that single table. The
hardware/profile-only mask and missing-observation fallback are implemented in
`hybrid_v2x_rl.mean_field.action_masks`; transient channel truth and predictions
cannot enter that API. `hybrid_v2x_rl.mean_field.action_ledger` now binds one
validated action to every active pair in canonical frame order and computes
exact committed RF demand without merging flows that share a physical endpoint.
Each ledger row also derives RF use, VLC use, duplication, activation cost, and
resource reward from the authoritative mapping and configured cost coefficients.
VLC-only rows explicitly release the current-frame RF reservation; ledgers are
rebuilt from current actions so an earlier RF or DUP reservation cannot carry
forward. Lifecycle snapshots keep born and final agents in their active frame,
emit bootstrap semantics, and mark final pair state for release only after that
packet is processed; inactive IDs are rejected by exact action coverage. The
test matrix covers every action in isolation and all 81 ordered two-pair action
combinations, including non-unit configured resource prices. The
reusable conservation audit now reconciles active-pair count, RF attempts, VLC
activations, RF/VLC/duplication counts, RF releases, activation cost, and reward
exactly between immutable per-agent rows and published population totals. It
passes for all actions, mixed and empty populations, and fails closed with
field-level diagnostics if an aggregate drifts. The
inherited three-action packet interface remains isolated until the population
ledger can replace it without changing validated legacy behavior.

### Objective

Translate simultaneous agent actions into exact VLC activation and shared RF-pool demand.

### Tasks

- [x] Implement a single authoritative action-to-resource mapping.
- [x] Apply action masks when a medium or reservation choice is physically unavailable.
- [x] Aggregate RF attempts across all active pairs in the frame.
- [x] Track per-packet RF use, VLC use, duplication, and activation cost.
- [x] Ensure VLC-only decisions release their RF reservation.
- [x] Define deterministic handling for inactive, born, and terminated agents.
- [x] Test every action individually and in mixed populations.
- [x] Test accounting conservation across per-agent and population totals.

### Deliverables

- Population action ledger
- Action-mask rules
- Exact resource-accounting unit tests

### Completion gate

Per-agent accounting sums exactly to population totals for all nine actions, and VLC-only traffic contributes zero RF demand.

---

## Phase 4 — Shared RF-pool dynamics

Status (2026-09-20): **in progress**. The Phase 4 input boundary now accepts
only a complete, conservation-audited population action ledger and snapshots
the current frame's canonical per-pair RF reservations. Its unclipped
``offered_rf_attempts`` is exactly contract quantity ``D_t``; VLC-only and empty
populations contribute zero, and each new joint action produces a fresh demand
without carrying an earlier reservation forward. Mapping this demand to pool
utilization, measured CBR, and the declared collision sensitivity bands remains
the next task.

### Objective

Make RF reliability depend on the joint actions selected in the current frame.

### Tasks

- [x] Recompute offered RF load after all population actions are known.
- [ ] Map offered load to pool utilization, CBR, and collision probability using the validated RF model.
- [ ] Combine contention risk with policy-independent RF propagation state.
- [ ] Generate matched/counterfactual randomness where required for fair policy comparison.
- [ ] Expose only delayed or measured congestion information to the next actor observation.
- [ ] Test empty, light-load, saturation, and overload conditions.
- [ ] Test monotonicity: increasing offered RF load must not reduce collision risk under fixed channel conditions.
- [ ] Cross-check limiting cases against analytical calculations or the existing fixed-point oracle.

### Deliverables

- Action-coupled RF-pool model
- Collision/load validation suite
- Analytical limiting-case checks

### Completion gate

The RF-pool response is numerically stable, physically interpretable, monotonic in offered load, and consistent with reference calculations.

---

## Phase 5 — Mean-field environment

### Objective

Combine trace replay, local observations, joint-action coupling, channel outcomes, rewards, costs, and lifecycle handling into one testable environment.

### Decision-frame sequence

```text
Load all active pairs
        ↓
Construct causal local observations
        ↓
Shared policy selects all actions
        ↓
Aggregate RF attempts
        ↓
Calculate pool load and collisions
        ↓
Evaluate RF and VLC outcomes
        ↓
Assign rewards and constraint costs
        ↓
Update action-dependent link feedback and delayed congestion
        ↓
Advance to the next population frame
```

### Tasks

- [ ] Implement a Gymnasium-compatible API or document any intentional deviation.
- [ ] Support variable numbers of active agents with masks and stable agent IDs.
- [ ] Construct causal actor observations from trace state and past feedback only.
- [ ] Keep centralized critic information separate from actor observations.
- [ ] Produce reward, sampled binary miss cost, and simulator conditional-risk diagnostics.
- [ ] Handle pair birth without fabricated history.
- [ ] Handle pair termination and trace truncation correctly for return estimation.
- [ ] Seed every stochastic component.
- [ ] Add invariant checks for finite values, valid probabilities, valid actions, and legal lifecycle transitions.
- [ ] Run long deterministic rollouts with random and scripted policies.

### Deliverables

- Vectorized population environment
- Environment checker and invariant suite
- Deterministic rollout script
- Observation-normalization workflow fitted on training data only

### Completion gate

Long rollouts show no future leakage, invalid action, NaN, probability violation, identity error, or incorrect episode boundary.

---

## Phase 6 — Baselines and oracle checks

### Objective

Establish whether RL has a genuine opportunity and define the comparisons required for a publishable claim.

### Required baselines

1. Always RF, including each permitted RF-attempt level where meaningful.
2. Always VLC.
3. Duplicate-all.
4. Observable geometry-threshold selector.
5. Contextual policy without temporal/link history.
6. Supervised optical-risk estimator plus analytical load allocation.
7. Analytical truth-risk equilibrium as a non-deployable oracle bound.

### Tasks

- [ ] Implement every baseline through the same environment and accounting path.
- [ ] Use matched trace splits and matched random numbers for policy comparison.
- [ ] Verify expected ordering in simple/limiting scenarios.
- [ ] Report reliability and resource metrics at each density.
- [ ] Measure the gap between the best deployable baseline and oracle.
- [ ] Test whether link history or population coupling creates a sequential advantage over contextual selection.

### RL go/no-go gate

Proceed to full PPO implementation and training only if at least one of these opportunities remains:

- A meaningful cost or RF-use gap exists between the strongest deployable baseline and the oracle.
- Temporal/link-history information improves decisions beyond a contextual policy.
- Population-coupled actions create congestion-management behavior that fixed-risk selection cannot express.

If none remains, revise the research question or environment before spending substantial training compute.

### Deliverables

- Reproducible baseline suite
- Oracle comparison
- Phase 6 opportunity report
- Explicit PPO go/no-go decision

### Completion gate

Resource accounting and baseline ordering are explainable, comparisons are statistically consistent, and a concrete learnable gap has been demonstrated.

---

## Phase 7 — Primal-dual PPO

### Objective

Implement a shared categorical policy that minimizes activation cost while learning separate reliability multipliers for each traffic density.

### Initial architecture

- Shared categorical actor over nine actions
- Reward critic
- Cost critic
- Two hidden layers of 64 units per network, subject to smoke-test revision
- Feed-forward policy initially; recurrence is postponed
- Separate nonnegative dual variable for each density
- Centralized critic inputs permitted only when explicitly documented

### Tasks

- [ ] Implement masked categorical action sampling and deterministic evaluation.
- [ ] Implement reward and cost generalized advantage estimation.
- [ ] Distinguish true terminations from trace/time-limit truncations during bootstrapping.
- [ ] Implement PPO clipped actor and critic updates.
- [ ] Implement per-density dual ascent with nonnegative projection.
- [ ] Decide and document whether the actor uses sampled binary costs, conditional risk, or a staged combination.
- [ ] Log policy loss, value losses, entropy, KL divergence, clip fraction, explained variance, constraint estimates, and dual variables.
- [ ] Save model, optimizers, duals, normalization state, configuration, seed state, and training counters in checkpoints.
- [ ] Restore a checkpoint and verify bitwise-identical deterministic actions where practical.

### Required tests

- [ ] GAE across termination and truncation boundaries
- [ ] Cost-advantage and reward-advantage separation
- [ ] PPO clipping behavior
- [ ] Action-mask correctness
- [ ] Dual update increases after violation and decreases/stays projected at zero after slack
- [ ] Density-to-dual assignment
- [ ] Checkpoint save/restore
- [ ] Tiny learnable environment convergence

### Deliverables

- Primal-dual PPO implementation
- Training/checkpoint API
- Algorithm unit tests
- Tiny-environment learning test

### Completion gate

All mathematical unit tests pass, the learner solves a small known problem, and checkpoint restoration preserves the complete training state.

---

## Phase 8 — Training

### Objective

Train reproducible policies gradually, selecting checkpoints by validation reliability before resource cost.

### Training sequence

1. Overfit a tiny deterministic environment to confirm the full learning path.
2. Run short smoke training on one trace and one traffic density.
3. Profile environment throughput, memory, and bottlenecks on the MacBook Pro M2.
4. Train jointly across densities 10, 20, and 30 vehicles per lane-kilometer.
5. Apply the reliability curriculum:

   ```text
   1e-2 → 1e-3 → 1e-4
   ```

6. Add a `1e-5` target only as a declared secondary experiment after the `1e-4` pipeline is stable and statistically supportable.
7. Train five independent policy seeds.
8. Select checkpoints using validation feasibility first and resource cost second.
9. Freeze all choices before opening the test split.

### Initial configuration to profile

- Policy seeds: `1001, 1002, 1003, 1004, 1005`
- Initial transition budget: 10 million per seed
- Rollout size: 32,768 packet decisions
- Minibatch size: 1,024
- PPO epochs per update: 10
- Initial learning rate: `3e-4`
- Discount: `0.99`
- GAE lambda: `0.95`
- PPO clip ratio: `0.2`
- Initial entropy coefficient: `0.01`

These are starting values, not fixed scientific constants. Change them only using training/validation evidence, record every change in versioned configuration, and never tune on the final test split.

### Compute policy

The M2 is sufficient for environment development, tests, profiling, smoke runs, and likely small-to-medium feed-forward PPO experiments. A GPU is optional, not required for correctness. Decide whether full five-seed training needs external compute only after measuring transitions per second and estimating wall-clock time from the actual environment.

### Deliverables

- Tiny-overfit result
- Smoke-run report
- Throughput and wall-clock estimate
- Versioned full-run configurations
- Five-seed training logs and checkpoints
- Validation checkpoint-selection table

### Completion gate

Training is stable, runs are reproducible, diagnostics show no silent collapse, and at least one validation-feasible checkpoint is available for every declared density.

---

## Phase 9 — Final evaluation

### Objective

Evaluate frozen policies on untouched traces and determine whether the research claim is supported.

### Evaluation protocol

- [ ] Freeze code version, configuration, normalization statistics, checkpoints, and decision rules.
- [ ] Evaluate every policy seed on untouched test traces.
- [ ] Use independent mobility replicates as reliability clusters.
- [ ] Use matched random numbers for fair comparisons where valid.
- [ ] Report both pooled summaries and seed/replicate-level results.
- [ ] Apply the same feasibility test to RL and all baselines.
- [ ] Evaluate declared density and channel-model shifts.
- [ ] Run all required ablations.
- [ ] Record failures and infeasible cases rather than filtering them out.

### Required metrics by density and policy

- Deadline-miss probability
- Realized miss count and evaluated packet count
- One-sided cluster-aware reliability upper confidence bound
- RF attempts per packet
- RF-use fraction
- VLC-use fraction
- Duplication fraction
- Mean activation/resource cost
- RF-pool demand and utilization
- CBR and collision probability
- Performance near junctions and blockage transitions
- Results for every policy seed and mobility replicate
- Training and inference compute cost

### Required ablations

1. With versus without action-dependent link history.
2. Population-coupled versus fixed RF risk.
3. With versus without the mean-field/congestion signal.
4. Local-only versus centralized critic information during training.
5. Medium-only actions versus medium plus RF-attempt control.
6. Conditional-risk training versus sampled binary-cost training.
7. Densities outside the training support.
8. RF/VLC channel parameters outside the nominal training support.

### Statistical decision rule

For every declared density, a policy is feasible only when its one-sided cluster-aware reliability upper confidence bound is at or below the stated miss budget. Among feasible policies, lower activation cost and lower RF-pool use determine superiority. If no learned policy is feasible, the correct result is a failed constraint claim, not a cost comparison based only on point estimates.

### Deliverables

- Locked test-evaluation configuration
- Baseline, oracle, and learned-policy comparison tables
- Reliability confidence analysis
- Ablation and sensitivity results
- Reproducibility manifest
- Paper-ready figures and tables generated from immutable result files

### Completion gate

At least one frozen learned policy meets the statistical reliability requirement at every claimed density and improves cost or RF-pool use over the strongest feasible deployable baseline. Otherwise, report the limitation and narrow or revise the claim.

---

## Phase dependencies

```text
Phase 0: reproducible repository
   ↓
Phase 1: frozen environment contract
   ↓
Phase 2: population-frame trace replay
   ↓
Phase 3: joint action/resource ledger
   ↓
Phase 4: action-coupled RF-pool dynamics
   ↓
Phase 5: complete mean-field environment
   ↓
Phase 6: baselines, oracle, and RL go/no-go
   ↓
Phase 7: primal-dual PPO
   ↓
Phase 8: staged multi-seed training
   ↓
Phase 9: locked final evaluation
```

Later phases may be scaffolded early, but no phase should be treated as scientifically complete before its dependency gates pass.

## Experiment hygiene

- Use configuration files as the source of truth for model, environment, training, and evaluation settings.
- Store the Git commit, configuration hash, trace split, seed, package versions, and machine information with every run.
- Fit observation normalization on the training split only and save it with checkpoints.
- Never select hyperparameters or checkpoints using final test results.
- Preserve raw per-packet or per-cluster evaluation summaries needed to recompute confidence bounds.
- Separate exploratory runs from confirmatory final runs.
- Use identical evaluation code paths for learned policies and deployable baselines.

## Main research risks and responses

| Risk | Evidence to monitor | Response |
|---|---|---|
| No RL advantage over strong baselines | Baseline-oracle gap is negligible | Stop at Phase 6 and revise the research question |
| Sparse reliability violations destabilize learning | Cost critic and dual estimates have extreme variance | Use validated conditional-risk signals for training while retaining sampled misses for final evaluation |
| Policy exploits noncausal simulator state | Ablation or feature audit reveals future/truth information | Remove the feature and repeat training |
| RF population feedback is modeled incorrectly | Load monotonicity or limiting-case tests fail | Do not train; repair Phase 4 first |
| One policy hides density-specific violations | Aggregate performance passes while one density fails | Maintain separate density constraints and dual variables |
| Test leakage | Test traces influence tuning or normalization | Invalidate affected result and rerun with frozen choices |
| Full runs are too slow locally | M2 throughput estimate implies impractical wall time | Optimize/vectorize first, then move only full runs to suitable external compute |
| Statistical evidence is too weak at very low miss rates | Confidence upper bound remains above the target despite few observed misses | Increase independent evaluation exposure or narrow the reliability claim |

## Deferred scope

The following items are intentionally postponed until the synthetic-trace study is complete:

- Real-world mobility or communication datasets
- Calibration against real RF/VLC measurements
- Communication-dependent vehicle motion
- Agent-specific multi-agent networks
- Recurrent policy architecture
- Hardware-in-the-loop or road deployment

These are future extensions, not prerequisites for demonstrating the first population-coupled RL result.

## Immediate next step

Phases 2 and 3 are complete, and Phase 4 has begun with an immutable,
action-coupled RF-demand boundary that consumes the audited current-frame
ledger. Implement the next Phase 4 task by mapping its unclipped ``D_t`` to pool
utilization, clipped CBR, and per-attempt collision probability across the
configured sensing-reliability bands. Preserve the existing analytical-model
scope and do not integrate propagation outcomes yet.
