# Phase 8 exploratory joint override

Status: completed; exact feasibility fails, but the explicitly authorized
exploratory training path is ready to be frozen under the nominal sensing cell

## Purpose and claim boundary

The combined receiver/block frontier found no exact or 10%-near-feasible
candidate. The best observed propagation-only pair reached `1.182919e-4`,
18.29% above the `1e-4` target. The user explicitly chose to continue toward
training with the best improvement rather than keep changing the physical
model to force an exact pass.

This document records that decision as an exploratory override. It does not
change the completed frontier, relax the scientific target, or convert failure
into feasibility. Any later paper result must state that the selected system
does not establish the `1e-4` reliability requirement.

## Frozen selected pair

- 300 B payload and 10 ms deadline
- 2.0 ms QPSK RF attempt with four current slots per attempt
- Four-action-attempt ceiling and canonical nine-action policy space
- Independent-ideal, integrated zero-loss, two-branch MRC
- Wide-60° optical configuration
- Propagation-only limiting mean: `1.182919e-4`
- Exact-budget multiple: 1.182919

The selected receiver is an optimistic sensitivity assumption, not the
hardware-primary model. This limitation must remain visible in training and
publication claims.

## Joint characterization grid

To prioritize the best attainable exploratory reliability without rerunning
inferior resource levels, the joint run uses the already-declared maximum
full-carrier capacity, `rf-capacity-4x` (four 10 MHz carrier allocations and
800 selection resources). It evaluates the contract `DUP-4` fallback under
all three already-declared sensing bands:

- nominal;
- pessimistic;
- optimistic.

Each cell covers all nine frozen validation windows and densities 10, 20, and
30. The existing exact-assignment cap and search-iteration limit are retained.
No adaptive cell, receiver, optical profile, or fallback diagnostic may be
added after execution starts.

The exact `1e-4` verdict is still computed and reported, but it does not gate
the explicitly authorized exploratory path. Once all three cells complete,
the nominal sensing cell becomes the training configuration and the other two
cells bound sensing-model sensitivity.

## Reproduction commands

Structural validation, with zero evaluated frames:

```bash
.venv/bin/python scripts/run_exploratory_joint_override.py
```

Execute or resume the three joint cells:

```bash
.venv/bin/python scripts/run_exploratory_joint_override.py --execute
```

The runner checkpoints after each cell and writes the final result to
`artifacts/evaluations/phase8_exploratory_joint_override.json`. It performs no
PPO training and does not open the test split.

## Completed joint result

The declaration SHA-256 is
`1be2914d82183b3e8c81e9cf17f189fce8b62bcf7045ab4d7a164507fbef4c7c`.
The completed result SHA-256 is
`7c463a464246ed15af6f7f634e2776c1587cef3315c142cc8a1121ebcdd643a4`.
All three cells evaluated 69,626 transitions over 117 nonempty frames. Their
realizable candidate means are:

| Sensing band | Density 10 | Density 20 | Density 30 | Worst/target |
|---|---:|---:|---:|---:|
| Optimistic | `8.130105e-4` | `7.132345e-3` | `1.245986e-2` | 124.60× |
| Nominal | `8.577271e-4` | `7.473536e-3` | `1.303173e-2` | 130.32× |
| Pessimistic | `9.296924e-4` | `8.014698e-3` | `1.391849e-2` | 139.18× |

Density 20 is formally infeasible in every sensing cell because its certified
lower bounds are `1.695127e-4`, `1.719182e-4`, and `1.757055e-4`, respectively.
At densities 10 and 30, the realizable candidates are far above budget but the
certificate gaps remain open, so their verdicts are inconclusive rather than
proven infeasible. Every complete cell is nevertheless infeasible because the
density-20 proof is sufficient.

The nominal oracle allocation uses `DUP-4` for 59.95% of transitions, `DUP-3`
for 35.20%, and `RF-4` for 4.85%; all other actions are unused. Mean RF demand
is 3.648 attempts per pair. This is not a learned policy, but it establishes
that the best realizable joint assignment relies heavily on maximum
duplication while contention and half-duplex losses still dominate.

## Training consequence

The full-system result is not close to `1e-4`: the nominal worst-density
candidate is approximately 130 times the target. The earlier 18.29% excess was
only a propagation-only necessary condition and did not include shared RF
access losses.

Under the recorded user override, exploratory PPO remains permitted for the
nominal cell. Its purpose is now to study whether PPO learns sensible hybrid
RF/VLC decisions and how closely it approaches the realizable joint oracle,
not to demonstrate satisfaction of the reliability constraint. The next task
is to freeze a training profile containing this exact physical configuration,
run a short smoke campaign, and verify action/regime learning before committing
to full independent seeds. The test split remains closed.
