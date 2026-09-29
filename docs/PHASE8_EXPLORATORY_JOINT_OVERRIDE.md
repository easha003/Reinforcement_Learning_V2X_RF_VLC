# Phase 8 exploratory joint override

Status: user-authorized, declared, and implemented; joint execution pending

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
