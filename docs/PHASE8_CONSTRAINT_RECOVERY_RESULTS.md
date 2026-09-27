# Phase 8 bounded constraint-pressure recovery results

Date completed: 2026-09-27

## Verdict

The predeclared recovery gate passed. Both dual-pressure treatments produced a
material validation response; the entropy-only treatment did not. The frozen
selection rule chose `dual_init_10` because it achieved the largest
row-weighted increase in feasible action probability mass.

This is a recovery-of-learning result, not a reliability result. The selected
arm's row-weighted policy-expected validation risk is `2.6301e-2`, still about
263 times the `1e-4` target. It authorizes a fresh full seed-1001 run with the
single selected change, but that run must independently pass the existing
density-specific reliability gate before any paper checkpoint can be selected.

## Frozen provenance and protocol

- Code and protocol commit: `4c8ba75` (`Add bounded constraint recovery experiment`)
- Policy seed in every arm: `1001`
- Transition budget per arm: `300,000`
- Completed per arm: `297,266` transitions in 9 density-balanced PPO updates
- Common stop reason: `insufficient_budget_for_balanced_round`
- Unused tail per arm: 2,734 transitions
- Final active curriculum budget: `1e-4`
- State-regime audit schema: `hybrid-rf-vlc-rl.state-regime-audit.v3`
- State-regime audit SHA-256:
  `35b240c2fc58b46f02999d225663b6a91803509309cf098b4927e19aa6a7b2a7`
- Audit exposure: 9 frozen validation windows and 33,732 overlapping regime
  label rows in every arm; test split unopened
- Automated analysis SHA-256:
  `10758d76fe2c0e43f9be7dc63e669e07268486b2110669428bb343ac7d923987`

Every arm contains nine immutable checkpoints, nine v3 iteration reports, nine
JSONL metric rows, and v1 constraint-pressure diagnostics in every iteration.
All persisted numeric values are finite. Session checkpoint hashes match the
files on disk.

## Primary gate result

The feasibility and risk columns below are row-weighted over the five causal
regimes. Feasibility is probability mass on actions whose exact
policy-induced-load risk meets the active budget; risk is policy-expected
conditional miss probability under the same load.

| Arm | Feasible mass | Gain vs. control | Regimes with gain >= 0.05 | Expected risk | Risk/control | Gate |
|---|---:|---:|---:|---:|---:|---|
| `control` | 0.383729 | -- | -- | 0.190319 | 1.000000 | reference |
| `dual_lr_5` | 0.572582 | +0.188852 | 5 / 5 | 0.075169 | 0.394966 | pass |
| `dual_init_10` | **0.590858** | **+0.207129** | 4 / 5 | **0.026301** | **0.138194** | **pass; selected** |
| `entropy_005` | 0.379745 | -0.003984 | 0 / 5 | 0.195039 | 1.024801 | fail |

No treatment lost more than the allowed 0.02 feasible mass in any regime. The
selected arm's easy-state gain was +0.03114, below the per-regime improvement
threshold, while its other four gains were +0.15204 to +0.34318. The arm still
passes because the frozen rule required at least three regimes to improve by
0.05 and prohibited any material regression.

## Regime-conditioned selected-arm result

| Validation regime | Control feasible mass | Selected feasible mass | Gain | Selected deterministic risk |
|---|---:|---:|---:|---:|
| Easy state | 0.282587 | 0.313724 | +0.031136 | 0.018016 |
| Moderate RF conditions | 0.481336 | 0.824516 | +0.343180 | 0.005422 |
| Poor VLC / usable RF | 0.309801 | 0.461841 | +0.152041 | 0.018440 |
| Uncertain mixed state | 0.422377 | 0.622382 | +0.200005 | 0.027877 |
| Heavy RF contention / optical permitted | 0.415655 | 0.689603 | +0.273948 | 0.017777 |

The result does not assert the originally hypothesized hand-written action
table. It shows that the constrained actor can move probability toward the
counterfactually feasible set in every regime while lowering risk.

## Constraint-pressure mechanism

The final pre-update ratio
`mean(abs(lambda * A_cost)) / mean(abs(A_reward))` changed as follows:

| Arm | Density 10 | Density 20 | Density 30 |
|---|---:|---:|---:|
| `control` | 0.0384 | 0.0292 | 0.0293 |
| `dual_lr_5` | 1.7637 | 1.5063 | 1.5932 |
| `dual_init_10` | 0.7543 | 0.8934 | 1.1111 |
| `entropy_005` | 0.0393 | 0.0304 | 0.0301 |

Control and entropy therefore left the cost advantage at only about 3--4% of
reward-advantage magnitude. Faster ascent made constraint pressure dominate,
while initialization at 10 produced roughly balanced reward and constraint
pressure and the best validation response.

Final rollout conditional-miss estimates tell the same directional story:

| Arm | Density 10 | Density 20 | Density 30 |
|---|---:|---:|---:|
| `control` | 0.191594 | 0.226959 | 0.272163 |
| `dual_lr_5` | 0.131745 | 0.158024 | 0.140766 |
| `dual_init_10` | **0.054830** | **0.068986** | **0.064751** |
| `entropy_005` | 0.206898 | 0.224795 | 0.259256 |

The selected arm also avoided the early cheap-action concentration. In its
final pre-update policy, `RF-1` held only 3.1--4.5% mean probability across
densities, while `DUP-1` held 22.7--29.8%, VLC 15.1--33.6%, and RF-2/RF-3
retained meaningful density-dependent probability. This is evidence of
constraint-driven policy movement, not proof of convergence.

## Artifact identities

| Arm | Config hash | Final checkpoint SHA-256 | Evaluation SHA-256 |
|---|---|---|---|
| `control` | `57c9c84e63603c48049a4b0bbfd6fd1efcbf3dd25564b84cce9c8ba1220d1fec` | `e7992d84025b42302a0d1c5d230a8d81125ef339a9a78b65fc6c449826a819d0` | `c20f7ba1ff566b49976175a1212fc59cfd36b954c8084f19f2479b5e06a0bc45` |
| `dual_lr_5` | `6f13caa517b05ffc4b9417d24a17b5f73d012b553dcd4d752066fe5d11875507` | `a8caf95efa0726de97f393fb1b0fd82ce895b5984081d0774ee6485728805751` | `dd154a07176a30ccabcd186b9f13ec0f2ac78399aa42f61e781400bb8a95c6c7` |
| `dual_init_10` | `678fb7789aac9ab8ceb04856e5be2804e5ded731141134b6a71fcfb0b140f3da` | `61c2e1465a20fddc326afc4ecc520cf6981a3b5fa63df45069cd68e9cee856ea` | `cc34f451bafdddca9b1f4489947d0539149cb5df6d1733e474277a4e16495317` |
| `entropy_005` | `e2a1e2ce5b84110a8667438161baf113ea94c15013b436c540e773cf634e122f` | `21e6df4b7f6121b11bf4f08d891a7bfac6d25002f64d2386c2a1ea0048f08471` | `3ea5b468a050693f21c0ac85441a46bfd38f00ac8c6962a165395573779f9170` |

## Decision and next task

Use the headline configuration plus
`configs/training/recovery_dual_init_10.yaml` for a fresh full seed-1001 run.
Do not include the bounded 300,000-transition layer, do not resume any old
checkpoint, and do not combine dual initialization with the faster dual rate.
The run must start at transition zero on one frozen commit. Codex progress
checks remain once per hour for full training, as requested by the user.
