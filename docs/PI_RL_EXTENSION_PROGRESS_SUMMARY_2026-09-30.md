# RL Extension Progress Summary for PI

**Date:** 2026-09-30  
**Project:** Constrained reinforcement learning for hybrid RF/VLC V2X resource
allocation  
**Service objective:** 300-byte packet, 10 ms deadline, packet miss probability
at most `1e-4` (at least 99.99% modeled delivery reliability)

## Executive summary

The RL environment, causal observation boundary, nine-action RF/VLC interface,
constrained PPO implementation, reproducible training/checkpoint pipeline, and
pair-local population physics are operational. PPO responds to constraint
pressure and materially improves its policy, but the current modeled system
does not meet the `1e-4` target. The work identified and corrected an important
modeling issue: the original RF pool counted the complete Manhattan population
as mutually contending even though the declared physical contention domain was
200 m. Correct pair-local contention improves RF-involving mean packet risk by
approximately 16%--32%, but the remaining gap is dominated by true local load
and rare states where RF is NLOS while the single direct VLC path is
geometrically unavailable.

VLC is already performing its intended complementary role. Under the current
10 ms pair-local fixed-action stress test, adding VLC to RF-4 reduces miss risk
by 41.1%, 72.9%, and 81.7% at densities 10, 20, and 30, respectively. It helps
most where RF contention is highest. Nevertheless, the corresponding DUP-4
risks remain `8.69e-4`, `7.58e-3`, and `1.35e-2`, above the target. The next
physical side project therefore studies optical spatial/angular diversity and
longer optical coding rather than further PPO tuning alone.

## Research question

Can a causal, decentralized policy reduce communication-resource use while
satisfying a strict deadline-miss constraint by selecting among VLC-only,
one-to-four RF attempts, and RF/VLC duplication, when all vehicles' actions
jointly affect local RF contention?

The work deliberately separates three questions:

1. Is the physical/action system capable of meeting the reliability target?
2. Can a policy infer useful actions from causal observations without simulator
   truth?
3. If the target is infeasible, how much failure is caused by policy regret
   versus the physical and population-load floor?

## Implemented system

- Nine categorical actions: `VLC`, `RF-1` through `RF-4`, and `DUP-1` through
  `DUP-4`.
- Exact resource accounting for every action and complete joint population
  decision.
- Causal noisy observations with temporal history, forecasts, delayed link
  feedback, action masks, and a training-only centralized critic.
- Conditional reliability cost plus sampled packet outcomes.
- Reproducible identity-addressed random tapes and deterministic
  checkpoint/resume behavior.
- Primal-dual PPO with density-specific constraint multipliers and staged
  reliability curriculum.
- Synthetic Manhattan mobility at 10, 20, and 30 vehicles per lane-kilometer,
  with frozen train/validation/test separation.
- Corrected 200 m pair-local RF contention, geometry-aware sensing, and
  endpoint-specific half-duplex scheduling.

## PPO training findings

The first full seed-1001 engineering run processed 9,998,802 of 10,000,000
configured transitions across 265 updates and all three curriculum stages. It
was numerically stable and exercised all training traces, but approximately
99.58% of its actions collapsed to `RF-1` or `VLC`; its final conditional miss
estimates remained approximately 0.065--0.072 under the `1e-4` target.

A bounded constraint-pressure experiment showed that the learner responds to
the constrained objective:

- feasible-action probability mass increased from `0.383729` to `0.590858`;
- policy-expected risk decreased from `0.190319` to `0.026301`;
- the predeclared selection rule chose initial dual multipliers of 10.

A fresh full seed-1001 run with that change reduced last-20 training risk by
approximately 76%--83% and frozen-validation risk by approximately 78%.
Feasible-action mass increased from `0.502278` to `0.655464`. However, all 13
observed validation density/regime cells remained approximately 59--200 times
above the `1e-4` target. The policy concentrated mainly on `VLC` and `RF-2`,
while the counterfactual evaluator frequently required `RF-4` or `DUP-4`.

This establishes that PPO learns in response to constraint pressure, but not
that PPO can overcome an infeasible physical/load system.

## Policy regret versus feasibility

The residual-feasibility decomposition found that approximately 95.0%--99.5%
of selected PPO risk was action regret. Better action learning therefore still
has substantial value. At the same time, the unilateral minimum-action floor
under the learned population load was:

| Density | Minimum-action conditional miss floor | Target verdict |
|---:|---:|---|
| 10 | `6.13e-5` | Pass |
| 20 | `2.23e-4` | Fail |
| 30 | `4.46e-4` | Fail |

Thus policy improvement alone could close much of the gap, but the population
load floor still prevented a target claim at densities 20 and 30.

## Correction from global to pair-local RF contention

The original population-joint oracle used a frame-global RF pool and reported
miss floors of `5.40e-4`, `1.48e-3`, and `1.85e-3`. A domain audit then showed:

- median frame-global/local active-flow ratio: 6.55;
- 84.74% of globally pooled flows lay outside the focal 200 m domain;
- no audited row had a genuinely frame-global 200 m contention domain.

The environment was therefore migrated atomically to pair-local contention,
pair-specific sensing/collision response, and endpoint-specific half-duplex.
Pre-migration PPO checkpoints are analysis-only and cannot be resumed for new
optimization.

A matched 18-cell A/B held actions, propagation, VLC truth, validation windows,
and randomness constant. Pair-local accounting reduced usable-row campaign
mean risk by:

| Action class | 3 ms reduction | 10 ms reduction |
|---|---:|---:|
| RF actions | 15.72%--25.36% | 15.85%--25.79% |
| DUP actions | 21.00%--31.50% | 20.91%--29.35% |
| VLC negative control | 0.00% | 0.00% |

The local model removes approximately 84.3%--85.4% of global utilization
incorrectly attributed to a typical focal domain. It does not uniformly lower
every row: endpoint-specific accounting correctly increases risk for some
receivers that are more exposed than the old population mean.

## Physical feasibility studies

The corrected pair-local 3 ms system-feasibility frontier evaluated 36 cells
over RF capacity, sensing uncertainty, optical FOV, and fallback treatment.
Every cell was certified infeasible. Even diagnostic zero-contention,
zero-half-duplex lower bounds were:

| Density | Certified lower bound | Budget multiple |
|---:|---:|---:|
| 10 | `2.271923e-4` | 2.27× |
| 20 | `6.895343e-4` | 6.90× |
| 30 | `3.897709e-4` | 3.90× |

The RF allocation model was then corrected so one 300-byte transport block
uses a complete 10 MHz/24-RB carrier allocation rather than half-carrier
subchannels. Receive-diversity, longer-block, and deadline-edge screens were
performed under this corrected mapping.

For the 10 ms extension:

- increasing RF airtime to 2.0 ms QPSK made densities 10 and 30 pass the
  propagation-only screen, but density 20 remained at `1.723841e-4`;
- the continuous 2.475 ms deadline-edge calculation reached only
  `1.294405e-4`;
- combining 2.0 ms QPSK with independent-ideal zero-loss two-branch MRC reached
  a best propagation-only worst-density mean of `1.182919e-4`, 18.29% above
  target;
- a user-authorized exploratory pair-local joint characterization gave
  nominal candidate means of `8.5773e-4`, `7.4735e-3`, and `1.3032e-2` at
  densities 10, 20, and 30.

The nominal joint worst-density result is therefore about 130.3 times the
target. It is an exploratory configuration, not an exact-feasibility result.

## Present contribution of VLC

The current 10 ms pair-local fixed-action results demonstrate complementary
behavior directly:

| Density | RF-4 miss | DUP-4 miss | Reduction from adding VLC |
|---:|---:|---:|---:|
| 10 | `1.4751e-3` | `8.6931e-4` | 41.1% |
| 20 | `2.7957e-2` | `7.5800e-3` | 72.9% |
| 30 | `7.3644e-2` | `1.3496e-2` | 81.7% |

Across the campaign, RF-4 has mean miss risk `5.0150e-2`, standalone VLC has
`2.5381e-1`, and DUP-4 has `1.0078e-2`. The combination is much stronger than
either medium alone because their dominant failure mechanisms differ.

The remaining risk is sharply concentrated. At density 20, only 16 of 23,204
rows carry essentially all of the earlier propagation floor; they combine RF
NLOS with occluded/out-of-FOV VLC. At density 30, the material rows combine RF
NLOS with beam-not-aimed or occluded VLC. A single concentrated 30-degree
receiver improves clear-path link budget but does not change this geometric
tail.

## Defensible conclusions

The following statements are supported:

1. The software and learning pipeline is reproducible and capable of learning
   under a constrained objective.
2. The initial global RF contention model was physically over-broad; the
   pair-local correction is both physically necessary and materially
   beneficial.
3. VLC already complements RF, with the largest measured relative benefit at
   the highest traffic density.
4. The present single direct VLC path is unavailable on rare RF-hard rows, so
   it cannot close the `1e-4` tail.
5. More PPO tuning cannot establish feasibility when the physical/joint gate
   fails.
6. The current work supports an exploratory policy-learning study and a
   mechanism-based negative/feasibility result, not a claim of achieving
   `1e-4`.

The following statements are not supported:

- real-world 99.99% reliability;
- compliance with a specific critical-maneuver standard;
- daytime, adverse-weather, pitch/grade, or lane-change robustness;
- independent optical blockage without calibrated evidence;
- resuming an old global-pool PPO checkpoint in the corrected environment.

## Next work

Two tracks are now separated:

1. **Main PPO track:** freeze the selected nominal 10 ms pair-local profile and
   run a bounded fresh PPO smoke campaign to study state-dependent RF/VLC
   behavior and oracle regret, without claiming `1e-4`.
2. **VLC diversity side project:** on branch `vlc-diversity-frontier`, audit
   RF-conditioned VLC failure and evaluate angular/spatial optical receiver
   diversity plus bounded longer VLC coding. Only propagation-screen survivors
   proceed to pair-local joint feasibility and, later, fresh PPO.

The side-project plan is recorded in
`docs/VLC_DIVERSITY_FRONTIER_SIDE_PROJECT_PLAN.md`.

## Suggested short verbal update

> We completed the constrained-PPO infrastructure and two full seed-1001
> engineering runs. The learner responds strongly to constraint pressure, but
> the current physical and population-load model does not meet the `10^-4`
> packet-miss target. We identified and corrected a significant global
> contention overcount: about 85% of globally pooled flows were outside a
> focal vehicle's 200 m contention region, and the pair-local correction lowers
> RF-involving mean risk by roughly 16% to 32%. VLC is already complementary;
> adding it to RF-4 lowers miss risk by 41%, 73%, and 82% from low to high
> density. The remaining failure is concentrated in rare RF-NLOS states where
> the single direct VLC path is also blocked or misaligned. We are therefore
> keeping exploratory PPO on the main path and moving a physically bounded VLC
> spatial/angular-diversity frontier to a separate branch.
