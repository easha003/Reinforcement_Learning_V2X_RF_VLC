# Phase 8 receive-diversity frontier declaration

Status: complete; the frozen experiment, physical implementation, resumable
executor, and validation screen are finished, with zero surviving profiles

Date frozen: 2026-09-28

## Decision

The next feasibility experiment uses one SISO control and a two-receive-branch
maximum-ratio-combining (MRC) intervention. It does **not** divide a failure
probability by an assumed gain. It must construct two correlated complex fading
branches, combine their instantaneous SNRs, and then pass that SNR through the
same finite-blocklength decoder already used by the SISO link.

The frozen MRC rule is

```text
gamma_mrc = gamma_1 + 10^(-L_2/10) gamma_2,
```

where `L_2` is the declared implementation loss on the second receiver branch.
Both receiver chains have equal-variance independent noise and perfect
per-attempt channel-state information. Those are explicit optimistic
assumptions, not hidden properties of the trace dataset.

Only the second branch's diffuse small-scale fading receives a new random
innovation. Both branches share path loss, propagation/blockage state,
lognormal shadowing, the configured Rician specular component, RF contention,
and receiver half-duplex exposure. Redrawing any of those mechanisms
independently would manufacture reliability that two antennas on the same
receiving vehicle do not provide.

## Calibration boundary

The declaration is literature-bounded rather than field-calibrated to this
repository's synthetic Manhattan traces.

- Abbas, Kåredal, and Tufvesson measured four antenna locations on two vehicles
  at 5.6 GHz across highway, urban, and rural routes. Their results support
  complementary roof/bumper placement and show that MRC can materially exceed
  selection combining. They also report that 2--4 m RF cables can add about
  3.5--7 dB attenuation, unless processing is moved close to the antenna:
  [DOI 10.1109/LAWP.2013.2250243](https://doi.org/10.1109/LAWP.2013.2250243).
- Sathyanarayanan et al. report a fabricated 5.9-GHz vehicular antenna with
  far-field envelope correlation coefficient (ECC) below `0.03`. Under the
  declared equal-power isotropic approximation, the conservative boundary
  `rho_h = sqrt(0.03) = 0.173205...` is used as a proxy for diffuse complex
  branch-innovation correlation:
  [DOI 10.1038/s41598-026-44515-3](https://doi.org/10.1038/s41598-026-44515-3).

The second source measures antenna-port/far-field correlation, not dynamic
on-road fading correlation. It therefore anchors the headline synthetic model
but does not establish a real-vehicle reliability guarantee. A high-correlation
case remains in the grid specifically to expose dependence on that limitation.

## Frozen sensitivity axes

The branch-correlation levels are:

| Level | Diffuse complex correlation `rho_h` | Role |
|---|---:|---|
| independent ideal | `0.0` | optimistic mathematical boundary |
| low-correlation hardware bound | `sqrt(0.03)` | headline literature-bounded case |
| correlated stress | `0.7` | adverse sensitivity, not a fitted coefficient |

The secondary-branch implementation losses are:

| Level | Loss | Role |
|---|---:|---|
| integrated zero loss | `0 dB` | optimistic local-processing boundary |
| short cable | `3.5 dB` | headline lower measured cable-loss bound |
| long cable | `7 dB` | adverse upper measured cable-loss bound |

The control plus the `3 x 3` MRC cross-product produces ten receive profiles.
Only
`rx2-mrc__low-correlation-hardware-bound__short-cable-loss` can participate in
a future training-authorization decision. The independent, zero-loss, and
stressed profiles are sensitivity evidence; a favorable result in one of them
alone cannot unblock PPO.

## Reused evidence and experiment size

The frozen declaration is
`configs/evaluation/receive_diversity_system_feasibility_frontier.yaml`,
SHA-256:

```text
9f607285d9b1b65fc411a0e69d69975d7d1731086134e02cc8d3c3736690193e
```

The declaration consumes the corrected full-carrier frontier at SHA-256
`60718bcdead081ad77b794ef64daadb054fb8c0fa2f052fbe4334f67a82d99c7`.
It retains the existing:

- 1/2/4 complete 10-MHz carrier capacity levels;
- nominal, pessimistic, and optimistic sensing bands;
- wide and concentrated optical configurations;
- contract and all-usable fallback views; and
- nine frozen validation windows over densities 10, 20, and 30.

Ten receive profiles times 18 source physical points times two fallback views
produce 360 predeclared evaluation cells before screening.

Execution is frozen in two stages. A cheap propagation-only necessary-condition
screen may remove a profile only when its optimistic lower bound already exceeds
`1e-4`. Every surviving profile must then use the certificate-aware pair-local
joint evaluator. No result may add a new profile, relax an axis, or substitute
the earlier abstract RF-failure divisor.

## Structural result

Run:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_receive_diversity_frontier.py
```

The structural validation resolves ten receive profiles, 18 source physical
points, two fallback views, and 360 cells. It instantiates all 180 physical
profile/point combinations, resolves nine validation windows, evaluates zero
channel frames, does not read the test split, uses no actor or checkpoint, and
cannot authorize training.

## What remains

The declaration alone does not show that `1e-4` is feasible. The keyed
correlated second-branch fading and MRC calculation are now implemented and
connected to the pair-local evaluator; see
`PHASE8_RECEIVE_DIVERSITY_PHYSICAL_MODEL.md`. The frozen propagation-only
screen and certificate-aware runner are documented in
`PHASE8_RECEIVE_DIVERSITY_EXECUTION.md`. All ten profiles fail the necessary
condition, including the headline profile at densities 20 and 30. Therefore
the survivor grid is empty, PPO remains blocked, and the next task is a
propagation-tail decomposition rather than another learner run.
