# Phase 8 receive-diversity physical model

Status: implementation and structural wiring complete; the resumable
two-stage feasibility executor is validated and execution is in progress; PPO
training remains blocked

Date completed: 2026-09-28

## Implemented boundary

The RF channel now supports exactly two receiver contracts:

1. the established one-antenna SISO path; and
2. the frozen two-receive-antenna maximum-ratio-combining (MRC) path.

The default remains SISO. Receive diversity is an explicit external experiment
profile and does not alter the resolved project configuration or any historical
artifact when it is absent.

For the diffuse complex processes, branch two is constructed as

```text
z_2,corr = rho z_1 + sqrt(1-rho^2) z_2,
```

where `z_1` is the unchanged primary process and `z_2` comes from a separately
keyed persistent generator. Both processes retain the existing temporal and
inter-carrier frequency correlation. The configured Rician specular component
is then applied equally to both branches. Consequently:

- `rho = 0` gives independent diffuse innovations;
- `rho = 1` gives the identical-branch limit; and
- the primary branch is bit-exact with the original SISO stream.

The implementation rejects receive diversity when the fading process was built
from one shared sequential RNG. This prevents a second branch from shifting the
primary or another pair's future samples. Production rollouts already use keyed
generators derived from root seed, trace ID, and pair identity.

## MRC calculation

For primary and secondary instantaneous pre-loss SNRs, the decoder receives

```text
gamma_mrc = gamma_1 + 10^(-L_2/10) gamma_2.
```

Equivalently, because both branches share path loss, shadowing, transmit power,
and receiver-noise level, the effective fading multiplier is

```text
g_eff = g_1 + 10^(-L_2/10) g_2.
```

The existing finite-blocklength function is evaluated once at this combined
SNR. No abstract failure-probability divisor remains in the physical path.
Zero-loss equal-gain branches therefore produce the expected `3.0103 dB`
instantaneous combining gain. Increasing secondary-chain loss continuously
returns the result toward the primary SISO link and can never make that primary
link worse.

## Mechanisms that remain shared

The second receive antenna does not create another packet transmission or
access opportunity. These mechanisms remain exactly shared:

- large-scale path loss;
- propagation class and vehicle blockage;
- lognormal shadowing;
- the configured Rician specular component;
- RF collision and sensing state; and
- receiver half-duplex exposure.

Only diffuse small-scale fading receives a branch-specific innovation. This is
the boundary that prevents two antennas from manufacturing independent
contention or independent blockage.

## Authoritative-path integration

The optional `RFReceiveDiversity` profile now flows through:

```text
frozen receive profile
  -> RF channel/lifecycle construction
  -> keyed physical rollout
  -> pair-local RF propagation truth
  -> local attempt-risk assembly
  -> certificate-aware joint evaluator
```

The deterministic rollout fingerprint includes the physical profile whenever
one is explicitly supplied. The existing identity-addressed access/decoding
tape remains matched, and the primary fading branch remains matched between
SISO and MRC comparisons.

The receive-diversity structural validator now instantiates all ten frozen
receive profiles across all 18 source physical points: 180 physical profile
instances feeding the 360 predeclared fallback-view cells. It resolves all nine
validation windows but evaluates zero channel frames.

## Verification

The focused physical, rollout, declaration, and frontier suite covers:

- SISO arithmetic and explicit/default SISO equivalence;
- exact preservation of the keyed primary fading stream;
- the declared power-correlation/ECC proxy;
- independent and perfectly correlated branch limits;
- keyed-generator enforcement and pair-state release;
- secondary-chain loss before MRC;
- the `3.0103 dB` equal-branch limit;
- decoding-risk monotonicity relative to the primary branch;
- aligned per-attempt branch tapes in the legacy packet path;
- profile provenance in deterministic rollout fingerprints; and
- propagation truth reaching the authoritative pair-local evaluator path.

No trace outcome, feasibility verdict, or training decision is produced by
these tests.

## Next task

The frozen propagation-only necessary-condition screen and certificate-aware
joint executor are implemented; see `PHASE8_RECEIVE_DIVERSITY_EXECUTION.md`.
The immediate task is to complete that bounded validation run, inspect its
certificate-aware result, and leave PPO and the held-out test split blocked
unless the headline receive profile passes its full authorization gate. The
one-hour monitoring interval applies to the CPU execution.
