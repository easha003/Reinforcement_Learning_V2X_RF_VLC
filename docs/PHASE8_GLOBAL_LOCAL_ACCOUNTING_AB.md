# Phase 8 Global-versus-Local RF Accounting A/B

## Status

Complete on 2026-09-29. The resumable CPU run evaluated all 18 declared
cells, wrote the result artifact, performed no training, used no actor or
checkpoint, and did not open the test split.

Result artifact:

- `artifacts/evaluations/phase8_global_local_accounting_ab.json`
- SHA-256: `554f5a71966a00124c8ffa929b176f8912b4615cfae3d578fd3fb63decec107a`

## Question

Does replacing the legacy frame-global RF pool and population-mean
half-duplex approximation with the current 200 m pair-local pool and
endpoint-specific half-duplex schedule materially change hybrid RF/VLC packet
risk?

This experiment answers that question separately at 3 ms and 10 ms. It does
not treat the two deadlines as an A/B pair and does not compare them as if the
other physical settings were equal.

## Matched design

Each deadline uses the same nine frozen validation windows: three 16-frame
windows at each of densities 10, 20, and 30 vehicles per lane-kilometer. Every
cell therefore contains 69,626 transitions: 68,135 actor-usable rows and 1,491
contract-fallback rows.

For each of the nine policy actions, every actor-usable row selects that fixed
action. An unusable row selects the unchanged `DUP-4` contract fallback. The
authoritative current rollout is executed once. The legacy counterfactual is
then reconstructed from that rollout's exact complete action ledger, RF
propagation realization, VLC realization, and environment seed. Only these
two mechanisms change:

| Model | Contention domain | Half-duplex exposure |
|---|---|---|
| Legacy | One frame-global RF pool | Population-mean committed RF airtime |
| Current | Focal transmitter's 200 m local RF domain | Focal receiver's current transmit duty cycle |

The action, propagation truth, VLC truth, retry count, sensing band, and
randomness are identical within every matched pair. The reported endpoint is
conditional miss probability, not a finite sample of binary packet misses.

The `VLC` usable-row cell is a negative control: because it has no RF leg, its
legacy and current risk must be exactly equal. The all-row `VLC` view is not a
pure negative control because its 1,491 unusable rows use the `DUP-4` fallback.

### 3 ms profile

- 300 B / 3 ms / `1e-4`
- nominal sensing, wide 60-degree optical receiver, SISO RF
- historical `rf-capacity-1x`: two subchannels and 400 candidate resources

The 3 ms capacity point preserves the old declaration solely for this
historical accounting comparison. Its two-12-RB interpretation was superseded
by the full-carrier allocation correction and must not be reused as a current
physical-capacity claim.

### 10 ms profile

- 300 B / 10 ms / `1e-4`
- 2.0 ms QPSK RF attempt
- nominal sensing, wide 60-degree optical receiver
- four full-carrier resources per slot and 800 candidate resources
- two-branch independent-ideal integrated zero-loss MRC

This is the nominal profile selected by the explicit exploratory override.

## Campaign results

The table reports actor-usable rows only, so each row isolates the named fixed
action. `Reduction` is `(legacy global - pair local) / legacy global`.

| Deadline | Action | Legacy global mean | Pair-local mean | Reduction |
|---|---:|---:|---:|---:|
| 3 ms | VLC | `2.538067e-1` | `2.538067e-1` | 0.00% |
| 3 ms | RF-1 | `2.531772e-1` | `2.133694e-1` | 15.72% |
| 3 ms | RF-2 | `2.061789e-1` | `1.620614e-1` | 21.40% |
| 3 ms | RF-3 | `2.185037e-1` | `1.660804e-1` | 23.99% |
| 3 ms | RF-4 | `2.526988e-1` | `1.886053e-1` | 25.36% |
| 3 ms | DUP-1 | `5.751409e-2` | `4.543353e-2` | 21.00% |
| 3 ms | DUP-2 | `4.491373e-2` | `3.261181e-2` | 27.39% |
| 3 ms | DUP-3 | `4.673315e-2` | `3.269103e-2` | 30.05% |
| 3 ms | DUP-4 | `5.353697e-2` | `3.667280e-2` | 31.50% |
| 10 ms | VLC | `2.538067e-1` | `2.538067e-1` | 0.00% |
| 10 ms | RF-1 | `1.519561e-1` | `1.278649e-1` | 15.85% |
| 10 ms | RF-2 | `8.416767e-2` | `6.576355e-2` | 21.87% |
| 10 ms | RF-3 | `6.860418e-2` | `5.177844e-2` | 24.53% |
| 10 ms | RF-4 | `6.758126e-2` | `5.014981e-2` | 25.79% |
| 10 ms | DUP-1 | `3.474651e-2` | `2.748272e-2` | 20.91% |
| 10 ms | DUP-2 | `1.835224e-2` | `1.338529e-2` | 27.06% |
| 10 ms | DUP-3 | `1.463654e-2` | `1.040282e-2` | 28.93% |
| 10 ms | DUP-4 | `1.426406e-2` | `1.007814e-2` | 29.35% |

The negative control is exact at both deadlines. Across RF-involving actions,
the campaign-level reduction is 15.72%--31.50% at 3 ms and 15.85%--29.35% at
10 ms. The direction therefore does not depend on the deadline profile.

## Mechanism diagnosis

`RF-1` provides the cleanest mechanism view because every usable pair reserves
one RF attempt. The following values are means over usable RF rows.

| Profile | Density | Global collision | Local collision | Global half-duplex | Local half-duplex | Global utilization | Local utilization |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 ms | 10 | 0.07781 | 0.05652 | 0.00500 | 0.00310 | 0.5428 | 0.0791 |
| 3 ms | 20 | 0.20353 | 0.17190 | 0.00500 | 0.00425 | 1.5204 | 0.2402 |
| 3 ms | 30 | 0.31208 | 0.26417 | 0.00500 | 0.00463 | 2.4973 | 0.3781 |
| 10 ms | 10 | 0.03970 | 0.02875 | 0.02000 | 0.01242 | 1.0856 | 0.1583 |
| 10 ms | 20 | 0.10757 | 0.09088 | 0.02000 | 0.01698 | 3.0408 | 0.4804 |
| 10 ms | 30 | 0.17062 | 0.14307 | 0.02000 | 0.01854 | 4.9945 | 0.7561 |

The local pool removes 84.3%--85.4% of the global utilization attributed to a
typical focal domain, consistent with the earlier finding that 84.74% of
globally pooled flows lie outside that domain. Lower mean collision and
endpoint-correct half-duplex exposure both contribute to the packet-risk
improvement.

Pair-local accounting is not merely a uniform downward scale. For RF-only
campaign cells it gives lower row risk on roughly 74.5%--75.0% of usable rows,
while the legacy model gives lower risk on roughly 25%. The endpoint-specific
schedule and heterogeneous local domains can expose a focal receiver more than
the old population mean even though the old global pool overstates campaign
load overall. For DUP cells, roughly 74.3% of rows are exactly equal because a
zero-risk VLC leg makes the RF-accounting difference irrelevant; pair-local is
lower on about 19% and global is lower on about 6%.

## Conclusions and claim boundary

1. The pair-local migration is beneficial and physically necessary. It lowers
   mean conditional risk for every RF-involving fixed-action stress test at
   both deadlines while the VLC negative control remains unchanged.
2. The main removed bias is global contention overcounting. Endpoint-specific
   half-duplex accounting also matters and correctly redistributes risk across
   receivers instead of guaranteeing a lower value for every row.
3. This result does not establish the `1e-4` target. Homogeneous fixed-action
   sweeps are mechanism stress tests, not coordinated oracle policies or
   learned policies, and their means are intentionally high under contention.
4. The result does not justify comparing 3 ms directly with 10 ms: RF airtime,
   pool capacity, and receive diversity also differ between the profiles.
5. The old global joint-oracle result remains valid only for the superseded
   code model. Current feasibility and future PPO evaluation must use the
   pair-local/endpoint-correct environment.

The A/B therefore closes the question of whether the local correction helped:
it did, materially, without changing non-RF channel truth. It does not change
the next experimental decision. The next task remains freezing the exact
nominal 10 ms exploratory PPO profile and running the bounded smoke campaign,
with no exact-`1e-4` claim and no test-split access.
