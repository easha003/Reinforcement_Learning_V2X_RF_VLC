"""Can a policy see enough to reach the targeting bound?

Run as::

    python scripts/fitted_risk_check.py [--density 30] [--packets 200000]

The targeting bound assumes a policy knows each packet's failure probability
exactly. It does not: it sees a thirty-five column observation derived from
measurements that are \\SI{50}{\\milli\\second} stale, metre-noisy, and
propagated \\SI{200}{\\milli\\second} forward. The bound is therefore an upper
limit on what *any* risk-aware policy can achieve, and the interesting
question is how much of it survives the observation.

This answers that without training anything. For each packet it records the
observation a policy would see and the risk the channel actually assigned,
fits the second from the first, and re-solves the same knapsack on the fitted
values. The gap between the two costs is the price of imperfect sight.

Both legs are fitted separately, because the knapsack ranks by the *reduction*
one duplication buys and not by either risk on its own. The two are not
equally hard, and the asymmetry is the point: the radio's risk is
collision-dominated, so a contender count predicts it almost exactly, while
the joint risk carries the optical leg and depends on blocker bodies and
alignment the policy can only estimate. Handing a fit the true joint risk
therefore hands it the entire difficult half, which is why the leaky variant
is reported beside the honest one rather than deleted.

**It is a screening test, not a bound on PPO.** A ridge regression is a weak
learner, so a poor result here does not prove a neural policy would fail. A
*good* result, though, is close to conclusive in the other direction: if a
linear fit already recovers most of the headroom, the information is present
in the observation and the remaining question is only whether the optimiser
finds it.

It also under-reports for a reason specific to this script. The baseline it
drives is RF-only, so the optical leg is never spent, never measured, and its
ten quality columns stay at their unset value for every packet. That is the
feedback channel behaving correctly -- an action teaches nothing about the
link it did not use -- but it means the fit works from a strictly smaller
observation than a policy that sometimes duplicates would enjoy.

The screening costs minutes. Training five seeds for ten million transitions
each does not, and finding out afterwards that the features cannot support the
task is the expensive way to learn it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_v2x_rl.config import headline_config_layers, load_config
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.env.campaign import TargetingBound
from hybrid_v2x_rl.env.episodes import TraceSource, iter_pair_instants
from hybrid_v2x_rl.env.feedback import measurements
from hybrid_v2x_rl.env.packet import ACTIONS
from hybrid_v2x_rl.env.perception import build_perception
from hybrid_v2x_rl.env.rollout import always

WARMUP_S = 400.0


def collect(config, density: int, packets: int, split: str, replicate: int, seed: int):
    """Observations a policy would see, beside the risk the channel assigned."""

    root = Path.cwd()
    trace = root / "artifacts" / "traces" / f"synthetic-d{density}-{split}-{replicate:03d}"
    source = TraceSource.discover(trace)
    rollout = build_rollout(config, buildings=(), root_seed=seed)
    perception = build_perception(config, root_seed=seed)

    features: list[tuple[float, ...]] = []
    rf_risk: list[float] = []
    joint_risk: list[float] = []
    started = time.time()

    for instant in iter_pair_instants(
        source, generation_period_s=config.service.generation_period_s,
        max_packets=packets, warmup_s=WARMUP_S,
    ):
        observation = perception.observe(instant)
        if observation is None:
            continue
        _, _, alternatives = rollout.evaluate_instant(
            trace_id=instant.trace_id, pair_id=instant.pair_id, index=instant.index,
            density=float(density), time_s=instant.time_s,
            transmitter=instant.transmitter, receiver=instant.receiver,
            neighbours=instant.neighbours, index_of_frame=instant.index_of_frame,
            choose=always(ACTIONS[0]), counterfactual=True,
        )
        rf = alternatives["RF"]
        features.append(observation)
        rf_risk.append(rf.rf_failure_probability)
        joint_risk.append(alternatives["DUP"].joint_failure_probability)
        # The policy learns from what it did; here nothing is chosen, so the
        # link history is fed the baseline outcome to keep it moving. The
        # measurement goes through the feedback channel rather than straight
        # from the outcome, because the exact SINR is oracle-side.
        perception.record(
            instant.pair_id, action=rf.action, at_s=instant.time_s,
            delivered=rf.delivered,
            measurements=measurements(
                rf, root_seed=seed, trace_id=instant.trace_id,
                pair_id=instant.pair_id, packet_index=instant.index,
            ),
        )
        if instant.final:
            rollout.release(instant.pair_id)
            perception.release(instant.pair_id)

    print(f"  collected {len(features):,} packets in {time.time() - started:.0f}s")
    return np.asarray(features), np.asarray(rf_risk), np.asarray(joint_risk)


def fit_risk(x: np.ndarray, y: np.ndarray, *, train_fraction: float = 0.5):
    """Ridge regression on log-risk, fitted and scored on disjoint halves.

    Log-risk because the target spans orders of magnitude and a squared error
    on the raw probability would be dominated by the few worst packets --
    exactly the packets a policy most needs to rank correctly against each
    other rather than merely place above the rest.

    The split is by position rather than at random, so the fit is scored on
    packets from later in the trace than it saw. A random split would let a
    pair's own neighbouring packets appear on both sides, and those are
    correlated enough to flatter any model.
    """

    cut = int(train_fraction * len(x))
    target = np.log(np.clip(y, 1e-12, None))

    mean, std = x[:cut].mean(axis=0), x[:cut].std(axis=0)
    std[std < 1e-9] = 1.0
    design = lambda a: np.column_stack([(a - mean) / std, np.ones(len(a))])  # noqa: E731

    train, test = design(x[:cut]), design(x[cut:])
    ridge = 1e-3 * np.eye(train.shape[1])
    weights = np.linalg.solve(train.T @ train + ridge, train.T @ target[:cut])

    predicted = np.exp(test @ weights)
    residual = target[cut:] - test @ weights
    total = target[cut:] - target[cut:].mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total)
    spearman = _spearman(predicted, y[cut:])
    return predicted, y[cut:], r2, spearman, cut


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation: the knapsack only cares about ordering."""

    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb) / denominator if denominator else 0.0


def solve(rf: np.ndarray, joint: np.ndarray, order: np.ndarray, budget: float):
    """Cost of duplicating in ``order`` until the true mean risk clears ``budget``.

    The order is the policy's, the arithmetic is the channel's. That is the
    whole construction: a policy that ranks packets badly still pays the true
    miss rate of the packets it left alone.
    """

    total = float(rf.sum())
    n = len(rf)
    if total / n <= budget:
        return 1.0, total / n
    duplicated = 0
    for index in order:
        if total / n <= budget:
            break
        total -= rf[index] - joint[index]
        duplicated += 1
    return 1.0 + duplicated / n, total / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--density", type=int, default=30)
    ap.add_argument("--packets", type=int, default=200_000)
    ap.add_argument("--split", default="test")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--band", default=None,
                    choices=("optimistic", "nominal", "pessimistic"))
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/evaluations/fitted-risk.json"))
    ap.add_argument(
        "--config", type=Path, nargs="+", default=None,
        help="explicit config layers, in order; defaults to the frozen headline stack. "
             "Pass configs/service/ev2x_300B_3ms_1e-5.yaml in place of the "
             "1e-4 layer to run the eV2X row.",
    )
    args = ap.parse_args()

    _root = Path.cwd()
    config = load_config(
        tuple(args.config) if args.config else headline_config_layers(_root),
        project_root=_root,
    )
    budget = config.service.miss_budget
    print(f"density {args.density}, up to {args.packets:,} packets, budget {budget:.0e}")

    x, rf, joint = collect(config, args.density, args.packets, args.split,
                           args.replicate, args.seed)
    if len(x) < 1000:
        print("too few packets to fit", file=sys.stderr)
        return 1

    # Both legs must be fitted. The knapsack ranks by risk *reduction*, so a
    # policy handed the true joint risk would be told the answer for the term
    # that carries all the optical geometry -- and would look far better than
    # it is. Fitting each separately and differencing the predictions is the
    # only version of this test where the policy sees nothing privileged.
    predicted_rf, _, r2, rho, cut = fit_risk(x, rf)
    predicted_joint, _, r2_joint, rho_joint, _ = fit_risk(x, joint)
    held_rf, held_joint = rf[cut:], joint[cut:]

    perfect = TargetingBound()
    for a, b in zip(held_rf, held_joint, strict=True):
        perfect.observe(float(a), float(b))
    perfect_cost, perfect_rate = perfect.solve(budget)

    fitted_order = np.argsort(-(predicted_rf - predicted_joint))
    fitted_cost, fitted_rate = solve(held_rf, held_joint, fitted_order, budget)

    # What the leaky version would have reported, kept as a diagnostic: the
    # distance between the two is how much of the apparent skill was the
    # oracle's rather than the fit's.
    leaky_order = np.argsort(-(predicted_rf - held_joint))
    leaky_cost, _ = solve(held_rf, held_joint, leaky_order, budget)

    rng = np.random.default_rng(0)
    blind_order = rng.permutation(len(held_rf))
    blind_cost, _ = solve(held_rf, held_joint, blind_order, budget)

    # The knapsack never sees either risk alone, only their difference, so
    # this is the correlation that decides the cost.
    reduction = _spearman(predicted_rf - predicted_joint, held_rf - held_joint)

    print(f"\nheld-out packets: {len(held_rf):,}")
    print(f"log-risk R^2   RF {r2:.3f} (rank {rho:.3f})   "
          f"joint {r2_joint:.3f} (rank {rho_joint:.3f})")
    print(f"rank correlation on the risk *reduction* the knapsack ranks by: "
          f"{reduction:.3f}")
    print(f"\n{'strategy':<38}{'mean cost':>11}")
    print(f"{'duplicate everything':<38}{2.0:>11.3f}")
    print(f"{'random targeting':<38}{blind_cost:>11.3f}")
    print(f"{'fitted risk (observation only)':<38}{fitted_cost:>11.3f}")
    print(f"{'  same, but handed the true joint risk':<38}{leaky_cost:>11.3f}")
    print(f"{'perfect risk (targeting bound)':<38}{perfect_cost:>11.3f}")

    recoverable = 2.0 - perfect_cost
    captured = (2.0 - fitted_cost) / recoverable if recoverable > 1e-9 else float("nan")
    leaked = (2.0 - leaky_cost) / recoverable if recoverable > 1e-9 else float("nan")
    print(f"\na linear fit on the observation captures {captured:.1%} of the "
          f"headroom the bound identifies")
    print(f"an oracle joint risk would have inflated that to {leaked:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "density": args.density, "budget": budget, "band": args.band or "nominal",
        "held_out_packets": int(len(held_rf)), "log_risk_r2": r2,
        "rank_correlation": rho, "log_joint_risk_r2": r2_joint,
        "joint_rank_correlation": rho_joint, "reduction_rank_correlation": reduction,
        "duplicate_all_cost": 2.0,
        "random_targeting_cost": blind_cost, "fitted_cost": fitted_cost,
        "oracle_joint_cost": leaky_cost, "headroom_with_oracle_joint": leaked,
        "perfect_cost": perfect_cost, "fitted_rate": fitted_rate,
        "perfect_rate": perfect_rate, "headroom_captured": captured,
    }, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
