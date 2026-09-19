"""Fixed policies and oracle bounds on the held-out split, from the caches.

Run as::

    python scripts/oracle_bounds.py --split test

The published targeting bound answers one question: given a radio-only
baseline, which packets are worth duplicating? It is the right bound for a
policy whose only lever is redundancy, and it is *not* a floor for a policy
that can also **substitute**. Both single-link actions cost one, so on a packet
whose optical path is clean the light is a cheaper route to the same
reliability than the radio plus the light together.

So two bounds are reported.

``duplication_only``
    Baseline RF everywhere, upgrade packets to DUP in order of risk reduction.
    This is the published quantity and the one the fixed baselines sit against.

``full_action_set``
    Baseline the cheaper single link per packet, then upgrade to DUP the same
    way. This is the bound a three-action policy actually competes against,
    and it is the honest comparison for the learned policy.

Both are greedy and both are exact. Every upgrade costs the same one unit, so
a knapsack with uniform weights is solved optimally by taking the largest risk
reductions first -- there is no fractional relaxation to worry about.

Both are oracles twice over: they read each packet's true marginals, and they
choose after seeing them. No causal policy can reach either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_v2x_rl.config import headline_config_layers, load_config
from hybrid_v2x_rl.env.cache import TransitionCache
from hybrid_v2x_rl.observation.builder import ObservationSchema


def greedy_bound(single: np.ndarray, joint: np.ndarray, budget: float) -> tuple[float, float]:
    """Cheapest mean activation cost whose mean risk clears ``budget``."""

    n = len(single)
    total = float(single.sum())
    if total / n <= budget:
        return 1.0, total / n

    reduction = single - joint
    order = np.argsort(-reduction)
    gained = np.cumsum(reduction[order])
    remaining = total - gained
    feasible = np.flatnonzero(remaining / n <= budget)
    if feasible.size == 0:
        return 2.0, float(remaining[-1] / n)
    upgrades = int(feasible[0]) + 1
    return 1.0 + upgrades / n, float(remaining[feasible[0]] / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caches", type=Path, default=Path("artifacts/caches"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--densities", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/evaluations/oracle-bounds.json"))
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
    schema = ObservationSchema(
        features=tuple(config.observation.features),
        history_packets=int(config.observation.history_packets),
    )

    payload: dict = {"budget": budget, "split": args.split, "densities": {}}
    print(f"budget {budget:.0e}, split {args.split}\n")
    header = (f"{'rho':>4} {'RF only':>11} {'VLC only':>11} {'best single':>12} "
              f"{'DUP all':>11} | {'dup-only':>9} {'all-actions':>12}")
    print(header)
    print("-" * len(header))

    for density in args.densities:
        directory = args.caches / f"{args.split}-d{density}-{args.replicate:03d}"
        if not (directory / "manifest.json").exists():
            print(f"  ! missing {directory}", file=sys.stderr)
            continue
        cache = TransitionCache.load(directory, schema=schema)
        risk = np.asarray(cache.risk, dtype=np.float64)
        p_rf, p_vlc = risk[:, 0], risk[:, 1]
        joint = p_rf * p_vlc
        best_single = np.minimum(p_rf, p_vlc)

        duplication_only = greedy_bound(p_rf, joint, budget)
        full_actions = greedy_bound(best_single, joint, budget)

        entry = {
            "packets": cache.packets,
            "rf_only": {"miss": float(p_rf.mean()), "cost": 1.0},
            "vlc_only": {"miss": float(p_vlc.mean()), "cost": 1.0},
            "best_single_oracle": {"miss": float(best_single.mean()), "cost": 1.0},
            "duplicate_all": {"miss": float(joint.mean()), "cost": 2.0},
            "duplication_only_bound": {"cost": duplication_only[0],
                                       "miss": duplication_only[1]},
            "full_action_set_bound": {"cost": full_actions[0], "miss": full_actions[1]},
            "vlc_cheaper_share": float((p_vlc < p_rf).mean()),
        }
        payload["densities"][str(density)] = entry
        print(f"{density:>4} {p_rf.mean():>11.3e} {p_vlc.mean():>11.3e} "
              f"{best_single.mean():>12.3e} {joint.mean():>11.3e} | "
              f"{duplication_only[0]:>9.3f} {full_actions[0]:>12.3f}")

    print("\nshare of packets where the optical leg is the safer single link:")
    for density, entry in payload["densities"].items():
        print(f"  rho={density}: {entry['vlc_cheaper_share']:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
