#!/usr/bin/env python3
"""Run the bounded causal-regime and counterfactual-feasibility audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.state_regime_audit import build_state_regime_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root containing configs and immutable trace artifacts",
    )
    parser.add_argument("--windows-per-trace", type=int, default=3)
    parser.add_argument("--frames-per-window", type=int, default=16)
    parser.add_argument("--threshold-max-rows", type=int, default=250_000)
    parser.add_argument("--threshold-seed", type=int, default=73)
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--minimum-rows", type=int, default=10_000)
    parser.add_argument("--minimum-clusters", type=int, default=200)
    parser.add_argument(
        "--replicates-per-density",
        type=int,
        help="bounded diagnostic only; omit to include every configured replicate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluations/phase8_state_regime_audit.json"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    config = load_headline_config(project_root)
    report = build_state_regime_audit(
        config,
        windows_per_trace=args.windows_per_trace,
        frames_per_window=args.frames_per_window,
        threshold_max_rows=args.threshold_max_rows,
        threshold_seed=args.threshold_seed,
        environment_seed=args.environment_seed,
        minimum_rows=args.minimum_rows,
        minimum_clusters=args.minimum_clusters,
        replicates_per_density=args.replicates_per_density,
    )
    output = args.out if args.out.is_absolute() else project_root / args.out
    report.write_json(output)
    supported = sum(bool(row["supported"]) for row in report.rows)
    print(
        f"state-regime coverage: supported={supported}/{len(report.rows)} "
        f"all_supported={report.all_regimes_supported}"
    )
    print(f"test split opened: {report.as_dict()['test_split_opened']}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
