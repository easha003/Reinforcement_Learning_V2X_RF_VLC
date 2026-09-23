#!/usr/bin/env python3
"""Measure Phase 6 history and population-coupling opportunities."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.sequential_opportunity import (
    build_sequential_opportunity_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition-cache-root",
        type=Path,
        default=Path("artifacts/caches"),
        help="root containing train-d*-* and test-d*-* transition caches",
    )
    parser.add_argument(
        "--frame-cache-root",
        type=Path,
        default=Path("artifacts/frame_caches"),
        help="root containing verified population-frame caches",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluations/phase6_sequential_opportunity.json"),
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    parser.add_argument(
        "--minimum-history-rows",
        type=int,
        help="override the configured held-out packet threshold",
    )
    parser.add_argument(
        "--minimum-history-clusters",
        type=int,
        help="override the configured held-out pair-episode threshold",
    )
    return parser


def _resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else root / expanded


def _discover(root: Path, split: str) -> tuple[Path, ...]:
    paths = tuple(
        sorted(path for path in root.glob(f"{split}-d*-*") if (path / "manifest.json").is_file())
    )
    if not paths:
        raise FileNotFoundError(
            f"no {split} transition caches beneath {root}; "
            "run scripts/build_training_cache.py first"
        )
    return paths


def main() -> int:
    args = _parser().parse_args()
    config = load_headline_config(PROJECT_ROOT)
    transition_root = _resolve(PROJECT_ROOT, args.transition_cache_root)
    frame_root = _resolve(PROJECT_ROOT, args.frame_cache_root)
    output = _resolve(PROJECT_ROOT, args.out)
    report = build_sequential_opportunity_report(
        config,
        training_cache_paths=_discover(transition_root, "train"),
        evaluation_cache_paths=_discover(transition_root, "test"),
        frame_cache_root=frame_root,
        confidence_level=args.confidence,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        minimum_history_rows=args.minimum_history_rows,
        minimum_history_clusters=args.minimum_history_clusters,
    )
    report.write_json(output)
    print(
        f"history: status={report.history.status} "
        f"rows={report.history.evaluation_rows:,} "
        f"clusters={report.history.evaluation_clusters:,}"
    )
    print(
        f"population coupling: status={report.population.status} "
        f"densities={len(report.population.densities)}"
    )
    print(f"PPO gate: {report.gate_decision}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
