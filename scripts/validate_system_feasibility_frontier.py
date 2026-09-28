#!/usr/bin/env python3
"""Validate the frozen pair-local system-feasibility frontier declaration."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.system_feasibility_frontier import (
    SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA,
    load_system_feasibility_frontier_declaration,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path(
    "configs/evaluation/pair_local_system_feasibility_frontier.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root containing configurations and frozen evidence",
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DEFAULT_DECLARATION,
        help="frontier declaration path, relative to the project root by default",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    declaration = load_system_feasibility_frontier_declaration(
        args.declaration,
        project_root=project_root,
        verify_evidence=True,
    )
    print(f"schema: {SYSTEM_FEASIBILITY_FRONTIER_DECLARATION_SCHEMA}")
    print(f"declaration SHA-256: {declaration.sha256}")
    print(f"miss budget: {declaration.miss_budget:.8g}")
    print(f"densities: {declaration.densities}")
    print(f"physical points: {len(declaration.physical_points)}")
    print(f"evaluation cells: {len(declaration.evaluation_cells)}")
    print(f"headline point: {declaration.headline_point.point_id}")
    print(f"frozen window source: {declaration.window_source}")
    print(f"planned output: {declaration.output_path}")
    print("test split opened: False")
    print("frontier executed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
