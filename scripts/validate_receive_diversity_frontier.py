#!/usr/bin/env python3
"""Validate the frozen receive-diversity frontier without executing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_v2x_rl.agents.receive_diversity_frontier import (
    load_receive_diversity_frontier_declaration,
    structural_receive_diversity_dry_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path(
    "configs/evaluation/receive_diversity_system_feasibility_frontier.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--skip-evidence-verification",
        action="store_true",
        help="validate structure without hashing the frozen source evidence",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    declaration = load_receive_diversity_frontier_declaration(
        args.declaration,
        project_root=args.project_root,
        verify_evidence=not args.skip_evidence_verification,
    )
    print(json.dumps(structural_receive_diversity_dry_run(declaration), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
