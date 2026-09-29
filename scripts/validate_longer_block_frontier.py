#!/usr/bin/env python3
"""Validate the frozen longer-block RF frontier without evaluating it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_v2x_rl.agents.longer_block_frontier import (
    load_longer_block_frontier_declaration,
    structural_longer_block_dry_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path("configs/evaluation/longer_block_rf_frontier.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--skip-evidence-verification",
        action="store_true",
        help="validate structure without hashing frozen result artifacts",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    declaration = load_longer_block_frontier_declaration(
        args.declaration,
        project_root=args.project_root,
        verify_evidence=not args.skip_evidence_verification,
    )
    print(
        json.dumps(
            structural_longer_block_dry_run(
                declaration,
                project_root=args.project_root,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
