#!/usr/bin/env python3
"""Dry-run or execute the frozen RF-decoding reliability-scaling diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.reliability_scaling_diagnostic import (
    execute_reliability_scaling_diagnostic,
    load_reliability_scaling_declaration,
    structural_scaling_dry_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path(
    "configs/evaluation/rf_decoding_reliability_scaling.yaml"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root containing configurations, traces, and evidence",
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DEFAULT_DECLARATION,
        help="scaling declaration path relative to the project root",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="evaluate frozen validation frames and write the result",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="override the declared result path; valid only with --execute",
    )
    return parser


def _resolve(project_root: Path, supplied: Path) -> Path:
    return supplied if supplied.is_absolute() else project_root / supplied


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    project_root = args.project_root.expanduser().resolve()
    declaration = load_reliability_scaling_declaration(
        args.declaration,
        project_root=project_root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_scaling_dry_run(declaration)
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"validation windows: {report['validation_windows']}")
        print(f"improvement factors: {report['improvement_factors']}")
        print(f"views: {report['views']}")
        print(f"density rows: {report['density_rows']}")
        print("joint action search used: False")
        print("training authorized: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    def progress(index: int, total: int, window: object) -> None:
        print(f"[{index:02d}/{total:02d}] evaluating {window}", flush=True)

    result = execute_reliability_scaling_diagnostic(
        declaration,
        progress=progress,
    )
    output = (
        _resolve(project_root, args.out)
        if args.out is not None
        else declaration.output_path
    )
    written = result.write_json(output)
    decision = result.decision()
    print(
        "contract minimum declared factor: "
        f"{decision['minimum_declared_factor_by_view']['contract-dup4']}"
    )
    print(
        "all-rows minimum declared factor: "
        f"{decision['minimum_declared_factor_by_view']['all-rows-oracle-controlled']}"
    )
    print("training authorized: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
