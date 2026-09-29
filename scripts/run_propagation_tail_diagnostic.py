#!/usr/bin/env python3
"""Dry-run or execute the frozen propagation-tail decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.propagation_tail_diagnostic import (
    execute_propagation_tail_diagnostic,
    load_propagation_tail_declaration,
    structural_propagation_tail_dry_run,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path("configs/evaluation/propagation_tail_decomposition.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root containing source evidence and validation traces",
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DEFAULT_DECLARATION,
        help="declaration path relative to the project root",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="evaluate the bounded validation windows; default is dry-run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="override the declared output path; valid only with --execute",
    )
    return parser


def _resolve(root: Path, supplied: Path) -> Path:
    return supplied if supplied.is_absolute() else root / supplied


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    project_root = args.project_root.expanduser().resolve()
    declaration = load_propagation_tail_declaration(
        args.declaration,
        project_root=project_root,
        verify_evidence=True,
    )
    dry_run = structural_propagation_tail_dry_run(
        declaration,
        project_root=project_root,
    )
    if not args.execute:
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"receive profile: {dry_run['receive_profile']}")
        print(f"optical configurations: {dry_run['optical_configurations']}")
        print(f"validation windows: {dry_run['validation_windows']}")
        print(f"declared frame evaluations: {dry_run['frames']}")
        print("channel frames evaluated: 0")
        print("training performed: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    output = _resolve(project_root, args.out) if args.out is not None else declaration.output_path

    def progress(index: int, total: int, label: str) -> None:
        print(
            f"[propagation-tail {index:02d}/{total:02d}] starting {label}",
            flush=True,
        )

    result = execute_propagation_tail_diagnostic(
        declaration,
        project_root=project_root,
        progress=progress,
    )
    written = result.write_json(output)
    for row in result.rows:
        print(
            f"{row['optical_configuration_name']} density "
            f"{row['density_vehicles_per_lane_km']:g}: "
            f"mean={row['mean_selected_risk']:.6e}, "
            f"budget_multiple={row['budget_multiple']:.6f}",
            flush=True,
        )
    print("training performed: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
