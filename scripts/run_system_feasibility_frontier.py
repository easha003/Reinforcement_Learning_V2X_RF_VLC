#!/usr/bin/env python3
"""Dry-run or execute the frozen pair-local system-feasibility frontier."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.system_feasibility_execution import (
    FrontierCellResult,
    execute_system_feasibility_frontier,
    load_frontier_progress,
    structural_dry_run,
    write_frontier_progress,
)
from hybrid_v2x_rl.agents.system_feasibility_frontier import (
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
        help="repository root containing configurations, traces, and evidence",
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        default=DEFAULT_DECLARATION,
        help="frontier declaration path relative to the project root",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run every expensive cell and write the final result; default is dry-run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="override the declared final result path; valid only with --execute",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="replace any matching cell-progress prefix; valid only with --execute",
    )
    return parser


def _resolve(project_root: Path, supplied: Path) -> Path:
    return supplied if supplied.is_absolute() else project_root / supplied


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    if args.restart and not args.execute:
        raise SystemExit("--restart is valid only with --execute")
    project_root = args.project_root.expanduser().resolve()
    declaration = load_system_feasibility_frontier_declaration(
        args.declaration,
        project_root=project_root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_dry_run(declaration, project_root=project_root)
        payload = report.as_dict()
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"validation windows: {len(report.windows)}")
        print(f"physical points: {payload['physical_points']}")
        print(f"evaluation cells: {payload['evaluation_cells']}")
        print("actor used: False")
        print("checkpoint used: False")
        print("test split opened: False")
        print("frontier executed: False")
        print("structural dry run: PASS")
        return 0

    output = (
        _resolve(project_root, args.out)
        if args.out is not None
        else declaration.output_path
    )
    progress_path = output.with_suffix(".progress.json")
    completed: tuple[FrontierCellResult, ...] = ()
    if progress_path.is_file() and not args.restart:
        completed = load_frontier_progress(
            progress_path,
            declaration=declaration,
        )
        print(
            f"resuming {len(completed)} / {len(declaration.evaluation_cells)} "
            f"completed cells from {progress_path}"
        )
    else:
        write_frontier_progress(
            progress_path,
            declaration=declaration,
            cells=(),
        )

    def progress(index: int, total: int, cell_id: str) -> None:
        print(f"[{index:02d}/{total:02d}] starting {cell_id}", flush=True)

    def checkpoint(cells: tuple[FrontierCellResult, ...]) -> None:
        write_frontier_progress(
            progress_path,
            declaration=declaration,
            cells=cells,
        )
        print(
            f"[{len(cells):02d}/{len(declaration.evaluation_cells):02d}] "
            f"completed {cells[-1].cell.cell_id}",
            flush=True,
        )

    result = execute_system_feasibility_frontier(
        declaration,
        project_root=project_root,
        progress=progress,
        completed_cells=completed,
        checkpoint=checkpoint,
    )
    written = result.write_json(output)
    decision = result.decision()
    print(f"current nominal verdict: {decision['current_system_nominal_verdict']}")
    print(f"current robust verdict: {decision['current_system_robust_verdict']}")
    print(f"training authorized: {decision['training_authorized']}")
    print(f"test split opened: {decision['test_split_opened']}")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
