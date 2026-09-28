#!/usr/bin/env python3
"""Dry-run or execute the frozen receive-diversity feasibility frontier."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.receive_diversity_execution import (
    PropagationScreenProfileResult,
    ReceiveDiversityCellResult,
    execute_receive_diversity_frontier,
    load_receive_diversity_progress,
    write_receive_diversity_progress,
)
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
        help="declaration path relative to the project root",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the screen and surviving joint cells; default is dry-run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="override the declared final result path; valid only with --execute",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard matching progress and restart; valid only with --execute",
    )
    return parser


def _resolve(root: Path, supplied: Path) -> Path:
    return supplied if supplied.is_absolute() else root / supplied


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    if args.restart and not args.execute:
        raise SystemExit("--restart is valid only with --execute")
    project_root = args.project_root.expanduser().resolve()
    declaration = load_receive_diversity_frontier_declaration(
        args.declaration,
        project_root=project_root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_receive_diversity_dry_run(
            declaration,
            project_root=project_root,
        )
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"receive profiles: {report['receive_profiles']}")
        print(f"source validation windows: {report['source_validation_windows']}")
        print(
            "evaluation cells before screening: "
            f"{report['evaluation_cells_before_screening']}"
        )
        print(f"physical profile instances: {report['physical_profile_instances']}")
        print("channel frames evaluated: 0")
        print("training performed: False")
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
    screens: tuple[PropagationScreenProfileResult, ...] | None = None
    completed: tuple[ReceiveDiversityCellResult, ...] = ()
    if progress_path.is_file() and not args.restart:
        screens, completed = load_receive_diversity_progress(
            progress_path,
            declaration=declaration,
        )
        survivors = sum(result.survives for result in screens)
        total = survivors * len(declaration.source_frontier.evaluation_cells)
        print(
            f"resuming completed propagation screen and {len(completed)} / "
            f"{total} joint cells from {progress_path}",
            flush=True,
        )

    def progress(stage: str, index: int, total: int, label: str) -> None:
        print(
            f"[{stage} {index:03d}/{total:03d}] starting {label}",
            flush=True,
        )

    def checkpoint(
        screen_results: tuple[PropagationScreenProfileResult, ...],
        cells: tuple[ReceiveDiversityCellResult, ...],
    ) -> None:
        write_receive_diversity_progress(
            progress_path,
            declaration=declaration,
            screen_results=screen_results,
            cells=cells,
        )
        if cells:
            print(
                f"[joint-frontier {len(cells):03d}] completed "
                f"{cells[-1].cell_id}",
                flush=True,
            )
        else:
            survivors = sum(result.survives for result in screen_results)
            print(
                "[propagation-screen] completed: "
                f"{survivors} / {len(screen_results)} profiles survive",
                flush=True,
            )

    result = execute_receive_diversity_frontier(
        declaration,
        project_root=project_root,
        progress=progress,
        screen_results=screens,
        completed_cells=completed,
        checkpoint=checkpoint,
    )
    written = result.write_json(output)
    decision = result.decision()
    print(
        "headline survived propagation screen: "
        f"{decision['headline_survived_propagation_screen']}"
    )
    print(f"training authorized: {decision['training_authorized']}")
    print(f"test split opened: {decision['test_split_opened']}")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
