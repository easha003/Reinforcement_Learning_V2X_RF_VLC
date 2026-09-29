#!/usr/bin/env python3
"""Dry-run, execute, or resume the frozen longer-block RF frontier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hybrid_v2x_rl.agents.longer_block_execution import (
    LongerBlockCandidateResult,
    execute_longer_block_frontier,
    load_longer_block_progress,
    write_longer_block_progress,
)
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
        "--execute",
        action="store_true",
        help="run the propagation screen; default is a structural dry run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="override the declared output path; valid only with --execute",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="ignore matching progress and restart; valid only with --execute",
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
    root = args.project_root.expanduser().resolve()
    declaration = load_longer_block_frontier_declaration(
        args.declaration,
        project_root=root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_longer_block_dry_run(declaration, project_root=root)
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"candidates: {report['candidates']}")
        print(f"optical configurations: {report['optical_configurations']}")
        print(f"validation windows: {report['validation_windows']}")
        grids = cast(list[dict[str, object]], report["resource_grids"])
        for row in grids:
            name = cast(str, row["candidate_name"])
            channel_uses = cast(int, row["finite_blocklength_channel_uses"])
            rf4_airtime_s = cast(float, row["rf4_total_airtime_s"])
            deadline_fit = cast(bool, row["rf4_fits_deadline"])
            print(
                f"{name}: {channel_uses} uses, "
                f"RF-4={1000.0 * rf4_airtime_s:.1f} ms, "
                f"deadline_fit={deadline_fit}"
            )
        print("channel frames evaluated: 0")
        print("training performed: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    output = _resolve(root, args.out) if args.out is not None else declaration.output_path
    progress_path = output.with_suffix(".progress.json")
    completed: tuple[LongerBlockCandidateResult, ...] = ()
    if progress_path.is_file() and not args.restart:
        completed = load_longer_block_progress(
            progress_path,
            declaration=declaration,
        )
        print(
            f"resuming {len(completed)} / {len(declaration.candidates)} "
            f"completed candidates from {progress_path}",
            flush=True,
        )

    def progress(index: int, total: int, name: str) -> None:
        print(
            f"[longer-block {index:02d}/{total:02d}] starting {name}",
            flush=True,
        )

    def checkpoint(results: tuple[LongerBlockCandidateResult, ...]) -> None:
        write_longer_block_progress(
            progress_path,
            declaration=declaration,
            results=results,
        )
        latest = results[-1]
        print(
            f"[longer-block {len(results):02d}/{len(declaration.candidates):02d}] "
            f"completed {latest.candidate.name}; survives={latest.survives}",
            flush=True,
        )

    result = execute_longer_block_frontier(
        declaration,
        project_root=root,
        completed_results=completed,
        progress=progress,
        checkpoint=checkpoint,
    )
    written = result.write_json(output)
    for candidate_result in result.candidate_results:
        print(f"{candidate_result.candidate.name}:", flush=True)
        for row in candidate_result.rows:
            print(
                f"  {row['optical_configuration_name']} density "
                f"{row['density_vehicles_per_lane_km']:g}: "
                f"mean={row['mean_optimistic_propagation_only_conditional_miss_lower_bound']:.6e}, "
                f"budget_multiple={row['budget_multiple']:.6f}",
                flush=True,
            )
    decision = result.decision()
    print(f"shortest passing candidate: {decision['shortest_passing_candidate_name']}")
    print(
        f"joint contention frontier authorized: {decision['joint_contention_frontier_authorized']}"
    )
    print("training performed: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
