#!/usr/bin/env python3
"""Dry-run, execute, or resume the exploratory joint override."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hybrid_v2x_rl.agents.exploratory_joint_override import (
    execute_exploratory_joint_override,
    load_exploratory_joint_override_declaration,
    load_exploratory_joint_progress,
    structural_exploratory_joint_dry_run,
    write_exploratory_joint_progress,
)
from hybrid_v2x_rl.agents.system_feasibility_execution import FrontierCellResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path("configs/evaluation/exploratory_joint_override.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the three joint cells; default is a structural dry run",
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


def _worst_mean(cell: FrontierCellResult) -> float:
    return max(
        cast(float, row["mean_pair_local_candidate_conditional_miss_risk"])
        for row in cell.density_rows
    )


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    if args.restart and not args.execute:
        raise SystemExit("--restart is valid only with --execute")
    root = args.project_root.expanduser().resolve()
    declaration = load_exploratory_joint_override_declaration(
        args.declaration,
        project_root=root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_exploratory_joint_dry_run(
            declaration,
            project_root=root,
        )
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"RF candidate: {report['selected_rf_candidate']}")
        print(f"receive profile: {report['selected_receive_profile']}")
        print(f"optical configuration: {report['selected_optical_configuration']}")
        print(f"propagation lower bound: {report['propagation_only_worst_density_mean']:.6e}")
        print(f"validation windows: {report['validation_windows']}")
        plans = cast(list[dict[str, object]], report["cells"])
        print(f"joint cells: {len(plans)}")
        for plan_row in plans:
            print(
                f"  {plan_row['cell_id']}: subchannels={plan_row['subchannels']}, "
                f"band={plan_row['sensing_band']}, fallback={plan_row['fallback_view']}"
            )
        print("channel frames evaluated: 0")
        print("training performed: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    output = _resolve(root, args.out) if args.out is not None else declaration.output_path
    progress_path = output.with_suffix(".progress.json")
    completed: tuple[FrontierCellResult, ...] = ()
    if progress_path.is_file() and not args.restart:
        completed = load_exploratory_joint_progress(
            progress_path,
            declaration=declaration,
        )
        print(
            f"resuming {len(completed)} / {len(declaration.selected_cells)} "
            f"completed cells from {progress_path}",
            flush=True,
        )

    def progress(index: int, total: int, cell_id: str) -> None:
        print(f"[override-joint {index:02d}/{total:02d}] starting {cell_id}", flush=True)

    def checkpoint(cells: tuple[FrontierCellResult, ...]) -> None:
        write_exploratory_joint_progress(
            progress_path,
            declaration=declaration,
            cells=cells,
        )
        latest = cells[-1]
        print(
            f"[override-joint {len(cells):02d}/{len(declaration.selected_cells):02d}] "
            f"completed {latest.cell.cell_id}; verdict={latest.verdict}; "
            f"worst_mean={_worst_mean(latest):.6e}",
            flush=True,
        )

    result = execute_exploratory_joint_override(
        declaration,
        project_root=root,
        completed_cells=completed,
        progress=progress,
        checkpoint=checkpoint,
    )
    written = result.write_json(output)
    for cell_result in result.cells:
        print(
            f"{cell_result.cell.cell_id}: verdict={cell_result.verdict}",
            flush=True,
        )
        for row, verdict in zip(
            cell_result.density_rows,
            cell_result.density_verdicts,
            strict=True,
        ):
            print(
                f"  density {row['density_vehicles_per_lane_km']:g}: "
                f"candidate={row['mean_pair_local_candidate_conditional_miss_risk']:.6e}, "
                f"lower={row['mean_certified_conditional_miss_lower_bound']:.6e}, "
                f"verdict={verdict}",
                flush=True,
            )
    decision = result.decision()
    print(f"exact system target met: {decision['exact_system_target_met']}")
    print(f"best joint cell: {decision['best_joint_cell_id']}")
    print(f"best joint worst-density mean: {decision['best_joint_worst_density_mean']}")
    print(f"nominal training cell: {decision['nominal_training_cell_id']}")
    print(f"exploratory training authorized: {decision['exploratory_training_authorized']}")
    print("training performed: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
