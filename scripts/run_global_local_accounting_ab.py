#!/usr/bin/env python3
"""Dry-run, execute, or resume the matched global/local accounting A/B."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hybrid_v2x_rl.agents.global_local_accounting_ab import (
    execute_global_local_accounting_ab,
    load_global_local_accounting_ab_declaration,
    load_global_local_accounting_ab_progress,
    structural_global_local_accounting_ab_dry_run,
    write_global_local_accounting_ab_progress,
    write_global_local_accounting_ab_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path("configs/evaluation/global_local_accounting_ab.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the 18 matched cells; default is a structural dry run",
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


def _campaign_means(cell: dict[str, object]) -> tuple[float, float, float]:
    campaign = cast(dict[str, object], cell["campaign"])
    usable = cast(dict[str, object], campaign["usable_rows_only"])
    global_mean = float(cast(float, usable["legacy_global_mean_conditional_miss_risk"]))
    local_mean = float(cast(float, usable["pair_local_mean_conditional_miss_risk"]))
    reduction = usable["relative_risk_reduction"]
    return global_mean, local_mean, float(cast(float, reduction)) if reduction is not None else 0.0


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    if args.restart and not args.execute:
        raise SystemExit("--restart is valid only with --execute")
    root = args.project_root.expanduser().resolve()
    declaration = load_global_local_accounting_ab_declaration(
        args.declaration,
        project_root=root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_global_local_accounting_ab_dry_run(
            declaration,
            project_root=root,
        )
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"validation windows: {report['validation_windows']}")
        print(f"matched cells: {report['expected_cells']}")
        for profile in cast(list[dict[str, object]], report["profiles"]):
            print(
                f"  {profile['profile_id']}: deadline={float(cast(float, profile['deadline_s'])) * 1e3:g} ms, "
                f"subchannels={profile['subchannels']}, cells={len(cast(list[object], profile['cells']))}"
            )
        print("channel frames evaluated: 0")
        print("training performed: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    output = _resolve(root, args.out) if args.out is not None else declaration.output_path
    progress_path = output.with_suffix(".progress.json")
    completed: tuple[dict[str, object], ...] = ()
    if progress_path.is_file() and not args.restart:
        completed = load_global_local_accounting_ab_progress(
            progress_path,
            declaration=declaration,
        )
        print(
            f"resuming {len(completed)} / {declaration.expected_cells} completed cells from {progress_path}",
            flush=True,
        )

    def progress(index: int, total: int, cell_id: str) -> None:
        print(f"[global-local-ab {index:02d}/{total:02d}] starting {cell_id}", flush=True)

    def checkpoint(cells: tuple[dict[str, object], ...]) -> None:
        write_global_local_accounting_ab_progress(
            progress_path,
            declaration=declaration,
            cells=cells,
        )
        global_mean, local_mean, reduction = _campaign_means(cells[-1])
        print(
            f"[global-local-ab {len(cells):02d}/{declaration.expected_cells:02d}] "
            f"completed {cells[-1]['cell_id']}; usable global={global_mean:.6e}; "
            f"local={local_mean:.6e}; relative reduction={reduction:.3%}",
            flush=True,
        )

    result = execute_global_local_accounting_ab(
        declaration,
        project_root=root,
        completed_cells=completed,
        progress=progress,
        checkpoint=checkpoint,
    )
    written = write_global_local_accounting_ab_result(output, result)
    for cell in cast(list[dict[str, object]], result["cells"]):
        global_mean, local_mean, reduction = _campaign_means(cell)
        print(
            f"{cell['cell_id']}: usable global={global_mean:.6e}; "
            f"local={local_mean:.6e}; relative reduction={reduction:.3%}",
            flush=True,
        )
    print("training performed: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
