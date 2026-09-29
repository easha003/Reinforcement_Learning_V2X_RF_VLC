#!/usr/bin/env python3
"""Dry-run, execute, or resume the combined receiver/block frontier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hybrid_v2x_rl.agents.combined_receiver_block_frontier import (
    CombinedProfileResult,
    execute_combined_receiver_block_frontier,
    load_combined_receiver_block_declaration,
    load_combined_receiver_block_progress,
    structural_combined_receiver_block_dry_run,
    write_combined_receiver_block_progress,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path("configs/evaluation/combined_receiver_block_frontier.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the validation screen; default is a structural dry run",
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


def _worst_mean(result: CombinedProfileResult) -> float:
    return max(
        cast(float, row["mean_optimistic_propagation_only_conditional_miss_lower_bound"])
        for row in result.rows
    )


def main() -> int:
    args = _parser().parse_args()
    if args.out is not None and not args.execute:
        raise SystemExit("--out is valid only with --execute")
    if args.restart and not args.execute:
        raise SystemExit("--restart is valid only with --execute")
    root = args.project_root.expanduser().resolve()
    declaration = load_combined_receiver_block_declaration(
        args.declaration,
        project_root=root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_combined_receiver_block_dry_run(
            declaration,
            project_root=root,
        )
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"RF candidate: {declaration.candidate.name}")
        print(f"receive profiles: {report['receive_profiles']}")
        print(f"optical configurations: {report['optical_configurations']}")
        print(f"densities: {report['densities']}")
        print(f"validation windows: {report['validation_windows']}")
        print(f"physical profile instances: {report['physical_profile_instances']}")
        print(f"evaluation rows: {report['evaluation_rows']}")
        print(f"exact budget: {declaration.miss_budget:.6e}")
        print(f"exploratory near budget: {declaration.near_budget:.6e}")
        print("channel frames evaluated: 0")
        print("training performed: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    output = _resolve(root, args.out) if args.out is not None else declaration.output_path
    progress_path = output.with_suffix(".progress.json")
    completed: tuple[CombinedProfileResult, ...] = ()
    if progress_path.is_file() and not args.restart:
        completed = load_combined_receiver_block_progress(
            progress_path,
            declaration=declaration,
        )
        print(
            f"resuming {len(completed)} / {len(declaration.receive_profiles)} "
            f"completed profiles from {progress_path}",
            flush=True,
        )

    def progress(index: int, total: int, name: str) -> None:
        print(f"[combined {index:02d}/{total:02d}] starting {name}", flush=True)

    def checkpoint(results: tuple[CombinedProfileResult, ...]) -> None:
        write_combined_receiver_block_progress(
            progress_path,
            declaration=declaration,
            results=results,
        )
        latest = results[-1]
        print(
            f"[combined {len(results):02d}/{len(declaration.receive_profiles):02d}] "
            f"completed {latest.combined_profile.profile.name}; "
            f"worst_mean={_worst_mean(latest):.6e}; "
            f"exact_optics={list(latest.exact_optical_configuration_names)}; "
            f"near_optics={list(latest.near_optical_configuration_names)}",
            flush=True,
        )

    result = execute_combined_receiver_block_frontier(
        declaration,
        project_root=root,
        completed_results=completed,
        progress=progress,
        checkpoint=checkpoint,
    )
    written = result.write_json(output)
    for profile_result in result.profile_results:
        print(f"{profile_result.combined_profile.profile.name}:", flush=True)
        for row in profile_result.rows:
            print(
                f"  {row['optical_configuration_name']} density "
                f"{row['density_vehicles_per_lane_km']:g}: "
                f"mean={row['mean_optimistic_propagation_only_conditional_miss_lower_bound']:.6e}, "
                f"exact={row['meets_exact_budget']}, "
                f"near={row['meets_exploratory_near_budget']}",
                flush=True,
            )
    decision = result.decision()
    print(f"exact propagation target met: {decision['exact_propagation_target_met']}")
    print(f"selection basis: {decision['selection_basis']}")
    print(f"selected receive profile: {decision['selected_receive_profile_name']}")
    print(f"selected optical configuration: {decision['selected_optical_configuration_name']}")
    print(f"selected worst-density mean: {decision['selected_worst_density_mean']}")
    print(
        f"joint contention frontier authorized: {decision['joint_contention_frontier_authorized']}"
    )
    print("training performed: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
