#!/usr/bin/env python3
"""Dry-run, execute, or resume the frozen deadline-edge threshold screen."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from hybrid_v2x_rl.agents.deadline_edge_threshold import (
    OpticalThresholdResult,
    execute_deadline_edge_threshold,
    load_deadline_edge_progress,
    load_deadline_edge_threshold_declaration,
    structural_deadline_edge_dry_run,
    write_deadline_edge_progress,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION = Path("configs/evaluation/deadline_edge_blocklength_threshold.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the density-20 replay; default is a structural dry run",
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
    declaration = load_deadline_edge_threshold_declaration(
        args.declaration,
        project_root=root,
        verify_evidence=True,
    )
    if not args.execute:
        report = structural_deadline_edge_dry_run(declaration, project_root=root)
        print(f"declaration SHA-256: {declaration.sha256}")
        print(f"source candidate: {report['source_candidate']}")
        print(f"density: {report['density_vehicles_per_lane_km']:g}")
        print(f"validation windows: {report['validation_windows']}")
        print(f"physical profile instances: {report['physical_profile_instances']}")
        print(
            "threshold interval: "
            f"{1000.0 * declaration.lower_airtime_s:.3f}-"
            f"{1000.0 * declaration.upper_airtime_s:.3f} ms/attempt"
        )
        print(
            "next full-slot candidate: "
            f"{1000.0 * declaration.next_full_slot_airtime_s:.1f} ms/attempt, "
            f"deadline_fit={report['next_full_slot_fits']}"
        )
        print("channel frames evaluated: 0")
        print("training performed: False")
        print("test split opened: False")
        print("structural dry run: PASS")
        return 0

    output = _resolve(root, args.out) if args.out is not None else declaration.output_path
    progress_path = output.with_suffix(".progress.json")
    completed: tuple[OpticalThresholdResult, ...] = ()
    if progress_path.is_file() and not args.restart:
        completed = load_deadline_edge_progress(
            progress_path,
            declaration=declaration,
        )
        print(
            f"resuming {len(completed)} / "
            f"{len(declaration.optical_configuration_names)} completed optical "
            f"profiles from {progress_path}",
            flush=True,
        )

    def progress(index: int, total: int, name: str) -> None:
        print(
            f"[deadline-edge {index:02d}/{total:02d}] starting {name}",
            flush=True,
        )

    def checkpoint(results: tuple[OpticalThresholdResult, ...]) -> None:
        write_deadline_edge_progress(
            progress_path,
            declaration=declaration,
            results=results,
        )
        latest = results[-1]
        threshold = latest.threshold
        print(
            f"[deadline-edge {len(results):02d}/"
            f"{len(declaration.optical_configuration_names):02d}] completed "
            f"{latest.optical_configuration_name}; upper_mean="
            f"{cast(float, threshold['upper_mean_selected_risk']):.6e}; "
            f"minimum_passing_uses={threshold['minimum_passing_channel_uses']}",
            flush=True,
        )

    result = execute_deadline_edge_threshold(
        declaration,
        project_root=root,
        completed_results=completed,
        progress=progress,
        checkpoint=checkpoint,
    )
    written = result.write_json(output)
    for optical in result.optical_results:
        threshold = optical.threshold
        minimum_airtime = threshold["minimum_passing_airtime_s"]
        formatted_airtime = (
            "none" if minimum_airtime is None else f"{1000.0 * cast(float, minimum_airtime):.6f} ms"
        )
        print(
            f"{optical.optical_configuration_name}: "
            f"control={optical.source_mean_selected_risk:.6e}, "
            f"upper={cast(float, threshold['upper_mean_selected_risk']):.6e}, "
            f"minimum={formatted_airtime}, "
            "full-slot-fit="
            f"{threshold['rounded_full_slot_candidate_fits_deadline']}",
            flush=True,
        )
    decision = result.decision()
    print(
        f"theoretical deadline boundary passes: {decision['theoretical_deadline_boundary_passes']}"
    )
    print(
        "current full-slot grid has passing candidate: "
        f"{decision['current_full_slot_grid_has_passing_candidate']}"
    )
    print(
        f"joint contention frontier authorized: {decision['joint_contention_frontier_authorized']}"
    )
    print("training performed: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
