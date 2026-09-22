#!/usr/bin/env python3
"""Run Phase 6 baselines over matched configured trace splits and random tapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.baselines import (
    BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_CONTEXTUAL,
    BASELINE_DUPLICATE_ALL,
    BASELINE_GEOMETRY_THRESHOLD,
    BASELINE_ORACLE,
    BASELINE_SUPERVISED,
    SupervisedOpticalRiskEstimator,
    baseline_policy,
)
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mean_field.matched_campaign import run_matched_policy_campaign

DEFAULT_POLICIES = (
    *BASELINE_ALWAYS_RF,
    BASELINE_ALWAYS_VLC,
    BASELINE_DUPLICATE_ALL,
    BASELINE_GEOMETRY_THRESHOLD,
    BASELINE_CONTEXTUAL,
    BASELINE_ORACLE,
)


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="canonical baseline name; repeat for multiple policies",
    )
    parser.add_argument(
        "--estimator",
        type=Path,
        help="training-fitted supervised optical-risk estimator JSON",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="per-trace diagnostic cutoff; use 0 for complete traces (default: 10)",
    )
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "artifacts/evaluations/phase6_matched_baselines.json",
    )
    return parser


def _load_estimator(path: Path | None) -> SupervisedOpticalRiskEstimator | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("estimator artifact must contain a JSON object")
    return SupervisedOpticalRiskEstimator.from_dict(payload)


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    config = load_headline_config(project_root)
    catalog = TraceCatalog.from_splits(
        config.paths.trace_root,
        config.environment.splits,
    )
    names = tuple(args.policies or DEFAULT_POLICIES)
    estimator = _load_estimator(args.estimator)
    if BASELINE_SUPERVISED in names and estimator is None:
        raise ValueError(
            "supervised-risk-allocation requires --estimator fitted on training traces"
        )
    policies = tuple(
        baseline_policy(name, supervised_estimator=estimator)
        for name in names
    )
    campaign = run_matched_policy_campaign(
        config,
        catalog,
        policies=policies,
        environment_seed=args.environment_seed,
        policy_seed=args.policy_seed,
        max_frames=None if args.frames == 0 else args.frames,
    )
    output = args.out.expanduser()
    if not output.is_absolute():
        output = project_root / output
    campaign.write_json(output)
    for comparison in campaign.comparisons:
        print(
            f"{comparison.source.trace_id}: policies={len(comparison.reports)} "
            f"frames={comparison.reports[0].frames} "
            f"tape={comparison.matched_tape_fingerprint}"
        )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
