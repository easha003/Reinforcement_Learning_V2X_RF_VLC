#!/usr/bin/env python3
"""Run replay-checked Phase 5 random and scripted population rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_v2x_rl.config.loader import load_headline_config
from hybrid_v2x_rl.mean_field.deterministic_rollout import (
    POLICY_NAMES,
    canonical_policy_name,
    run_deterministic_rollout,
)
from hybrid_v2x_rl.mean_field.frames import FrameTraceSource


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=("Compose the complete Phase 5 frame path and verify deterministic replay.")
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--trace",
        help="canonical trace ID (default: first configured training trace)",
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help=(
            "policy to run; repeat for more (random, cycle, or a fixed action such "
            "as VLC/RF-2/DUP-4; default: random and cycle)"
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=600,
        help="diagnostic frame cutoff; use 0 for the full trace (default: 600)",
    )
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--policy-seed", type=int, default=7001)
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "artifacts/logs/phase5_deterministic_rollouts.json",
    )
    parser.add_argument(
        "--no-replay-check",
        action="store_true",
        help="skip the second identical run used to verify the fingerprint",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    config = load_headline_config(project_root)
    trace_id = args.trace or config.environment.splits.train[0]
    source = FrameTraceSource.discover(config.paths.trace_root / trace_id)
    policies = tuple(
        canonical_policy_name(policy) for policy in (args.policies or list(POLICY_NAMES))
    )
    max_frames = None if args.frames == 0 else args.frames

    reports: list[dict[str, object]] = []
    for policy in policies:
        report = run_deterministic_rollout(
            config,
            source,
            policy=policy,
            environment_seed=args.environment_seed,
            policy_seed=args.policy_seed,
            max_frames=max_frames,
        )
        replay_verified = False
        if not args.no_replay_check:
            repeated = run_deterministic_rollout(
                config,
                source,
                policy=policy,
                environment_seed=args.environment_seed,
                policy_seed=args.policy_seed,
                max_frames=max_frames,
            )
            if repeated != report:
                raise RuntimeError(
                    f"deterministic replay mismatch for policy {policy}: "
                    f"{report.fingerprint} != {repeated.fingerprint}"
                )
            replay_verified = True
        row = report.as_dict()
        row["replay_verified"] = replay_verified
        reports.append(row)
        print(
            f"{policy}: frames={report.frames} transitions={report.transitions} "
            f"usable={report.usable_transitions} misses={report.misses} "
            f"fingerprint={report.fingerprint} replay_verified={replay_verified}"
        )

    payload = {
        "schema": "hybrid-rf-vlc-rl.phase5-deterministic-rollouts.v1",
        "trace_id": trace_id,
        "policies": reports,
    }
    output = args.out.expanduser()
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
