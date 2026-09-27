#!/usr/bin/env python3
"""Apply the predeclared gate to four bounded PPO regime evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.recovery_analysis import (
    RECOVERY_ARMS,
    build_constraint_recovery_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    for arm in RECOVERY_ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluations/phase8_constraint_recovery_analysis.json"),
    )
    return parser


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def main() -> int:
    args = _parser().parse_args()
    root = args.project_root.expanduser().resolve()
    paths = {arm: _resolve(root, getattr(args, arm)) for arm in RECOVERY_ARMS}
    report = build_constraint_recovery_analysis(paths)
    output = report.write_json(_resolve(root, args.out))
    for arm, row in report.comparisons.items():
        print(
            f"{arm}: feasible_gain={row['weighted_feasible_mass_gain']:.6f} "
            f"risk_ratio={row['weighted_expected_risk_ratio']} "
            f"pass={row['passes_primary_gate']}"
        )
    print(f"selected arm: {report.selected_arm}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
