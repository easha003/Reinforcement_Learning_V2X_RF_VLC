#!/usr/bin/env python3
"""Diagnose residual PPO reliability floor and action-selection regret."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.residual_feasibility import (
    build_residual_feasibility_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path(
            "artifacts/evaluations/"
            "phase8_seed1001_dual_init10_full_ppo_regime_evaluation_v2.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/evaluations/phase8_seed1001_residual_feasibility.json"
        ),
    )
    return parser


def _resolve(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def main() -> int:
    args = _parser().parse_args()
    report = build_residual_feasibility_report(_resolve(args.evaluation))
    output = report.write_json(_resolve(args.out))
    print(f"campaign diagnosis: {report.campaign['diagnosis']}")
    for row in report.densities:
        induced = row["counterfactuals"]["policy_induced_load"]
        offload = row["counterfactuals"]["vlc_offload"]
        print(
            f"density={row['density_vehicles_per_lane_km']:g} "
            f"policy_floor={induced['mean_minimum_action_conditional_miss_risk']:.8g} "
            f"offload_floor={offload['mean_minimum_action_conditional_miss_risk']:.8g} "
            f"diagnosis={row['diagnosis']}"
        )
    print(
        "standard PPO recovery arm authorized: "
        f"{report.standard_ppo_recovery_arm_authorized}"
    )
    print(f"next action: {report.next_action}")
    print("test split opened: False")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
