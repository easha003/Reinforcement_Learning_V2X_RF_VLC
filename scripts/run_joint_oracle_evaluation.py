#!/usr/bin/env python3
"""Evaluate the exact population-joint reliability floor on validation windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.joint_oracle_evaluation import (
    build_joint_oracle_evaluation,
)
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root containing configuration and trace artifacts",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="layered YAML path; repeat in merge order; defaults to headline layers",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--state-regime-audit",
        type=Path,
        default=Path("artifacts/evaluations/phase8_state_regime_audit.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/evaluations/phase8_seed1001_population_joint_oracle.json"
        ),
    )
    return parser


def _resolve(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    layers = (
        tuple(_resolve(project_root, path) for path in args.config)
        if args.config
        else headline_config_layers(project_root)
    )
    config = load_config(layers, project_root=project_root)
    report = build_joint_oracle_evaluation(
        config,
        checkpoint_path=_resolve(project_root, args.checkpoint),
        state_regime_audit_path=_resolve(project_root, args.state_regime_audit),
    )
    output = report.write_json(_resolve(project_root, args.out))
    print(f"joint oracle windows: {len(report.windows)}")
    for row in report.densities:
        print(
            f"density={row['density_vehicles_per_lane_km']:g} "
            f"risk={row['mean_joint_oracle_conditional_miss_risk']:.8g} "
            f"budget_multiple={row['joint_oracle_risk_budget_multiple']:.4g} "
            f"meets_budget={row['joint_oracle_mean_meets_budget']}"
        )
    decision = report.as_dict()["decision"]
    print(
        "all densities meet joint floor: "
        f"{decision['all_densities_meet_joint_oracle_floor']}"
    )
    print("test split opened: False")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
