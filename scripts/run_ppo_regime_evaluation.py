#!/usr/bin/env python3
"""Evaluate a frozen PPO checkpoint by causal state regime on validation traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.agents.regime_evaluation import build_ppo_regime_evaluation
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
        help=(
            "Layered YAML configuration path; repeat in merge order. "
            "Defaults to the headline layers."
        ),
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
        default=Path("artifacts/evaluations/phase8_seed1001_ppo_regime_evaluation.json"),
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
    report = build_ppo_regime_evaluation(
        config,
        checkpoint_path=_resolve(project_root, args.checkpoint),
        state_regime_audit_path=_resolve(project_root, args.state_regime_audit),
    )
    output = _resolve(project_root, args.out)
    report.write_json(output)
    print(
        "PPO regime evaluation: "
        f"windows={len(report.windows)} "
        f"checkpoint_iteration={report.checkpoint_completed_iterations}"
    )
    for row in report.campaign_rows:
        actual = row["actual_policy_load"]
        assert isinstance(actual, dict)
        print(
            f"{row['regime']}: rows={row['rows']} "
            f"risk={actual['mean_selected_conditional_miss_risk']} "
            f"feasible={actual['selected_feasible_fraction']}"
        )
    print("test split opened: False")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
