#!/usr/bin/env python3
"""Audit global RF pooling against the declared 200 m contention domain."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.mean_field.contention_domain_audit import (
    build_contention_domain_audit,
)

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
    parser.add_argument(
        "--state-regime-audit",
        type=Path,
        default=Path("artifacts/evaluations/phase8_state_regime_audit.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/evaluations/phase8_rf_contention_domain_audit.json"),
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
    report = build_contention_domain_audit(
        config,
        state_regime_audit_path=_resolve(project_root, args.state_regime_audit),
    )
    output = report.write_json(_resolve(project_root, args.out))
    print(f"contention radius: {report.radius_m:g} m")
    for row in report.densities:
        ratio = row["global_to_local_pair_domain_ratio"]
        outside = row["pair_weighted_fraction_of_global_flows_outside_local_domain"]
        print(
            f"density={row['density_vehicles_per_lane_km']:g} "
            f"rows={row['pair_rows']} "
            f"median_global/local={ratio['p50']:.4g} "
            f"outside_fraction={outside:.4%}"
        )
    decision = report.as_dict()["decision"]
    print(
        "global pool matches local domain: "
        f"{decision['global_frame_pool_matches_declared_local_domain']}"
    )
    print("test split opened: False")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
