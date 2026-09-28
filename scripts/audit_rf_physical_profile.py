#!/usr/bin/env python3
"""Audit the headline RF profile before freezing a physical intervention."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_v2x_rl.channels.rf.profile_audit import (
    build_rf_physical_profile_audit,
    write_rf_physical_profile_audit,
)
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "artifacts/evaluations/phase8_rf_full_carrier_profile_audit.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output = args.out if args.out.is_absolute() else project_root / args.out
    config = load_config(
        headline_config_layers(project_root), project_root=project_root
    )
    payload = build_rf_physical_profile_audit(config)
    written = write_rf_physical_profile_audit(payload, output)
    active = payload["active_profile"]
    pool = payload["collision_pool_interpretation"]
    decision = payload["decision"]
    assert isinstance(active, dict)
    assert isinstance(pool, dict)
    assert isinstance(decision, dict)
    print(
        "active FBL: "
        f"n={active['active_finite_blocklength_channel_uses']} "
        f"k={active['information_bits']}"
    )
    print(
        "resource allocation: "
        f"{active['resource_blocks']} RB/link versus "
        f"{pool['resource_blocks_per_subchannel']} RB/pool-subchannel"
    )
    print(
        "configured block fits one pool subchannel: "
        f"{payload['checks']['configured_block_fits_one_pool_subchannel']}"
    )
    print(
        "allocation contract consistent: "
        f"{decision['allocation_contract_consistent']}"
    )
    print(
        "physical profile freeze ready: "
        f"{decision['physical_profile_freeze_ready']}"
    )
    print("training authorized: False")
    print("test split opened: False")
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
