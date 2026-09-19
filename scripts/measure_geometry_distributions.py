"""Measure the pair-geometry distributions the coupling figures rest on.

Run as::

    python scripts/measure_geometry_distributions.py [--stride 8]

Emits ``artifacts/evaluations/geometry-distributions.json``: per density, the
distribution of optical path length and of contender count over the same
tagged-pair instants the campaign evaluates.

These two quantities are the paper's central claim in numerical form. The
optical path is the car-following gap, so it *shortens* as traffic densifies;
the contender count is what the radio must share a resource pool with, so it
*rises*. Measuring them from one trace is what makes their opposition a
property of the traffic rather than of two independently chosen distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_v2x_rl.config import headline_config_layers, load_config
from hybrid_v2x_rl.core.geometry import Segment
from hybrid_v2x_rl.core.link_endpoints import optical_link_path
from hybrid_v2x_rl.core.pair_geometry import pair_geometry
from hybrid_v2x_rl.env.rollout import CONTENTION_RADIUS_M
from hybrid_v2x_rl.geometry.spatial_index import SpatialIndex

COLUMNS = ["time_s", "vehicle_id", "x_m", "y_m", "heading_rad", "speed_mps",
           "length_m", "width_m", "height_m"]
WARMUP_S = 400.0
QUANTILES = [5, 10, 25, 50, 75, 90, 95]


class Pose:
    __slots__ = tuple(c for c in COLUMNS if c != "time_s")

    def __init__(self, row):
        for name in Pose.__slots__:
            setattr(self, name, row[name])


def measure(trace: Path, fov_rad: float, stride: int):
    pairs = pq.read_table(
        trace / "pairs.parquet", columns=["tx_id", "rx_id", "start_s", "end_s"]
    ).to_pylist()
    pairs.sort(key=lambda p: p["start_s"])
    starts = [p["start_s"] for p in pairs]

    optical: list[float] = []
    separation: list[float] = []
    contenders: list[int] = []
    frame: dict[str, Pose] = {}
    current = None
    emitted = 0

    def flush(time_s):
        nonlocal emitted
        if current is None or time_s < WARMUP_S:
            return
        emitted += 1
        if emitted % stride:
            return
        index = SpatialIndex.build(tuple(frame.values()))
        for p in pairs[: bisect_right(starts, time_s)]:
            if p["end_s"] < time_s:
                continue
            tx, rx = frame.get(p["tx_id"]), frame.get(p["rx_id"])
            if tx is None or rx is None:
                continue
            geometry = pair_geometry(tx, rx, fov_half_angle_rad=fov_rad)
            optical.append(geometry.optical_path_length_m)
            separation.append(geometry.separation_m)
            origin = optical_link_path(tx, rx).segment.start
            near = index.candidates(Segment(origin, origin), margin_m=CONTENTION_RADIUS_M)
            contenders.append(sum(
                1 for other in near
                if other.vehicle_id != tx.vehicle_id
                and math.hypot(other.x_m - tx.x_m, other.y_m - tx.y_m) <= CONTENTION_RADIUS_M
            ))

    for part in sorted((trace / "vehicles").glob("part-*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=65_536, columns=COLUMNS):
            for row in batch.to_pylist():
                if row["time_s"] != current:
                    flush(current)
                    frame, current = {}, row["time_s"]
                frame[row["vehicle_id"]] = Pose(row)
    flush(current)
    return optical, separation, contenders


def summarise(values, *, cdf_points: int = 200) -> dict:
    a = np.sort(np.asarray(values, dtype=float))
    grid = np.linspace(0, 100, cdf_points)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "quantiles": {str(q): float(np.percentile(a, q)) for q in QUANTILES},
        # A coarse CDF rather than the raw sample: enough to draw the curve,
        # small enough that the artifact stays reviewable.
        "cdf_x": [float(x) for x in np.percentile(a, grid)],
        "cdf_p": [float(p) / 100.0 for p in grid],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stride", type=int, default=8,
                    help="evaluate one frame in STRIDE; the distributions are "
                         "smooth, so subsampling costs resolution and not shape")
    ap.add_argument("--split", default="test")
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--densities", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--out", type=Path,
                    default=Path("artifacts/evaluations/geometry-distributions.json"))
    ap.add_argument(
        "--config", type=Path, nargs="+", default=None,
        help="explicit config layers, in order; defaults to the frozen headline stack. "
             "Pass configs/service/ev2x_300B_3ms_1e-5.yaml in place of the "
             "1e-4 layer to run the eV2X row.",
    )
    args = ap.parse_args()

    root = Path.cwd()
    _root = root
    config = load_config(
        tuple(args.config) if args.config else headline_config_layers(_root),
        project_root=_root,
    )
    fov = math.radians(config.vlc.receiver_fov_deg)

    payload = {"split": args.split, "replicate": args.replicate,
               "warmup_s": WARMUP_S, "stride": args.stride,
               "contention_radius_m": CONTENTION_RADIUS_M, "densities": {}}
    for density in args.densities:
        trace = root / "artifacts" / "traces" / \
            f"synthetic-d{density}-{args.split}-{args.replicate:03d}"
        if not trace.exists():
            print(f"  ! missing {trace.name}", file=sys.stderr)
            continue
        optical, separation, contenders = measure(trace, fov, args.stride)
        payload["densities"][str(density)] = {
            "optical_path_m": summarise(optical),
            "separation_m": summarise(separation),
            "contenders": summarise(contenders),
        }
        print(f"  rho={density}: {len(optical):,} instants, "
              f"optical path median {np.median(optical):.1f} m, "
              f"contenders median {np.median(contenders):.0f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
