"""Measure the policy-independent half of every transition, once.

Run as::

    python scripts/build_training_cache.py --split train --densities 10 20 30

Writes ``artifacts/caches/<split>-d<density>-<replicate>/`` for each density.
See :mod:`hybrid_v2x_rl.env.cache` for why this is sound: every draw a packet
consumes is seeded from the packet's identity, so the risks, the outcomes and
the reported qualities are functions of the trace and not of the policy. Only
the link history depends on what was chosen, and the replay rebuilds that.

Rows are written grouped by episode and ordered within it. The stream arrives
interleaved -- many pairs are live at once and it is ordered by time -- but GAE
runs along a trajectory, so a replay wants each episode contiguous. Sorting
once here costs a few seconds; discovering episode boundaries on every epoch of
every seed would not.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_v2x_rl.config import config_hash, headline_config_layers, load_config
from hybrid_v2x_rl.core.enums import Link
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.env.cache import ColumnPlan, TransitionCache
from hybrid_v2x_rl.env.episodes import TraceSource, iter_pair_instants
from hybrid_v2x_rl.env.feedback import measurements
from hybrid_v2x_rl.env.packet import ACTIONS, DUP
from hybrid_v2x_rl.env.perception import build_perception
from hybrid_v2x_rl.env.rollout import always
from hybrid_v2x_rl.observation.builder import ObservationSchema

WARMUP_S = 400.0


def build(config, *, density: int, split: str, replicate: int, packets: int,
          seed: int, out: Path) -> dict:
    schema = ObservationSchema(
        features=tuple(config.observation.features),
        history_packets=int(config.observation.history_packets),
    )
    plan = ColumnPlan.from_schema(schema)

    root = Path.cwd()
    trace_dir = root / "artifacts" / "traces" / f"synthetic-d{density}-{split}-{replicate:03d}"
    source = TraceSource.discover(trace_dir)
    rollout = build_rollout(config, buildings=(), root_seed=seed)
    perception = build_perception(config, root_seed=seed)

    # Preallocated rather than appended. The stream overshoots ``packets``
    # slightly because it truncates on a frame boundary, so the buffers carry
    # headroom and grow only if that is not enough; a list of a million
    # one-row arrays costs more in object overhead than the data itself, and
    # three densities building at once would not fit.
    capacity = int(packets * 1.1) + 10_000
    width = len(plan.trace_columns)
    rows = np.zeros((capacity, width), dtype=np.float32)
    risk = np.zeros((capacity, 2), dtype=np.float32)
    delivered = np.zeros((capacity, 2), dtype=np.uint8)
    quality = np.zeros((capacity, 2), dtype=np.float32)
    stamps = np.zeros(capacity, dtype=np.float64)
    episode = np.zeros(capacity, dtype=np.int64)
    order = np.zeros(capacity, dtype=np.int64)
    episode_of: dict[str, int] = {}
    count = 0
    started = time.time()

    for instant in iter_pair_instants(
        source, generation_period_s=config.service.generation_period_s,
        max_packets=packets, warmup_s=WARMUP_S,
    ):
        observation = perception.observe(instant)
        if observation is None:
            continue
        _, _, alternatives = rollout.evaluate_instant(
            trace_id=instant.trace_id, pair_id=instant.pair_id, index=instant.index,
            density=float(density), time_s=instant.time_s,
            transmitter=instant.transmitter, receiver=instant.receiver,
            neighbours=instant.neighbours, index_of_frame=instant.index_of_frame,
            choose=always(ACTIONS[0]), counterfactual=True,
        )
        rf, vlc, dup = alternatives["RF"], alternatives["VLC"], alternatives["DUP"]

        # DUP spends both legs, so it is the one outcome carrying both
        # qualities. The matched-tape rule makes them equal to what RF-only and
        # VLC-only would have reported, which is asserted in the cache tests.
        reported = measurements(
            dup, root_seed=seed, trace_id=instant.trace_id,
            pair_id=instant.pair_id, packet_index=instant.index,
        )

        if count == len(stamps):
            grown = int(len(stamps) * 1.5)
            rows = np.resize(rows, (grown, width))
            risk, delivered = np.resize(risk, (grown, 2)), np.resize(delivered, (grown, 2))
            quality = np.resize(quality, (grown, 2))
            stamps, episode = np.resize(stamps, grown), np.resize(episode, grown)
            order = np.resize(order, grown)

        vector = np.asarray(observation, dtype=np.float32)
        rows[count] = vector[plan.trace_columns]
        risk[count] = (rf.rf_failure_probability, vlc.vlc_failure_probability)
        delivered[count] = (int(rf.rf_delivered), int(vlc.vlc_delivered))
        quality[count] = (reported[Link.RF], reported[Link.VLC])
        stamps[count] = instant.time_s
        episode[count] = episode_of.setdefault(instant.pair_id, len(episode_of))
        order[count] = instant.index
        count += 1

        if instant.final:
            rollout.release(instant.pair_id)
            perception.release(instant.pair_id)

    if not count:
        raise SystemExit(f"no usable packets in {trace_dir.name}")

    rows, risk, delivered = rows[:count], risk[:count], delivered[:count]
    quality, stamps = quality[:count], stamps[:count]

    # Group each episode's rows together and order them, so a replay can take a
    # trajectory as a contiguous slice.
    shuffle = np.lexsort((order[:count], episode[:count]))
    episodes = episode[:count][shuffle]
    final = np.zeros(len(episodes), dtype=np.uint8)
    final[-1] = 1
    final[:-1] = (episodes[1:] != episodes[:-1]).astype(np.uint8)

    manifest = {
        "density": density, "split": split, "replicate": replicate, "seed": seed,
        "trace": trace_dir.name, "warmup_s": WARMUP_S,
        "config_hash": config_hash(config),
        "rf_attempts_per_packet": int(config.service.rf_attempts_per_packet),
        "vlc_attempts_per_packet": int(config.service.vlc_attempts_per_packet),
        "trace_features": list(plan.trace_features),
        "history_packets": int(config.observation.history_packets),
        "episodes": int(len(episode_of)),
        "activation_costs": {action.name: action.activation_cost for action in ACTIONS},
        "duplication_cost": DUP.activation_cost,
    }
    TransitionCache.write(
        out, trace=rows[shuffle], risk=risk[shuffle], delivered=delivered[shuffle],
        quality=quality[shuffle], time_s=stamps[shuffle],
        episode=episodes.astype(np.int32), final=final, manifest=manifest,
    )
    elapsed = time.time() - started
    print(f"  rho={density}: {count:,} packets in {len(episode_of):,} episodes, "
          f"{elapsed:.0f}s ({count / elapsed:.0f}/s) -> {out}", flush=True)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--densities", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--replicate", type=int, default=0)
    ap.add_argument("--packets", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--out", type=Path, default=Path("artifacts/caches"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing cache instead of skipping it")
    ap.add_argument(
        "--config", type=Path, nargs="+", default=None,
        help="explicit config layers, in order; defaults to the frozen headline stack. "
             "Pass configs/service/ev2x_300B_3ms_1e-5.yaml in place of the "
             "1e-4 layer to run the eV2X row.",
    )
    args = ap.parse_args()

    _root = Path.cwd()
    config = load_config(
        tuple(args.config) if args.config else headline_config_layers(_root),
        project_root=_root,
    )
    for density in args.densities:
        out = args.out / f"{args.split}-d{density}-{args.replicate:03d}"
        if (out / "manifest.json").exists() and not args.force:
            print(f"  rho={density}: {out} exists, skipping (--force to rebuild)")
            continue
        build(config, density=density, split=args.split, replicate=args.replicate,
              packets=args.packets, seed=args.seed, out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
