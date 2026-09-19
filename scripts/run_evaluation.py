"""Produce the paper's evaluation tables from the held-out test split.

Run as::

    python scripts/run_evaluation.py [--packets 1000000] [--split test]

**Replicates are pooled at the cluster level, not averaged.** Three test traces
per density are independent by construction -- different seeds, disjoint vehicle
populations -- so their pair episodes join one cluster set and the bootstrap
resamples across all of them. Averaging three separate interval estimates would
throw away exactly the between-trace variation the interval is supposed to
capture.

**The train split is not used here.** Every number reported so far came from
``train-000``, which is the right thing while developing the environment and the
wrong thing to publish. Selecting a profile on a trace and then reporting that
trace's numbers is how a result gets tuned into existence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_v2x_rl.channels.rf.collision import SensitivityBand
from hybrid_v2x_rl.config.hashing import scope_hash
from hybrid_v2x_rl.config.loader import headline_config_layers, load_config
from hybrid_v2x_rl.config.models import ProjectConfig
from hybrid_v2x_rl.env import campaign
from hybrid_v2x_rl.env.assembly import build_rollout
from hybrid_v2x_rl.env.episodes import TraceSource, iter_pair_instants
from hybrid_v2x_rl.env.statistics import MIN_CLUSTERS, _normal_quantile

#: The trace opens in its seeded formation -- evenly spaced, aligned, every pair
#: pointing straight down its lane -- and reads a geometric outage of 4.6% there
#: against 17.4% over its length. Nothing before this is representative.
WARMUP_S = 400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=int, default=1_000_000,
                        help="packets per density, pooled across replicates")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--densities", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--out", type=Path, default=Path("artifacts/evaluations"))
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing result file")
    parser.add_argument(
        "--band", default=None, choices=("optimistic", "nominal", "pessimistic"),
        help="collision sensitivity band. The contention model is analytical "
             "with a declared uncertainty range; this selects which end of it "
             "to evaluate.")
    parser.add_argument(
        "--fov-deg", type=float, default=None,
        help="receiver acceptance half-angle, overriding the profile. The "
             "headline is 60 deg; 30 deg is the concentrated sensitivity, "
             "which buys optical gain and costs availability.")
    parser.add_argument(
        "--ambient", default=None, choices=("clear_night", "clear_day"),
        help="ambient optical condition. Daylight raises shot noise by four "
             "orders of magnitude and is the harshest declared case.")
    parser.add_argument(
        "--pattern-variant", default=None, choices=("wide", "narrow"),
        help="override the headlamp pattern. 'wide' is the headline (the "
             "model's own falloff past the last regulated angle); 'narrow' is "
             "the sensitivity (a steeper decay there). ECE R112 specifies "
             "nothing beyond 9L/9R, so results that depend on that region are "
             "a band across the two rather than a single number.",
    )
    parser.add_argument(
        "--no-equilibrium", action="store_true",
        help="skip the mean-field allocation, which costs a second pass over "
             "every trace",
    )
    parser.add_argument(
        "--config", type=Path, nargs="+", default=None,
        help="explicit config layers, in order; defaults to the frozen headline stack. "
             "Pass configs/service/ev2x_300B_3ms_1e-5.yaml in place of the "
             "1e-4 layer to run the eV2X row.",
    )
    return parser.parse_args()


def report_to_dict(
    report: campaign.DensityReport,
    budget: float,
    replicates: int,
    config: ProjectConfig,
) -> dict:
    """Everything a table or a figure needs, with the raw counts kept.

    ``report_raw_counts`` is set in the evaluation profile for a reason: "36
    joint failures in 500,075 packets" is checkable in a way that "99.9925%" is
    not.
    """

    policies = {}
    for name, stats in report.policies.items():
        policies[name] = {
            "packets": stats.packets,
            "misses": stats.misses,
            "clusters": stats.rate.cluster_count,
            "mean_cost": stats.mean_cost,
            "expected_miss_rate": stats.expected_miss_rate,
            "realized_miss_rate": stats.realized_miss_rate,
            "expected_bootstrap_upper": stats.upper_bound(replicates=replicates),
            "realized_bootstrap_upper": stats.realized_upper(replicates=replicates),
            "realized_wilson_upper": stats.wilson_upper(),
            # Keyed off the bound, not the point estimate: that is the
            # difference between "we measured 7.5e-5" and "we showed 1e-4".
            "meets_budget": stats.upper_bound(replicates=replicates) <= budget,
            "failure_causes": dict(stats.causes),
            "action_choices": dict(stats.choices),
        }
    return {
        "density_veh_per_lane_km": report.density,
        "packets": report.packets,
        "clusters": report.cluster_count,
        "sufficient_clusters": report.cluster_count >= MIN_CLUSTERS,
        # Recorded beside the miss rates because it can invalidate all of them.
        # Above one the profile committed more airtime than the pool supplies,
        # and every reliability number here was computed by a collision model
        # that does not check whether the selections it prices can all be
        # honoured.
        "pool": {
            "mean_contenders": report.mean_contenders,
            "resource_demand": report.resource_demand,
            "deliverable": report.deliverable,
            "attempts_per_packet": config.service.rf_attempts_per_packet,
        },
        "optical": {
            "geometric_outage": report.optical_outage,
            "total_failure_rate": report.optical_failure_rate,
            "geometric_failures": report.optical_geometric_failures,
            "failures": report.optical_failures,
        },
        "complementarity": {
            "rate": report.complementarity,
            "rf_lost_vlc_saved": report.rf_lost_vlc_saved,
            "vlc_lost_rf_saved": report.vlc_lost_rf_saved,
        },
        "dependence": {
            "ratio": report.dependence_ratio,
            "realized_joint_failures": report.joint_failures,
            "predicted_joint_failures": report.predicted_joint,
        },
        "policies": policies,
        "targeting": {
            "risk_concentration_top_1pct": report.targeting.risk_concentration(0.01),
            "risk_concentration_top_10pct": report.targeting.risk_concentration(0.10),
            "frontier": [
                {"budget": b, "mean_cost": c, "achieved": a, "feasible": f}
                for b, c, a, f in report.targeting.frontier(
                    campaign.DEFAULT_FRONTIER_BUDGETS
                )
            ],
        },
    }


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    config = load_config(
        tuple(args.config) if args.config else headline_config_layers(root),
        project_root=root,
    )
    if args.pattern_variant:
        artifact = (root / "artifacts" / "calibration" / "vlc"
                    / f"headlamp-r112-compliant-{args.pattern_variant}-v2")
        if not artifact.exists():
            print(f"missing pattern artifact {artifact}\n"
                  f"  build it with scripts/build_headlamp_pattern.py", file=sys.stderr)
            return 1
        config = config.model_copy(
            update={"vlc": config.vlc.model_copy(update={"pattern_artifact": artifact})},
            deep=True,
        )
        print(f"headlamp pattern: {artifact.name}")
    vlc_updates: dict = {}
    if args.fov_deg is not None:
        vlc_updates["receiver_fov_deg"] = args.fov_deg
    if args.ambient is not None:
        vlc_updates["ambient_condition"] = args.ambient
    if vlc_updates:
        config = config.model_copy(
            update={"vlc": config.vlc.model_copy(update=vlc_updates)}, deep=True)
        print(f"vlc overrides: {vlc_updates}")

    band = SensitivityBand(args.band) if args.band else None
    if band:
        print(f"collision band: {band.value}")

    budget = config.service.miss_budget
    per_replicate = max(1, args.packets // args.replicates)

    # Named and guarded before anything runs, because the artifact is now
    # written after every density rather than once at the end. A nine-hour run
    # lost two finished densities when the third crashed mid-pass: completed
    # work should survive whatever comes after it. The packet count stays in the
    # name because a smoke run and a publication run must not share a path --
    # one did, and a 6,000 packet check overwrote a 3,000,000 packet result.
    parts = []
    # The service profile decides the answer and did not appear in the name, so
    # a 1e-5 run and a 1e-4 run at the same packet count and pattern collided on
    # one path. Omitted for the headline so existing artifacts keep their names.
    if config.project.experiment != "headline":
        parts.append(config.project.experiment.replace("_", "-"))
    parts.append(args.pattern_variant or "config")
    if args.band:
        parts.append(args.band)
    if args.fov_deg is not None:
        parts.append(f"fov{args.fov_deg:g}")
    if args.ambient:
        parts.append(args.ambient)
    destination = (args.out
                   / f"campaign-{args.split}-{'-'.join(parts)}-{args.packets}.json")

    # What has to match for two runs to belong in one artifact. The filename
    # already separates packet count, pattern, band, FOV and ambient; this
    # catches the rest, so densities measured on different seeds or a different
    # config cannot be merged into something that reads as one campaign.
    provenance = {
        "budget": budget,
        "split": args.split,
        "root_seed": args.seed,
        "packets_requested_per_density": args.packets,
        "bootstrap_replicates": args.bootstrap,
        "collision_band": args.band or "nominal",
        "config_hash": scope_hash(config, "mobility"),
    }
    prior: list = []
    if destination.exists():
        held = json.loads(destination.read_text())
        clash = {k: (held.get(k), v) for k, v in provenance.items()
                 if held.get(k) != v}
        if clash and not args.force:
            print(
                f"refusing to merge into {destination}\n"
                f"  it holds a run this one does not match: {clash}\n"
                f"  pass --force if that is what you want",
                file=sys.stderr,
            )
            return 1
        prior = [] if clash else held.get("densities", [])
        if prior:
            print(f"merging into {destination}, which already holds densities "
                  f"{[e['density_veh_per_lane_km'] for e in prior]}")

    def write(entries: list) -> Path:
        """Persist every finished density, merged with any already on disk.

        Densities are measured one at a time and a run can cover a subset, so a
        later run has to compose with an earlier one rather than replace it --
        otherwise finishing the campaign in two sittings means keeping the
        answer in two files. Entries from this run win over stored ones for the
        same density, which is what a deliberate re-measurement should do.
        """

        fresh = {e["density_veh_per_lane_km"] for e in entries}
        merged = sorted(
            [e for e in prior if e["density_veh_per_lane_km"] not in fresh] + entries,
            key=lambda e: e["density_veh_per_lane_km"],
        )
        args.out.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            **provenance,
            "warmup_s": WARMUP_S,
            "confidence_level": config.evaluation.confidence_level,
            "one_sided_upper_bound": True,
            "normal_quantile_z": _normal_quantile(config.evaluation.confidence_level),
            "interval_method": "trajectory_block_bootstrap",
            "headlamp_pattern_variant": args.pattern_variant or "as configured",
            "receiver_fov_deg": config.vlc.receiver_fov_deg,
            "ambient_condition": config.vlc.ambient_condition,
            # Which densities this file holds against which were asked for, so a
            # partial artifact cannot be read as a complete one.
            "densities_requested": list(args.densities),
            "densities_complete": [e["density_veh_per_lane_km"] for e in merged],
            "densities": merged,
        }, indent=2))
        return destination

    results = []
    for density in args.densities:
        report = campaign.DensityReport(density=float(density))
        used = []
        sources = []
        started = time.time()
        for replicate in range(args.replicates):
            trace_id = f"synthetic-d{density}-{args.split}-{replicate:03d}"
            path = root / "artifacts" / "traces" / trace_id
            if not path.exists():
                print(f"  ! missing {trace_id}, skipping", flush=True)
                continue
            source = TraceSource.discover(path)
            sources.append(source)
            # A fresh rollout per trace: shadowing and fading state must not
            # carry across traces, and the seed is tied to the trace so a rerun
            # of one replicate reproduces exactly.
            rollout = build_rollout(
                config, buildings=(), root_seed=args.seed + density * 10 + replicate,
                band=band,
            )
            campaign.run(
                rollout,
                iter_pair_instants(
                    source,
                    generation_period_s=config.service.generation_period_s,
                    max_packets=per_replicate,
                    warmup_s=WARMUP_S,
                ),
                density=float(density),
                into=report,
            )
            used.append(trace_id)
            print(f"  {trace_id}: {report.packets:,} packets pooled "
                  f"({time.time() - started:.0f}s)", flush=True)

        if not report.packets:
            print(f"density {density}: no traces found for split {args.split!r}")
            continue

        # The allocation, scored on the channel it produces, into its own
        # report: run_equilibrium counts its own packets and contenders, and
        # pooling those into the baseline report would double every denominator
        # on it. Only the policy's statistics cross over, so format_report
        # prints it beside the baselines it has to be judged against.
        equilibrium = None
        if sources and not args.no_equilibrium:
            started_eq = time.time()
            equilibrium = campaign.run_equilibrium(
                config, sources=sources, density=float(density), budget=budget,
                root_seed=args.seed + density * 10, band=band,
                max_packets=per_replicate, warmup_s=WARMUP_S,
                generation_period_s=config.service.generation_period_s,
            )
            report.policies[campaign.EQUILIBRIUM] = equilibrium.policies[
                campaign.EQUILIBRIUM
            ]
            print(f"  equilibrium: {equilibrium.packets:,} packets, pool claim "
                  f"{equilibrium.resource_demand:.2f} "
                  f"({time.time() - started_eq:.0f}s)", flush=True)

        print()
        print(campaign.format_report(report, budget=budget,
                                     bootstrap_replicates=args.bootstrap))
        print(f"  traces: {', '.join(used)}")
        print(f"  [{time.time() - started:.0f}s]\n")

        entry = report_to_dict(report, budget, args.bootstrap, config)
        if equilibrium is not None:
            # Its own pool claim, which is the whole point: the baseline block
            # above reports what the profile asks at full radio use, and this
            # reports what is left once the optical leg has taken its share.
            entry["equilibrium"] = {
                "packets": equilibrium.packets,
                "mean_contenders": equilibrium.mean_contenders,
                "resource_demand": equilibrium.resource_demand,
                "deliverable": equilibrium.deliverable,
            }
        entry["traces"] = used
        entry["split"] = args.split
        results.append(entry)
        written = write(results)
        held = json.loads(written.read_text())["densities_complete"]
        print(f"  -> wrote {written}  "
              f"({len(results)} of {len(args.densities)} this run; "
              f"file now holds {held})\n", flush=True)

    if not results:
        print("nothing evaluated", file=sys.stderr)
        return 1

    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
