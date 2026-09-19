"""Command-line entry point for Hybrid RF/VLC RL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from hybrid_v2x_rl import __version__
from hybrid_v2x_rl.config import config_hash as hash_config
from hybrid_v2x_rl.config import headline_config_layers, load_config
from hybrid_v2x_rl.core.errors import HybridV2XError
from hybrid_v2x_rl.doctor import doctor_succeeded, run_doctor
from hybrid_v2x_rl.mean_field.frame_campaign import validate_frame_campaign
from hybrid_v2x_rl.mean_field.frames import TraceCatalog
from hybrid_v2x_rl.mobility.pipeline import (
    GridTracePipeline,
    SplitCounts,
    TraceCampaignPlan,
    TraceCampaignResult,
)

app = typer.Typer(
    name="hybrid-v2x-rl",
    help="Hybrid RF/VLC RL research implementation commands.",
    no_args_is_help=True,
)
mobility_app = typer.Typer(
    name="mobility",
    help="Analytic Manhattan-grid mobility commands.",
    no_args_is_help=True,
)
frames_app = typer.Typer(
    name="frames",
    help="Population-frame replay and validation commands.",
    no_args_is_help=True,
)
app.add_typer(mobility_app)
app.add_typer(frames_app)
console = Console()


@app.callback()
def main() -> None:
    """Run a Hybrid RF/VLC RL command."""


@app.command()
def doctor(
    config: Annotated[
        list[Path] | None,
        typer.Option(
            "--config",
            help="Layered YAML configuration path; repeat in merge order.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option(
            "--project-root",
            help="Project root containing configs/.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON."),
    ] = False,
) -> None:
    """Check Python, dependencies, and the resolved headline config."""

    resolved_project_root = project_root if project_root is not None else Path.cwd()
    checks = run_doctor(
        project_root=resolved_project_root,
        config_paths=tuple(config or ()),
    )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "ok": doctor_succeeded(checks),
                    "checks": [check.to_dict() for check in checks],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        table = Table(title="Hybrid RF/VLC RL environment")
        table.add_column("Check")
        table.add_column("Status")
        table.add_column("Detail")
        for check in checks:
            style = {"pass": "green", "warning": "yellow", "fail": "red"}[check.status]
            table.add_row(check.name, f"[{style}]{check.status}[/{style}]", check.detail)
        console.print(table)

    if not doctor_succeeded(checks):
        raise typer.Exit(code=1)


def _render_campaign(result: TraceCampaignResult) -> None:
    table = Table(title="Gate 1 — mobility credibility")
    table.add_column("Target ρ", justify="right")
    table.add_column("Realized", justify="right")
    table.add_column("Trace")
    table.add_column("Split")
    table.add_column("Pairs", justify="right")
    table.add_column("Gate 1")
    for density in result.densities:
        style = "green" if density.passed else "red"
        table.add_row(
            f"{density.target_density_veh_per_lane_km:g}",
            f"{density.realized_density_veh_per_lane_km:.2f}",
            density.trace_id,
            density.split,
            str(len(density.pair_segments)),
            f"[{style}]{'pass' if density.passed else 'FAIL'}[/{style}]",
        )
    console.print(table)
    for density in result.densities:
        failed = [check for check in density.validation.gate1_checks if not check.passed]
        if failed:
            console.print(f"[red]{density.trace_id} failed Gate-1 checks:[/red]")
            for check in failed:
                console.print(f"  - {check.name}: {check.detail}")


@mobility_app.command("generate-traces")
def mobility_generate_traces(
    output: Annotated[
        Path,
        typer.Option("--output", help="Artifact root for generated traces.", file_okay=False),
    ],
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", help="Layered YAML path; repeat in merge order.", exists=True),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option("--project-root", help="Project root containing configs/.", exists=True),
    ] = None,
    density: Annotated[
        list[float] | None,
        typer.Option("--density", help="Override target densities in veh/lane-km."),
    ] = None,
    train: Annotated[
        int, typer.Option("--train", help="Independent realizations in the train split.")
    ] = 3,
    validation: Annotated[
        int, typer.Option("--validation", help="Independent realizations in validation.")
    ] = 1,
    test: Annotated[
        int, typer.Option("--test", help="Independent realizations in the test split.")
    ] = 3,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable JSON.")
    ] = False,
) -> None:
    """Generate traces, extract tagged pairs, and report Gate 1.

    Returns nonzero when any target density fails a Gate-1 condition.
    """

    root = project_root if project_root is not None else Path.cwd()
    layers = tuple(config) if config else headline_config_layers(root)
    try:
        resolved = load_config(layers, project_root=root)
        pipeline = GridTracePipeline(
            resolved,
            TraceCampaignPlan(output_root=output, code_version=__version__),
        )
        console.print(f"resolved configuration hash: [bold]{pipeline.config_hash}[/bold]")
        result = pipeline.run(
            target_densities=density or None,
            splits=SplitCounts(train=train, validation=validation, test=test),
        )
    except HybridV2XError as error:
        console.print(f"[red]{type(error).__name__}: {error}[/red]")
        raise typer.Exit(code=1) from error

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "gate1_passed": result.gate1_passed,
                    "config_hash": result.config_hash,
                    "manifest_path": str(result.manifest_path),
                    "densities": [item.to_record() for item in result.densities],
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
    else:
        _render_campaign(result)
        console.print(f"campaign manifest: {result.manifest_path}")

    if not result.gate1_passed:
        raise typer.Exit(code=1)


@frames_app.command("validate-campaign")
def frames_validate_campaign(
    config: Annotated[
        list[Path] | None,
        typer.Option("--config", help="Layered YAML path; repeat in merge order.", exists=True),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option("--project-root", help="Project root containing configs/.", exists=True),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help="Artifact root for frame caches; defaults to configured paths.artifact_root.",
            file_okay=False,
        ),
    ] = None,
    report_path: Annotated[
        Path | None,
        typer.Option(
            "--report",
            help="Machine-readable campaign report path.",
            dir_okay=False,
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the complete machine-readable report."),
    ] = False,
) -> None:
    """Validate all configured traces and publish compact immutable frame caches."""

    root = project_root if project_root is not None else Path.cwd()
    layers = tuple(config) if config else headline_config_layers(root)
    try:
        resolved = load_config(layers, project_root=root)
        digest = hash_config(resolved)
        output_root = artifact_root or resolved.paths.artifact_root
        report = validate_frame_campaign(
            TraceCatalog.from_splits(resolved.paths.trace_root, resolved.environment.splits),
            generation_period_s=resolved.service.generation_period_s,
            expected_config_hash=digest,
            artifact_root=output_root,
            code_version=__version__,
        )
        destination = report.write_json(
            report_path or output_root / "frame_campaign_validation.json"
        )
    except HybridV2XError as error:
        console.print(f"[red]{type(error).__name__}: {error}[/red]")
        raise typer.Exit(code=1) from error

    payload = report.as_dict()
    if json_output:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        totals = cast(dict[str, object], payload["totals"])
        console.print(
            "[green]Phase 2 frame validation passed[/green]: "
            f"{payload['trace_count']} traces, {totals['frames']} frames, "
            f"{totals['decision_pair_episodes']} pair episodes"
        )
        console.print(f"campaign report: {destination}")


if __name__ == "__main__":
    app()
