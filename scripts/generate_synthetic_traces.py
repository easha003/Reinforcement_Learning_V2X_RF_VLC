#!/usr/bin/env python3
"""Generate the frozen synthetic mobility traces and report Gate 1.

Thin batch driver.  All business logic lives in ``hybrid_v2x_rl.mobility.pipeline``
and is reached through the CLI, per ``CODE_IMPLEMENTATION_SPEC.md`` §23.

Usage::

    python scripts/generate_synthetic_traces.py
    python scripts/generate_synthetic_traces.py --density 40 --json

Arguments are forwarded to ``hybrid-v2x-rl mobility generate-traces``. Exits nonzero
when any target density fails a Gate-1 condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

from hybrid_v2x_rl.cli import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Forward to the CLI with project-root defaults applied."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(item.startswith("--output") for item in arguments):
        arguments += ["--output", str(PROJECT_ROOT / "artifacts")]
    if not any(item.startswith("--project-root") for item in arguments):
        arguments += ["--project-root", str(PROJECT_ROOT)]

    try:
        app(["mobility", "generate-traces", *arguments], standalone_mode=False)
    except SystemExit as exit_signal:  # typer.Exit propagates as SystemExit
        return int(exit_signal.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
