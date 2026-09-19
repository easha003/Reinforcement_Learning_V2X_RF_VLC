"""Read-only diagnostics for the local Hybrid RF/VLC RL development environment."""

from __future__ import annotations

import importlib
import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from hybrid_v2x_rl.config import config_hash, load_config, load_headline_config

CheckStatus = Literal["pass", "fail", "warning"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One machine-readable environment check."""

    name: str
    status: CheckStatus
    detail: str
    required: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _python_check() -> DoctorCheck:
    supported = sys.version_info[:2] in {(3, 11), (3, 12)}
    return DoctorCheck(
        name="python",
        status="pass" if supported else "fail",
        detail=f"{platform.python_version()} at {sys.executable}",
    )


def _dependency_check(module_name: str) -> DoctorCheck:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return DoctorCheck(
            name=f"dependency:{module_name}",
            status="fail",
            detail=str(exc),
        )
    version = getattr(module, "__version__", "installed")
    return DoctorCheck(
        name=f"dependency:{module_name}",
        status="pass",
        detail=str(version),
    )


def _config_check(config_paths: tuple[Path, ...], project_root: Path) -> DoctorCheck:
    try:
        if config_paths:
            config = load_config(config_paths, project_root=project_root)
        else:
            config = load_headline_config(project_root=project_root)
    except Exception as exc:
        return DoctorCheck(
            name="headline_config",
            status="fail",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return DoctorCheck(
        name="headline_config",
        status="pass",
        detail=f"schema={config.schema_version}; sha256={config_hash(config)}",
    )


def run_doctor(
    *,
    project_root: Path,
    config_paths: tuple[Path, ...] = (),
) -> tuple[DoctorCheck, ...]:
    """Run all required M0 diagnostics."""

    checks = [
        _python_check(),
        *(_dependency_check(name) for name in ("numpy", "pydantic", "yaml", "typer")),
        _config_check(config_paths, project_root.resolve()),
    ]
    return tuple(checks)


def doctor_succeeded(checks: Iterable[DoctorCheck]) -> bool:
    """Return whether every required check passed."""

    return all(not check.required or check.status == "pass" for check in checks)
