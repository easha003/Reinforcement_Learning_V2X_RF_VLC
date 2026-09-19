"""Project-specific exceptions with reproducibility context."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class HybridV2XError(Exception):
    """Base error that can identify the configuration and artifact involved.

    Parameters
    ----------
    message:
        Human-readable description of the failure.
    config_hash:
        Canonical resolved-configuration hash, when one exists at the failure
        site.
    artifact_path:
        Path of the artifact involved, when the failure concerns an artifact.
    context:
        Additional small, serializable diagnostic fields.
    """

    def __init__(
        self,
        message: str,
        *,
        config_hash: str | None = None,
        artifact_path: str | os.PathLike[str] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("error message must be a non-empty string")
        if config_hash is not None and (
            not isinstance(config_hash, str) or not config_hash.strip()
        ):
            raise ValueError("config_hash must be a non-empty string when provided")

        resolved_path: str | None = None
        if artifact_path is not None:
            resolved_path = os.fspath(artifact_path)
            if not isinstance(resolved_path, str) or not resolved_path:
                raise ValueError("artifact_path must resolve to a non-empty text path")

        self.message = message.strip()
        self.config_hash = config_hash
        self.artifact_path = resolved_path
        self.context = dict(context or {})
        super().__init__(self.message)

    def __str__(self) -> str:
        """Render the message together with available reproduction context."""

        details: list[str] = []
        if self.config_hash is not None:
            details.append(f"config_hash={self.config_hash}")
        if self.artifact_path is not None:
            details.append(f"artifact_path={self.artifact_path}")
        details.extend(f"{key}={value!r}" for key, value in sorted(self.context.items()))
        if not details:
            return self.message
        return f"{self.message} [{', '.join(details)}]"


class ConfigurationError(HybridV2XError):
    """Raised when resolved configuration violates a project contract."""


class ArtifactVersionError(HybridV2XError):
    """Raised when an artifact schema version is unsupported."""


class PhysicalInfeasibilityError(HybridV2XError):
    """Raised when a requested physical transmission cannot meet its contract."""


class ObservationLeakageError(HybridV2XError):
    """Raised when deployable observations contain hidden simulator state."""


class TraceIntegrityError(HybridV2XError):
    """Raised when a mobility or packet trace fails integrity checks."""


class CalibrationError(HybridV2XError):
    """Raised when calibration data or model support is invalid."""


class StatisticalPowerError(HybridV2XError):
    """Raised when evidence is insufficient for a requested reliability claim."""


__all__ = [
    "ArtifactVersionError",
    "CalibrationError",
    "ConfigurationError",
    "HybridV2XError",
    "ObservationLeakageError",
    "PhysicalInfeasibilityError",
    "StatisticalPowerError",
    "TraceIntegrityError",
]
