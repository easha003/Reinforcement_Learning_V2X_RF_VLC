"""Versioned provenance manifests for immutable research artifacts."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Annotated

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "1.0.0"
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]
Seed = Annotated[int, Field(strict=True, ge=0, lt=2**64)]


class ManifestModel(BaseModel):
    """Strict base model for data persisted in a manifest."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class FileDigest(ManifestModel):
    """Digest and size of one regular file relative to an artifact root."""

    path: NonEmptyString
    sha256: Sha256
    size_bytes: Annotated[int, Field(ge=0)]

    @field_validator("path")
    @classmethod
    def path_is_safe_and_canonical(cls, value: str) -> str:
        """Reject absolute, platform-specific, or escaping paths."""

        if value == "." or "\\" in value:
            raise ValueError("manifest file paths must use POSIX separators")
        path = PurePosixPath(value)
        if path.is_absolute() or value != path.as_posix():
            raise ValueError("manifest file path must be canonical and relative")
        if any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("manifest file path cannot contain empty, '.' or '..' parts")
        if path.as_posix() == MANIFEST_FILENAME:
            raise ValueError("manifest cannot contain a digest of itself")
        return value


class ArtifactReference(ManifestModel):
    """Stable pointer to a consumed artifact and its exact manifest."""

    artifact_type: NonEmptyString
    artifact_id: NonEmptyString
    manifest_sha256: Sha256


class ArtifactManifest(ManifestModel):
    """Complete, versioned provenance record stored beside artifact files."""

    schema_version: NonEmptyString = MANIFEST_SCHEMA_VERSION
    artifact_type: NonEmptyString
    artifact_id: NonEmptyString
    created_at_utc: AwareDatetime
    config_hash: NonEmptyString
    code_version: NonEmptyString
    #: Git commit the producing code came from, ``-dirty`` when the tree was
    #: modified.  ``None`` only when the artifact was built outside a checkout.
    #: ``code_version`` alone is insufficient: it is the package version and
    #: moves on a release, so it cannot distinguish two engines that differ by
    #: a day's commits.  Work plan §11.2.
    code_commit: str | None = None
    #: Digests of the configuration subsets this artifact's producer can read,
    #: keyed by scope name (``hybrid_v2x_rl.config.hashing.HASH_SCOPES``).  The
    #: question "may this artifact be reused?" is answered here; ``config_hash``
    #: answers the broader "was this the same run?" and moves whenever any
    #: physics-bearing field changes, including ones the producer never reads.
    config_scope_hashes: dict[str, str] = Field(default_factory=dict)
    input_artifacts: list[ArtifactReference] = Field(default_factory=list)
    random_seeds: dict[str, Seed] = Field(default_factory=dict)
    software_versions: dict[str, str] = Field(default_factory=dict)
    files: list[FileDigest] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @field_validator("created_at_utc")
    @classmethod
    def timestamp_is_utc(cls, value: datetime) -> datetime:
        if value.utcoffset() != timedelta(0):
            raise ValueError("created_at_utc must use UTC")
        return value

    @field_validator("random_seeds")
    @classmethod
    def seeds_are_unsigned_64_bit(cls, value: dict[str, int]) -> dict[str, int]:
        for name in value:
            if not name.strip():
                raise ValueError("random seed names must be non-empty")
        return value

    @field_validator("software_versions")
    @classmethod
    def software_version_entries_are_nonempty(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not name.strip() or not version.strip() for name, version in value.items()):
            raise ValueError("software version names and values must be non-empty")
        return value

    @model_validator(mode="after")
    def entries_are_unique_and_canonical(self) -> ArtifactManifest:
        file_paths = [file.path for file in self.files]
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("manifest file paths must be unique")
        if file_paths != sorted(file_paths):
            raise ValueError("manifest file entries must be sorted by path")

        references = [
            (reference.artifact_type, reference.artifact_id) for reference in self.input_artifacts
        ]
        if len(references) != len(set(references)):
            raise ValueError("input artifact references must be unique")
        return self


__all__ = [
    "ArtifactManifest",
    "ArtifactReference",
    "FileDigest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
]
