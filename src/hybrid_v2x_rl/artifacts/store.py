"""Atomic, immutable, manifest-verified artifact storage."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from hybrid_v2x_rl.artifacts.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactReference,
    FileDigest,
)
from hybrid_v2x_rl.core.errors import ArtifactVersionError, TraceIntegrityError
from hybrid_v2x_rl.core.provenance import code_commit

_HASH_CHUNK_BYTES = 1024 * 1024
ArtifactProducer = Callable[[Path], None]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_json_bytes(manifest: ArtifactManifest) -> bytes:
    """Serialize a manifest deterministically for stable references."""

    payload: dict[str, Any] = manifest.model_dump(mode="json")
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    """A successfully verified immutable artifact."""

    path: Path
    manifest: ArtifactManifest
    manifest_sha256: str

    @property
    def reference(self) -> ArtifactReference:
        """Return the compact provenance pointer consumed by child artifacts."""

        return ArtifactReference(
            artifact_type=self.manifest.artifact_type,
            artifact_id=self.manifest.artifact_id,
            manifest_sha256=self.manifest_sha256,
        )


class ArtifactStore:
    """Create and open immutable artifacts below one root directory.

    A ``family`` is the on-disk collection name (for example ``"traces"``).
    It is intentionally distinct from the semantic ``artifact_type`` stored in
    a manifest (for example ``"mobility_trace"``).
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def artifact_path(self, family: str, artifact_id: str) -> Path:
        """Return the final path after validating both path components."""

        return (
            self.root
            / _safe_component(family, field="family")
            / _safe_component(artifact_id, field="artifact_id")
        )

    def create(
        self,
        *,
        family: str,
        artifact_id: str,
        config_hash: str,
        code_version: str,
        producer: ArtifactProducer,
        artifact_type: str | None = None,
        required_files: Sequence[str] = (),
        input_artifacts: Iterable[ArtifactReference] = (),
        random_seeds: dict[str, int] | None = None,
        software_versions: dict[str, str] | None = None,
        notes: Iterable[str] = (),
        created_at_utc: datetime | None = None,
        schema_version: str = MANIFEST_SCHEMA_VERSION,
        code_commit_override: str | None = None,
        config_scope_hashes: dict[str, str] | None = None,
    ) -> StoredArtifact:
        """Produce, validate, and atomically publish one artifact.

        ``producer`` receives an otherwise empty temporary directory.  It must
        return normally before any content is hashed.  The store writes
        ``manifest.json`` last and promotes the complete directory in one
        rename.  An existing final directory is never replaced.
        """

        family_name = _safe_component(family, field="family")
        artifact_name = _safe_component(artifact_id, field="artifact_id")
        semantic_type = artifact_type if artifact_type is not None else family_name
        if not semantic_type.strip():
            raise ValueError("artifact_type must be non-empty")

        family_path = self.root / family_name
        family_path.mkdir(parents=True, exist_ok=True)
        final_path = family_path / artifact_name
        if final_path.exists() or final_path.is_symlink():
            raise FileExistsError(f"immutable artifact already exists: {final_path}")

        reservation_path = family_path / f".{artifact_name}.publish-lock"
        try:
            reservation_descriptor = os.open(
                reservation_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as error:
            raise FileExistsError(f"artifact is already being published: {final_path}") from error
        temporary_path: Path | None = None
        try:
            try:
                os.write(reservation_descriptor, f"pid={os.getpid()}\n".encode())
            finally:
                os.close(reservation_descriptor)
            temporary_path = Path(
                tempfile.mkdtemp(prefix=f".{artifact_name}.tmp-", dir=family_path)
            )
            producer(temporary_path)
            manifest_path = temporary_path / MANIFEST_FILENAME
            if manifest_path.exists() or manifest_path.is_symlink():
                raise TraceIntegrityError(
                    "artifact producer must not write manifest.json",
                    artifact_path=temporary_path,
                )

            normalized_required = tuple(_safe_relative_path(item) for item in required_files)
            _check_required_files(temporary_path, normalized_required)
            file_digests = _collect_file_digests(temporary_path)
            manifest = ArtifactManifest(
                schema_version=schema_version,
                artifact_type=semantic_type,
                artifact_id=artifact_name,
                created_at_utc=created_at_utc or datetime.now(UTC),
                config_hash=config_hash,
                code_version=code_version,
                code_commit=code_commit_override or code_commit(),
                config_scope_hashes=dict(config_scope_hashes or {}),
                input_artifacts=list(input_artifacts),
                random_seeds=dict(random_seeds or {}),
                software_versions=dict(software_versions or {}),
                files=file_digests,
                notes=list(notes),
            )
            _write_manifest_last(manifest_path, manifest)

            verified = verify_artifact(
                temporary_path,
                expected_artifact_type=semantic_type,
                expected_artifact_id=artifact_name,
            )
            if final_path.exists() or final_path.is_symlink():
                raise FileExistsError(f"immutable artifact already exists: {final_path}")
            temporary_path.rename(final_path)
            _fsync_directory(family_path)
            return StoredArtifact(
                path=final_path,
                manifest=verified.manifest,
                manifest_sha256=verified.manifest_sha256,
            )
        except BaseException:
            if temporary_path is not None and temporary_path.exists():
                shutil.rmtree(temporary_path)
            raise
        finally:
            reservation_path.unlink(missing_ok=True)

    def open(
        self,
        *,
        family: str,
        artifact_id: str,
        expected_artifact_type: str | None = None,
    ) -> StoredArtifact:
        """Open an artifact only after verifying its manifest and every file."""

        path = self.artifact_path(family, artifact_id)
        return verify_artifact(
            path,
            expected_artifact_type=expected_artifact_type,
            expected_artifact_id=artifact_id,
        )


def verify_artifact(
    path: str | os.PathLike[str],
    *,
    expected_artifact_type: str | None = None,
    expected_artifact_id: str | None = None,
) -> StoredArtifact:
    """Verify manifest schema, inventory, sizes, and SHA-256 file digests."""

    artifact_path = Path(path)
    if not artifact_path.is_dir() or artifact_path.is_symlink():
        raise TraceIntegrityError(
            "artifact path is not a regular directory",
            artifact_path=artifact_path,
        )

    manifest_path = artifact_path / MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise TraceIntegrityError(
            "artifact is incomplete: manifest.json is missing",
            artifact_path=artifact_path,
        )

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = ArtifactManifest.model_validate_json(manifest_bytes)
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as error:
        raise TraceIntegrityError(
            "artifact manifest is invalid",
            artifact_path=artifact_path,
            context={"reason": str(error)},
        ) from error

    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ArtifactVersionError(
            "unsupported artifact manifest schema",
            artifact_path=artifact_path,
            context={
                "actual": manifest.schema_version,
                "supported": MANIFEST_SCHEMA_VERSION,
            },
        )
    if expected_artifact_type is not None and (manifest.artifact_type != expected_artifact_type):
        raise TraceIntegrityError(
            "artifact type does not match expectation",
            artifact_path=artifact_path,
            context={
                "actual": manifest.artifact_type,
                "expected": expected_artifact_type,
            },
        )
    if expected_artifact_id is not None and manifest.artifact_id != expected_artifact_id:
        raise TraceIntegrityError(
            "artifact ID does not match expectation",
            artifact_path=artifact_path,
            context={"actual": manifest.artifact_id, "expected": expected_artifact_id},
        )

    expected_files = {digest.path: digest for digest in manifest.files}
    actual_files = _inventory_relative_files(artifact_path)
    expected_inventory = set(expected_files) | {MANIFEST_FILENAME}
    if actual_files != expected_inventory:
        raise TraceIntegrityError(
            "artifact file inventory does not match manifest",
            artifact_path=artifact_path,
            context={
                "missing": sorted(expected_inventory - actual_files),
                "unexpected": sorted(actual_files - expected_inventory),
            },
        )

    for relative, digest in expected_files.items():
        file_path = artifact_path / relative
        size_bytes = file_path.stat().st_size
        if size_bytes != digest.size_bytes:
            raise TraceIntegrityError(
                "artifact file size does not match manifest",
                artifact_path=artifact_path,
                context={
                    "file": relative,
                    "actual": size_bytes,
                    "expected": digest.size_bytes,
                },
            )
        actual_hash = sha256_file(file_path)
        if actual_hash != digest.sha256:
            raise TraceIntegrityError(
                "artifact file digest does not match manifest",
                artifact_path=artifact_path,
                context={
                    "file": relative,
                    "actual": actual_hash,
                    "expected": digest.sha256,
                },
            )

    return StoredArtifact(
        path=artifact_path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def _safe_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty, trimmed string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{field} must be one safe path component")
    return value


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("required file path must be a non-empty string")
    if value == "." or "\\" in value:
        raise ValueError("required file paths must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("required file path must be canonical and relative")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("required file path cannot contain '.', '..', or empty parts")
    if value == MANIFEST_FILENAME:
        raise ValueError("manifest.json is managed by ArtifactStore")
    return value


def _check_required_files(root: Path, required_files: Sequence[str]) -> None:
    missing = [
        relative
        for relative in required_files
        if not (root / relative).is_file() or (root / relative).is_symlink()
    ]
    if missing:
        raise TraceIntegrityError(
            "artifact producer did not create all required files",
            artifact_path=root,
            context={"missing": missing},
        )


def _inventory_relative_files(root: Path) -> set[str]:
    inventory: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise TraceIntegrityError(
                "artifact cannot contain symbolic links",
                artifact_path=root,
                context={"file": relative},
            )
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise TraceIntegrityError(
                "artifact contains a non-regular filesystem entry",
                artifact_path=root,
                context={"file": relative},
            )
        inventory.add(relative)
    return inventory


def _collect_file_digests(root: Path) -> list[FileDigest]:
    paths = sorted(_inventory_relative_files(root))
    return [
        FileDigest(
            path=relative,
            sha256=sha256_file(root / relative),
            size_bytes=(root / relative).stat().st_size,
        )
        for relative in paths
    ]


def _write_manifest_last(path: Path, manifest: ArtifactManifest) -> None:
    payload = manifest_json_bytes(manifest)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier; some filesystems reject directory fsync."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "ArtifactProducer",
    "ArtifactStore",
    "StoredArtifact",
    "manifest_json_bytes",
    "sha256_file",
    "verify_artifact",
]
