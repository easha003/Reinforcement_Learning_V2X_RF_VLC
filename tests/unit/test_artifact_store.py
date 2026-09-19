"""Tests for atomic, immutable, manifest-verified artifacts."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from hybrid_v2x_rl.artifacts.manifest import MANIFEST_FILENAME
from hybrid_v2x_rl.artifacts.store import ArtifactStore, verify_artifact
from hybrid_v2x_rl.core.errors import TraceIntegrityError


def _write_one_file(root: Path) -> None:
    assert not (root / MANIFEST_FILENAME).exists()
    (root / "payload.txt").write_text("published data\n", encoding="utf-8")


def test_create_writes_manifest_last_and_returns_verified_reference(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = store.create(
        family="calibration",
        artifact_type="unit_test",
        artifact_id="artifact-001",
        config_hash="c" * 64,
        code_version="test-version",
        producer=_write_one_file,
        required_files=("payload.txt",),
        random_seeds={"mobility": 7},
        software_versions={"python": "3.11"},
        created_at_utc=datetime(2026, 7, 28, tzinfo=UTC),
    )

    assert artifact.path == tmp_path / "artifacts/calibration/artifact-001"
    assert artifact.manifest.files[0].path == "payload.txt"
    assert artifact.manifest.files[0].size_bytes == len(b"published data\n")
    assert artifact.reference.artifact_id == "artifact-001"
    assert artifact.reference.manifest_sha256 == artifact.manifest_sha256
    assert len(artifact.manifest_sha256) == 64
    assert not any(
        path.name.startswith(".artifact-001.tmp-") for path in artifact.path.parent.iterdir()
    )


def test_existing_artifact_is_never_overwritten(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.create(
        family="traces",
        artifact_id="trace-a",
        config_hash="hash",
        code_version="version",
        producer=_write_one_file,
    )

    called = False

    def second_producer(root: Path) -> None:
        nonlocal called
        called = True
        (root / "other.txt").write_text("other", encoding="utf-8")

    with pytest.raises(FileExistsError):
        store.create(
            family="traces",
            artifact_id="trace-a",
            config_hash="different",
            code_version="different",
            producer=second_producer,
        )
    assert not called
    assert (tmp_path / "traces/trace-a/payload.txt").read_text(encoding="utf-8") == (
        "published data\n"
    )


def test_failed_production_leaves_no_final_or_temporary_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    def partial(root: Path) -> None:
        (root / "not-the-required-file.txt").write_text("partial", encoding="utf-8")

    with pytest.raises(TraceIntegrityError, match="required"):
        store.create(
            family="traces",
            artifact_id="trace-b",
            config_hash="hash",
            code_version="version",
            producer=partial,
            required_files=("required.txt",),
        )
    family = tmp_path / "traces"
    assert list(family.iterdir()) == []


def test_open_rejects_modified_missing_and_unexpected_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    artifact = store.create(
        family="data",
        artifact_id="digest-test",
        config_hash="hash",
        code_version="version",
        producer=_write_one_file,
    )

    (artifact.path / "payload.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="size|digest"):
        store.open(family="data", artifact_id="digest-test")

    (artifact.path / "payload.txt").write_text("published data\n", encoding="utf-8")
    (artifact.path / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(TraceIntegrityError, match="inventory"):
        verify_artifact(artifact.path)

    (artifact.path / "unexpected.txt").unlink()
    (artifact.path / MANIFEST_FILENAME).unlink()
    with pytest.raises(TraceIntegrityError, match="incomplete"):
        verify_artifact(artifact.path)


@pytest.mark.parametrize(
    ("family", "artifact_id"),
    [
        ("../traces", "trace"),
        ("traces/path", "trace"),
        ("traces", "../trace"),
        ("traces", "trace/path"),
    ],
)
def test_store_rejects_path_traversal(tmp_path: Path, family: str, artifact_id: str) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="path component"):
        store.artifact_path(family, artifact_id)
