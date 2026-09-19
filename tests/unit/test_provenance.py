"""Work plan §11.2: an artifact must record which code produced it.

``code_version`` is the package version and moves only on a release, so it
cannot tell two engines apart that differ by a day's commits.  On 2026-08-04
the mobility engine changed twice under an unchanged ``0.1.0``, which is the
concrete failure these tests exist to prevent recurring.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hybrid_v2x_rl.core.provenance import DIRTY_SUFFIX, code_commit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def git(*args: str, cwd: Path) -> None:
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(
    code_commit(PROJECT_ROOT) is None,
    reason="the new repository has not received its initial commit yet",
)
def test_the_running_code_reports_a_commit() -> None:
    commit = code_commit(PROJECT_ROOT)

    assert commit is not None, "the project is a checkout, so a commit must be reported"
    digest = commit.removesuffix(DIRTY_SUFFIX)
    assert len(digest) == 40
    assert all(character in "0123456789abcdef" for character in digest)


def test_a_clean_tree_reports_a_bare_commit(tmp_path: Path) -> None:
    git("init", "--quiet", cwd=tmp_path)
    git("config", "user.email", "test@example.com", cwd=tmp_path)
    git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("one", encoding="utf-8")
    git("add", "file.txt", cwd=tmp_path)
    git("commit", "--quiet", "-m", "first", cwd=tmp_path)

    commit = code_commit(tmp_path)
    assert commit is not None
    assert not commit.endswith(DIRTY_SUFFIX)


def test_an_uncommitted_change_is_marked_dirty(tmp_path: Path) -> None:
    """An artifact built from modified code is not reproducible from a commit.

    Recording the clean hash would assert a reproducibility that does not hold,
    which is worse than recording nothing.
    """

    git("init", "--quiet", cwd=tmp_path)
    git("config", "user.email", "test@example.com", cwd=tmp_path)
    git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("one", encoding="utf-8")
    git("add", "file.txt", cwd=tmp_path)
    git("commit", "--quiet", "-m", "first", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("two", encoding="utf-8")

    commit = code_commit(tmp_path)
    assert commit is not None
    assert commit.endswith(DIRTY_SUFFIX)


def test_an_untracked_file_also_counts_as_dirty(tmp_path: Path) -> None:
    """A new module that is not yet committed still changes behaviour."""

    git("init", "--quiet", cwd=tmp_path)
    git("config", "user.email", "test@example.com", cwd=tmp_path)
    git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("one", encoding="utf-8")
    git("add", "file.txt", cwd=tmp_path)
    git("commit", "--quiet", "-m", "first", cwd=tmp_path)
    (tmp_path / "extra.py").write_text("x = 1\n", encoding="utf-8")

    commit = code_commit(tmp_path)
    assert commit is not None
    assert commit.endswith(DIRTY_SUFFIX)


def test_outside_a_checkout_the_commit_is_absent_rather_than_invented(
    tmp_path: Path,
) -> None:
    assert code_commit(tmp_path) is None


@pytest.mark.parametrize("field", ["code_commit"])
def test_the_manifest_model_carries_the_field(field: str) -> None:
    from hybrid_v2x_rl.artifacts.manifest import ArtifactManifest

    assert field in ArtifactManifest.model_fields
