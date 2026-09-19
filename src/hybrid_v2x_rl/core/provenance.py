"""Which code produced an artifact.

Work plan §11.2 requires every run to be keyed by configuration hash, mobility
trace hash, channel calibration version, **code commit**, policy seed and
evaluation scenario seed.  The commit was the missing one: manifests recorded
``code_version``, which is the package version string and moves only on a
release.  Two traces can therefore share a ``config_hash`` and a
``code_version`` and still come from materially different simulators — which is
exactly what happened on 2026-08-04, when the mobility engine changed twice
under an unchanged ``0.1.0``.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

#: Marks a commit whose working tree carried uncommitted changes.
DIRTY_SUFFIX = "-dirty"

_GIT_TIMEOUT_S = 5.0


def _git(*args: str, cwd: Path | None = None) -> str | None:
    """Run one git command, or return ``None`` if git cannot answer."""

    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


@lru_cache(maxsize=8)
def code_commit(project_root: Path | None = None) -> str | None:
    """The commit the running code came from, or ``None`` outside a checkout.

    A ``-dirty`` suffix is appended when the working tree has uncommitted
    changes.  This matters more than it looks: an artifact built from modified
    code cannot be reproduced from the commit alone, and recording the clean
    hash would assert a reproducibility that does not hold.  Recording nothing
    would be worse still, because the artifact would look like it came from
    whatever was committed at the time.

    Cached, since it shells out and the answer cannot change within a run.
    """

    root = project_root or Path(__file__).resolve().parents[3]
    commit = _git("rev-parse", "HEAD", cwd=root)
    if commit is None:
        return None

    status = _git("status", "--porcelain", cwd=root)
    if status:
        return f"{commit}{DIRTY_SUFFIX}"
    return commit


__all__ = [
    "DIRTY_SUFFIX",
    "code_commit",
]
