from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from hybrid_v2x_rl.cli import app
from hybrid_v2x_rl.doctor import DoctorCheck, doctor_succeeded


def _write_executable(path: Path, *, exit_code: int = 0) -> None:
    path.write_text(f"#!/bin/sh\nexit {exit_code}\n", encoding="utf-8")
    path.chmod(0o755)


def test_doctor_succeeded_requires_all_required_checks() -> None:
    passing = DoctorCheck(name="required", status="pass", detail="ok")
    warning = DoctorCheck(
        name="optional",
        status="warning",
        detail="not installed",
        required=False,
    )
    failure = DoctorCheck(name="broken", status="fail", detail="bad")

    assert doctor_succeeded((passing, warning))
    assert not doctor_succeeded((passing, failure))


def test_doctor_cli_json_is_machine_readable(monkeypatch) -> None:
    checks = (DoctorCheck(name="python", status="pass", detail="3.11"),)
    monkeypatch.setattr("hybrid_v2x_rl.cli.run_doctor", lambda **_: checks)

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {
        "checks": [
            {
                "detail": "3.11",
                "name": "python",
                "required": True,
                "status": "pass",
            }
        ],
        "ok": True,
    }


def test_doctor_cli_returns_nonzero_on_required_failure(monkeypatch) -> None:
    checks = (DoctorCheck(name="dependency:numpy", status="fail", detail="not found"),)
    monkeypatch.setattr("hybrid_v2x_rl.cli.run_doctor", lambda **_: checks)

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
