from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from devtools.gate import GATES_BY_NAME
from devtools.verify_webui import main


def test_webui_generate_check_script_resolves_from_ci_working_directory() -> None:
    root = Path(__file__).resolve().parents[3]
    # The script's ``uv run`` would otherwise sync the project, rebuilding the
    # editable package and writing its build metadata into the checkout. The
    # law here is only that the script resolves from CI's working directory,
    # so it runs against the already provisioned environment.
    result = subprocess.run(
        ["npm", "run", "generate:check"],
        cwd=root / "webui",
        env={**os.environ, "UV_NO_SYNC": "1"},
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "render webui-design-system: sync OK" in result.stdout


def test_webui_verification_is_catalogued() -> None:
    assert GATES_BY_NAME["webui"].args == ("devtools.verify_webui",)


def test_webui_verification_propagates_package_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Result:
        returncode = 7
        stdout = "package checks failed\n"
        stderr = ""

    monkeypatch.setattr("devtools.verify_webui.subprocess.run", lambda *args, **kwargs: Result())
    assert main([]) == 7
    assert "verify webui: red" in capsys.readouterr().out
