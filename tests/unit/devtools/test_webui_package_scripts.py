from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools.gate import GATES_BY_NAME
from devtools.verify_webui import main


def test_webui_generate_check_script_resolves_from_ci_working_directory() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["npm", "run", "generate:check"],
        cwd=root / "webui",
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
