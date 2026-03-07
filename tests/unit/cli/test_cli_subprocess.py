"""Tests for the subprocess CLI harness."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.infra.cli_subprocess import run_cli


def test_run_cli_honors_explicit_cwd(tmp_path: Path) -> None:
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"], cwd=tmp_path)

    command = mock_run.call_args.args[0]
    kwargs = mock_run.call_args.kwargs

    assert kwargs["cwd"] == tmp_path
    assert command[:4] == ["uv", "run", "--project", str(Path(__file__).parents[3])]


def test_run_cli_defaults_cwd_to_project_root() -> None:
    completed = MagicMock(returncode=0, stdout="ok", stderr="")

    with patch("tests.infra.cli_subprocess.subprocess.run", return_value=completed) as mock_run:
        run_cli(["--help"])

    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == Path(__file__).parents[3]
