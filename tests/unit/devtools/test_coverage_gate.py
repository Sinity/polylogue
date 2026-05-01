from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools import coverage_gate


def write_pyproject(path: Path, threshold: int | float) -> Path:
    pyproject = path / "pyproject.toml"
    pyproject.write_text(
        f"[tool.coverage.report]\nfail_under = {threshold}\n",
        encoding="utf-8",
    )
    return pyproject


def test_read_coverage_threshold_uses_pyproject_report_floor(tmp_path: Path) -> None:
    pyproject = write_pyproject(tmp_path, 84)

    assert coverage_gate.read_coverage_threshold(pyproject) == 84


def test_read_coverage_threshold_rejects_bool(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[tool.coverage.report]\nfail_under = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fail_under"):
        coverage_gate.read_coverage_threshold(pyproject)


def test_build_coverage_command_uses_threshold_and_local_options(tmp_path: Path) -> None:
    pyproject = write_pyproject(tmp_path, 84)

    command = coverage_gate.build_coverage_command(
        pyproject_path=pyproject,
        ignore_integration=True,
        term_missing=True,
        extra_args=("--maxfail=1",),
    )

    assert command == [
        "pytest",
        "--cov=polylogue",
        "--cov-report=xml",
        "--cov-report=term-missing:skip-covered",
        "--cov-fail-under",
        "84",
        "-q",
        "--ignore=tests/integration",
        "--maxfail=1",
    ]


def test_main_strips_separator_and_runs_coverage_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject = write_pyproject(tmp_path, 84)
    captured: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        captured.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("devtools.coverage_gate.subprocess.run", fake_run)

    assert coverage_gate.main(["--pyproject", str(pyproject), "--ignore-integration", "--", "--maxfail=1"]) == 0
    assert captured == [
        [
            "pytest",
            "--cov=polylogue",
            "--cov-report=xml",
            "--cov-fail-under",
            "84",
            "-q",
            "--ignore=tests/integration",
            "--maxfail=1",
        ]
    ]
