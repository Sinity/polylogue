"""An absent virtualenv is reported as such, not as a missing tool.

A ``.claude/worktrees/<agent>`` checkout that has never entered the devshell has
no ``.venv`` at all. Reporting the first gate's ``ruff`` as unavailable names
neither the cause nor the remedy, and ruff is in fact installed.

Anti-vacuity: delete the ``unprovisioned_environment`` branch in
``executable_gate_result`` and ``test_an_absent_venv_is_named_with_its_remedy``
goes red -- the diagnosis falls back to ``gate_missing_executable``.
"""

from __future__ import annotations

from pathlib import Path

from devtools.required_gate import executable_gate_result, unprovisioned_environment


def test_an_absent_venv_is_named_with_its_remedy(tmp_path: Path) -> None:
    result = executable_gate_result([str(tmp_path / ".venv" / "bin" / "ruff"), "format"], gate="gate format")
    assert result.ok is False
    assert result.diagnosis == "gate_unprovisioned_environment"
    assert str(tmp_path / ".venv") in result.details[0]
    assert "nix develop" in result.details[0]
    # The remedy must not suggest borrowing another checkout's environment.
    assert "Never share or symlink" in result.details[0]


def test_a_provisioned_venv_missing_one_tool_is_still_a_missing_executable(tmp_path: Path) -> None:
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    result = executable_gate_result([str(tmp_path / ".venv" / "bin" / "ruff"), "format"], gate="gate format")
    assert result.diagnosis == "gate_missing_executable"


def test_an_installed_tool_passes(tmp_path: Path) -> None:
    binary = tmp_path / ".venv" / "bin" / "ruff"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    result = executable_gate_result([str(binary), "format"], gate="gate format")
    assert result.ok is True
    assert result.diagnosis == "gate_passed"


def test_a_path_resolved_tool_is_not_classified_as_a_venv(tmp_path: Path) -> None:
    del tmp_path
    assert unprovisioned_environment("some-tool-not-on-path") is None
    assert unprovisioned_environment(None) is None
    assert unprovisioned_environment("/usr/bin/ruff") is None
