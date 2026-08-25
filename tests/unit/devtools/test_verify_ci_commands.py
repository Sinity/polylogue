from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_ci_commands


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, str]:
    monkeypatch.setattr(verify_ci_commands, "repo_root", lambda: root)
    return verify_ci_commands.main(list(argv)), capsys.readouterr().out


def test_current_ci_devtools_commands_match_the_catalog(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from devtools import repo_root

    rc, _out = _run(repo_root(), monkeypatch, capsys)
    assert rc == 0


def test_unknown_github_run_command_fails_without_scanning_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tmp_path,
        ".github/workflows/ci.yml",
        """
name: devtools imaginary prose is not executable
jobs:
  test:
    steps:
      - name: devtools also-imaginary
        run: uv run devtools verify definitely-unknown
""",
    )

    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert "unknown devtools command 'verify definitely-unknown'" in out


def test_circle_command_mapping_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        tmp_path,
        ".circleci/config.yml",
        """
version: 2.1
jobs:
  gate:
    steps:
      - run:
          name: broken
          command: uv run devtools nonexistent-command
""",
    )

    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert "unknown devtools command 'nonexistent-command'" in out


def test_invalid_workflow_yaml_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, ".github/workflows/bad.yml", ": invalid: [")

    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert ".github/workflows/bad.yml: invalid YAML:" in out


def test_empty_workflow_population_fails_closed_without_a_success_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out = _run(tmp_path, monkeypatch, capsys, "--json")
    payload = json.loads(out)
    assert rc == 1
    assert payload["required_gate"]["diagnosis"] == "gate_empty_required_population"
    assert "CI devtools commands match" not in out


def test_unreadable_workflow_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, ".github/workflows/bad.yml", "name: valid\njobs: {}\n")
    workflow = tmp_path / ".github" / "workflows" / "bad.yml"
    workflow.write_bytes(b"\xff\xfe")

    rc, out = _run(tmp_path, monkeypatch, capsys)
    assert rc == 1
    assert "invalid YAML" in out
