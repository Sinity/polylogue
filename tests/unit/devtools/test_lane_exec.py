"""Unit contracts for the shared direct-module lane re-exec helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from devtools import lane_exec


def _fake_polylogue_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "lane"
    (root / "polylogue" / "cli").mkdir(parents=True)
    (root / "polylogue" / "cli" / "click_app.py").write_text("", encoding="utf-8")
    (root / ".git").mkdir()
    return root


def test_reexec_into_lane_no_ops_without_an_enclosing_checkout(tmp_path: Path) -> None:
    assert lane_exec.reexec_into_lane("devtools", ["status"], cwd=tmp_path) is None


def test_reexec_into_lane_no_ops_without_a_provisioned_venv(tmp_path: Path) -> None:
    root = _fake_polylogue_checkout(tmp_path)
    assert lane_exec.reexec_into_lane("devtools", ["status"], cwd=root) is None


def test_reexec_into_lane_no_ops_when_already_the_lane_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_polylogue_checkout(tmp_path)
    lane_python = root / ".venv" / "bin" / "python"
    lane_python.parent.mkdir(parents=True)
    lane_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(lane_python))

    assert lane_exec.reexec_into_lane("devtools", ["status"], cwd=root) is None


def test_reexec_into_lane_re_execs_once_on_interpreter_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_polylogue_checkout(tmp_path)
    lane_python = root / ".venv" / "bin" / "python"
    lane_python.parent.mkdir(parents=True)
    lane_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", "/some/other/foreign/python")
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)

        class _Result:
            returncode = 7

        return _Result()

    monkeypatch.setattr(lane_exec.subprocess, "run", _fake_run)

    exit_code = lane_exec.reexec_into_lane("devtools.pre_push_gate", ["/tmp/updates"], cwd=root)

    assert exit_code == 7
    assert calls == [[str(lane_python), "-m", "devtools.pre_push_gate", "/tmp/updates"]]
