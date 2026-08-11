"""Tests for the ``devtools test`` focused runner (devtools/run_tests.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from devtools import run_tests, verify
from devtools.verify_runs import git_head


def test_build_pytest_cmd_defaults_to_single_process() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit/pipeline"])
    assert cmd[:5] == [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "devtools.pytest_progress_plugin",
    ]
    assert "tests/unit/pipeline" in cmd
    assert cmd[-2:] == ["-n", "0"]


def test_build_pytest_cmd_respects_explicit_worker_flag() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit", "-n", "4"])
    # No injected -n when the caller already chose one.
    assert cmd.count("-n") == 1
    assert cmd[-2:] == ["-n", "4"]


@pytest.mark.parametrize(
    "selection",
    [
        ["tests/unit", "-n4"],
        ["tests/unit", "-n=4"],
        ["tests/unit", "--numprocesses", "8"],
        ["tests/unit", "--numprocesses=8"],
    ],
)
def test_build_pytest_cmd_forwards_all_xdist_worker_spellings(selection: list[str]) -> None:
    command = run_tests.build_pytest_cmd(selection)

    for arg in selection:
        assert arg in command


def test_build_pytest_cmd_honors_workers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "8")
    cmd = run_tests.build_pytest_cmd(["tests/unit"])
    assert cmd[-2:] == ["-n", "8"]


def test_subprocess_env_anchors_pytest_artifacts_to_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(run_tests.ROOT / "tests")

    env = verify._subprocess_env()

    assert env["POLYLOGUE_PYTEST_EVENTS_PATH"] == str(run_tests.ROOT / verify.PYTEST_EVENTS_PATH)
    assert env["POLYLOGUE_PYTEST_SELECTION_PATH"] == str(run_tests.ROOT / verify.PYTEST_SELECTION_PATH)
    assert env["POLYLOGUE_PYTEST_SUMMARY_PATH"] == str(run_tests.ROOT / verify.PYTEST_SUMMARY_PATH)


def test_main_requires_a_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_tests.main([]) == 2
    err = capsys.readouterr().err
    assert "give a selection" in err
    assert "devtools verify" in err


def test_main_strips_dispatch_json_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(label: str, cmd: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["label"] = label
        captured["cmd"] = cmd
        captured["run"] = kwargs["run"]
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.setenv("POLYLOGUE_TEST_NO_LOCK", "1")
    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr("devtools.run_tests._run", _fake_run)
    monkeypatch.setattr("devtools.run_tests.git_head", lambda _root: "abc123")
    assert run_tests.main(["tests/unit/pipeline", "--json"]) == 0
    assert "--json" not in captured["cmd"]
    assert "tests/unit/pipeline" in captured["cmd"]
    assert captured["label"] == "pytest focused"
    assert captured["run"].run_id
    assert captured["run"]._payload["git_head"] == "abc123"
    assert isinstance(captured["run"]._payload["git_dirty"], bool)
    assert captured["run"]._payload["verification_scope"] == "affected"
    assert captured["run"]._payload["release_baseline_allowed"] is False


def test_main_returns_pytest_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(label: str, cmd: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        return 5, 0.01, {"diagnosis": "pytest_failed"}

    monkeypatch.setenv("POLYLOGUE_TEST_NO_LOCK", "1")
    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr("devtools.run_tests._run", _fake_run)
    assert run_tests.main(["tests/unit/does_not_exist"]) == 5


def test_git_head_records_checkout_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert cmd == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == tmp_path
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(cmd, 0, stdout="deadbeef\n", stderr="")

    monkeypatch.setattr("devtools.verify_runs.subprocess.run", _fake_run)
    assert git_head(tmp_path) == "deadbeef"


def test_git_head_degrades_to_none_without_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="not a git repository")

    monkeypatch.setattr("devtools.verify_runs.subprocess.run", _fake_run)
    assert git_head(tmp_path) is None


def test_git_head_degrades_to_none_when_probe_cannot_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("git missing")

    monkeypatch.setattr("devtools.verify_runs.subprocess.run", _fake_run)
    assert git_head(tmp_path) is None
