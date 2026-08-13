"""Tests for the ``devtools test`` focused runner (devtools/run_tests.py)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from devtools import run_tests, verify
from devtools.verify_runs import CURRENT_STATISTICS_PATH, git_head, pytest_command_worker_request


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
    assert cmd[-3:] == ["-n", "4", "--dist=loadgroup"]


@pytest.mark.parametrize(
    ("selection", "expected_request"),
    [
        (["tests/unit", "-n4"], "4"),
        (["tests/unit", "-n=4"], "4"),
        (["tests/unit", "--numprocesses", "8"], "8"),
        (["tests/unit", "--numprocesses=8"], "8"),
    ],
)
def test_build_pytest_cmd_forwards_exactly_one_xdist_worker_request(
    selection: list[str], expected_request: str
) -> None:
    command = run_tests.build_pytest_cmd(selection)

    for arg in selection:
        assert arg in command
    worker_flags = [
        arg for arg in command if arg in {"-n", "--numprocesses"} or arg.startswith(("-n", "--numprocesses="))
    ]
    assert len(worker_flags) == 1
    assert pytest_command_worker_request(command) == expected_request


def test_build_pytest_cmd_honors_workers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "8")
    cmd = run_tests.build_pytest_cmd(["tests/unit"])
    assert cmd[-3:] == ["-n", "8", "--dist=loadgroup"]


def test_build_pytest_cmd_preserves_explicit_xdist_distribution() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit", "-n", "4", "--dist=worksteal"])

    assert cmd.count("--dist=worksteal") == 1
    assert "--dist=loadgroup" not in cmd


def test_build_pytest_cmd_does_not_add_distribution_for_serial_run() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit", "-n", "0"])

    assert not any(arg.startswith("--dist") for arg in cmd)


def test_subprocess_env_anchors_pytest_artifacts_to_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(run_tests.ROOT / "tests")

    env = verify._subprocess_env()

    assert env["POLYLOGUE_PYTEST_EVENTS_DIR"] == str(run_tests.ROOT / verify.PYTEST_EVENTS_DIR)
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
    monkeypatch.setattr("devtools.run_tests.append_verify_history", lambda payload: captured.update(history=payload))
    assert run_tests.main(["tests/unit/pipeline", "--json"]) == 0
    assert "--json" not in captured["cmd"]
    assert "tests/unit/pipeline" in captured["cmd"]
    assert captured["label"] == "pytest focused"
    assert captured["run"].run_id
    assert captured["run"]._payload["git_head"] == "abc123"
    assert isinstance(captured["run"]._payload["git_dirty"], bool)
    assert captured["run"]._payload["verification_scope"] == "affected"
    assert captured["run"]._payload["release_baseline_allowed"] is False
    assert captured["history"]["run_id"] == captured["run"].run_id
    assert captured["history"]["status"] == "success"


def test_main_preserves_relative_selection_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(_label: str, cmd: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cmd"] = cmd
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.chdir(run_tests.ROOT / "tests" / "unit")
    monkeypatch.setenv("POLYLOGUE_TEST_NO_LOCK", "1")
    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr("devtools.run_tests._run", _fake_run)
    monkeypatch.setattr("devtools.run_tests.append_verify_history", lambda _payload: None)

    assert run_tests.main(["core/test_identity_law.py::test_session_id_is_origin_native_id"]) == 0

    assert "tests/unit/core/test_identity_law.py::test_session_id_is_origin_native_id" in captured["cmd"]


def test_main_returns_pytest_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(label: str, cmd: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        return 5, 0.01, {"diagnosis": "pytest_failed"}

    monkeypatch.setenv("POLYLOGUE_TEST_NO_LOCK", "1")
    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr("devtools.run_tests._run", _fake_run)
    assert run_tests.main(["tests/unit/does_not_exist"]) == 5


@pytest.mark.parametrize("invocation_location", ["inside", "external"])
def test_main_anchors_and_refreshes_root_artifacts_from_any_invocation_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invocation_location: str
) -> None:
    root = tmp_path / "checkout"
    subdirectory = root / "devtools"
    external_directory = tmp_path / "unrelated"
    subdirectory.mkdir(parents=True)
    external_directory.mkdir()
    stale_report = root / verify.PYTEST_REPORT_PATH
    stale_statistics = root / CURRENT_STATISTICS_PATH
    stale_report.parent.mkdir(parents=True)
    stale_statistics.parent.mkdir(parents=True, exist_ok=True)
    stale_report.write_text('{"stale": true}')
    stale_statistics.write_text('{"stale": true}')
    captured: dict[str, object] = {}

    def fake_run(_label: str, _cmd: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cwd"] = kwargs["cwd"]
        Path(verify.PYTEST_REPORT_PATH).write_text('{"fresh": true}')
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.setattr(run_tests, "_LOCK_PATH", root / ".cache" / "test-run.lock")
    monkeypatch.setattr(
        run_tests,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=root / "polylogue", as_dict=lambda: {}),
    )
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "worktree_fingerprint", lambda _root: "fingerprint")
    monkeypatch.setattr(run_tests, "_run", fake_run)
    monkeypatch.setattr(run_tests, "append_verify_history", lambda _payload: None)
    monkeypatch.setenv("POLYLOGUE_TEST_NO_LOCK", "1")
    invocation_directory = subdirectory if invocation_location == "inside" else external_directory
    monkeypatch.chdir(invocation_directory)

    assert run_tests.main(["tests/unit/example.py"]) == 0

    assert captured["cwd"] == str(root)
    assert stale_report.read_text() == '{"fresh": true}'
    assert not stale_statistics.exists()
    assert not (invocation_directory / ".cache").exists()


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
