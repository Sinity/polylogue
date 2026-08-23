"""Tests for the ``devtools test`` focused runner (devtools/run_tests.py)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from devtools import run_tests
from devtools.verify_runs import (
    CURRENT_RUN_PATH,
    CURRENT_STATISTICS_PATH,
    git_head,
    pytest_command_worker_request,
)


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


def test_main_requires_a_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_tests.main([]) == 2
    err = capsys.readouterr().err
    assert "give a selection" in err
    assert "devtools verify" in err


def test_main_strips_dispatch_json_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def direct_subprocess(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr("devtools.run_tests.subprocess.run", direct_subprocess)
    monkeypatch.setattr("devtools.run_tests.git_head", lambda _root: "abc123")
    monkeypatch.setattr("devtools.run_tests.append_verify_history", lambda payload: captured.update(history=payload))
    assert run_tests.main(["tests/unit/pipeline", "--json"]) == 0
    assert "--json" not in captured["cmd"]
    assert "tests/unit/pipeline" in captured["cmd"]
    assert captured["env"]["POLYLOGUE_PYTEST_EVENTS_DIR"].endswith("/events")
    assert captured["history"]["git_head"] == "abc123"
    assert isinstance(captured["history"]["git_dirty"], bool)
    assert captured["history"]["verification_scope"] == "affected"
    assert captured["history"]["status"] == "success"


def test_main_preserves_relative_selection_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(_label: str, cmd: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cmd"] = cmd
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.chdir(run_tests.ROOT / "tests" / "unit")
    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr("devtools.run_tests._run", _fake_run)
    monkeypatch.setattr("devtools.run_tests.append_verify_history", lambda _payload: None)

    assert run_tests.main(["core/test_identity_law.py::test_session_id_is_origin_native_id"]) == 0

    assert "tests/unit/core/test_identity_law.py::test_session_id_is_origin_native_id" in captured["cmd"]


def test_main_preserves_path_valued_options_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(_label: str, cmd: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cmd"] = cmd
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    invocation = tmp_path / "nested"
    invocation.mkdir()
    (invocation / "fixtures").mkdir()
    monkeypatch.chdir(invocation)
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr(run_tests, "_run", _fake_run)
    monkeypatch.setattr(run_tests, "append_verify_history", lambda _payload: None)

    assert (
        run_tests.main(
            [
                "-k",
                "proof",
                "--basetemp=diagnostic",
                "--rootdir",
                ".",
                "--ignore",
                "fixtures",
                "--ignore-glob=fixtures/*.json",
                "--junit-xml",
                "reports/results.xml",
            ]
        )
        == 0
    )

    command = cast(list[str], captured["cmd"])
    assert f"--basetemp={invocation / 'diagnostic'}" in command
    assert command[command.index("--rootdir") + 1] == str(invocation)
    assert command[command.index("--ignore") + 1] == str(invocation / "fixtures")
    assert f"--ignore-glob={invocation / 'fixtures' / '*.json'}" in command
    assert command[command.index("--junit-xml") + 1] == str(invocation / "reports" / "results.xml")


def test_main_persists_interrupted_direct_cli_result_to_local_run_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history: dict[str, Any] = {}

    def interrupt(*_args: Any, **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_tests,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=tmp_path / "polylogue", as_dict=lambda: {}),
    )
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr(run_tests, "_run", interrupt)
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "append_verify_history", lambda payload: history.update(payload))

    assert run_tests.main(["tests/unit/example.py"]) == 130

    run_payload = json.loads((tmp_path / history["artifact_dir"] / "run.json").read_text())
    current_payload = json.loads((tmp_path / CURRENT_RUN_PATH).read_text())
    for payload in (history, run_payload, current_payload):
        assert payload["diagnosis"] == "pytest_interrupted"
        assert payload["pytest_aggregate"]["selection_mode"] == "focused"
        assert payload["git_head"] == "head"
        assert payload["final_git_head"] == "head"


def test_normalize_selection_paths_preserves_pytest_path_option_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invocation = tmp_path / "invocation"
    expanded_root = tmp_path / "expanded-root"
    invocation.mkdir()
    expanded_root.mkdir()
    absolute_config = tmp_path / "absolute.ini"
    monkeypatch.setenv("PYTEST_ROOT", str(expanded_root))

    normalized = run_tests._normalize_selection_paths(
        [
            "-cconfig/pytest.ini",
            "-c",
            "separate/pytest.ini",
            "--config-file=other/pytest.ini",
            "--config-file",
            "separate-config/pytest.ini",
            "--log-file",
            "logs/test.log",
            "--log-file=logs/joined.log",
            "--debug",
            "logs/debug-separated.log",
            "--debug=logs/debug.log",
            "--rootdir",
            "$PYTEST_ROOT/relative",
            "--rootdir=$PYTEST_ROOT/joined",
            "--junitxml=reports/junit.xml",
            "--junit-xml",
            "reports/junit-alias.xml",
            "--ignore-glob=fixtures/*.json",
            "--basetemp",
            str(absolute_config),
            "--config-file",
            "$PYTEST_ROOT/literal.ini",
        ],
        invocation_directory=invocation,
    )

    assert f"-c{invocation / 'config' / 'pytest.ini'}" in normalized
    assert normalized[normalized.index("-c") + 1] == str(invocation / "separate" / "pytest.ini")
    assert f"--config-file={invocation / 'other' / 'pytest.ini'}" in normalized
    assert normalized[normalized.index("--config-file") + 1] == str(invocation / "separate-config" / "pytest.ini")
    assert normalized[normalized.index("--log-file") + 1] == str(invocation / "logs" / "test.log")
    assert f"--log-file={invocation / 'logs' / 'joined.log'}" in normalized
    assert normalized[normalized.index("--debug") + 1] == str(invocation / "logs" / "debug-separated.log")
    assert f"--debug={invocation / 'logs' / 'debug.log'}" in normalized
    assert normalized[normalized.index("--rootdir") + 1] == str(expanded_root / "relative")
    assert f"--rootdir={expanded_root / 'joined'}" in normalized
    assert f"--junitxml={invocation / 'reports' / 'junit.xml'}" in normalized
    assert normalized[normalized.index("--junit-xml") + 1] == str(invocation / "reports" / "junit-alias.xml")
    assert f"--ignore-glob={invocation / 'fixtures' / '*.json'}" in normalized
    assert normalized[normalized.index("--basetemp") + 1] == str(absolute_config)
    assert normalized[-1] == str(invocation / "$PYTEST_ROOT" / "literal.ini")


def test_normalize_selection_paths_preserves_pytest_symlinks_and_optional_debug(
    tmp_path: Path,
) -> None:
    invocation = tmp_path / "invocation"
    invocation.mkdir()
    target = invocation / "target.ini"
    target.write_text("[pytest]\n", encoding="utf-8")
    config_link = invocation / "config-link.ini"
    config_link.symlink_to(target.name)

    normalized = run_tests._normalize_selection_paths(
        ["-c", "config-link.ini", "-c=config-link.ini", "--debug", "-k", "focused"],
        invocation_directory=invocation,
    )

    lexical_link = str(invocation / "config-link.ini")
    assert normalized[:2] == ["-c", lexical_link]
    assert normalized[2] == f"-c{lexical_link}"
    assert normalized[3:] == ["--debug", "-k", "focused"]
    assert str(target) not in normalized


def test_main_preserves_keyword_and_marker_values_from_tests_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    monkeypatch.chdir(run_tests.ROOT / "tests")
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)

    def capture(_label: str, command: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured.extend(command)
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.setattr(
        run_tests,
        "_run",
        capture,
    )
    monkeypatch.setattr(run_tests, "append_verify_history", lambda _payload: None)

    assert run_tests.main(["-k", "unit", "-m", "unit"]) == 0

    keyword_index = captured.index("-k")
    marker_index = next(index for index in range(keyword_index + 1, len(captured)) if captured[index] == "-m")
    assert captured[keyword_index + 1] == "unit"
    assert captured[marker_index + 1] == "unit"


def test_main_finalizes_runner_exception_after_open_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    history: dict[str, Any] = {}
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        run_tests,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=tmp_path / "polylogue", as_dict=lambda: {}),
    )
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "append_verify_history", lambda payload: history.update(payload))

    def explode(_label: str, command: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        run = kwargs["run"]
        run.start_step(label="pytest focused", cmd=command)
        raise RuntimeError("focused runner exploded")

    monkeypatch.setattr(run_tests, "_run", explode)

    assert run_tests.main(["focused-selector", "--json"]) == 125
    assert history["exit_code"] == 125
    assert history["diagnosis"] == "focused_test_runner_exception"
    assert history["steps"][0]["status"] == "failed"
    assert history["steps"][0]["exit"] == 125


def test_main_returns_pytest_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(label: str, cmd: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        return 5, 0.01, {"diagnosis": "pytest_failed"}

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
    stale_report = root / run_tests.PYTEST_REPORT_PATH
    stale_statistics = root / CURRENT_STATISTICS_PATH
    stale_report.parent.mkdir(parents=True)
    stale_statistics.parent.mkdir(parents=True, exist_ok=True)
    stale_report.write_text('{"stale": true}')
    stale_statistics.write_text('{"stale": true}')
    captured: dict[str, object] = {}

    def fake_run(_label: str, _cmd: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cwd"] = kwargs["cwd"]
        Path(run_tests.PYTEST_REPORT_PATH).write_text('{"fresh": true}')
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.setattr(
        run_tests,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: SimpleNamespace(polylogue_import_path=root / "polylogue", as_dict=lambda: {}),
    )
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "_run", fake_run)
    monkeypatch.setattr(run_tests, "append_verify_history", lambda _payload: None)
    invocation_directory = subdirectory if invocation_location == "inside" else external_directory
    monkeypatch.chdir(invocation_directory)

    assert run_tests.main(["tests/unit/example.py"]) == 0

    assert captured["cwd"] == str(root)
    assert stale_report.read_text() == '{"fresh": true}'
    assert json.loads(stale_statistics.read_text())["canonical_report_status"] == "missing"
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
