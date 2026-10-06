"""Tests for the ``devtools test`` focused runner (devtools/run_tests.py)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from devtools import pytest_slot, run_tests
from devtools.pytest_invocation import (
    CLEAR_CONFIGURED_ADDOPTS,
    IGNORED_COLLECTION_ARGS,
    SUITE_COST_PLUGIN_NAME,
    TESTMON_RETENTION_PLUGIN_NAME,
    devtools_plugin_args,
    managed_plugin_args,
)
from devtools.pytest_slot import SlotOutcome
from devtools.verify_runs import (
    CURRENT_RUN_PATH,
    CURRENT_STATISTICS_PATH,
    VERIFY_RUNS_DIR,
    VerifyRun,
    git_head,
    git_worktree_content_sha256,
    pytest_command_worker_request,
)
from devtools.worker_memory import CHARGE_PROFILE_ENV, CORPUS_MAX_WORKERS, FOCUSED_MAX_WORKERS, resize_worker_argument

_HOLD_SELECTION_LOCK = run_tests._hold_selection_lock


@pytest.fixture(autouse=True)
def _no_receipt_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` tests exercise the run path, never a reused receipt from this checkout.

    The outer ``devtools test`` running this file holds the real checkout's
    selection lock, so ``main`` here never takes it; the lock has its own law.
    """
    monkeypatch.setenv(run_tests.REUSE_ENV, "0")
    monkeypatch.setattr(run_tests, "_hold_selection_lock", lambda _selection: None)
    # History resolves relative to the root under test, never the host's.
    monkeypatch.setenv("POLYLOGUE_VERIFY_HISTORY_PATH", ".cache/verify/history.jsonl")


@pytest.fixture
def isolated_focused_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic main calls own their receipts and current-event mirror."""
    from tests.infra.devtools_admission_fixture import make_focused_checkout

    root = make_focused_checkout(tmp_path / "checkout")
    monkeypatch.chdir(root)
    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    return root


def _write_passing_evidence(root: Path, run: VerifyRun) -> None:
    step = run._payload["steps"][-1]
    step_dir = run.run_dir / "steps" / step["step_id"]
    (step_dir / "selection.json").write_text(json.dumps({"selected_count": 1}), encoding="utf-8")
    (step_dir / "summary.json").write_text(json.dumps({"exitstatus": 0}), encoding="utf-8")
    events = step_dir / "events"
    events.mkdir()
    (events / "gw0.jsonl").write_text(
        json.dumps({"event": "collection_finished", "updated_at": "2026-01-01T00:00:00Z"}) + "\n", encoding="utf-8"
    )
    report = step_dir / "pytest-report.json"
    report.write_text(json.dumps({"tests": [{"nodeid": "test_ok", "outcome": "passed"}]}), encoding="utf-8")


def test_build_pytest_cmd_defaults_to_single_process() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools/test_run_tests.py"])
    assert cmd[:5] == [
        str(run_tests.ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-p",
        "devtools.pytest_progress_plugin",
    ]
    assert "tests/unit/devtools/test_run_tests.py" in cmd
    assert "-n" not in cmd


@pytest.mark.parametrize(
    ("sentinel", "expected"),
    [
        ("--help", "Usage: devtools test"),
        ("-h", "Usage: devtools test"),
        ("--version", "pytest "),
        ("-V", "pytest "),
    ],
)
def test_meta_options_never_become_a_test_selection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    sentinel: str,
    expected: str,
) -> None:
    """The pytest CLI's sibling help/version spellings never enter admission."""
    monkeypatch.setattr(run_tests, "run_pytest", lambda *_args, **_kwargs: pytest.fail("pytest launched"))

    assert run_tests.main([sentinel]) == 0
    output = capsys.readouterr().out
    assert expected in output


def test_build_pytest_cmd_uses_the_managed_plugin_contract() -> None:
    """The repository's own plugins load, then the third-party contract, in order.

    Anti-vacuity: drop one name from ``DEVTOOLS_PLUGIN_ARGS`` in the built
    command and the first slice comparison fails; reorder the two blocks and
    the second does.
    """
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools/test_run_tests.py"])

    focused_plugins = devtools_plugin_args(testmon=False)
    devtools_start = cmd.index(focused_plugins[0])
    assert [*focused_plugins] == cmd[devtools_start : devtools_start + len(focused_plugins)]
    managed_start = devtools_start + len(focused_plugins)
    assert cmd[managed_start : managed_start + 2] == ["-p", SUITE_COST_PLUGIN_NAME]
    managed_start += 2
    serial_plugins = managed_plugin_args(testmon=False, xdist=False)
    assert [*serial_plugins] == cmd[managed_start : managed_start + len(serial_plugins)]
    assert TESTMON_RETENTION_PLUGIN_NAME not in cmd
    assert "pytest-testmon" not in cmd
    assert "xdist" not in cmd
    assert CLEAR_CONFIGURED_ADDOPTS in cmd
    assert "--assert=plain" in cmd
    ignored_start = cmd.index(IGNORED_COLLECTION_ARGS[0])
    assert [*IGNORED_COLLECTION_ARGS] == cmd[ignored_start : ignored_start + len(IGNORED_COLLECTION_ARGS)]


def test_build_pytest_cmd_keeps_benchmark_collection_for_explicit_target() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/benchmarks/test_scale_tiers.py"])

    assert IGNORED_COLLECTION_ARGS[0] not in cmd


def test_build_pytest_cmd_respects_explicit_worker_flag() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit", "-n", "4"])
    # No injected -n when the caller already chose one.
    assert cmd.count("-n") == 1
    assert cmd[-3:] == ["-n", "4", "--dist=loadgroup"]
    assert "xdist" in cmd


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

    worker_flags = [
        arg for arg in command if arg in {"-n", "--numprocesses"} or arg.startswith(("-n", "--numprocesses="))
    ]
    assert len(worker_flags) == 1
    assert pytest_command_worker_request(command) == expected_request


def test_build_pytest_cmd_ignores_workers_env_for_focused_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "8")
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools/test_run_tests.py"])
    assert "-n" not in cmd
    large = run_tests.build_pytest_cmd(["tests/unit"])
    assert large[large.index("-n") + 1] == str(FOCUSED_MAX_WORKERS)


def test_a_large_selection_runs_under_xdist_and_a_small_one_does_not() -> None:
    """Width follows the selection's module count, never an ambient setting.

    Anti-vacuity: return ``[]`` unconditionally from ``_worker_args`` and the
    large selection runs in one process; drop the module threshold and the
    single file is spread over workers.
    """
    small = run_tests.build_pytest_cmd(["tests/unit/devtools/test_run_tests.py", "tests/unit/devtools/test_verify.py"])
    assert "-n" not in small
    assert "xdist" not in small

    large = run_tests.build_pytest_cmd(["tests/unit/devtools"])
    assert large[large.index("-n") + 1] == str(FOCUSED_MAX_WORKERS)
    assert "xdist" in large
    assert "--dist=loadgroup" in large


def test_focused_environment_declares_the_focused_charge_profile(tmp_path: Path) -> None:
    run = VerifyRun(tier="focused-test", argv=["tests"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])

    environment = run_tests.focused_pytest_env(run=run, artifacts=artifacts)

    assert environment[CHARGE_PROFILE_ENV] == "focused"


def test_build_pytest_cmd_preserves_explicit_xdist_distribution() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit", "-n", "4", "--dist=worksteal"])

    assert cmd.count("--dist=worksteal") == 1
    assert "--dist=loadgroup" not in cmd


@pytest.mark.parametrize(("cluster", "expected"), [("-vn2", "2"), ("-qn2", "2"), ("-xvn3", "3")])
def test_a_worker_count_inside_a_short_option_cluster_is_the_callers(cluster: str, expected: str) -> None:
    """Anti-vacuity (Codex P2, #5708): detect ``-n`` only as a whole-argument
    prefix and ``-vn2`` gets a managed ``-n`` appended after it, which argparse
    lets override the caller's own worker request. The cluster reaches pytest
    as separate options, so the slot's resizer sees its worker count."""
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", cluster])

    assert cmd.count("-n") == 1
    assert pytest_command_worker_request(cmd) == expected


def test_an_n_inside_an_attached_value_is_not_a_worker_count() -> None:
    """``-kn`` is ``-k`` with the attached expression ``n``, not a worker request."""
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", "-kn"])

    assert cmd[cmd.index("-n") + 1] == str(FOCUSED_MAX_WORKERS)


def test_build_pytest_cmd_does_not_add_distribution_for_serial_run() -> None:
    cmd = run_tests.build_pytest_cmd(["tests/unit", "-n", "0"])

    assert not any(arg.startswith("--dist") for arg in cmd)


def test_prepare_nodatacow_parent_marks_basetemp_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], object, object]] = []

    class FakeProcess:
        def wait(self) -> None:
            pass

    def fake_popen(command: list[str], *, stdout: object, stderr: object) -> FakeProcess:
        calls.append((command, stdout, stderr))
        return FakeProcess()

    basetemp = tmp_path / "verify" / "pytest-run"
    monkeypatch.setattr("devtools.run_tests.subprocess.Popen", fake_popen)

    run_tests._prepare_nodatacow_parent(basetemp)

    assert calls == [(["chattr", "+C", str(basetemp.parent)], subprocess.DEVNULL, subprocess.DEVNULL)]
    assert basetemp.parent.is_dir()


def test_prepare_nodatacow_parent_is_best_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("chattr")

    monkeypatch.setattr("devtools.run_tests.subprocess.Popen", unavailable)

    run_tests._prepare_nodatacow_parent(tmp_path / "pytest-run")


def test_main_requires_a_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_tests.main([]) == 2
    err = capsys.readouterr().err
    assert "give a selection" in err
    assert "devtools verify" in err


def test_outliers_aggregate_phases_and_report_test_and_file_shares(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_dir = tmp_path / run_tests.PYTEST_REPORT_DIR
    report_dir.mkdir(parents=True)
    (report_dir / "last-pytest-parallel-1-of-3.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "tests/unit/slow.py::test_a",
                        "setup": {"duration": 1.0},
                        "call": {"duration": 9.0},
                        "teardown": {"duration": 2.0},
                    },
                    {
                        "nodeid": "tests/unit/slow.py::test_b",
                        "setup": {"duration": 0.5},
                        "call": {"duration": 3.0},
                        "teardown": {"duration": 0.5},
                    },
                    {
                        "nodeid": "tests/unit/fast.py::test_c",
                        "setup": {"duration": 0.1},
                        "call": {"duration": 1.0},
                        "teardown": {"duration": 0.1},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    for name, nodeid, duration in (
        ("last-pytest-serial.json", "tests/unit/serial.py::test_serial", 4.0),
        ("last-pytest-storage-scale.json", "tests/unit/storage_scale.py::test_scale", 5.0),
    ):
        (report_dir / name).write_text(
            json.dumps({"tests": [{"nodeid": nodeid, "call": {"duration": duration}}]}), encoding="utf-8"
        )

    run_dir = tmp_path / VERIFY_RUNS_DIR / "completed"
    steps = []
    for index, path in enumerate(sorted(report_dir.glob("last-pytest-*.json"))):
        step_id = f"{index:02d}-pytest-lane"
        destination = run_dir / "steps" / step_id / "pytest-report.json"
        destination.parent.mkdir(parents=True)
        path.rename(destination)
        steps.append({"name": f"pytest lane {index}", "step_id": step_id})
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "tier": "all",
                "status": "success",
                "finished_at": "2026-01-01T00:00:00Z",
                "pytest_aggregate": {"complete_corpus_covered": True},
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )

    assert run_tests.print_outliers(5, root=tmp_path) == 0
    output = capsys.readouterr().out
    assert "Full-run receipts: 3; tests: 5; serial time: 26.20s" in output
    assert "tests/unit/serial.py::test_serial" in output
    assert "tests/unit/storage_scale.py::test_scale" in output
    assert "Top 5 slowest tests (100.0% of serial time):" in output
    assert "tests/unit/slow.py::test_a" in output
    assert "Top 4 slowest files (100.0% of serial time):" in output
    assert "tests/unit/slow.py" in output


def test_parse_outliers_supports_default_and_explicit_limits() -> None:
    assert run_tests._parse_outliers(["--outliers"]) == (10, [])
    assert run_tests._parse_outliers(["--outliers", "25"]) == (25, [])
    assert run_tests._parse_outliers(["--outliers=3"]) == (3, [])


def test_main_strips_dispatch_json_flag(monkeypatch: pytest.MonkeyPatch, isolated_focused_checkout: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_pytest(cmd: list[str], **kwargs: Any) -> SlotOutcome:
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        env = kwargs["env"]
        Path(env["POLYLOGUE_PYTEST_SELECTION_PATH"]).write_text(json.dumps({"selected_count": 1}), encoding="utf-8")
        Path(env["POLYLOGUE_PYTEST_SUMMARY_PATH"]).write_text(json.dumps({"exitstatus": 0}), encoding="utf-8")
        events = Path(env["POLYLOGUE_PYTEST_EVENTS_DIR"])
        events.mkdir()
        (events / "gw0.jsonl").write_text(
            json.dumps({"event": "collection_finished", "updated_at": "2026-01-01T00:00:00Z"}) + "\n", encoding="utf-8"
        )
        report_arg = next(arg for arg in cmd if arg.startswith("--polylogue-report-file="))
        report = Path(report_arg.split("=", 1)[1])
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"tests": [{"nodeid": "test_ok", "outcome": "passed"}]}), encoding="utf-8")
        return SlotOutcome(returncode=0, slot="held")

    monkeypatch.setattr("devtools.run_tests._clear_pytest_report", lambda _cmd: None)
    # The seam is the pytest slot, not subprocess: a run outside the slot is
    # queued rather than executed here.
    monkeypatch.setattr("devtools.run_tests.run_pytest", fake_run_pytest)
    monkeypatch.setattr("devtools.run_tests.git_head", lambda _root: "abc123")
    monkeypatch.setattr("devtools.run_tests.append_verify_history", lambda payload: captured.update(history=payload))
    monkeypatch.setattr(
        "devtools.run_tests.append_verification_evidence", lambda payload: captured.update(evidence=payload)
    )
    assert run_tests.main(["tests/unit/pipeline", "--json"]) == 0
    assert "--json" not in captured["cmd"]
    assert "tests/unit/pipeline" in captured["cmd"]
    assert captured["env"]["POLYLOGUE_PYTEST_EVENTS_DIR"].endswith("/events")
    assert captured["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTEST_ADDOPTS" not in captured["env"]
    assert "PYTEST_PLUGINS" not in captured["env"]
    assert captured["history"]["git_head"] == "abc123"
    assert isinstance(captured["history"]["git_dirty"], bool)
    assert captured["history"]["verification_scope"] == "affected"
    assert captured["history"]["status"] == "success"
    assert captured["evidence"] == captured["history"]


def test_queued_focused_receipt_identifies_execution_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A mutation after submission changes the content named by run.json."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    ignored = tmp_path / ".gitignore"
    # The autouse fixture puts the XDG homes, and with them the durable
    # verification evidence lane, inside tmp_path; they are not worktree content.
    ignored.write_text(".cache/\nxdg-*/\n", encoding="utf-8")
    source = tmp_path / "test_input.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "test_input.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        cwd=tmp_path,
        check=True,
    )
    submitted_head = git_head(tmp_path)
    submitted_digest = git_worktree_content_sha256(tmp_path)
    assert submitted_head is not None and submitted_digest is not None

    untracked = tmp_path / "new_input.py"
    executed: dict[str, Any] = {}

    def queued_pytest(cmd: list[str], **kwargs: Any) -> SlotOutcome:
        source.write_text("value = 2\n", encoding="utf-8")
        tracked_edit_digest = git_worktree_content_sha256(tmp_path)
        untracked.write_text("value = 3\n", encoding="utf-8")
        execution_digest = git_worktree_content_sha256(tmp_path)
        assert tracked_edit_digest not in (None, submitted_digest)
        assert execution_digest not in (None, submitted_digest, tracked_edit_digest)
        executed["digest"] = execution_digest
        executed["provenance"] = pytest_slot._focused_worktree_provenance(kwargs["cwd"], kwargs["env"])
        env = kwargs["env"]
        Path(env["POLYLOGUE_PYTEST_SELECTION_PATH"]).write_text(json.dumps({"selected_count": 1}), encoding="utf-8")
        Path(env["POLYLOGUE_PYTEST_SUMMARY_PATH"]).write_text(json.dumps({"exitstatus": 0}), encoding="utf-8")
        events = Path(env["POLYLOGUE_PYTEST_EVENTS_DIR"])
        events.mkdir()
        (events / "gw0.jsonl").write_text(
            json.dumps({"event": "collection_finished", "updated_at": "2026-01-01T00:00:00Z"}) + "\n",
            encoding="utf-8",
        )
        report_arg = next(arg for arg in cmd if arg.startswith("--polylogue-report-file="))
        Path(report_arg.split("=", 1)[1]).write_text(
            json.dumps({"tests": [{"nodeid": "test_input.py", "outcome": "passed"}]}), encoding="utf-8"
        )
        return SlotOutcome(
            returncode=0,
            slot="agentctl job 1",
            receipt={"worktree_provenance": executed["provenance"]},
        )

    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _path: None)
    monkeypatch.setattr(run_tests, "run_pytest", queued_pytest)
    assert run_tests.main(["test_input.py"]) == 0

    receipt = json.loads((tmp_path / CURRENT_RUN_PATH).read_text(encoding="utf-8"))
    assert receipt["git_head"] == submitted_head
    assert receipt["git_dirty"] is True
    assert receipt["git_worktree_content_sha256"] == executed["digest"]
    assert receipt["worktree_capture_source"] == "pytest_slot_start"
    assert receipt["git_branch"] == "test/feature"
    assert json.loads((tmp_path / receipt["artifact_dir"] / "run.json").read_text(encoding="utf-8")) == receipt

    source.write_text("value = 1\n", encoding="utf-8")
    untracked.unlink()
    assert git_worktree_content_sha256(tmp_path) == submitted_digest


@pytest.mark.parametrize("runner", ["managed", "isolated"])
def test_run_uses_the_requested_runner(monkeypatch: pytest.MonkeyPatch, runner: str) -> None:
    """Managed remains the default; isolation is an explicit CI route."""
    called: list[str] = []

    def managed(*_args: Any, **_kwargs: Any) -> SlotOutcome:
        called.append("managed")
        return SlotOutcome(returncode=0, slot="managed")

    def isolated(*_args: Any, **_kwargs: Any) -> SlotOutcome:
        called.append("isolated")
        return SlotOutcome(returncode=0, slot="isolated")

    monkeypatch.setattr(run_tests, "run_pytest", managed)
    monkeypatch.setattr(run_tests, "run_pytest_isolated", isolated)
    monkeypatch.setattr(run_tests, "write_run_receipt", lambda _path: None)

    exit_code, _elapsed, metadata = run_tests._run(
        "pytest focused",
        ["pytest"],
        cwd=".",
        env={},
        run=cast(VerifyRun, None),
        artifacts=cast(Any, SimpleNamespace(step_dir=Path("."))),
        report_path=Path("missing-report.json"),
        runner=runner,
    )

    assert exit_code == 0
    assert called == [runner]
    assert metadata["pytest_slot"] == runner


def test_main_preserves_relative_selection_from_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
    isolated_focused_checkout: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(_label: str, cmd: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cmd"] = cmd
        _write_passing_evidence(run_tests.ROOT, _kwargs["run"])
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
    isolated_focused_checkout: Path,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_run(_label: str, cmd: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured["cmd"] = cmd
        _write_passing_evidence(run_tests.ROOT, _kwargs["run"])
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
        lambda *_args, **_kwargs: None,
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


def test_interrupted_focused_run_publishes_an_interrupted_evidence_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ctrl-C during ``devtools test`` reaches the evidence lane as ``interrupted``.

    Anti-vacuity: dropping ``termination_reason`` from the focused aggregate,
    or not treating ``pytest_interrupted`` as an interruption, publishes the
    receipt with status ``failed``.
    """
    from devtools.verify_runs import canonical_verification_receipt

    published: list[dict[str, Any]] = []

    def interrupt(*_args: Any, **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr(run_tests, "_run", interrupt)
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "append_verify_history", lambda _payload: None)
    monkeypatch.setattr(run_tests, "append_verification_evidence", lambda payload: published.append(dict(payload)))

    assert run_tests.main(["tests/unit/example.py"]) == 130

    [payload] = published
    assert payload["pytest_aggregate"]["termination_reason"] == "operator_interrupt"
    assert canonical_verification_receipt(payload)["status"] == "interrupted"


def test_main_records_rewritten_focused_exit_and_why_surfaces_the_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutation: letting a stale zero overwrite the finalized step hides this failure from ``why``."""
    from devtools import why

    history: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "append_verify_history", lambda payload: history.update(payload))
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    original_start = VerifyRun.start_step

    def start_step(self: VerifyRun, **kwargs: Any) -> Any:
        artifacts = original_start(self, **kwargs)
        captured["artifacts"] = artifacts
        return artifacts

    def stale_zero(*_args: Any, **_kwargs: Any) -> tuple[int, float, dict[str, str]]:
        artifacts = captured["artifacts"]
        artifacts.selection_path.write_text(json.dumps({"selected_count": 1}), encoding="utf-8")
        artifacts.summary_path.write_text(json.dumps({"exitstatus": 0}), encoding="utf-8")
        artifacts.events_dir.mkdir()
        (artifacts.events_dir / "gw0.jsonl").write_text(
            json.dumps({"event": "collection_finished", "updated_at": "2026-01-01T00:00:00Z"}) + "\n",
            encoding="utf-8",
        )
        report = artifacts.step_dir / "pytest-report.json"
        report.write_text(json.dumps({"tests": []}), encoding="utf-8")
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.setattr(VerifyRun, "start_step", start_step)
    monkeypatch.setattr(run_tests, "_run", stale_zero)

    assert run_tests.main(["tests/unit/example.py"]) == 1
    step = history["steps"][0]
    assert step["process_exit"] == 0
    assert step["exit"] == 1
    assert step["status"] == "failed"
    assert step["diagnosis"] == "pytest_report_incomplete"
    stream = __import__("io").StringIO()
    why._render(history, stream)
    assert step["step_id"] in stream.getvalue()


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
    isolated_focused_checkout: Path,
) -> None:
    captured: list[str] = []
    monkeypatch.chdir(run_tests.ROOT / "tests")
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)

    def capture(_label: str, command: list[str], **_kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        captured.extend(command)
        _write_passing_evidence(run_tests.ROOT, _kwargs["run"])
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
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "append_verify_history", lambda payload: history.update(payload))

    def explode(_label: str, command: list[str], **kwargs: Any) -> tuple[int, float, dict[str, Any]]:
        raise RuntimeError("focused runner exploded")

    monkeypatch.setattr(run_tests, "_run", explode)

    assert run_tests.main(["focused-selector", "--json"]) == 125
    assert history["exit_code"] == 125
    assert history["diagnosis"] == "focused_test_runner_exception"
    assert history["steps"][0]["status"] == "failed"
    assert history["steps"][0]["exit"] == 125


def test_main_returns_pytest_exit_code(monkeypatch: pytest.MonkeyPatch, isolated_focused_checkout: Path) -> None:
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
        _write_passing_evidence(run_tests.ROOT, kwargs["run"])
        return 0, 0.01, {"diagnosis": "pytest_passed"}

    monkeypatch.setattr(run_tests, "ROOT", root)
    monkeypatch.setattr(
        run_tests,
        "assert_polylogue_matches_checkout",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "_run", fake_run)
    monkeypatch.setattr(run_tests, "append_verify_history", lambda _payload: None)
    invocation_directory = subdirectory if invocation_location == "inside" else external_directory
    monkeypatch.chdir(invocation_directory)

    assert run_tests.main(["tests/unit/example.py"]) == 0

    assert captured["cwd"] == str(root)
    assert json.loads(stale_report.read_text())["tests"][0]["outcome"] == "passed"
    assert json.loads(stale_statistics.read_text())["canonical_report_status"] == "present"
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


def test_absent_paths_are_resolved_against_the_checkout_not_the_caller_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pytest exit 5 must not be explained as "these paths do not exist".

    Anti-vacuity: resolving the selection against the process working
    directory instead of the checkout root reports every relative path as
    absent, which is the misleading refusal this guards.
    """
    checkout = tmp_path / "checkout"
    (checkout / "tests" / "unit").mkdir(parents=True)
    (checkout / "tests" / "unit" / "test_present.py").write_text("", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    selection = [
        "tests/unit/test_present.py",
        "tests/unit/test_present.py::test_case",
        "tests/unit/test_deleted.py",
        "-k",
        "hybrid",
    ]

    assert run_tests.absent_selection_paths(selection, root=checkout) == ["tests/unit/test_deleted.py"]


def test_main_refuses_a_missing_path_before_queueing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing selection path is refused before pool admission.

    Anti-vacuity: drop the pre-admission check in ``main`` and the fake slot
    below is reached, failing the test, which is the minutes-long queue wait a
    typo'd path used to cost.
    """

    def must_not_queue(*_args: Any, **_kwargs: Any) -> SlotOutcome:
        raise AssertionError("a selection with a missing path was queued for the pytest slot")

    monkeypatch.setattr("devtools.run_tests.run_pytest", must_not_queue)

    assert run_tests.main(["tests/unit/devtools/test_run_tests.py", "tests/unit/test_no_such_file.py"]) == 4
    assert "tests/unit/test_no_such_file.py" in capsys.readouterr().err


def test_pre_admission_gate_ignores_option_values_that_look_like_paths() -> None:
    """Output paths passed to options are not selections, even when they end in ``.py``.

    Anti-vacuity: treat every absent ``*.py`` argument as a selection and the
    ``--log-file`` value below is refused before pytest runs.
    """
    assert not run_tests._is_test_module_name("output.py")
    assert not run_tests._is_test_module_name("results.xml")
    assert run_tests._is_test_module_name("test_widget.py")
    assert run_tests._is_test_module_name("widget_test.py")
    assert run_tests._certain_selections(
        ["--ignore", "tests/unit/test_retired.py", "tests/unit/test_a.py", "--junit-xml=out.xml", "tests/test_b.py"]
    ) == ["--ignore", "tests/unit/test_a.py", "--junit-xml=out.xml", "tests/test_b.py"]


def test_focused_run_never_loads_or_names_a_testmon_graph(tmp_path: Path) -> None:
    """Focused checks preserve the broad graph rather than making a scratch one."""
    run = VerifyRun(tier="focused-test", argv=["tests"], git_head="head", root=tmp_path)
    artifacts = run.start_step(label="pytest focused", cmd=["pytest"])

    environment = run_tests.focused_pytest_env(run=run, artifacts=artifacts)

    assert "TESTMON_DATAFILE" not in environment
    assert not (tmp_path / ".cache" / "testmon").exists()


def _focused_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    result: tuple[int, float, dict[str, Any]],
    write_evidence: bool,
) -> dict[str, Any]:
    """Drive ``devtools test`` against a throwaway checkout, returning its history payload."""
    history: dict[str, Any] = {}
    captured: dict[str, Any] = {}
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_tests, "assert_polylogue_matches_checkout", lambda *_a, **_k: None)
    monkeypatch.setattr(run_tests, "git_head", lambda _root: "head")
    monkeypatch.setattr(run_tests, "append_verify_history", lambda payload: history.update(payload))
    monkeypatch.setattr(run_tests, "_clear_pytest_report", lambda _cmd: None)
    original_start = VerifyRun.start_step

    def start_step(self: VerifyRun, **kwargs: Any) -> Any:
        captured["run"] = self
        return original_start(self, **kwargs)

    def fake_run(*_a: Any, **_k: Any) -> tuple[int, float, dict[str, Any]]:
        if write_evidence:
            _write_passing_evidence(tmp_path, cast(VerifyRun, captured["run"]))
        return result

    monkeypatch.setattr(VerifyRun, "start_step", start_step)
    monkeypatch.setattr(run_tests, "_run", fake_run)
    history["exit"] = run_tests.main(["tests/unit/example.py"])
    return history


def test_every_run_names_the_receipt_it_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The last line carries the verdict and a receipt that exists.

    Anti-vacuity: naming a path the runner never writes fails the existence
    check, and printing the footer only for failures fails on this green run.
    """
    history = _focused_run(monkeypatch, tmp_path, result=(0, 0.01, {"diagnosis": "pytest_passed"}), write_evidence=True)
    assert history["exit"] == 0

    final = capsys.readouterr().err.strip().splitlines()[-1]
    assert final.startswith("devtools test: PASSED exit=0 diagnosis=pytest_passed receipt=")
    receipt = tmp_path / final.split("receipt=", 1)[1].split()[0]
    # The line names the checkout it tested, so a cited receipt is self-identifying.
    assert " checkout=" in final and " branch=" in final and " head=" in final
    assert receipt.is_file()
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert recorded["exit_code"] == 0
    assert recorded["run_id"] == history["run_id"]
    assert recorded["pytest_aggregate"]["outcomes"] == {"passed": 1}


def test_a_run_that_never_acquired_the_slot_keeps_its_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anti-vacuity: without the terminal-diagnosis carve-out the receipt reports
    the absence of evidence instead of the refusal that caused it."""
    history = _focused_run(
        monkeypatch,
        tmp_path,
        result=(
            125,
            0.01,
            {
                "diagnosis": "pytest_slot_unavailable",
                "error": "the runtime is unreachable",
                "termination_reason": "pytest_slot_unavailable",
            },
        ),
        write_evidence=False,
    )
    assert history["exit"] == 125
    assert history["diagnosis"] == "pytest_slot_unavailable"
    assert history["steps"][0]["diagnosis"] == "pytest_slot_unavailable"

    lines = capsys.readouterr().err.strip().splitlines()
    # The artifact pointer precedes the verdict, which stays the last line and
    # names the checkout it tested.
    assert lines[-2].startswith("devtools test: artifacts=")
    assert lines[-1].startswith("devtools test: FAILED exit=125 diagnosis=pytest_slot_unavailable receipt=")
    assert " branch=" in lines[-1]
    receipt = tmp_path / run_tests.PYTEST_REPORT_DIR / "runs" / history["run_id"] / "run.json"
    recorded = json.loads(receipt.read_text(encoding="utf-8"))
    assert recorded["exit_code"] == 125
    assert recorded["diagnosis"] == "pytest_slot_unavailable"


@pytest.mark.parametrize("runner", ["managed", "isolated"])
def test_a_focused_failure_retains_its_first_report_without_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runner: str
) -> None:
    report_path = tmp_path / "pytest-report.json"
    original = json.dumps({"exitcode": 1, "tests": [{"nodeid": "tests/test_a.py::test_red", "outcome": "failed"}]})
    report_path.write_text(original, encoding="utf-8")
    calls: list[list[str]] = []

    def execute(command: list[str], **kwargs: Any) -> SlotOutcome:
        calls.append(command)
        return SlotOutcome(returncode=1, slot="held")

    monkeypatch.setattr(run_tests, "write_run_receipt", lambda _path: None)
    monkeypatch.setattr(run_tests, "run_pytest", execute)
    monkeypatch.setattr(run_tests, "run_pytest_isolated", execute)
    exit_code, _elapsed, metadata = run_tests._run(
        "pytest focused",
        ["pytest"],
        cwd=str(tmp_path),
        env={},
        run=cast(VerifyRun, None),
        artifacts=cast(Any, SimpleNamespace(step_dir=tmp_path)),
        report_path=report_path,
        runner=runner,
    )
    assert exit_code == 1
    assert metadata["diagnosis"] == "pytest_failed"
    assert "rerun" not in metadata
    assert calls == [["pytest"]]
    assert report_path.read_text(encoding="utf-8") == original


def test_an_unfinishable_focused_run_is_never_adjudicated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An internal execution failure also retains its original exit."""
    report_path = tmp_path / "pytest-report.json"
    report_path.write_text(
        json.dumps({"tests": [{"nodeid": "tests/test_a.py::test_x", "outcome": "failed"}]}), encoding="utf-8"
    )

    monkeypatch.setattr(run_tests, "write_run_receipt", lambda _path: None)
    monkeypatch.setattr(run_tests, "run_pytest", lambda *_a, **_k: SlotOutcome(returncode=3, slot="held"))

    exit_code, _elapsed, metadata = run_tests._run(
        "pytest focused",
        ["pytest"],
        cwd=str(tmp_path),
        env={},
        run=cast(VerifyRun, None),
        artifacts=cast(Any, SimpleNamespace(step_dir=tmp_path)),
        report_path=report_path,
    )

    assert exit_code == 3
    assert "rerun" not in metadata


def _green_receipt(runs: Path, name: str, *, argv: list[str], digest: str, **overrides: Any) -> Path:
    import platform
    import sys

    # Reuse consults Git for ignored paths, so the checkout is a repository,
    # and only named files that exist are reusable.
    checkout = runs.parents[2]
    if not (checkout / ".git").exists():
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    for argument in argv:
        if not argument.startswith("-"):
            module = checkout / argument.split("::", 1)[0]
            module.parent.mkdir(parents=True, exist_ok=True)
            module.touch()

    run_dir = runs / name
    run_dir.mkdir(parents=True)
    payload: dict[str, Any] = {
        "status": "success",
        "exit_code": 0,
        "argv": argv,
        "git_worktree_content_sha256": digest,
        "pytest_aggregate": {"terminal_green": True},
        "environment_fingerprint": {
            "checkout_root": str(checkout.absolute()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
    }
    payload.update(overrides)
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    return run_dir / "run.json"


def test_a_green_run_of_the_same_selection_and_tree_is_reused(tmp_path: Path) -> None:
    """Only an exact match on selection, tree digest and interpreter is reused.

    Anti-vacuity: drop any one comparison in ``reusable_green_receipt`` and one
    of the near-miss receipts below is returned instead of ``None``.
    """
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    expected = _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") == expected
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d2") is None
    assert run_tests.reusable_green_receipt(["tests/unit/test_b.py"], root=tmp_path, content_sha256="d1") is None
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256=None) is None


def test_a_receipt_from_another_checkout_is_never_reused(tmp_path: Path) -> None:
    import platform
    import sys

    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(
        runs,
        "20260101T000000Z-focused-test-1-a",
        argv=selection,
        digest="d1",
        environment_fingerprint={
            "checkout_root": str((tmp_path / "sibling").absolute()),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
        },
    )

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "failed", "exit_code": 1},
        {"pytest_aggregate": {"terminal_green": False}},
        {"environment_fingerprint": {"python_executable": "/other/python", "python_version": "3.0.0"}},
    ],
)
def test_a_red_or_foreign_run_is_never_reused(tmp_path: Path, overrides: dict[str, Any]) -> None:
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1", **overrides)

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_green_older_than_a_pruned_red_is_not_reused(tmp_path: Path) -> None:
    """Anti-vacuity: skip the history check and the surviving older green is
    returned although a later red of unknown inputs was pruned."""
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    expected = _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")
    history = tmp_path / ".cache" / "verify" / "history.jsonl"
    history.write_text(
        json.dumps({"run_id": "20260101T000000Z-focused-test-1-a", "status": "success"}) + "\n",
        encoding="utf-8",
    )
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") == expected

    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run_id": "20260102T000000Z-focused-test-2-b", "status": "failed"}) + "\n")

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_pruned_red_later_in_the_same_second_blocks_reuse(tmp_path: Path) -> None:
    """Anti-vacuity: compare history run ids lexically and the pruned red,
    whose suffix sorts below the green's, is taken for an earlier run."""
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(
        runs,
        "20260101T000000Z-focused-test-1-ff",
        argv=selection,
        digest="d1",
        started_at="2026-01-01T00:00:00.1+00:00",
    )
    (tmp_path / ".cache" / "verify" / "history.jsonl").write_text(
        json.dumps(
            {
                "run_id": "20260101T000000Z-focused-test-1-00",
                "status": "failed",
                "started_at": "2026-01-01T00:00:00.9+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_later_red_in_the_same_second_outranks_a_green(tmp_path: Path) -> None:
    """Anti-vacuity: order by directory name alone and the green, whose random
    suffix sorts higher, is returned although the red started after it."""
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(
        runs,
        "20260101T000000Z-focused-test-1-ff",
        argv=selection,
        digest="d1",
        started_at="2026-01-01T00:00:00.1+00:00",
    )
    _green_receipt(
        runs,
        "20260101T000000Z-focused-test-1-00",
        argv=selection,
        digest="d1",
        started_at="2026-01-01T00:00:00.9+00:00",
        status="failed",
        exit_code=1,
    )

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_main_reuses_a_green_receipt_without_queueing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Anti-vacuity: without the reuse branch ``main`` reaches the fake slot and fails."""
    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    receipt = tmp_path / "run.json"
    receipt.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    monkeypatch.setattr(run_tests, "reusable_green_receipt", lambda *_a, **_k: receipt)
    monkeypatch.setattr(run_tests, "git_worktree_content_sha256", lambda _root: "d1")

    def must_not_queue(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a reusable green run was queued again")

    monkeypatch.setattr("devtools.run_tests.run_pytest", must_not_queue)

    assert run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly"]) == 0
    assert f"receipt={receipt}" in capsys.readouterr().err


def test_identical_selections_in_one_checkout_share_one_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second caller with the same selection waits until the first releases.

    Anti-vacuity: remove the blocking acquisition in ``_hold_selection_lock``
    and the second caller returns while the first still holds the lock, so
    the first ``wait`` below sees it done.
    """
    import fcntl
    import hashlib
    import os
    import threading

    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    lock_dir = tmp_path / ".cache" / "verify" / "inflight"
    lock_dir.mkdir(parents=True)
    digest = hashlib.sha256(json.dumps(selection).encode("utf-8")).hexdigest()[:24]
    held = os.open(lock_dir / f"{digest}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    done = threading.Event()

    def second_caller() -> None:
        _HOLD_SELECTION_LOCK(selection)
        done.set()

    thread = threading.Thread(target=second_caller, daemon=True)
    thread.start()
    try:
        assert not done.wait(timeout=0.5), "the second caller did not wait for the first"
        fcntl.flock(held, fcntl.LOCK_UN)
        assert done.wait(timeout=10), "the second caller never acquired after release"
    finally:
        os.close(held)
        thread.join(timeout=10)
        for handle in run_tests._SELECTION_LOCKS.values():
            os.close(handle)
        run_tests._SELECTION_LOCKS.clear()


def test_reuse_is_keyed_on_the_execution_environment(tmp_path: Path) -> None:
    """A run under a different Hypothesis profile never answers from another's receipt.

    Anti-vacuity: drop the ``execution_environment_key`` comparison and the
    ``default``-profile lookup returns the ``verify``-profile receipt.
    """
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/property/test_a.py", "--randomly-seed=1"]
    verify_key = run_tests.execution_environment_key({"HYPOTHESIS_PROFILE": "verify", "SHELL": "/bin/zsh"})
    default_key = run_tests.execution_environment_key({"HYPOTHESIS_PROFILE": "default", "SHELL": "/bin/zsh"})
    receipt = _green_receipt(
        runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1", execution_environment_key=verify_key
    )

    assert verify_key != default_key
    # A variable outside the declared inputs does not split the key.
    assert run_tests.execution_environment_key({"HYPOTHESIS_PROFILE": "verify", "SHELL": "/bin/bash"}) == verify_key
    assert (
        run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1", environment_key=verify_key)
        == receipt
    )
    assert (
        run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1", environment_key=default_key)
        is None
    )


def test_a_reused_receipt_is_emitted_as_json_when_asked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Anti-vacuity: drop the ``use_json`` branch on reuse and stdout is empty."""
    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    receipt = tmp_path / "run.json"
    receipt.write_text(json.dumps({"status": "success", "run_id": "r1"}), encoding="utf-8")
    monkeypatch.setattr(run_tests, "reusable_green_receipt", lambda *_a, **_k: receipt)
    monkeypatch.setattr(run_tests, "git_worktree_content_sha256", lambda _root: "d1")

    assert run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "r1"


@pytest.mark.parametrize("flag", ["--lf", "--last-failed", "--ff", "--sw", "--lfnf=all"])
def test_stateful_selectors_are_never_answered_from_a_receipt(tmp_path: Path, flag: str) -> None:
    """``--lf`` selects from pytest's mutable cache, so identical argv is not identical work.

    Anti-vacuity: drop the stateful-selector refusal and the matching receipt
    below is returned.
    """
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", flag]
    _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


@pytest.mark.parametrize(
    ("selection", "eligible"),
    [
        (["tests/unit/test_a.py", "-k", "fast", "-x", "--tb=short", "-p", "no:randomly"], True),
        (["tests/unit/test_a.py", "--randomly-seed=7"], True),
        # pytest-randomly draws a new order each run unless one is fixed.
        (["tests/unit/test_a.py"], False),
        (["tests/unit/test_a.py", "--randomly-seed=last"], False),
        (["tests/unit/test_a.py", "-v"], False),
        (["tests/unit/test_a.py", "--junitxml=/tmp/report.xml"], False),
        (["tests/unit/test_a.py", "--cache-clear"], False),
        (["/tmp/test_external.py"], False),
        (["tests/unit/test_a.py", "-c", "/tmp/pytest.ini"], False),
        # A directory may hold ignored, collectable modules the digest omits.
        (["tests/unit"], False),
        # -s is asked for to see live output, which a receipt cannot replay.
        (["tests/unit/test_a.py", "-s"], False),
    ],
)
def test_only_checkout_local_selections_with_inert_options_are_reused(
    tmp_path: Path, selection: list[str], eligible: bool
) -> None:
    """A receipt answers only for what its tree digest covers and what a rerun would redo.

    Anti-vacuity: accept any option, or any path, in ``_reuse_eligible`` and
    one of the refused selections is answered from a receipt without running.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (tmp_path / "tests" / "unit" / "test_a.py").touch()
    assert run_tests._reuse_eligible(selection, root=tmp_path) is eligible


def test_the_database_revision_is_read_from_its_marker_without_a_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity (Codex P1, #5708): enumerate the example database for the
    key and every focused run pays a walk that grows with the database."""
    examples = tmp_path / ".cache" / "hypothesis" / "examples"
    (examples / "abc").mkdir(parents=True)
    (examples / "abc" / "def").write_bytes(b"counterexample")

    def refuse_walk(self: Path, pattern: str) -> object:
        raise AssertionError("the example database was walked")

    monkeypatch.setattr(Path, "rglob", refuse_walk)
    assert run_tests.hypothesis_database_revision(tmp_path) == "absent"


def test_a_new_hypothesis_counterexample_or_golden_switch_changes_the_key(tmp_path: Path) -> None:
    """Inputs outside the tree digest still decide whether a receipt answers.

    Anti-vacuity: drop the database revision or ``UPDATE_GOLDEN`` from the key
    and the corresponding pair below compares equal.
    """
    from devtools.hypothesis_database import RevisionedExampleDatabase

    before = run_tests.hypothesis_database_revision(tmp_path)
    RevisionedExampleDatabase(tmp_path / ".cache" / "hypothesis" / "examples").save(b"key", b"counterexample")
    assert run_tests.hypothesis_database_revision(tmp_path) != before

    assert run_tests.execution_environment_key({}) != run_tests.execution_environment_key({"UPDATE_GOLDEN": "1"})


def test_a_git_ignored_selection_is_never_reused(tmp_path: Path) -> None:
    """The tree digest omits ignored files, so an ignored test is always run.

    Anti-vacuity: drop the ``_git_ignored`` check and ``.cache/test_x.py`` is
    eligible for reuse.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".cache/\n", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "test_x.py").write_text("", encoding="utf-8")
    (tmp_path / "test_y.py").write_text("", encoding="utf-8")

    assert run_tests._reuse_eligible([".cache/test_x.py"], root=tmp_path) is False
    assert run_tests._reuse_eligible(["test_y.py", "-p", "no:randomly"], root=tmp_path) is True


def test_reuse_is_refused_when_the_tree_changes_during_lookup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A save between digest and return means the receipt describes another tree.

    Anti-vacuity: drop the second digest comparison and ``main`` returns the
    reused receipt instead of reaching the (fake) slot.
    """
    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    digests = iter(["before", "after", "after", "after", "after"])
    monkeypatch.setattr(run_tests, "git_worktree_content_sha256", lambda _root: next(digests, "after"))
    monkeypatch.setattr(run_tests, "reusable_green_receipt", lambda *_a, **_k: tmp_path / "run.json")

    queued: list[bool] = []

    def reached_the_slot(*_args: Any, **_kwargs: Any) -> Any:
        queued.append(True)
        raise RuntimeError("stop after admission")

    monkeypatch.setattr("devtools.run_tests.run_pytest", reached_the_slot)

    assert run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly"]) != 0
    assert queued == [True]


def test_a_branch_switch_during_lookup_refuses_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: drop the admission check at the point of reuse and the
    feature branch's receipt is returned on the default branch."""
    from devtools.checkout_identity import CheckoutIdentity

    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    monkeypatch.setattr(run_tests, "git_worktree_content_sha256", lambda _root: "same")
    feature = CheckoutIdentity(root=tmp_path, branch="feature", head="h1", default_branch="master")
    default = CheckoutIdentity(root=tmp_path, branch="master", head="h1", default_branch="master")
    identities = iter([feature, feature, default])
    monkeypatch.setattr(run_tests, "checkout_identity", lambda _root: next(identities, default))

    def lookup(*_a: Any, **_k: Any) -> Path:
        return tmp_path / "run.json"

    monkeypatch.setattr(run_tests, "reusable_green_receipt", lookup)

    from devtools.checkout_identity import REFUSAL_EXIT

    assert run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly"]) == REFUSAL_EXIT


def test_a_standalone_flag_before_a_large_directory_keeps_xdist() -> None:
    """Anti-vacuity: treat ``-x`` as taking an operand and ``tests/unit/devtools`` counts nothing."""
    cmd = run_tests.build_pytest_cmd(["-x", "tests/unit/devtools"])
    assert cmd[cmd.index("-n") + 1] == str(FOCUSED_MAX_WORKERS)


def test_benchmark_selections_never_get_automatic_workers() -> None:
    """Anti-vacuity: drop the benchmark exclusion and ``-n`` is appended beside ``-p no:xdist``."""
    cmd = run_tests.build_pytest_cmd(["tests/benchmarks", "--benchmark-enable", "-p", "no:xdist"])
    assert "-n" not in cmd


def test_explicit_xdist_disablement_is_honored_for_any_selection() -> None:
    """Anti-vacuity: guard only benchmarks and ``-p no:xdist`` gets ``-n 4`` beside it."""
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", "-p", "no:xdist"])
    assert "-n" not in cmd


@pytest.mark.parametrize("capture", [["-s"], ["--capture=no"], ["--capture", "no"]])
def test_uncaptured_output_keeps_a_large_selection_serial(capture: list[str]) -> None:
    """Anti-vacuity: drop the capture override and ``-n 4`` swallows the live output ``-s`` asked for."""
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", *capture])
    assert "-n" not in cmd


@pytest.mark.parametrize("value_option", [["-r", "f"], ["--color", "yes"], ["--show-capture", "no"]])
def test_an_option_value_is_not_a_path(value_option: list[str]) -> None:
    """Anti-vacuity: classify arity from a hand-kept list that omits the option
    and its value is the only (missing) path, so a pathless run counts zero
    modules and the whole suite runs serially."""
    assert run_tests._selected_test_modules(value_option) == run_tests._selected_test_modules([])


@pytest.mark.parametrize("debugger", ["--pdb", "--trace"])
def test_an_interactive_debugger_keeps_a_large_selection_serial(debugger: str) -> None:
    """Anti-vacuity: drop the debugger override and ``-n 4`` gives the
    debugger a worker with no standard input."""
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", debugger])
    assert "-n" not in cmd


def test_a_forced_rerun_takes_the_selection_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``--rerun`` holds the same lock as a reusing caller, while skipping reuse.

    Anti-vacuity: take the lock only on the reuse branch and ``--rerun`` never
    reaches ``_hold_selection_lock``, so its red receipt can land while another
    caller is still answering from an older green one.
    """

    class LockedError(Exception):
        pass

    held: list[list[str]] = []

    def record(selection: list[str]) -> None:
        held.append(list(selection))
        raise LockedError

    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    monkeypatch.setattr(run_tests, "_hold_selection_lock", record)

    with pytest.raises(LockedError):
        run_tests.main(["tests/unit/devtools/test_run_tests.py", "--rerun"])
    assert held == [["tests/unit/devtools/test_run_tests.py"]]


def test_an_isolated_run_is_never_answered_from_a_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: allow reuse for ``--runner isolated`` and the managed receipt returns without running."""
    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    monkeypatch.setattr(run_tests, "reusable_green_receipt", lambda *_a, **_k: tmp_path / "run.json")
    ran: list[bool] = []

    def isolated(*_args: Any, **_kwargs: Any) -> Any:
        ran.append(True)
        raise RuntimeError("stop after admission")

    monkeypatch.setattr("devtools.run_tests.run_pytest_isolated", isolated)

    run_tests.main(["tests/unit/devtools/test_run_tests.py", "--runner", "isolated"])
    assert ran == [True]


def test_reuse_is_refused_when_the_example_database_moves_during_lookup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: recheck only the tree digest and a counterexample saved by
    a concurrent selection mid-lookup is never replayed."""
    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    monkeypatch.setattr(run_tests, "git_worktree_content_sha256", lambda _root: "same")
    keys = iter(["d0", "d1"])
    monkeypatch.setattr(run_tests, "_reuse_environment_key", lambda: next(keys, "d1"))
    monkeypatch.setattr(run_tests, "reusable_green_receipt", lambda *_a, **_k: tmp_path / "run.json")
    queued: list[bool] = []

    def reached_the_slot(*_args: Any, **_kwargs: Any) -> Any:
        queued.append(True)
        raise RuntimeError("stop after admission")

    monkeypatch.setattr("devtools.run_tests.run_pytest", reached_the_slot)

    run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly"])
    assert queued == [True]


def test_home_is_part_of_the_reuse_key() -> None:
    """Anti-vacuity: drop HOME from the key and these two environments compare equal."""
    assert run_tests.execution_environment_key({"HOME": "/tmp/home-a"}) != run_tests.execution_environment_key(
        {"HOME": "/tmp/home-b"}
    )


def test_a_receipt_pruned_during_lookup_sends_the_selection_to_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: suppress the read error and ``--json`` exits 0 with empty stdout."""
    monkeypatch.setenv(run_tests.REUSE_ENV, "1")
    monkeypatch.setattr(run_tests, "reusable_green_receipt", lambda *_a, **_k: tmp_path / "pruned" / "run.json")
    queued: list[bool] = []

    def reached_the_slot(*_args: Any, **_kwargs: Any) -> Any:
        queued.append(True)
        raise RuntimeError("stop after admission")

    monkeypatch.setattr("devtools.run_tests.run_pytest", reached_the_slot)

    run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly", "--json"])
    assert queued == [True]


def test_a_broad_selection_is_sized_by_the_corpus_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Anti-vacuity: keep the focused profile for any selection and ``tests``
    is admitted at four focused workers the corpus model says cannot fit."""
    seen: dict[str, str | None] = {}

    def capture(_cmd: list[str], **kwargs: Any) -> Any:
        seen["profile"] = kwargs["env"].get(CHARGE_PROFILE_ENV)
        raise RuntimeError("stop after admission")

    monkeypatch.setattr("devtools.run_tests.run_pytest", capture)
    monkeypatch.setattr(run_tests, "_selected_test_modules", lambda _selection: run_tests.BROAD_SELECTION_MODULES)
    run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly"])
    assert seen["profile"] is None

    monkeypatch.setattr(run_tests, "_selected_test_modules", lambda _selection: 1)
    run_tests.main(["tests/unit/devtools/test_run_tests.py", "-p", "no:randomly"])
    assert seen["profile"] == "focused"


def test_an_ignored_conftest_disables_reuse(tmp_path: Path) -> None:
    """A named test still loads its ancestors' conftest, which the digest omits when ignored.

    Anti-vacuity: drop the ignored-source guard and the receipt below is reused.
    """
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/foo/test_a.py", "--randomly-seed=1"]
    receipt = _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") == receipt

    (tmp_path / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("tests/unit/foo/conftest.py\n", encoding="utf-8")
    (tmp_path / "tests" / "unit" / "foo" / "conftest.py").write_text("", encoding="utf-8")

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_standalone_long_flag_before_a_directory_keeps_xdist() -> None:
    """Anti-vacuity: treat every option as taking a value and ``--strict-markers``
    swallows the directory, so no workers are requested."""
    cmd = run_tests.build_pytest_cmd(["--strict-markers", "tests/unit/devtools"])
    assert cmd[cmd.index("-n") + 1] == str(FOCUSED_MAX_WORKERS)


def test_a_pathless_selection_counts_as_the_whole_test_tree() -> None:
    """``-k expr`` alone collects ``testpaths``, so it is broad work.

    Anti-vacuity: count only explicit operands and ``-k`` counts zero modules,
    running the whole tree serially on the focused sizing.
    """
    assert run_tests._selected_test_modules(["-k", "never_matches"]) >= run_tests.BROAD_SELECTION_MODULES


def test_generated_worker_options_go_before_the_path_separator() -> None:
    """Anti-vacuity: append after ``--`` and pytest looks for a file named ``-n``."""
    cmd = run_tests.build_pytest_cmd(["--", "tests/unit/devtools"])
    separator = cmd.index("--")
    assert cmd.index("-n") < separator
    assert cmd[separator + 1 :] == ["tests/unit/devtools"]


def test_node_ids_of_one_file_count_as_one_module() -> None:
    """Anti-vacuity: count selectors instead of files and eight node ids of one
    file trigger xdist."""
    selection = [f"tests/unit/devtools/test_run_tests.py::test_{index}" for index in range(8)]
    assert run_tests._selected_test_modules(selection) == 1


def test_an_ignored_fixture_disables_reuse(tmp_path: Path) -> None:
    """Anti-vacuity: look only at ignored ``*.py`` and an ignored JSON fixture a
    parametrization globs leaves the receipt reusable."""
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    receipt = _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") == receipt
    (tmp_path / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("*.local.json\n", encoding="utf-8")
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "extra.local.json").write_text("{}", encoding="utf-8")
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_an_ignored_root_pytest_config_disables_reuse(tmp_path: Path) -> None:
    """Anti-vacuity: omit root config files from the guard and an ignored
    ``pytest.ini`` that changes collection leaves the receipt reusable."""
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")
    (tmp_path / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("pytest.ini\n", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\npython_functions = nope_*\n", encoding="utf-8")
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_newer_red_run_outranks_an_older_green(tmp_path: Path) -> None:
    """Anti-vacuity: skip non-green receipts while scanning and the older green is returned."""
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")
    _green_receipt(runs, "20260102T000000Z-focused-test-2-b", argv=selection, digest="d1", status="failed", exit_code=1)
    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_clustered_capture_flag_keeps_xdist_off() -> None:
    """Anti-vacuity (Codex P2, #5708): match ``-s`` only as a whole argument and
    ``-sv`` gets an automatic worker count, losing the live output it asked for."""
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", "-sv"])

    assert "-n" not in cmd


def test_an_explicit_automatic_width_starts_pytest_at_the_capped_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The caller's ``-vnauto`` passes through ``devtools test``; the slot's resizer caps it.

    Anti-vacuity: leave ``auto`` for xdist to resolve (the resizer used to
    return it unchanged) and the started command asks for every CPU after the
    memory cap has run.
    """
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", str(CORPUS_MAX_WORKERS + 8))
    cmd = run_tests.build_pytest_cmd(["tests/unit/devtools", "-vnauto"])
    assert pytest_command_worker_request(cmd) == "auto"

    started, basis = resize_worker_argument(
        cmd, meminfo=tmp_path / "meminfo", process_cgroup=tmp_path / "cgroup", cgroup_root=tmp_path
    )

    assert basis is not None and basis["requested_workers"] == CORPUS_MAX_WORKERS + 8
    assert pytest_command_worker_request(started) == str(basis["workers"])
    assert basis["workers"] <= CORPUS_MAX_WORKERS


def test_pruned_red_history_is_read_from_the_configured_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pruned-red check reads the history the writer appends to.

    Anti-vacuity (Codex P1, #5708): hard-code the checkout-local
    ``.cache/verify/history.jsonl`` and a red recorded at the configured
    (XDG) history path is missed, so the older green is reused.
    """
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(runs, "20260101T000000Z-focused-test-1-a", argv=selection, digest="d1")
    history = tmp_path / "state" / "history.jsonl"
    history.parent.mkdir()
    history.write_text(
        json.dumps({"run_id": "20260101T000000Z-focused-test-1-a", "status": "success"})
        + "\n"
        + json.dumps({"run_id": "20260102T000000Z-focused-test-2-b", "status": "failed"})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("POLYLOGUE_VERIFY_HISTORY_PATH", str(history))

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


@pytest.mark.parametrize(
    ("cluster", "expanded"),
    [(["-vn8"], ["-v", "-n", "8"]), (["-vpno:xdist"], ["-v", "-p", "no:xdist"]), (["-k", "-vx"], ["-k", "-vx"])],
)
def test_short_clusters_are_expanded_as_argparse_reads_them(cluster: list[str], expanded: list[str]) -> None:
    """Anti-vacuity (Codex P1/P2, #5708): leave ``-vn8`` whole and the slot's
    worker resizer, which reads only ``-n``/``--numprocesses``, keeps eight
    workers; leave ``-vpno:xdist`` whole and ``-n 4`` is added to a run that
    disabled xdist. An option's value is never split."""
    from devtools.pytest_options import expand_short_clusters

    assert expand_short_clusters(cluster) == expanded


def test_a_clustered_xdist_disable_gets_no_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity (Codex P2, #5708): check only the cluster's flags and
    ``-vpno:xdist`` still gets ``-n 4`` appended."""
    monkeypatch.setattr(run_tests, "_selected_test_modules", lambda _selection: run_tests.LARGE_SELECTION_MODULES)
    command = run_tests.build_pytest_cmd(["-vpno:xdist", "tests/unit"], report_path=tmp_path / "report.json")

    assert "-n" not in command


def test_an_interrupted_example_write_still_moves_the_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A save that fails midway leaves a revision no earlier receipt carries.

    Anti-vacuity (Codex P2, #5708): bump the marker only after the write and
    a failed or killed save keeps the old token, so a stale green is reused.
    """
    from hypothesis.database import DirectoryBasedExampleDatabase

    from devtools.hypothesis_database import RevisionedExampleDatabase, read_revision

    examples = tmp_path / "examples"
    database = RevisionedExampleDatabase(examples)
    database.save(b"k", b"v1")
    before = read_revision(examples)

    def interrupted(self: DirectoryBasedExampleDatabase, key: bytes, value: bytes) -> None:
        raise OSError("killed mid-write")

    monkeypatch.setattr(DirectoryBasedExampleDatabase, "save", interrupted)
    with pytest.raises(OSError):
        database.save(b"k", b"v2")

    assert read_revision(examples) != before


def test_rerun_after_the_separator_is_a_path() -> None:
    """Anti-vacuity (Codex P1, #5708): strip ``--rerun`` everywhere and a file
    literally named ``--rerun`` after ``--`` vanishes, running the corpus."""
    assert run_tests._parse_rerun(["--rerun", "tests/unit"]) == (True, ["tests/unit"])
    assert run_tests._parse_rerun(["--", "--rerun"]) == (False, ["--", "--rerun"])


def test_benchmark_selections_are_never_reused(tmp_path: Path) -> None:
    """Anti-vacuity (Codex P2, #5708): judge reuse by path alone and a green
    wall-clock benchmark answers later runs without timing anything."""
    benchmark = tmp_path / "tests" / "benchmarks" / "test_budget.py"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("def test_x():\n    pass\n", encoding="utf-8")

    assert run_tests._reuse_eligible(["tests/benchmarks/test_budget.py"], root=tmp_path) is False


def test_an_oomd_killed_queued_run_is_typed_oom_killed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A unit systemd-oomd killed reads as ``oom_killed``, not a missing receipt.

    The kill takes the in-unit receipt writer, so without the termination the
    run would be ``worktree_provenance_unavailable`` at exit 125.

    Anti-vacuity: drop the ``systemd_result`` read from the job's AgentCTL
    outcome, or the ``oom_killed`` early return, and the diagnosis reverts to
    the missing receipt.
    """
    state = tmp_path / "jobs"
    state.mkdir()
    reference = "polylogue-pytest_focused-0badf00d"
    (state / f"{reference}.outcome").write_text(
        json.dumps({"exit_code": 137, "outcome": "failed", "systemd_result": "oom-kill", "unit": "unit.service"}),
        encoding="utf-8",
    )
    monkeypatch.setattr("devtools.verify_runs._agentctl_state_root", lambda env=None: state)
    termination = pytest_slot._job_termination(reference)
    assert termination == {"killer": "oom-kill", "unit": "unit.service"}
    assert pytest_slot._job_termination("polylogue-pytest_focused-absent") is None

    killed = SlotOutcome(returncode=137, slot="agentctl job 9", termination=termination)
    monkeypatch.setattr(run_tests, "run_pytest", lambda *_a, **_k: killed)
    rc, _elapsed, metadata = run_tests._run(
        "pytest focused",
        ["pytest"],
        cwd=str(tmp_path),
        env={},
        run=cast(Any, None),
        artifacts=cast(Any, None),
        report_path=tmp_path / "report.json",
    )
    assert rc == 137
    assert metadata["diagnosis"] == "oom_killed"
    assert metadata["termination_killer"] == "oom-kill"
    assert metadata["termination_unit"] == "unit.service"
    # The kill took the slot receipt, so the tested tree is unknown.
    assert metadata["worktree_provenance_unknown"] is True


def test_an_oom_killed_step_keeps_its_diagnosis_over_the_missing_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: without the ``oom_killed`` terminal carve-out in
    ``finish_step`` the receipt reports ``pytest_no_report`` instead."""
    history = _focused_run(
        monkeypatch,
        tmp_path,
        result=(
            137,
            0.01,
            {
                "diagnosis": "oom_killed",
                "termination_killer": "oom-kill",
                "termination_unit": "unit.service",
                "worktree_provenance_unknown": True,
            },
        ),
        write_evidence=False,
    )
    assert history["exit"] == 137
    assert history["diagnosis"] == "oom_killed"
    assert history["steps"][0]["termination_killer"] == "oom-kill"
    # Anti-vacuity: keep the submission head and the receipt names a tree
    # pytest may never have run against.
    assert history["git_head"] is None
    assert history["worktree_capture_source"] == "unavailable"
    from devtools import verify_runs

    # The checkout at finalization must not stand in for the unknown tree.
    history["final_git_head"] = "moved-after-the-kill"
    canonical = verify_runs.canonical_verification_receipt(history)
    assert canonical["source_revision"] is None
    assert canonical["git_dirty"] is None
    # The durable projections keep who ended the step, not only that it failed.
    for durable in (
        verify_runs.canonical_verification_receipt(history)["steps"][0],
        verify_runs._semantic_history_row(history)["steps"][0],
    ):
        assert durable["termination_killer"] == "oom-kill"
        assert durable["termination_unit"] == "unit.service"


def test_a_queued_run_keeps_its_slot_receipt_and_any_recorded_killer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Anti-vacuity: drop ``pytest_slot_receipt`` from the OOM return, or the
    termination merge from the other returns, and these fields disappear."""
    provenance = {"git_head": "abc", "git_branch": "b", "git_dirty": False, "git_worktree_content_sha256": "s"}
    receipt = {"worktree_provenance": provenance, "memory_peak_mib": 900}

    def run_with(outcome: SlotOutcome) -> dict[str, Any]:
        monkeypatch.setattr(run_tests, "run_pytest", lambda *_a, **_k: outcome)
        monkeypatch.setattr(run_tests, "write_run_receipt", lambda _path: None)
        _rc, _elapsed, metadata = run_tests._run(
            "pytest focused",
            ["pytest"],
            cwd=str(tmp_path),
            env={},
            run=cast(Any, None),
            artifacts=cast(Any, None),
            report_path=tmp_path / "report.json",
        )
        return metadata

    oom = run_with(
        SlotOutcome(
            returncode=137, slot="agentctl job 9", receipt=receipt, termination={"killer": "oom-kill", "unit": "u"}
        )
    )
    assert oom["diagnosis"] == "oom_killed"
    assert oom["pytest_slot_receipt"] == receipt
    assert oom["worktree_provenance"] == provenance

    timed_out = run_with(
        SlotOutcome(
            returncode=124, slot="agentctl job 10", receipt=receipt, termination={"killer": "timeout", "unit": "u"}
        )
    )
    assert timed_out["diagnosis"] == "pytest_failed"
    assert timed_out["termination_killer"] == "timeout"
    assert timed_out["termination_unit"] == "u"


def test_a_selection_measuring_the_real_clock_is_never_reused(tmp_path: Path) -> None:
    """A module that declares ``uses_real_clock`` measures current timing.

    Anti-vacuity (Codex P2, #5708): exclude only ``tests/benchmarks`` and a
    green latency bound is answered from a receipt without measuring.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "test_latency.py").write_text(
        "import pytest\n\npytestmark = pytest.mark.uses_real_clock('asserts current CLI latency')\n",
        encoding="utf-8",
    )
    (tmp_path / "test_plain.py").write_text("def test_x() -> None: ...\n", encoding="utf-8")

    assert run_tests._reuse_eligible(["test_latency.py", "-p", "no:randomly"], root=tmp_path) is False
    assert run_tests._reuse_eligible(["test_plain.py", "-p", "no:randomly"], root=tmp_path) is True


def test_a_revision_read_waits_for_an_unfinished_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader never records a token taken while a write is still mutating.

    Anti-vacuity (Codex P2, #5708): bump without a shared lock and the reader
    returns the writer's first token at once; a writer killed after its
    mutation then leaves that token standing for changed examples.
    """
    import threading

    from hypothesis.database import DirectoryBasedExampleDatabase

    from devtools.hypothesis_database import RevisionedExampleDatabase, read_revision

    examples = tmp_path / "examples"
    database = RevisionedExampleDatabase(examples)
    database.save(b"k", b"v0")
    entered, release, read_done = threading.Event(), threading.Event(), threading.Event()
    original = DirectoryBasedExampleDatabase.save

    def paused(self: DirectoryBasedExampleDatabase, key: bytes, value: bytes) -> None:
        entered.set()
        assert release.wait(10)
        original(self, key, value)

    monkeypatch.setattr(DirectoryBasedExampleDatabase, "save", paused)
    writer = threading.Thread(target=database.save, args=(b"k", b"v1"))
    writer.start()
    assert entered.wait(10)
    seen: list[str] = []

    def read() -> None:
        seen.append(read_revision(examples))
        read_done.set()

    reader = threading.Thread(target=read)
    reader.start()
    assert not read_done.wait(0.3)
    release.set()
    writer.join(10)
    reader.join(10)

    assert seen == [read_revision(examples)]


def test_the_option_probe_ignores_ambient_pytest_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe loads the plugins the admitted run loads, not ``PYTEST_PLUGINS``.

    Anti-vacuity (Codex P2, #5708): inherit ``PYTEST_PLUGINS`` and a missing
    ambient plugin fails sizing before a valid selection reaches the pool.
    """
    from devtools import pytest_options

    seen: dict[str, str] = {}

    def fake_run(*_args: object, env: dict[str, str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.update(env)
        return subprocess.CompletedProcess([], 0, stdout='{"-x": 0}\n', stderr="")

    monkeypatch.setenv("PYTEST_PLUGINS", "missing_plugin")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    monkeypatch.setattr("devtools.pytest_options.subprocess.run", fake_run)
    pytest_options.pytest_option_nargs.cache_clear()
    try:
        assert pytest_options.pytest_option_nargs(("probe-only",)) == {"-x": 0}
    finally:
        pytest_options.pytest_option_nargs.cache_clear()

    assert "PYTEST_PLUGINS" not in seen
    assert "PYTEST_XDIST_WORKER" not in seen


def test_the_newest_matching_run_decides_however_many_share_its_second(tmp_path: Path) -> None:
    """A later red in a crowded second is never skipped for an earlier green.

    Anti-vacuity (Codex P1, #5708): cut the candidates at 50 by name before
    ordering by recorded start and the red, whose name sorts low, is dropped.
    """
    runs = tmp_path / ".cache" / "verify" / "runs"
    selection = ["tests/unit/test_a.py", "--randomly-seed=1"]
    _green_receipt(
        runs, "20260101T000000Z-focused-test-9-zzzz", argv=selection, digest="d1", started_at="2026-01-01T00:00:00.100"
    )
    for index in range(60):
        _green_receipt(
            runs,
            f"20260101T000000Z-focused-test-5-m{index:03d}",
            argv=["tests/unit/test_other.py"],
            digest="d1",
            started_at="2026-01-01T00:00:00.200",
        )
    _green_receipt(
        runs,
        "20260101T000000Z-focused-test-1-aaaa",
        argv=selection,
        digest="d1",
        started_at="2026-01-01T00:00:00.900",
        status="failed",
        exit_code=1,
    )

    assert run_tests.reusable_green_receipt(selection, root=tmp_path, content_sha256="d1") is None


def test_a_pathless_selection_is_never_reused(tmp_path: Path) -> None:
    """A ``-m``/``-k`` selection without files names modules this check cannot read.

    Anti-vacuity (Codex P2, #5708): accept a fixed-order selection with no path
    and ``-m uses_real_clock`` is answered from a receipt.
    """
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    assert run_tests._reuse_eligible(["-m", "uses_real_clock", "-p", "no:randomly"], root=tmp_path) is False


def test_python_runtime_settings_are_part_of_the_reuse_key() -> None:
    """A different hash seed or locale never answers from another's receipt.

    Anti-vacuity (Codex P2, #5708): key only ``HYPOTHESIS_``/``PYTEST_``/
    ``POLYLOGUE_`` settings and ``PYTHONHASHSEED=2`` reuses the seed-1 green.
    """
    assert run_tests.execution_environment_key({"PYTHONHASHSEED": "1"}) != run_tests.execution_environment_key(
        {"PYTHONHASHSEED": "2"}
    )
    assert run_tests.execution_environment_key({"LC_ALL": "C"}) != run_tests.execution_environment_key(
        {"LC_ALL": "pl_PL.UTF-8"}
    )


def test_arguments_after_the_separator_are_counted_as_paths(tmp_path: Path) -> None:
    """``-- -test_x.py`` names one file, not the whole tree.

    Anti-vacuity (Codex P2, #5708): apply the leading-dash test after ``--`` and
    the selection counts as the configured ``tests`` tree, crossing the xdist
    threshold.
    """
    assert run_tests._split_separator(["-x", "--", "-test_x.py"]) == (["-x"], ["-test_x.py"])
    assert run_tests._selected_test_modules(["--", "-no-such-test.py"]) == 0
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "-test_x.py").write_text("def test_x() -> None: ...\n", encoding="utf-8")
    assert run_tests._reuse_eligible(["-p", "no:randomly", "--", "-test_x.py"], root=tmp_path) is True


@pytest.mark.parametrize("selection", ["all", "affected", "descriptor"])
def test_verify_pytest_command_keeps_plain_assertions(selection: str) -> None:
    """Anti-vacuity: the verify step clears configured addopts; without
    ``--assert=plain`` in the shared closed-world args the corpus run rewrites
    assertions and retains their ASTs.
    """
    from devtools import verify

    cmd = verify._pytest_command(selection=selection, worker_args=(), hypothesis_profile=None, explicit_tests=())
    assert CLEAR_CONFIGURED_ADDOPTS in cmd
    assert "--assert=plain" in cmd
