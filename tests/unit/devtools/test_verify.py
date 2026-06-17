from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from devtools.verify import (
    PYTEST_REPORT_PATH,
    ROOT,
    _format_completion_notification,
    _parse_pytest_test_count,
    _pytest_command_metadata,
    _pytest_metadata_from_report,
    _pytest_stall_timeout_s,
    _pytest_timeout_s,
    _read_pytest_report,
    _run,
    _stop_after_failed_step,
    _testmon_preflight,
    build_verify_steps,
    main,
)


def test_quick_verify_omits_pytest() -> None:
    steps = build_verify_steps(quick=True, lab=False, skip_slow=False)

    labels = [label for label, _command in steps]
    assert labels == [
        "ruff format",
        "ruff check",
        "mypy",
        "render-all",
        "verify-topology",
        "verify-layering",
        "verify-closure-matrix",
        "verify-schema-roundtrip",
        "verify-manifests",
        "verify-ci-workflows",
        "verify-doc-commands",
        "verify-lane-assertions",
        "verify-test-infra-currency",
        "verify-test-clock-hygiene",
        "public-surface-audit",
    ]


def test_default_verify_uses_pytest_testmon() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    assert "--testmon" in command
    assert "--testmon-noselect" not in command
    assert "--testmon-forceselect" in command
    assert "-n" in command
    assert command[command.index("-n") + 1] == "0"


def test_pytest_step_requests_structured_json_report() -> None:
    """Every pytest invocation must emit the report consumed by verify and dashboards (#1026)."""
    for kwargs in (
        {"seed_testmon": True},
        {"full_pytest": True},
        {},  # default testmon
    ):
        steps = build_verify_steps(quick=False, lab=False, skip_slow=False, **kwargs)
        pytest_steps = [(label, command) for label, command in steps if label.startswith("pytest")]
        assert pytest_steps, kwargs
        # Every pytest lane emits a structured JSON report.
        for label, command in pytest_steps:
            assert "--json-report" in command, f"{label}: {command}"
            assert any(arg.startswith("--json-report-file=") for arg in command), label
        # The canonical report path consumed by verify/dashboards is emitted by
        # the primary lane; the #1775 isolated lane writes its own file.
        expected_target = f"--json-report-file={PYTEST_REPORT_PATH}"
        assert any(expected_target in command for _label, command in pytest_steps), kwargs


def test_seed_testmon_runs_full_collection_without_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon"
    assert "--testmon" in command
    assert "--testmon-noselect" in command
    assert "-n" in command
    assert command[command.index("-n") + 1] == "16"


def test_full_verify_includes_full_pytest_without_testmon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, full_pytest=True)

    # #1775: the full diagnostic runs as two lanes — a parallel bulk lane plus a
    # single-process isolated lane for load-sensitive/tui tests. Neither uses
    # testmon; the bulk lane keeps xdist parallelism, the isolated lane forces -n 0.
    labels = [label for label, _command in steps]
    assert labels[-2:] == ["pytest full (parallel)", "pytest load-sensitive (isolated)"]

    bulk_label, bulk_command = steps[-2]
    assert bulk_label == "pytest full (parallel)"
    assert "--testmon" not in bulk_command
    assert "-n" in bulk_command
    assert bulk_command[bulk_command.index("-n") + 1] == "16"

    isolated_label, isolated_command = steps[-1]
    assert isolated_label == "pytest load-sensitive (isolated)"
    assert "--testmon" not in isolated_command
    assert isolated_command[isolated_command.index("-n") + 1] == "0"


def test_seed_testmon_worker_count_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "4")

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon"
    assert command[command.index("-n") + 1] == "4"


def test_default_testmon_worker_count_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "3")

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    assert command[command.index("-n") + 1] == "3"


def test_marker_filters_keep_testmon_selection_forced() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    marker_expr = command[command.index("-m") + 1]
    assert "not scale_medium" in marker_expr
    assert "not scale_large" in marker_expr
    assert "--testmon-forceselect" in command


def test_skip_slow_composes_with_forced_testmon_selection() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=True)

    label, command = steps[-1]
    assert label == "pytest testmon"
    # Scale-tier policy (#1183): the default verify gate filters out
    # ``scale_medium``/``scale_large``; ``--skip-slow`` composes with that
    # filter via ``and`` rather than replacing it.
    marker_expr = command[command.index("-m") + 1]
    assert "not slow" in marker_expr
    assert "not scale_medium" in marker_expr
    assert "not scale_large" in marker_expr
    assert "--testmon-forceselect" in command


def test_default_verify_excludes_medium_and_large_scale_markers() -> None:
    """Default verify pytest step deselects the medium/large scale tiers (#1183)."""
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    marker_expr = command[command.index("-m") + 1]
    assert "not scale_medium" in marker_expr
    assert "not scale_large" in marker_expr
    # ``scale_small`` is *not* excluded — it runs in the default gate.
    assert "scale_small" not in marker_expr


def test_lab_verify_includes_medium_scale_marker() -> None:
    """``--lab`` lets ``scale_medium`` into the pytest step but still gates ``scale_large`` (#1183)."""
    steps = build_verify_steps(quick=False, lab=True, skip_slow=False)

    pytest_step = next((label, command) for label, command in steps if label.startswith("pytest"))
    label, command = pytest_step
    marker_expr = command[command.index("-m") + 1]
    assert "not scale_large" in marker_expr
    assert "not scale_medium" not in marker_expr
    assert "scale_small" not in marker_expr


def test_lab_verify_delegates_to_lab_scenario() -> None:
    steps = build_verify_steps(quick=True, lab=True, skip_slow=False)

    labels = [label for label, _command in steps]
    assert "lab scenario" in labels
    assert "verify-slos" in labels
    lab_step = next(step for step in steps if step[0] == "lab scenario")
    assert lab_step == (
        "lab scenario",
        [sys.executable, "-m", "devtools", "lab-scenario", "run", "archive-smoke", "--tier", "0"],
    )


def test_testmon_preflight_requires_seed_when_database_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert "devtools verify --seed-testmon" in message


def test_testmon_preflight_requires_seed_stamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".testmondata").write_text("partial")

    message = _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False)

    assert message is not None
    assert ".cache/testmon/seed.json" in message


def test_testmon_preflight_accepts_seeded_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".testmondata").write_text("seeded")
    seed_stamp = tmp_path / ".cache" / "testmon" / "seed.json"
    seed_stamp.parent.mkdir(parents=True)
    seed_stamp.write_text("{}")

    assert _testmon_preflight(seed_testmon=False, full_pytest=False, quick=False, commit=False) is None


def test_testmon_preflight_allows_seed_and_full_without_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert _testmon_preflight(seed_testmon=True, full_pytest=False, quick=False, commit=False) is None
    assert _testmon_preflight(seed_testmon=False, full_pytest=True, quick=False, commit=False) is None


def test_parse_pytest_test_count_from_summary() -> None:
    output = "bringing up nodes...\n\n6 passed, 2 skipped, 1 xfailed in 8.49s\n"

    assert _parse_pytest_test_count(output) == 9


def test_parse_pytest_test_count_handles_no_tests() -> None:
    assert _parse_pytest_test_count("no tests ran in 0.02s\n") == 0


def test_run_records_pytest_count_metadata_from_terminal_fallback() -> None:
    """When the JSON report is missing, _run falls back to terminal scraping."""
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="....\n4 passed in 1.23s\n",
        stderr="",
    )

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=None),
    ):
        rc, _elapsed, metadata = _run("pytest affected", ["pytest"])

    assert rc == 0
    assert metadata["count"] == 4
    assert metadata["report_path"] is None
    assert metadata["pytest_workers"] == "unset"
    assert metadata["pytest_selection"] == "full"


def test_run_forces_subprocesses_to_current_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_ROOT", "/stale/main")
    monkeypatch.setenv("POLYLOGUE_REPO_ROOT", "/stale/main")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/stale/main/.cache/pycache")
    completed = subprocess.CompletedProcess(args=["devtools"], returncode=0, stdout="", stderr="")

    with patch("devtools.verify.subprocess.run", return_value=completed) as run:
        rc, _elapsed, _metadata = _run("render-all", ["devtools", "render-all", "--check"])

    assert rc == 0
    env = run.call_args.kwargs["env"]
    assert env["POLYLOGUE_ROOT"] == str(ROOT)
    assert env["POLYLOGUE_REPO_ROOT"] == str(ROOT)
    assert env["PYTHONPYCACHEPREFIX"] == str(ROOT / ".cache" / "pycache")


def test_run_reads_structured_pytest_report() -> None:
    """The structured pytest report is the primary metadata source (#1026, #998)."""
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="",
        stderr="",
    )
    report = {
        "summary": {"passed": 10, "failed": 1, "skipped": 2, "total": 13},
        "duration": 4.56,
    }

    with (
        patch("devtools.verify._run_pytest_with_heartbeat", return_value=completed),
        patch("devtools.verify._read_pytest_report", return_value=report),
    ):
        rc, _elapsed, metadata = _run("pytest testmon", ["pytest", "--testmon", "-n", "8"])

    assert rc == 0
    assert metadata["count"] == 13  # passed+failed+skipped
    assert metadata["passed"] == 10
    assert metadata["failed"] == 1
    assert metadata["skipped"] == 2
    assert metadata["total"] == 13
    assert metadata["pytest_duration_s"] == 4.56
    assert metadata["report_path"] == str(PYTEST_REPORT_PATH)
    assert metadata["pytest_workers"] == "8"
    assert metadata["pytest_selection"] == "testmon"


def test_pytest_run_emits_heartbeat_for_long_silent_child(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.1")

    rc, _elapsed, metadata = _run("pytest heartbeat", [sys.executable, "-c", "import time; time.sleep(0.25)"])

    captured = capsys.readouterr()
    assert rc == 0
    assert metadata["heartbeat_s"] == 0.1
    assert "command:" in captured.err
    assert "still running: pid=" in captured.err
    assert "elapsed=" in captured.err


def test_pytest_run_terminates_after_runtime_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "0.15")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0")

    rc, _elapsed, metadata = _run("pytest timeout", [sys.executable, "-c", "import time; time.sleep(5)"])

    captured = capsys.readouterr()
    assert rc == 124
    assert metadata["timeout_s"] == 0.15
    assert metadata["stall_timeout_s"] == 0.0
    assert "pytest runtime exceeded 0.15s" in captured.err
    assert "terminated pytest process group" in captured.err


def test_pytest_run_terminates_after_output_stall(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POLYLOGUE_VERIFY_HEARTBEAT_S", "0.05")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "0")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "0.15")

    rc, _elapsed, metadata = _run(
        "pytest stall",
        [sys.executable, "-c", "import time; print('progress', flush=True); time.sleep(5)"],
    )

    captured = capsys.readouterr()
    assert rc == 124
    assert metadata["timeout_s"] == 0.0
    assert metadata["stall_timeout_s"] == 0.15
    assert "progress" in captured.err
    assert "pytest produced no output for 0.15s" in captured.err
    assert "terminated pytest process group" in captured.err


def test_pytest_timeout_env_defaults_and_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", raising=False)
    monkeypatch.delenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", raising=False)
    assert _pytest_timeout_s() > 0
    assert _pytest_stall_timeout_s() > 0

    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S", "-1")
    monkeypatch.setenv("POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S", "nope")
    assert _pytest_timeout_s() == 0.0
    assert _pytest_stall_timeout_s() > 0


def test_pytest_command_metadata_reports_worker_and_selection_policy() -> None:
    assert _pytest_command_metadata(["pytest", "--testmon", "-n", "8"]) == {
        "pytest_workers": "8",
        "pytest_selection": "testmon",
    }
    assert _pytest_command_metadata(["pytest", "--testmon", "--testmon-noselect", "-n", "16"]) == {
        "pytest_workers": "16",
        "pytest_selection": "testmon-noselect",
    }
    assert _pytest_command_metadata(["pytest", "-n", "16"]) == {
        "pytest_workers": "16",
        "pytest_selection": "full",
    }


def test_pytest_metadata_handles_empty_summary() -> None:
    """Robustness: a malformed/empty report still yields a metadata dict."""
    assert _pytest_metadata_from_report({}) == {"report_path": str(PYTEST_REPORT_PATH)}


def test_read_pytest_report_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert _read_pytest_report(tmp_path / "missing.json") is None


def test_read_pytest_report_returns_none_for_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert _read_pytest_report(bad) is None


def test_read_pytest_report_returns_none_for_non_object(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]")
    assert _read_pytest_report(bad) is None


def test_read_pytest_report_parses_valid_payload(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text('{"summary": {"passed": 3}, "duration": 1.0}')
    parsed = _read_pytest_report(path)
    assert parsed == {"summary": {"passed": 3}, "duration": 1.0}


def test_verify_continues_after_failed_cheap_step(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fake_run(label: str, command: list[str]) -> tuple[int, float, dict[str, object]]:
        calls.append(label)
        return 1, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
    ):
        rc = main(["--quick", "--json"])

    assert rc == 1
    assert calls == [label for label, _command in build_verify_steps(quick=True, lab=False, skip_slow=False)]
    payload = capsys.readouterr().out
    assert '"exit_code": 1' in payload


def test_verify_stops_after_failed_heavy_step(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def fake_run(label: str, command: list[str]) -> tuple[int, float, dict[str, object]]:
        calls.append(label)
        return (1 if label.startswith("pytest") else 0), 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify"),
        patch("devtools.verify._testmon_preflight", return_value=None),
    ):
        rc = main(["--json"])

    assert rc == 1
    assert calls[-1].startswith("pytest")
    payload = capsys.readouterr().out
    assert '"exit_code": 1' in payload


def test_failed_step_stop_policy_distinguishes_cheap_and_heavy_steps() -> None:
    assert _stop_after_failed_step("ruff check") is False
    assert _stop_after_failed_step("verify-layering") is False
    assert _stop_after_failed_step("pytest testmon") is True
    assert _stop_after_failed_step("lab scenario") is True
    assert _stop_after_failed_step("verify-slos") is True


def test_completion_notification_uses_pytest_count() -> None:
    summary = _format_completion_notification(
        exit_code=0,
        total_duration=118.2,
        step_results=[
            {"name": "ruff check", "duration_s": 0.1, "exit": 0},
            {"name": "pytest affected", "duration_s": 100.0, "exit": 0, "count": 12},
        ],
    )

    assert summary == "PASS (118s), 12 tests"


def test_completion_notification_omits_unknown_pytest_count() -> None:
    summary = _format_completion_notification(
        exit_code=0,
        total_duration=118.2,
        step_results=[{"name": "pytest", "duration_s": 100.0, "exit": 0}],
    )

    assert summary == "PASS (118s)"


def test_default_testmon_step_pairs_marker_filter_with_forceselect() -> None:
    """#1632: any pytest -m marker filter in the default lane MUST be paired with --testmon-forceselect.

    Without ``--testmon-forceselect``, a marker selector deactivates
    pytest-testmon's affected-test selection and the run silently
    expands to the whole suite — PR #1550 fixed exactly this regression
    after a full week of every default verify running 9.5K tests
    instead of the affected subset. This invariant is the regression
    guard so the footgun cannot re-land silently again.
    """
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)
    label, command = steps[-1]
    assert label == "pytest testmon"
    if "-m" in command:
        assert "--testmon-forceselect" in command, (
            f"marker filter without --testmon-forceselect re-introduces the #1550 silent-deselection footgun: {command}"
        )


def test_skip_slow_testmon_step_keeps_forceselect_with_compound_marker() -> None:
    """``--skip-slow`` composes the marker; the pairing invariant must still hold."""
    steps = build_verify_steps(quick=False, lab=False, skip_slow=True)
    label, command = steps[-1]
    assert label == "pytest testmon"
    assert "-m" in command
    assert "--testmon-forceselect" in command


def test_verify_does_not_notify_on_pass() -> None:
    """Passing verify runs stay silent — only failures send a desktop popup."""

    def fake_run(label: str, command: list[str]) -> tuple[int, float, dict[str, object]]:
        return 0, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify") as notify,
    ):
        rc = main(["--quick", "--json"])

    assert rc == 0
    notify.assert_not_called()


def test_verify_notifies_on_failure() -> None:
    """Failing verify runs still send a desktop popup so the operator notices."""

    def fake_run(label: str, command: list[str]) -> tuple[int, float, dict[str, object]]:
        return 1, 0.01, {}

    with (
        patch("devtools.verify._run", side_effect=fake_run),
        patch("devtools.verify._git_head", return_value="head"),
        patch("devtools.verify._save_history"),
        patch("devtools.verify._stamp_head"),
        patch("devtools.verify._notify") as notify,
    ):
        rc = main(["--quick", "--json"])

    assert rc == 1
    notify.assert_called_once()
