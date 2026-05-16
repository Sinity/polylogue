from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from devtools.verify import (
    _format_completion_notification,
    _is_testmon_global_invalidator,
    _parse_pytest_test_count,
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
        "verify-file-budgets",
        "verify-test-ownership",
        "verify-schema-roundtrip",
        "verify-cross-cuts",
        "verify-suppressions",
        "verify-manifests",
        "verify-ci-workflows",
        "verify-witness-lifecycle",
        "verify-lane-assertions",
        "verification-impact check",
    ]


def test_default_verify_uses_pytest_testmon() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False)

    label, command = steps[-1]
    assert label == "pytest testmon"
    assert "--testmon" in command
    assert "--testmon-noselect" not in command
    assert "-n" in command
    assert "0" in command


def test_seed_testmon_runs_full_collection_without_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon"
    assert "--testmon" in command
    assert "--testmon-noselect" in command
    assert "-n" in command
    assert "8" in command


def test_full_verify_includes_full_pytest_without_testmon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)
    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, full_pytest=True)

    label, command = steps[-1]
    assert label == "pytest full"
    assert "--testmon" not in command
    assert "-n" in command
    assert "8" in command


def test_seed_testmon_worker_count_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "4")

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, seed_testmon=True)

    label, command = steps[-1]
    assert label == "pytest seed-testmon"
    assert command[command.index("-n") + 1] == "4"


def test_global_testmon_invalidator_runs_full_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PYTEST_WORKERS", raising=False)

    steps = build_verify_steps(quick=False, lab=False, skip_slow=False, testmon_global=True)

    label, command = steps[-1]
    assert label == "pytest testmon-global"
    assert "--testmon" in command
    assert "--testmon-noselect" in command
    assert command[command.index("-n") + 1] == "8"


def test_skip_slow_keeps_testmon_selection_forced() -> None:
    steps = build_verify_steps(quick=False, lab=False, skip_slow=True)

    label, command = steps[-1]
    assert label == "pytest testmon"
    assert command[command.index("-m") + 1] == "not slow"
    assert "--testmon-forceselect" in command


def test_global_testmon_invalidators_cover_harness_and_config() -> None:
    assert _is_testmon_global_invalidator("pyproject.toml")
    assert _is_testmon_global_invalidator("uv.lock")
    assert _is_testmon_global_invalidator("tests/conftest.py")
    assert _is_testmon_global_invalidator("tests/infra/storage_records.py")
    assert _is_testmon_global_invalidator("tests/unit/pytest.ini")
    assert not _is_testmon_global_invalidator("polylogue/cli/click_app.py")
    assert not _is_testmon_global_invalidator("docs/execution-plan.md")


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


def test_run_records_pytest_count_metadata() -> None:
    completed = subprocess.CompletedProcess(
        args=["pytest"],
        returncode=0,
        stdout="....\n4 passed in 1.23s\n",
        stderr="",
    )

    with patch("devtools.verify.subprocess.run", return_value=completed):
        rc, _elapsed, metadata = _run("pytest affected", ["pytest"])

    assert rc == 0
    assert metadata == {"count": 4}


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
