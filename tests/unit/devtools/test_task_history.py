"""Tests for ``devtools workspace tasks`` task-history surface and harness wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import devtools.__main__ as devtools_main
from devtools import task_history


@pytest.fixture(autouse=True)
def isolated_task_history_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect task-history storage into ``tmp_path`` and clear opt-out env."""
    path = tmp_path / "tasks.jsonl"
    monkeypatch.setenv("POLYLOGUE_TASK_HISTORY_FILE", str(path))
    monkeypatch.delenv("POLYLOGUE_TASK_HISTORY_DISABLE", raising=False)
    return path


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("verify", "verify"),
        ("verify topology", "verify"),
        ("render all", "render"),
        ("render cli-reference", "render"),
        ("lab smoke", "lab"),
        ("bench mutation", "campaign"),
        ("bench campaign", "campaign"),
        ("bench synthetic", "campaign"),
        ("status", "query"),
        ("workspace tasks", "query"),
        ("workspace failure-context", "query"),
        ("workspace worktree-gc", "query"),
        ("release build-package", "render"),
        ("totally-unknown-command", "other"),
        ("", "other"),
    ],
)
def test_classify_command_buckets(command: str, expected: str) -> None:
    assert task_history.classify_command(command) == expected


# ---------------------------------------------------------------------------
# log / recent / stats round-trip
# ---------------------------------------------------------------------------


def test_log_and_recent_round_trip(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        task_history.main(
            [
                "log",
                "--command",
                "verify",
                "--duration-ms",
                "1500",
                "--exit-code",
                "0",
                "--cwd",
                "/repo",
            ]
        )
        == 0
    )
    assert task_history.main(["recent", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    entry = payload[0]
    assert entry["command"] == "verify"
    assert entry["class"] == "verify"
    assert entry["duration_ms"] == 1500
    assert entry["exit_code"] == 0


def test_stats_by_class_and_slowest(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    samples = [
        ("verify", 1000.0, 0),
        ("verify", 2000.0, 0),
        ("verify", 9000.0, 1),
        ("render all", 200.0, 0),
        ("status", 50.0, 0),
    ]
    for cmd, dur, code in samples:
        task_history.record_invocation(
            command=cmd,
            args=[],
            duration_ms=dur,
            exit_code=code,
        )

    assert task_history.main(["stats", "--by-class", "--slowest", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 5
    assert "by_class" in payload
    verify_dist = payload["by_class"]["verify"]
    assert verify_dist["count"] == 3
    assert verify_dist["max_ms"] == 9000.0
    # median of [1000, 2000, 9000] is 2000
    assert verify_dist["median_ms"] == 2000.0
    # slowest contains the two slowest, ordered desc
    assert [t["duration_ms"] for t in payload["slowest"]] == [9000.0, 2000.0]


def test_record_invocation_enriches_verify_run_metadata(
    isolated_task_history_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    current = tmp_path / ".cache" / "verify" / "current-run.json"
    current.parent.mkdir(parents=True)
    current.write_text(
        json.dumps(
            {
                "run_id": "run-abc",
                "artifact_dir": ".cache/verify/runs/run-abc",
                "status": "failed",
                "diagnosis": "report_missing_after_sessionfinish_success",
                "steps": [
                    {
                        "name": "pytest seed-testmon",
                        "diagnosis": "report_missing_after_sessionfinish_success",
                        "selected_count": 11295,
                        "peak_tree_rss_mb": 8192.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("devtools.task_history._get_root", lambda: tmp_path)

    task_history.record_invocation(command="verify", args=["--seed-testmon"], duration_ms=100.0, exit_code=-15)
    assert task_history.main(["recent", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    entry = payload[0]
    assert entry["verify_run_id"] == "run-abc"
    assert entry["verify_diagnosis"] == "report_missing_after_sessionfinish_success"
    assert entry["pytest_peak_tree_rss_mb"] == 8192.0


def test_stats_resources_reports_peak_distribution(
    isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task_history.record_invocation(command="verify", args=[], duration_ms=100.0, exit_code=0)
    records = [json.loads(line) for line in isolated_task_history_file.read_text().splitlines()]
    records[0]["pytest_peak_tree_rss_mb"] = 100.0
    isolated_task_history_file.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    assert task_history.main(["stats", "--resources", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["resources"]["count"] == 1
    assert payload["resources"]["peak_rss_mb_max"] == 100.0


def test_stats_slow_tests_reads_latest_pytest_report(
    isolated_task_history_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    verify_cache = tmp_path / ".cache" / "verify"
    verify_cache.mkdir(parents=True)
    (verify_cache / "last-pytest.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "tests/test_fast.py::test_fast",
                        "outcome": "passed",
                        "setup": {"duration": 0.1},
                        "call": {"duration": 0.2},
                        "teardown": {"duration": 0.1},
                    },
                    {
                        "nodeid": "tests/test_slow.py::test_slow",
                        "outcome": "passed",
                        "setup": {"duration": 1.0},
                        "call": {"duration": 2.0},
                        "teardown": {"duration": 0.5},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (verify_cache / "last-pytest-isolated.json").write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": "tests/test_isolated.py::test_isolated",
                        "outcome": "passed",
                        "setup": {"duration": 0.5},
                        "call": {"duration": 4.0},
                        "teardown": {"duration": 0.25},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("devtools.task_history._get_root", lambda: tmp_path)
    task_history.record_invocation(command="verify", args=[], duration_ms=100.0, exit_code=0)

    assert task_history.main(["stats", "--slow-tests", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["slow_tests"] == [
        {
            "call_s": 4.0,
            "nodeid": "tests/test_isolated.py::test_isolated",
            "outcome": "passed",
            "report": "last-pytest-isolated.json",
            "setup_s": 0.5,
            "teardown_s": 0.25,
            "total_s": 4.75,
        }
    ]


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_budget_passes_when_p95_within(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for dur in (100.0, 200.0, 300.0, 400.0, 500.0):
        task_history.record_invocation(command="verify", args=[], duration_ms=dur, exit_code=0)
    code = task_history.main(["budget", "--class", "verify", "--max-ms", "1000", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["within_budget"] is True
    assert payload["status"] == "ok"


def test_budget_fails_when_p95_exceeds(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for dur in (100.0, 200.0, 300.0, 400.0, 50000.0):
        task_history.record_invocation(command="verify", args=[], duration_ms=dur, exit_code=0)
    code = task_history.main(["budget", "--class", "verify", "--max-ms", "1000", "--json"])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["within_budget"] is False
    assert payload["status"] == "over-budget"


def test_budget_no_data_default_fails(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = task_history.main(["budget", "--class", "verify", "--max-ms", "1000", "--json"])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no-data"


def test_budget_no_data_allow_empty(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = task_history.main(["budget", "--class", "verify", "--max-ms", "1000", "--allow-empty", "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no-data"


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def test_prune_keeps_latest_n(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for i in range(10):
        task_history.record_invocation(command=f"render-{i}", args=[], duration_ms=float(i), exit_code=0)
    assert task_history.main(["prune", "--keep", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["before"] == 10
    assert payload["after"] == 3
    assert payload["removed"] == 7

    # Verify the kept records are the latest three
    lines = isolated_task_history_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    kept = [json.loads(line) for line in lines]
    assert [t["command"] for t in kept] == ["render-7", "render-8", "render-9"]


def test_prune_noop_when_below_keep(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task_history.record_invocation(command="verify", args=[], duration_ms=1.0, exit_code=0)
    assert task_history.main(["prune", "--keep", "10", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed"] == 0
    assert payload["after"] == 1


def test_prune_zero_keeps_nothing(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for i in range(3):
        task_history.record_invocation(command=f"x-{i}", args=[], duration_ms=1.0, exit_code=0)
    assert task_history.main(["prune", "--keep", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["after"] == 0
    assert isolated_task_history_file.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_dry_run_round_trips_argv(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task_history.record_invocation(
        command="render all",
        args=["--check", "--verbose"],
        duration_ms=42.0,
        exit_code=0,
        cwd="/tmp/repo",
    )
    assert task_history.main(["replay", "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "render all"
    assert payload["args"] == ["--check", "--verbose"]
    assert payload["argv"] == ["render all", "--check", "--verbose"]
    assert payload["index"] == 1


def test_replay_dry_run_with_explicit_index(
    isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    task_history.record_invocation(command="status", args=[], duration_ms=10.0, exit_code=0)
    task_history.record_invocation(command="verify", args=["--quick"], duration_ms=20.0, exit_code=0)
    # index 2 = the older one (status)
    assert task_history.main(["replay", "2", "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "status"


def test_replay_errors_on_empty(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert task_history.main(["replay"]) == 1
    assert "no tasks recorded" in capsys.readouterr().err


def test_replay_errors_on_out_of_range(isolated_task_history_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    task_history.record_invocation(command="status", args=[], duration_ms=1.0, exit_code=0)
    assert task_history.main(["replay", "5"]) == 1
    assert "exceeds" in capsys.readouterr().err


def test_replay_executes_via_subprocess(isolated_task_history_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay should shell out to ``python -m devtools <argv>``."""
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(cmd, 0)

    task_history.record_invocation(
        command="status",
        args=["--json"],
        duration_ms=1.0,
        exit_code=0,
        cwd=str(isolated_task_history_file.parent),
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert task_history.main(["replay"]) == 0
    assert captured["cmd"] == [sys.executable, "-m", "devtools", "status", "--json"]
    assert captured["cwd"] == str(isolated_task_history_file.parent)


# ---------------------------------------------------------------------------
# Harness auto-logging
# ---------------------------------------------------------------------------


def test_devtools_main_auto_logs_invocation(isolated_task_history_file: Path) -> None:
    """A normal ``devtools`` call appends a record to the JSONL log."""
    rc = devtools_main.main(["--list-commands", "--json"])
    assert rc == 0
    # Root-level option only — by design we skip auto-log for bare options.
    assert not isolated_task_history_file.exists() or isolated_task_history_file.read_text() == ""

    rc = devtools_main.main(["status", "--json"])
    # status command may return non-zero in tmp, but we only care that we logged.
    records = [
        json.loads(line) for line in isolated_task_history_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(records) == 1
    entry = records[0]
    assert entry["command"] == "status"
    assert entry["class"] == "query"
    assert "duration_ms" in entry
    assert "exit_code" in entry
    assert "timestamp" in entry
    assert entry["exit_code"] == rc


def test_devtools_main_respects_task_history_disable(
    isolated_task_history_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("POLYLOGUE_TASK_HISTORY_DISABLE", "1")
    devtools_main.main(["status", "--json"])
    assert not isolated_task_history_file.exists() or isolated_task_history_file.read_text() == ""


def test_record_invocation_swallows_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Auto-log must never raise even if the file path is unwritable."""
    bad = tmp_path / "ro_dir" / "nope.jsonl"
    monkeypatch.setenv("POLYLOGUE_TASK_HISTORY_FILE", str(bad))
    # Force _ensure_file to fail by making the parent a regular file.
    bad.parent.parent.mkdir(parents=True, exist_ok=True)
    bad.parent.write_text("not a dir", encoding="utf-8")
    # Should not raise:
    task_history.record_invocation(command="verify", args=[], duration_ms=1.0, exit_code=0)
