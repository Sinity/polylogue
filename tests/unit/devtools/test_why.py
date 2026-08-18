"""Contracts for `devtools why`.

The property under test is that the command RENDERS what a receipt recorded and
never invents advice: an unrecognised diagnosis must be reported verbatim rather
than mapped to a plausible-sounding remedy, because a confident wrong remedy
sends the reader further from the cause than silence does.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from devtools.why import _EXPLANATIONS, _latest_run, _render


def _write_run(root: Path, run_id: str, payload: dict[str, object]) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    path = run_dir / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_latest_run_is_the_most_recently_written(tmp_path: Path) -> None:
    older = _write_run(tmp_path, "20260101T000000Z-a", {"status": "success"})
    newer = _write_run(tmp_path, "20260102T000000Z-b", {"status": "failed"})
    import os

    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    assert _latest_run(tmp_path) == newer


def test_unknown_diagnosis_is_reported_verbatim_without_invented_advice() -> None:
    stream = io.StringIO()

    _render({"tier": "testmon", "status": "failed", "diagnosis": "some_unmapped_token"}, stream)

    output = stream.getvalue()
    assert "some_unmapped_token" in output
    assert "no recorded explanation" in output
    assert "do    :" not in output, "an unmapped diagnosis must not be given a remedy"


def test_bootstrap_explanation_points_at_the_field_that_distinguishes_the_two_causes() -> None:
    stream = io.StringIO()

    _render(
        {
            "tier": "testmon",
            "status": "success",
            "diagnosis": "native_testmon_graph_invalid",
            "testmon_selection": {
                "selection_mode": "bootstrap",
                "state_status": "incomplete",
                "state_reason": "changed executable modules are absent from the native dependency graph",
                "missing_executable_paths": ["polylogue/daemon/http.py"],
                "runtime_data_paths": [],
            },
        },
        stream,
    )

    output = stream.getvalue()
    assert "state_status" in output, "the remedy must name the field that separates absent from incomplete"
    assert "incomplete" in output
    assert "polylogue/daemon/http.py" in output, "the uncovered files are the actionable part"


def test_failing_steps_and_non_green_tests_are_surfaced() -> None:
    stream = io.StringIO()

    _render(
        {
            "tier": "testmon",
            "status": "failed",
            "exit_code": 1,
            "pytest_aggregate": {"non_green_count": 2, "non_green_sample": ["tests/unit/a.py::test_x"]},
            "steps": [
                {"step_id": "01-ok", "exit": 0},
                {"step_id": "11-pytest-native-parallel", "exit": 1, "diagnosis": "pytest_failed"},
            ],
        },
        stream,
    )

    output = stream.getvalue()
    assert "tests/unit/a.py::test_x" in output
    assert "11-pytest-native-parallel" in output
    assert "01-ok" not in output, "passing steps are noise in a failure explanation"


def test_import_mismatch_remedy_warns_that_the_obvious_probe_lies() -> None:
    """Regression guard for a real trap: `python -c 'import polylogue'` run from
    inside a worktree reports that worktree because cwd leads sys.path, so it
    "confirms" a correct checkout while the active environment belongs to
    another one."""
    remedy = _EXPLANATIONS["checkout_import_mismatch"].remedy

    assert "cwd" in remedy
    assert "VIRTUAL_ENV" in remedy


def test_history_mode_reports_where_the_time_went(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The question "where did the last N hours go" kept requiring an ad hoc
    DuckDB query against a substrate that materialises on its own cadence and
    was 17 hours stale when it mattered. The history file is that data at its
    source, current by construction, and covers every checkout and worktree."""
    from datetime import UTC, datetime, timedelta

    from devtools import why

    recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    stale = (datetime.now(UTC) - timedelta(hours=100)).isoformat()
    history = tmp_path / "verify-history.jsonl"
    history.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                {
                    "started_at": recent,
                    "tier": "testmon",
                    "duration_s": 2700.0,
                    "diagnosis": "native_testmon_graph_invalid",
                    "checkout_root": "/realm/project/polylogue",
                    "pytest_aggregate": {"selected_union_count": 0, "terminal_union_count": 20000},
                },
                {
                    "started_at": recent,
                    "tier": "focused-test",
                    "duration_s": 10.0,
                    "diagnosis": "pytest_passed",
                    "checkout_root": "/realm/worktrees/lane-a",
                },
                {"started_at": stale, "tier": "quick", "duration_s": 9999.0, "diagnosis": "pytest_passed"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(why, "VERIFY_HISTORY_PATH", history)
    stream = io.StringIO()

    assert why._render_history(24.0, stream) == 0

    output = stream.getvalue()
    assert "2 run(s)" in output, "the 100-hour-old run is outside the window"
    assert "9999" not in output
    assert "lane-a" in output, "lanes must be visible; their receipts die with the worktree"
    assert "selected nothing and ran the full corpus" in output
    assert "under-counted" in output, "the record omits killed runs and must say so"
