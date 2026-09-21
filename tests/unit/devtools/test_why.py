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

from devtools.why import _EXPLANATIONS, _history_projection, _latest_run, _render


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

    _render({"tier": "all", "status": "failed", "diagnosis": "some_unmapped_token"}, stream)

    output = stream.getvalue()
    assert "some_unmapped_token" in output
    assert "no recorded explanation" in output
    assert "do    :" not in output, "an unmapped diagnosis must not be given a remedy"


def test_failing_steps_are_surfaced() -> None:
    stream = io.StringIO()

    _render(
        {
            "tier": "all",
            "status": "failed",
            "exit_code": 1,
            "steps": [
                {"step_id": "01-ok", "exit": 0},
                {"step_id": "11-pytest-native-parallel", "exit": 1, "diagnosis": "pytest_failed"},
            ],
        },
        stream,
    )

    output = stream.getvalue()
    assert "11-pytest-native-parallel" in output
    assert "01-ok" not in output, "passing steps are noise in a failure explanation"


@pytest.mark.parametrize(
    ("diagnosis", "remedy"),
    [
        ("pytest_report_incomplete", "Re-run the same selection"),
        ("pytest_collection_only", "Re-run without --collect-only"),
        ("pytest_no_tests_selected", "Use a selector"),
        ("gate_missing_executable", "make it available on PATH"),
        ("gate_missing_input", "Restore the input"),
        ("gate_unreadable_input", "Make the input"),
        ("render_input_missing", "Restore the declared input"),
        ("render_input_unreadable", "Make the declared input readable"),
        ("render_input_invalid", "Fix the declared input"),
        ("render_surface_failed", "Fix the generated surface"),
        ("render_surface_invalid_result", "return an integer status"),
        ("render_surface_system_exit", "Read the recorded surface message"),
        ("gate_semantic_violation", "Read the detailed layering finding"),
        ("not_enforced", "Enable the gate's enforcement option"),
    ],
)
def test_new_typed_diagnoses_have_recorded_remedies(diagnosis: str, remedy: str) -> None:
    stream = io.StringIO()

    _render({"tier": "quick", "status": "failed", "diagnosis": diagnosis}, stream)

    assert remedy in stream.getvalue()


def test_import_mismatch_remedy_names_the_retained_contract() -> None:
    remedy = _EXPLANATIONS["checkout_import_mismatch"].remedy

    assert "imports polylogue" in remedy


@pytest.mark.uses_real_clock("builds relative history timestamps for the why report")
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
                    "tier": "all",
                    "duration_s": 2700.0,
                    "diagnosis": "pytest_failed",
                    "pytest_aggregate": {"selected_union_count": 0, "terminal_union_count": 20000},
                },
                {
                    "started_at": recent,
                    "tier": "focused-test",
                    "duration_s": 10.0,
                    "diagnosis": "pytest_passed",
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
    assert "selected nothing and ran the full corpus" in output
    assert "under-counted" in output, "the record omits killed runs and must say so"


def test_history_projection_carries_the_receipt_columns() -> None:
    projection = _history_projection(
        {
            "run_id": "run-1",
            "tier": "all",
            "status": "failed",
            "duration_s": 12.5,
            "testmon_selection": {"selection_mode": "affected", "graph_status": "usable"},
            "pytest_aggregate": {"selection_mode": "affected", "selected_union_count": 7},
        }
    )

    assert projection == {
        "run_id": "run-1",
        "started_at": None,
        "tier": "all",
        "graph_status": "usable",
        "selection_mode": "affected",
        "selected_count": 7,
        "wall_time_s": 12.5,
        "outcome": "failed",
        "diagnosis": None,
    }


def test_why_names_the_tree_a_receipt_covers() -> None:
    """polylogue-p2mbi: a result detached from its tree is not readable evidence.

    Anti-vacuity: drop the line and a reader of `devtools why` gets a verdict
    with no way to tell which commit it is about -- which is how 53 commits of
    drift went unnoticed on the 03:00 corpus run.
    """
    stream = io.StringIO()
    _render({"tier": "all", "status": "success", "git_head": "a" * 40, "final_git_head": "a" * 40}, stream)
    assert f"tested tree: {'a' * 40}" in stream.getvalue()


def test_why_says_so_when_a_receipt_records_no_tree() -> None:
    """Anti-vacuity: printing nothing here makes an unattributable receipt look ordinary."""
    stream = io.StringIO()
    _render({"tier": "all", "status": "success"}, stream)
    assert "tested tree: not recorded" in stream.getvalue()


def test_why_shows_a_tree_that_moved_under_the_run() -> None:
    """A merge landing mid-run is visible, not averaged into one SHA.

    Anti-vacuity: print only `git_head` and a run that straddled a merge reads
    as a run on one tree.
    """
    stream = io.StringIO()
    _render({"tier": "all", "status": "success", "git_head": "a" * 40, "final_git_head": "b" * 40}, stream)
    output = stream.getvalue()
    assert f"{'a' * 40} -> {'b' * 40}" in output


def test_why_marks_a_dirty_tested_tree() -> None:
    """Anti-vacuity: hide the flag and a receipt from a modified tree reads as a commit."""
    stream = io.StringIO()
    _render({"tier": "all", "status": "success", "git_head": "a" * 40, "git_dirty": True}, stream)
    assert "(dirty)" in stream.getvalue()


def test_why_names_the_killer_instead_of_sending_the_reader_to_the_journal() -> None:
    """polylogue-yk0zz: three killers, one previously undifferentiated report.

    Anti-vacuity: drop the `ended:` line and an oom-killed corpus run renders
    identically to a cancelled one -- which is what made identifying the
    2026-09-16..20 killers a multi-day manual investigation.
    """
    stream = io.StringIO()
    _render(
        {
            "tier": "all",
            "status": "failed",
            "exit_code": 137,
            "diagnosis": "verification_abandoned",
            "termination_reason": "failed",
            "termination_killer": "oom-kill",
            "termination_unit": "agentctl-pytest-heavy-polylogue-verify_all-b0ccb32f.service",
        },
        stream,
    )
    output = stream.getvalue()
    assert "ended: failed; systemd recorded oom-kill" in output
    assert "agentctl-pytest-heavy-polylogue-verify_all-b0ccb32f.service" in output
    # The remedy no longer sends the reader to look up what the receipt holds.
    assert "journalctl" not in output


def test_why_does_not_invent_a_killer_for_a_clean_cancel() -> None:
    """Anti-vacuity: always printing a killer would attribute a cancel to systemd."""
    stream = io.StringIO()
    _render({"tier": "all", "status": "failed", "termination_reason": "cancelled"}, stream)
    output = stream.getvalue()
    assert "ended: cancelled" in output
    assert "systemd recorded" not in output


def test_why_says_nothing_about_an_ending_nobody_recorded() -> None:
    """A run that closed itself has no external ending to report.

    Anti-vacuity: emit an empty `ended:` line and every ordinary receipt gains
    a blank field that reads as missing evidence.
    """
    stream = io.StringIO()
    _render({"tier": "quick", "status": "success", "exit_code": 0}, stream)
    assert "ended:" not in stream.getvalue()
