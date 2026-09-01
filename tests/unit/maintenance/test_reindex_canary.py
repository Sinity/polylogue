"""Source-authoritative canary contracts."""

import sqlite3
from pathlib import Path

import pytest

from polylogue.maintenance.reindex_canary import (
    CanarySelectionError,
    DifferenceOperation,
    compare_reindex_generations,
    select_canary_sessions,
)


def _index(path: Path, text: str) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE sessions(session_id TEXT, origin TEXT, raw_id TEXT, sort_key_ms INTEGER); CREATE TABLE blocks(session_id TEXT, text TEXT);"
        )
        db.execute("INSERT INTO sessions VALUES ('codex:a', 'codex', 'raw-a', 1)")
        db.execute("INSERT INTO blocks VALUES ('codex:a', ?)", (text,))


def test_selector_is_deterministic_and_requires_a_sealed_cohort(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    _index(path, "stable")
    first = select_canary_sessions(path, sessions_per_origin=1)
    second = select_canary_sessions(path, sessions_per_origin=1)
    assert first.to_dict() == second.to_dict()
    empty = tmp_path / "empty.db"
    with sqlite3.connect(empty) as db:
        db.execute("CREATE TABLE sessions(session_id TEXT, origin TEXT, raw_id TEXT, sort_key_ms INTEGER)")
    with pytest.raises(CanarySelectionError, match="empty"):
        select_canary_sessions(empty)


def test_comparison_is_forensic_and_reports_real_row_mutation(tmp_path: Path) -> None:
    current, candidate = tmp_path / "current.db", tmp_path / "candidate.db"
    _index(current, "stable")
    _index(candidate, "changed")
    report = compare_reindex_generations(current, candidate, session_ids=("codex:a",))
    assert len(report.differences) == 1
    assert report.differences[0].operation is DifferenceOperation.CHANGED
    assert "classification" not in report.to_dict()["differences"][0]
