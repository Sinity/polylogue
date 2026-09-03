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
            "CREATE TABLE sessions(session_id TEXT, origin TEXT, raw_id TEXT, sort_key_ms INTEGER); CREATE TABLE blocks(block_id TEXT PRIMARY KEY, session_id TEXT, text TEXT);"
        )
        db.execute("INSERT INTO sessions VALUES ('codex:a', 'codex', 'raw-a', 1)")
        db.execute("INSERT INTO blocks VALUES ('block-0', 'codex:a', ?)", (text,))


def test_selector_is_deterministic_and_records_the_sealed_cohort(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    _index(path, "stable")
    seal = "a" * 64
    first = select_canary_sessions(path, sessions_per_origin=1, source_manifest_digest=seal)
    second = select_canary_sessions(path, sessions_per_origin=1, source_manifest_digest=seal)
    assert first.to_dict() == second.to_dict()
    assert first.source_manifest_digest == seal
    empty = tmp_path / "empty.db"
    with sqlite3.connect(empty) as db:
        db.execute("CREATE TABLE sessions(session_id TEXT, origin TEXT, raw_id TEXT, sort_key_ms INTEGER)")
    with pytest.raises(CanarySelectionError, match="empty"):
        select_canary_sessions(empty, source_manifest_digest=seal)


def test_selector_requires_a_sealed_source_manifest(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    _index(path, "stable")
    with pytest.raises(CanarySelectionError, match="sealed source manifest"):
        select_canary_sessions(path)


def test_selector_rejects_missing_origins_and_duplicate_raw_members(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    _index(path, "stable")
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO sessions VALUES ('codex:b', 'codex', 'raw-a', 2)")

    seal = "a" * 64
    with pytest.raises(CanarySelectionError, match="gemini"):
        select_canary_sessions(path, source_manifest_digest=seal, required_origins=("codex", "gemini"))
    with pytest.raises(CanarySelectionError, match="duplicate"):
        select_canary_sessions(path, source_manifest_digest=seal, sessions_per_origin=2)


def test_comparison_is_forensic_and_reports_real_row_mutation(tmp_path: Path) -> None:
    current, candidate = tmp_path / "current.db", tmp_path / "candidate.db"
    _index(current, "stable")
    _index(candidate, "changed")
    report = compare_reindex_generations(current, candidate, session_ids=("codex:a",))
    assert len(report.differences) == 1
    assert report.differences[0].operation is DifferenceOperation.CHANGED
    differences = report.to_dict()["differences"]
    assert isinstance(differences, list)
    assert "classification" not in differences[0]


def test_comparison_retains_each_child_row_for_one_session(tmp_path: Path) -> None:
    current, candidate = tmp_path / "current.db", tmp_path / "candidate.db"
    _index(current, "stable")
    _index(candidate, "stable")
    with sqlite3.connect(current) as db:
        db.execute("INSERT INTO blocks VALUES ('block-1', 'codex:a', 'first')")
    with sqlite3.connect(candidate) as db:
        db.execute("INSERT INTO blocks VALUES ('block-1', 'codex:a', 'changed')")
    report = compare_reindex_generations(current, candidate, session_ids=("codex:a",))
    assert [(item.table, item.operation) for item in report.differences] == [("blocks", DifferenceOperation.CHANGED)]
