from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot


def test_raw_materialization_snapshot_ignores_skipped_raw_rows(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(raw_id, origin, validation_status, parse_error, parsed_at_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("raw-materializable", "chatgpt-export", "valid", None, 123),
                ("raw-skipped", "aistudio-drive", "skipped", None, None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, raw_id TEXT)")

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["available"] is True
    assert snapshot["raw_artifact_count"] == 1
    assert snapshot["materialized_raw_artifact_count"] == 0
    assert snapshot["archive_session_count"] == 0
    assert snapshot["join_gap_count"] == 1
    assert snapshot["total"] == 1
    assert snapshot["unchecked"] == 1
    assert snapshot["affected_unchecked"] == 1
    assert snapshot["category_counts"] == {
        "raw_id_join_gap": 1,
        "skipped": 0,
        "parse_failed": 0,
        "parsed_without_index_session": 1,
    }
    assert snapshot["source_family_counts"] == {"chatgpt-export": 1}


def test_raw_materialization_snapshot_counts_raw_artifacts_once(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(raw_id, origin, validation_status, parse_error, parsed_at_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("raw-shared", "claude-code-session", "valid", None, 123),
                ("raw-gap", "codex-session", "valid", None, 124),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, raw_id TEXT)")
        conn.executemany(
            "INSERT INTO sessions(session_id, raw_id) VALUES (?, ?)",
            [
                ("session-one", "raw-shared"),
                ("session-two", "raw-shared"),
            ],
        )

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["raw_artifact_count"] == 2
    assert snapshot["materialized_raw_artifact_count"] == 1
    assert snapshot["archive_session_count"] == 2
    assert snapshot["join_gap_count"] == 1
    assert snapshot["total"] == 1
    assert snapshot["source_family_counts"] == {"codex-session": 1}
