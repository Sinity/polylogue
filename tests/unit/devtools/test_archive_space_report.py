"""Tests for ``polylogue ops diagnostics space``."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import cast

import pytest

from devtools.archive_space_report import build_space_report, main


def _seed_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY, blob_size INTEGER NOT NULL);
            CREATE TABLE sessions(session_id TEXT PRIMARY KEY, provider_meta TEXT);
            CREATE TABLE messages(message_id TEXT PRIMARY KEY, session_id TEXT, text TEXT);
            CREATE INDEX idx_messages_session ON messages(session_id);
            INSERT INTO raw_sessions VALUES ('raw-1', 100);
            INSERT INTO sessions VALUES ('c1', '{"source":"test"}');
            INSERT INTO messages VALUES ('m1', 'c1', 'hello');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_build_space_report_groups_dbstat_objects(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_db(db)

    report = build_space_report(db, limit=10, include_objects=True)

    assert report["ok"] is True
    assert report["dbstat_available"] is True
    assert cast(int, report["file_bytes"]) > 0
    categories = report["category_totals"]
    assert isinstance(categories, dict)
    assert "raw" in categories
    assert "archive" in categories
    assert "index" in categories
    objects = report["objects"]
    assert isinstance(objects, list)
    assert any(item["name"] == "raw_sessions" for item in objects)


def test_missing_database_reports_error(tmp_path: Path) -> None:
    report = build_space_report(tmp_path / "missing.db")

    assert report == {
        "ok": False,
        "report_version": 1,
        "db_path": str(tmp_path / "missing.db"),
        "error": "database_not_found",
    }


def test_build_space_report_skips_object_scan_by_default(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    _seed_db(db)

    report = build_space_report(db)

    assert report["ok"] is True
    assert report["object_scan_requested"] is False
    assert report["dbstat_error"] == "dbstat_skipped"
    assert report["objects"] == []


def test_main_json_returns_success_for_existing_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "index.db"
    _seed_db(db)

    assert main(["--db", str(db), "--json", "--objects", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert '"ok": true' in out
    assert '"object_count"' in out
