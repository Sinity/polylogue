"""The source train persists verified member identity on every raw row."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_MIGRATION = Path("polylogue/storage/sqlite/migrations/source/003_raw_member_identity.sql")


def test_source_train_backfills_raw_member_identity() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY, source_index INTEGER NOT NULL);
        CREATE TABLE raw_container_coordinates(
            raw_id TEXT PRIMARY KEY,
            addressing_mode TEXT,
            content_identity TEXT
        );
        INSERT INTO raw_sessions VALUES ('raw-1', 7);
        INSERT INTO raw_container_coordinates VALUES
            ('raw-1', 'element_of_container', 'a' || printf('%063d', 0));
        """
    )
    connection.executescript(_MIGRATION.read_text(encoding="utf-8"))

    columns = {row[1] for row in connection.execute("PRAGMA table_info(raw_sessions)")}
    assert {"source_index", "addressing_mode", "content_identity"} <= columns
    assert connection.execute(
        "SELECT source_index, addressing_mode, content_identity FROM raw_sessions"
    ).fetchone() == (7, "element_of_container", "a" + "0" * 63)


def test_source_train_does_not_reclassify_without_coordinate_evidence() -> None:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE raw_sessions(raw_id TEXT PRIMARY KEY, source_index INTEGER NOT NULL);
        CREATE TABLE raw_container_coordinates(
            raw_id TEXT PRIMARY KEY,
            addressing_mode TEXT,
            content_identity TEXT
        );
        INSERT INTO raw_sessions VALUES ('raw-1', 7);
        """
    )
    connection.executescript(_MIGRATION.read_text(encoding="utf-8"))

    assert connection.execute("SELECT addressing_mode, content_identity FROM raw_sessions").fetchone() == (None, None)
