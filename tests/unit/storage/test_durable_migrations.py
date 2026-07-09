from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.migration_runner import MigrationError, migrate_archive_tier


def _write_backup_manifest(path: Path, *, included_tiers: list[str]) -> Path:
    path.mkdir()
    manifest = {
        "format": "polylogue-backup-v1",
        "profile": "user_overlays",
        "included_tiers": included_tiers,
        "omitted_tiers": [],
        "backed_up_files": [],
        "warnings": [],
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _create_user_v3(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE assertions (
                assertion_id        TEXT PRIMARY KEY,
                scope_ref           TEXT,
                target_ref          TEXT NOT NULL,
                key                 TEXT,
                kind                TEXT NOT NULL,
                value_json          TEXT,
                body_text           TEXT,
                author_ref          TEXT DEFAULT 'user:local',
                author_kind         TEXT DEFAULT 'user',
                evidence_refs_json  TEXT DEFAULT '[]',
                status              TEXT DEFAULT 'active',
                visibility          TEXT DEFAULT 'private',
                confidence          REAL,
                staleness_json      TEXT,
                context_policy_json TEXT DEFAULT '{"inject":false}',
                supersedes_json     TEXT DEFAULT '[]',
                created_at_ms       INTEGER NOT NULL,
                updated_at_ms       INTEGER NOT NULL
            ) STRICT;
            CREATE INDEX idx_assertions_target_kind
            ON assertions(target_ref, kind);
            CREATE INDEX idx_assertions_kind_status_updated
            ON assertions(kind, status, updated_at_ms);
            CREATE INDEX idx_assertions_target_kind_status_visibility
            ON assertions(target_ref, kind, status, visibility);
            PRAGMA user_version = 3;
            """
        )
        conn.commit()
    finally:
        conn.close()


def _create_source_v1(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id                  TEXT PRIMARY KEY,
                origin                  TEXT NOT NULL,
                native_id               TEXT,
                source_path             TEXT NOT NULL,
                source_index            INTEGER NOT NULL DEFAULT 0,
                blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
                blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
                acquired_at_ms          INTEGER NOT NULL,
                file_mtime_ms           INTEGER,
                parsed_at_ms            INTEGER,
                parse_error             TEXT,
                validated_at_ms         INTEGER,
                validation_status       TEXT,
                validation_error        TEXT,
                validation_drift_count  INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
                validation_mode         TEXT,
                detection_warnings_json TEXT NOT NULL DEFAULT '[]'
            ) STRICT;
            CREATE INDEX idx_raw_sessions_origin ON raw_sessions(origin);
            CREATE UNIQUE INDEX idx_raw_sessions_origin_native
            ON raw_sessions(origin, native_id)
            WHERE native_id IS NOT NULL;
            PRAGMA user_version = 1;
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_user_tier_v3_migrates_to_v4_with_backup_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    _create_user_v3(db_path)
    manifest = _write_backup_manifest(tmp_path / "backup", included_tiers=["user.db"])

    conn = sqlite3.connect(db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert result.from_version == 3
        assert result.to_version == USER_SCHEMA_VERSION
        assert result.applied_versions == (4,)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
        conn.execute(
            "INSERT INTO user_settings (setting_key, value_json, updated_at_ms) VALUES (?, ?, ?)",
            ("reader.theme", '"system"', 123),
        )
        assert (
            conn.execute("SELECT value_json FROM user_settings WHERE setting_key = ?", ("reader.theme",)).fetchone()[0]
            == '"system"'
        )
    finally:
        conn.close()


def test_source_tier_v1_migrates_to_current_without_native_uniqueness(tmp_path: Path) -> None:
    """v1 -> current applies both additive migrations in sequence.

    002 relaxes the ``origin``/``native_id`` uniqueness constraint; 003 drops
    ``pending_blob_refs`` (polylogue-v7e0 — the lease mechanism it backed was
    never reachable from any production ingest caller). A v1 fixture predates
    the table entirely, so 003's ``DROP TABLE IF EXISTS`` is a no-op here —
    this pins that the chain still applies cleanly when the table never
    existed, not just when migrating forward from v2.
    """
    db_path = tmp_path / "source.db"
    _create_source_v1(db_path)
    manifest = _write_backup_manifest(tmp_path / "backup", included_tiers=["source.db"])

    conn = sqlite3.connect(db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
        assert result.from_version == 1
        assert result.to_version == SOURCE_SCHEMA_VERSION
        assert result.applied_versions == (2, 3)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'"
        ).fetchone()

        index_rows = conn.execute("PRAGMA index_list('raw_sessions')").fetchall()
        origin_native_index = next(row for row in index_rows if row[1] == "idx_raw_sessions_origin_native")
        assert int(origin_native_index[2]) == 0

        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw:direct", "chatgpt-export", "conversation-1", "/direct.json", 0, b"0" * 32, 2, 1),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw:browser", "chatgpt-export", "conversation-1", "/browser.json", 0, b"1" * 32, 2, 2),
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM raw_sessions WHERE origin = ? AND native_id = ?",
                ("chatgpt-export", "conversation-1"),
            ).fetchone()[0]
            == 2
        )
    finally:
        conn.close()


def _create_source_v2_with_pending_blob_refs(path: Path) -> None:
    """A v2 source tier that still carries ``pending_blob_refs``.

    Mirrors the pre-polylogue-v7e0 schema so migration 003's
    ``DROP TABLE IF EXISTS pending_blob_refs`` has a real table to drop,
    proving the migration is effective, not just a no-op ``IF EXISTS``.
    """
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id                  TEXT PRIMARY KEY,
                origin                  TEXT NOT NULL,
                native_id               TEXT,
                source_path             TEXT NOT NULL,
                source_index            INTEGER NOT NULL DEFAULT 0,
                blob_hash               BLOB NOT NULL CHECK(length(blob_hash) = 32),
                blob_size               INTEGER NOT NULL CHECK(blob_size >= 0),
                acquired_at_ms          INTEGER NOT NULL,
                file_mtime_ms           INTEGER,
                parsed_at_ms            INTEGER,
                parse_error             TEXT,
                validated_at_ms         INTEGER,
                validation_status       TEXT,
                validation_error        TEXT,
                validation_drift_count  INTEGER NOT NULL DEFAULT 0 CHECK(validation_drift_count >= 0),
                validation_mode         TEXT,
                detection_warnings_json TEXT NOT NULL DEFAULT '[]'
            ) STRICT;
            CREATE INDEX idx_raw_sessions_origin ON raw_sessions(origin);
            CREATE INDEX idx_raw_sessions_origin_native
            ON raw_sessions(origin, native_id)
            WHERE native_id IS NOT NULL;
            CREATE TABLE pending_blob_refs (
                blob_hash       BLOB NOT NULL CHECK(length(blob_hash) = 32),
                operation_id    TEXT NOT NULL,
                ref_type        TEXT NOT NULL,
                ref_id          TEXT NOT NULL,
                acquired_at_ms  INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, operation_id, ref_type, ref_id)
            );
            CREATE INDEX idx_pending_blob_refs_operation
            ON pending_blob_refs(operation_id);
            PRAGMA user_version = 2;
            """
        )
        conn.execute(
            "INSERT INTO pending_blob_refs (blob_hash, operation_id, ref_type, ref_id, acquired_at_ms) "
            "VALUES (?, 'op-1', 'raw_payload', 'op-1', 0)",
            (b"a" * 32,),
        )
        conn.commit()
    finally:
        conn.close()


def test_source_tier_v2_migrates_to_v3_dropping_pending_blob_refs(tmp_path: Path) -> None:
    """Migration 003 actually drops a populated ``pending_blob_refs`` table.

    Regression coverage for polylogue-v7e0: the lease mechanism the table
    backed was never reachable in production, so the table is removed
    rather than left as dead schema. Starts from a v2 fixture where the
    table exists and has a row, unlike the v1 fixture above where it never
    existed.
    """
    db_path = tmp_path / "source.db"
    _create_source_v2_with_pending_blob_refs(db_path)
    manifest = _write_backup_manifest(tmp_path / "backup", included_tiers=["source.db"])

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'").fetchone()

        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)

        assert result.from_version == 2
        assert result.to_version == SOURCE_SCHEMA_VERSION
        assert result.applied_versions == (3,)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'"
        ).fetchone()
    finally:
        conn.close()


def test_user_tier_migration_requires_manifest_with_target_tier(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    _create_user_v3(db_path)
    manifest = _write_backup_manifest(tmp_path / "backup", included_tiers=["source.db"])

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="does not include user.db"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


def test_initialize_database_refuses_old_durable_tier_without_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    _create_user_v3(db_path)

    with pytest.raises(RuntimeError, match="explicit durable-tier migration"):
        initialize_archive_database(db_path, ArchiveTier.USER)


def test_initialize_database_can_apply_explicit_user_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    _create_user_v3(db_path)
    manifest = _write_backup_manifest(tmp_path / "backup", included_tiers=["user.db"])

    initialize_archive_database(db_path, ArchiveTier.USER, migration_backup_manifest=manifest)

    conn = sqlite3.connect(db_path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


def test_derived_tiers_do_not_use_migration_runner(tmp_path: Path) -> None:
    manifest = _write_backup_manifest(tmp_path / "backup", included_tiers=["index.db"])
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA user_version = 1")
        with pytest.raises(MigrationError, match="does not support in-place migrations"):
            migrate_archive_tier(conn, ArchiveTier.INDEX, backup_manifest=manifest)
    finally:
        conn.close()
