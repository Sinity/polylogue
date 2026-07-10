from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.daemon.backup import BackupProfile, backup_archive
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user import USER_SCHEMA_VERSION
from polylogue.storage.sqlite.migration_runner import MigrationError, migrate_archive_tier


def _verified_backup_manifest(
    output_dir: Path,
    *,
    profile: BackupProfile = "rebuildable_cache_exclude",
) -> Path:
    result = backup_archive(output_dir=output_dir, profile=profile, verify=True)
    assert result.ok, result.verification
    assert result.verified is True
    assert result.output_path is not None
    receipt = Path(str(result.verification["receipt_path"]))
    assert receipt.exists()
    return Path(result.output_path) / "manifest.json"


def _unverified_backup_manifest(output_dir: Path, *, profile: BackupProfile = "user_overlays") -> Path:
    result = backup_archive(output_dir=output_dir, profile=profile, verify=False)
    assert result.ok
    assert result.output_path is not None
    assert not (Path(result.output_path) / "verification-receipt.json").exists()
    return Path(result.output_path) / "manifest.json"


def _block_migration_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(_conn: sqlite3.Connection, _sql: str) -> None:
        pytest.fail("migration SQL executed before backup receipt validation completed")

    monkeypatch.setattr("polylogue.storage.sqlite.migration_runner._execute_migration_sql", fail_if_called)


def _tamper_manifest(manifest: Path) -> None:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["warnings"] = ["tampered"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")


def _tamper_receipt(manifest: Path) -> None:
    receipt = manifest.with_name("verification-receipt.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["format"] = "tampered"
    receipt.write_text(json.dumps(payload), encoding="utf-8")


def _tamper_backup_tier(manifest: Path) -> None:
    conn = sqlite3.connect(manifest.with_name("user.db"))
    try:
        conn.execute("CREATE TABLE tamper (value TEXT)")
        conn.commit()
    finally:
        conn.close()


def _create_user_v3(path: Path) -> None:
    path.unlink(missing_ok=True)
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
    path.unlink(missing_ok=True)
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


def test_user_tier_v3_migrates_to_current_with_verified_backup_receipt(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")

    conn = sqlite3.connect(db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert result.from_version == 3
        assert result.to_version == USER_SCHEMA_VERSION
        assert result.applied_versions == (4, 5)
        assert result.backup_receipt == manifest.with_name("verification-receipt.json")
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_deliveries'"
        ).fetchone()
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


def test_source_tier_v1_migrates_to_current_without_native_uniqueness(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """v1 -> current applies all durable source migrations in sequence.

    002 relaxes the ``origin``/``native_id`` uniqueness constraint; 003 drops
    ``pending_blob_refs`` (polylogue-v7e0 — the lease mechanism it backed was
    never reachable from any production ingest caller). A v1 fixture predates
    the table entirely, so 003's ``DROP TABLE IF EXISTS`` is a no-op here —
    this pins that the chain still applies cleanly when the table never
    existed, not just when migrating forward from v2.
    """
    db_path = workspace_env["archive_root"] / "source.db"
    _create_source_v1(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup")

    conn = sqlite3.connect(db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
        assert result.from_version == 1
        assert result.to_version == SOURCE_SCHEMA_VERSION
        assert result.applied_versions == (2, 3, 4, 5, 6, 7)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info('raw_sessions')")}
        assert "predecessor_source_revision" in columns
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blob_publication_reservations'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_session_memberships'"
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
    path.unlink(missing_ok=True)
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


def _create_source_v3_with_referenced_blob(path: Path, blob_hash: str) -> None:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32)
            ) STRICT;
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
                ref_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                PRIMARY KEY(blob_hash, ref_type, ref_id)
            ) STRICT;
            PRAGMA user_version = 3;
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash) VALUES (?, ?)", ("raw-one", bytes.fromhex(blob_hash))
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, ref_id, ref_type) VALUES (?, ?, ?)",
            (bytes.fromhex(blob_hash), "raw-one", "raw_payload"),
        )
        conn.commit()
    finally:
        conn.close()


def test_source_tier_v2_migrates_to_v3_dropping_pending_blob_refs(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Migration 003 actually drops a populated ``pending_blob_refs`` table.

    Regression coverage for polylogue-v7e0: the lease mechanism the table
    backed was never reachable in production, so the table is removed
    rather than left as dead schema. Starts from a v2 fixture where the
    table exists and has a row, unlike the v1 fixture above where it never
    existed.
    """
    db_path = workspace_env["archive_root"] / "source.db"
    _create_source_v2_with_pending_blob_refs(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup")

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'").fetchone()

        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)

        assert result.from_version == 2
        assert result.to_version == SOURCE_SCHEMA_VERSION
        assert result.applied_versions == (3, 4, 5, 6, 7)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blob_publication_reservations'"
        ).fetchone()
    finally:
        conn.close()


def test_source_tier_v3_adds_publication_reservations_with_verified_backup_receipt(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = workspace_env["archive_root"] / "source.db"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32)
            ) STRICT;
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL CHECK(length(blob_hash) = 32),
                ref_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                PRIMARY KEY(blob_hash, ref_type, ref_id)
            ) STRICT;
            PRAGMA user_version = 3;
            """
        )
        conn.commit()
    finally:
        conn.close()
    manifest = _verified_backup_manifest(tmp_path / "backup")

    conn = sqlite3.connect(db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
        assert result.applied_versions == (4, 5, 6, 7)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        conn.execute(
            """
            INSERT INTO blob_publication_reservations (
                publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
            ) VALUES ('receipt-1', ?, 32, 'publisher', 1)
            """,
            (b"r" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blob_publication_reservations (
                publication_id, blob_hash, size_bytes, publisher_id, reserved_at_ms
            ) VALUES ('receipt-2', ?, 32, 'publisher-2', 2)
            """,
            (b"r" * 32,),
        )
        assert conn.execute("SELECT COUNT(*) FROM blob_publication_reservations").fetchone()[0] == 2
        indexes = {row[1] for row in conn.execute("PRAGMA index_list('blob_publication_reservations')")}
        assert "idx_blob_publication_reservations_hash" in indexes
    finally:
        conn.close()


def test_user_tier_migration_requires_receipt_with_target_tier(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="diagnostics_bundle")

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="does not include user.db"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


def test_unverified_backup_manifest_cannot_authorize_user_migration(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _unverified_backup_manifest(tmp_path / "backup-unverified")
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="successful backup verification receipt"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


def test_failed_backup_verification_cannot_authorize_user_migration(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    monkeypatch.setattr(
        "polylogue.daemon.backup._verify_archive_file_set_backup",
        lambda _path: {"ok": False, "error": "forced verification failure"},
    )
    result = backup_archive(output_dir=tmp_path / "backup-failed-verify", profile="user_overlays", verify=True)
    assert not result.ok
    assert result.output_path is not None
    manifest = Path(result.output_path) / "manifest.json"
    assert not manifest.with_name("verification-receipt.json").exists()
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="successful backup verification receipt"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("label", "mutate", "match"),
    [
        ("manifest", _tamper_manifest, "does not match manifest"),
        ("receipt", _tamper_receipt, "unsupported format"),
        ("backup-tier", _tamper_backup_tier, "tier artifact .* mismatch"),
    ],
)
def test_migration_receipt_detects_manifest_receipt_and_backup_db_mutations(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    mutate: Callable[[Path], None],
    match: str,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / f"backup-{label}", profile="user_overlays")
    mutate(manifest)
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match=match):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


def test_migration_rejects_receipt_transplanted_from_another_verified_backup(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    first_manifest = _verified_backup_manifest(tmp_path / "backup-first", profile="user_overlays")
    second_manifest = _verified_backup_manifest(tmp_path / "backup-second", profile="user_overlays")
    shutil.copy2(
        first_manifest.with_name("verification-receipt.json"), second_manifest.with_name("verification-receipt.json")
    )
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="does not match manifest"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=second_manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
    finally:
        conn.close()


def test_migration_receipt_detects_stale_live_tier_bytes(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup-stale-live", profile="user_overlays")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE local_mutation (value TEXT)")
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="live tier .* mismatch"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
    finally:
        conn.close()


def test_migration_rejects_live_writes_committed_after_receipt_validation(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
    manifest = _verified_backup_manifest(tmp_path / "backup-race", profile="user_overlays")
    original_validate = migration_runner.validate_migration_backup_manifest
    validation_count = 0

    def validate_then_commit(
        path: Path,
        tier: ArchiveTier,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Path:
        nonlocal validation_count
        receipt = original_validate(path, tier, connection=connection)
        validation_count += 1
        if validation_count == 1:
            with sqlite3.connect(db_path) as concurrent_writer:
                concurrent_writer.execute("CREATE TABLE committed_after_validation (value TEXT)")
        return receipt

    monkeypatch.setattr(
        "polylogue.storage.sqlite.migration_runner.validate_migration_backup_manifest",
        validate_then_commit,
    )
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="changed before the migration lock"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'committed_after_validation'"
        ).fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize("mutation", ["added", "removed", "resized", "hash-mismatched"])
def test_migration_receipt_detects_blob_inventory_mutations(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    archive_root = workspace_env["archive_root"]
    db_path = archive_root / "source.db"
    payload = b"migration receipt blob evidence"
    blob_hash, _ = BlobStore(archive_root / "blob").write_from_bytes(payload)
    _create_source_v3_with_referenced_blob(db_path, blob_hash)
    manifest = _verified_backup_manifest(tmp_path / f"backup-blob-{mutation}")
    copied_blob = manifest.parent / "blob" / blob_hash[:2] / blob_hash[2:]
    if mutation == "added":
        extra_payload = b"extra blob evidence"
        extra_hash = hashlib.sha256(extra_payload).hexdigest()
        extra_path = manifest.parent / "blob" / extra_hash[:2] / extra_hash[2:]
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        extra_path.write_bytes(extra_payload)
    elif mutation == "removed":
        copied_blob.unlink()
    elif mutation == "resized":
        copied_blob.write_bytes(b"short")
    else:
        copied_blob.write_bytes(b"x" * len(payload))
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="blob"):
            migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blob_publication_reservations'"
        ).fetchone()
    finally:
        conn.close()


def test_initialize_database_refuses_old_durable_tier_without_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    _create_user_v3(db_path)

    with pytest.raises(RuntimeError, match="explicit durable-tier migration"):
        initialize_archive_database(db_path, ArchiveTier.USER)


def test_initialize_database_can_apply_explicit_user_migration(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")

    initialize_archive_database(db_path, ArchiveTier.USER, migration_backup_manifest=manifest)

    conn = sqlite3.connect(db_path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
    finally:
        conn.close()


def test_derived_tiers_do_not_use_migration_runner(tmp_path: Path) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA user_version = 1")
        with pytest.raises(MigrationError, match="does not support in-place migrations"):
            migrate_archive_tier(conn, ArchiveTier.INDEX, backup_manifest=tmp_path / "missing-manifest.json")
    finally:
        conn.close()
