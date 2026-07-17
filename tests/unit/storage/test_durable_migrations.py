from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from polylogue.annotations.schema import DELEGATION_DISCOURSE_SCHEMA
from polylogue.daemon import backup as backup_mod
from polylogue.daemon.backup import BackupProfile, backup_archive
from polylogue.storage.backup_attestation import (
    ATTESTATION_ALGORITHM,
    ATTESTATION_FORMAT,
    VERIFICATION_RECEIPT_FORMAT,
    attestation_key_path,
    load_attestation_key,
    sign_verification_receipt,
    tier_attestation_id,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite import migration_runner
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL, SOURCE_SCHEMA_VERSION
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
    payload["verdict"] = "failure"
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


def _create_user_v5(path: Path) -> None:
    """Create the exact pre-annotation durable user tier."""

    _create_user_v3(path)
    migrations = Path(__file__).parents[3] / "polylogue" / "storage" / "sqlite" / "migrations" / "user"
    conn = sqlite3.connect(path)
    try:
        for version, filename in ((4, "004_user_settings.sql"), (5, "005_context_deliveries.sql")):
            conn.executescript((migrations / filename).read_text(encoding="utf-8"))
            conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    finally:
        conn.close()


_USER_DURABLE_SCHEMA_OBJECTS = (
    "annotation_batches",
    "annotation_schemas",
    "idx_annotation_batches_schema_target_time",
    "idx_annotation_batches_source_result_time",
    "idx_assertions_scope_kind_status",
    "queries",
    "query_names",
    "idx_query_names_query_hash",
    "idx_query_names_watch",
    "result_sets",
    "idx_result_sets_query_epoch",
    "result_set_members",
    "query_edges",
    "idx_query_edges_dst_kind",
    "retained_query_runs",
    "query_evaluation_receipts",
    "idx_query_evaluation_receipts_query_time",
    "watched_query_baselines",
    "retained_query_runs_result_set_query_match_insert",
    "retained_query_runs_result_set_query_match_update",
    "query_evaluation_receipts_result_set_query_match_insert",
    "query_evaluation_receipts_result_set_query_match_update",
    "watched_query_baselines_result_set_query_match_insert",
    "watched_query_baselines_result_set_query_match_update",
    "result_set_holdout_policies",
    "holdout_access_receipts",
    "idx_holdout_access_receipts_result_set",
)


def _user_durable_schema_sql(conn: sqlite3.Connection) -> tuple[tuple[object, ...], ...]:
    placeholders = ",".join("?" for _ in _USER_DURABLE_SCHEMA_OBJECTS)
    rows = conn.execute(
        f"""
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name IN ({placeholders})
            ORDER BY type, name
            """,
        _USER_DURABLE_SCHEMA_OBJECTS,
    ).fetchall()
    return tuple((str(row[0]), str(row[1]), str(row[2]), _normalize_schema_sql(str(row[3]))) for row in rows)


def _normalize_schema_sql(sql: str) -> str:
    """Compare SQLite DDL semantics despite ALTER TABLE's punctuation layout."""
    collapsed = re.sub(r"\s+", " ", sql).strip()
    collapsed = re.sub(r"\s*,\s*", ",", collapsed)
    collapsed = re.sub(r"\(\s*", "(", collapsed)
    return re.sub(r"\s*\)", ")", collapsed)


def _assert_query_provenance_binding_triggers(conn: sqlite3.Connection) -> None:
    """Exercise migration and fresh DDL against raw SQL bypasses."""
    first_hash, second_hash = "b" * 64, "c" * 64
    conn.executemany(
        """
        INSERT INTO queries (
            query_hash, canonical_plan_json, grain, lane, rank_policy, created_at_ms
        ) VALUES (?, '{}', 'session', 'dialogue', 'mixed', 1)
        """,
        ((first_hash,), (second_hash,)),
    )
    conn.executemany(
        """
        INSERT INTO result_sets (
            result_set_id, query_hash, grain, corpus_epoch, member_count,
            membership_merkle_root, ordered_rank_hash, exactness, persistence_class, created_at_ms
        ) VALUES (?, ?, 'session', 'index:g1', 0, ?, ?, 'exact', 'watch', 1)
        """,
        (("binding-first", first_hash, "1" * 64, "2" * 64), ("binding-second", second_hash, "3" * 64, "4" * 64)),
    )
    with pytest.raises(sqlite3.IntegrityError, match="same query"):
        conn.execute(
            "INSERT INTO retained_query_runs (run_id, query_hash, result_set_id, retained_at_ms) VALUES ('qr_raw', ?, 'binding-second', 1)",
            (first_hash,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="same query"):
        conn.execute(
            """
            INSERT INTO query_evaluation_receipts (
                receipt_id, query_hash, result_set_id, source_generation, user_generation,
                index_generation, runtime_build_ref, model_refs_json, resolved_bounds_json,
                degradation_json, created_at_ms
            ) VALUES ('receipt-raw', ?, 'binding-second', 's', 'u', 'i', 'b', '[]', '{}', '{}', 1)
            """,
            (first_hash,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="same query"):
        conn.execute(
            "INSERT INTO watched_query_baselines (query_hash, result_set_id, updated_at_ms) VALUES (?, 'binding-second', 1)",
            (first_hash,),
        )


def _assert_user_v6_annotation_checks(conn: sqlite3.Connection, *, suffix: str) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO annotation_schemas (
                schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
            ) VALUES (?, 1, ?, ?, 1)
            """,
            (f"check.{suffix}", '{"schema_id":"different","version":1}', "a" * 64),
        )
    for missing_key_definition in (
        '{"version":1}',
        '{"schema_id":"check.missing"}',
    ):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO annotation_schemas (
                    schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
                ) VALUES ('check.missing', 1, ?, ?, 1)
                """,
                (missing_key_definition, "b" * 64),
            )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO annotation_batches (
                batch_id, schema_id, schema_version, target_ref, source_result_ref,
                actor_ref, model_ref, prompt_ref, total_count, valid_count,
                invalid_count, abstained_count, assertion_refs_json,
                validation_failures_json, metadata_json, created_at_ms
            ) VALUES (?, 'delegation.discourse', 1, 'delegation:check', 'result-set:check',
                      'agent:check', 'agent:model', 'block:prompt:0', 2, 1, 0, 0,
                      '["assertion:one"]', '[]', '{}', 1)
            """,
            (f"bad-counts-{suffix}",),
        )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO annotation_batches (
                batch_id, schema_id, schema_version, target_ref, source_result_ref,
                actor_ref, model_ref, prompt_ref, total_count, valid_count,
                invalid_count, abstained_count, assertion_refs_json,
                validation_failures_json, metadata_json, created_at_ms
            ) VALUES (?, 'delegation.discourse', 1, 'delegation:check', 'result-set:check',
                      'agent:check', 'agent:model', 'block:prompt:0', 1, 1, 0, 0,
                      '[]', '[]', '{}', 1)
            """,
            (f"bad-ref-count-{suffix}",),
        )


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
        assert result.applied_versions == (4, 5, 6, 7, 8, 9)
        assert result.backup_receipt == manifest.with_name("verification-receipt.json")
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_settings'").fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='context_deliveries'"
        ).fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='annotation_schemas'"
        ).fetchone()
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='annotation_batches'"
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


def test_user_tier_v5_annotation_migration_requires_verified_backup_and_matches_fresh_ddl(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v5(db_path)
    with sqlite3.connect(db_path) as seed_conn:
        seed_conn.execute(
            """
            INSERT INTO assertions (
                assertion_id, target_ref, kind, created_at_ms, updated_at_ms
            ) VALUES ('sentinel', 'session:sentinel', 'annotation', 1, 1)
            """
        )
        seed_conn.execute(
            """
            INSERT INTO assertions (
                assertion_id, target_ref, key, kind, value_json, created_at_ms, updated_at_ms
            ) VALUES ('saved-query-sentinel', 'saved_view:sentinel', 'recent', 'saved_query', ?, 1, 1)
            """,
            (json.dumps({"origin": "codex-session", "limit": 10}),),
        )
        seed_conn.commit()
    manifest = _verified_backup_manifest(tmp_path / "backup-v5", profile="user_overlays")

    conn = sqlite3.connect(db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert result.from_version == 5
        assert result.to_version == USER_SCHEMA_VERSION == 9
        assert result.applied_versions == (6, 7, 8, 9)
        assert result.backup_receipt == manifest.with_name("verification-receipt.json")
        assert conn.execute("SELECT assertion_id FROM assertions WHERE assertion_id = 'sentinel'").fetchone()
        saved_target = conn.execute(
            "SELECT target_ref FROM assertions WHERE assertion_id = 'saved-query-sentinel'"
        ).fetchone()
        assert saved_target is not None and str(saved_target[0]).startswith("query:")
        assert conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 1
        schema_row = conn.execute(
            """
            SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
            FROM annotation_schemas
            WHERE schema_id = 'delegation.discourse' AND schema_version = 1
            """
        ).fetchone()
        expected_seed = (
            DELEGATION_DISCOURSE_SCHEMA.schema_id,
            DELEGATION_DISCOURSE_SCHEMA.version,
            DELEGATION_DISCOURSE_SCHEMA.canonical_definition_json(),
            DELEGATION_DISCOURSE_SCHEMA.definition_fingerprint,
            0,
        )
        assert schema_row == expected_seed
        assert {str(row[1]) for row in conn.execute("PRAGMA index_list(assertions)")} >= {
            "idx_assertions_scope_kind_status"
        }
        _assert_query_provenance_binding_triggers(conn)

        fresh_db = tmp_path / "fresh-user-v6.db"
        initialize_archive_database(fresh_db, ArchiveTier.USER)
        with sqlite3.connect(fresh_db) as fresh_conn:
            assert _user_durable_schema_sql(conn) == _user_durable_schema_sql(fresh_conn)
            _assert_query_provenance_binding_triggers(fresh_conn)
            assert tuple(conn.execute("PRAGMA foreign_key_list(annotation_batches)")) == tuple(
                fresh_conn.execute("PRAGMA foreign_key_list(annotation_batches)")
            )
            assert (
                fresh_conn.execute(
                    """
                SELECT schema_id, schema_version, definition_json, definition_sha256, registered_at_ms
                FROM annotation_schemas
                WHERE schema_id = 'delegation.discourse' AND schema_version = 1
                """
                ).fetchone()
                == expected_seed
            )
            _assert_user_v6_annotation_checks(conn, suffix="migrated")
            _assert_user_v6_annotation_checks(fresh_conn, suffix="fresh")
    finally:
        conn.close()


def test_symlinked_user_tier_uses_resolved_attestation_authority(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    archive_root = workspace_env["archive_root"]
    physical_db_path = tmp_path / "durable" / "user.db"
    physical_db_path.parent.mkdir()
    _create_user_v3(physical_db_path)
    configured_db_path = archive_root / "user.db"
    configured_db_path.unlink(missing_ok=True)
    configured_db_path.symlink_to(physical_db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")

    conn = sqlite3.connect(configured_db_path)
    try:
        result = migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert result.from_version == 3
        assert result.to_version == USER_SCHEMA_VERSION
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == USER_SCHEMA_VERSION
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
        assert result.applied_versions == (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info('raw_sessions')")}
        assert "predecessor_source_revision" in columns
        assert "capture_mode" in columns
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blob_publication_reservations'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_session_memberships'"
        ).fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sinex_publication_obligations'"
        ).fetchone()
        conn.execute(
            """
            INSERT INTO sinex_publication_obligations (
                object_id, protocol_version, revision_id, manifest_digest, mode,
                created_at_ms, updated_at_ms
            ) VALUES ('claude-code-session:migrated', 'polylogue.material-protocol/v1', 'rev-1', 'digest-1', 'mirror', 1, 1)
            """
        )
        assert conn.execute("SELECT status FROM sinex_publication_obligations").fetchone()[0] == "pending"

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


def test_source_publication_backfill_requires_verified_backup(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    db_path = workspace_env["archive_root"] / "source.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE raw_authority_parser_census")
        conn.execute("DROP TABLE raw_authority_blockers")
        conn.execute("DROP TABLE raw_authority_census_post_plans")
        conn.execute("DROP TABLE raw_authority_census_plans")
        conn.execute("DROP TABLE raw_authority_plans")
        conn.execute("DROP TABLE raw_authority_censuses")
        conn.execute("DROP TABLE excised_content")
        conn.execute("DROP TABLE sinex_publication_segments")
        conn.execute("DROP TABLE sinex_publication_receipts")
        conn.execute("DROP TABLE sinex_publication_payloads")
        conn.execute("DROP TABLE sinex_publication_obligations")
        conn.execute("PRAGMA user_version = 9")
        conn.commit()

        with pytest.raises(MigrationError, match="verified backup manifest"):
            migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=None)

    manifest = _verified_backup_manifest(tmp_path / "backup-source-publication")
    with sqlite3.connect(db_path) as conn:
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)

        assert result.from_version == 9
        assert result.to_version == SOURCE_SCHEMA_VERSION == 13
        assert result.applied_versions == (10, 11, 12, 13)
        assert result.backup_receipt == manifest.with_name("verification-receipt.json")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {
            "sinex_publication_obligations",
            "sinex_publication_payloads",
            "sinex_publication_segments",
            "sinex_publication_receipts",
            "excised_content",
            "raw_authority_censuses",
            "raw_authority_parser_census",
            "raw_authority_plans",
            "raw_authority_census_plans",
            "raw_authority_census_post_plans",
            "raw_authority_blockers",
        } <= tables


def test_additive_no_backup_marker_must_be_the_header_not_a_substring() -> None:
    """CodeRabbit #2905: substring matching would waive the backup requirement
    if the marker text ever appeared in a comment, string literal, or later in
    the file. It must be the file's first non-blank line."""
    header = migration_runner._ADDITIVE_NO_BACKUP_MARKER

    assert (
        migration_runner._requires_migration_backup(f"{header}\n-- a real migration\nCREATE TABLE t (x INTEGER);")
        is False
    )
    assert migration_runner._requires_migration_backup(f"{header} trailing text\nCREATE TABLE t (x INTEGER);") is True
    assert (
        migration_runner._requires_migration_backup(f"-- unrelated comment\n{header}\nCREATE TABLE t (x INTEGER);")
        is True
    )
    assert migration_runner._requires_migration_backup(f"CREATE TABLE t (x TEXT DEFAULT '{header}');") is True
    assert migration_runner._requires_migration_backup("CREATE TABLE t (x INTEGER);") is True
    assert migration_runner._requires_migration_backup(f"\n\n  {header}  \nCREATE TABLE t (x INTEGER);") is False


def test_source_tier_v7_expands_origin_checks_with_verified_backup(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The Beads origin copy-forward retains v7 evidence and accepts new raws."""
    db_path = workspace_env["archive_root"] / "source.db"
    db_path.unlink(missing_ok=True)
    # sinex_publication_obligations (migration 010) postdates v7; migration
    # 010 issues its own CREATE TABLE, so the v7 fixture must not already
    # carry it (unlike the tables migrations 8/9 rebuild in place, this one
    # is untouched by the origin-CHECK copy-forward under test).
    sinex_obligations_marker = "\nCREATE TABLE IF NOT EXISTS sinex_publication_obligations"
    old_ddl = SOURCE_DDL[: SOURCE_DDL.index(sinex_obligations_marker)] + "\n"
    old_ddl = old_ddl.replace(", 'beads-issue'", "")
    old_ddl = old_ddl.replace(
        "    capture_mode            TEXT CHECK ((capture_mode IN "
        "('chatgpt', 'claude-ai', 'claude-code', 'codex', 'gemini', 'gemini-cli', "
        "'hermes', 'antigravity', 'beads', 'grok', 'drive', 'unknown') "
        "OR capture_mode IS NULL)),\n",
        "",
    )
    old_ddl = old_ddl.replace(
        "    ,source_revision         TEXT\n"
        "    ,predecessor_source_revision TEXT\n"
        "    ,predecessor_raw_id      TEXT\n"
        "    ,baseline_raw_id         TEXT\n"
        "    ,append_start_offset     INTEGER CHECK(append_start_offset >= 0)\n"
        "    ,append_end_offset       INTEGER CHECK(append_end_offset > append_start_offset)\n"
        "    ,acquisition_generation  INTEGER CHECK(acquisition_generation >= 0)\n"
        "    ,revision_authority      TEXT NOT NULL DEFAULT 'quarantined'\n"
        "        CHECK(revision_authority IN ('asserted', 'byte_proven', 'quarantined'))\n",
        "    ,source_revision         TEXT\n"
        "    ,predecessor_raw_id      TEXT\n"
        "    ,baseline_raw_id         TEXT\n"
        "    ,append_start_offset     INTEGER CHECK(append_start_offset >= 0)\n"
        "    ,append_end_offset       INTEGER CHECK(append_end_offset > append_start_offset)\n"
        "    ,acquisition_generation  INTEGER CHECK(acquisition_generation >= 0)\n"
        "    ,revision_authority      TEXT NOT NULL DEFAULT 'quarantined'\n"
        "        CHECK(revision_authority IN ('asserted', 'byte_proven', 'quarantined'))\n"
        "    ,predecessor_source_revision TEXT\n",
    )
    authority_start = old_ddl.index("-- Durable authority reconciliation ledger.")
    authority_end = old_ddl.index("CREATE TABLE IF NOT EXISTS blob_refs", authority_start)
    old_ddl = old_ddl[:authority_start] + old_ddl[authority_end:]
    # Migration 010 adds `excised_content` (polylogue-27m) -- a v7 snapshot
    # predates it, same as it predates the beads-origin/capture_mode diffs
    # stripped above. Without this, the fixture (built from the CURRENT
    # SOURCE_DDL) already contains the table, and migration 010's
    # non-`IF NOT EXISTS` `CREATE TABLE excised_content` collides with it.
    old_ddl = old_ddl.replace(
        "\n"
        "-- Durable removed-content ledger (polylogue-27m). A row here is the\n"
        '-- authoritative "this content is forgotten on purpose" marker for\n'
        "-- standalone/off-mode excision: the acquire-time write chokepoint\n"
        "-- (``write_source_raw_session``) refuses to re-store a raw payload whose\n"
        "-- ``blob_hash`` matches ``removed_hash``, so an ordinary re-ingest of\n"
        "-- unmodified source files cannot resurrect excised content even after an\n"
        "-- index.db rebuild. ``span_start``/``span_end`` are populated only for a\n"
        "-- sub-payload excision (e.g. a detected secret candidate span); both are\n"
        "-- NULL for a whole-raw-session excision. This table is never queried for\n"
        "-- its own sake by a reader -- it exists purely as a write-time gate plus\n"
        "-- forensic trail, so no secret span coordinates ever carry the removed\n"
        "-- literal, only byte offsets into the (now-deleted) payload.\n"
        "CREATE TABLE IF NOT EXISTS excised_content (\n"
        "    removed_hash    BLOB NOT NULL CHECK(length(removed_hash) = 32),\n"
        "    hash_kind       TEXT NOT NULL DEFAULT 'blob_hash' CHECK(hash_kind IN ('blob_hash')),\n"
        "    reason          TEXT NOT NULL,\n"
        "    actor           TEXT NOT NULL,\n"
        "    prior_revision  TEXT,\n"
        "    span_start      INTEGER CHECK(span_start IS NULL OR span_start >= 0),\n"
        "    span_end        INTEGER CHECK(span_end IS NULL OR span_end > span_start),\n"
        "    excised_at_ms   INTEGER NOT NULL,\n"
        "    PRIMARY KEY(removed_hash, hash_kind)\n"
        ") STRICT;\n",
        "",
    )
    blob_hash, blob_size = BlobStore(workspace_env["archive_root"] / "blob").write_from_bytes(b"before-beads")
    with sqlite3.connect(db_path) as conn:
        conn.executescript(old_ddl)
        conn.execute("PRAGMA user_version = 7")
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size,
                acquired_at_ms, logical_source_key, revision_kind, source_revision,
                predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                append_start_offset, append_end_offset, acquisition_generation, revision_authority
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "before-beads",
                "codex-session",
                "session-1",
                "/before.jsonl",
                0,
                bytes.fromhex(blob_hash),
                blob_size,
                1,
                "codex:session-1",
                "append",
                "source-revision-1",
                "source-revision-0",
                "raw-predecessor",
                "raw-baseline",
                3,
                7,
                2,
                "byte_proven",
            ),
        )
        conn.commit()
    manifest = _verified_backup_manifest(tmp_path / "backup-v7")

    with sqlite3.connect(db_path) as conn:
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
        assert result.from_version == 7
        assert result.to_version == SOURCE_SCHEMA_VERSION == 13
        assert result.applied_versions == (8, 9, 10, 11, 12, 13)
        assert conn.execute(
            """
            SELECT predecessor_source_revision, predecessor_raw_id, baseline_raw_id,
                   append_start_offset, append_end_offset, acquisition_generation,
                   revision_authority
            FROM raw_sessions WHERE raw_id = 'before-beads'
            """
        ).fetchone() == (
            "source-revision-0",
            "raw-predecessor",
            "raw-baseline",
            3,
            7,
            2,
            "byte_proven",
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("beads-raw", "beads-issue", "polylogue-7fj", "/repo/.beads/interactions.jsonl", 0, b"b" * 32, 1, 2),
        )
        assert conn.execute("SELECT raw_id FROM raw_sessions WHERE origin = 'beads-issue'").fetchone()


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
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-one", "codex-session", "session-1", "/fixture.jsonl", 0, bytes.fromhex(blob_hash), 1, 0),
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
        assert result.applied_versions == (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
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
        assert result.applied_versions == (4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
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


@pytest.mark.parametrize("attestation_mode", ["missing", "forged-mac"])
def test_public_hash_forged_receipt_cannot_authorize_user_migration(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attestation_mode: str,
) -> None:
    """Public backup hashes cannot impersonate the scratch verifier."""
    archive_root = workspace_env["archive_root"]
    db_path = archive_root / "user.db"
    _create_user_v3(db_path)

    _verified_backup_manifest(tmp_path / "trusted", profile="user_overlays")
    manifest = _unverified_backup_manifest(tmp_path / "unverified")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    forged: dict[str, object] = {
        "format": VERIFICATION_RECEIPT_FORMAT,
        "verdict": "success",
        "verified_at": "2026-07-11T00:00:00+00:00",
        "mode": "archive_file_set",
        "profile": manifest_payload["profile"],
        "manifest_path": "manifest.json",
        **backup_mod._receipt_evidence(manifest.parent),
        "verification": {"ok": True, "scratch_restore": "claimed"},
    }
    if attestation_mode == "forged-mac":
        forged["attestations"] = [
            {
                "format": ATTESTATION_FORMAT,
                "algorithm": ATTESTATION_ALGORITHM,
                "tier": "user",
                "resource_id": tier_attestation_id(db_path),
                "key_id": hashlib.sha256(load_attestation_key(db_path)).hexdigest(),
                "mac": "0" * 64,
            }
        ]
    manifest.with_name("verification-receipt.json").write_text(
        json.dumps(forged, indent=2, sort_keys=True), encoding="utf-8"
    )
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="receipt authentication failed"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("key_state", "match"),
    [
        ("missing", "attestation key is missing"),
        ("rotated", "attestation key does not match"),
    ],
)
def test_verified_receipt_loses_authority_when_local_key_changes(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key_state: str,
    match: str,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / f"backup-{key_state}", profile="user_overlays")
    key_path = attestation_key_path(db_path)
    if key_state == "missing":
        key_path.unlink()
    else:
        key_path.write_bytes(os.urandom(32))
        key_path.chmod(0o600)
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match=match):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
    finally:
        conn.close()


def test_migration_rejects_signed_artifact_source_fingerprint_mismatch(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE newer_live_state (value TEXT)")
    newer_fingerprint = backup_mod._sqlite_source_fingerprint(db_path)

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["tier_source_fingerprints"]["user.db"] = newer_fingerprint
    manifest.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True), encoding="utf-8")

    receipt_path = manifest.with_name("verification-receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_size_bytes"] = manifest.stat().st_size
    receipt["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    user_artifact = next(item for item in receipt["tier_artifacts"] if item["tier"] == "user")
    user_artifact["source_fingerprint"] = newer_fingerprint
    unsigned_receipt = {key: value for key, value in receipt.items() if key != "attestations"}
    signed_receipt = sign_verification_receipt(unsigned_receipt, authority_paths={"user": db_path})
    receipt_path.write_text(json.dumps(signed_receipt, indent=2, sort_keys=True), encoding="utf-8")
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="artifact does not match its live source fingerprint"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_migration_rejects_tier_artifact_aliases_to_live_database(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_kind: str,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")
    copied_user_db = manifest.with_name("user.db")
    copied_user_db.unlink()
    if alias_kind == "symlink":
        copied_user_db.symlink_to(db_path)
    else:
        os.link(db_path, copied_user_db)
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="real regular file|multiple hard links|aliases the live tier"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
    finally:
        conn.close()


def test_migration_rejects_post_receipt_wal_without_main_file_change(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode = WAL").fetchone()[0] == "wal"
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")
    copied_user_db = manifest.with_name("user.db")
    main_hash = hashlib.sha256(copied_user_db.read_bytes()).hexdigest()
    tamper_conn = sqlite3.connect(copied_user_db)
    try:
        tamper_conn.execute("PRAGMA wal_autocheckpoint = 0")
        tamper_conn.execute("CREATE TABLE post_receipt_tamper (value TEXT)")
        tamper_conn.commit()
        wal_path = Path(f"{copied_user_db}-wal")
        assert wal_path.stat().st_size > 0
        assert hashlib.sha256(copied_user_db.read_bytes()).hexdigest() == main_hash
        _block_migration_sql(monkeypatch)

        conn = sqlite3.connect(db_path)
        try:
            with pytest.raises(MigrationError, match="unbound SQLite sidecar"):
                migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
            assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
            assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
        finally:
            conn.close()
    finally:
        tamper_conn.close()


def test_migration_rejects_linked_sqlite_sidecar(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")
    Path(f"{manifest.with_name('user.db')}-wal").symlink_to(db_path)
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="unbound SQLite sidecar"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
    finally:
        conn.close()


def test_migration_rejects_unbound_extra_backup_artifact(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = workspace_env["archive_root"] / "user.db"
    _create_user_v3(db_path)
    manifest = _verified_backup_manifest(tmp_path / "backup", profile="user_overlays")
    manifest.with_name("unexpected.txt").write_text("not in the verified file set", encoding="utf-8")
    _block_migration_sql(monkeypatch)

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(MigrationError, match="closed artifact inventory"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'user_settings'").fetchone()
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
        ("receipt", _tamper_receipt, "attestation MAC is invalid"),
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


def test_backup_artifact_inventory_scan_is_cached_across_both_durable_tier_migrations(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One activation authenticates the immutable backup inventory once.

    Without caching, ``migrate_archive_tier`` scans and SHA-256's the whole
    backup artifact tree (including the blob inventory) four times for one
    source+user activation: once before ``BEGIN`` and once inside the
    transaction, for each of the two durable tiers.  The backup directory is
    asserted immutable for the run, so those are pure duplication -- observed
    as 100+ GiB of repeat reads against a 35 GiB index during the 2026-07-13
    v35->v36 cutover.  This proves the expensive scan (``_backup_artifact_inventory``)
    now runs exactly once and both tier migrations still succeed off the
    shared, cached result.
    """
    archive_root_path = workspace_env["archive_root"]
    source_path = archive_root_path / "source.db"
    user_path = archive_root_path / "user.db"
    _create_source_v1(source_path)
    _create_user_v3(user_path)
    # Default "rebuildable_cache_exclude" profile includes source+user+embeddings.
    manifest = _verified_backup_manifest(tmp_path / "backup")

    scan_calls: list[Path] = []
    original_scan = migration_runner._backup_artifact_inventory

    def counting_scan(backup_root: Path) -> list[dict[str, object]]:
        scan_calls.append(backup_root)
        return original_scan(backup_root)

    monkeypatch.setattr(migration_runner, "_backup_artifact_inventory", counting_scan)

    with sqlite3.connect(source_path) as conn:
        source_result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
    with sqlite3.connect(user_path) as conn:
        user_result = migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)

    assert source_result.to_version == SOURCE_SCHEMA_VERSION
    assert user_result.to_version == USER_SCHEMA_VERSION
    assert len(scan_calls) == 1


def test_cached_backup_inventory_still_detects_tamper_between_tier_migrations(
    workspace_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache hit can only skip work, never launder a mutated backup.

    The source-tier migration populates the cache; tampering the backup
    afterward must still be caught by the user-tier migration, proving the
    cheap stat-signature check actually invalidates on real mutation instead
    of silently trusting a stale scan.
    """
    archive_root_path = workspace_env["archive_root"]
    source_path = archive_root_path / "source.db"
    user_path = archive_root_path / "user.db"
    _create_source_v1(source_path)
    _create_user_v3(user_path)
    manifest = _verified_backup_manifest(tmp_path / "backup")

    with sqlite3.connect(source_path) as conn:
        migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)

    _tamper_backup_tier(manifest)
    _block_migration_sql(monkeypatch)

    with sqlite3.connect(user_path) as conn:
        with pytest.raises(MigrationError, match="tier artifact .* mismatch"):
            migrate_archive_tier(conn, ArchiveTier.USER, backup_manifest=manifest)


def test_backup_inventory_cache_signature_rejects_stale_entry_after_size_change(tmp_path: Path) -> None:
    """Unit-level proof that a changed artifact invalidates the cached scan."""
    backup_root = tmp_path / "backup-cache"
    backup_root.mkdir()
    artifact = backup_root / "example.db"
    artifact.write_bytes(b"original-bytes")
    migration_runner._backup_artifact_inventory_cache.clear()

    first = migration_runner._cached_backup_artifact_inventory(backup_root)
    first_hash = next(item["sha256"] for item in first if item["path"] == "example.db")

    artifact.write_bytes(b"mutated-bytes-are-longer")
    second = migration_runner._cached_backup_artifact_inventory(backup_root)
    second_hash = next(item["sha256"] for item in second if item["path"] == "example.db")

    assert first_hash != second_hash
    assert second_hash == hashlib.sha256(b"mutated-bytes-are-longer").hexdigest()
