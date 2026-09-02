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
from tests.infra.durable_schema_reset import reset_source_fixture_to_version


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


def _tamper_receipt_attestation(manifest: Path) -> None:
    """Alter a receipt field the MAC covers while leaving the verdict valid.

    _tamper_receipt flips the verdict, which the status check rejects before
    the MAC is ever verified, so the MAC path needs its own case.
    """
    receipt = manifest.with_name("verification-receipt.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["verified_at"] = "1999-01-01T00:00:00Z"
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


def _reset_source_fixture_to_version(conn: sqlite3.Connection, version: int) -> None:
    reset_source_fixture_to_version(conn, version)


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
        assert result.applied_versions == (4, 5, 6, 7, 8, 9, 10, 11)
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
        assert result.to_version == USER_SCHEMA_VERSION == 11
        assert result.applied_versions == (6, 7, 8, 9, 10, 11)
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


def _restore_source_pre_v30_raw_artifact_indexes(conn: sqlite3.Connection) -> None:
    """Restore the raw-artifact index shape that existed before migration 030."""
    conn.executescript(
        """
        DROP INDEX IF EXISTS idx_raw_artifacts_failure_identity;
        DROP INDEX IF EXISTS idx_raw_artifacts_source_identity;
        CREATE UNIQUE INDEX idx_raw_artifacts_source_identity
        ON raw_artifacts(origin, source_path, source_index);
        """
    )


# Oldest source version from which the numbered migrations still reach
# canonical-DDL parity. Migrations 009, 021 and 027 rebuild raw_sessions and
# raw_hook_events with enum-membership CHECK constraints that the canonical
# source DDL no longer carries -- vocabulary is validated at the write boundary
# (#4495) -- so a tier replayed across any of them diverges from a fresh
# bootstrap by those CHECK definitions. 027 is the newest of the three.
_SOURCE_DDL_PARITY_FLOOR = 27


def _create_source_at_ddl_parity_floor(path: Path) -> None:
    """Build an empty source tier at ``_SOURCE_DDL_PARITY_FLOOR``.

    Derived from canonical DDL -- the production bootstrap route -- then
    stripped of every object a migration above that version introduces.
    """
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SOURCE_DDL)
        _restore_source_pre_v30_raw_artifact_indexes(conn)
        _reset_source_fixture_to_version(conn, _SOURCE_DDL_PARITY_FLOOR)
        conn.execute(f"PRAGMA user_version = {_SOURCE_DDL_PARITY_FLOOR}")
        conn.commit()
    finally:
        conn.close()


def _create_source_v29_raw_failure_fixture(path: Path, *, archive_root: Path) -> tuple[str, str, str]:
    """Build the exact pre-v30 source shape used by migration 030."""
    current_indexes = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_source_identity
ON raw_artifacts(origin, source_path, source_index)
WHERE artifact_kind NOT IN (
    'deferred_hot_jsonl_capture',
    'deferred_claude_code_partial_jsonl',
    'deferred_cas_frontier',
    'deferred_codex_cas_frontier',
    'terminal_corrupt_input',
    'terminal_superseded_deferred_cas_frontier',
    'terminal_unknown_json_decode',
    'terminal_unknown_export_no_session',
    'terminal_unsupported_shape'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_failure_identity
ON raw_artifacts(raw_id, origin, source_path, source_index)
WHERE artifact_kind IN (
    'deferred_hot_jsonl_capture',
    'deferred_claude_code_partial_jsonl',
    'deferred_cas_frontier',
    'deferred_codex_cas_frontier',
    'terminal_corrupt_input',
    'terminal_superseded_deferred_cas_frontier',
    'terminal_unknown_json_decode',
    'terminal_unknown_export_no_session',
    'terminal_unsupported_shape'
);
"""
    legacy_index = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_artifacts_source_identity
ON raw_artifacts(origin, source_path, source_index);
"""
    v29_ddl = SOURCE_DDL.replace(current_indexes, legacy_index)
    assert v29_ddl != SOURCE_DDL
    path.unlink(missing_ok=True)
    blob_store = BlobStore(archive_root / "blob")
    ordinary_blob, ordinary_size = blob_store.write_from_bytes(b"ordinary-v29")
    failure_a_blob, failure_a_size = blob_store.write_from_bytes(b"failure-a-v29")
    failure_b_blob, failure_b_size = blob_store.write_from_bytes(b"failure-b-v29")
    conn = sqlite3.connect(path)
    try:
        conn.executescript(v29_ddl)
        _reset_source_fixture_to_version(conn, 29)
        conn.execute("PRAGMA user_version = 29")
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, 'codex-session', ?, ?, ?, ?, ?)
            """,
            [
                ("raw-v29-ordinary", "/v29/ordinary.json", 0, bytes.fromhex(ordinary_blob), ordinary_size, 1),
                ("raw-v29-failure-a", "/v29/failure.jsonl", 0, bytes.fromhex(failure_a_blob), failure_a_size, 2),
                ("raw-v29-failure-b", "/v29/failure.jsonl", 1, bytes.fromhex(failure_b_blob), failure_b_size, 3),
            ],
        )
        conn.execute(
            """
            INSERT INTO raw_artifacts (
                artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                support_status, classification_reason, first_observed_at_ms, last_observed_at_ms
            ) VALUES ('artifact-v29-ordinary', 'raw-v29-ordinary', 'codex-session',
                      '/v29/ordinary.json', 0, 'session_export', 'supported_parseable',
                      'ordinary-v29', 1, 1)
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_artifacts (
                artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                support_status, classification_reason, first_observed_at_ms, last_observed_at_ms
            ) VALUES (?, ?, 'codex-session', '/v29/failure.jsonl', ?,
                      'deferred_cas_frontier', 'partial_decode', 'deferred-v29', ?, ?)
            """,
            [
                ("artifact-v29-failure-a", "raw-v29-failure-a", 0, 2, 2),
                ("artifact-v29-failure-b", "raw-v29-failure-b", 1, 3, 3),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return ordinary_blob, failure_a_blob, failure_b_blob


def test_source_tier_v29_migration_030_splits_raw_artifact_indexes(
    workspace_env: dict[str, Path],
    tmp_path: Path,
) -> None:
    """The source-v30 index split preserves ordinary and retained-raw evidence."""
    db_path = workspace_env["archive_root"] / "source.db"
    _create_source_v29_raw_failure_fixture(db_path, archive_root=workspace_env["archive_root"])
    manifest = _verified_backup_manifest(tmp_path / "backup-source-v29")

    with sqlite3.connect(db_path) as conn:
        # "Only migration 030" was true while 30 was the head version. The
        # runner enforces canonical-DDL parity against the CURRENT head, so a
        # pinned target cannot pass; assert that 030 ran and that the index
        # split it performs is intact, which is what this test is really for.
        result = migrate_archive_tier(conn, ArchiveTier.SOURCE, backup_manifest=manifest)
        assert result.from_version == 29
        assert 30 in result.applied_versions
        assert result.to_version == SOURCE_SCHEMA_VERSION
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION

        index_names = {str(row[1]) for row in conn.execute("PRAGMA index_list('raw_artifacts')")}
        assert {"idx_raw_artifacts_source_identity", "idx_raw_artifacts_failure_identity"} <= index_names
        index_columns = {
            name: tuple(str(column[2]) for column in conn.execute(f"PRAGMA index_info('{name}')"))
            for name in ("idx_raw_artifacts_source_identity", "idx_raw_artifacts_failure_identity")
        }
        assert index_columns["idx_raw_artifacts_source_identity"] == ("origin", "source_path", "source_index")
        assert index_columns["idx_raw_artifacts_failure_identity"] == (
            "raw_id",
            "origin",
            "source_path",
            "source_index",
        )
        assert {
            tuple(row)
            for row in conn.execute(
                """
                SELECT artifact_id, raw_id, source_path, source_index, artifact_kind
                FROM raw_artifacts
                ORDER BY artifact_id
                """
            )
        } == {
            (
                "artifact-v29-failure-a",
                "raw-v29-failure-a",
                "/v29/failure.jsonl",
                0,
                "deferred_cas_frontier",
            ),
            (
                "artifact-v29-failure-b",
                "raw-v29-failure-b",
                "/v29/failure.jsonl",
                1,
                "deferred_cas_frontier",
            ),
            (
                "artifact-v29-ordinary",
                "raw-v29-ordinary",
                "/v29/ordinary.json",
                0,
                "session_export",
            ),
        }

        conn.execute(
            """
            INSERT INTO raw_artifacts (
                artifact_id, raw_id, origin, source_path, source_index, artifact_kind,
                support_status, classification_reason, first_observed_at_ms, last_observed_at_ms
            ) VALUES ('artifact-v30-failure-same-coordinate', 'raw-v29-ordinary',
                      'codex-session', '/v29/failure.jsonl', 0, 'deferred_cas_frontier',
                      'partial_decode', 'new-v30', 4, 4)
            """
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE source_path = '/v29/failure.jsonl' AND source_index = 0"
        ).fetchone() == (2,)


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


def test_source_tier_fresh_bootstrap_accepts_claude_design_session(
    workspace_env: dict[str, Path],
) -> None:
    """A freshly bootstrapped source.db (``SOURCE_DDL``, no migration
    involved) already accepts ``claude-design-session`` --
    ``archive_tiers/source.py`` generates the origin CHECK from
    ``core.enums.Origin`` directly via ``check()``/``nullable_check()``, so
    fresh archives never had this bug. Hard-coding any of the five origin
    CHECKs in ``source.py`` back to a literal list that omits
    ``claude-design-session`` makes the INSERT below raise
    ``sqlite3.IntegrityError``.
    """
    db_path = workspace_env["archive_root"] / "source.db"
    with sqlite3.connect(db_path) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
            "VALUES ('fresh-design', 'claude-design-session', '/fresh.json', ?, 1, 1)",
            (b"f" * 32,),
        )
        conn.commit()
        assert conn.execute("SELECT origin FROM raw_sessions WHERE raw_id = 'fresh-design'").fetchone() == (
            "claude-design-session",
        )


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
        assert result.applied_versions == tuple(range(3, SOURCE_SCHEMA_VERSION + 1))
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == SOURCE_SCHEMA_VERSION
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pending_blob_refs'"
        ).fetchone()
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='gc_generations'").fetchone()
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='blob_publication_reservations'"
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO raw_hook_events (
                    hook_event_id, origin, source_path, event_type, payload_json, observed_at_ms
                ) VALUES ('invalid-origin', 'invalid-origin', '/fixture.json', 'PreToolUse', '{}', 1)
                """
            )
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
        ("receipt", _tamper_receipt, "receipt is not a successful verification"),
        ("receipt-attestation", _tamper_receipt_attestation, "attestation MAC is invalid"),
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


def test_initialize_database_does_not_apply_explicit_user_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    _create_user_v3(db_path)
    with pytest.raises(RuntimeError, match="explicit durable-tier migration"):
        initialize_archive_database(db_path, ArchiveTier.USER)
    conn = sqlite3.connect(db_path)
    try:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 3
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
    _create_source_at_ddl_parity_floor(source_path)
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
    _create_source_at_ddl_parity_floor(source_path)
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


def test_historical_source_projection_matches_every_supported_train_target(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    """Each admitted source train matches a genuine numbered migration chain.

    The migrated side begins at ``_SOURCE_DDL_PARITY_FLOOR`` and applies
    numbered SQL to each target. The projected side begins at canonical DDL --
    the production bootstrap route -- and removes later train riders, so the
    inventory comparison covers definitions as well as table/index/trigger/view
    names.

    Anti-vacuity: a migration that adds a column, index, table, trigger or view
    the canonical DDL does not carry (or canonical DDL that carries one no
    migration adds) makes the inventory diverge and this red. The companion
    test below proves the comparison itself is not inert.
    """
    db_path = workspace_env["archive_root"] / "source.db"

    for target_version in range(_SOURCE_DDL_PARITY_FLOOR, SOURCE_SCHEMA_VERSION + 1):
        db_path.unlink(missing_ok=True)
        _create_source_at_ddl_parity_floor(db_path)
        manifest = _verified_backup_manifest(tmp_path / f"source-projection-backup-v{target_version}")
        with sqlite3.connect(db_path) as migrated:
            result = migrate_archive_tier(
                migrated,
                ArchiveTier.SOURCE,
                backup_manifest=manifest,
                target_version=target_version,
            )
            assert result.to_version == target_version
            with sqlite3.connect(":memory:") as projected:
                projected.executescript(SOURCE_DDL)
                projected.execute(f"PRAGMA user_version = {SOURCE_SCHEMA_VERSION}")
                migration_runner._prepare_fresh_connection_for_target(projected, ArchiveTier.SOURCE, target_version)
                parity = migration_runner.prove_durable_fresh_ddl_parity(
                    ArchiveTier.SOURCE,
                    target_version,
                    migrated_connection=migrated,
                    fresh_connection=projected,
                    evidence_ref=f"test:source-projection-matrix:v{target_version}",
                )
        assert parity.matches, parity


def test_historical_source_projection_rejects_a_missing_replaced_index(
    workspace_env: dict[str, Path], tmp_path: Path
) -> None:
    """Anti-vacuity: a v28 projection without its replaced index is rejected."""
    target_version = 28
    db_path = workspace_env["archive_root"] / "source.db"
    db_path.unlink(missing_ok=True)
    _create_source_at_ddl_parity_floor(db_path)
    manifest = _verified_backup_manifest(tmp_path / "source-projection-corruption-backup")

    with sqlite3.connect(db_path) as migrated, sqlite3.connect(":memory:") as projected:
        migrate_archive_tier(
            migrated,
            ArchiveTier.SOURCE,
            backup_manifest=manifest,
            target_version=target_version,
        )
        projected.executescript(SOURCE_DDL)
        projected.execute(f"PRAGMA user_version = {SOURCE_SCHEMA_VERSION}")
        migration_runner._prepare_fresh_connection_for_target(projected, ArchiveTier.SOURCE, target_version)
        projected.execute("DROP INDEX idx_raw_artifacts_source_identity")
        parity = migration_runner.prove_durable_fresh_ddl_parity(
            ArchiveTier.SOURCE,
            target_version,
            migrated_connection=migrated,
            fresh_connection=projected,
            evidence_ref="test:source-projection-matrix:corruption",
        )

    assert parity.matches is False
    assert parity.unexpected_objects == ("index:idx_raw_artifacts_source_identity",)
