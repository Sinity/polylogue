"""Clone-only contracts for the v35 archive schema fast-forward."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

import devtools.archive_schema_fast_forward as fast_forward_module
from devtools.archive_schema_fast_forward import (
    INDEX_CLONE_CHECKPOINT_SCHEMA,
    SchemaFastForwardError,
    _parser,
    _promote_index_generation,
    _require_receipt_identity,
    activate_prepared_forward,
    atomic_promote,
    beads_evidence,
    fast_forward_embeddings_clone,
    fast_forward_index_clone,
    plan_clone_forward,
    reuse_index_clone,
    write_index_clone_checkpoint,
)
from polylogue.storage.index_generation import ActiveWriterLease, RebuildLeaseUnavailableError
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDINGS_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec


def _clear_wal_sidecars(path: Path) -> None:
    """Leave a WAL-mode fixture stable enough for clone preflight."""
    with sqlite3.connect(path) as conn:
        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        assert checkpoint is not None and checkpoint[0] == 0
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _create_v35_index(path: Path, *, wal_mode: bool = False) -> None:
    # The fixture is an actual v35-shaped Origin CHECK schema, not simply a
    # lowered PRAGMA version on current DDL.
    legacy_ddl = INDEX_DDL.replace(", 'beads-issue'", "")
    with sqlite3.connect(path) as conn:
        if wal_mode:
            conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(legacy_ddl)
        conn.executescript(
            """
            CREATE TABLE session_runs (
                run_ref TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE
            ) STRICT;
            CREATE TABLE session_observed_events (
                event_ref TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                run_ref TEXT NOT NULL
            ) STRICT;
            CREATE TABLE session_context_snapshots (
                snapshot_ref TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                run_ref TEXT NOT NULL
            ) STRICT;
            """
        )
        conn.execute("PRAGMA user_version = 35")
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("parent", "chatgpt-export", b"p" * 32),
        )
        conn.execute(
            "INSERT INTO session_runs(run_ref, session_id) VALUES (?, ?)",
            ("run:parent", "chatgpt-export:parent"),
        )
        conn.execute(
            "INSERT INTO session_observed_events(event_ref, session_id, run_ref) VALUES (?, ?, ?)",
            ("event:parent", "chatgpt-export:parent", "run:parent"),
        )
        conn.execute(
            "INSERT INTO session_context_snapshots(snapshot_ref, session_id, run_ref) VALUES (?, ?, ?)",
            ("snapshot:parent", "chatgpt-export:parent", "run:parent"),
        )
        conn.execute(
            "INSERT INTO sessions(native_id, origin, parent_session_id, content_hash) VALUES (?, ?, ?, ?)",
            ("child", "chatgpt-export", "chatgpt-export:parent", b"c" * 32),
        )
        conn.execute(
            """INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?)""",
            ("chatgpt-export:child", "chatgpt-export", "parent", "resume", 1),
        )
        conn.execute(
            """INSERT INTO session_events(
                session_id, position, event_type, summary
            ) VALUES (?, ?, ?, ?)""",
            ("chatgpt-export:child", 0, "fixture", "keeps inbound FK declarations honest"),
        )
        conn.commit()
    if wal_mode:
        _clear_wal_sidecars(path)


def _fk_declarations(path: Path) -> dict[str, tuple[tuple[object, ...], ...]]:
    with sqlite3.connect(path) as conn:
        tables = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return {
            table: tuple(tuple(row) for row in conn.execute(f'PRAGMA foreign_key_list("{table}")')) for table in tables
        }


def _create_v1_embeddings(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            pytest.skip(f"sqlite-vec unavailable: {error}")
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
        conn.execute("DROP INDEX idx_embedding_failures_active")
        conn.execute("DROP TABLE embedding_failures")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    _clear_wal_sidecars(path)


def _build_v35_archive_root(root: Path) -> None:
    """Build a full 5-tier archive at the exact versions ``prepare`` expects."""
    root.mkdir(parents=True, exist_ok=True)
    initialize_archive_database(root / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(root / "user.db", ArchiveTier.USER)
    initialize_archive_database(root / "ops.db", ArchiveTier.OPS)
    _create_v35_index(root / "index.db")
    _create_v1_embeddings(root / "embeddings.db")


def test_index_clone_copy_forward_preserves_fk_graph_ddl_and_rows(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    clone = tmp_path / "index-v36.db"
    _create_v35_index(source, wal_mode=True)
    before_fk = _fk_declarations(source)
    _clear_wal_sidecars(source)

    result = fast_forward_index_clone(source, clone)

    assert result.source.user_version == 35
    assert result.clone.user_version == INDEX_SCHEMA_VERSION
    assert {
        name: count
        for name, count in result.source.table_counts.items()
        if name not in {"session_runs", "session_observed_events", "session_context_snapshots"}
    } == result.clone.table_counts
    assert not any(Path(f"{clone}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))
    assert _fk_declarations(clone) == {
        name: declarations
        for name, declarations in before_fk.items()
        if name not in {"session_runs", "session_observed_events", "session_context_snapshots"}
    }
    with sqlite3.connect(clone) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        sessions_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()[
            0
        ]
        links_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='session_links'"
        ).fetchone()[0]
        assert {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "('session_runs', 'session_observed_events', 'session_context_snapshots')"
            )
        } == set()
    assert "beads-issue" in sessions_sql
    assert "beads-issue" in links_sql


def test_beads_origin_or_path_fails_closed_before_clone(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    _create_v35_index(source)
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            """INSERT INTO session_links(
                src_session_id, dst_origin, dst_native_id, link_type, observed_at_ms
            ) VALUES (?, ?, ?, ?, ?)""",
            ("chatgpt-export:child", "beads-issue", "bead", "resume", 2),
        )
        conn.commit()

    assert beads_evidence(source) == {"session_links.dst_origin": 1}

    with pytest.raises(SchemaFastForwardError, match="Beads evidence"):
        fast_forward_index_clone(source, tmp_path / "clone.db")


def test_beads_census_rejects_wal_before_it_can_create_a_shm_sidecar(tmp_path: Path) -> None:
    """A sidecar-bearing archive must fail before a read-only census opens it.

    ``mode=ro`` alone creates ``-shm`` while a writer's WAL exists.  This is
    the exact state that used to make prepare reject sidecars it had just
    introduced itself.  Keep the writer open, remove only its visible SHM
    pathname, then prove the census neither reconnects nor recreates it.
    """
    source = tmp_path / "index-v35.db"
    writer = sqlite3.connect(source)
    try:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("CREATE TABLE retained (value TEXT)")
        writer.execute("INSERT INTO retained VALUES ('fixture')")
        writer.commit()
        wal = Path(f"{source}-wal")
        shm = Path(f"{source}-shm")
        assert wal.exists()
        shm.unlink(missing_ok=True)
        assert not shm.exists()

        with pytest.raises(SchemaFastForwardError, match="SQLite sidecars"):
            beads_evidence(source)

        assert wal.exists()
        assert not shm.exists()
    finally:
        writer.close()


def test_embedding_clone_preserves_vectors_and_installs_failure_lifecycle(tmp_path: Path) -> None:
    source = tmp_path / "embeddings-v1.db"
    clone = tmp_path / "embeddings-v2.db"
    with sqlite3.connect(source) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            pytest.skip(f"sqlite-vec unavailable: {error}")
        initialize_archive_tier(conn, ArchiveTier.EMBEDDINGS)
        conn.execute("DROP INDEX idx_embedding_failures_active")
        conn.execute("DROP TABLE embedding_failures")
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO message_embeddings_meta(message_id, model, dimension, content_hash) VALUES (?, ?, ?, ?)",
            ("message:1", "fixture", 1024, b"m" * 32),
        )
        conn.execute("INSERT INTO embedding_status(session_id, origin) VALUES (?, ?)", ("session:1", "chatgpt-export"))
        conn.commit()
    _clear_wal_sidecars(source)

    result = fast_forward_embeddings_clone(source, clone)

    assert result.source.user_version == 1
    assert result.clone.user_version == EMBEDDINGS_SCHEMA_VERSION
    assert not any(Path(f"{clone}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))
    with sqlite3.connect(clone) as conn:
        loaded, error = try_load_sqlite_vec(conn)
        assert loaded, error
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM embedding_status").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM embedding_failures").fetchone()[0] == 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == EMBEDDINGS_SCHEMA_VERSION


def test_reuse_index_clone_reproves_v36_staging_before_atomic_move(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    expected = fast_forward_index_clone(source, staged)

    result = reuse_index_clone(source, staged, destination)

    assert result.source == expected.source
    assert result.clone == expected.clone
    assert staged.exists()
    assert destination.exists()
    assert not any(Path(f"{destination}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


def test_reuse_index_clone_copies_across_subvolumes_before_local_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    archive.mkdir()
    staging.mkdir()
    source = archive / "index-v35.db"
    completed = archive / "completed-index-v36.db"
    destination = staging / "new-run" / "index.db"
    _create_v35_index(source)
    fast_forward_index_clone(source, completed)
    real_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def reject_cross_device_replace(
        source_path: str | os.PathLike[str], destination_path: str | os.PathLike[str]
    ) -> None:
        source_item = Path(source_path)
        destination_item = Path(destination_path)
        replace_calls.append((source_item, destination_item))
        if source_item.parent != destination_item.parent:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(os, "replace", reject_cross_device_replace)

    reuse_index_clone(source, completed, destination)

    assert completed.exists()
    assert destination.exists()
    assert all(source_item.parent == destination_item.parent for source_item, destination_item in replace_calls)
    assert not list(destination.parent.glob(f".{destination.name}.schema-forward-*.tmp"))


def test_atomic_promote_keeps_a_named_rollback(tmp_path: Path) -> None:
    active = tmp_path / "index.db"
    clone = tmp_path / "prepared-index.db"
    rollback = tmp_path / "index.db.rollback"
    active.write_text("old", encoding="utf-8")
    clone.write_text("new", encoding="utf-8")

    promoted = atomic_promote(clone, active, rollback)

    assert active.read_text(encoding="utf-8") == "new"
    assert rollback.read_text(encoding="utf-8") == "old"
    assert promoted["rollback"] == str(rollback)


def test_atomic_promote_copies_across_subvolumes_before_local_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion must not issue a cross-device rename for staging artifacts."""
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    archive.mkdir()
    staging.mkdir()
    active = archive / "embeddings.db"
    clone = staging / "prepared-embeddings.db"
    rollback = staging / "rollback" / "embeddings.db"
    active.write_text("old", encoding="utf-8")
    clone.write_text("new", encoding="utf-8")
    real_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def reject_cross_device_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        replace_calls.append((source_path, destination_path))
        if source_path.parent != destination_path.parent:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", reject_cross_device_replace)

    atomic_promote(clone, active, rollback)

    assert active.read_text(encoding="utf-8") == "new"
    assert rollback.read_text(encoding="utf-8") == "old"
    assert not clone.exists()
    assert all(source.parent == destination.parent for source, destination in replace_calls)


def test_activation_failure_restores_durable_snapshots_without_cross_device_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-migration snapshots are rollback inputs, never promoted entries."""
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    archive.mkdir()
    staging.mkdir()
    source = archive / "source.db"
    user = archive / "user.db"
    for database in (source, user, archive / "index.db", archive / "embeddings.db", archive / "ops.db"):
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE retained (value TEXT)")
            conn.execute("INSERT INTO retained VALUES ('original')")
            conn.commit()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "polylogue.archive-schema-fast-forward.v1",
                "status": "prepared",
                "archive_root": str(archive),
                "staging_root": str(staging),
                "backup_manifest": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("devtools.archive_schema_fast_forward._require_service_stopped", lambda _service: None)
    monkeypatch.setattr("devtools.archive_schema_fast_forward._require_receipt_identity", lambda *_args: None)
    monkeypatch.setattr("devtools.archive_schema_fast_forward.require_no_beads_evidence", lambda *_paths: None)

    def fail_after_source_mutation(conn: sqlite3.Connection, tier: ArchiveTier, **_kwargs: object) -> None:
        assert tier is ArchiveTier.SOURCE
        conn.execute("CREATE TABLE migration_marker (value TEXT)")
        conn.commit()
        raise RuntimeError("synthetic source migration failure")

    monkeypatch.setattr("devtools.archive_schema_fast_forward.migrate_archive_tier", fail_after_source_mutation)
    real_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def reject_cross_device_replace(source_path: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_item = Path(source_path)
        destination_item = Path(destination)
        replace_calls.append((source_item, destination_item))
        if source_item.parent != destination_item.parent:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source_path, destination)

    monkeypatch.setattr(os, "replace", reject_cross_device_replace)

    with pytest.raises(RuntimeError, match="synthetic source migration failure"):
        activate_prepared_forward(receipt_path=receipt, backup_manifest=manifest)

    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT value FROM retained").fetchall() == [("original",)]
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='migration_marker'").fetchone() is None
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "rolled_back"
    assert all(source_item.parent == destination_item.parent for source_item, destination_item in replace_calls)


def test_activation_final_evidence_failure_is_rolled_back_inside_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-promotion evidence must not strand an unrecorded activation."""
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    archive.mkdir()
    staging.mkdir()
    source = archive / "source.db"
    user = archive / "user.db"
    for database in (source, user, archive / "index.db", archive / "embeddings.db", archive / "ops.db"):
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE retained (value TEXT)")
            conn.execute("INSERT INTO retained VALUES ('original')")
            conn.commit()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "polylogue.archive-schema-fast-forward.v1",
                "status": "prepared",
                "archive_root": str(archive),
                "staging_root": str(staging),
                "backup_manifest": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("devtools.archive_schema_fast_forward._require_service_stopped", lambda _service: None)
    monkeypatch.setattr("devtools.archive_schema_fast_forward._require_receipt_identity", lambda *_args: None)
    monkeypatch.setattr("devtools.archive_schema_fast_forward.require_no_beads_evidence", lambda *_paths: None)
    monkeypatch.setattr("devtools.archive_schema_fast_forward.migrate_archive_tier", lambda *_args, **_kwargs: None)
    finalized: list[Path] = []
    monkeypatch.setattr(
        "devtools.archive_schema_fast_forward._finalize_clone_database", lambda path: finalized.append(path)
    )
    monkeypatch.setattr(
        "devtools.archive_schema_fast_forward._promote_index_generation",
        lambda *_args: {"kind": "file", "active": str(archive / "index.db"), "rollback": "unused"},
    )
    monkeypatch.setattr(
        "devtools.archive_schema_fast_forward.atomic_promote",
        lambda _clone, active, _rollback: {"kind": "file", "active": str(active), "rollback": "unused"},
    )
    monkeypatch.setattr("devtools.archive_schema_fast_forward._restore_promoted", lambda _item: None)
    monkeypatch.setattr(
        "devtools.archive_schema_fast_forward._database_evidence",
        lambda _path: (_ for _ in ()).throw(RuntimeError("synthetic final evidence failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic final evidence failure"):
        activate_prepared_forward(receipt_path=receipt, backup_manifest=manifest)

    assert finalized == [source, user]
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "rolled_back"
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT value FROM retained").fetchall() == [("original",)]


def test_index_generation_promotion_preserves_active_symlink_protocol(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    generations = archive / ".index-generations"
    old_target = generations / "gen-v35" / "index.db"
    old_target.parent.mkdir(parents=True)
    old_target.write_text("v35", encoding="utf-8")
    active = archive / "index.db"
    active.symlink_to(old_target)
    clone = tmp_path / "prepared-index.db"
    clone.write_text("v36", encoding="utf-8")

    promoted = _promote_index_generation(clone, active)

    assert active.is_symlink()
    assert active.resolve().read_text(encoding="utf-8") == "v36"
    assert old_target.read_text(encoding="utf-8") == "v35"
    assert Path(promoted["rollback"]) == old_target
    assert Path(promoted["generation_target"]).is_relative_to(generations)


def test_index_generation_promotion_copies_staging_clone_before_local_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged index never crosses a device boundary through ``os.replace``."""
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    generations = archive / ".index-generations"
    old_target = generations / "gen-v35" / "index.db"
    old_target.parent.mkdir(parents=True)
    staging.mkdir()
    old_target.write_text("v35", encoding="utf-8")
    active = archive / "index.db"
    active.symlink_to(old_target)
    clone = staging / "prepared-index.db"
    clone.write_text("v36", encoding="utf-8")
    real_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def reject_cross_device_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        replace_calls.append((source_path, destination_path))
        if source_path.parent != destination_path.parent:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", reject_cross_device_replace)

    promoted = _promote_index_generation(clone, active)

    assert active.is_symlink()
    assert active.resolve().read_text(encoding="utf-8") == "v36"
    assert old_target.read_text(encoding="utf-8") == "v35"
    assert Path(promoted["rollback"]) == old_target
    assert not clone.exists()
    assert all(source.parent == destination.parent for source, destination in replace_calls)


def test_activation_receipt_rejects_any_active_tier_identity_drift(tmp_path: Path) -> None:
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE retained (value TEXT)")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    from devtools.archive_schema_fast_forward import _database_evidence

    evidence = _database_evidence(database)
    receipt: dict[str, object] = {
        "source": {
            "path": str(database),
            "user_version": 7,
            "sha256": evidence.sha256,
            "size_bytes": database.stat().st_size,
        }
    }
    _require_receipt_identity(receipt, "source", database)

    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA user_version = 8")
        conn.commit()
    with pytest.raises(SchemaFastForwardError, match="changed since clone proof"):
        _require_receipt_identity(receipt, "source", database)


def test_operator_cli_exposes_only_prepare_and_existing_manifest_activation() -> None:
    parser = _parser()
    prepare = parser.parse_args(
        [
            "prepare",
            "--archive-root",
            "/tmp/archive",
            "--staging-root",
            "/tmp/staging",
            "--receipt",
            "/tmp/receipt.json",
            "--backup-manifest",
            "/tmp/manifest.json",
            "--reuse-index-clone",
            "/tmp/completed-index.db",
        ]
    )
    activate = parser.parse_args(
        [
            "activate",
            "--receipt",
            "/tmp/receipt.json",
            "--backup-manifest",
            "/tmp/manifest.json",
        ]
    )

    assert str(prepare.reuse_index_clone) == "/tmp/completed-index.db"
    assert activate.command == "activate"
    assert str(activate.backup_manifest) == "/tmp/manifest.json"
    assert activate.service == "polylogued.service"


# --- polylogue-qg6x: resumable schema-forward clone proofs -----------------


def test_write_index_clone_checkpoint_records_a_self_checking_receipt(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    clone = tmp_path / "index-v36.db"
    _create_v35_index(source)

    result = fast_forward_index_clone(source, clone)
    checkpoint = write_index_clone_checkpoint(result, clone_path=clone)

    checkpoint_path = clone.with_name(f"{clone.name}.clone-checkpoint.json")
    assert checkpoint_path.exists()
    on_disk = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert on_disk == checkpoint
    assert checkpoint["schema"] == INDEX_CLONE_CHECKPOINT_SCHEMA
    assert on_disk["source"]["user_version"] == 35
    assert on_disk["clone"]["user_version"] == INDEX_SCHEMA_VERSION
    assert checkpoint["quick_check"] == ["ok"]
    assert checkpoint["foreign_key_check"] == []
    assert checkpoint["foreign_key_declarations_preserved"] is True
    assert checkpoint["clone_beads_findings"] == {}
    assert isinstance(checkpoint["canonical_ddl_sha256"], str) and len(checkpoint["canonical_ddl_sha256"]) == 64
    # Self-checking: the receipt hash covers every other field.
    body = {key: value for key, value in checkpoint.items() if key != "receipt_sha256"}
    assert (
        checkpoint["receipt_sha256"]
        == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    )


def test_reuse_index_clone_with_valid_checkpoint_skips_clone_census_fk_and_quick_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    result = fast_forward_index_clone(source, staged)
    write_index_clone_checkpoint(result, clone_path=staged)

    evidence_calls: list[Path] = []
    original_evidence = fast_forward_module._database_evidence

    def tracking_evidence(path: Path) -> object:
        evidence_calls.append(Path(path))
        return original_evidence(path)

    monkeypatch.setattr(fast_forward_module, "_database_evidence", tracking_evidence)

    def fail_finalize(path: Path) -> None:
        pytest.fail("checkpoint reuse must not re-finalize the already-finalized clone")

    monkeypatch.setattr(fast_forward_module, "_finalize_clone_database", fail_finalize)

    reused = reuse_index_clone(source, staged, destination)

    assert reused.source == result.source
    assert reused.clone == result.clone
    assert reused.quick_check == ("ok",)
    assert destination.exists()
    # The source is always freshly evidenced (it can legitimately drift);
    # the clone's table census is trusted from the checkpoint instead of a
    # fresh (table-scanning) `_database_evidence` call.
    assert evidence_calls == [source]


def test_reuse_index_clone_ignores_checkpoint_when_source_has_drifted_since(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    result = fast_forward_index_clone(source, staged)
    write_index_clone_checkpoint(result, clone_path=staged)

    with sqlite3.connect(source) as conn:
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("drift", "chatgpt-export", b"d" * 32),
        )
        conn.commit()

    # Source drift invalidates the checkpoint (source identity mismatch), so
    # this falls back to a full reprove against the now-larger source -- and
    # correctly rejects the stale clone rather than silently trusting it.
    with pytest.raises(SchemaFastForwardError, match="structural row counts"):
        reuse_index_clone(source, staged, destination)


def test_reuse_index_clone_ignores_checkpoint_when_clone_bytes_tampered(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    result = fast_forward_index_clone(source, staged)
    write_index_clone_checkpoint(result, clone_path=staged)

    with sqlite3.connect(staged) as conn:
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("tampered", "chatgpt-export", b"t" * 32),
        )
        conn.commit()

    # The clone's bytes no longer match what the checkpoint recorded, so the
    # lightweight identity check discards the checkpoint and a full reprove
    # against the (unchanged) source correctly rejects the tampered clone.
    with pytest.raises(SchemaFastForwardError, match="structural row counts"):
        reuse_index_clone(source, staged, destination)


def test_reuse_index_clone_ignores_a_tampered_checkpoint_receipt(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    result = fast_forward_index_clone(source, staged)
    write_index_clone_checkpoint(result, clone_path=staged)

    checkpoint_path = staged.with_name(f"{staged.name}.clone-checkpoint.json")
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["quick_check"] = ["forged"]
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    # A checkpoint whose recorded body no longer matches its own hash cannot
    # be trusted, but that must not corrupt reuse of an otherwise-sound
    # clone: it just forces the (still-correct) full reprove path.
    reused = reuse_index_clone(source, staged, destination)
    assert reused.source == result.source
    assert reused.clone == result.clone
    assert reused.quick_check == ("ok",)


def test_reuse_index_clone_checkpoint_fast_path_still_rejects_clone_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    result = fast_forward_index_clone(source, staged)
    write_index_clone_checkpoint(result, clone_path=staged)

    # Keep the writer open: SQLite auto-checkpoints (and can remove) the WAL
    # sidecar once the last connection to a WAL-mode database closes, so the
    # sidecar must stay visible for the duration of this check -- matching
    # the existing beads-census WAL fixture pattern above.
    writer = sqlite3.connect(staged)
    try:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("CREATE TABLE scratch (value TEXT)")
        writer.commit()
        assert Path(f"{staged}-wal").exists()

        with pytest.raises(SchemaFastForwardError, match="SQLite sidecars"):
            reuse_index_clone(source, staged, destination)
    finally:
        writer.close()


# --- polylogue-b5l.1: writer-exclusive, crash-resumable rebuilds -----------


def test_plan_clone_forward_fails_before_first_write_when_writer_already_owns_archive(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _build_v35_archive_root(archive)
    staging = tmp_path / "staging"
    receipt = tmp_path / "receipt.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    writer = ActiveWriterLease(archive)
    writer.acquire()
    try:
        with pytest.raises(SchemaFastForwardError, match="another writer owns the archive"):
            plan_clone_forward(
                archive_root=archive,
                staging_root=staging,
                receipt_path=receipt,
                backup_manifest=manifest,
            )
    finally:
        writer.close()

    assert not receipt.exists()
    assert not staging.exists()


def test_plan_clone_forward_writes_index_clone_checkpoint_before_later_tier_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive"
    _build_v35_archive_root(archive)
    staging = tmp_path / "staging"
    receipt = tmp_path / "receipt.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fast_forward_module, "_require_service_stopped", lambda _service: None)

    payload = plan_clone_forward(
        archive_root=archive,
        staging_root=staging,
        receipt_path=receipt,
        backup_manifest=manifest,
    )

    run_root = Path(str(payload["staging_root"]))
    checkpoint_path = run_root / "index.db.clone-checkpoint.json"
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema"] == INDEX_CLONE_CHECKPOINT_SCHEMA
    assert checkpoint["quick_check"] == ["ok"]


def test_plan_clone_forward_reuse_flag_consumes_the_prior_runs_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive"
    _build_v35_archive_root(archive)
    staging = tmp_path / "staging"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(fast_forward_module, "_require_service_stopped", lambda _service: None)

    first_receipt = tmp_path / "receipt-1.json"
    first_payload = plan_clone_forward(
        archive_root=archive,
        staging_root=staging,
        receipt_path=first_receipt,
        backup_manifest=manifest,
    )
    first_run_root = Path(str(first_payload["staging_root"]))
    first_index_clone = first_run_root / "index.db"

    census_calls: list[Path] = []
    original_evidence = fast_forward_module._database_evidence

    def tracking_evidence(path: Path) -> object:
        census_calls.append(Path(path))
        return original_evidence(path)

    monkeypatch.setattr(fast_forward_module, "_database_evidence", tracking_evidence)

    second_receipt = tmp_path / "receipt-2.json"
    second_payload = plan_clone_forward(
        archive_root=archive,
        staging_root=staging,
        receipt_path=second_receipt,
        backup_manifest=manifest,
        reuse_index_clone_path=first_index_clone,
    )

    assert second_payload["reused_index_clone"] == str(first_index_clone)
    # This is the exact scenario the operator hit live on 2026-07-13: the
    # first attempt's clone survives (fully proven) and a retry consumes it
    # via --reuse-index-clone.  The retry must never re-census the *staged*
    # clone -- only the live archive's index.db (which can legitimately
    # drift) is freshly evidenced.
    assert first_index_clone not in census_calls
    assert (archive / "index.db") in census_calls


def test_activate_prepared_forward_holds_writer_exclusion_across_the_whole_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    archive.mkdir()
    staging.mkdir()
    source = archive / "source.db"
    user = archive / "user.db"
    for database in (source, user, archive / "index.db", archive / "embeddings.db", archive / "ops.db"):
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE retained (value TEXT)")
            conn.commit()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "polylogue.archive-schema-fast-forward.v1",
                "status": "prepared",
                "archive_root": str(archive),
                "staging_root": str(staging),
                "backup_manifest": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fast_forward_module, "_require_service_stopped", lambda _service: None)
    monkeypatch.setattr(fast_forward_module, "_require_receipt_identity", lambda *_args: None)
    monkeypatch.setattr(fast_forward_module, "require_no_beads_evidence", lambda *_paths: None)

    observed_conflict: list[bool] = []

    def observe_exclusion_then_fail(conn: sqlite3.Connection, tier: object, **_kwargs: object) -> None:
        del conn, tier
        writer = ActiveWriterLease(archive)
        try:
            writer.acquire()
        except RebuildLeaseUnavailableError:
            observed_conflict.append(True)
        else:
            observed_conflict.append(False)
            writer.close()
        raise RuntimeError("synthetic stop after observing exclusion")

    monkeypatch.setattr(fast_forward_module, "migrate_archive_tier", observe_exclusion_then_fail)

    with pytest.raises(RuntimeError, match="synthetic stop after observing exclusion"):
        activate_prepared_forward(receipt_path=receipt, backup_manifest=manifest)

    assert observed_conflict == [True]
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "rolled_back"


def test_activate_prepared_forward_fails_before_first_write_when_writer_already_owns_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    archive.mkdir()
    staging.mkdir()
    for database in (
        archive / "source.db",
        archive / "user.db",
        archive / "index.db",
        archive / "embeddings.db",
        archive / "ops.db",
    ):
        with sqlite3.connect(database) as conn:
            conn.execute("CREATE TABLE retained (value TEXT)")
            conn.execute("INSERT INTO retained VALUES ('original')")
            conn.commit()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "polylogue.archive-schema-fast-forward.v1",
                "status": "prepared",
                "archive_root": str(archive),
                "staging_root": str(staging),
                "backup_manifest": str(manifest),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fast_forward_module, "_require_service_stopped", lambda _service: None)
    monkeypatch.setattr(
        fast_forward_module,
        "migrate_archive_tier",
        lambda *_args, **_kwargs: pytest.fail("must fail before migrate_archive_tier is ever called"),
    )

    writer = ActiveWriterLease(archive)
    writer.acquire()
    try:
        with pytest.raises(SchemaFastForwardError, match="another writer owns the archive"):
            activate_prepared_forward(receipt_path=receipt, backup_manifest=manifest)
    finally:
        writer.close()

    # Nothing was attempted -- the receipt is untouched, not marked rolled_back.
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "prepared"
    with sqlite3.connect(archive / "source.db") as conn:
        assert conn.execute("SELECT value FROM retained").fetchall() == [("original",)]
