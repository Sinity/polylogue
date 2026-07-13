"""Clone-only contracts for the v35 archive schema fast-forward."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from devtools.archive_schema_fast_forward import (
    SchemaFastForwardError,
    _parser,
    _promote_index_generation,
    _require_receipt_identity,
    atomic_promote,
    beads_evidence,
    fast_forward_embeddings_clone,
    fast_forward_index_clone,
    reuse_index_clone,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
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
        conn.execute("PRAGMA user_version = 35")
        conn.execute(
            "INSERT INTO sessions(native_id, origin, content_hash) VALUES (?, ?, ?)",
            ("parent", "chatgpt-export", b"p" * 32),
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


def test_index_clone_copy_forward_preserves_fk_graph_ddl_and_rows(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    clone = tmp_path / "index-v36.db"
    _create_v35_index(source, wal_mode=True)
    before_fk = _fk_declarations(source)
    _clear_wal_sidecars(source)

    result = fast_forward_index_clone(source, clone)

    assert result.source.user_version == 35
    assert result.clone.user_version == 36
    assert result.source.table_counts == result.clone.table_counts
    assert not any(Path(f"{clone}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))
    assert _fk_declarations(clone) == before_fk
    with sqlite3.connect(clone) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        sessions_sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='sessions'").fetchone()[
            0
        ]
        links_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='session_links'"
        ).fetchone()[0]
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
    assert result.clone.user_version == 2
    assert not any(Path(f"{clone}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))
    with sqlite3.connect(clone) as conn:
        loaded, error = try_load_sqlite_vec(conn)
        assert loaded, error
        assert conn.execute("SELECT COUNT(*) FROM message_embeddings_meta").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM embedding_status").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM embedding_failures").fetchone()[0] == 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2


def test_reuse_index_clone_reproves_v36_staging_before_atomic_move(tmp_path: Path) -> None:
    source = tmp_path / "index-v35.db"
    staged = tmp_path / "completed-index-v36.db"
    destination = tmp_path / "new-run" / "index.db"
    _create_v35_index(source)
    expected = fast_forward_index_clone(source, staged)

    result = reuse_index_clone(source, staged, destination)

    assert result.source == expected.source
    assert result.clone == expected.clone
    assert not staged.exists()
    assert destination.exists()
    assert not any(Path(f"{destination}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal"))


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
