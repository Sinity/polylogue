from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.storage.source_sessions import session_ids_for_source_path, session_ids_for_source_paths
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def test_source_session_lookup_reads_archive_file_set(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    source_path = tmp_path / "sessions" / "current.jsonl"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-current", "codex-session", "current-native", str(source_path), 0, b"a" * 32, 10, 1),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("current-native", "codex-session", "raw-current", 0, b"b" * 32, 1, 1),
        )
        assert session_ids_for_source_path(conn, source_path) == ["codex-session:current-native"]
        assert session_ids_for_source_paths(conn, [source_path]) == {source_path: ["codex-session:current-native"]}


def test_source_session_lookup_requires_source_tier(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "current.jsonl"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    with sqlite3.connect(index_db) as conn:
        assert session_ids_for_source_path(conn, source_path) == []


def test_embedding_path_scope_uses_archive_source_tier_when_index_is_generation(tmp_path: Path) -> None:
    """Watcher path scopes survive an active index generation outside the archive root.

    Anti-vacuity: deriving ``source.db`` from the generation directory returns
    no IDs, and the embedding callback treats a changed source as already done.
    """

    from polylogue.operations.embedding_derivation import embedding_session_ids_for_paths

    root = tmp_path / "archive"
    index_db = root / ".index-generations" / "gen-1" / "index.db"
    source_db = root / "source.db"
    source_path = root / "sessions" / "current.jsonl"
    index_db.parent.mkdir(parents=True)
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, source_index,
                blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-generation", "codex-session", "generation-native", str(source_path), 0, b"g" * 32, 10, 1),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                native_id, origin, raw_id, message_count, content_hash, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("generation-native", "codex-session", "raw-generation", 0, b"h" * 32, 1, 1),
        )

    assert embedding_session_ids_for_paths(index_db, archive_root=root, paths=(source_path,)) == (
        "codex-session:generation-native",
    )
