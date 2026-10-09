from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import compute_cancel
from polylogue.core.errors import ArchiveTierUnavailableError
from polylogue.operations import embedding_derivation
from polylogue.storage.source_sessions import session_ids_for_source_path, session_ids_for_source_paths
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture


def test_source_session_lookup_reads_archive_file_set(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    source_path = tmp_path / "sessions" / "current.jsonl"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(source_db)
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
        with pytest.raises(ArchiveTierUnavailableError) as refusal:
            session_ids_for_source_path(conn, source_path)
        assert refusal.value.tier == "source"
        assert not (tmp_path / "source.db").exists()


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
    initialize_runtime_source_fixture(source_db)
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


@pytest.fixture
def matching_source_scope(tmp_path: Path) -> tuple[Path, Path]:
    index_db = tmp_path / "index.db"
    initialize_archive_database(index_db, ArchiveTier.INDEX)
    initialize_runtime_source_fixture(tmp_path / "source.db")
    source_path = tmp_path / "current.jsonl"
    with sqlite3.connect(tmp_path / "source.db") as source:
        source.execute(
            "INSERT INTO raw_sessions (raw_id,origin,native_id,source_path,source_index,blob_hash,blob_size,acquired_at_ms) "
            "VALUES ('raw-scope','codex-session','scope',?,0,?,10,1)",
            (str(source_path), b"a" * 32),
        )
    with sqlite3.connect(index_db) as index:
        index.execute(
            "INSERT INTO sessions (native_id,origin,raw_id,message_count,content_hash,created_at_ms,updated_at_ms) "
            "VALUES ('scope','codex-session','raw-scope',0,?,1,1)",
            (b"b" * 32,),
        )
    return index_db, source_path


def test_interrupted_embedding_source_scope_refuses_then_retries_matching_session(
    matching_source_scope: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    index_db, source_path = matching_source_scope
    original = open_readonly_connection

    def interrupted(path: Path, **kwargs: object) -> sqlite3.Connection:
        # The real SQLite progress callback interrupts the reader, including
        # a matching Source/Index relation; no provider is constructed.
        connection = original(path, timeout_class="background-read", validate_schema=False)
        connection.set_progress_handler(lambda: 1, 1)
        return connection

    with monkeypatch.context() as scope:
        scope.setattr(embedding_derivation, "open_readonly_connection", interrupted)
        with pytest.raises(sqlite3.OperationalError) as failure:
            embedding_derivation.embedding_session_ids_for_paths(
                index_db, archive_root=index_db.parent, paths=(source_path,)
            )
        assert failure.value.sqlite_errorcode == sqlite3.SQLITE_INTERRUPT
    assert embedding_derivation.embedding_session_ids_for_paths(
        index_db, archive_root=index_db.parent, paths=(source_path,)
    ) == ("codex-session:scope",)


def test_locked_source_scope_propagates_native_busy_and_retries_without_empty_success(
    matching_source_scope: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    index_db, source_path = matching_source_scope
    source_db = index_db.parent / "source.db"
    with sqlite3.connect(source_db) as blocker:
        blocker.execute("PRAGMA journal_mode=DELETE")
        blocker.execute("BEGIN EXCLUSIVE")
        original = open_readonly_connection

        def no_wait(path: Path, **kwargs: object) -> sqlite3.Connection:
            return original(path, timeout_class="background-read", validate_schema=False, timeout=0)

        with monkeypatch.context() as scope:
            scope.setattr(embedding_derivation, "open_readonly_connection", no_wait)
            with pytest.raises(sqlite3.OperationalError) as failure:
                embedding_derivation.embedding_session_ids_for_paths(
                    index_db, archive_root=index_db.parent, paths=(source_path,)
                )
            assert failure.value.sqlite_errorcode == sqlite3.SQLITE_BUSY
        blocker.rollback()
    assert embedding_derivation.embedding_session_ids_for_paths(
        index_db, archive_root=index_db.parent, paths=(source_path,)
    ) == ("codex-session:scope",)


def test_cancelled_source_lookup_preserves_owner_cancellation(
    matching_source_scope: tuple[Path, Path],
) -> None:
    index_db, source_path = matching_source_scope
    cancelled = threading.Event()
    cancelled.set()
    token = compute_cancel.set(cancelled)
    try:
        with pytest.raises(DaemonOperationCancelled):
            embedding_derivation.embedding_session_ids_for_paths(
                index_db, archive_root=index_db.parent, paths=(source_path,)
            )
    finally:
        compute_cancel.reset(token)
    assert embedding_derivation.embedding_session_ids_for_paths(
        index_db, archive_root=index_db.parent, paths=(source_path,)
    ) == ("codex-session:scope",)


def test_successfully_empty_source_scope_is_distinct_from_lookup_failure(
    matching_source_scope: tuple[Path, Path],
) -> None:
    index_db, source_path = matching_source_scope
    with sqlite3.connect(index_db) as connection:
        assert session_ids_for_source_paths(connection, (source_path.with_name("absent.jsonl"),)) == {
            source_path.with_name("absent.jsonl"): []
        }
    assert (
        embedding_derivation.embedding_session_ids_for_paths(
            index_db, archive_root=index_db.parent, paths=(source_path.with_name("absent.jsonl"),)
        )
        == ()
    )
