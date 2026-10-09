"""Canonical synthetic vector archives for production projection tests."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
from typing import Literal

import pytest

from polylogue.core.enums import Origin
from polylogue.storage.embeddings.identity import vector_derivation_hash
from polylogue.storage.search_providers.sqlite_vec import SqliteVecProvider
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.embedding_write import ArchiveEmbeddingWrite, upsert_message_embeddings
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.identity import fixture_block_content_identity


def seed_vector_archive(
    root: Path,
    samples: Sequence[tuple[str, str, str, list[float] | None]],
    *,
    model: str = "voyage-4",
) -> dict[tuple[str, str], tuple[str, str]]:
    """Seed exact index prose and its purchased vector, returning generated identities."""
    root.mkdir(parents=True, exist_ok=True)
    identities: dict[tuple[str, str], tuple[str, str]] = {}
    with (
        closing(sqlite3.connect(root / "index.db")) as index,
        closing(sqlite3.connect(root / "embeddings.db")) as vectors,
        closing(index.cursor()) as index_cursor,
    ):
        initialize_archive_tier(index, ArchiveTier.INDEX)
        initialize_archive_tier(vectors, ArchiveTier.EMBEDDINGS)
        for native_session, native_message, text, vector in samples:
            session_id = f"codex-session:{native_session}"
            message_id = f"{session_id}:n:{native_message}"
            index_cursor.execute(
                "INSERT OR IGNORE INTO sessions (native_id, origin, title, title_source, content_hash) VALUES (?, ?, ?, 'origin', ?)",
                (native_session, Origin.CODEX_SESSION.value, native_session, b"s" * 32),
            )
            index_cursor.execute(
                """INSERT INTO messages (session_id, native_id, position, role, material_origin, word_count, content_hash)
                   VALUES (?, ?, (SELECT COUNT(*) FROM messages WHERE session_id = ?),
                           'user', 'human_authored', ?, ?)""",
                (session_id, native_message, session_id, len(text.split()), b"m" * 32),
            )
            index_cursor.execute(
                "INSERT INTO blocks (session_id, message_id, position, block_type, text, content_hash, content_identity, content_occurrence) VALUES (?, ?, 0, 'text', ?, ?, ?, 0)",
                (
                    session_id,
                    message_id,
                    text,
                    b"b" * 32,
                    fixture_block_content_identity("text", text),
                ),
            )
            if vector is not None:
                upsert_message_embeddings(
                    vectors,
                    [
                        ArchiveEmbeddingWrite(
                            message_id=message_id,
                            session_id=session_id,
                            origin=Origin.CODEX_SESSION,
                            embedding=vector,
                            model=model,
                            embedded_at_ms=1_767_225_700_000,
                            vector_derivation_hash=vector_derivation_hash(model=model, input_text=text),
                            message_content_hash=b"m" * 32,
                        )
                    ],
                )
            identities[(native_session, native_message)] = (session_id, message_id)
        index_cursor.execute(
            "UPDATE sessions SET message_count = (SELECT COUNT(*) FROM messages m WHERE m.session_id=sessions.session_id), word_count = (SELECT COALESCE(SUM(word_count),0) FROM messages m WHERE m.session_id=sessions.session_id)"
        )
        index.commit()
        vectors.commit()
    return identities


def record_owned_vector_closes(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Check owned handles after release on their worker, avoiding a false thread-affinity refusal."""
    original_release = SqliteVecProvider._release_connection
    closed: list[bool] = []

    def release(provider: SqliteVecProvider, connection: sqlite3.Connection) -> None:
        owned = connection is not provider._snapshot_connection
        original_release(provider, connection)
        if owned:
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
            closed.append(True)

    monkeypatch.setattr(SqliteVecProvider, "_release_connection", release)
    return closed


def record_similarity_read_closes(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Prove the route's ordinary preflight handles close on their creating thread."""
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    original_open = open_readonly_connection
    closed: list[bool] = []

    def open_read(path: str | Path, *, timeout_class: Literal["interactive-read"]) -> sqlite3.Connection:
        connection = original_open(path, timeout_class=timeout_class)
        original_close = connection.close
        creator = threading.get_ident()

        def close() -> None:
            assert threading.get_ident() == creator
            original_close()
            with pytest.raises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")
            closed.append(True)

        monkeypatch.setattr(connection, "close", close)
        return connection

    monkeypatch.setattr("polylogue.daemon.similarity.open_readonly_connection", open_read)
    return closed
