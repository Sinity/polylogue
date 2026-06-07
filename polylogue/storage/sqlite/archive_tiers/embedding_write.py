"""Archive embedding metadata write helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.core.enums import Origin
from polylogue.storage.search_providers.sqlite_vec_support import _serialize_f32
from polylogue.storage.sqlite.archive_tiers.embeddings import EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingStatus:
    session_id: str
    origin: str
    message_count_embedded: int
    last_embedded_at_ms: int | None
    needs_reindex: bool
    error_message: str | None


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingMeta:
    message_id: str
    model: str
    dimension: int
    content_hash: bytes
    embedded_at_ms: int | None
    needs_reindex: bool


def upsert_message_embedding(
    conn: sqlite3.Connection,
    *,
    message_id: str,
    session_id: str,
    origin: Origin | str,
    embedding: list[float],
    model: str,
    embedded_at_ms: int,
    content_hash: bytes | None = None,
) -> ArchiveEmbeddingMeta:
    """Upsert one message vector plus its reproducibility metadata."""
    if len(embedding) != EMBEDDING_DIMENSION:
        raise ValueError(f"embedding must have {EMBEDDING_DIMENSION} dimensions")
    if content_hash is None:
        raise ValueError("content_hash is required for message embedding metadata")
    origin_value = _enum_value(origin)
    with conn:
        conn.execute("DELETE FROM message_embeddings WHERE message_id = ?", (message_id,))
        conn.execute(
            """
            INSERT INTO message_embeddings (message_id, embedding, session_id, origin)
            VALUES (?, ?, ?, ?)
            """,
            (message_id, _serialize_f32(embedding), session_id, origin_value),
        )
        conn.execute(
            """
            INSERT INTO message_embeddings_meta (
                message_id, model, dimension, content_hash, embedded_at_ms, needs_reindex
            ) VALUES (?, ?, ?, ?, ?, 0)
            ON CONFLICT(message_id) DO UPDATE SET
                model = excluded.model,
                dimension = excluded.dimension,
                content_hash = excluded.content_hash,
                embedded_at_ms = excluded.embedded_at_ms,
                needs_reindex = 0
            """,
            (message_id, model, EMBEDDING_DIMENSION, content_hash, embedded_at_ms),
        )
    return read_embedding_meta(conn, message_id)


def mark_session_embedding_error(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: Origin | str,
    error_message: str,
) -> ArchiveEmbeddingStatus:
    """Record a resumable embedding error for one session."""
    origin_value = _enum_value(origin)
    with conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, 0, NULL, 1, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                needs_reindex = 1,
                error_message = excluded.error_message
            """,
            (session_id, origin_value, error_message),
        )
    return read_embedding_status(conn, session_id)


def read_embedding_meta(conn: sqlite3.Connection, target_id: str) -> ArchiveEmbeddingMeta:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT message_id, model, dimension, content_hash, embedded_at_ms, needs_reindex
        FROM message_embeddings_meta
        WHERE message_id = ?
        """,
        (target_id,),
    ).fetchone()
    if row is None:
        raise KeyError(target_id)
    return ArchiveEmbeddingMeta(
        message_id=row["message_id"],
        model=row["model"],
        dimension=row["dimension"],
        content_hash=row["content_hash"],
        embedded_at_ms=row["embedded_at_ms"],
        needs_reindex=bool(row["needs_reindex"]),
    )


def read_embedding_status(conn: sqlite3.Connection, session_id: str) -> ArchiveEmbeddingStatus:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
        FROM embedding_status
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(session_id)
    return ArchiveEmbeddingStatus(
        session_id=row["session_id"],
        origin=row["origin"],
        message_count_embedded=row["message_count_embedded"],
        last_embedded_at_ms=row["last_embedded_at_ms"],
        needs_reindex=bool(row["needs_reindex"]),
        error_message=row["error_message"],
    )


def _enum_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


__all__ = [
    "ArchiveEmbeddingMeta",
    "ArchiveEmbeddingStatus",
    "mark_session_embedding_error",
    "read_embedding_meta",
    "read_embedding_status",
    "upsert_message_embedding",
]
