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
    target_id: str
    target_type: str
    model: str
    dimension: int
    embedded_at_ms: int
    content_hash: bytes | None
    origin: str | None


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
            INSERT INTO embeddings_meta (
                target_id, target_type, model, dimension, embedded_at_ms, content_hash, origin
            ) VALUES (?, 'message', ?, ?, ?, ?, ?)
            ON CONFLICT(target_id) DO UPDATE SET
                target_type = excluded.target_type,
                model = excluded.model,
                dimension = excluded.dimension,
                embedded_at_ms = excluded.embedded_at_ms,
                content_hash = excluded.content_hash,
                origin = excluded.origin
            """,
            (message_id, model, EMBEDDING_DIMENSION, embedded_at_ms, content_hash, origin_value),
        )
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, 1, ?, 0, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                message_count_embedded = (
                    SELECT COUNT(*) FROM message_embeddings
                    WHERE session_id = excluded.session_id
                ),
                last_embedded_at_ms = excluded.last_embedded_at_ms,
                needs_reindex = 0,
                error_message = NULL
            """,
            (session_id, origin_value, embedded_at_ms),
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
                session_id, origin, message_count_embedded, needs_reindex, error_message
            ) VALUES (?, ?, 0, 1, ?)
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
        SELECT target_id, target_type, model, dimension, embedded_at_ms, content_hash, origin
        FROM embeddings_meta
        WHERE target_id = ?
        """,
        (target_id,),
    ).fetchone()
    if row is None:
        raise KeyError(target_id)
    return ArchiveEmbeddingMeta(
        target_id=row["target_id"],
        target_type=row["target_type"],
        model=row["model"],
        dimension=row["dimension"],
        embedded_at_ms=row["embedded_at_ms"],
        content_hash=row["content_hash"],
        origin=row["origin"],
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
