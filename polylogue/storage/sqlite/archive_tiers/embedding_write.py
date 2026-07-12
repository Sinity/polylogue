"""Archive embedding metadata write helpers.

Writer module: embeddings.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

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


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingWrite:
    message_id: str
    session_id: str
    origin: Origin | str
    embedding: list[float]
    model: str
    embedded_at_ms: int
    content_hash: bytes


EmbeddingFailureState = Literal["retryable", "terminal", "acknowledged", "superseded", "resolved"]
EmbeddingFailureResolution = Literal["acknowledge", "requeue", "supersede"]


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingFailure:
    failure_id: str
    session_id: str
    origin: str
    message_refs: tuple[str, ...]
    provider: str
    model: str
    error_class: str
    error_message: str
    retryable: bool
    lifecycle_state: EmbeddingFailureState
    created_at_ms: int
    updated_at_ms: int
    resolved_at_ms: int | None
    resolution_action: str | None
    resolution_note: str | None
    superseded_by: str | None


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
    if content_hash is None:
        raise ValueError("content_hash is required for message embedding metadata")
    upsert_message_embeddings(
        conn,
        [
            ArchiveEmbeddingWrite(
                message_id=message_id,
                session_id=session_id,
                origin=origin,
                embedding=embedding,
                model=model,
                embedded_at_ms=embedded_at_ms,
                content_hash=content_hash,
            )
        ],
    )
    return read_embedding_meta(conn, message_id)


def upsert_message_embeddings(
    conn: sqlite3.Connection,
    writes: Sequence[ArchiveEmbeddingWrite],
) -> None:
    """Upsert message vectors and metadata in one transaction."""
    for write in writes:
        if len(write.embedding) != EMBEDDING_DIMENSION:
            raise ValueError(f"embedding must have {EMBEDDING_DIMENSION} dimensions")

    with conn:
        for write in writes:
            origin_value = _enum_value(write.origin)
            conn.execute("DELETE FROM message_embeddings WHERE message_id = ?", (write.message_id,))
            conn.execute(
                """
                INSERT INTO message_embeddings (message_id, embedding, session_id, origin)
                VALUES (?, ?, ?, ?)
                """,
                (write.message_id, _serialize_f32(write.embedding), write.session_id, origin_value),
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
                (
                    write.message_id,
                    write.model,
                    EMBEDDING_DIMENSION,
                    write.content_hash,
                    write.embedded_at_ms,
                ),
            )


def mark_session_embedding_error(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: Origin | str,
    error_message: str,
    retryable: bool = True,
) -> ArchiveEmbeddingStatus:
    """Record a resumable embedding error for one session."""
    origin_value = _enum_value(origin)
    needs_reindex = 1 if retryable else 0
    with conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, 0, NULL, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                needs_reindex = excluded.needs_reindex,
                error_message = excluded.error_message
            """,
            (session_id, origin_value, needs_reindex, error_message),
        )
    return read_embedding_status(conn, session_id)


def record_embedding_failure(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: Origin | str,
    message_refs: Sequence[str],
    provider: str,
    model: str,
    error_class: str,
    error_message: str,
    retryable: bool,
    occurred_at_ms: int | None = None,
) -> ArchiveEmbeddingFailure:
    """Persist one inspectable failure event and its current retry lifecycle."""

    now_ms = int(time.time() * 1000) if occurred_at_ms is None else occurred_at_ms
    failure_id = f"embedding-failure:{uuid.uuid4()}"
    state: EmbeddingFailureState = "retryable" if retryable else "terminal"
    origin_value = _enum_value(origin)
    refs = tuple(dict.fromkeys(str(ref) for ref in message_refs))
    with conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, 0, NULL, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                needs_reindex = excluded.needs_reindex,
                error_message = excluded.error_message
            """,
            (session_id, origin_value, 1 if retryable else 0, error_message),
        )
        conn.execute(
            """
            UPDATE embedding_failures
            SET lifecycle_state = 'superseded', updated_at_ms = ?, resolved_at_ms = ?,
                resolution_action = 'superseded', superseded_by = ?
            WHERE session_id = ? AND lifecycle_state IN ('retryable', 'terminal')
            """,
            (now_ms, now_ms, failure_id, session_id),
        )
        conn.execute(
            """
            INSERT INTO embedding_failures (
                failure_id, session_id, origin, message_refs_json, provider, model,
                error_class, error_message, retryable, lifecycle_state, created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                failure_id,
                session_id,
                origin_value,
                json.dumps(refs),
                provider,
                model,
                error_class,
                error_message,
                int(retryable),
                state,
                now_ms,
                now_ms,
            ),
        )
    return read_embedding_failure(conn, failure_id)


def resolve_embedding_failure(
    conn: sqlite3.Connection,
    *,
    failure_id: str,
    action: EmbeddingFailureResolution,
    note: str | None = None,
    superseded_by: str | None = None,
    resolved_at_ms: int | None = None,
) -> ArchiveEmbeddingFailure:
    """Explicitly acknowledge, supersede, or requeue an active failure."""

    now_ms = int(time.time() * 1000) if resolved_at_ms is None else resolved_at_ms
    state = cast(
        EmbeddingFailureState,
        {
            "acknowledge": "acknowledged",
            "supersede": "superseded",
            "requeue": "resolved",
        }[action],
    )
    with conn:
        row = conn.execute(
            "SELECT session_id FROM embedding_failures WHERE failure_id = ? AND lifecycle_state IN ('retryable', 'terminal')",
            (failure_id,),
        ).fetchone()
        if row is None:
            raise KeyError(failure_id)
        conn.execute(
            """
            UPDATE embedding_failures
            SET lifecycle_state = ?, updated_at_ms = ?, resolved_at_ms = ?, resolution_action = ?,
                resolution_note = ?, superseded_by = ?
            WHERE failure_id = ?
            """,
            (state, now_ms, now_ms, action, note, superseded_by, failure_id),
        )
        if action == "requeue":
            conn.execute(
                "UPDATE embedding_status SET needs_reindex = 1, error_message = NULL WHERE session_id = ?",
                (str(row[0]),),
            )
    return read_embedding_failure(conn, failure_id)


def resolve_open_embedding_failures_for_session(
    conn: sqlite3.Connection, *, session_id: str, resolved_at_ms: int | None = None
) -> int:
    """Preserve prior failures while marking a later successful embedding as resolution."""

    now_ms = int(time.time() * 1000) if resolved_at_ms is None else resolved_at_ms
    with conn:
        cursor = conn.execute(
            """
            UPDATE embedding_failures
            SET lifecycle_state = 'resolved', updated_at_ms = ?, resolved_at_ms = ?, resolution_action = 'embedded'
            WHERE session_id = ? AND lifecycle_state IN ('retryable', 'terminal')
            """,
            (now_ms, now_ms, session_id),
        )
    return max(0, cursor.rowcount)


def list_active_embedding_failures(conn: sqlite3.Connection, *, limit: int = 25) -> tuple[ArchiveEmbeddingFailure, ...]:
    """Return bounded current failure identities for status and agent surfaces."""

    rows = conn.execute(
        """
        SELECT failure_id, session_id, origin, message_refs_json, provider, model, error_class, error_message,
               retryable, lifecycle_state, created_at_ms, updated_at_ms, resolved_at_ms, resolution_action,
               resolution_note, superseded_by
        FROM embedding_failures
        WHERE lifecycle_state IN ('retryable', 'terminal')
        ORDER BY updated_at_ms DESC, failure_id ASC
        LIMIT ?
        """,
        (max(0, limit),),
    ).fetchall()
    return tuple(_failure_from_row(row) for row in rows)


def read_embedding_failure(conn: sqlite3.Connection, failure_id: str) -> ArchiveEmbeddingFailure:
    row = conn.execute(
        """
        SELECT failure_id, session_id, origin, message_refs_json, provider, model, error_class, error_message,
               retryable, lifecycle_state, created_at_ms, updated_at_ms, resolved_at_ms, resolution_action,
               resolution_note, superseded_by
        FROM embedding_failures WHERE failure_id = ?
        """,
        (failure_id,),
    ).fetchone()
    if row is None:
        raise KeyError(failure_id)
    return _failure_from_row(row)


def _failure_from_row(row: sqlite3.Row | tuple[object, ...]) -> ArchiveEmbeddingFailure:
    message_refs_raw = row[3]
    try:
        message_refs = tuple(str(item) for item in json.loads(str(message_refs_raw)))
    except (TypeError, ValueError, json.JSONDecodeError):
        message_refs = ()
    return ArchiveEmbeddingFailure(
        failure_id=str(row[0]),
        session_id=str(row[1]),
        origin=str(row[2]),
        message_refs=message_refs,
        provider=str(row[4]),
        model=str(row[5]),
        error_class=str(row[6]),
        error_message=str(row[7]),
        retryable=bool(row[8]),
        lifecycle_state=cast(EmbeddingFailureState, str(row[9])),
        created_at_ms=int(str(row[10])),
        updated_at_ms=int(str(row[11])),
        resolved_at_ms=None if row[12] is None else int(str(row[12])),
        resolution_action=None if row[13] is None else str(row[13]),
        resolution_note=None if row[14] is None else str(row[14]),
        superseded_by=None if row[15] is None else str(row[15]),
    )


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
    "ArchiveEmbeddingFailure",
    "EmbeddingFailureResolution",
    "EmbeddingFailureState",
    "ArchiveEmbeddingStatus",
    "ArchiveEmbeddingWrite",
    "list_active_embedding_failures",
    "mark_session_embedding_error",
    "read_embedding_failure",
    "read_embedding_meta",
    "read_embedding_status",
    "upsert_message_embedding",
    "upsert_message_embeddings",
    "record_embedding_failure",
    "resolve_embedding_failure",
    "resolve_open_embedding_failures_for_session",
]
