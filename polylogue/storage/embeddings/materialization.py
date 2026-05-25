"""Substrate-side embedding execution (no CLI / click coupling).

Provides three primitives that surfaces compose into their own UI:

* :func:`iter_pending_conversations` — list conversations that need embedding.
* :func:`embed_conversation_sync` — embed messages for one conversation.
* :class:`EmbedConversationOutcome` — typed outcome record.

CLI (:mod:`polylogue.cli.shared.embed_runtime`) and pipeline
(:mod:`polylogue.pipeline.run_stages`) layer their progress and message
formatting on top.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from polylogue.archive.models import Conversation
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
    from polylogue.storage.runtime import MessageRecord


EmbedSingleStatus = Literal["embedded", "no_messages", "no_embeddable_messages", "not_found", "error"]


@dataclass(frozen=True, slots=True)
class PendingConversation:
    """Identifier and (optional) display title for one pending conversation."""

    conversation_id: str
    title: str | None = None
    message_count: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingCatchupLimits:
    """Bound one resumable embedding catch-up pass."""

    max_conversations: int | None = None
    max_messages: int | None = None
    stop_after_seconds: int | None = None
    max_errors: int | None = None


@dataclass(frozen=True, slots=True)
class EmbedConversationOutcome:
    """Typed outcome for embedding one conversation."""

    status: EmbedSingleStatus
    conversation_id: str
    title: str | None = None
    embedded_message_count: int = 0
    error: str | None = None


class _EmbedConversationStore(Protocol):
    @property
    def backend(self) -> RepositoryBackendProtocol: ...

    async def get_messages(self, conversation_id: str) -> list[MessageRecord]: ...

    async def view(self, conversation_id: str) -> Conversation | None: ...


def _row_value(row: object, index: int, key: str) -> object:
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, sqlite3.Row):
        try:
            return row[key]
        except (IndexError, KeyError):
            return None
    if isinstance(row, tuple):
        return row[index] if index < len(row) else None
    try:
        return getattr(row, key)
    except AttributeError:
        return None


def _row_int(row: object, index: int, key: str) -> int:
    value = _row_value(row, index, key)
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def iter_pending_conversations(
    backend: RepositoryBackendProtocol,
    *,
    rebuild: bool = False,
    max_conversations: int | None = None,
    max_messages: int | None = None,
) -> list[PendingConversation]:
    """Return conversations needing embedding.

    With ``rebuild=True`` returns every conversation; otherwise returns
    rows missing from ``embedding_status`` or flagged ``needs_reindex``.
    """
    from polylogue.storage.sqlite.connection import open_read_connection

    with open_read_connection(backend.db_path) as conn:
        return select_pending_conversation_window(
            conn,
            rebuild=rebuild,
            max_conversations=max_conversations,
            max_messages=max_messages,
        )


def select_pending_conversation_window(
    conn: sqlite3.Connection,
    *,
    conversation_ids: list[str] | tuple[str, ...] | None = None,
    rebuild: bool = False,
    max_conversations: int | None = None,
    max_messages: int | None = None,
) -> list[PendingConversation]:
    """Return one bounded, resumable pending-conversation window."""

    pending: list[PendingConversation] = []
    message_total = 0
    params: list[object] = []
    id_filter = ""
    unique_ids = tuple(dict.fromkeys(conversation_ids or ()))
    if unique_ids:
        placeholders = ", ".join("?" for _ in unique_ids)
        id_filter = f"AND c.conversation_id IN ({placeholders})"
        params.extend(unique_ids)

    status_exists = _table_exists(conn, "embedding_status")
    stats_exists = max_messages is None and _table_exists(conn, "conversation_stats")
    where_clause = "1 = 1" if rebuild or not status_exists else "(e.conversation_id IS NULL OR e.needs_reindex = 1)"

    join_clause = "LEFT JOIN embedding_status e ON c.conversation_id = e.conversation_id" if status_exists else ""
    if stats_exists:
        cursor = conn.execute(
            f"""
            SELECT
                c.conversation_id,
                c.title,
                COALESCE(
                    cs.message_count,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id)
                ) AS message_count
            FROM conversations c
            {join_clause}
            LEFT JOIN conversation_stats cs ON cs.conversation_id = c.conversation_id
            WHERE {where_clause}
              {id_filter}
            ORDER BY COALESCE(c.updated_at, ''), c.conversation_id
            """,
            tuple(params),
        )
    else:
        cursor = conn.execute(
            f"""
            SELECT
                c.conversation_id,
                c.title,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS message_count
            FROM conversations c
            {join_clause}
            WHERE {where_clause}
              {id_filter}
            ORDER BY COALESCE(c.updated_at, ''), c.conversation_id
            """,
            tuple(params),
        )
    while True:
        rows = cursor.fetchmany(500)
        if not rows:
            break
        for row in rows:
            conversation_id = str(_row_value(row, 0, "conversation_id"))
            title_value = _row_value(row, 1, "title")
            title = None if title_value is None else str(title_value)
            message_count = _row_int(row, 2, "message_count")
            if max_conversations is not None and len(pending) >= max_conversations:
                return pending
            if max_messages is not None and pending and message_total + message_count > max_messages:
                return pending
            pending.append(
                PendingConversation(
                    conversation_id=conversation_id,
                    title=title,
                    message_count=message_count,
                )
            )
            message_total += message_count
            if max_messages is not None and message_total >= max_messages:
                return pending
    return pending


def mark_all_conversations_needs_reindex(backend: RepositoryBackendProtocol) -> None:
    """Flag every conversation for a rebuild-backed resumable catch-up."""

    _ensure_embedding_status_table(backend.db_path)
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(backend.db_path) as conn:
        conn.execute(
            """
            INSERT INTO embedding_status (conversation_id, message_count_embedded, needs_reindex, error_message)
            SELECT conversation_id, 0, 1, NULL
            FROM conversations
            ON CONFLICT(conversation_id) DO UPDATE SET
                needs_reindex = 1,
                error_message = NULL
            """
        )
        conn.commit()


def embed_conversation_sync(
    repo: _EmbedConversationStore,
    vec_provider: VectorProvider,
    conversation_id: str,
    *,
    fetch_title: bool = False,
) -> EmbedConversationOutcome:
    """Embed one conversation. Returns an outcome — does not raise on no-op.

    ``fetch_title=True`` issues an extra ``view`` lookup so callers can
    display a friendly label; when False the title field is left ``None``.
    """
    from polylogue.api.sync.bridge import run_coroutine_sync

    title: str | None = None
    if fetch_title:

        async def _view_title() -> Conversation | None:
            return await repo.view(conversation_id)

        conv = run_coroutine_sync(_view_title())
        if conv is None:
            return EmbedConversationOutcome(status="not_found", conversation_id=conversation_id)
        title = conv.title
        full_id = str(conv.id)
    else:
        full_id = conversation_id

    try:
        messages = run_coroutine_sync(repo.get_messages(full_id))
        if not messages:
            _record_embedding_success(repo.backend.db_path, full_id, message_count=0)
            return EmbedConversationOutcome(status="no_messages", conversation_id=full_id, title=title)
        vec_provider.upsert(full_id, messages)
        if not _embedding_status_row_exists(repo.backend.db_path, full_id):
            _record_embedding_success(repo.backend.db_path, full_id, message_count=0)
            return EmbedConversationOutcome(
                status="no_embeddable_messages",
                conversation_id=full_id,
                title=title,
                embedded_message_count=0,
            )
    except Exception as exc:
        _record_embedding_failure(repo.backend.db_path, full_id, str(exc))
        return EmbedConversationOutcome(status="error", conversation_id=full_id, title=title, error=str(exc))
    return EmbedConversationOutcome(
        status="embedded",
        conversation_id=full_id,
        title=title,
        embedded_message_count=len(messages),
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _usable_db_path(db_path: object) -> Path | None:
    if isinstance(db_path, Path):
        return db_path
    if isinstance(db_path, str):
        return Path(db_path)
    return None


def _ensure_embedding_status_table(db_path: object) -> bool:
    path = _usable_db_path(db_path)
    if path is None:
        return False

    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_status (
                conversation_id TEXT PRIMARY KEY,
                message_count_embedded INTEGER DEFAULT 0,
                last_embedded_at TEXT,
                needs_reindex INTEGER DEFAULT 0,
                error_message TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding_status_needs
            ON embedding_status(needs_reindex) WHERE needs_reindex = 1
        """)
        conn.commit()
    finally:
        conn.close()
    return True


def _record_embedding_success(db_path: object, conversation_id: str, *, message_count: int) -> None:
    if not _ensure_embedding_status_table(db_path):
        return

    path = _usable_db_path(db_path)
    if path is None:
        return
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute(
            """
            INSERT INTO embedding_status (
                conversation_id, message_count_embedded, last_embedded_at, needs_reindex, error_message
            ) VALUES (?, ?, datetime('now'), 0, NULL)
            ON CONFLICT(conversation_id) DO UPDATE SET
                message_count_embedded = excluded.message_count_embedded,
                last_embedded_at = excluded.last_embedded_at,
                needs_reindex = 0,
                error_message = NULL
            """,
            (conversation_id, message_count),
        )
        conn.commit()
    finally:
        conn.close()


def _record_embedding_failure(db_path: object, conversation_id: str, error: str) -> None:
    if not _ensure_embedding_status_table(db_path):
        return

    path = _usable_db_path(db_path)
    if path is None:
        return
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute(
            """
            INSERT INTO embedding_status (
                conversation_id, message_count_embedded, last_embedded_at, needs_reindex, error_message
            ) VALUES (?, 0, datetime('now'), 1, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                last_embedded_at = excluded.last_embedded_at,
                needs_reindex = 1,
                error_message = excluded.error_message
            """,
            (conversation_id, error),
        )
        conn.commit()
    finally:
        conn.close()


def _embedding_status_row_exists(db_path: object, conversation_id: str) -> bool:
    path = _usable_db_path(db_path)
    if path is None or not path.exists():
        return True

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
        try:
            if not _table_exists(conn, "embedding_status"):
                return False
            row = conn.execute(
                "SELECT 1 FROM embedding_status WHERE conversation_id = ? LIMIT 1",
                (conversation_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return True


__all__ = [
    "EmbeddingCatchupLimits",
    "EmbedConversationOutcome",
    "EmbedSingleStatus",
    "PendingConversation",
    "embed_conversation_sync",
    "iter_pending_conversations",
    "mark_all_conversations_needs_reindex",
    "select_pending_conversation_window",
]
