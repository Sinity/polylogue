"""Substrate-side embedding execution (no CLI / click coupling).

Provides three primitives that surfaces compose into their own UI:

* :func:`iter_pending_sessions` — list sessions that need embedding.
* :func:`embed_session_sync` — embed messages for one session.
* :class:`EmbedSessionOutcome` — typed outcome record.

CLI (:mod:`polylogue.cli.shared.embed_runtime`) and pipeline
(:mod:`polylogue.pipeline.run_stages`) layer their progress and message
formatting on top.
"""

from __future__ import annotations

import contextlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from polylogue.archive.models import Session
    from polylogue.protocols import VectorProvider
    from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
    from polylogue.storage.runtime import MessageRecord


EmbedSingleStatus = Literal["embedded", "no_messages", "no_embeddable_messages", "not_found", "error"]


@dataclass(frozen=True, slots=True)
class PendingSession:
    """Identifier and (optional) display title for one pending session."""

    session_id: str
    title: str | None = None
    message_count: int = 0


@dataclass(frozen=True, slots=True)
class EmbeddingCatchupLimits:
    """Bound one resumable embedding catch-up pass."""

    max_sessions: int | None = None
    max_messages: int | None = None
    stop_after_seconds: int | None = None
    max_errors: int | None = None


@dataclass(frozen=True, slots=True)
class ArchiveEmbeddingSessionState:
    """Eligible-session embedding completion counts for an archive index."""

    eligible_sessions: int
    embedded_sessions: int
    pending_sessions: int


def archive_embeddable_message_where(alias: str = "m") -> str:
    """SQL predicate for authored prose messages eligible for embedding."""

    return f"""
{alias}.message_type = 'message'
AND {alias}.role IN ('user', 'assistant')
AND {alias}.material_origin IN ('human_authored', 'assistant_authored')
AND {alias}.word_count > 0
"""


@dataclass(frozen=True, slots=True)
class EmbedSessionOutcome:
    """Typed outcome for embedding one session."""

    status: EmbedSingleStatus
    session_id: str
    title: str | None = None
    embedded_message_count: int = 0
    error: str | None = None


class _EmbedSessionStore(Protocol):
    @property
    def backend(self) -> RepositoryBackendProtocol: ...

    async def get_messages(self, session_id: str) -> list[MessageRecord]: ...

    async def view(self, session_id: str) -> Session | None: ...


class _EmbeddingTextProvider(Protocol):
    model: str
    dimension: int

    def _get_embeddings(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...


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


def iter_pending_sessions(
    backend: RepositoryBackendProtocol,
    *,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
) -> list[PendingSession]:
    """Return sessions needing embedding.

    With ``rebuild=True`` returns every session; otherwise returns
    rows missing from ``embedding_status`` or flagged ``needs_reindex``.
    """
    from polylogue.storage.sqlite.connection import open_read_connection

    with open_read_connection(backend.db_path) as conn:
        return select_pending_session_window(
            conn,
            rebuild=rebuild,
            max_sessions=max_sessions,
            max_messages=max_messages,
        )


def select_pending_session_window(
    conn: sqlite3.Connection,
    *,
    session_ids: list[str] | tuple[str, ...] | None = None,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
) -> list[PendingSession]:
    """Return one bounded, resumable pending-session window.

    Windows are ordered newest-first (``updated_at_ms`` DESC). When the
    embedding budget is smaller than the corpus — the common case, since a
    full embed of a large archive can exceed the provider's free-token
    allotment — newest-first ensures the most query-relevant (recently
    active) sessions are embedded first. It is also correct for the daemon's
    ambient catch-up, which should cover just-ingested sessions promptly.
    """

    pending: list[PendingSession] = []
    message_total = 0
    params: list[object] = []
    id_filter = ""
    unique_ids = tuple(dict.fromkeys(session_ids or ()))
    if unique_ids:
        placeholders = ", ".join("?" for _ in unique_ids)
        id_filter = f"AND c.session_id IN ({placeholders})"
        params.extend(unique_ids)

    status_exists = _table_exists(conn, "embedding_status")
    where_clause = "1 = 1" if rebuild or not status_exists else "(e.session_id IS NULL OR e.needs_reindex = 1)"

    join_clause = "LEFT JOIN embedding_status e ON c.session_id = e.session_id" if status_exists else ""
    if max_messages is None:
        cursor = conn.execute(
            f"""
            SELECT
                c.session_id,
                c.title,
                c.message_count AS message_count
            FROM sessions c
            {join_clause}
            WHERE {where_clause}
              {id_filter}
            ORDER BY COALESCE(c.updated_at_ms, 0) DESC, c.session_id
            """,
            tuple(params),
        )
    else:
        cursor = conn.execute(
            f"""
            SELECT
                c.session_id,
                c.title,
                (SELECT COUNT(*) FROM messages m WHERE m.session_id = c.session_id) AS message_count
            FROM sessions c
            {join_clause}
            WHERE {where_clause}
              {id_filter}
            ORDER BY COALESCE(c.updated_at_ms, 0) DESC, c.session_id
            """,
            tuple(params),
        )
    while True:
        rows = cursor.fetchmany(500)
        if not rows:
            break
        for row in rows:
            session_id = str(_row_value(row, 0, "session_id"))
            title_value = _row_value(row, 1, "title")
            title = None if title_value is None else str(title_value)
            message_count = _row_int(row, 2, "message_count")
            if max_sessions is not None and len(pending) >= max_sessions:
                return pending
            if max_messages is not None and message_count > max_messages:
                continue
            if max_messages is not None and pending and message_total + message_count > max_messages:
                return pending
            pending.append(
                PendingSession(
                    session_id=session_id,
                    title=title,
                    message_count=message_count,
                )
            )
            message_total += message_count
            if max_messages is not None and message_total >= max_messages:
                return pending
    return pending


def select_pending_archive_session_window(
    conn: sqlite3.Connection,
    *,
    status_table: str | None = None,
    session_ids: list[str] | tuple[str, ...] | None = None,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
    min_messages: int | None = None,
) -> list[PendingSession]:
    """Return one bounded pending-session window from a archive index.

    ``min_messages`` skips sessions below a message-count floor so a limited
    embedding budget is not spent on trivial sessions (provider smoke tests,
    empty stubs). It complements the budget bounds (``max_*``) with a quality
    floor, making selective embedding affordable for real users.
    """

    pending: list[PendingSession] = []
    message_total = 0
    params: list[object] = []
    id_filter = ""
    unique_ids = tuple(dict.fromkeys(session_ids or ()))
    if unique_ids:
        placeholders = ", ".join("?" for _ in unique_ids)
        id_filter = f"AND s.session_id IN ({placeholders})"
        params.extend(unique_ids)

    if status_table is None:
        status_table = "embedding_status" if _table_exists(conn, "embedding_status") else ""
    exact_count_cte = _archive_session_embeddable_count_cte(conn)
    aggregate_expr = None if exact_count_cte is not None else _archive_session_embeddable_count_expression(conn)
    status_select = "e.session_id, e.needs_reindex, e.message_count_embedded" if status_table else "NULL, NULL, NULL"
    join_clause = f"LEFT JOIN {status_table} e ON e.session_id = s.session_id" if status_table else ""
    count_join = ""
    if exact_count_cte is not None:
        count_select = "ec.embeddable_message_count AS estimated_message_count"
        count_join = "JOIN embeddable_counts ec ON ec.session_id = s.session_id"
        floor_filter = "AND ec.embeddable_message_count >= ?" if min_messages is not None else ""
        if min_messages is not None:
            params.append(min_messages)
        pending_filter = (
            ""
            if rebuild or not status_table
            else """
              AND (
                    e.session_id IS NULL
                 OR e.needs_reindex = 1
                 OR e.message_count_embedded < ec.embeddable_message_count
              )
            """
        )
    elif aggregate_expr is None:
        count_select = "NULL AS estimated_message_count"
        floor_filter = ""
        pending_filter = ""
    else:
        count_select = f"{aggregate_expr} AS estimated_message_count"
        floor_filter = f"AND {aggregate_expr} >= ?" if min_messages is not None else ""
        if min_messages is not None:
            params.append(min_messages)
        pending_filter = (
            ""
            if rebuild or not status_table
            else """
              AND (
                    e.session_id IS NULL
                 OR e.needs_reindex = 1
                 OR e.message_count_embedded < {aggregate_expr}
              )
            """
        ).format(aggregate_expr=aggregate_expr)
    cursor = conn.execute(
        f"""
        {exact_count_cte or ""}
        SELECT s.session_id, s.title, {status_select}, {count_select}
        FROM sessions s
        {count_join}
        {join_clause}
        WHERE 1 = 1
          {id_filter}
          {floor_filter}
          {pending_filter}
        ORDER BY (s.sort_key_ms IS NULL), s.sort_key_ms DESC, s.session_id
        """,
        tuple(params),
    )
    while True:
        rows = cursor.fetchmany(500)
        if not rows:
            break
        for row in rows:
            session_id = str(_row_value(row, 0, "session_id"))
            title_value = _row_value(row, 1, "title")
            title = None if title_value is None else str(title_value)
            status_session_id = _row_value(row, 2, "status_session_id")
            needs_reindex = _row_int(row, 3, "needs_reindex") if status_session_id is not None else 0
            message_count_embedded = _row_int(row, 4, "message_count_embedded") if status_session_id is not None else 0
            estimated_count = _row_int(row, 5, "estimated_message_count")
            if estimated_count <= 0:
                continue
            if min_messages is not None and estimated_count < min_messages:
                continue
            if not (
                rebuild
                or not status_table
                or status_session_id is None
                or needs_reindex == 1
                or message_count_embedded < estimated_count
            ):
                continue
            if max_sessions is not None and len(pending) >= max_sessions:
                return pending
            if max_messages is not None and estimated_count > max_messages:
                continue
            if max_messages is not None and pending and message_total + estimated_count > max_messages:
                return pending
            pending.append(PendingSession(session_id=session_id, title=title, message_count=estimated_count))
            message_total += estimated_count
            if max_messages is not None and message_total >= max_messages:
                return pending
    return pending


def count_archive_session_embeddable_messages(conn: sqlite3.Connection, session_id: str) -> int:
    """Count authored-prose messages eligible for embedding in one session."""

    if not _table_exists(conn, "messages"):
        return 0
    messages_ref = archive_embedding_messages_table_ref(conn, alias="m")
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {messages_ref}
        WHERE {archive_embeddable_message_where("m")}
          AND m.session_id = ?
        """,
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def count_archive_embedding_session_state(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    rebuild: bool = False,
) -> ArchiveEmbeddingSessionState:
    """Count eligible sessions and their embedding completion state."""

    if not _table_exists(conn, "messages"):
        return ArchiveEmbeddingSessionState(eligible_sessions=0, embedded_sessions=0, pending_sessions=0)

    if rebuild or not status_table:
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT m.session_id
                FROM {archive_messages_table_ref(conn, alias="m")}
                WHERE {archive_embeddable_message_where("m")}
                GROUP BY m.session_id
            ) eligible_counts
            """
        ).fetchone()
        eligible = int(row[0] or 0) if row is not None else 0
        return ArchiveEmbeddingSessionState(
            eligible_sessions=eligible,
            embedded_sessions=0,
            pending_sessions=eligible,
        )

    row = conn.execute(
        f"""
        WITH eligible_counts AS (
            SELECT m.session_id, COUNT(*) AS message_count
            FROM {archive_messages_table_ref(conn, alias="m")}
            WHERE {archive_embeddable_message_where("m")}
            GROUP BY m.session_id
        )
        SELECT
            COUNT(*) AS eligible_sessions,
            SUM(
                CASE
                    WHEN e.session_id IS NOT NULL
                     AND e.needs_reindex = 0
                     AND e.message_count_embedded >= ec.message_count
                    THEN 1 ELSE 0
                END
            ) AS embedded_sessions,
            SUM(
                CASE
                    WHEN e.session_id IS NULL
                      OR e.needs_reindex = 1
                      OR e.message_count_embedded < ec.message_count
                    THEN 1 ELSE 0
                END
            ) AS pending_sessions
        FROM eligible_counts ec
        LEFT JOIN {status_table} e ON e.session_id = ec.session_id
        """
    ).fetchone()
    if row is None:
        return ArchiveEmbeddingSessionState(eligible_sessions=0, embedded_sessions=0, pending_sessions=0)
    return ArchiveEmbeddingSessionState(
        eligible_sessions=int(row[0] or 0),
        embedded_sessions=int(row[1] or 0),
        pending_sessions=int(row[2] or 0),
    )


def archive_messages_table_ref(conn: sqlite3.Connection, *, alias: str) -> str:
    if _index_exists(conn, "idx_messages_message_type"):
        return f"messages AS {alias} INDEXED BY idx_messages_message_type"
    return f"messages AS {alias}"


def archive_embedding_messages_table_ref(conn: sqlite3.Connection, *, alias: str) -> str:
    if _index_exists(conn, "idx_messages_embedding_prose"):
        return f"messages AS {alias} INDEXED BY idx_messages_embedding_prose"
    if _index_exists(conn, "idx_messages_session_material_origin"):
        return f"messages AS {alias} INDEXED BY idx_messages_session_material_origin"
    return archive_messages_table_ref(conn, alias=alias)


def _archive_session_embeddable_count_expression(conn: sqlite3.Connection) -> str | None:
    """Return a minimal-fixture fallback when exact message columns are absent."""

    columns = _table_columns(conn, "sessions")
    if {"authored_user_message_count", "assistant_message_count"}.issubset(columns):
        return "COALESCE(s.authored_user_message_count, 0) + COALESCE(s.assistant_message_count, 0)"
    if "message_count" in columns:
        return "COALESCE(s.message_count, 0)"
    return None


def _archive_session_embeddable_count_cte(conn: sqlite3.Connection) -> str | None:
    """Return exact per-session prose counts for canonical archive indexes."""

    message_columns = _table_columns(conn, "messages")
    required_columns = {"session_id", "message_type", "role", "material_origin", "word_count"}
    if not required_columns.issubset(message_columns):
        return None
    messages_ref = archive_embedding_messages_table_ref(conn, alias="m")
    return f"""
        WITH embeddable_counts AS (
            SELECT m.session_id, COUNT(*) AS embeddable_message_count
            FROM {messages_ref}
            WHERE {archive_embeddable_message_where("m")}
            GROUP BY m.session_id
        )
    """


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def mark_all_archive_sessions_needs_reindex(index_db_path: Path) -> None:
    """Flag every archive session for embedding rebuild."""

    embeddings_db_path = index_db_path.with_name("embeddings.db")
    conn = sqlite3.connect(embeddings_db_path, timeout=30.0)
    try:
        conn.execute("ATTACH DATABASE ? AS idx", (str(index_db_path),))
        with conn:
            conn.execute(
                """
                INSERT INTO embedding_status (session_id, origin, message_count_embedded, needs_reindex, error_message)
                SELECT session_id, origin, 0, 1, NULL
                FROM idx.sessions
                ON CONFLICT(session_id) DO UPDATE SET
                    needs_reindex = 1,
                    error_message = NULL
                """
            )
    finally:
        conn.close()


def embed_session_sync(
    repo: _EmbedSessionStore,
    vec_provider: VectorProvider,
    session_id: str,
    *,
    fetch_title: bool = False,
) -> EmbedSessionOutcome:
    """Embed one session. Returns an outcome — does not raise on no-op.

    ``fetch_title=True`` issues an extra ``view`` lookup so callers can
    display a friendly label; when False the title field is left ``None``.
    """
    from polylogue.api.sync.bridge import run_coroutine_sync

    title: str | None = None
    if fetch_title:

        async def _view_title() -> Session | None:
            return await repo.view(session_id)

        conv = run_coroutine_sync(_view_title())
        if conv is None:
            return EmbedSessionOutcome(status="not_found", session_id=session_id)
        title = conv.title
        full_id = str(conv.id)
    else:
        full_id = session_id

    try:
        messages = run_coroutine_sync(repo.get_messages(full_id))
        if not messages:
            _record_embedding_success(repo.backend.db_path, full_id, message_count=0)
            return EmbedSessionOutcome(status="no_messages", session_id=full_id, title=title)
        vec_provider.upsert(full_id, messages)
        if not _embedding_status_row_exists(repo.backend.db_path, full_id):
            _record_embedding_success(repo.backend.db_path, full_id, message_count=0)
            return EmbedSessionOutcome(
                status="no_embeddable_messages",
                session_id=full_id,
                title=title,
                embedded_message_count=0,
            )
    except Exception as exc:
        _record_embedding_failure(repo.backend.db_path, full_id, str(exc))
        return EmbedSessionOutcome(status="error", session_id=full_id, title=title, error=str(exc))
    return EmbedSessionOutcome(
        status="embedded",
        session_id=full_id,
        title=title,
        embedded_message_count=len(messages),
    )


def embed_archive_session_sync(
    index_db_path: Path,
    vec_provider: VectorProvider,
    session_id: str,
) -> EmbedSessionOutcome:
    """Embed one archive session without routing through the archive repository."""

    text_provider = cast(_EmbeddingTextProvider, vec_provider)
    if not hasattr(text_provider, "_get_embeddings"):
        return EmbedSessionOutcome(
            status="error",
            session_id=session_id,
            error="vector provider does not expose text embedding generation",
        )

    embeddings_db_path = index_db_path.with_name("embeddings.db")
    index_conn = sqlite3.connect(f"file:{index_db_path}?mode=ro", uri=True, timeout=30.0)
    index_conn.row_factory = sqlite3.Row
    embeddings_conn = sqlite3.connect(embeddings_db_path, timeout=30.0)
    try:
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        loaded, error = try_load_sqlite_vec(embeddings_conn)
        if not loaded:
            raise RuntimeError("archive embedding materialization requires sqlite-vec") from error
        session = index_conn.execute(
            "SELECT session_id, origin, title, message_count FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session is None:
            return EmbedSessionOutcome(status="not_found", session_id=session_id)
        messages_ref = archive_embedding_messages_table_ref(index_conn, alias="m")
        rows = index_conn.execute(
            f"""
            SELECT m.message_id, m.role, m.content_hash, m.material_origin, m.message_type,
                   GROUP_CONCAT(b.text, char(10) || char(10)) AS text
            FROM {messages_ref}
            LEFT JOIN blocks AS b INDEXED BY idx_blocks_session_position
              ON b.session_id = m.session_id
             AND b.message_id = m.message_id
             AND b.block_type = 'text'
             AND b.text IS NOT NULL
            WHERE m.session_id = ?
              AND {archive_embeddable_message_where("m")}
            GROUP BY m.message_id, m.role, m.content_hash, m.material_origin, m.message_type, m.position, m.variant_index
            ORDER BY m.position, m.variant_index
            """,
            (session_id,),
        ).fetchall()
        if not rows:
            _record_archive_embedding_success(
                embeddings_conn, session_id=session_id, origin=str(session["origin"]), message_count=0
            )
            return EmbedSessionOutcome(
                status="no_messages" if int(session["message_count"] or 0) <= 0 else "no_embeddable_messages",
                session_id=session_id,
                title=None if session["title"] is None else str(session["title"]),
            )

        embeddable = [
            row
            for row in rows
            if _should_embed_archive_message(row["material_origin"], row["message_type"], row["role"], row["text"])
        ]
        if not embeddable:
            _record_archive_embedding_success(
                embeddings_conn, session_id=session_id, origin=str(session["origin"]), message_count=0
            )
            return EmbedSessionOutcome(
                status="no_embeddable_messages",
                session_id=session_id,
                title=None if session["title"] is None else str(session["title"]),
            )

        embeddings = text_provider._get_embeddings([str(row["text"]) for row in embeddable], input_type="document")
        if len(embeddings) != len(embeddable):
            raise RuntimeError("embedding provider returned a mismatched vector count")
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        from polylogue.storage.sqlite.archive_tiers.embedding_write import upsert_message_embedding

        for row, embedding in zip(embeddable, embeddings, strict=True):
            upsert_message_embedding(
                embeddings_conn,
                message_id=str(row["message_id"]),
                session_id=session_id,
                origin=str(session["origin"]),
                embedding=embedding,
                model=text_provider.model,
                embedded_at_ms=now_ms,
                content_hash=bytes(row["content_hash"]) if row["content_hash"] is not None else None,
            )
        _record_archive_embedding_success(
            embeddings_conn,
            session_id=session_id,
            origin=str(session["origin"]),
            message_count=len(embeddable),
        )
    except Exception as exc:
        try:
            from polylogue.storage.sqlite.archive_tiers.embedding_write import mark_session_embedding_error

            origin_row = index_conn.execute(
                "SELECT origin FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if origin_row is not None:
                mark_session_embedding_error(
                    embeddings_conn,
                    session_id=session_id,
                    origin=str(origin_row["origin"]),
                    error_message=str(exc),
                )
        finally:
            with contextlib.suppress(sqlite3.Error):
                index_conn.close()
            with contextlib.suppress(sqlite3.Error):
                embeddings_conn.close()
        return EmbedSessionOutcome(status="error", session_id=session_id, error=str(exc))
    finally:
        with contextlib.suppress(sqlite3.Error):
            index_conn.close()
        with contextlib.suppress(sqlite3.Error):
            embeddings_conn.close()
    return EmbedSessionOutcome(
        status="embedded",
        session_id=session_id,
        title=None if session["title"] is None else str(session["title"]),
        embedded_message_count=len(embeddable),
    )


def _record_archive_embedding_success(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    origin: str,
    message_count: int,
) -> None:
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    with conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, ?, ?, 0, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                message_count_embedded = excluded.message_count_embedded,
                last_embedded_at_ms = excluded.last_embedded_at_ms,
                needs_reindex = 0,
                error_message = NULL
            """,
            (session_id, origin, message_count, now_ms),
        )


_PROSE_MATERIAL_ORIGINS = frozenset({"human_authored", "assistant_authored"})
_PROSE_ROLES = frozenset({"user", "assistant"})


def _should_embed_archive_message(material_origin: object, message_type: object, role: object, text: object) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    if str(message_type) != "message":
        return False
    if str(role) not in _PROSE_ROLES:
        return False
    return str(material_origin) in _PROSE_MATERIAL_ORIGINS


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def _index_exists(conn: sqlite3.Connection, index: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=? LIMIT 1",
        (index,),
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
                session_id TEXT PRIMARY KEY,
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


def _record_embedding_success(db_path: object, session_id: str, *, message_count: int) -> None:
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
                session_id, message_count_embedded, last_embedded_at, needs_reindex, error_message
            ) VALUES (?, ?, datetime('now'), 0, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                message_count_embedded = excluded.message_count_embedded,
                last_embedded_at = excluded.last_embedded_at,
                needs_reindex = 0,
                error_message = NULL
            """,
            (session_id, message_count),
        )
        conn.commit()
    finally:
        conn.close()


def _record_embedding_failure(db_path: object, session_id: str, error: str) -> None:
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
                session_id, message_count_embedded, last_embedded_at, needs_reindex, error_message
            ) VALUES (?, 0, datetime('now'), 1, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_embedded_at = excluded.last_embedded_at,
                needs_reindex = 1,
                error_message = excluded.error_message
            """,
            (session_id, error),
        )
        conn.commit()
    finally:
        conn.close()


def _embedding_status_row_exists(db_path: object, session_id: str) -> bool:
    path = _usable_db_path(db_path)
    if path is None or not path.exists():
        return True

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
        try:
            if not _table_exists(conn, "embedding_status"):
                return False
            row = conn.execute(
                "SELECT 1 FROM embedding_status WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return True


__all__ = [
    "EmbeddingCatchupLimits",
    "ArchiveEmbeddingSessionState",
    "EmbedSessionOutcome",
    "EmbedSingleStatus",
    "PendingSession",
    "archive_embeddable_message_where",
    "archive_embedding_messages_table_ref",
    "archive_messages_table_ref",
    "count_archive_embedding_session_state",
    "count_archive_session_embeddable_messages",
    "embed_session_sync",
    "embed_archive_session_sync",
    "iter_pending_sessions",
    "mark_all_archive_sessions_needs_reindex",
    "select_pending_session_window",
    "select_pending_archive_session_window",
]
