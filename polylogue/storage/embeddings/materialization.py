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

from polylogue.config import load_polylogue_config
from polylogue.storage.embeddings.identity import (
    EMBEDDING_DERIVATION_KEY_SQL_FUNCTION,
    EMBEDDING_MESSAGE_KEY_SQL_FUNCTION,
    EMBEDDING_SOURCE_HASH_SQL_FUNCTION,
    EmbeddingRecipe,
    EmbeddingSourceDigest,
    message_embedding_derivation_key,
    register_embedding_identity_sql,
)
from polylogue.storage.table_existence import table_exists as _table_exists

if TYPE_CHECKING:
    from polylogue.archive.models import Session
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
    from polylogue.storage.runtime import MessageRecord


EmbedSingleStatus = Literal["embedded", "no_messages", "no_embeddable_messages", "not_found", "error"]
ARCHIVE_EMBED_MESSAGE_BATCH_SIZE = 128
# Kept under the polylogue-ve9z heuristic purge (h01l): these match HTTP
# status codes in the embedding provider's OWN live error responses (retry
# policy triage), not prose inference over archived session content.
TERMINAL_PROVIDER_ERROR_MARKERS = (
    "http 400",
    "status 400",
    "400 bad request",
)


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
    blocked_sessions: int = 0


@dataclass(frozen=True, slots=True)
class _ArchiveEmbeddingFreshnessPredicate:
    """One exact desired-key predicate shared by selection and status counts."""

    cte_sql: str
    join_sql: str
    fresh_sql: str
    blocked_sql: str
    pending_sql: str


def is_terminal_embedding_provider_error(error_message: object) -> bool:
    """Return whether a provider error should leave visible non-retried debt."""

    if not isinstance(error_message, str):
        return False
    normalized = " ".join(error_message.lower().split())
    return any(marker in normalized for marker in TERMINAL_PROVIDER_ERROR_MARKERS)


def embedding_error_class(error_message: object) -> str:
    """Classify provider failures without discarding their original evidence."""

    normalized = " ".join(str(error_message).lower().split())
    if "http 400" in normalized or "status 400" in normalized or "400 bad request" in normalized:
        return "provider_http_400"
    if "http 429" in normalized or "status 429" in normalized:
        return "provider_http_429"
    if "timeout" in normalized or "timed out" in normalized:
        return "provider_timeout"
    return "provider_error"


def archive_embeddable_message_where(alias: str = "m") -> str:
    """SQL predicate for authored prose messages eligible for embedding."""

    return f"""
{alias}.message_type = 'message'
AND {alias}.role IN ('user', 'assistant')
AND {alias}.material_origin IN ('human_authored', 'assistant_authored')
AND {alias}.word_count > 0
"""


def message_prose_sql(
    alias: str = "m",
    *,
    separator: str = "'\n'",
    block_types: tuple[str, ...] = ("text",),
) -> str:
    """Reconstruct message prose as ordered GROUP_CONCAT with block-type filtering.

    Messages have no ``text`` column; text lives in the ``blocks`` table
    (one row per content block). This builder concatenates block text in
    position order, filtering by block_type to exclude tool responses,
    thinking, protocol noise, etc.

    Args:
        alias: Table alias for the messages table (e.g., "m" for "messages AS m").
        separator: SQL string literal for joining blocks (default "'\n'" for backfill,
                   "char(10)||char(10)" for embeddings).
        block_types: Tuple of block_type values to include (default ("text",)
                     to exclude thinking, tool_result, etc.).

    Returns:
        A GROUP_CONCAT SQL expression (without "AS text" clause) that when selected
        with an alias produces ordered, filtered block prose.
        The expression uses a correlated subquery for proper ordering.

    Example:
        >>> prose_expr = message_prose_sql("m", separator="'\\n'", block_types=("text",))
        >>> query = f"SELECT m.message_id, {prose_expr} AS text FROM messages AS m ..."
    """
    # Format block_types as SQL list for IN clause
    block_types_sql = ", ".join(f"'{bt}'" for bt in block_types)

    return f"""GROUP_CONCAT(
        (
            SELECT b.text
            FROM blocks b
            WHERE b.message_id = {alias}.message_id
              AND b.block_type IN ({block_types_sql})
              AND b.text IS NOT NULL
            ORDER BY b.position
        ),
        {separator}
    )"""


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


def _configured_embedding_recipe() -> EmbeddingRecipe:
    cfg = load_polylogue_config()
    return EmbeddingRecipe.current(
        model=str(cfg.embedding_model),
        dimensions=int(cfg.embedding_dimension),
    )


def _archive_embedding_sibling_table(
    conn: sqlite3.Connection,
    status_table: str | None,
    table_name: str,
) -> str:
    if not status_table:
        return ""
    if "." in status_table:
        schema, _, _ = status_table.rpartition(".")
        candidate = f"{schema}.{table_name}"
    else:
        candidate = table_name
    return candidate if _qualified_table_exists(conn, candidate) else ""


def _archive_embedding_freshness_predicate(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    recipe: EmbeddingRecipe,
) -> _ArchiveEmbeddingFreshnessPredicate | None:
    """Build the single exact-key predicate used by every archive selector.

    A legacy archive without the v3 derivation ledger returns ``None`` and is
    evaluated by the pre-existing content-aware fallback below. Fresh v3
    archives always take this route. A missing embeddings database is simpler:
    every exact eligible source session is pending.
    """

    message_columns = _table_columns(conn, "messages")
    if not _archive_session_embeddable_counts_available(conn) or "content_hash" not in message_columns:
        return None

    state_table = _archive_embedding_sibling_table(conn, status_table, "embedding_derivation_state")
    meta_table = _archive_embedding_sibling_table(conn, status_table, "message_embeddings_meta")
    if status_table:
        required_status = {
            "session_id",
            "message_count_embedded",
            "needs_reindex",
            "error_message",
        }
        required_state = {
            "session_id",
            "generation",
            "derivation_key",
            "source_hash",
            "recipe_hash",
            "output_contract_hash",
            "attempt_state",
            "message_count",
        }
        required_meta = {
            "message_id",
            "content_hash",
            "needs_reindex",
            "recipe_hash",
            "derivation_key",
            "generation",
        }
        if (
            not state_table
            or not meta_table
            or not required_status.issubset(_qualified_table_columns(conn, status_table))
            or not required_state.issubset(_qualified_table_columns(conn, state_table))
            or not required_meta.issubset(_qualified_table_columns(conn, meta_table))
        ):
            return None

    register_embedding_identity_sql(conn)
    relation = archive_embeddable_messages_relation(conn, alias="desired_source")
    cte_sql = f"""
        WITH desired_messages AS (
            SELECT desired_source.message_id, desired_source.session_id, desired_source.content_hash
            FROM {relation}
        ),
        desired_sessions AS (
            SELECT
                session_id,
                COUNT(*) AS message_count,
                {EMBEDDING_SOURCE_HASH_SQL_FUNCTION}(message_id, content_hash) AS source_hash
            FROM desired_messages
            GROUP BY session_id
        )
    """
    if not status_table:
        return _ArchiveEmbeddingFreshnessPredicate(
            cte_sql=cte_sql,
            join_sql="",
            fresh_sql="0",
            blocked_sql="0",
            pending_sql="1",
        )

    recipe_hash_sql = f"X'{recipe.recipe_hash.hex()}'"
    output_hash_sql = f"X'{recipe.output_contract_hash.hex()}'"
    desired_key_sql = (
        f"{EMBEDDING_DERIVATION_KEY_SQL_FUNCTION}(s.session_id, ds.source_hash, {recipe_hash_sql}, {output_hash_sql})"
    )
    key_is_current = f"""(
        d.session_id IS NOT NULL
        AND d.derivation_key = {desired_key_sql}
        AND d.source_hash = ds.source_hash
        AND d.recipe_hash = {recipe_hash_sql}
        AND d.output_contract_hash = {output_hash_sql}
    )"""
    message_key_sql = (
        f"{EMBEDDING_MESSAGE_KEY_SQL_FUNCTION}(dm.message_id, dm.content_hash, {recipe_hash_sql}, {output_hash_sql})"
    )
    materialization_is_current = f"""(
        e.session_id IS NOT NULL
        AND COALESCE(e.needs_reindex, 0) = 0
        AND e.error_message IS NULL
        AND e.message_count_embedded = ds.message_count
        AND d.message_count = ds.message_count
        AND NOT EXISTS (
            SELECT 1
            FROM desired_messages AS dm
            LEFT JOIN {meta_table} AS em ON em.message_id = dm.message_id
            WHERE dm.session_id = s.session_id
              AND (
                    em.message_id IS NULL
                 OR COALESCE(em.needs_reindex, 0) = 1
                 OR em.content_hash != dm.content_hash
                 OR em.recipe_hash IS NULL
                 OR em.recipe_hash != {recipe_hash_sql}
                 OR em.derivation_key IS NULL
                 OR em.derivation_key != {message_key_sql}
                 OR em.generation != d.generation
              )
        )
    )"""
    fresh_sql = f"({key_is_current} AND d.attempt_state = 'succeeded' AND {materialization_is_current})"
    blocked_sql = f"({key_is_current} AND d.attempt_state = 'failed_terminal')"
    pending_sql = f"(NOT ({fresh_sql}) AND NOT ({blocked_sql}))"
    return _ArchiveEmbeddingFreshnessPredicate(
        cte_sql=cte_sql,
        join_sql=f"""
            LEFT JOIN {status_table} AS e ON e.session_id = s.session_id
            LEFT JOIN {state_table} AS d ON d.session_id = s.session_id
        """,
        fresh_sql=fresh_sql,
        blocked_sql=blocked_sql,
        pending_sql=pending_sql,
    )


def _select_pending_archive_session_window_by_derivation(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    recipe: EmbeddingRecipe,
    session_ids: tuple[str, ...],
    rebuild: bool,
    max_sessions: int | None,
    max_messages: int | None,
    min_messages: int | None,
) -> list[PendingSession] | None:
    predicate = _archive_embedding_freshness_predicate(
        conn,
        status_table=status_table,
        recipe=recipe,
    )
    if predicate is None:
        return None

    params: list[object] = []
    id_filter = ""
    if session_ids:
        placeholders = ", ".join("?" for _ in session_ids)
        id_filter = f"AND s.session_id IN ({placeholders})"
        params.extend(session_ids)
    floor_filter = "AND ds.message_count >= ?"
    params.append(max(1, min_messages or 1))
    ceiling_filter = ""
    if max_messages is not None:
        ceiling_filter = "AND ds.message_count <= ?"
        params.append(max_messages)
    pending_filter = "" if rebuild else f"AND {predicate.pending_sql}"

    rows = conn.execute(
        f"""
        {predicate.cte_sql}
        SELECT s.session_id, s.title, ds.message_count
        FROM desired_sessions AS ds
        JOIN sessions AS s ON s.session_id = ds.session_id
        {predicate.join_sql}
        WHERE 1 = 1
          {id_filter}
          {floor_filter}
          {ceiling_filter}
          {pending_filter}
        ORDER BY (s.sort_key_ms IS NULL), s.sort_key_ms DESC, s.session_id
        """,
        tuple(params),
    )

    pending: list[PendingSession] = []
    message_total = 0
    while True:
        batch = rows.fetchmany(500)
        if not batch:
            break
        for row in batch:
            session_id = str(_row_value(row, 0, "session_id"))
            title_value = _row_value(row, 1, "title")
            title = None if title_value is None else str(title_value)
            message_count = _row_int(row, 2, "message_count")
            if max_sessions is not None and len(pending) >= max_sessions:
                return pending
            if max_messages is not None and pending and message_total + message_count > max_messages:
                return pending
            pending.append(PendingSession(session_id=session_id, title=title, message_count=message_count))
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
    recipe: EmbeddingRecipe | None = None,
) -> list[PendingSession]:
    """Return one bounded pending-session window from a archive index.

    ``min_messages`` skips sessions below a message-count floor so a limited
    embedding budget is not spent on trivial sessions (provider smoke tests,
    empty stubs). It complements the budget bounds (``max_*``) with a quality
    floor, making selective embedding affordable for real users.
    """

    unique_ids = tuple(dict.fromkeys(session_ids or ()))
    if status_table is None:
        status_table = "embedding_status" if _table_exists(conn, "embedding_status") else ""
    derivation_window = _select_pending_archive_session_window_by_derivation(
        conn,
        status_table=status_table,
        recipe=recipe or _configured_embedding_recipe(),
        session_ids=unique_ids,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
        min_messages=min_messages,
    )
    if derivation_window is not None:
        return derivation_window

    # Compatibility for pre-v3 or deliberately minimal fixtures: this fallback
    # remains content-aware, but no caller can disable its stale checks.
    include_stale_checks = True
    pending: list[PendingSession] = []
    message_total = 0
    params: list[object] = []
    id_filter = ""
    if unique_ids:
        placeholders = ", ".join("?" for _ in unique_ids)
        id_filter = f"AND s.session_id IN ({placeholders})"
        params.extend(unique_ids)

    stale_message_table = _archive_embedding_meta_table_for_status(conn, status_table)
    stale_message_clause = (
        _archive_stale_message_clause(conn, stale_message_table, session_alias="s") if include_stale_checks else ""
    )
    exact_counts_available = _archive_session_embeddable_counts_available(conn)
    aggregate_expr = _archive_session_embeddable_count_expression(conn)
    clean_status_is_authoritative = _archive_session_embeddable_count_uses_prose_rollups(conn)
    clean_status_needs_exact_refinement = bool(
        include_stale_checks
        and exact_counts_available
        and clean_status_is_authoritative
        and status_table
        and not rebuild
    )
    status_select = (
        "e.session_id AS status_session_id, e.needs_reindex, e.message_count_embedded, e.error_message"
        if status_table
        else "NULL AS status_session_id, NULL AS needs_reindex, NULL AS message_count_embedded, NULL AS error_message"
    )
    join_clause = f"LEFT JOIN {status_table} e ON e.session_id = s.session_id" if status_table else ""
    max_message_filter = ""
    if exact_counts_available:
        count_select = (
            f"{aggregate_expr} AS estimated_message_count" if aggregate_expr else "NULL AS estimated_message_count"
        )
        floor_filter = f"AND {aggregate_expr} >= ?" if aggregate_expr else ""
        if floor_filter:
            params.append(max(1, min_messages or 1))
        max_message_filter = f"AND {aggregate_expr} <= ?" if aggregate_expr and max_messages is not None else ""
        if max_message_filter:
            params.append(max_messages)
        if rebuild or not status_table or aggregate_expr is None:
            pending_filter = ""
        elif include_stale_checks:
            pending_filter = f"""
              AND (
                    e.session_id IS NULL
                 OR e.needs_reindex = 1
                 OR (
                    e.error_message IS NULL
                    AND (
                        e.message_count_embedded < {aggregate_expr}
                        {stale_message_clause}
                    )
                 )
              )
            """
        else:
            pending_filter = """
              AND (
                    e.session_id IS NULL
                 OR e.needs_reindex = 1
              )
            """
    elif aggregate_expr is None:
        count_select = "NULL AS estimated_message_count"
        floor_filter = ""
        pending_filter = ""
    else:
        count_select = f"{aggregate_expr} AS estimated_message_count"
        floor_filter = f"AND {aggregate_expr} >= ?"
        params.append(max(1, min_messages or 1))
        max_message_filter = f"AND {aggregate_expr} <= ?" if max_messages is not None else ""
        if max_message_filter:
            params.append(max_messages)
        if rebuild or not status_table:
            pending_filter = ""
        elif include_stale_checks:
            pending_filter = f"""
              AND (
                    e.session_id IS NULL
                 OR e.needs_reindex = 1
                 OR (
                    e.error_message IS NULL
                    AND (
                        e.message_count_embedded < {aggregate_expr}
                        {stale_message_clause}
                    )
                 )
              )
            """
        else:
            pending_filter = """
              AND (
                    e.session_id IS NULL
                 OR e.needs_reindex = 1
              )
            """
    limit_clause = ""
    if aggregate_expr is not None and max_sessions is not None and not clean_status_needs_exact_refinement:
        limit_clause = "LIMIT ?"
        params.append(max_sessions)
    cursor = conn.execute(
        f"""
        SELECT s.session_id, s.title, {status_select}, {count_select}
        FROM sessions s
        {join_clause}
        WHERE 1 = 1
          {id_filter}
          {floor_filter}
          {max_message_filter}
          {pending_filter}
        ORDER BY (s.sort_key_ms IS NULL), s.sort_key_ms DESC, s.session_id
        {limit_clause}
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
            error_message = _row_value(row, 5, "error_message") if status_session_id is not None else None
            estimated_count = _row_int(row, 6, "estimated_message_count")
            clean_status_row = status_session_id is not None and error_message is None
            needs_exact_count = (
                include_stale_checks
                and exact_counts_available
                and clean_status_row
                and needs_reindex == 0
                and message_count_embedded > 0
                and (not clean_status_is_authoritative or message_count_embedded < estimated_count)
            )
            if needs_exact_count:
                estimated_count = count_archive_session_embeddable_messages(conn, session_id)
            if estimated_count <= 0:
                continue
            if min_messages is not None and estimated_count < min_messages:
                continue
            if not (
                rebuild
                or not status_table
                or status_session_id is None
                or needs_reindex == 1
                or (include_stale_checks and clean_status_row and message_count_embedded < estimated_count)
                or (
                    include_stale_checks
                    and clean_status_row
                    and archive_session_has_stale_embeddings(conn, session_id, stale_message_table)
                )
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


def _archive_embedding_meta_table_for_status(conn: sqlite3.Connection, status_table: str | None) -> str:
    if not status_table:
        return ""
    if "." in status_table:
        schema, _, _ = status_table.rpartition(".")
        candidate = f"{schema}.message_embeddings_meta"
    else:
        candidate = "message_embeddings_meta"
    return candidate if _qualified_table_exists(conn, candidate) else ""


def _qualified_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    if "." not in table:
        return _table_exists(conn, table)
    schema, _, name = table.rpartition(".")
    if not schema.replace("_", "").isalnum() or not name.replace("_", "").isalnum():
        return False
    try:
        row = conn.execute(
            f"SELECT 1 FROM {schema}.sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _qualified_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if "." not in table:
        return _table_columns(conn, table)
    schema, _, name = table.rpartition(".")
    if not schema.replace("_", "").isalnum() or not name.replace("_", "").isalnum():
        return set()
    try:
        rows = conn.execute(f"PRAGMA {schema}.table_info({name})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _archive_stale_message_clause(conn: sqlite3.Connection, meta_table: str, *, session_alias: str) -> str:
    if not meta_table or not _table_exists(conn, "messages"):
        return ""
    relation = archive_embeddable_messages_relation(conn, alias="stale_m")
    return f"""
                 OR EXISTS (
                    SELECT 1
                    FROM {relation}
                    JOIN {meta_table} em
                      ON em.message_id = stale_m.message_id
                    WHERE stale_m.session_id = {session_alias}.session_id
                      AND stale_m.content_hash IS NOT NULL
                      AND em.content_hash IS NOT NULL
                      AND em.content_hash != stale_m.content_hash
                 )
    """


def archive_session_has_stale_embeddings(conn: sqlite3.Connection, session_id: str, meta_table: str) -> bool:
    if not meta_table or not _table_exists(conn, "messages"):
        return False
    relation = archive_embeddable_messages_relation(conn, alias="stale_m")
    row = conn.execute(
        f"""
        SELECT 1
        FROM {relation}
        JOIN {meta_table} em
          ON em.message_id = stale_m.message_id
        WHERE stale_m.session_id = ?
          AND stale_m.content_hash IS NOT NULL
          AND em.content_hash IS NOT NULL
          AND em.content_hash != stale_m.content_hash
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()
    return row is not None


def count_archive_session_embeddable_messages(conn: sqlite3.Connection, session_id: str) -> int:
    """Count authored-prose messages eligible for embedding in one session."""

    if not _table_exists(conn, "messages"):
        return 0
    if _archive_message_blocks_available(conn):
        messages_ref = archive_embedding_messages_table_ref(conn, alias="m")
        blocks_ref = (
            "blocks AS b INDEXED BY idx_blocks_session_position"
            if _index_exists(conn, "idx_blocks_session_position")
            else "blocks AS b"
        )
        prose_expr = message_prose_sql("m", separator="char(10)||char(10)", block_types=("text",))
        row = conn.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT m.message_id, {prose_expr} AS text
                FROM {messages_ref}
                LEFT JOIN {blocks_ref}
                  ON b.session_id = m.session_id
                 AND b.message_id = m.message_id
                 AND b.block_type = 'text'
                 AND b.text IS NOT NULL
                WHERE {archive_embeddable_message_where("m")}
                  AND m.session_id = ?
                GROUP BY m.message_id, m.position, m.variant_index
                HAVING LENGTH(TRIM(COALESCE(text, ''))) >= 20
            ) embeddable_messages
            """,
            (session_id,),
        ).fetchone()
        return int(row[0] or 0) if row is not None else 0
    text_filter = ""
    if "text" in _table_columns(conn, "messages"):
        text_filter = "AND LENGTH(TRIM(COALESCE(m.text, ''))) >= 20"
    row = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM messages AS m
        WHERE {archive_embeddable_message_where("m")}
          AND m.session_id = ?
          {text_filter}
        """,
        (session_id,),
    ).fetchone()
    return int(row[0] or 0) if row is not None else 0


def count_archive_embedding_session_state(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    rebuild: bool = False,
    recipe: EmbeddingRecipe | None = None,
) -> ArchiveEmbeddingSessionState:
    """Count eligible sessions and their embedding completion state."""

    if not _table_exists(conn, "messages"):
        return ArchiveEmbeddingSessionState(eligible_sessions=0, embedded_sessions=0, pending_sessions=0)

    predicate = _archive_embedding_freshness_predicate(
        conn,
        status_table=status_table,
        recipe=recipe or _configured_embedding_recipe(),
    )
    if predicate is not None:
        pending_sql = "1" if rebuild else predicate.pending_sql
        fresh_sql = "0" if rebuild else predicate.fresh_sql
        blocked_sql = "0" if rebuild else predicate.blocked_sql
        row = conn.execute(
            f"""
            {predicate.cte_sql}
            SELECT
                COUNT(*) AS eligible_sessions,
                COALESCE(SUM(CASE WHEN {fresh_sql} THEN 1 ELSE 0 END), 0) AS embedded_sessions,
                COALESCE(SUM(CASE WHEN {pending_sql} THEN 1 ELSE 0 END), 0) AS pending_sessions,
                COALESCE(SUM(CASE WHEN {blocked_sql} THEN 1 ELSE 0 END), 0) AS blocked_sessions
            FROM desired_sessions AS ds
            JOIN sessions AS s ON s.session_id = ds.session_id
            {predicate.join_sql}
            """
        ).fetchone()
        if row is None:
            return ArchiveEmbeddingSessionState(eligible_sessions=0, embedded_sessions=0, pending_sessions=0)
        return ArchiveEmbeddingSessionState(
            eligible_sessions=int(row[0] or 0),
            embedded_sessions=int(row[1] or 0),
            pending_sessions=int(row[2] or 0),
            blocked_sessions=int(row[3] or 0),
        )

    stale_message_table = _archive_embedding_meta_table_for_status(conn, status_table)
    stale_session_clause = _archive_stale_message_clause(conn, stale_message_table, session_alias="s")
    stale_eligible_clause = _archive_stale_message_clause(conn, stale_message_table, session_alias="ec")
    aggregate_expr = _archive_session_embeddable_count_expression(conn)
    if aggregate_expr is not None:
        if rebuild or not status_table:
            row = conn.execute(
                f"""
                SELECT COUNT(*)
                FROM sessions s
                WHERE {aggregate_expr} > 0
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
            SELECT
                COUNT(*) AS eligible_sessions,
                SUM(
                    CASE
                        WHEN e.session_id IS NOT NULL
                         AND e.needs_reindex = 0
                         AND e.error_message IS NULL
                         AND NOT (
                            0
                            {stale_session_clause}
                         )
                        THEN 1 ELSE 0
                    END
                ) AS embedded_sessions,
                SUM(
                    CASE
                        WHEN e.session_id IS NULL
                          OR e.needs_reindex = 1
                          OR (
                            e.error_message IS NULL
                            AND (
                                0
                                {stale_session_clause}
                            )
                          )
                        THEN 1 ELSE 0
                    END
                ) AS pending_sessions,
                SUM(
                    CASE
                        WHEN e.session_id IS NOT NULL
                         AND e.needs_reindex = 0
                         AND e.error_message IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS blocked_sessions
            FROM sessions s
            LEFT JOIN {status_table} e ON e.session_id = s.session_id
            WHERE {aggregate_expr} > 0
            """
        ).fetchone()
        if row is None:
            return ArchiveEmbeddingSessionState(eligible_sessions=0, embedded_sessions=0, pending_sessions=0)
        return ArchiveEmbeddingSessionState(
            eligible_sessions=int(row[0] or 0),
            embedded_sessions=int(row[1] or 0),
            pending_sessions=int(row[2] or 0),
            blocked_sessions=int(row[3] or 0),
        )

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
                     AND e.error_message IS NULL
                     AND e.message_count_embedded >= ec.message_count
                     AND NOT (
                        0
                        {stale_eligible_clause}
                     )
                    THEN 1 ELSE 0
                END
            ) AS embedded_sessions,
                SUM(
                    CASE
                        WHEN e.session_id IS NULL
                          OR e.needs_reindex = 1
                          OR (
                            e.error_message IS NULL
                            AND (
                                e.message_count_embedded < ec.message_count
                                {stale_eligible_clause}
                            )
                          )
                        THEN 1 ELSE 0
                    END
                ) AS pending_sessions,
                SUM(
                    CASE
                        WHEN e.session_id IS NOT NULL
                         AND e.needs_reindex = 0
                         AND e.error_message IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) AS blocked_sessions
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
        blocked_sessions=int(row[3] or 0),
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


def archive_embeddable_messages_relation(conn: sqlite3.Connection, *, alias: str) -> str:
    """Return a relation containing messages the archive embedder will send."""

    message_columns = _table_columns(conn, "messages")
    base_alias = f"{alias}_base"
    messages_ref = archive_embedding_messages_table_ref(conn, alias=base_alias)
    content_hash_expr = f"{base_alias}.content_hash" if "content_hash" in message_columns else "NULL"
    selected_columns = (
        f"{base_alias}.message_id AS message_id, "
        f"{base_alias}.session_id AS session_id, "
        f"{content_hash_expr} AS content_hash"
    )
    base_where = archive_embeddable_message_where(base_alias)
    if _archive_message_blocks_available(conn):
        blocks_ref = (
            "blocks AS b INDEXED BY idx_blocks_session_position"
            if _index_exists(conn, "idx_blocks_session_position")
            else "blocks AS b"
        )
        prose_expr = message_prose_sql(base_alias, separator="char(10)||char(10)", block_types=("text",))
        return f"""
        (
            SELECT {selected_columns}
            FROM {messages_ref}
            LEFT JOIN {blocks_ref}
              ON b.session_id = {base_alias}.session_id
             AND b.message_id = {base_alias}.message_id
             AND b.block_type = 'text'
             AND b.text IS NOT NULL
            WHERE {base_where}
            GROUP BY {base_alias}.message_id, {base_alias}.session_id, {content_hash_expr}
            HAVING LENGTH(TRIM(COALESCE({prose_expr}, ''))) >= 20
        ) AS {alias}
        """
    if "text" in message_columns:
        return f"""
        (
            SELECT {selected_columns}
            FROM {messages_ref}
            WHERE {base_where}
              AND LENGTH(TRIM(COALESCE({base_alias}.text, ''))) >= 20
        ) AS {alias}
        """
    return f"""
    (
        SELECT {selected_columns}
        FROM {messages_ref}
        WHERE {base_where}
    ) AS {alias}
    """


def _archive_session_embeddable_count_expression(conn: sqlite3.Connection) -> str | None:
    """Return a minimal-fixture fallback when exact message columns are absent."""

    columns = _table_columns(conn, "sessions")
    if {"authored_user_message_count", "assistant_message_count"}.issubset(columns):
        return "COALESCE(s.authored_user_message_count, 0) + COALESCE(s.assistant_message_count, 0)"
    if "message_count" in columns:
        return "COALESCE(s.message_count, 0)"
    return None


def _archive_session_embeddable_count_uses_prose_rollups(conn: sqlite3.Connection) -> bool:
    columns = _table_columns(conn, "sessions")
    return {"authored_user_message_count", "assistant_message_count"}.issubset(columns)


def _archive_session_embeddable_counts_available(conn: sqlite3.Connection) -> bool:
    """Return whether exact per-session prose counts can be computed."""

    message_columns = _table_columns(conn, "messages")
    required_columns = {"session_id", "message_type", "role", "material_origin", "word_count"}
    return required_columns.issubset(message_columns)


def _archive_message_blocks_available(conn: sqlite3.Connection) -> bool:
    block_columns = _table_columns(conn, "blocks")
    required_columns = {"session_id", "message_id", "block_type", "text"}
    return required_columns.issubset(block_columns)


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
            if _table_exists(conn, "embedding_derivation_state"):
                conn.execute(
                    """
                    UPDATE embedding_derivation_state
                    SET generation = generation + 1,
                        attempt_state = 'pending',
                        message_count = 0,
                        updated_at_ms = ?
                    """,
                    (int(datetime.now(UTC).timestamp() * 1000),),
                )
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


class _ProviderRequestError(RuntimeError):
    """Marks an exception raised by the embedding provider call itself."""


def _archive_embedding_source_hash(rows: list[sqlite3.Row]) -> bytes:
    digest = EmbeddingSourceDigest()
    normalized: list[tuple[str, bytes]] = []
    for row in rows:
        content_hash = row["content_hash"]
        if content_hash is None:
            raise ValueError("content_hash is required for embedding source identity")
        normalized.append((str(row["message_id"]), bytes(content_hash)))
    for message_id, content_hash in sorted(normalized):
        digest.update(message_id, content_hash)
    return digest.digest()


def _read_archive_embedding_source_snapshot(
    conn: sqlite3.Connection,
    session_id: str,
) -> tuple[bytes, int]:
    relation = archive_embeddable_messages_relation(conn, alias="current_source")
    rows = conn.execute(
        f"""
        SELECT current_source.message_id, current_source.content_hash
        FROM {relation}
        WHERE current_source.session_id = ?
        ORDER BY current_source.message_id
        """,
        (session_id,),
    ).fetchall()
    return _archive_embedding_source_hash(rows), len(rows)


def embed_archive_session_sync(
    index_db_path: Path,
    vec_provider: VectorProvider,
    session_id: str,
) -> EmbedSessionOutcome:
    """Embed one archive session with exact-key, generation-guarded publication."""

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
    attempted_message_refs: tuple[str, ...] = ()
    attempt = None
    session: sqlite3.Row | None = None
    embeddable: list[sqlite3.Row] = []
    try:
        from polylogue.storage.sqlite.archive_tiers.embedding_write import (
            ArchiveEmbeddingWrite,
            begin_embedding_attempt,
            complete_embedding_attempt_success,
            supersede_embedding_attempt,
        )
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
        prose_expr = message_prose_sql("m", separator="char(10)||char(10)", block_types=("text",))
        rows = index_conn.execute(
            f"""
            SELECT m.message_id, m.role, m.content_hash, m.material_origin, m.message_type,
                   {prose_expr} AS text
            FROM {messages_ref}
            LEFT JOIN blocks AS b INDEXED BY idx_blocks_session_position
              ON b.session_id = m.session_id
             AND b.message_id = m.message_id
             AND b.block_type = 'text'
             AND b.text IS NOT NULL
            WHERE m.session_id = ?
              AND {archive_embeddable_message_where("m")}
            GROUP BY m.message_id, m.role, m.content_hash, m.material_origin, m.message_type,
                     m.position, m.variant_index
            ORDER BY m.position, m.variant_index
            """,
            (session_id,),
        ).fetchall()
        embeddable = [
            row
            for row in rows
            if _should_embed_archive_message(row["material_origin"], row["message_type"], row["role"], row["text"])
        ]

        recipe = EmbeddingRecipe.current(
            model=str(text_provider.model),
            dimensions=int(text_provider.dimension),
        )
        source_hash = _archive_embedding_source_hash(embeddable)
        attempt = begin_embedding_attempt(
            embeddings_conn,
            session_id=session_id,
            origin=str(session["origin"]),
            source_hash=source_hash,
            recipe=recipe,
        )

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        writes: list[ArchiveEmbeddingWrite] = []
        batch_size = max(1, ARCHIVE_EMBED_MESSAGE_BATCH_SIZE)
        for start in range(0, len(embeddable), batch_size):
            batch = embeddable[start : start + batch_size]
            attempted_message_refs = tuple(str(row["message_id"]) for row in batch)
            try:
                vectors = text_provider._get_embeddings(
                    [str(row["text"]) for row in batch],
                    input_type="document",
                )
            except Exception as exc:
                raise _ProviderRequestError(str(exc)) from exc
            if len(vectors) != len(batch):
                raise _ProviderRequestError("embedding provider returned a mismatched vector count")
            for row, vector in zip(batch, vectors, strict=True):
                content_hash = row["content_hash"]
                if content_hash is None:
                    raise ValueError("content_hash is required for message embedding metadata")
                content_hash_bytes = bytes(content_hash)
                writes.append(
                    ArchiveEmbeddingWrite(
                        message_id=str(row["message_id"]),
                        session_id=session_id,
                        origin=str(session["origin"]),
                        embedding=vector,
                        model=text_provider.model,
                        embedded_at_ms=now_ms,
                        content_hash=content_hash_bytes,
                        recipe_hash=attempt.recipe_hash,
                        derivation_key=message_embedding_derivation_key(
                            message_id=str(row["message_id"]),
                            content_hash=content_hash_bytes,
                            recipe=recipe,
                        ).digest(),
                        generation=attempt.generation,
                    )
                )

        current_source_hash, current_message_count = _read_archive_embedding_source_snapshot(index_conn, session_id)
        current_recipe = _configured_embedding_recipe()
        if (
            current_source_hash != attempt.source_hash
            or current_message_count != len(embeddable)
            or current_recipe.recipe_hash != attempt.recipe_hash
            or current_recipe.output_contract_hash != attempt.output_contract_hash
        ):
            supersede_embedding_attempt(
                embeddings_conn,
                attempt=attempt,
                source_hash=current_source_hash,
                recipe=current_recipe,
            )
            return EmbedSessionOutcome(
                status="error",
                session_id=session_id,
                title=None if session["title"] is None else str(session["title"]),
                error="embedding source or recipe changed during materialization; retry queued",
            )

        committed = complete_embedding_attempt_success(
            embeddings_conn,
            attempt=attempt,
            writes=writes,
            completed_at_ms=now_ms,
        )
        if not committed:
            return EmbedSessionOutcome(
                status="error",
                session_id=session_id,
                title=None if session["title"] is None else str(session["title"]),
                error="embedding attempt was superseded before publication; retry queued",
            )
    except Exception as exc:
        from polylogue.storage.sqlite.archive_tiers.embedding_write import record_embedding_failure

        origin_row = None
        with contextlib.suppress(sqlite3.Error):
            origin_row = index_conn.execute(
                "SELECT origin FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if origin_row is not None:
            if isinstance(exc, _ProviderRequestError):
                provider = "voyage"
                error_class = embedding_error_class(exc)
                retryable = not is_terminal_embedding_provider_error(str(exc))
            else:
                provider = "local"
                error_class = "internal_error"
                retryable = True
            record_embedding_failure(
                embeddings_conn,
                session_id=session_id,
                origin=str(origin_row["origin"]),
                message_refs=attempted_message_refs,
                provider=provider,
                model=text_provider.model,
                error_class=error_class,
                error_message=str(exc),
                retryable=retryable,
                attempt=attempt,
            )
        return EmbedSessionOutcome(status="error", session_id=session_id, error=str(exc))
    finally:
        with contextlib.suppress(sqlite3.Error):
            index_conn.close()
        with contextlib.suppress(sqlite3.Error):
            embeddings_conn.close()

    assert session is not None
    if not embeddable:
        no_op_status: EmbedSingleStatus = (
            "no_messages" if int(session["message_count"] or 0) <= 0 else "no_embeddable_messages"
        )
        return EmbedSessionOutcome(
            status=no_op_status,
            session_id=session_id,
            title=None if session["title"] is None else str(session["title"]),
        )
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
    model: str | None = None,
) -> None:
    """Record the terminal write of one archive-session embed pass.

    ``model`` is the model the embed pass actually used (``text_provider.model``
    at the call site), when the pass computed any embeddings. It is compared
    against the *currently* configured model to close a race with
    ``_reconcile_embedding_config_change`` (``daemon/convergence_stages.py``):
    that function can bulk-mark ``needs_reindex = 1`` on every session while
    an embed pass for this session is already mid-flight under the
    previously-configured model. If this terminal write blindly cleared
    ``needs_reindex``, it would silently clobber that reindex request and
    leave the session marked "fresh" while holding stale-model embeddings.

    So: only clear ``needs_reindex`` when the model used for this pass still
    matches what's configured *now*. A mismatch means the configuration
    moved on since this pass started reading messages — its embeddings are
    already superseded, so ``needs_reindex`` is left set (or forced back to 1)
    so the session gets picked up again under the new model.

    ``model=None`` covers the "nothing to embed" outcomes (no messages / no
    embeddable messages): no embedding was computed, so there is nothing that
    can be stale, and the clear is unconditionally safe.
    """
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    if model is None:
        needs_reindex = 0
    else:
        configured_model = load_polylogue_config().embedding_model
        needs_reindex = 0 if model == configured_model else 1
    with conn:
        conn.execute(
            """
            INSERT INTO embedding_status (
                session_id, origin, message_count_embedded, last_embedded_at_ms, needs_reindex, error_message
            ) VALUES (?, ?, ?, ?, ?, NULL)
            ON CONFLICT(session_id) DO UPDATE SET
                origin = excluded.origin,
                message_count_embedded = excluded.message_count_embedded,
                last_embedded_at_ms = excluded.last_embedded_at_ms,
                needs_reindex = excluded.needs_reindex,
                error_message = NULL
            """,
            (session_id, origin, message_count, now_ms, needs_reindex),
        )
    # Every terminal success outcome — including "nothing to embed" — resolves
    # the session's open failures, or they linger as phantom debt.
    from polylogue.storage.sqlite.archive_tiers.embedding_write import resolve_open_embedding_failures_for_session

    resolve_open_embedding_failures_for_session(conn, session_id=session_id)


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
    "archive_embeddable_messages_relation",
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
