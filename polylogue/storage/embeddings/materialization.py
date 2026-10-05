"""Substrate-side embedding execution (no CLI / click coupling).

Provides three primitives that surfaces compose into their own UI:

* :func:`iter_pending_sessions` — list sessions that need embedding.
* :func:`embed_archive_session_sync` — embed messages for one session.
* :class:`EmbedSessionOutcome` — typed outcome record.

The daemon's embedding convergence owns execution; no CLI route embeds
in-process.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

from polylogue.config import load_polylogue_config
from polylogue.core.enums import Origin
from polylogue.core.sqlite_introspection import index_exists as _index_exists
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.storage.archive_identity import archive_root_for_index_path, demo_owned_session_ids
from polylogue.storage.archive_tuple_location import InactiveTierDestination
from polylogue.storage.embeddings.generations import EmbeddingGenerationBinding
from polylogue.storage.embeddings.identity import (
    EMBEDDING_DERIVATION_KEY_SQL_FUNCTION,
    EMBEDDING_SOURCE_HASH_SQL_FUNCTION,
    VECTOR_DERIVATION_HASH_SQL_FUNCTION,
    EmbeddingProvenanceError,
    EmbeddingRecipe,
    EmbeddingRequestSpec,
    EmbeddingSourceDigest,
    available_embedding_predicate,
    message_embedding_derivation_key,
    register_embedding_identity_sql,
)
from polylogue.storage.embeddings.tuple_generation import (
    EmbeddingTupleGeneration,
)
from polylogue.storage.embeddings.tuple_generation import (
    prepare_inactive_embedding_generation as _prepare_inactive_embedding_generation,
)
from polylogue.storage.sqlite.connection_profile import (
    open_isolated_write_connection,
    open_readonly_connection,
)


def ensure_embedding_lifecycle(archive_root: Path, *, active_path: Path | None = None) -> Path:
    """Enter the archive-bound generation collector before embedding writes."""
    from polylogue.storage.embeddings.generations import ensure_embedding_lifecycle as _ensure

    return _ensure(archive_root, active_path=active_path)


def prepare_inactive_archive_embedding_generation(
    destination: InactiveTierDestination,
    *,
    recipe: EmbeddingRecipe,
    source_generation: str,
    index_generation: str,
) -> EmbeddingTupleGeneration:
    """Construct the embeddings member of a daemon-owned inactive tuple."""
    return _prepare_inactive_embedding_generation(
        destination,
        recipe=recipe,
        source_generation=source_generation,
        index_generation=index_generation,
    )


def resolve_embedding_failure_with_lifecycle(
    embeddings_db: Path,
    *,
    failure_id: str,
    action: Literal["acknowledge", "requeue", "supersede"],
    note: str | None = None,
    superseded_by: str | None = None,
) -> ArchiveEmbeddingFailure:
    """Apply a failure resolution while holding the lifecycle writer lock."""
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore
    from polylogue.storage.sqlite.archive_tiers.embedding_write import resolve_embedding_failure
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease("embedding failure resolution", archive_root=embeddings_db.parent)
    store = EmbeddingGenerationStore(embeddings_db.parent, active_path=embeddings_db)
    with store.writer_lock(prepare_active=True) as binding:
        store.assert_binding(binding)
        conn = open_isolated_write_connection(
            binding.database_path,
            purpose="embedding failure resolution",
            timeout=30.0,
            archive_root=binding.archive_root,
        )
        try:
            with conn:
                return resolve_embedding_failure(
                    conn,
                    failure_id=failure_id,
                    action=action,
                    note=note,
                    superseded_by=superseded_by,
                )
        finally:
            conn.close()


if TYPE_CHECKING:
    from polylogue.core.protocols import VectorProvider
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore
    from polylogue.storage.repository.repository_contracts import RepositoryBackendProtocol
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        ArchiveEmbeddingAttempt,
        ArchiveEmbeddingFailure,
        ArchiveEmbeddingWrite,
    )


T = TypeVar("T")

EmbedSingleStatus = Literal["embedded", "no_messages", "no_embeddable_messages", "not_found", "error", "deferred"]
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


class EmbeddingAcquisitionExcludedError(RuntimeError):
    """The completed demo owner excludes this exact session from acquisition."""


def embedding_acquisition_allowed(conn: sqlite3.Connection, session_id: str) -> bool:
    """Apply acquisition policy without certifying or deleting stored outputs."""
    with contextlib.closing(conn.execute("PRAGMA database_list")) as cursor:
        index_path = next((str(row[2]) for row in cursor if row[1] == "main"), "")
    return not index_path or session_id not in demo_owned_session_ids(archive_root_for_index_path(Path(index_path)))


def embedding_acquisition_predicate(conn: sqlite3.Connection, alias: str) -> str:
    """Pin the completed demo membership once for this SQL work selection."""
    with contextlib.closing(conn.execute("PRAGMA database_list")) as cursor:
        index_path = next((str(row[2]) for row in cursor if row[1] == "main"), "")
    excluded = demo_owned_session_ids(archive_root_for_index_path(Path(index_path))) if index_path else frozenset()
    conn.create_function("polylogue_embedding_acquisition_allowed", 1, lambda sid: int(sid not in excluded))
    return f"polylogue_embedding_acquisition_allowed({alias}.session_id)"


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
        A self-contained scalar subquery (without "AS text" clause) that when
        selected with an alias produces ordered, filtered, concatenated block
        prose for that one message. The GROUP_CONCAT aggregation happens
        *inside* the subquery, over a pre-sorted derived table -- it does not
        rely on the outer query's own GROUP BY/JOIN shape. This matters
        because every caller LEFT JOINs `blocks` to filter/order by block
        columns, which fans a multi-block message out into multiple outer
        rows; wrapping GROUP_CONCAT around a *correlated scalar* subquery
        (the previous implementation) evaluated that scalar once per fanned-
        out row and returned only a single block's text, repeated -- multi
        TEXT-block messages silently lost every block but the first.

    Example:
        >>> prose_expr = message_prose_sql("m", separator="'\\n'", block_types=("text",))
        >>> query = f"SELECT m.message_id, {prose_expr} AS text FROM messages AS m ..."
    """
    # Format block_types as SQL list for IN clause
    block_types_sql = ", ".join(f"'{bt}'" for bt in block_types)

    return f"""(
        SELECT GROUP_CONCAT(prose_block.text, {separator})
        FROM (
            SELECT b.text
            FROM blocks b
            WHERE b.message_id = {alias}.message_id
              AND b.block_type IN ({block_types_sql})
              AND b.text IS NOT NULL
            ORDER BY b.position
        ) AS prose_block
    )"""


@dataclass(frozen=True, slots=True)
class EmbedSessionOutcome:
    """Typed outcome for embedding one session."""

    status: EmbedSingleStatus
    session_id: str
    title: str | None = None
    embedded_message_count: int = 0
    error: str | None = None
    deferred: bool = False


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
    archive_root: Path,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
) -> list[PendingSession]:
    """Return sessions needing embedding.

    With ``rebuild=True`` returns every session; otherwise returns
    rows missing from ``embedding_status`` or flagged ``needs_reindex``.
    """
    from polylogue.storage.sqlite.connection import open_read_connection

    with open_read_connection(backend.db_path, archive_root=archive_root) as conn:
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
    min_messages: int | None = None,
    limit_reached: list[bool] | None = None,
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

    # Filter in SQL before the session or message window is applied.
    if min_messages is not None and min_messages > 0:
        id_filter += " AND c.message_count >= ?"
        params.append(min_messages)

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
                if limit_reached is not None:
                    limit_reached.append(True)
                return pending
            if max_messages is not None and message_count > max_messages:
                if limit_reached is not None:
                    limit_reached.append(True)
                continue
            if max_messages is not None and pending and message_total + message_count > max_messages:
                if limit_reached is not None:
                    limit_reached.append(True)
                return pending
            pending.append(
                PendingSession(
                    session_id=session_id,
                    title=title,
                    message_count=message_count,
                )
            )
            message_total += message_count
    return pending


def _configured_embedding_recipe() -> EmbeddingRecipe:
    cfg = load_polylogue_config()
    return EmbeddingRecipe.current(
        model=str(cfg.embedding_model),
        dimensions=int(cfg.embedding_dimension),
    )


def _archive_embedding_sibling_table(status_table: str, table_name: str) -> str:
    """Name ``table_name`` in the same schema as ``status_table``.

    The embeddings DDL creates the status, derivation-ledger, and vector
    metadata tables together, so a sibling of an existing status table exists.
    """
    schema, dot, _ = status_table.rpartition(".")
    return f"{schema}{dot}{table_name}"


def _archive_embedding_freshness_predicate(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    recipe: EmbeddingRecipe,
) -> _ArchiveEmbeddingFreshnessPredicate:
    """Select missing outputs separately from free occurrence-binding work.

    Attempt receipts retain their exact computation identity and terminal
    refusal; they do not invalidate a usable retained output after an
    explicitly compatible document-model selection.
    """

    register_embedding_identity_sql(conn)
    relation = archive_embeddable_messages_relation(conn, alias="desired_source", recipe=recipe)
    cte_sql = f"""
        WITH desired_messages AS (
            SELECT desired_source.message_id, desired_source.session_id, desired_source.content_hash, desired_source.origin, desired_source.text, desired_source.vector_derivation_hash
            FROM {relation}
        ),
        desired_sessions AS (
            SELECT
                session_id,
                COUNT(*) AS message_count,
                {EMBEDDING_SOURCE_HASH_SQL_FUNCTION}(vector_derivation_hash) AS source_hash
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

    state_table = _archive_embedding_sibling_table(status_table, "embedding_derivation_state")
    meta_table = _archive_embedding_sibling_table(status_table, "message_embeddings_meta")
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
    refs_table = _archive_embedding_sibling_table(status_table, "message_embedding_refs")
    vectors_table = _archive_embedding_sibling_table(status_table, "message_embeddings")
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        raise RuntimeError(f"embedding vector inspection unavailable: {error}")
    valid = available_embedding_predicate(
        recipe=recipe,
        source="dm",
        refs="r",
        meta="em",
        vectors_table=vectors_table,
        meta_table=meta_table,
    )
    fresh_sql = f"""NOT EXISTS (
        SELECT 1 FROM desired_messages AS dm
        WHERE dm.session_id = s.session_id AND NOT EXISTS (
            SELECT 1 FROM (SELECT 1) AS anchor
            LEFT JOIN {refs_table} AS r ON r.message_id = dm.message_id
            LEFT JOIN {meta_table} AS em ON em.vector_derivation_hash = r.vector_derivation_hash
            WHERE COALESCE({valid}, 0)
        )
    )"""
    blocked_sql = f"(NOT ({fresh_sql}) AND {key_is_current} AND d.attempt_state = 'failed_terminal')"
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


def archive_embedding_blocked_counts_sql(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    recipe: EmbeddingRecipe,
) -> str | None:
    """Return SQL counting session keys whose derivation is terminally refused.

    "Blocked" is not a parallel status counter: it is the embeddings domain's
    own ``blocked`` branch of the freshness predicate that
    ``_select_pending_archive_session_window_by_derivation`` uses to keep these
    keys out of the work set.  A key is blocked when its *current* derivation
    key (session, source hash, recipe hash, output contract) carries
    ``attempt_state = 'failed_terminal'`` -- a refusal that no automatic retry
    will clear.  Such a key is neither valid nor pending: reporting it as
    pending claims work the writer will never do.

    The two counted columns are the blocked session keys and, within them, the
    required messages that still have no vector, so a caller can subtract a
    blocked key's backlog from a message-level pending count on the same basis.
    Returns ``None`` when the archive cannot express the classification.
    """

    predicate = _archive_embedding_freshness_predicate(
        conn,
        status_table=status_table,
        recipe=recipe,
    )
    if predicate.blocked_sql == "0":
        return None
    meta_table = _archive_embedding_sibling_table(status_table, "message_embeddings_meta")
    refs_table = _archive_embedding_sibling_table(status_table, "message_embedding_refs")
    vectors_table = _archive_embedding_sibling_table(status_table, "message_embeddings")
    valid = available_embedding_predicate(
        recipe=recipe,
        source="dm",
        refs="r",
        meta="em",
        vectors_table=vectors_table,
        meta_table=meta_table,
    )
    unembedded_sql = f"""(SELECT COUNT(*) FROM desired_messages AS dm
        WHERE dm.session_id = s.session_id AND NOT EXISTS (
            SELECT 1 FROM (SELECT 1) AS anchor
            LEFT JOIN {refs_table} AS r ON r.message_id = dm.message_id
            LEFT JOIN {meta_table} AS em ON em.vector_derivation_hash = r.vector_derivation_hash
            WHERE COALESCE({valid}, 0)
        ))"""
    return f"""
        {predicate.cte_sql}
        SELECT COUNT(*), COALESCE(SUM({unembedded_sql}), 0)
        FROM desired_sessions AS ds
        JOIN sessions AS s ON s.session_id = ds.session_id
        {predicate.join_sql}
        WHERE {predicate.blocked_sql}
    """


def archive_embedding_session_window_sql(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    recipe: EmbeddingRecipe,
    session_ids: tuple[str, ...],
    rebuild: bool,
    max_sessions: int | None,
    max_messages: int | None,
    min_messages: int | None,
) -> tuple[str, tuple[object, ...]]:
    predicate = _archive_embedding_freshness_predicate(
        conn,
        status_table=status_table,
        recipe=recipe,
    )

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
    acquisition_filter = embedding_acquisition_predicate(conn, "s")

    sql = f"""
        {predicate.cte_sql}
        , window_candidates AS (
        SELECT s.session_id, s.title, ds.message_count, s.sort_key_ms
        FROM desired_sessions AS ds
        JOIN sessions AS s ON s.session_id = ds.session_id
        {predicate.join_sql}
        WHERE 1 = 1
          {id_filter}
          {floor_filter}
          {ceiling_filter}
          {pending_filter}
          AND {acquisition_filter}
        ), window_ranked AS (
            SELECT session_id, title, message_count,
                   ROW_NUMBER() OVER (ORDER BY (sort_key_ms IS NULL), sort_key_ms DESC, session_id) AS ordinal,
                   SUM(message_count) OVER (
                       ORDER BY (sort_key_ms IS NULL), sort_key_ms DESC, session_id
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                   ) AS message_total
            FROM window_candidates
        )
        SELECT session_id, title, message_count FROM window_ranked WHERE 1 = 1
        """
    if max_sessions is not None:
        sql += " AND ordinal <= ?"
        params.append(max_sessions)
    if max_messages is not None:
        sql += " AND message_total <= ?"
        params.append(max_messages)
    return sql + " ORDER BY ordinal", tuple(params)


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
) -> list[PendingSession]:
    sql, params = archive_embedding_session_window_sql(
        conn,
        status_table=status_table,
        recipe=recipe,
        session_ids=session_ids,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
        min_messages=min_messages,
    )
    pending: list[PendingSession] = []
    with contextlib.closing(conn.execute(sql, params)) as rows:
        while batch := rows.fetchmany(500):
            for row in batch:
                title_value = _row_value(row, 1, "title")
                pending.append(
                    PendingSession(
                        session_id=str(_row_value(row, 0, "session_id")),
                        title=None if title_value is None else str(title_value),
                        message_count=_row_int(row, 2, "message_count"),
                    )
                )
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
    return _select_pending_archive_session_window_by_derivation(
        conn,
        status_table=status_table,
        recipe=recipe or _configured_embedding_recipe(),
        session_ids=unique_ids,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=max_messages,
        min_messages=min_messages,
    )


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


def archive_embeddable_messages_relation(conn: sqlite3.Connection, *, alias: str, recipe: EmbeddingRecipe) -> str:
    """Return a relation containing messages the archive embedder will send.

    The relation projects ``message_id``/``session_id``/``content_hash`` and
    ``vector_derivation_hash`` -- computed via the registered SQL function from
    exactly the same prose expression that will be sent to the embedder -- the
    identity-free vector key the freshness predicate, embedding
    materialization, and rescue compare against.

    The *whole* recipe is carried, not just its model: addresses built from the
    model alone had to assume ``dimensions=1024``, so at any other configured
    dimension the SQL-side address disagreed with the embed-time one and every
    message stayed pending forever (polylogue-crcst).
    """

    base_alias = f"{alias}_base"
    messages_ref = archive_embedding_messages_table_ref(conn, alias=base_alias)
    content_hash_expr = f"{base_alias}.content_hash"
    origin_expr = f"(SELECT source_session.origin FROM sessions AS source_session WHERE source_session.session_id = {base_alias}.session_id)"
    register_embedding_identity_sql(conn, recipe=recipe)
    recipe_literal = f"X'{recipe.recipe_hash.hex()}'"

    base_where = archive_embeddable_message_where(base_alias)
    blocks_ref = (
        "blocks AS b INDEXED BY idx_blocks_session_position"
        if _index_exists(conn, "idx_blocks_session_position")
        else "blocks AS b"
    )
    prose_expr = message_prose_sql(base_alias, separator="char(10)||char(10)", block_types=("text",))
    hash_expr = f"{VECTOR_DERIVATION_HASH_SQL_FUNCTION}({recipe_literal}, {prose_expr})"
    selected_columns = (
        f"{base_alias}.message_id AS message_id, "
        f"{base_alias}.session_id AS session_id, "
        f"{content_hash_expr} AS content_hash, {origin_expr} AS origin, "
        f"{hash_expr} AS vector_derivation_hash, {prose_expr} AS text"
    )
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


def mark_all_archive_sessions_needs_reindex(index_db_path: Path, *, embeddings_db_path: Path | None = None) -> None:
    """Flag every archive session for embedding rebuild under lifecycle admission."""
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore
    from polylogue.storage.sqlite.write_lease import require_write_lease

    resolved_embeddings = (
        embeddings_db_path if embeddings_db_path is not None else index_db_path.with_name("embeddings.db")
    )
    require_write_lease("embedding lifecycle bootstrap", archive_root=resolved_embeddings.parent)
    if not resolved_embeddings.exists():
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        initialize_archive_database(resolved_embeddings, ArchiveTier.EMBEDDINGS)
    store = EmbeddingGenerationStore(resolved_embeddings.parent, active_path=resolved_embeddings)
    with store.writer_lock() as binding:
        _mark_all_archive_sessions_needs_reindex(index_db_path, binding)
        store.assert_binding(binding)


def _mark_all_archive_sessions_needs_reindex(
    index_db_path: Path, embeddings_binding: EmbeddingGenerationBinding
) -> None:
    conn = open_isolated_write_connection(
        embeddings_binding.database_path,
        purpose="embedding lifecycle reindex",
        timeout=30.0,
        archive_root=embeddings_binding.archive_root,
    )
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


class _ProviderRequestError(RuntimeError):
    """Marks an exception raised by the embedding provider call itself."""


def _present_vector_addresses(
    conn: sqlite3.Connection, hashes: Iterable[bytes], *, recipe: EmbeddingRecipe
) -> set[bytes]:
    """Return the subset of ``hashes`` that already own a vector in this tier."""

    wanted = sorted(set(hashes))
    present: set[bytes] = set()
    chunk = 500
    for start in range(0, len(wanted), chunk):
        window = wanted[start : start + chunk]
        placeholders = ",".join("?" for _ in window)
        with contextlib.closing(
            conn.execute(
                f"""SELECT em.vector_derivation_hash, em.model, em.dimension, em.recipe_hash, em.output_contract_hash,
                       v.vector_derivation_hash IS NOT NULL
                FROM message_embeddings_meta AS em
                LEFT JOIN message_embeddings AS v ON v.vector_derivation_hash = lower(hex(em.vector_derivation_hash))
                WHERE em.vector_derivation_hash IN ({placeholders})""",
                window,
            )
        ) as cursor:
            rows = cursor.fetchall()
        for address, model, dimension, recipe_hash, output_hash, vector_present in rows:
            producer = recipe.proven_stored_producer(
                model=str(model), dimension=int(dimension), recipe_hash=bytes(recipe_hash)
            )
            if producer is None:
                raise EmbeddingProvenanceError("stored embedding producer provenance is unproven")
            if (
                producer.model == recipe.model
                and recipe.retrieval_compatible(replace(producer, input_schema_version=recipe.input_schema_version))
                and bytes(output_hash) == producer.output_contract_hash
            ):
                if vector_present:
                    present.add(bytes(address))
            else:
                raise EmbeddingProvenanceError("stored embedding output is incompatible with the selected producer")
    return present


def _archive_embedding_source_hash_from_pairs(pairs: Iterable[tuple[str, bytes]]) -> bytes:
    """Session source identity from ``(message_id, vector_derivation_hash)`` pairs.

    Only the hash VALUES are digested (message_id is accepted for caller
    convenience -- e.g. a ``dict[message_id, hash].items()`` -- but excluded
    from the digest itself): source identity must stay content-only so a
    message-set that is renumbered by a rebuild but unchanged in content
    does not bust the session's derivation key (polylogue-q88p).
    """
    digest = EmbeddingSourceDigest()
    for _message_id, input_hash in sorted(pairs, key=lambda item: item[1]):
        digest.update(input_hash)
    return digest.digest()


def _archive_embedding_source_hash(rows: list[sqlite3.Row]) -> bytes:
    """Source identity from a relation carrying a precomputed ``vector_derivation_hash`` column."""

    normalized: list[tuple[str, bytes]] = []
    for row in rows:
        input_hash = row["vector_derivation_hash"]
        if input_hash is None:
            raise ValueError("vector_derivation_hash is required for embedding source identity")
        normalized.append((str(row["message_id"]), bytes(input_hash)))
    return _archive_embedding_source_hash_from_pairs(normalized)


def _read_archive_embedding_source_snapshot(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    recipe: EmbeddingRecipe,
) -> tuple[bytes, int, tuple[str, ...]]:
    """Re-read live source identity to detect drift during materialization.

    Queried fresh (not from the Python-side hashes computed before the
    provider call) so a text/message-set change racing the embed pass is
    caught: the relation recomputes ``vector_derivation_hash`` from whatever
    text is in ``index.db`` right now.
    """
    relation = archive_embeddable_messages_relation(conn, alias="current_source", recipe=recipe)
    rows = conn.execute(
        f"""
        SELECT current_source.message_id, current_source.vector_derivation_hash
        FROM {relation}
        WHERE current_source.session_id = ?
        ORDER BY current_source.message_id
        """,
        (session_id,),
    ).fetchall()
    return _archive_embedding_source_hash(rows), len(rows), tuple(sorted(str(row["message_id"]) for row in rows))


def inline_embedding_admission(actor: str, function: Callable[[], T]) -> T:
    """Run one embedding write phase directly: the caller is its own writer."""
    del actor
    return function()


class EmbeddingWriteAdmission(Protocol):
    """Run one short archive-embedding write phase under writer authority.

    The provider call may not run while the writer lease or the embedding
    generation lock is held, so the archive route is a sequence of *admitted*
    phases separated by lease-free computation: reserve the attempt, embed,
    publish each window, finalize. A caller that is already the process's sole
    writer passes nothing and gets :func:`inline_embedding_admission`; the
    daemon passes an adapter that admits every phase through its write
    coordinator, so each phase's authority begins and ends inside the phase.
    """

    def __call__(self, actor: str, function: Callable[[], T], /) -> T: ...


@dataclass(frozen=True, slots=True)
class _ArchiveEmbeddingInput:
    """One message the provider still has to embed for this attempt."""

    message_id: str
    text: str
    input_hash: bytes
    message_content_hash: bytes | None


@dataclass(frozen=True, slots=True)
class _ArchiveEmbeddingPlan:
    """The immutable input snapshot one reserved attempt was computed from.

    Everything the provider call and the publication phases need is captured
    here while the generation lock is held, so computation reads no database
    and publication revalidates against exactly what was reserved.
    """

    index_db_path: Path
    embeddings_path: Path
    binding: EmbeddingGenerationBinding
    session_id: str
    origin: str
    title: str | None
    session_message_count: int
    model: str
    recipe: EmbeddingRecipe
    configured_recipe_before: EmbeddingRecipe
    attempt: ArchiveEmbeddingAttempt
    embeddable_message_ids: tuple[str, ...]
    pending_count: int
    to_embed: tuple[_ArchiveEmbeddingInput, ...]
    now_ms: int
    published_before_compute: int


def _embedding_lifecycle_store(embeddings_path: Path) -> EmbeddingGenerationStore:
    from polylogue.storage.embeddings.generations import EmbeddingGenerationStore

    return EmbeddingGenerationStore(embeddings_path.parent, active_path=embeddings_path)


def _open_bound_embedding_connection(binding: EmbeddingGenerationBinding) -> sqlite3.Connection:
    conn = open_isolated_write_connection(
        Path(binding.database_path),
        purpose="embedding materialization",
        timeout=30.0,
        archive_root=Path(binding.archive_root),
    )
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

    loaded, error = try_load_sqlite_vec(conn)
    if not loaded:
        with contextlib.suppress(sqlite3.Error):
            conn.close()
        raise RuntimeError("archive embedding materialization requires sqlite-vec") from error
    return conn


_SESSION_ATTEMPT_LOCK_GUARD = threading.Lock()
_SESSION_ATTEMPT_LOCKS: dict[str, tuple[threading.Lock, int]] = {}


@contextlib.contextmanager
def _session_attempt_lock(key: str) -> Iterator[None]:
    """Serialize one session's provider work across live and backlog owners."""
    with _SESSION_ATTEMPT_LOCK_GUARD:
        lock, users = _SESSION_ATTEMPT_LOCKS.get(key, (threading.Lock(), 0))
        _SESSION_ATTEMPT_LOCKS[key] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _SESSION_ATTEMPT_LOCK_GUARD:
            current_lock, users = _SESSION_ATTEMPT_LOCKS[key]
            if users == 1:
                del _SESSION_ATTEMPT_LOCKS[key]
            else:
                _SESSION_ATTEMPT_LOCKS[key] = (current_lock, users - 1)


def embed_archive_session_sync(
    index_db_path: Path,
    vec_provider: VectorProvider,
    session_id: str,
    *,
    embeddings_db_path: Path | None = None,
    stop_after_seconds: float | None = None,
    admit: EmbeddingWriteAdmission | None = None,
) -> EmbedSessionOutcome:
    """Own a session attempt exclusively until provider work and publication settle."""
    embeddings_path = embeddings_db_path or index_db_path.with_name("embeddings.db")
    key = f"{embeddings_path.resolve(strict=False)}\0{session_id}"
    with _session_attempt_lock(key):
        return _embed_archive_session_sync_unlocked(
            index_db_path,
            vec_provider,
            session_id,
            embeddings_db_path=embeddings_path,
            stop_after_seconds=stop_after_seconds,
            admit=admit,
        )


def _embed_archive_session_sync_unlocked(
    index_db_path: Path,
    vec_provider: VectorProvider,
    session_id: str,
    *,
    embeddings_db_path: Path | None = None,
    stop_after_seconds: float | None = None,
    admit: EmbeddingWriteAdmission | None = None,
) -> EmbedSessionOutcome:
    """Embed one archive session: admitted reservation, lease-free compute, admitted publication.

    ``admit`` is the seam that keeps the provider call outside writer
    ownership. Each phase below runs inside one ``admit`` call and acquires --
    then releases -- both the writer authority and the embedding generation
    lock; ``vec_provider._get_embeddings`` runs between those calls holding
    neither, so a slow or hung provider cannot block an unrelated archive
    writer. Publication re-reserves a fresh generation binding and refuses a
    window whose reserved attempt no longer owns the session.
    """
    admit_phase: EmbeddingWriteAdmission = inline_embedding_admission if admit is None else admit
    text_provider = cast(_EmbeddingTextProvider, vec_provider)
    if not hasattr(text_provider, "_get_embeddings"):
        return EmbedSessionOutcome(
            status="error",
            session_id=session_id,
            error="vector provider does not expose text embedding generation",
        )
    resolved_embeddings = (
        embeddings_db_path if embeddings_db_path is not None else index_db_path.with_name("embeddings.db")
    )

    try:
        prepared = admit_phase(
            "embedding.prepare",
            lambda: _prepare_archive_embedding_attempt(
                index_db_path,
                text_provider,
                session_id,
                embeddings_path=resolved_embeddings,
            ),
        )
    except Exception as exc:
        # Reservation failed before any attempt existed -- a lifecycle refusal,
        # a writer-hold bound, a missing tier. There is no attempt to record a
        # receipt against, so this stays a typed outcome rather than an
        # exception escaping a route whose contract is "returns an outcome".
        return EmbedSessionOutcome(status="error", session_id=session_id, error=str(exc))
    if isinstance(prepared, EmbedSessionOutcome):
        return prepared
    plan = prepared

    # ── lease-free computation ──────────────────────────────────────────
    # Nothing below opens a write connection, holds the generation lock, or
    # owns the daemon's writer gate until the next ``admit_phase`` call.
    batch_size = max(1, ARCHIVE_EMBED_MESSAGE_BATCH_SIZE)
    started_at = time.monotonic()
    published_count = plan.published_before_compute
    deferred = False
    for start in range(0, len(plan.to_embed), batch_size):
        if stop_after_seconds is not None and time.monotonic() - started_at >= stop_after_seconds:
            deferred = True
            break
        batch = plan.to_embed[start : start + batch_size]
        attempted_refs = tuple(item.message_id for item in batch)
        try:
            vectors = text_provider._get_embeddings([item.text for item in batch], input_type="document")
            if len(vectors) != len(batch):
                raise _ProviderRequestError("embedding provider returned a mismatched vector count")
        except Exception as exc:
            provider_error = exc if isinstance(exc, _ProviderRequestError) else _ProviderRequestError(str(exc))
            return _fail_archive_embedding_attempt(
                plan,
                provider_error,
                attempted_message_refs=attempted_refs,
                admit=admit_phase,
            )
        writes = tuple(
            _archive_embedding_write(plan, item, vector) for item, vector in zip(batch, vectors, strict=True)
        )
        try:
            published = admit_phase(
                "embedding.publish",
                partial(_publish_archive_embedding_window, plan, writes),
            )
        except Exception as exc:
            return _fail_archive_embedding_attempt(
                plan,
                exc,
                attempted_message_refs=attempted_refs,
                admit=admit_phase,
            )
        if not published:
            return EmbedSessionOutcome(
                status="error",
                session_id=plan.session_id,
                title=plan.title,
                error="embedding attempt superseded",
            )
        published_count += len(batch)
        if stop_after_seconds is not None and time.monotonic() - started_at >= stop_after_seconds:
            deferred = start + len(batch) < len(plan.to_embed)
            if deferred:
                break

    if deferred:
        return EmbedSessionOutcome(
            status="deferred",
            session_id=plan.session_id,
            title=plan.title,
            embedded_message_count=len(plan.embeddable_message_ids) - plan.pending_count + published_count,
            deferred=True,
        )

    try:
        return admit_phase("embedding.finalize", lambda: _finalize_archive_embedding_attempt(plan))
    except Exception as exc:
        return _fail_archive_embedding_attempt(plan, exc, attempted_message_refs=(), admit=admit_phase)


def _archive_embedding_write(
    plan: _ArchiveEmbeddingPlan,
    item: _ArchiveEmbeddingInput,
    vector: list[float],
) -> ArchiveEmbeddingWrite:
    from polylogue.storage.sqlite.archive_tiers.embedding_write import ArchiveEmbeddingWrite

    return ArchiveEmbeddingWrite(
        message_id=item.message_id,
        session_id=plan.session_id,
        origin=plan.origin,
        embedding=vector,
        model=plan.model,
        embedded_at_ms=plan.now_ms,
        vector_derivation_hash=item.input_hash,
        message_content_hash=item.message_content_hash,
        recipe_hash=plan.attempt.recipe_hash,
        derivation_key=message_embedding_derivation_key(
            message_id=item.message_id,
            vector_derivation_hash=item.input_hash,
            recipe=plan.recipe,
        ).digest(),
        generation=plan.attempt.generation,
    )


def _prepare_archive_embedding_attempt(
    index_db_path: Path,
    text_provider: _EmbeddingTextProvider,
    session_id: str,
    *,
    embeddings_path: Path,
) -> _ArchiveEmbeddingPlan | EmbedSessionOutcome:
    """Reserve one attempt and snapshot its inputs, under writer authority.

    Short by construction: it reads the session's embeddable messages, hashes
    them, reserves the attempt, and republishes vectors that already exist by
    content address. It never calls the provider, so the generation lock it
    holds is released before any network work begins.
    """
    from polylogue.storage.embeddings.generations import EmbeddingGenerationError
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        begin_embedding_attempt,
        publish_embedding_attempt_window,
    )
    from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec
    from polylogue.storage.sqlite.write_lease import require_write_lease

    require_write_lease("embedding archive bootstrap", archive_root=embeddings_path.parent)
    if not embeddings_path.exists():
        from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

        initialize_archive_database(embeddings_path, ArchiveTier.EMBEDDINGS)

    store = _embedding_lifecycle_store(embeddings_path)
    with store.writer_lock() as binding:
        index_conn = open_readonly_connection(index_db_path, timeout_class="background-read", validate_schema=False)
        index_conn.row_factory = sqlite3.Row
        embeddings_conn: sqlite3.Connection | None = None
        session: sqlite3.Row | None = None
        attempt = None
        try:
            # Opened before the extension check so a tier without sqlite-vec
            # still reaches the failure ledger below instead of failing
            # silently with nothing recorded.
            embeddings_conn = open_isolated_write_connection(
                Path(binding.database_path),
                purpose="embedding materialization",
                timeout=30.0,
                archive_root=Path(binding.archive_root),
            )
            loaded, extension_error = try_load_sqlite_vec(embeddings_conn)
            if not loaded:
                raise RuntimeError("archive embedding materialization requires sqlite-vec") from extension_error
            session = index_conn.execute(
                "SELECT session_id, origin, title, message_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                return EmbedSessionOutcome(status="not_found", session_id=session_id)

            if not embedding_acquisition_allowed(index_conn, session_id):
                return EmbedSessionOutcome(
                    status="deferred", session_id=session_id, deferred=True, error="demo_acquisition_excluded"
                )

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
            # Snapshot the *configured* recipe as it stands before any provider
            # call. The finalize-time guard asks "did the recipe change while we
            # worked", and it can only answer that by comparing configuration to
            # configuration. Comparing it to ``recipe`` -- which is derived from
            # the provider actually in use -- reports a mismatch that was already
            # true before the work began, so any caller whose provider differs
            # from the configured one (a catch-up run pinned to an existing
            # generation's model, for instance) supersedes every attempt and
            # re-queues forever.
            configured_recipe_before = _configured_embedding_recipe()
            store.require_recipe_compatible(embeddings_conn, recipe)
            # Derived directly from row["text"] -- the exact same string handed
            # to the embedder -- so hash validity equals vector validity by
            # construction (polylogue-q88p). Computed once here, not re-derived
            # from stored identity, and reused for both the write and the
            # session source-identity digest.
            input_hash_by_message_id: dict[str, bytes] = {
                str(row["message_id"]): EmbeddingRequestSpec(
                    recipe=recipe, input_text=str(row["text"])
                ).vector_derivation_hash
                for row in embeddable
            }
            source_hash = _archive_embedding_source_hash_from_pairs(input_hash_by_message_id.items())
            store.assert_binding(binding)
            attempt = begin_embedding_attempt(
                embeddings_conn,
                session_id=session_id,
                origin=str(session["origin"]),
                source_hash=source_hash,
                recipe=recipe,
            )

            now_ms = int(datetime.now(UTC).timestamp() * 1000)
            existing_refs = {
                str(row[0]): row
                for row in embeddings_conn.execute(
                    """
                    SELECT r.message_id, r.vector_derivation_hash, r.message_content_hash, em.model, em.dimension, em.recipe_hash, em.output_contract_hash,
                           EXISTS(SELECT 1 FROM message_embeddings AS v WHERE v.vector_derivation_hash = lower(hex(r.vector_derivation_hash))), r.origin
                    FROM message_embedding_refs AS r
                    JOIN message_embeddings_meta AS em
                      ON em.vector_derivation_hash = r.vector_derivation_hash
                    WHERE r.session_id = ?
                    """,
                    (session_id,),
                ).fetchall()
            }
            pending_embeddable = []
            for row in embeddable:
                retained = existing_refs.get(str(row["message_id"]))
                if (
                    retained is not None
                    and retained[2] is not None
                    and row["content_hash"] is not None
                    and bytes(retained[2]) == bytes(row["content_hash"])
                    and recipe.proven_stored_producer(
                        model=str(retained[3]), dimension=int(retained[4]), recipe_hash=bytes(retained[5])
                    )
                    is None
                ):
                    raise EmbeddingProvenanceError("stored embedding producer provenance is unproven")
                valid = (
                    retained is not None
                    and retained[2] is not None
                    and row["content_hash"] is not None
                    and bytes(retained[2]) == bytes(row["content_hash"])
                    and recipe.stored_output_matches(
                        model=str(retained[3]),
                        dimension=int(retained[4]),
                        recipe_hash=bytes(retained[5]),
                        output_contract_hash=bytes(retained[6]),
                        vector_hash=bytes(retained[1]),
                        text=str(row["text"]),
                    )
                    and bool(retained[7])
                    and str(retained[8]) == str(session["origin"])
                )
                if not valid:
                    pending_embeddable.append(row)
            present_hashes = _present_vector_addresses(
                embeddings_conn,
                (input_hash_by_message_id[str(row["message_id"])] for row in pending_embeddable),
                recipe=recipe,
            )
            plan = _ArchiveEmbeddingPlan(
                index_db_path=index_db_path,
                embeddings_path=embeddings_path,
                binding=binding,
                session_id=session_id,
                origin=str(session["origin"]),
                title=None if session["title"] is None else str(session["title"]),
                session_message_count=int(session["message_count"] or 0),
                model=str(text_provider.model),
                recipe=recipe,
                configured_recipe_before=configured_recipe_before,
                attempt=attempt,
                embeddable_message_ids=tuple(str(row["message_id"]) for row in embeddable),
                pending_count=len(pending_embeddable),
                to_embed=tuple(
                    _ArchiveEmbeddingInput(
                        message_id=str(row["message_id"]),
                        text=str(row["text"]),
                        input_hash=input_hash_by_message_id[str(row["message_id"])],
                        message_content_hash=None if row["content_hash"] is None else bytes(row["content_hash"]),
                    )
                    for row in pending_embeddable
                    if input_hash_by_message_id[str(row["message_id"])] not in present_hashes
                ),
                now_ms=now_ms,
                published_before_compute=0,
            )
            reusable_writes = [
                _archive_embedding_write(
                    plan,
                    _ArchiveEmbeddingInput(
                        message_id=str(row["message_id"]),
                        text="",
                        input_hash=input_hash_by_message_id[str(row["message_id"])],
                        message_content_hash=None if row["content_hash"] is None else bytes(row["content_hash"]),
                    ),
                    [],
                )
                for row in pending_embeddable
                if input_hash_by_message_id[str(row["message_id"])] in present_hashes
            ]
            if reusable_writes:
                if not publish_embedding_attempt_window(
                    embeddings_conn,
                    attempt=attempt,
                    writes=reusable_writes,
                    completed_at_ms=now_ms,
                ):
                    return EmbedSessionOutcome(
                        status="error",
                        session_id=session_id,
                        title=plan.title,
                        error="embedding attempt superseded",
                    )
                store.refresh_binding_contract(binding)
            return replace(plan, published_before_compute=len(reusable_writes))
        except Exception as exc:
            if isinstance(exc, EmbeddingGenerationError):
                # A hostile pointer/root replacement invalidates the whole
                # operation. Do not turn that failed reservation into a failure
                # receipt in the replacement generation.
                return EmbedSessionOutcome(status="error", session_id=session_id, error=str(exc))
            if embeddings_conn is None:
                return EmbedSessionOutcome(status="error", session_id=session_id, error=str(exc))
            _record_archive_embedding_attempt_error(
                embeddings_conn,
                index_conn=index_conn,
                session_id=session_id,
                session=session,
                model=str(text_provider.model),
                error=exc,
                attempted_message_refs=(),
                attempt=attempt,
            )
            return EmbedSessionOutcome(status="error", session_id=session_id, error=str(exc))
        finally:
            with contextlib.suppress(sqlite3.Error):
                index_conn.close()
            if embeddings_conn is not None:
                with contextlib.suppress(sqlite3.Error):
                    embeddings_conn.close()


def _publish_archive_embedding_window(
    plan: _ArchiveEmbeddingPlan,
    writes: Sequence[ArchiveEmbeddingWrite],
) -> bool:
    """Publish one computed window under a freshly reserved generation binding.

    The binding is re-acquired rather than inherited: the pointer may have
    moved while the provider was working, and ``assert_binding`` is what makes
    publishing into a replacement generation impossible rather than merely
    unlikely. ``publish_embedding_attempt_window`` then refuses a window whose
    reserved attempt no longer owns the session.
    """
    from polylogue.storage.sqlite.archive_tiers.embedding_write import publish_embedding_attempt_window

    store = _embedding_lifecycle_store(plan.embeddings_path)
    with store.writer_lock() as binding:
        store.assert_binding(plan.binding)
        conn = _open_bound_embedding_connection(binding)
        try:
            store.require_recipe_compatible(conn, plan.recipe)
            published = publish_embedding_attempt_window(
                conn,
                attempt=plan.attempt,
                writes=writes,
                completed_at_ms=plan.now_ms,
            )
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        if published:
            store.refresh_binding_contract(binding)
        return published


def _finalize_archive_embedding_attempt(plan: _ArchiveEmbeddingPlan) -> EmbedSessionOutcome:
    """Revalidate the snapshot the vectors were computed from, then close the attempt."""
    from polylogue.storage.sqlite.archive_tiers.embedding_write import (
        finalize_embedding_attempt_success,
        supersede_embedding_attempt,
    )

    store = _embedding_lifecycle_store(plan.embeddings_path)
    with store.writer_lock() as binding:
        store.assert_binding(plan.binding)
        index_conn = open_readonly_connection(
            plan.index_db_path, timeout_class="background-read", validate_schema=False
        )
        index_conn.row_factory = sqlite3.Row
        try:
            conn = _open_bound_embedding_connection(binding)
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                index_conn.close()
            raise
        try:
            configured_recipe_now = _configured_embedding_recipe()
            configuration_changed = (
                configured_recipe_now.recipe_hash != plan.configured_recipe_before.recipe_hash
                or configured_recipe_now.output_contract_hash != plan.configured_recipe_before.output_contract_hash
            )
            # Source identity includes the provider request. Compare it under
            # the recipe that produced this attempt, independently of whether
            # the configured recipe changed while the provider was working.
            current_source_hash, current_message_count, current_message_ids = _read_archive_embedding_source_snapshot(
                index_conn, plan.session_id, recipe=plan.recipe
            )
            if (
                current_source_hash != plan.attempt.source_hash
                or current_message_count != len(plan.embeddable_message_ids)
                or current_message_ids != tuple(sorted(plan.embeddable_message_ids))
                or configuration_changed
            ):
                successor_recipe = configured_recipe_now if configuration_changed else plan.recipe
                if configuration_changed:
                    # A configured-recipe transition queues that new request,
                    # so its source digest must use the same recipe as its key.
                    current_source_hash, _, _ = _read_archive_embedding_source_snapshot(
                        index_conn, plan.session_id, recipe=successor_recipe
                    )
                supersede_embedding_attempt(
                    conn,
                    attempt=plan.attempt,
                    source_hash=current_source_hash,
                    recipe=successor_recipe,
                )
                return EmbedSessionOutcome(
                    status="error",
                    session_id=plan.session_id,
                    title=plan.title,
                    error="embedding source or recipe changed during materialization; retry queued",
                )
            committed = finalize_embedding_attempt_success(
                conn,
                attempt=plan.attempt,
                message_ids=list(plan.embeddable_message_ids),
                completed_at_ms=plan.now_ms,
            )
            if not committed:
                return EmbedSessionOutcome(
                    status="error",
                    session_id=plan.session_id,
                    title=plan.title,
                    error="embedding attempt was superseded before publication; retry queued",
                )
        finally:
            with contextlib.suppress(sqlite3.Error):
                index_conn.close()
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        store.refresh_binding_contract(binding)

    if not plan.embeddable_message_ids:
        no_op_status: EmbedSingleStatus = "no_messages" if plan.session_message_count <= 0 else "no_embeddable_messages"
        return EmbedSessionOutcome(status=no_op_status, session_id=plan.session_id, title=plan.title)
    return EmbedSessionOutcome(
        status="embedded",
        session_id=plan.session_id,
        title=plan.title,
        embedded_message_count=len(plan.embeddable_message_ids),
    )


def _fail_archive_embedding_attempt(
    plan: _ArchiveEmbeddingPlan,
    error: BaseException,
    *,
    attempted_message_refs: Sequence[str],
    admit: EmbeddingWriteAdmission,
) -> EmbedSessionOutcome:
    """Record one attempt's failure receipt through an admitted short write."""
    from polylogue.storage.embeddings.generations import EmbeddingGenerationError

    def record() -> str | None:
        store = _embedding_lifecycle_store(plan.embeddings_path)
        with store.writer_lock() as binding:
            # A hostile pointer/root replacement invalidates the whole
            # operation. Do not turn that failed publication into a failure
            # receipt in the replacement generation; the caller must retry
            # against a fresh bind.
            store.assert_binding(plan.binding)
            conn = _open_bound_embedding_connection(binding)
            try:
                _record_archive_embedding_attempt_error(
                    conn,
                    index_conn=None,
                    session_id=plan.session_id,
                    session=None,
                    origin=plan.origin,
                    model=plan.model,
                    error=error,
                    attempted_message_refs=attempted_message_refs,
                    attempt=plan.attempt,
                )
            finally:
                with contextlib.suppress(sqlite3.Error):
                    conn.close()
        return None

    try:
        admit("embedding.failure", record)
    except EmbeddingGenerationError as binding_error:
        return EmbedSessionOutcome(status="error", session_id=plan.session_id, error=str(binding_error))
    return EmbedSessionOutcome(status="error", session_id=plan.session_id, title=plan.title, error=str(error))


def _record_archive_embedding_attempt_error(
    conn: sqlite3.Connection,
    *,
    index_conn: sqlite3.Connection | None,
    session_id: str,
    session: sqlite3.Row | None,
    model: str,
    error: BaseException,
    attempted_message_refs: Sequence[str],
    attempt: ArchiveEmbeddingAttempt | None,
    origin: str | None = None,
) -> None:
    """Write one embedding failure receipt, never losing the row to a lookup failure."""
    from polylogue.storage.sqlite.archive_tiers.embedding_write import record_embedding_failure

    # The failure ledger must never lose a row merely because the origin lookup
    # itself failed (polylogue-es7b): the attempt's own origin is used when it
    # is known, then the session row, then a best-effort re-read, and finally an
    # explicit unknown sentinel so ``record_embedding_failure`` is always called.
    if origin is not None:
        origin_value = origin
    elif session is not None:
        origin_value = str(session["origin"])
    else:
        origin_value = str(Origin.UNKNOWN_EXPORT)
        if index_conn is not None:
            with contextlib.suppress(sqlite3.Error):
                origin_row = index_conn.execute(
                    "SELECT origin FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                if origin_row is not None:
                    origin_value = str(origin_row["origin"])
    if isinstance(error, _ProviderRequestError):
        provider = "voyage"
        error_class = embedding_error_class(error)
        retryable = not is_terminal_embedding_provider_error(str(error))
    else:
        provider = "local"
        error_class = "internal_error"
        retryable = True
    record_embedding_failure(
        conn,
        session_id=session_id,
        origin=origin_value,
        message_refs=tuple(attempted_message_refs),
        provider=provider,
        model=model,
        error_class=error_class,
        error_message=str(error),
        retryable=retryable,
        attempt=attempt,
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
    "embed_archive_session_sync",
    "iter_pending_sessions",
    "mark_all_archive_sessions_needs_reindex",
    "prepare_inactive_archive_embedding_generation",
    "select_pending_session_window",
    "select_pending_archive_session_window",
]
