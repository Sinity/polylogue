"""Shared helpers for optional embedding-related archive statistics."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

import aiosqlite

from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.sqlite_introspection import table_exists_async as _table_exists_async
from polylogue.storage.derived.session.runtime import SessionInsightStatusSnapshot
from polylogue.storage.embeddings.sql import EMBEDDED_MESSAGES_SQL

StatsRow = sqlite3.Row | tuple[object, ...]


def _stats_row(row: object) -> StatsRow | None:
    if row is None:
        return None
    if isinstance(row, (sqlite3.Row, tuple)):
        return row
    return None


def _sqlite_rows(rows: Iterable[object]) -> list[sqlite3.Row]:
    return [row for row in rows if isinstance(row, sqlite3.Row)]


def _coerce_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
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
            return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def build_retrieval_bands_from_status(
    *,
    total_sessions: int,
    embedded_sessions: int,
    embedded_messages: int,
    pending_sessions: int,
    stale_messages: int,
    missing_provenance: int,
    session_status: SessionInsightStatusSnapshot,
) -> dict[str, dict[str, object]]:
    transcript_ready = total_sessions == 0 or (
        embedded_sessions == total_sessions
        and pending_sessions == 0
        and stale_messages == 0
        and missing_provenance == 0
    )
    transcript_status = "empty" if total_sessions == 0 else ("ready" if transcript_ready else "pending")

    evidence_source_rows = session_status.profile_row_count
    evidence_materialized_rows = session_status.profile_row_count
    evidence_ready = True

    inference_source_rows = session_status.profile_row_count
    inference_materialized_rows = session_status.profile_row_count
    inference_ready = session_status.profile_inference_fts_duplicate_count == 0
    enrichment_source_rows = session_status.profile_row_count
    enrichment_materialized_rows = session_status.profile_row_count
    enrichment_ready = True

    return {
        "transcript_embeddings": {
            "status": transcript_status,
            "ready": transcript_ready,
            "source_documents": total_sessions,
            "materialized_documents": embedded_sessions,
            "materialized_rows": embedded_messages,
            "pending_documents": pending_sessions,
            "stale_rows": stale_messages,
            "missing_provenance_rows": missing_provenance,
            "detail": (
                f"Transcript embeddings ready ({embedded_sessions:,}/{total_sessions:,} sessions, {embedded_messages:,} messages)"
                if transcript_ready
                else (
                    f"Transcript embeddings pending ({embedded_sessions:,}/{total_sessions:,} sessions, "
                    f"pending {pending_sessions:,}, stale {stale_messages:,}, missing provenance {missing_provenance:,})"
                )
            ),
        },
        "evidence_retrieval": {
            "status": "ready" if evidence_ready else "pending",
            "ready": evidence_ready,
            "source_rows": evidence_source_rows,
            "materialized_rows": evidence_materialized_rows,
            "pending_rows": max(0, evidence_source_rows - evidence_materialized_rows),
            "stale_rows": session_status.profile_evidence_fts_duplicate_count,
            "detail": (
                f"Evidence retrieval ready ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows)"
                if evidence_ready
                else (
                    f"Evidence retrieval pending ({evidence_materialized_rows:,}/{evidence_source_rows:,} supporting rows; "
                    "profile evidence FTS pending)"
                )
            ),
        },
        "inference_retrieval": {
            "status": "ready" if inference_ready else "pending",
            "ready": inference_ready,
            "source_rows": inference_source_rows,
            "materialized_rows": inference_materialized_rows,
            "pending_rows": max(0, inference_source_rows - inference_materialized_rows),
            "stale_rows": session_status.profile_inference_fts_duplicate_count,
            "detail": (
                f"Inference retrieval ready ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows)"
                if inference_ready
                else (
                    f"Inference retrieval pending ({inference_materialized_rows:,}/{inference_source_rows:,} supporting rows; "
                    "profile inference FTS pending)"
                )
            ),
        },
        "enrichment_retrieval": {
            "status": "ready" if enrichment_ready else "pending",
            "ready": enrichment_ready,
            "source_rows": enrichment_source_rows,
            "materialized_rows": enrichment_materialized_rows,
            "pending_rows": max(0, enrichment_source_rows - enrichment_materialized_rows),
            "stale_rows": session_status.profile_enrichment_fts_duplicate_count,
            "detail": (
                f"Enrichment retrieval ready ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows)"
                if enrichment_ready
                else (
                    f"Enrichment retrieval pending ({enrichment_materialized_rows:,}/{enrichment_source_rows:,} supporting rows)"
                )
            ),
        },
    }


class EmbeddingCoverageUnmeasurableError(RuntimeError):
    """The embeddings tier could not be inspected, so coverage is unknown.

    Raised instead of returning a zero when a read fails for a reason that is
    an *inability to measure* rather than evidence of absence -- today, a
    SQLite module (``vec0``) that would not load, so a physically present
    virtual table cannot be read. ``embeddings.db`` is the expensive-to-rebuild
    tier whose vectors are re-purchased from a paid provider, never replayed
    from source, so reporting "cannot tell" as "nothing embedded" prescribes
    exactly the most expensive wrong action.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def is_missing_table_error(exc: sqlite3.OperationalError) -> bool:
    """True only for a genuinely absent relation -- a measured absence.

    ``no such module: vec0`` was once folded in here and turned an unloadable
    extension into a measured zero; it is classified by
    :func:`is_unmeasurable_coverage_error` instead. ``no such column`` is not
    absence either: the relation is present in a shape this reader cannot
    read, which :func:`is_unreadable_relation_error` classifies.
    """

    message = str(exc).lower()
    return "no such table" in message or "does not exist" in message or "table not found" in message


#: The typed reason a readiness reader reports when a relation it reads is
#: present but not in the shape it reads.
READINESS_RELATION_UNAVAILABLE = "readiness_relation_unavailable"


def is_unreadable_relation_error(exc: sqlite3.OperationalError) -> bool:
    """True when a present relation lacks a column the reader requires.

    Fresh archives carry one schema, so a missing column is never an older
    optional shape that means "not recorded"; it means the reader cannot
    measure, and is reported as ``readiness_relation_unavailable``.
    """

    return "no such column" in str(exc).lower()


def is_unmeasurable_coverage_error(exc: sqlite3.OperationalError) -> bool:
    """True when the failure means *cannot tell*, not *nothing is there*.

    A missing SQLite module is never optional-feature detection: the vectors
    are physically there and simply cannot be read.
    """

    message = str(exc).lower()
    return "no such module" in message


def _classify_optional_error(exc: sqlite3.OperationalError) -> None:
    """Re-raise as unmeasurable, or return so the caller may report absence."""

    if is_unmeasurable_coverage_error(exc):
        raise EmbeddingCoverageUnmeasurableError(str(exc)) from exc
    if is_unreadable_relation_error(exc):
        raise EmbeddingCoverageUnmeasurableError(f"{READINESS_RELATION_UNAVAILABLE}: {exc}") from exc
    if is_missing_table_error(exc):
        return
    raise exc


def table_exists_sync_missing_safe(conn: sqlite3.Connection, table: str) -> bool:
    # Thin wrapper (not a duplicate): swallows a still-in-flight
    # sqlite3.OperationalError ("no such table") that `introspection.table_exists`
    # itself never raises but a caller mid-migration/attach can still hit.
    try:
        return _table_exists(conn, table)
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return False


async def table_exists_async_missing_safe(conn: aiosqlite.Connection, table: str) -> bool:
    try:
        return await _table_exists_async(conn, table)
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return False


def optional_count_sync(conn: sqlite3.Connection, sql: str) -> int:
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return 0
    return int(row[0]) if row is not None else 0


def embedded_message_count_sync(conn: sqlite3.Connection) -> int:
    """Count messages that have a current embedding.

    ``message_embeddings_meta`` is keyed by ``vector_derivation_hash`` and
    deduped -- its row count is the number of *distinct vectors*, not messages
    (identical content across sessions shares one row).
    ``message_embedding_refs`` (message_id -> hash) is the per-message count;
    the embeddings DDL always creates it beside the vector tables, so its
    absence means no embeddings tier is visible on this connection.
    """
    return optional_count_sync(conn, EMBEDDED_MESSAGES_SQL)


def optional_row_sync(conn: sqlite3.Connection, sql: str) -> StatsRow | None:
    try:
        return _stats_row(conn.execute(sql).fetchone())
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return None


def optional_rows_sync(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return []


async def optional_count_async(conn: aiosqlite.Connection, sql: str) -> int:
    try:
        cursor = await conn.execute(sql)
        row = await cursor.fetchone()
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return 0
    return _coerce_int(row[0]) if row is not None else 0


async def embedded_message_count_async(conn: aiosqlite.Connection) -> int:
    """Count messages that have a current embedding (see sync counterpart)."""
    return await optional_count_async(conn, EMBEDDED_MESSAGES_SQL)


async def optional_row_async(conn: aiosqlite.Connection, sql: str) -> StatsRow | None:
    try:
        cursor = await conn.execute(sql)
        return _stats_row(await cursor.fetchone())
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return None


async def optional_rows_async(conn: aiosqlite.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        cursor = await conn.execute(sql)
        return _sqlite_rows(await cursor.fetchall())
    except sqlite3.OperationalError as exc:
        _classify_optional_error(exc)
        return []
