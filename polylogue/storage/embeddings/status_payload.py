"""Embedding-status payload builder (substrate, click-free).

Counts come from a sync read connection over the embedding-status tables.
Surfaces (CLI, MCP, dashboards) consume :func:`embedding_status_payload`
and render in their own dialect.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from typing_extensions import TypedDict

from polylogue.storage.embeddings.materialization import (
    archive_embeddable_message_where,
    archive_embedding_messages_table_ref,
    count_archive_embedding_session_state,
)
from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.embeddings.progress import EmbeddingCatchupRunPayload
from polylogue.storage.search_providers.sqlite_vec_support import (
    ESTIMATED_TOKENS_PER_MESSAGE,
    VOYAGE_4_COST_PER_1M_TOKENS,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from polylogue.config import Config

DETAIL_QUERY_TIMEOUT_MS = 2_000
METADATA_SUMMARY_TIMEOUT_MS = 5_000
STATUS_READ_BUSY_TIMEOUT_MS = 1_000


class _HasConfig(Protocol):
    @property
    def config(self) -> Config: ...


class RetrievalBandPayload(TypedDict, total=False):
    ready: bool
    status: str
    materialized_rows: int
    source_rows: int
    materialized_documents: int
    source_documents: int


class EmbeddingNextActionPayload(TypedDict):
    code: str
    command: str | None
    reason: str


class EmbeddingStatusPayload(TypedDict):
    config_enabled: bool
    has_voyage_api_key: bool
    daemon_stage_enabled: bool
    configured_model: str
    configured_dimension: int
    monthly_cost_cap_usd: float
    status: str
    total_sessions: int
    embedded_sessions: int
    embedded_messages: int
    pending_sessions: int
    pending_messages: int | None
    pending_messages_exact: bool
    embedding_coverage_percent: float
    retrieval_ready: bool
    freshness_status: str
    stale_messages: int
    messages_missing_provenance: int
    oldest_embedded_at: str | None
    newest_embedded_at: str | None
    embedding_models: dict[str, int]
    embedding_dimensions: dict[int, int]
    retrieval_bands: dict[str, dict[str, object]]
    failure_count: int
    total_estimated_cost_usd: float
    latest_catchup_run: EmbeddingCatchupRunPayload | None
    next_action: EmbeddingNextActionPayload


def _payload_int(value: object) -> int:
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


def _total_sessions(conn: sqlite3.Connection) -> int:
    from polylogue.storage.embeddings.support import optional_count_sync

    return optional_count_sync(conn, "SELECT COUNT(*) FROM sessions")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        is not None
    )


def _attached_table_exists(conn: sqlite3.Connection, schema_name: str, table_name: str) -> bool:
    quoted_schema = '"' + schema_name.replace('"', '""') + '"'
    return (
        conn.execute(
            f"SELECT 1 FROM {quoted_schema}.sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        is not None
    )


def _attached_table_name(conn: sqlite3.Connection, schema_name: str, table_name: str) -> str:
    if _attached_table_exists(conn, schema_name, table_name):
        return f"{schema_name}.{table_name}"
    return ""


def _scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    from polylogue.storage.embeddings.support import is_missing_table_error

    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        if is_missing_table_error(exc):
            return 0
        raise
    if row is None:
        return 0
    return _payload_int(row[0])


def _scalar_int_with_timeout(conn: sqlite3.Connection, sql: str, *, timeout_ms: int) -> int | None:
    """Return an exact scalar count, or ``None`` when the live archive cannot answer quickly."""

    from polylogue.storage.embeddings.support import is_missing_table_error

    deadline = time.monotonic() + (timeout_ms / 1000.0)

    def _interrupt_when_expired() -> int:
        return 1 if time.monotonic() >= deadline else 0

    conn.set_progress_handler(_interrupt_when_expired, 10_000)
    try:
        row = conn.execute(sql).fetchone()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if is_missing_table_error(exc):
            return 0
        if "interrupted" in message or "locked" in message or "busy" in message:
            return None
        raise
    finally:
        conn.set_progress_handler(None, 0)
    if row is None:
        return 0
    return _payload_int(row[0])


def _rows_with_timeout(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_ms: int,
) -> list[sqlite3.Row | tuple[object, ...]] | None:
    """Return query rows, or ``None`` when the live archive cannot answer quickly."""

    from polylogue.storage.embeddings.support import is_missing_table_error

    deadline = time.monotonic() + (timeout_ms / 1000.0)

    def _interrupt_when_expired() -> int:
        return 1 if time.monotonic() >= deadline else 0

    conn.set_progress_handler(_interrupt_when_expired, 10_000)
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        message = str(exc).lower()
        if is_missing_table_error(exc):
            return []
        if "interrupted" in message or "locked" in message or "busy" in message:
            return None
        raise
    finally:
        conn.set_progress_handler(None, 0)
    return list(rows)


def _iso_from_epoch_ms(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        epoch_ms = value
    elif isinstance(value, float):
        epoch_ms = int(value)
    elif isinstance(value, str):
        try:
            epoch_ms = int(value)
        except ValueError:
            return None
    else:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000.0, UTC).isoformat()


def _archive_index_path(db_path: Path) -> Path | None:
    if db_path.name == "index.db":
        return db_path if db_path.exists() else None
    index_db = db_path.with_name("index.db")
    return index_db if index_db.exists() else None


def _coverage_percent(*, embedded_sessions: int, eligible_sessions: int, total_sessions: int) -> float:
    if eligible_sessions <= 0:
        return 100.0 if total_sessions > 0 else 0.0
    return embedded_sessions / eligible_sessions * 100


def _embedding_status(
    *,
    total_sessions: int,
    embedded_sessions: int,
    pending_sessions: int,
) -> str:
    if total_sessions <= 0:
        return "empty"
    if pending_sessions <= 0:
        return "complete"
    if embedded_sessions <= 0:
        return "none"
    return "partial"


def _freshness_status(status: str, stats: EmbeddingStatsSnapshot) -> str:
    if stats.embedded_messages > 0 and (stats.stale_messages > 0 or stats.messages_missing_provenance > 0):
        return "stale"
    return status


def _retrieval_ready(stats: EmbeddingStatsSnapshot) -> bool:
    return stats.embedded_messages > stats.stale_messages


def _estimated_cost(message_count: int) -> float:
    estimated_tokens = message_count * ESTIMATED_TOKENS_PER_MESSAGE
    return round(estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000, 2)


def _next_action(
    *,
    config_enabled: bool,
    has_voyage_api_key: bool,
    total_sessions: int,
    embedded_sessions: int,
    pending_sessions: int,
    retrieval_ready: bool,
    stale_messages: int,
    failure_count: int,
) -> EmbeddingNextActionPayload:
    if total_sessions <= 0:
        return {
            "code": "archive_empty",
            "command": None,
            "reason": "Archive contains no sessions to embed.",
        }
    if not has_voyage_api_key:
        return {
            "code": "set_voyage_key",
            "command": "polylogue ops embed enable --voyage-api-key ...",
            "reason": "Semantic retrieval needs a Voyage API key before embedding can run.",
        }
    if not config_enabled:
        if pending_sessions <= 0 and retrieval_ready:
            return {
                "code": "ready",
                "command": "polylogue --semantic <query>",
                "reason": "Embeddings are retrieval-ready.",
            }
        if embedded_sessions > 0 and pending_sessions > 0:
            return {
                "code": "continue_backfill",
                "command": "polylogue ops embed backfill --max-sessions 10",
                "reason": (
                    "Manual embedding coverage exists, but daemon convergence is disabled; "
                    "continue bounded backfill or enable daemon catch-up."
                ),
            }
        return {
            "code": "enable_embeddings",
            "command": "polylogue ops embed enable --yes",
            "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
        }
    if failure_count > 0 and pending_sessions > 0:
        return {
            "code": "inspect_failures",
            "command": "polylogue ops embed status --detail",
            "reason": "Recent embedding failures exist while backlog remains.",
        }
    if stale_messages > 0:
        return {
            "code": "refresh_stale",
            "command": "polylogue ops embed backfill --max-sessions 10",
            "reason": "Existing vectors are stale for at least one message.",
        }
    if pending_sessions > 0:
        return {
            "code": "drain_backlog",
            "command": "polylogue ops embed backfill --max-sessions 10",
            "reason": "Embedding convergence is enabled and pending sessions remain.",
        }
    if retrieval_ready:
        return {
            "code": "ready",
            "command": "polylogue --semantic <query>",
            "reason": "Embeddings are retrieval-ready.",
        }
    return {
        "code": "run_preflight",
        "command": "polylogue ops embed preflight --detail",
        "reason": "Embedding state is inconclusive; inspect exact pending-message and retrieval-band details.",
    }


def _payload_from_stats(
    *,
    config_enabled: bool,
    has_voyage_api_key: bool,
    configured_model: str,
    configured_dimension: int,
    monthly_cost_cap_usd: float,
    total_sessions: int,
    stats: EmbeddingStatsSnapshot,
    latest_catchup_run: EmbeddingCatchupRunPayload | None,
    pending_messages_exact: bool,
) -> EmbeddingStatusPayload:
    embedded_sessions = stats.embedded_sessions
    pending_sessions = stats.pending_sessions
    eligible_sessions = embedded_sessions + pending_sessions
    status = _embedding_status(
        total_sessions=total_sessions,
        embedded_sessions=embedded_sessions,
        pending_sessions=pending_sessions,
    )
    retrieval_ready = _retrieval_ready(stats)
    return {
        "config_enabled": config_enabled,
        "has_voyage_api_key": has_voyage_api_key,
        "daemon_stage_enabled": config_enabled and has_voyage_api_key,
        "configured_model": configured_model,
        "configured_dimension": configured_dimension,
        "monthly_cost_cap_usd": monthly_cost_cap_usd,
        "status": status,
        "total_sessions": total_sessions,
        "embedded_sessions": embedded_sessions,
        "embedded_messages": stats.embedded_messages,
        "pending_sessions": pending_sessions,
        "pending_messages": stats.pending_messages if pending_messages_exact else None,
        "pending_messages_exact": pending_messages_exact,
        "embedding_coverage_percent": round(
            _coverage_percent(
                embedded_sessions=embedded_sessions,
                eligible_sessions=eligible_sessions,
                total_sessions=total_sessions,
            ),
            1,
        ),
        "retrieval_ready": retrieval_ready,
        "freshness_status": _freshness_status(status, stats),
        "stale_messages": stats.stale_messages,
        "messages_missing_provenance": stats.messages_missing_provenance,
        "oldest_embedded_at": stats.oldest_embedded_at,
        "newest_embedded_at": stats.newest_embedded_at,
        "embedding_models": stats.model_counts,
        "embedding_dimensions": stats.dimension_counts,
        "retrieval_bands": stats.retrieval_bands,
        "failure_count": stats.failure_count,
        "total_estimated_cost_usd": stats.total_estimated_cost_usd,
        "latest_catchup_run": latest_catchup_run,
        "next_action": _next_action(
            config_enabled=config_enabled,
            has_voyage_api_key=has_voyage_api_key,
            total_sessions=total_sessions,
            embedded_sessions=embedded_sessions,
            pending_sessions=pending_sessions,
            retrieval_ready=retrieval_ready,
            stale_messages=stats.stale_messages,
            failure_count=stats.failure_count,
        ),
    }


def _archive_embedding_status_payload(
    db_path: Path,
    *,
    cfg: object,
    include_detail: bool,
) -> EmbeddingStatusPayload | None:
    index_db = _archive_index_path(db_path)
    if index_db is None:
        return None
    conn = open_readonly_connection(index_db, timeout=STATUS_READ_BUSY_TIMEOUT_MS / 1000.0)
    conn.execute(f"PRAGMA busy_timeout = {STATUS_READ_BUSY_TIMEOUT_MS}")
    try:
        if not _table_exists(conn, "sessions"):
            return None
        embeddings_db = index_db.with_name("embeddings.db")
        if embeddings_db.exists():
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
            status_table = _attached_table_name(conn, "embeddings", "embedding_status")
            vector_table = _attached_table_name(conn, "embeddings", "message_embeddings")
            meta_table = _attached_table_name(conn, "embeddings", "message_embeddings_meta")
        else:
            status_table = ""
            vector_table = ""
            meta_table = ""
        has_messages = _table_exists(conn, "messages")
        has_status = bool(status_table)
        has_meta = bool(meta_table)
        total_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM sessions")
        session_state = count_archive_embedding_session_state(conn, status_table=status_table, rebuild=False)
        embedded_sessions = session_state.embedded_sessions if has_status else 0
        pending_sessions = session_state.pending_sessions
        if has_status:
            embedded_messages = _scalar_int(
                conn,
                f"SELECT COALESCE(SUM(message_count_embedded), 0) FROM {status_table}",
            )
        elif has_meta:
            exact_embedded_messages = _scalar_int_with_timeout(
                conn,
                f"""
                SELECT COUNT(*)
                FROM {meta_table}
                """,
                timeout_ms=DETAIL_QUERY_TIMEOUT_MS,
            )
            embedded_messages = exact_embedded_messages if exact_embedded_messages is not None else 0
        else:
            embedded_messages = 0
        failure_count = (
            _scalar_int(conn, f"SELECT COUNT(*) FROM {status_table} WHERE error_message IS NOT NULL")
            if has_status
            else 0
        )
        pending_messages = 0
        pending_messages_exact = include_detail
        stale_messages = 0
        missing_provenance = 0
        oldest_embedded_at: str | None = None
        newest_embedded_at: str | None = None
        model_counts: dict[str, int] = {}
        dimension_counts: dict[int, int] = {}
        if has_meta:
            model_rows = _rows_with_timeout(
                conn,
                f"""
                SELECT model, COUNT(*)
                FROM {meta_table}
                GROUP BY model
                ORDER BY COUNT(*) DESC, model ASC
                """,
                timeout_ms=METADATA_SUMMARY_TIMEOUT_MS,
            )
            if model_rows is not None:
                model_counts = {str(row[0]): _payload_int(row[1]) for row in model_rows if row[0] is not None}
            dimension_rows = _rows_with_timeout(
                conn,
                f"""
                SELECT dimension, COUNT(*)
                FROM {meta_table}
                GROUP BY dimension
                ORDER BY COUNT(*) DESC, dimension ASC
                """,
                timeout_ms=METADATA_SUMMARY_TIMEOUT_MS,
            )
            if dimension_rows is not None:
                dimension_counts = {
                    _payload_int(row[0]): _payload_int(row[1]) for row in dimension_rows if row[0] is not None
                }
        if include_detail and has_messages:
            messages_ref = archive_embedding_messages_table_ref(conn, alias="m")
            embeddable_where = archive_embeddable_message_where("m")
            total_messages = _scalar_int_with_timeout(
                conn,
                f"SELECT COUNT(*) FROM {messages_ref} WHERE {embeddable_where}",
                timeout_ms=DETAIL_QUERY_TIMEOUT_MS,
            )
            if total_messages is None:
                total_messages = 0
                pending_messages_exact = False
            if has_meta and embedded_messages == 0:
                pending_messages = total_messages
            elif has_meta:
                meta_join = "ON em.message_id = m.message_id"
                meta_missing_column = "em.message_id"
                status_join = f"LEFT JOIN {status_table} e ON e.session_id = m.session_id" if has_status else ""
                status_reindex_clause = "OR COALESCE(e.needs_reindex, 0) = 1" if has_status else ""
                exact_pending_messages = _scalar_int_with_timeout(
                    conn,
                    f"""
                    SELECT COUNT(*)
                    FROM {messages_ref}
                    LEFT JOIN {meta_table} em {meta_join}
                    {status_join}
                    WHERE {embeddable_where}
                      AND (
                        {meta_missing_column} IS NULL
                        OR COALESCE(em.needs_reindex, 0) = 1
                        {status_reindex_clause}
                      )
                    """,
                    timeout_ms=DETAIL_QUERY_TIMEOUT_MS,
                )
                if exact_pending_messages is None:
                    pending_messages = 0
                    pending_messages_exact = False
                else:
                    pending_messages = exact_pending_messages
            else:
                pending_messages = total_messages
            if has_meta and embedded_messages == 0:
                missing_provenance = 0
                stale_messages = 0
            elif has_meta and pending_messages_exact:
                meta_join = "ON em.message_id = m.message_id"
                meta_missing_column = "em.message_id"
                if vector_table:
                    exact_missing_provenance = _scalar_int_with_timeout(
                        conn,
                        f"""
                        SELECT COUNT(*)
                        FROM {vector_table} me
                        LEFT JOIN {meta_table} em
                          ON em.message_id = me.message_id
                        WHERE em.message_id IS NULL
                        """,
                        timeout_ms=DETAIL_QUERY_TIMEOUT_MS,
                    )
                    if exact_missing_provenance is None:
                        missing_provenance = 0
                        pending_messages_exact = False
                    else:
                        missing_provenance = exact_missing_provenance
                if pending_messages_exact:
                    exact_stale_messages = _scalar_int_with_timeout(
                        conn,
                        f"""
                        SELECT COUNT(*)
                        FROM {messages_ref}
                        JOIN {meta_table} em {meta_join}
                        WHERE {embeddable_where}
                          AND (
                            COALESCE(em.needs_reindex, 0) = 1
                            OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
                          )
                        """,
                        timeout_ms=DETAIL_QUERY_TIMEOUT_MS,
                    )
                    if exact_stale_messages is None:
                        stale_messages = 0
                        pending_messages_exact = False
                    else:
                        stale_messages = exact_stale_messages
                if pending_messages_exact:
                    bounds = conn.execute(
                        f"""
                        SELECT MIN(embedded_at_ms), MAX(embedded_at_ms)
                        FROM {meta_table}
                        """
                    ).fetchone()
                    if bounds is not None:
                        oldest_embedded_at = _iso_from_epoch_ms(bounds[0])
                        newest_embedded_at = _iso_from_epoch_ms(bounds[1])
        stats = EmbeddingStatsSnapshot(
            embedded_sessions=embedded_sessions,
            embedded_messages=embedded_messages,
            pending_sessions=pending_sessions,
            pending_messages=pending_messages,
            stale_messages=stale_messages,
            messages_missing_provenance=missing_provenance,
            oldest_embedded_at=oldest_embedded_at,
            newest_embedded_at=newest_embedded_at,
            model_counts=model_counts,
            dimension_counts=dimension_counts,
            retrieval_bands={},
            failure_count=failure_count,
            total_estimated_cost_usd=_estimated_cost(pending_messages) if include_detail else 0.0,
        )
    finally:
        conn.close()

    latest_catchup_run = _archive_latest_catchup_run(index_db.with_name("ops.db"))

    return _payload_from_stats(
        config_enabled=bool(getattr(cfg, "embedding_enabled", False)),
        has_voyage_api_key=bool(getattr(cfg, "voyage_api_key", None)),
        configured_model=str(getattr(cfg, "embedding_model", "")),
        configured_dimension=_payload_int(getattr(cfg, "embedding_dimension", 0)),
        monthly_cost_cap_usd=float(getattr(cfg, "embedding_max_cost_usd", 0.0) or 0.0),
        total_sessions=total_sessions,
        stats=stats,
        latest_catchup_run=latest_catchup_run,
        pending_messages_exact=pending_messages_exact,
    )


def _archive_latest_catchup_run(ops_db: Path) -> EmbeddingCatchupRunPayload | None:
    if not ops_db.exists():
        return None
    try:
        from polylogue.storage.sqlite.archive_tiers.ops_write import list_embedding_catchup_runs

        conn = open_readonly_connection(ops_db)
        try:
            runs = list_embedding_catchup_runs(conn)
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not runs:
        return None
    run = runs[0]
    started_at = _iso_from_epoch_ms(run.started_at_ms) or ""
    finished_at = _iso_from_epoch_ms(run.finished_at_ms)
    return {
        "run_id": run.run_id,
        "started_at": started_at,
        "updated_at": finished_at or started_at,
        "completed_at": finished_at,
        "status": run.status,
        "stop_reason": run.error_message,
        "rebuild": False,
        "max_sessions": None,
        "max_messages": None,
        "stop_after_seconds": None,
        "max_errors": None,
        "planned_sessions": run.scanned_sessions,
        "planned_messages": 0,
        "processed_sessions": run.scanned_sessions,
        "embedded_sessions": 0,
        "skipped_sessions": 0,
        "error_count": 1 if run.error_message else 0,
        "embedded_messages": run.embedded_messages,
        "estimated_cost_usd": float(run.estimated_cost_usd or 0.0),
        "last_session_id": None,
    }


def embedding_status_payload(
    env: _HasConfig,
    *,
    include_retrieval_bands: bool = False,
    include_detail: bool = False,
) -> EmbeddingStatusPayload:
    """Read canonical embedding-status statistics for operator surfaces."""
    from polylogue.config import load_polylogue_config
    from polylogue.storage.embeddings.embedding_stats import read_embedding_stats_sync
    from polylogue.storage.embeddings.progress import latest_embedding_catchup_run
    from polylogue.storage.embeddings.support import table_exists_sync

    cfg = load_polylogue_config()
    db_path = Path(env.config.db_path)
    archive_payload = _archive_embedding_status_payload(db_path, cfg=cfg, include_detail=include_detail)
    if archive_payload is not None:
        return archive_payload
    if not db_path.exists():
        return _payload_from_stats(
            config_enabled=bool(cfg.embedding_enabled),
            has_voyage_api_key=bool(cfg.voyage_api_key),
            configured_model=cfg.embedding_model,
            configured_dimension=cfg.embedding_dimension,
            monthly_cost_cap_usd=cfg.embedding_max_cost_usd,
            total_sessions=0,
            stats=EmbeddingStatsSnapshot(),
            latest_catchup_run=None,
            pending_messages_exact=include_detail,
        )

    conn = open_readonly_connection(db_path)
    try:
        total_sessions = _total_sessions(conn)
        embedding_stats = read_embedding_stats_sync(
            conn,
            include_retrieval_bands=include_retrieval_bands,
            detail=include_detail,
        )
        latest_run = latest_embedding_catchup_run(conn) if table_exists_sync(conn, "embedding_catchup_runs") else None
    finally:
        conn.close()

    return _payload_from_stats(
        config_enabled=bool(cfg.embedding_enabled),
        has_voyage_api_key=bool(cfg.voyage_api_key),
        configured_model=cfg.embedding_model,
        configured_dimension=cfg.embedding_dimension,
        monthly_cost_cap_usd=cfg.embedding_max_cost_usd,
        total_sessions=total_sessions,
        stats=embedding_stats,
        latest_catchup_run=latest_run,
        pending_messages_exact=include_detail,
    )
