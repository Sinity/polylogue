"""Embedding-status payload builder (substrate, click-free).

Counts come from a sync read connection over the embedding-status tables.
Surfaces (CLI, MCP, dashboards) consume :func:`embedding_status_payload`
and render in their own dialect.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from typing_extensions import TypedDict

from polylogue.storage.embeddings.models import EmbeddingStatsSnapshot
from polylogue.storage.embeddings.progress import EmbeddingCatchupRunPayload
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

if TYPE_CHECKING:
    from polylogue.config import Config


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


def _scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    if row is None:
        return 0
    return _payload_int(row[0])


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


def _coverage_percent(*, embedded_sessions: int, total_sessions: int) -> float:
    if total_sessions <= 0:
        return 0.0
    return embedded_sessions / total_sessions * 100


def _embedding_status(
    *,
    total_sessions: int,
    embedded_sessions: int,
    pending_sessions: int,
) -> str:
    if total_sessions <= 0:
        return "empty"
    if embedded_sessions <= 0:
        return "none"
    if pending_sessions > 0:
        return "partial"
    return "complete"


def _freshness_status(status: str, stats: EmbeddingStatsSnapshot) -> str:
    if stats.embedded_messages > 0 and (stats.stale_messages > 0 or stats.messages_missing_provenance > 0):
        return "stale"
    return status


def _retrieval_ready(stats: EmbeddingStatsSnapshot) -> bool:
    return stats.embedded_messages > stats.stale_messages


def _next_action(
    *,
    config_enabled: bool,
    has_voyage_api_key: bool,
    total_sessions: int,
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
            "command": "polylogue embed enable --voyage-api-key ...",
            "reason": "Semantic retrieval needs a Voyage API key before embedding can run.",
        }
    if not config_enabled:
        return {
            "code": "enable_embeddings",
            "command": "polylogue embed enable --yes",
            "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
        }
    if failure_count > 0 and pending_sessions > 0:
        return {
            "code": "inspect_failures",
            "command": "polylogue embed status --detail",
            "reason": "Recent embedding failures exist while backlog remains.",
        }
    if stale_messages > 0:
        return {
            "code": "refresh_stale",
            "command": "polylogue embed backfill --max-sessions 10",
            "reason": "Existing vectors are stale for at least one message.",
        }
    if pending_sessions > 0:
        return {
            "code": "drain_backlog",
            "command": "polylogue embed backfill --max-sessions 10",
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
        "command": "polylogue embed preflight --detail",
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
    pending_sessions = stats.pending_sessions or max(total_sessions - embedded_sessions, 0)
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
    conn = open_readonly_connection(index_db)
    try:
        if not _table_exists(conn, "sessions"):
            return None
        embeddings_db = index_db.with_name("embeddings.db")
        if embeddings_db.exists():
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
            status_table = "embeddings.embedding_status"
            embeddings_table = "embeddings.message_embeddings"
            meta_table = "embeddings.embeddings_meta"
        else:
            status_table = ""
            embeddings_table = ""
            meta_table = ""
        has_messages = _table_exists(conn, "messages")
        has_status = bool(status_table)
        has_embeddings = bool(embeddings_table)
        has_meta = bool(meta_table)
        total_sessions = _scalar_int(conn, "SELECT COUNT(*) FROM sessions")
        embedded_sessions = (
            _scalar_int(
                conn,
                f"""
                SELECT COUNT(*)
                FROM sessions s
                JOIN {status_table} e ON e.session_id = s.session_id
                WHERE e.needs_reindex = 0
                  AND e.message_count_embedded >= s.message_count
                """,
            )
            if has_status
            else 0
        )
        pending_sessions = max(total_sessions - embedded_sessions, 0)
        embedded_messages = _scalar_int(conn, f"SELECT COUNT(*) FROM {embeddings_table}") if has_embeddings else 0
        failure_count = (
            _scalar_int(conn, f"SELECT COUNT(*) FROM {status_table} WHERE error_message IS NOT NULL")
            if has_status
            else 0
        )
        pending_messages = 0
        stale_messages = 0
        missing_provenance = 0
        oldest_embedded_at: str | None = None
        newest_embedded_at: str | None = None
        model_counts: dict[str, int] = {}
        dimension_counts: dict[int, int] = {}
        if include_detail and has_messages:
            total_messages = _scalar_int(conn, "SELECT COUNT(*) FROM messages")
            if has_embeddings:
                pending_messages = _scalar_int(
                    conn,
                    f"""
                    SELECT COUNT(*)
                    FROM messages m
                    LEFT JOIN {embeddings_table} me ON me.message_id = m.message_id
                    LEFT JOIN {status_table} e ON e.session_id = m.session_id
                    WHERE me.message_id IS NULL
                       OR e.session_id IS NULL
                       OR e.needs_reindex = 1
                    """,
                )
            else:
                pending_messages = total_messages
            if has_embeddings and has_meta:
                missing_provenance = _scalar_int(
                    conn,
                    f"""
                    SELECT COUNT(*)
                    FROM {embeddings_table} me
                    LEFT JOIN {meta_table} em
                      ON em.target_id = me.message_id
                     AND em.target_type = 'message'
                    WHERE em.target_id IS NULL
                    """,
                )
                stale_messages = _scalar_int(
                    conn,
                    f"""
                    SELECT COUNT(*)
                    FROM {embeddings_table} me
                    JOIN messages m ON m.message_id = me.message_id
                    LEFT JOIN {meta_table} em
                      ON em.target_id = me.message_id
                     AND em.target_type = 'message'
                    WHERE em.target_id IS NULL
                       OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
                    """,
                )
                bounds = conn.execute(
                    f"""
                    SELECT MIN(embedded_at_ms), MAX(embedded_at_ms)
                    FROM {meta_table}
                    WHERE target_type = 'message'
                    """
                ).fetchone()
                if bounds is not None:
                    oldest_embedded_at = _iso_from_epoch_ms(bounds[0])
                    newest_embedded_at = _iso_from_epoch_ms(bounds[1])
                model_counts = {
                    str(row[0]): _payload_int(row[1])
                    for row in conn.execute(
                        f"""
                        SELECT model, COUNT(*)
                        FROM {meta_table}
                        WHERE target_type = 'message'
                        GROUP BY model
                        ORDER BY COUNT(*) DESC, model ASC
                        """
                    ).fetchall()
                    if row[0] is not None
                }
                dimension_counts = {
                    _payload_int(row[0]): _payload_int(row[1])
                    for row in conn.execute(
                        f"""
                        SELECT dimension, COUNT(*)
                        FROM {meta_table}
                        WHERE target_type = 'message'
                        GROUP BY dimension
                        ORDER BY COUNT(*) DESC, dimension ASC
                        """
                    ).fetchall()
                    if row[0] is not None
                }
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
            total_estimated_cost_usd=0.0,
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
        pending_messages_exact=include_detail,
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
