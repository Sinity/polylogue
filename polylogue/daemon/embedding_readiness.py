"""Embedding readiness snapshot for daemon status surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import polylogue.config as polylogue_config
from polylogue.storage.embeddings.status_payload import embedding_status_payload
from polylogue.storage.search_providers.sqlite_vec_support import (
    ESTIMATED_TOKENS_PER_MESSAGE,
    VOYAGE_4_COST_PER_1M_TOKENS,
)
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _defaults(*, enabled: bool, config_enabled: bool, has_key: bool, model: str, dimension: int) -> dict[str, object]:
    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_status": "empty",
        "embedding_freshness_status": "empty",
        "embedding_retrieval_ready": False,
        "embedding_pending_count": 0,
        "embedding_pending_message_count": 0,
        "embedding_pending_message_count_exact": False,
        "embedding_stale_count": 0,
        "embedding_coverage_percent": 0.0,
        "embedding_failure_count": 0,
        "embedding_estimated_cost_usd": 0.0,
        "embedding_latest_catchup_run": None,
    }


def embedding_readiness_info(db_file: Path, *, detail: bool = False) -> dict[str, object]:
    """Query embedding tables for bounded daemon status visibility."""
    cfg = polylogue_config.load_polylogue_config()
    config_enabled = bool(cfg.embedding_enabled)
    has_key = cfg.voyage_api_key is not None
    enabled = config_enabled and has_key
    model = cfg.embedding_model
    dimension = cfg.embedding_dimension
    index_db = _archive_index_path_for(db_file)

    if index_db is not None:
        archive_info = _archive_embedding_readiness_info(
            index_db,
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
            detail=detail,
        )
        if archive_info is not None:
            return archive_info

    if not db_file.exists():
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
        )

    try:
        payload = embedding_status_payload(
            SimpleNamespace(config=SimpleNamespace(db_path=db_file)),
            include_retrieval_bands=False,
            include_detail=detail,
        )
    except (sqlite3.Error, OSError):
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
        )

    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_status": payload["status"],
        "embedding_freshness_status": payload["freshness_status"],
        "embedding_retrieval_ready": payload["retrieval_ready"],
        "embedding_pending_count": payload["pending_sessions"],
        "embedding_pending_message_count": payload["pending_messages"] or 0,
        "embedding_pending_message_count_exact": payload["pending_messages_exact"],
        "embedding_stale_count": payload["stale_messages"],
        "embedding_coverage_percent": payload["embedding_coverage_percent"],
        "embedding_failure_count": payload["failure_count"],
        "embedding_estimated_cost_usd": payload["total_estimated_cost_usd"],
        "embedding_latest_catchup_run": payload["latest_catchup_run"],
    }


def _archive_index_path_for(db_file: Path) -> Path | None:
    if db_file.name == "index.db":
        return db_file if db_file.exists() else None
    index_db = db_file.with_name("index.db")
    return index_db if index_db.exists() else None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
            (table_name,),
        ).fetchone()
        is not None
    )


def _scalar_int(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    if row is None or row[0] is None:
        return 0
    value = row[0]
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


def _estimated_cost(message_count: int) -> float:
    estimated_tokens = message_count * ESTIMATED_TOKENS_PER_MESSAGE
    return round(estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000, 2)


def _archive_embedding_readiness_info(
    index_db: Path,
    *,
    enabled: bool,
    config_enabled: bool,
    has_key: bool,
    model: str,
    dimension: int,
    detail: bool,
) -> dict[str, object] | None:
    try:
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
            total_messages = 0
            if detail and has_messages:
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
            status = _embedding_status(
                total_sessions=total_sessions,
                embedded_sessions=embedded_sessions,
                pending_sessions=pending_sessions,
            )
            freshness_status = "stale" if embedded_messages > 0 and stale_messages > 0 else status
            return {
                "embedding_enabled": enabled,
                "embedding_config_enabled": config_enabled,
                "embedding_has_voyage_key": has_key,
                "embedding_model": model,
                "embedding_dimension": dimension,
                "embedding_status": status,
                "embedding_freshness_status": freshness_status,
                "embedding_retrieval_ready": embedded_messages > stale_messages,
                "embedding_pending_count": pending_sessions,
                "embedding_pending_message_count": pending_messages,
                "embedding_pending_message_count_exact": detail,
                "embedding_stale_count": stale_messages,
                "embedding_coverage_percent": round(
                    (embedded_sessions / total_sessions) * 100,
                    1,
                )
                if total_sessions > 0
                else 0.0,
                "embedding_failure_count": failure_count,
                "embedding_estimated_cost_usd": _estimated_cost(pending_messages) if detail else 0.0,
                "embedding_latest_catchup_run": None,
            }
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
        )


__all__ = ["embedding_readiness_info"]
