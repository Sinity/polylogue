"""Embedding readiness snapshot for daemon status surfaces."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polylogue.config as polylogue_config
from polylogue.storage.embeddings.support import embedded_message_count_sync, optional_count_sync, table_exists_sync
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
        "embedding_pending_count": 0,
        "embedding_pending_message_count": 0,
        "embedding_stale_count": 0,
        "embedding_coverage_percent": 0.0,
        "embedding_failure_count": 0,
        "embedding_estimated_cost_usd": 0.0,
        "embedding_latest_catchup_run": None,
    }


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not table_exists_sync(conn, table):
        return False
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})").fetchall())


def embedding_readiness_info(db_file: Path, *, detail: bool = False) -> dict[str, object]:
    """Query embedding tables for bounded daemon status visibility."""
    cfg = polylogue_config.load_polylogue_config()
    config_enabled = bool(cfg.embedding_enabled)
    has_key = cfg.voyage_api_key is not None
    enabled = config_enabled and has_key
    model = cfg.embedding_model
    dimension = cfg.embedding_dimension

    if not db_file.exists():
        return _defaults(
            enabled=enabled,
            config_enabled=config_enabled,
            has_key=has_key,
            model=model,
            dimension=dimension,
        )

    pending = pending_messages = stale = failure = total = total_messages = total_conv = embedded_conv = 0
    cost = 0.0
    latest_catchup_run: object | None = None
    try:
        conn = open_readonly_connection(db_file)
        try:
            from polylogue.storage.embeddings.progress import latest_embedding_catchup_run

            embedded_msg = embedded_message_count_sync(conn)
            if table_exists_sync(conn, "embedding_catchup_runs"):
                latest_catchup_run = latest_embedding_catchup_run(conn)
            if _column_exists(conn, "embedding_status", "error_message"):
                failure = optional_count_sync(
                    conn, "SELECT COUNT(*) FROM embedding_status WHERE error_message IS NOT NULL"
                )
            embedded_conv = optional_count_sync(conn, "SELECT COUNT(*) FROM embedding_status WHERE needs_reindex = 0")
            total_conv = (
                int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] or 0)
                if table_exists_sync(conn, "conversations")
                else 0
            )
            if detail:
                pending = optional_count_sync(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM conversations c
                    LEFT JOIN embedding_status e ON e.conversation_id = c.conversation_id
                    WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                    """,
                )
                pending_messages = optional_count_sync(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM messages m
                    JOIN conversations c ON c.conversation_id = m.conversation_id
                    LEFT JOIN embedding_status e ON e.conversation_id = c.conversation_id
                    WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                    """,
                )
                stale = optional_count_sync(
                    conn,
                    """
                    SELECT COUNT(*)
                    FROM message_embeddings me
                    JOIN messages m ON m.message_id = me.message_id
                    LEFT JOIN embeddings_meta em
                      ON em.target_id = me.message_id
                     AND em.target_type = 'message'
                    WHERE em.target_id IS NULL
                       OR (em.content_hash IS NOT NULL AND em.content_hash != m.content_hash)
                    """,
                )
                total_messages = (
                    int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] or 0)
                    if table_exists_sync(conn, "messages")
                    else 0
                )
            pending = max(pending, total_conv - embedded_conv)
            if detail:
                pending_messages = max(pending_messages, total_messages - embedded_msg if pending > 0 else 0)
            total = embedded_msg if total_conv > 0 else 0
            estimated_tokens = (total + pending_messages) * ESTIMATED_TOKENS_PER_MESSAGE
            cost = round(estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000, 2)
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        pass

    coverage = (max(0, total_conv - pending) / total_conv * 100) if total_conv > 0 else 0.0
    return {
        "embedding_enabled": enabled,
        "embedding_config_enabled": config_enabled,
        "embedding_has_voyage_key": has_key,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "embedding_pending_count": pending,
        "embedding_pending_message_count": pending_messages,
        "embedding_stale_count": stale,
        "embedding_coverage_percent": round(coverage, 1),
        "embedding_failure_count": failure,
        "embedding_estimated_cost_usd": cost,
        "embedding_latest_catchup_run": latest_catchup_run,
    }


__all__ = ["embedding_readiness_info"]
