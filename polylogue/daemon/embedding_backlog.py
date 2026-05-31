"""Bounded ambient embedding backlog drain for the daemon."""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path

from polylogue.config import load_polylogue_config
from polylogue.logging import get_logger
from polylogue.sources.live.sqlite_locking import is_transient_sqlite_lock
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS = 60


async def periodic_embedding_backlog_check() -> None:
    """Periodically drain one bounded pending-embedding window."""
    from polylogue.paths import db_path

    db = db_path()
    while True:
        await asyncio.sleep(EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS)
        if not db.exists():
            continue
        try:
            processed = await asyncio.to_thread(drain_embedding_backlog_once, db)
            if processed:
                logger.info("embed: drained %d pending conversation(s)", processed)
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("embed: archive busy; retrying backlog on next tick: %s", exc)
                continue
            logger.warning("embed: backlog check failed", exc_info=True)
        except Exception:
            logger.warning("embed: backlog check failed", exc_info=True)


def drain_embedding_backlog_once(db_path: Path) -> int:
    """Run one bounded daemon embedding catch-up window over pending backlog."""

    from polylogue.daemon.convergence_stages import (
        _DAEMON_EMBED_MAX_CONVERSATIONS,
        _DAEMON_EMBED_MAX_ERRORS,
        _DAEMON_EMBED_MAX_MESSAGES,
        _DAEMON_EMBED_STOP_AFTER_SECONDS,
        _embed_conversations_sync,
        _embedding_config_enabled,
    )
    from polylogue.storage.embeddings.materialization import select_pending_conversation_window

    if not db_path.exists() or not _embedding_config_enabled():
        return 0

    cfg = load_polylogue_config()
    monthly_cap = float(str(cfg.get("embedding_max_cost_usd", 0.0)))
    try:
        with closing(open_readonly_connection(db_path, timeout=5.0)) as conn:
            if monthly_cap > 0:
                month_spend = embedding_catchup_estimated_cost_this_month(conn)
                if month_spend >= monthly_cap:
                    logger.info(
                        "embed: monthly cost cap exhausted (%.4f >= %.2f); backlog drain paused",
                        month_spend,
                        monthly_cap,
                    )
                    return 0
                remaining_cap = max(monthly_cap - month_spend, 0.0)
            else:
                remaining_cap = None
            pending = select_pending_conversation_window(
                conn,
                max_conversations=_DAEMON_EMBED_MAX_CONVERSATIONS,
                max_messages=_DAEMON_EMBED_MAX_MESSAGES,
            )
    except Exception:
        logger.warning("embed: failed to inspect pending backlog", exc_info=True)
        return 0
    if not pending:
        return 0
    conversation_ids = tuple(item.conversation_id for item in pending)
    ok = _embed_conversations_sync(
        db_path,
        conversation_ids,
        max_errors=_DAEMON_EMBED_MAX_ERRORS,
        stop_after_seconds=_DAEMON_EMBED_STOP_AFTER_SECONDS,
        max_cost_usd=remaining_cap,
    )
    return len(conversation_ids) if ok else 0


def embedding_catchup_estimated_cost_this_month(conn: sqlite3.Connection) -> float:
    if not _table_exists(conn, "embedding_catchup_runs"):
        return 0.0
    row = conn.execute(
        """
        SELECT COALESCE(SUM(estimated_cost_usd), 0.0)
        FROM embedding_catchup_runs
        WHERE strftime('%Y-%m', started_at) = strftime('%Y-%m', 'now')
        """
    ).fetchone()
    return float(row[0] or 0.0) if row is not None else 0.0


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


__all__ = [
    "EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS",
    "drain_embedding_backlog_once",
    "embedding_catchup_estimated_cost_this_month",
    "periodic_embedding_backlog_check",
]
