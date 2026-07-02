"""Bounded ambient embedding backlog drain for the daemon."""

from __future__ import annotations

import asyncio
import sqlite3
import time
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
    from polylogue.paths import active_index_db_path

    db = active_index_db_path()
    while True:
        await asyncio.sleep(EMBEDDING_BACKLOG_RETRY_INTERVAL_SECONDS)
        try:
            processed = await asyncio.to_thread(drain_embedding_backlog_once, db)
            if processed:
                logger.info("embed: drained %d pending session(s)", processed)
        except sqlite3.OperationalError as exc:
            if is_transient_sqlite_lock(exc):
                logger.info("embed: archive busy; retrying backlog on next tick: %s", exc)
                continue
            logger.warning("embed: backlog check failed", exc_info=True)
        except Exception:
            logger.warning("embed: backlog check failed", exc_info=True)


def drain_embedding_backlog_once(db_path: Path) -> int:
    """Run one bounded daemon embedding catch-up window over pending backlog."""

    from polylogue.daemon.convergence_stages import _embedding_config_enabled

    if not _embedding_config_enabled():
        return 0

    index_db = _active_archive_index_path(db_path)
    if index_db is None:
        return 0
    return _drain_archive_embedding_backlog_once(index_db)


def _active_archive_index_path(db_path: Path) -> Path | None:
    from polylogue.paths import active_index_db_path

    candidates = []
    active_db = active_index_db_path()
    if active_db.name == "index.db":
        candidates.append(active_db)
    if db_path.name == "index.db":
        candidates.append(db_path)
    candidates.append(db_path.with_name("index.db"))
    index_db = next((candidate for candidate in dict.fromkeys(candidates) if candidate.exists()), None)
    if index_db is None:
        return None
    try:
        with closing(open_readonly_connection(index_db, timeout=5.0)) as conn:
            return index_db if _table_exists(conn, "sessions") else None
    except Exception:
        logger.warning("embed: failed to inspect archive index", exc_info=True)
        return None


def _drain_archive_embedding_backlog_once(index_db: Path) -> int:
    from polylogue.daemon.convergence_stages import (
        _DAEMON_EMBED_MAX_ERRORS,
        _DAEMON_EMBED_MAX_MESSAGES,
        _DAEMON_EMBED_MAX_SESSIONS,
        _DAEMON_EMBED_STOP_AFTER_SECONDS,
    )
    from polylogue.storage.embeddings.materialization import (
        embed_archive_session_sync,
        select_pending_archive_session_window,
    )
    from polylogue.storage.search_providers import create_vector_provider
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    cfg = load_polylogue_config()
    voyage_key = cfg.get("voyage_api_key")
    if not voyage_key:
        return 0
    embeddings_db = index_db.with_name("embeddings.db")
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(embeddings_db, ArchiveTier.EMBEDDINGS)
    monthly_cap = float(str(cfg.get("embedding_max_cost_usd", 0.0)))
    ops_db = index_db.with_name("ops.db")
    if monthly_cap > 0:
        month_spend = _archive_embedding_catchup_estimated_cost_this_month(ops_db)
        if month_spend >= monthly_cap:
            logger.info(
                "embed: archive monthly cost cap exhausted (%.4f >= %.2f); backlog drain paused",
                month_spend,
                monthly_cap,
            )
            return 0
        monthly_cap = max(monthly_cap - month_spend, 0.0)
    try:
        with closing(open_readonly_connection(index_db, timeout=5.0)) as conn:
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
            pending = select_pending_archive_session_window(
                conn,
                status_table="embeddings.embedding_status",
                max_sessions=_DAEMON_EMBED_MAX_SESSIONS,
                max_messages=_DAEMON_EMBED_MAX_MESSAGES,
            )
    except Exception:
        logger.warning("embed: failed to inspect archive pending backlog", exc_info=True)
        return 0
    if not pending:
        return 0

    started_at_ms = int(time.time() * 1000)
    run_id = _upsert_archive_embedding_catchup_run(
        ops_db,
        status="running",
        started_at_ms=started_at_ms,
        scanned_sessions=0,
        embedded_messages=0,
        estimated_cost_usd=0.0,
    )
    vec_provider = create_vector_provider(
        voyage_api_key=str(voyage_key),
        db_path=embeddings_db,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
    )
    if vec_provider is None:
        logger.warning("embed: archive vector provider unavailable")
        _upsert_archive_embedding_catchup_run(
            ops_db,
            run_id=run_id,
            status="failed",
            started_at_ms=started_at_ms,
            finished_at_ms=int(time.time() * 1000),
            error_message="vector provider unavailable",
        )
        return 0

    embedded = 0
    embedded_messages = 0
    errors = 0
    skipped = 0
    processed = 0
    error_message: str | None = None
    cumulative_cost = 0.0
    start_monotonic = time.monotonic()
    for item in pending:
        if time.monotonic() - start_monotonic >= _DAEMON_EMBED_STOP_AFTER_SECONDS:
            break
        estimated_batch_cost = (
            item.message_count * ESTIMATED_TOKENS_PER_MESSAGE * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
        )
        if monthly_cap > 0.0 and cumulative_cost + estimated_batch_cost > monthly_cap:
            logger.info(
                "embed: archive cost cap would be exceeded (%.4f > %.2f); backlog drain paused",
                cumulative_cost + estimated_batch_cost,
                monthly_cap,
            )
            break
        outcome = embed_archive_session_sync(index_db, vec_provider, item.session_id)
        processed += 1
        if outcome.status == "embedded":
            embedded += 1
            embedded_messages += outcome.embedded_message_count
            cumulative_cost += (
                outcome.embedded_message_count * ESTIMATED_TOKENS_PER_MESSAGE * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
            )
            if monthly_cap > 0.0 and cumulative_cost > monthly_cap:
                logger.info(
                    "embed: archive cost cap reached (%.4f > %.2f); backlog drain paused",
                    cumulative_cost,
                    monthly_cap,
                )
                break
        elif outcome.status in {"no_messages", "no_embeddable_messages"}:
            skipped += 1
            logger.info("embed: archive %s has no embeddable messages", item.session_id)
        elif outcome.status == "error":
            errors += 1
            error_message = outcome.error
            logger.warning("embed: archive %s failed: %s", outcome.session_id, outcome.error)
            if errors >= _DAEMON_EMBED_MAX_ERRORS:
                break
    logger.info("embed: archive %d done, %d errors, est. cost $%.4f", embedded, errors, cumulative_cost)
    _upsert_archive_embedding_catchup_run(
        ops_db,
        run_id=run_id,
        status="failed" if errors else "completed",
        started_at_ms=started_at_ms,
        finished_at_ms=int(time.time() * 1000),
        scanned_sessions=processed,
        embedded_sessions=embedded,
        skipped_sessions=skipped,
        error_count=errors,
        embedded_messages=embedded_messages,
        estimated_cost_usd=cumulative_cost,
        error_message=error_message,
    )
    return processed if errors == 0 else 0


def _archive_embedding_catchup_estimated_cost_this_month(ops_db: Path) -> float:
    if not ops_db.exists():
        return 0.0
    try:
        with closing(open_readonly_connection(ops_db, timeout=5.0)) as conn:
            if not _table_exists(conn, "embedding_catchup_runs"):
                return 0.0
            row = conn.execute(
                """
                SELECT COALESCE(SUM(estimated_cost_usd), 0.0)
                FROM embedding_catchup_runs
                WHERE strftime('%Y-%m', started_at_ms / 1000, 'unixepoch') = strftime('%Y-%m', 'now')
                """
            ).fetchone()
            return float(row[0] or 0.0) if row is not None else 0.0
    except Exception:
        logger.warning("embed: failed to inspect archive catch-up spend", exc_info=True)
        return 0.0


def _upsert_archive_embedding_catchup_run(
    ops_db: Path,
    *,
    status: str,
    started_at_ms: int,
    run_id: str | None = None,
    finished_at_ms: int | None = None,
    scanned_sessions: int = 0,
    embedded_sessions: int = 0,
    skipped_sessions: int = 0,
    error_count: int = 0,
    embedded_messages: int = 0,
    estimated_cost_usd: float | None = None,
    error_message: str | None = None,
) -> str:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db.parent.mkdir(parents=True, exist_ok=True)
    from polylogue.storage.sqlite.connection_profile import open_daemon_connection

    with open_daemon_connection(ops_db, timeout=30.0) as conn:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        return upsert_embedding_catchup_run(
            conn,
            run_id=run_id,
            status=status,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            scanned_sessions=scanned_sessions,
            embedded_sessions=embedded_sessions,
            skipped_sessions=skipped_sessions,
            error_count=error_count,
            embedded_messages=embedded_messages,
            estimated_cost_usd=estimated_cost_usd,
            error_message=error_message,
        )


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
