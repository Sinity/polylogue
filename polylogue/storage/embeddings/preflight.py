"""Read-only embedding catch-up preflight estimates.

The CLI, MCP, and daemon-facing operator surfaces share this module so
semantic-search catch-up planning does not depend on Click command internals.
The report never contacts Voyage; it only reads the archive and local config.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Cost preflight numbers for the configured archive."""

    total_sessions: int
    pending_sessions: int
    pending_messages: int
    estimated_tokens: int
    estimated_cost_usd: float
    model: str
    dimension: int
    cost_cap_usd: float
    windowed: bool = False
    max_sessions: int | None = None
    max_messages: int | None = None
    max_cost_usd: float | None = None


def message_window_for_cost(max_cost_usd: float | None) -> int | None:
    if max_cost_usd is None:
        return None
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    cost_per_message = ESTIMATED_TOKENS_PER_MESSAGE * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
    return max(1, int(max_cost_usd / cost_per_message))


def effective_message_window(max_messages: int | None, max_cost_usd: float | None) -> int | None:
    cost_messages = message_window_for_cost(max_cost_usd)
    if max_messages is None:
        return cost_messages
    if cost_messages is None:
        return max_messages
    return min(max_messages, cost_messages)


def effective_cost_cap(config_cap_usd: float, run_cap_usd: float | None) -> float:
    if run_cap_usd is None:
        return config_cap_usd
    if config_cap_usd <= 0:
        return run_cap_usd
    return min(config_cap_usd, run_cap_usd)


def read_pending_message_count(
    db_path: Path,
    *,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(total_convs, pending_convs, pending_messages)``.

    Pending = no ``embedding_status`` row, or ``needs_reindex = 1``.
    Reading happens against a sync read connection so the command works even
    when the daemon is not running.
    """
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    index_db = _archive_index_path(db_path)
    if index_db is not None and _is_archive_index(index_db):
        return _read_archive_pending_message_count(
            index_db,
            rebuild=rebuild,
            max_sessions=max_sessions,
            max_messages=max_messages,
        )

    if not db_path.exists():
        return 0, 0, 0

    conn = open_readonly_connection(db_path)
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        if max_sessions is not None or max_messages is not None:
            from polylogue.api import select_pending_embedding_session_window

            pending = select_pending_embedding_session_window(
                conn,
                rebuild=rebuild,
                max_sessions=max_sessions,
                max_messages=max_messages,
            )
            return total, len(pending), sum(item.message_count for item in pending)
        try:
            if rebuild:
                pending_convs = total
                pending_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
                return total, pending_convs, pending_messages
            pending_convs = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM sessions c
                    LEFT JOIN embedding_status e
                      ON c.session_id = e.session_id
                    WHERE e.session_id IS NULL OR e.needs_reindex = 1
                    """
                ).fetchone()[0]
            )
            pending_messages = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages m
                    JOIN sessions c ON c.session_id = m.session_id
                    LEFT JOIN embedding_status e
                      ON c.session_id = e.session_id
                    WHERE e.session_id IS NULL OR e.needs_reindex = 1
                    """
                ).fetchone()[0]
            )
        except sqlite3.OperationalError:
            # embedding_status table may not exist on a fresh / never-embedded DB.
            pending_convs = total
            pending_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    finally:
        conn.close()
    return total, pending_convs, pending_messages


def _archive_index_path(db_path: Path) -> Path | None:
    if db_path.name == "index.db":
        return db_path if db_path.exists() else None
    index_db = db_path.with_name("index.db")
    if index_db.exists():
        return index_db
    from polylogue.paths import archive_root

    configured_index_db = archive_root() / "index.db"
    return configured_index_db if configured_index_db.exists() else None


def _read_archive_pending_message_count(
    index_db: Path,
    *,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
) -> tuple[int, int, int]:
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    conn = open_readonly_connection(index_db)
    try:
        if not _table_exists(conn, "sessions"):
            return 0, 0, 0
        embeddings_db = index_db.with_name("embeddings.db")
        if embeddings_db.exists():
            conn.execute("ATTACH DATABASE ? AS embeddings", (str(embeddings_db),))
            status_table = "embeddings.embedding_status"
        else:
            status_table = ""
        total = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        if max_sessions is not None or max_messages is not None:
            pending = _select_archive_pending_window(
                conn,
                status_table=status_table,
                rebuild=rebuild,
                max_sessions=max_sessions,
                max_messages=max_messages,
            )
            return total, len(pending), sum(item[1] for item in pending)
        if rebuild or not status_table:
            pending_convs = total
            pending_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
            return total, pending_convs, pending_messages
        pending_convs = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM sessions s
                LEFT JOIN {status_table} e ON e.session_id = s.session_id
                WHERE e.session_id IS NULL OR e.needs_reindex = 1
                """
            ).fetchone()[0]
        )
        pending_messages = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                FROM messages m
                JOIN sessions s ON s.session_id = m.session_id
                LEFT JOIN {status_table} e ON e.session_id = s.session_id
                WHERE e.session_id IS NULL OR e.needs_reindex = 1
                """
            ).fetchone()[0]
        )
    finally:
        conn.close()
    return total, pending_convs, pending_messages


def _is_archive_index(path: Path) -> bool:
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    try:
        conn = open_readonly_connection(path)
    except sqlite3.Error:
        return False
    try:
        return _table_exists(conn, "sessions")
    finally:
        conn.close()


def _select_archive_pending_window(
    conn: sqlite3.Connection,
    *,
    status_table: str,
    rebuild: bool,
    max_sessions: int | None,
    max_messages: int | None,
) -> list[tuple[str, int]]:
    pending: list[tuple[str, int]] = []
    message_total = 0
    where_clause = "1 = 1" if rebuild or not status_table else "(e.session_id IS NULL OR e.needs_reindex = 1)"
    join_clause = f"LEFT JOIN {status_table} e ON e.session_id = s.session_id" if status_table else ""
    cursor = conn.execute(
        f"""
        SELECT s.session_id, s.message_count
        FROM sessions s
        {join_clause}
        WHERE {where_clause}
        ORDER BY s.sort_key_ms IS NULL, s.sort_key_ms, s.session_id
        """
    )
    while True:
        rows = cursor.fetchmany(500)
        if not rows:
            break
        for row in rows:
            session_id = str(row[0])
            message_count = int(row[1] or 0)
            if max_sessions is not None and len(pending) >= max_sessions:
                return pending
            if max_messages is not None and pending and message_total + message_count > max_messages:
                return pending
            pending.append((session_id, message_count))
            message_total += message_count
            if max_messages is not None and message_total >= max_messages:
                return pending
    return pending


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ? LIMIT 1",
        (table,),
    ).fetchone()
    return row is not None


def build_preflight_report(
    db_path: Path,
    *,
    rebuild: bool = False,
    max_sessions: int | None = None,
    max_messages: int | None = None,
    max_cost_usd: float | None = None,
) -> PreflightReport:
    """Build a :class:`PreflightReport` without contacting Voyage."""
    from polylogue.config import load_polylogue_config
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    cfg = load_polylogue_config()
    effective_max_messages = effective_message_window(max_messages, max_cost_usd)
    total, pending, pending_messages = read_pending_message_count(
        db_path,
        rebuild=rebuild,
        max_sessions=max_sessions,
        max_messages=effective_max_messages,
    )
    estimated_tokens = pending_messages * ESTIMATED_TOKENS_PER_MESSAGE
    estimated_cost = estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
    return PreflightReport(
        total_sessions=total,
        pending_sessions=pending,
        pending_messages=pending_messages,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_cost,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
        cost_cap_usd=cfg.embedding_max_cost_usd,
        windowed=max_sessions is not None or max_messages is not None or max_cost_usd is not None,
        max_sessions=max_sessions,
        max_messages=effective_max_messages,
        max_cost_usd=max_cost_usd,
    )


def preflight_backfill_args(report: PreflightReport) -> list[str] | None:
    if report.pending_sessions <= 0:
        return None
    args = ["ops", "embed", "backfill", "--yes"]
    if report.max_sessions is not None:
        args.extend(["--max-sessions", str(report.max_sessions)])
    if report.max_messages is not None:
        args.extend(["--max-messages", str(report.max_messages)])
    if report.max_cost_usd is not None:
        args.extend(["--max-cost-usd", f"{report.max_cost_usd:.4f}".rstrip("0").rstrip(".")])
    return args


def preflight_payload(report: PreflightReport) -> dict[str, object]:
    from polylogue.storage.search_providers.sqlite_vec_support import (
        ESTIMATED_TOKENS_PER_MESSAGE,
        VOYAGE_4_COST_PER_1M_TOKENS,
    )

    backfill_args = preflight_backfill_args(report)
    return {
        "total_sessions": report.total_sessions,
        "pending_sessions": report.pending_sessions,
        "pending_messages": report.pending_messages,
        "estimated_tokens": report.estimated_tokens,
        "estimated_cost_usd": report.estimated_cost_usd,
        "model": report.model,
        "dimension": report.dimension,
        "monthly_cost_cap_usd": report.cost_cap_usd,
        "effective_cost_cap_usd": effective_cost_cap(report.cost_cap_usd, report.max_cost_usd),
        "windowed": report.windowed,
        "max_sessions": report.max_sessions,
        "max_messages": report.max_messages,
        "max_cost_usd": report.max_cost_usd,
        "pricing": {
            "estimated_tokens_per_message": ESTIMATED_TOKENS_PER_MESSAGE,
            "cost_usd_per_1m_tokens": VOYAGE_4_COST_PER_1M_TOKENS,
            "approximate": True,
        },
        "backfill_args": backfill_args,
        "backfill_command": "polylogue " + " ".join(backfill_args) if backfill_args is not None else None,
    }
