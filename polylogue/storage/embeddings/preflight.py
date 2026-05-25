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

    total_conversations: int
    pending_conversations: int
    pending_messages: int
    estimated_tokens: int
    estimated_cost_usd: float
    model: str
    dimension: int
    cost_cap_usd: float
    windowed: bool = False
    max_conversations: int | None = None
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
    max_conversations: int | None = None,
    max_messages: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(total_convs, pending_convs, pending_messages)``.

    Pending = no ``embedding_status`` row, or ``needs_reindex = 1``.
    Reading happens against a sync read connection so the command works even
    when the daemon is not running.
    """
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    if not db_path.exists():
        return 0, 0, 0

    conn = open_readonly_connection(db_path)
    try:
        total = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
        if max_conversations is not None or max_messages is not None:
            from polylogue.api import select_pending_embedding_conversation_window

            pending = select_pending_embedding_conversation_window(
                conn,
                rebuild=rebuild,
                max_conversations=max_conversations,
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
                    SELECT COUNT(*) FROM conversations c
                    LEFT JOIN embedding_status e
                      ON c.conversation_id = e.conversation_id
                    WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
                    """
                ).fetchone()[0]
            )
            pending_messages = int(
                conn.execute(
                    """
                    SELECT COUNT(*) FROM messages m
                    JOIN conversations c ON c.conversation_id = m.conversation_id
                    LEFT JOIN embedding_status e
                      ON c.conversation_id = e.conversation_id
                    WHERE e.conversation_id IS NULL OR e.needs_reindex = 1
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


def build_preflight_report(
    db_path: Path,
    *,
    rebuild: bool = False,
    max_conversations: int | None = None,
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
        max_conversations=max_conversations,
        max_messages=effective_max_messages,
    )
    estimated_tokens = pending_messages * ESTIMATED_TOKENS_PER_MESSAGE
    estimated_cost = estimated_tokens * VOYAGE_4_COST_PER_1M_TOKENS / 1_000_000
    return PreflightReport(
        total_conversations=total,
        pending_conversations=pending,
        pending_messages=pending_messages,
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_cost,
        model=cfg.embedding_model,
        dimension=cfg.embedding_dimension,
        cost_cap_usd=cfg.embedding_max_cost_usd,
        windowed=max_conversations is not None or max_messages is not None or max_cost_usd is not None,
        max_conversations=max_conversations,
        max_messages=effective_max_messages,
        max_cost_usd=max_cost_usd,
    )


def preflight_backfill_args(report: PreflightReport) -> list[str] | None:
    if report.pending_conversations <= 0:
        return None
    args = ["embed", "backfill", "--yes"]
    if report.max_conversations is not None:
        args.extend(["--max-conversations", str(report.max_conversations)])
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
        "total_conversations": report.total_conversations,
        "pending_conversations": report.pending_conversations,
        "pending_messages": report.pending_messages,
        "estimated_tokens": report.estimated_tokens,
        "estimated_cost_usd": report.estimated_cost_usd,
        "model": report.model,
        "dimension": report.dimension,
        "monthly_cost_cap_usd": report.cost_cap_usd,
        "effective_cost_cap_usd": effective_cost_cap(report.cost_cap_usd, report.max_cost_usd),
        "windowed": report.windowed,
        "max_conversations": report.max_conversations,
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
