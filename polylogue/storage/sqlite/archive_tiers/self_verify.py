"""Archive self-verify helpers for the archive database."""

from __future__ import annotations

import sqlite3
from typing import Any

from polylogue.storage.sqlite.archive_tiers.write import read_archive_session_envelope, search_archive_blocks


def build_archive_session_self_verify_envelope(
    conn: sqlite3.Connection,
    session_id: str,
    query: str,
) -> dict[str, Any]:
    """Build a tiny, stable envelope for archive self-verification."""
    envelope = read_archive_session_envelope(conn, session_id)
    return {
        "session_id": envelope.session_id,
        "origin": envelope.origin,
        "active_leaf_message_id": envelope.active_leaf_message_id,
        "counts": {
            "messages": len(envelope.messages),
            "blocks": sum(len(message.blocks) for message in envelope.messages),
            "session_events": _count(conn, "SELECT COUNT(*) FROM session_events WHERE session_id = ?", session_id),
            "provider_usage_events": _count(
                conn,
                "SELECT COUNT(*) FROM session_provider_usage_events WHERE session_id = ?",
                session_id,
            ),
            "session_links_outbound": _count(
                conn,
                "SELECT COUNT(*) FROM session_links WHERE src_session_id = ?",
                session_id,
            ),
            "attachments": _count(
                conn,
                """
                SELECT COUNT(DISTINCT attachment_id)
                FROM attachment_refs
                WHERE session_id = ?
                """,
                session_id,
            ),
            "attachment_refs": _count(conn, "SELECT COUNT(*) FROM attachment_refs WHERE session_id = ?", session_id),
            "tags": _count(conn, "SELECT COUNT(*) FROM session_tags WHERE session_id = ?", session_id),
            "insight_materializations": _count(
                conn,
                "SELECT COUNT(*) FROM insight_materialization WHERE session_id = ?",
                session_id,
            ),
            "work_events": _count(conn, "SELECT COUNT(*) FROM session_work_events WHERE session_id = ?", session_id),
            "phases": _count(conn, "SELECT COUNT(*) FROM session_phases WHERE session_id = ?", session_id),
        },
        "ordered_message_ids": [message.message_id for message in envelope.messages],
        "ordered_block_ids": [block.block_id for message in envelope.messages for block in message.blocks],
        "session_event_ids": _string_list(
            conn,
            "SELECT event_id FROM session_events WHERE session_id = ? ORDER BY position",
            session_id,
        ),
        "attachment_ids": _string_list(
            conn,
            """
            SELECT DISTINCT attachment_id
            FROM attachment_refs
            WHERE session_id = ?
            ORDER BY attachment_id
            """,
            session_id,
        ),
        "tag_keys": _string_list(
            conn,
            "SELECT tag_source || ':' || tag FROM session_tags WHERE session_id = ? ORDER BY tag_source, tag",
            session_id,
        ),
        "insight_materialization_keys": _string_list(
            conn,
            """
            SELECT insight_type || ':' || materializer_version
            FROM insight_materialization
            WHERE session_id = ?
            ORDER BY insight_type
            """,
            session_id,
        ),
        "work_event_ids": _string_list(
            conn,
            "SELECT event_id FROM session_work_events WHERE session_id = ? ORDER BY position",
            session_id,
        ),
        "phase_ids": _string_list(
            conn,
            "SELECT phase_id FROM session_phases WHERE session_id = ? ORDER BY position",
            session_id,
        ),
        "fts_hits": {
            "query": query,
            "block_ids": search_archive_blocks(conn, query) if query else [],
        },
    }


def _count(conn: sqlite3.Connection, query: str, session_id: str) -> int:
    return int(conn.execute(query, (session_id,)).fetchone()[0])


def _string_list(conn: sqlite3.Connection, query: str, session_id: str) -> list[str]:
    return [str(row[0]) for row in conn.execute(query, (session_id,)).fetchall()]


__all__ = ["build_archive_session_self_verify_envelope"]
