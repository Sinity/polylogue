"""Read helper for the ``session_links`` table.

polylogue-4ts.10: this module used to also carry a full async
resolve/cycle-detect/quarantine write engine
(``upsert_session_links``/``resolve_session_links_for_session``/
``resolve_unresolved_links_for_child``/``_would_create_cycle``/
``_quarantine_link``/``count_quarantined_session_links``). A 2026-08-03
structural audit found that engine had zero production callers -- the sole
production writer is ``_resolve_session_graph``/``_resolve_outbound_session_links``
in ``storage/sqlite/archive_tiers/write.py`` (invoked from
``write_parsed_session_to_archive``), which grew its own equivalent
cycle-detection + quarantine implementation (``_would_create_cycle``/
``_quarantine_session_link`` there) in #3643. The dead engine was exercised
only by wrong-oracle tests certifying behavior production code could not
exhibit; those tests were retargeted at the live path and the dead engine
was deleted here rather than kept as an unreachable second implementation.
Only ``list_session_links_for_session`` (a genuine production read path via
``query_store_archive.py``) remains.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import aiosqlite

SESSION_LINK_COLUMNS = """src_session_id, dst_origin, dst_native_id, link_type,
               resolved_dst_session_id, branch_point_message_id, inheritance,
               status, parent_tool_use_block_id, method, confidence,
               evidence_json, observed_at_ms, resolved_at_ms"""
"""The canonical projection column set shared by every topology read.

A sync reader (the archive tier's rooted tree) and the async readers must select
the same columns, or one surface silently loses provenance the other keeps.
"""


def read_session_digest_links(conn: sqlite3.Connection, session_id: str) -> list[dict[str, object]]:
    """Read the original digest's outbound assertions and unlinked children on one pin."""
    with closing(conn.cursor()) as cursor:
        cursor.row_factory = sqlite3.Row
        links = [
            dict(row)
            for row in cursor.execute(
                f"SELECT {SESSION_LINK_COLUMNS} FROM session_links WHERE src_session_id = ? "
                "ORDER BY link_type, dst_origin, dst_native_id",
                (session_id,),
            )
        ]
        links.extend(
            {
                "dst_origin": row["origin"],
                "dst_native_id": row["native_id"],
                "resolved_dst_session_id": row["session_id"],
                "status": "resolved",
                "link_type": row["branch_type"] or "child",
            }
            for row in cursor.execute(
                "SELECT session_id, origin, native_id, branch_type FROM sessions "
                "WHERE parent_session_id = ? ORDER BY sort_key_ms DESC, session_id",
                (session_id,),
            )
        )
        return links


async def list_session_links_for_session(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    bounded = "" if limit is None else " LIMIT ?"
    params: tuple[object, ...] = (session_id,) if limit is None else (session_id, limit)
    cursor = await conn.execute(
        f"""
        SELECT {SESSION_LINK_COLUMNS}
         FROM session_links
         WHERE src_session_id = ?
         ORDER BY link_type, dst_origin, dst_native_id
        """
        + bounded,
        params,
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def list_session_links_to_session(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    limit: int | None,
) -> list[dict[str, object]]:
    """Stable reverse lookup for canonical topology BFS, optionally bounded."""

    cursor = await conn.execute(
        f"""
        SELECT {SESSION_LINK_COLUMNS}
          FROM session_links
         WHERE resolved_dst_session_id = ?
         ORDER BY src_session_id, dst_origin, dst_native_id, link_type

        """
        + (" LIMIT ?" if limit is not None else ""),
        (session_id, limit) if limit is not None else (session_id,),
    )
    return [dict(row) for row in await cursor.fetchall()]


async def list_session_links(conn: aiosqlite.Connection) -> list[dict[str, object]]:
    """Return canonical topology assertions in stable natural-key order."""

    cursor = await conn.execute(
        f"""
        SELECT {SESSION_LINK_COLUMNS}
          FROM session_links
         ORDER BY src_session_id, dst_origin, dst_native_id, link_type
        """
    )
    return [dict(row) for row in await cursor.fetchall()]


__all__ = [
    "read_session_digest_links",
    "list_session_links_for_session",
    "list_session_links",
    "list_session_links_to_session",
    "SESSION_LINK_COLUMNS",
]
