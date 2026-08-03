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

import aiosqlite


async def list_session_links_for_session(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[dict[str, object]]:
    cursor = await conn.execute(
        """
        SELECT src_session_id, dst_origin, dst_native_id, link_type,
               resolved_dst_session_id, status, parent_tool_use_block_id, method, confidence,
               evidence_json, observed_at_ms, resolved_at_ms
          FROM session_links
         WHERE src_session_id = ?
         ORDER BY link_type, dst_origin, dst_native_id
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


__all__ = [
    "list_session_links_for_session",
]
