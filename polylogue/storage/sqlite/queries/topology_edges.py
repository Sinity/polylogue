"""SQL helpers for the ``topology_edges`` table (#1258 / #866 slice A).

Two operations are exposed:

- :func:`upsert_topology_edges` — write ingest-time edges. Idempotent on
  ``(src_conversation_id, dst_provider_native_id, edge_type)``. Never demotes
  ``resolved`` → ``unresolved`` if a later ingest carries less information
  than the prior one.
- :func:`resolve_topology_edges_for_conversation` — flip pending
  ``unresolved`` rows pointing at ``(provider_name, provider_conversation_id)``
  to ``resolved`` once their parent has been ingested. Used by
  ``save_via_backend`` so out-of-order ingest backfills the resolver state
  the moment the parent lands.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import aiosqlite

from polylogue.archive.topology.edge import TopologyEdgeRecord, TopologyEdgeStatus


def _make_edge_id(src: str, dst_provider_native_id: str, edge_type: str) -> str:
    seed = f"{src}|{dst_provider_native_id}|{edge_type}"
    return f"edge-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


async def upsert_topology_edges(
    conn: aiosqlite.Connection,
    edges: Iterable[TopologyEdgeRecord],
) -> int:
    """Upsert topology edges. Returns number of rows written."""
    written = 0
    for edge in edges:
        edge_id = _make_edge_id(
            str(edge.src_conversation_id),
            edge.dst_provider_native_id,
            str(edge.edge_type),
        )
        await conn.execute(
            """
            INSERT INTO topology_edges (
                edge_id,
                src_conversation_id,
                dst_provider_native_id,
                dst_provider_name,
                edge_type,
                resolved_dst_conversation_id,
                raw_evidence,
                confidence,
                status,
                observed_at,
                resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (src_conversation_id, dst_provider_native_id, edge_type) DO UPDATE SET
                dst_provider_name = excluded.dst_provider_name,
                resolved_dst_conversation_id = COALESCE(
                    excluded.resolved_dst_conversation_id,
                    topology_edges.resolved_dst_conversation_id
                ),
                raw_evidence = COALESCE(excluded.raw_evidence, topology_edges.raw_evidence),
                confidence = excluded.confidence,
                status = CASE
                    -- never demote resolved → unresolved
                    WHEN topology_edges.status = 'resolved' AND excluded.status = 'unresolved'
                        THEN topology_edges.status
                    ELSE excluded.status
                END,
                resolved_at = COALESCE(excluded.resolved_at, topology_edges.resolved_at)
            """,
            (
                edge_id,
                str(edge.src_conversation_id),
                edge.dst_provider_native_id,
                edge.dst_provider_name,
                str(edge.edge_type),
                str(edge.resolved_dst_conversation_id) if edge.resolved_dst_conversation_id else None,
                edge.raw_evidence,
                edge.confidence,
                str(edge.status),
                edge.observed_at,
                edge.resolved_at,
            ),
        )
        written += 1
    return written


async def resolve_topology_edges_for_conversation(
    conn: aiosqlite.Connection,
    *,
    conversation_id: str,
    provider_name: str,
    provider_conversation_id: str,
    resolved_at: str,
) -> int:
    """Flip unresolved edges whose parent is the just-saved conversation.

    Returns the number of rows updated. Used by the conversation save path
    to backfill resolver state for out-of-order ingest: when a child is
    ingested before its parent, its edge is written ``unresolved``; this
    helper runs after the parent's row hits ``conversations`` and flips
    every dangling edge that was waiting for that parent's native id.
    """
    cursor = await conn.execute(
        """
        UPDATE topology_edges
           SET resolved_dst_conversation_id = ?,
               status = 'resolved',
               resolved_at = ?
         WHERE status = 'unresolved'
           AND dst_provider_name = ?
           AND dst_provider_native_id = ?
        """,
        (conversation_id, resolved_at, provider_name, provider_conversation_id),
    )
    return cursor.rowcount or 0


async def list_topology_edges_for_conversation(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[dict[str, object]]:
    """Read every outbound edge for a conversation. Used by tests."""
    cursor = await conn.execute(
        """
        SELECT edge_id, src_conversation_id, dst_provider_native_id, dst_provider_name,
               edge_type, resolved_dst_conversation_id, raw_evidence, confidence,
               status, observed_at, resolved_at
          FROM topology_edges
         WHERE src_conversation_id = ?
         ORDER BY edge_type, dst_provider_native_id
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


__all__ = [
    "TopologyEdgeStatus",
    "list_topology_edges_for_conversation",
    "resolve_topology_edges_for_conversation",
    "upsert_topology_edges",
]
