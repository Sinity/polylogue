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

from polylogue.archive.conversation.branch_type import BranchType
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

    Slice B (#1259 / #866) — also backfills the resolved children's
    ``conversations.parent_conversation_id`` (and ``branch_type``, where the
    edge type maps to a valid ``BranchType``) so the fast-path ancestry
    walk used by ``derive_session_topology_async``, ``thread_root_id`` and
    other downstream readers benefits from late-parent arrival without
    requiring the child to be re-ingested. The backfill is conservative:
    it only writes ``parent_conversation_id`` when the column is NULL and
    only writes ``branch_type`` when the column is NULL and the edge type
    is a member of ``BranchType``. This keeps the operation idempotent —
    running the resolver twice for the same parent produces no further
    changes after the first pass.
    """
    # Collect the children that the upcoming UPDATE will flip, *before*
    # the UPDATE runs, so we can use them to backfill the conversations
    # rows in the same transaction. Selecting child IDs + edge types lets
    # us avoid issuing one UPDATE per child while still classifying the
    # branch type per row.
    cursor = await conn.execute(
        """
        SELECT src_conversation_id, edge_type
          FROM topology_edges
         WHERE status = 'unresolved'
           AND dst_provider_name = ?
           AND dst_provider_native_id = ?
        """,
        (provider_name, provider_conversation_id),
    )
    pending_rows = list(await cursor.fetchall())

    update_cursor = await conn.execute(
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
    flipped = update_cursor.rowcount or 0

    if flipped == 0 or not pending_rows:
        return flipped

    # Backfill the resolved fast-path on each child conversation. The
    # ``parent_conversation_id IS NULL`` guard preserves whatever was set
    # at the child's original write time and makes the backfill idempotent;
    # the ``branch_type`` guard mirrors the same constraint and restricts
    # writes to the closed ``BranchType`` vocabulary so the CHECK constraint
    # on ``conversations.branch_type`` cannot be violated by future
    # ``RESUME`` / ``BRANCH`` / ``REPAIRED`` edge types.
    valid_branch_types: set[str] = {bt.value for bt in BranchType}
    for row in pending_rows:
        child_id = row["src_conversation_id"]
        edge_type = row["edge_type"]
        branch_type: str | None = edge_type if edge_type in valid_branch_types else None
        await conn.execute(
            """
            UPDATE conversations
               SET parent_conversation_id = COALESCE(parent_conversation_id, ?),
                   branch_type = COALESCE(branch_type, ?)
             WHERE conversation_id = ?
            """,
            (conversation_id, branch_type, child_id),
        )

    return flipped


async def resolve_unresolved_edges_for_child(
    conn: aiosqlite.Connection,
    *,
    src_conversation_id: str,
    resolved_at: str,
) -> int:
    """Resolve any of ``src_conversation_id``'s own unresolved edges whose
    parent is already in ``conversations``.

    Closes the concurrent-ingest race where the parent's
    :func:`resolve_topology_edges_for_conversation` pass ran before the
    child's edge was upserted. After the child's edge lands, this helper
    sweeps every unresolved edge it just wrote, joins against
    ``conversations`` on ``(provider_name, provider_conversation_id)``,
    and flips the edge if a matching parent row now exists.

    Backfills the same ``conversations.parent_conversation_id`` /
    ``branch_type`` fast-path columns as
    :func:`resolve_topology_edges_for_conversation`, with the same
    conservative ``COALESCE`` guards so the operation is idempotent.
    """
    cursor = await conn.execute(
        """
        SELECT te.edge_id, te.edge_type, c.conversation_id AS parent_cid
          FROM topology_edges AS te
          JOIN conversations  AS c
            ON c.provider_name = te.dst_provider_name
           AND c.provider_conversation_id = te.dst_provider_native_id
         WHERE te.src_conversation_id = ?
           AND te.status = 'unresolved'
        """,
        (src_conversation_id,),
    )
    rows = list(await cursor.fetchall())
    if not rows:
        return 0

    valid_branch_types: set[str] = {bt.value for bt in BranchType}
    resolved = 0
    for row in rows:
        edge_id = row["edge_id"]
        edge_type = row["edge_type"]
        parent_cid = row["parent_cid"]
        await conn.execute(
            """
            UPDATE topology_edges
               SET resolved_dst_conversation_id = ?,
                   status = 'resolved',
                   resolved_at = ?
             WHERE edge_id = ?
               AND status = 'unresolved'
            """,
            (parent_cid, resolved_at, edge_id),
        )
        branch_type: str | None = edge_type if edge_type in valid_branch_types else None
        await conn.execute(
            """
            UPDATE conversations
               SET parent_conversation_id = COALESCE(parent_conversation_id, ?),
                   branch_type = COALESCE(branch_type, ?)
             WHERE conversation_id = ?
            """,
            (parent_cid, branch_type, src_conversation_id),
        )
        resolved += 1
    return resolved


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
    "resolve_unresolved_edges_for_child",
    "upsert_topology_edges",
]
