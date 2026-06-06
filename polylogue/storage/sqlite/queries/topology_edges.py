"""SQL helpers for the ``topology_edges`` table (#1258 / #866 slice A).

Two operations are exposed:

- :func:`upsert_topology_edges` — write ingest-time edges. Idempotent on
  ``(src_session_id, dst_provider_native_id, edge_type)``. Never demotes
  ``resolved`` → ``unresolved`` if a later ingest carries less information
  than the prior one.
- :func:`resolve_topology_edges_for_session` — flip pending
  ``unresolved`` rows pointing at ``(source_name, provider_session_id)``
  to ``resolved`` once their parent has been ingested. Used by
  ``save_via_backend`` so out-of-order ingest backfills the resolver state
  the moment the parent lands.

Cycle quarantine (#1260 / #866 slice C)
---------------------------------------
Both resolver paths refuse to backfill a child's
``sessions.parent_session_id`` when doing so would introduce a
cycle into the resolved-parent graph (A→B→A, 3-node, etc., including
the self-cycle A→A). The offending edge is left in the
``topology_edges`` table with ``status='quarantined'`` and a
``raw_evidence`` JSON document recording the detected cycle path so the
operator can audit which session chain was rejected. The child's
``parent_session_id`` is NOT written for quarantined edges — the
fast-path ancestry walk therefore continues to terminate at the child
rather than enter the cycle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import aiosqlite

from polylogue.archive.session.branch_type import BranchType
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
            str(edge.src_session_id),
            edge.dst_provider_native_id,
            str(edge.edge_type),
        )
        await conn.execute(
            """
            INSERT INTO topology_edges (
                edge_id,
                src_session_id,
                dst_provider_native_id,
                dst_provider_name,
                edge_type,
                resolved_dst_session_id,
                raw_evidence,
                confidence,
                status,
                observed_at,
                resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (src_session_id, dst_provider_native_id, edge_type) DO UPDATE SET
                dst_provider_name = excluded.dst_provider_name,
                resolved_dst_session_id = COALESCE(
                    excluded.resolved_dst_session_id,
                    topology_edges.resolved_dst_session_id
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
                str(edge.src_session_id),
                edge.dst_provider_native_id,
                edge.dst_provider_name,
                str(edge.edge_type),
                str(edge.resolved_dst_session_id) if edge.resolved_dst_session_id else None,
                edge.raw_evidence,
                edge.confidence,
                str(edge.status),
                edge.observed_at,
                edge.resolved_at,
            ),
        )
        written += 1
    return written


_CYCLE_WALK_BUDGET = 1024
"""Hard cap on the resolved-parent ancestry walk used by cycle detection.

Any session chain longer than this is treated as cyclic regardless
of whether the actual cycle was reached, because no legitimate
session lineage is expected to exceed it and a runaway walk would
otherwise pin the resolver."""


async def _would_create_cycle(
    conn: aiosqlite.Connection,
    *,
    child_id: str,
    proposed_parent_id: str,
) -> list[str] | None:
    """Return the cycle path if making ``proposed_parent_id`` the parent
    of ``child_id`` would create a cycle in ``sessions.parent_session_id``.

    A cycle exists when ``proposed_parent_id == child_id`` (self-cycle) or
    when walking ``proposed_parent_id``'s existing ancestry via
    ``parent_session_id`` reaches ``child_id``. Returns ``None`` if no
    cycle would form. The returned list is the chain
    ``[child_id, proposed_parent_id, ..., child_id]`` so the operator can
    see exactly which edge would have closed the loop.
    """

    # Self-cycle: child claiming itself as parent.
    if proposed_parent_id == child_id:
        return [child_id, child_id]

    path: list[str] = [child_id, proposed_parent_id]
    current = proposed_parent_id
    steps = 0
    while True:
        if steps >= _CYCLE_WALK_BUDGET:
            # Treat budget-exceeded as cyclic; legitimate chains do not
            # reach this length and continuing risks a runaway walk.
            path.append("...budget-exceeded")
            return path
        row = await (
            await conn.execute(
                "SELECT parent_session_id FROM sessions WHERE session_id = ?",
                (current,),
            )
        ).fetchone()
        if row is None:
            return None
        next_parent = row["parent_session_id"]
        if next_parent is None:
            return None
        if next_parent == child_id:
            path.append(child_id)
            return path
        path.append(next_parent)
        current = next_parent
        steps += 1


async def _quarantine_edge(
    conn: aiosqlite.Connection,
    *,
    edge_id: str | None,
    src_session_id: str,
    dst_provider_name: str | None,
    dst_provider_native_id: str | None,
    cycle_path: list[str],
    observed_at: str,
) -> None:
    """Mark an edge as quarantined and record the cycle path in raw_evidence.

    If ``edge_id`` is given the update targets that row directly; otherwise
    the ``(dst_provider_name, dst_provider_native_id)`` lookup is used,
    matching the resolver-by-parent code path.
    """

    evidence = json.dumps(
        {
            "reason": "cycle_rejected",
            "cycle_path": cycle_path,
            "detected_at": observed_at,
        },
        sort_keys=True,
    )
    if edge_id is not None:
        await conn.execute(
            """
            UPDATE topology_edges
               SET status = 'quarantined',
                   raw_evidence = ?,
                   resolved_at = ?
             WHERE edge_id = ?
            """,
            (evidence, observed_at, edge_id),
        )
        return
    await conn.execute(
        """
        UPDATE topology_edges
           SET status = 'quarantined',
               raw_evidence = ?,
               resolved_at = ?
         WHERE src_session_id = ?
           AND dst_provider_name = ?
           AND dst_provider_native_id = ?
        """,
        (
            evidence,
            observed_at,
            src_session_id,
            dst_provider_name,
            dst_provider_native_id,
        ),
    )


async def count_quarantined_topology_edges(conn: aiosqlite.Connection) -> int:
    """Return the number of topology edges currently in ``quarantined`` state.

    Surfaced through the diagnostic/readiness report (#1260) so cycle
    rejections are visible to the operator rather than silently absorbed.
    """

    row = await (await conn.execute("SELECT COUNT(*) AS n FROM topology_edges WHERE status = 'quarantined'")).fetchone()
    return int(row["n"]) if row is not None else 0


async def resolve_topology_edges_for_session(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    source_name: str,
    provider_session_id: str,
    resolved_at: str,
) -> int:
    """Flip unresolved edges whose parent is the just-saved session.

    Returns the number of rows updated. Used by the session save path
    to backfill resolver state for out-of-order ingest: when a child is
    ingested before its parent, its edge is written ``unresolved``; this
    helper runs after the parent's row hits ``sessions`` and flips
    every dangling edge that was waiting for that parent's native id.

    Slice B (#1259 / #866) — also backfills the resolved children's
    ``sessions.parent_session_id`` (and ``branch_type``, where the
    edge type maps to a valid ``BranchType``) so the fast-path ancestry
    walk used by ``derive_session_topology_async``, ``thread_root_id`` and
    other downstream readers benefits from late-parent arrival without
    requiring the child to be re-ingested. The backfill is conservative:
    it only writes ``parent_session_id`` when the column is NULL and
    only writes ``branch_type`` when the column is NULL and the edge type
    is a member of ``BranchType``. This keeps the operation idempotent —
    running the resolver twice for the same parent produces no further
    changes after the first pass.
    """
    # Collect the children that would be flipped, *before* any UPDATE runs,
    # so we can partition them into (resolved, quarantined) buckets based
    # on whether backfilling their ``parent_session_id`` would create
    # a cycle (#1260). Per-row processing lets the cycle detector walk the
    # ancestry of ``session_id`` (the proposed parent) for each child.
    cursor = await conn.execute(
        """
        SELECT src_session_id, edge_type
          FROM topology_edges
         WHERE status = 'unresolved'
           AND dst_provider_name = ?
           AND dst_provider_native_id = ?
        """,
        (source_name, provider_session_id),
    )
    pending_rows = list(await cursor.fetchall())

    if not pending_rows:
        return 0

    valid_branch_types: set[str] = {bt.value for bt in BranchType}
    flipped = 0
    for row in pending_rows:
        child_id = row["src_session_id"]
        edge_type = row["edge_type"]

        cycle_path = await _would_create_cycle(
            conn,
            child_id=child_id,
            proposed_parent_id=session_id,
        )
        if cycle_path is not None:
            # Quarantine the offending edge instead of resolving it. Leave
            # ``sessions.parent_session_id`` untouched so the
            # fast-path ancestry walk does not enter the cycle.
            await _quarantine_edge(
                conn,
                edge_id=None,
                src_session_id=child_id,
                dst_provider_name=source_name,
                dst_provider_native_id=provider_session_id,
                cycle_path=cycle_path,
                observed_at=resolved_at,
            )
            continue

        await conn.execute(
            """
            UPDATE topology_edges
               SET resolved_dst_session_id = ?,
                   status = 'resolved',
                   resolved_at = ?
             WHERE status = 'unresolved'
               AND src_session_id = ?
               AND dst_provider_name = ?
               AND dst_provider_native_id = ?
            """,
            (session_id, resolved_at, child_id, source_name, provider_session_id),
        )
        flipped += 1
        # Backfill the resolved fast-path on the child session. The
        # ``parent_session_id IS NULL`` guard preserves whatever was set
        # at the child's original write time and makes the backfill idempotent;
        # the ``branch_type`` guard mirrors the same constraint and restricts
        # writes to the closed ``BranchType`` vocabulary so the CHECK constraint
        # on ``sessions.branch_type`` cannot be violated by future
        # ``RESUME`` / ``BRANCH`` / ``REPAIRED`` edge types.
        branch_type: str | None = edge_type if edge_type in valid_branch_types else None
        await conn.execute(
            """
            UPDATE sessions
               SET parent_session_id = COALESCE(parent_session_id, ?),
                   branch_type = COALESCE(branch_type, ?)
             WHERE session_id = ?
            """,
            (session_id, branch_type, child_id),
        )

    return flipped


async def resolve_unresolved_edges_for_child(
    conn: aiosqlite.Connection,
    *,
    src_session_id: str,
    resolved_at: str,
) -> int:
    """Resolve any of ``src_session_id``'s own unresolved edges whose
    parent is already in ``sessions``.

    Closes the concurrent-ingest race where the parent's
    :func:`resolve_topology_edges_for_session` pass ran before the
    child's edge was upserted. After the child's edge lands, this helper
    sweeps every unresolved edge it just wrote, joins against
    ``sessions`` on ``(source_name, provider_session_id)``,
    and flips the edge if a matching parent row now exists.

    Backfills the same ``sessions.parent_session_id`` /
    ``branch_type`` fast-path columns as
    :func:`resolve_topology_edges_for_session`, with the same
    conservative ``COALESCE`` guards so the operation is idempotent.
    """
    cursor = await conn.execute(
        """
        SELECT te.edge_id, te.edge_type, c.session_id AS parent_cid
          FROM topology_edges AS te
          JOIN sessions  AS c
            ON c.source_name = te.dst_provider_name
           AND c.provider_session_id = te.dst_provider_native_id
         WHERE te.src_session_id = ?
           AND te.status = 'unresolved'
        """,
        (src_session_id,),
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

        cycle_path = await _would_create_cycle(
            conn,
            child_id=src_session_id,
            proposed_parent_id=parent_cid,
        )
        if cycle_path is not None:
            # Quarantine the offending edge (#1260). Do not flip to resolved
            # and do not backfill the sessions row, so the fast-path
            # ancestry walk does not enter the cycle.
            await _quarantine_edge(
                conn,
                edge_id=edge_id,
                src_session_id=src_session_id,
                dst_provider_name=None,
                dst_provider_native_id=None,
                cycle_path=cycle_path,
                observed_at=resolved_at,
            )
            continue

        await conn.execute(
            """
            UPDATE topology_edges
               SET resolved_dst_session_id = ?,
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
            UPDATE sessions
               SET parent_session_id = COALESCE(parent_session_id, ?),
                   branch_type = COALESCE(branch_type, ?)
             WHERE session_id = ?
            """,
            (parent_cid, branch_type, src_session_id),
        )
        resolved += 1
    return resolved


async def list_topology_edges_for_session(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[dict[str, object]]:
    """Read every outbound edge for a session. Used by tests."""
    cursor = await conn.execute(
        """
        SELECT edge_id, src_session_id, dst_provider_native_id, dst_provider_name,
               edge_type, resolved_dst_session_id, raw_evidence, confidence,
               status, observed_at, resolved_at
          FROM topology_edges
         WHERE src_session_id = ?
         ORDER BY edge_type, dst_provider_native_id
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


__all__ = [
    "TopologyEdgeStatus",
    "count_quarantined_topology_edges",
    "list_topology_edges_for_session",
    "resolve_topology_edges_for_session",
    "resolve_unresolved_edges_for_child",
    "upsert_topology_edges",
]
