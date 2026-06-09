"""SQL helpers for the current ``session_links`` table."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime

import aiosqlite

from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.topology.edge import TopologyEdgeRecord, TopologyEdgeStatus


def _timestamp_ms(value: str | None) -> int | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _status_value(status: TopologyEdgeStatus) -> str | None:
    if status is TopologyEdgeStatus.QUARANTINED:
        return TopologyEdgeStatus.QUARANTINED.value
    if status is TopologyEdgeStatus.REPAIRED:
        return TopologyEdgeStatus.REPAIRED.value
    return None


async def upsert_session_links(
    conn: aiosqlite.Connection,
    links: Iterable[TopologyEdgeRecord],
) -> int:
    """Upsert session links. Returns number of rows written."""
    written = 0
    for link in links:
        await conn.execute(
            """
            INSERT INTO session_links (
                src_session_id,
                dst_origin,
                dst_native_id,
                link_type,
                resolved_dst_session_id,
                status,
                method,
                confidence,
                evidence_json,
                observed_at_ms,
                resolved_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (src_session_id, dst_origin, dst_native_id, link_type) DO UPDATE SET
                resolved_dst_session_id = COALESCE(
                    excluded.resolved_dst_session_id,
                    session_links.resolved_dst_session_id
                ),
                status = CASE
                    WHEN session_links.status = 'quarantined' THEN session_links.status
                    ELSE excluded.status
                END,
                method = COALESCE(excluded.method, session_links.method),
                confidence = excluded.confidence,
                evidence_json = COALESCE(NULLIF(excluded.evidence_json, '[]'), session_links.evidence_json),
                resolved_at_ms = COALESCE(excluded.resolved_at_ms, session_links.resolved_at_ms)
            """,
            (
                str(link.src_session_id),
                link.dst_origin.value,
                link.dst_native_id,
                str(link.link_type),
                str(link.resolved_dst_session_id) if link.resolved_dst_session_id else None,
                _status_value(link.status),
                "parser-parent",
                link.confidence,
                link.evidence_json,
                _timestamp_ms(link.observed_at) or 0,
                _timestamp_ms(link.resolved_at),
            ),
        )
        written += 1
    return written


_CYCLE_WALK_BUDGET = 1024


async def _would_create_cycle(
    conn: aiosqlite.Connection,
    *,
    child_id: str,
    proposed_parent_id: str,
) -> list[str] | None:
    if proposed_parent_id == child_id:
        return [child_id, child_id]

    path: list[str] = [child_id, proposed_parent_id]
    current = proposed_parent_id
    steps = 0
    while True:
        if steps >= _CYCLE_WALK_BUDGET:
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


async def _quarantine_link(
    conn: aiosqlite.Connection,
    *,
    src_session_id: str,
    dst_origin: str | None,
    dst_native_id: str | None,
    link_type: str | None,
    cycle_path: list[str],
    observed_at_ms: int,
) -> None:
    evidence = json.dumps(
        {
            "reason": "cycle_rejected",
            "cycle_path": cycle_path,
            "detected_at_ms": observed_at_ms,
        },
        sort_keys=True,
    )
    if dst_origin is None or dst_native_id is None or link_type is None:
        await conn.execute(
            """
            UPDATE session_links
               SET status = 'quarantined',
                   evidence_json = ?,
                   resolved_at_ms = ?
             WHERE src_session_id = ?
            """,
            (evidence, observed_at_ms, src_session_id),
        )
        return
    await conn.execute(
        """
        UPDATE session_links
           SET status = 'quarantined',
               evidence_json = ?,
               resolved_at_ms = ?
         WHERE src_session_id = ?
           AND dst_origin = ?
           AND dst_native_id = ?
           AND link_type = ?
        """,
        (evidence, observed_at_ms, src_session_id, dst_origin, dst_native_id, link_type),
    )


async def count_quarantined_session_links(conn: aiosqlite.Connection) -> int:
    row = await (await conn.execute("SELECT COUNT(*) AS n FROM session_links WHERE status = 'quarantined'")).fetchone()
    return int(row["n"]) if row is not None else 0


async def resolve_session_links_for_session(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    origin: str,
    native_id: str,
    resolved_at: str,
) -> int:
    observed_at_ms = _timestamp_ms(resolved_at) or 0
    cursor = await conn.execute(
        """
        SELECT src_session_id, link_type
          FROM session_links
         WHERE resolved_dst_session_id IS NULL
           AND status IS NULL
           AND dst_origin = ?
           AND dst_native_id = ?
        """,
        (origin, native_id),
    )
    pending_rows = list(await cursor.fetchall())
    if not pending_rows:
        return 0

    valid_branch_types: set[str] = {bt.value for bt in BranchType}
    flipped = 0
    for row in pending_rows:
        child_id = row["src_session_id"]
        link_type = row["link_type"]

        cycle_path = await _would_create_cycle(conn, child_id=child_id, proposed_parent_id=session_id)
        if cycle_path is not None:
            await _quarantine_link(
                conn,
                src_session_id=child_id,
                dst_origin=origin,
                dst_native_id=native_id,
                link_type=link_type,
                cycle_path=cycle_path,
                observed_at_ms=observed_at_ms,
            )
            continue

        await conn.execute(
            """
            UPDATE session_links
               SET resolved_dst_session_id = ?,
                   resolved_at_ms = ?
             WHERE resolved_dst_session_id IS NULL
               AND status IS NULL
               AND src_session_id = ?
               AND dst_origin = ?
               AND dst_native_id = ?
               AND link_type = ?
            """,
            (session_id, observed_at_ms, child_id, origin, native_id, link_type),
        )
        flipped += 1
        branch_type: str | None = link_type if link_type in valid_branch_types else None
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


async def resolve_unresolved_links_for_child(
    conn: aiosqlite.Connection,
    *,
    src_session_id: str,
    resolved_at: str,
) -> int:
    observed_at_ms = _timestamp_ms(resolved_at) or 0
    cursor = await conn.execute(
        """
        SELECT link.link_type, link.dst_origin, link.dst_native_id, dst.session_id AS parent_id
          FROM session_links AS link
          JOIN sessions AS dst
            ON dst.origin = link.dst_origin
           AND dst.native_id = link.dst_native_id
         WHERE link.src_session_id = ?
           AND link.resolved_dst_session_id IS NULL
           AND link.status IS NULL
        """,
        (src_session_id,),
    )
    rows = list(await cursor.fetchall())
    if not rows:
        return 0

    valid_branch_types: set[str] = {bt.value for bt in BranchType}
    resolved = 0
    for row in rows:
        link_type = row["link_type"]
        parent_id = row["parent_id"]

        cycle_path = await _would_create_cycle(conn, child_id=src_session_id, proposed_parent_id=parent_id)
        if cycle_path is not None:
            await _quarantine_link(
                conn,
                src_session_id=src_session_id,
                dst_origin=row["dst_origin"],
                dst_native_id=row["dst_native_id"],
                link_type=link_type,
                cycle_path=cycle_path,
                observed_at_ms=observed_at_ms,
            )
            continue

        await conn.execute(
            """
            UPDATE session_links
               SET resolved_dst_session_id = ?,
                   resolved_at_ms = ?
             WHERE src_session_id = ?
               AND dst_origin = ?
               AND dst_native_id = ?
               AND link_type = ?
               AND resolved_dst_session_id IS NULL
               AND status IS NULL
            """,
            (parent_id, observed_at_ms, src_session_id, row["dst_origin"], row["dst_native_id"], link_type),
        )
        branch_type: str | None = link_type if link_type in valid_branch_types else None
        await conn.execute(
            """
            UPDATE sessions
               SET parent_session_id = COALESCE(parent_session_id, ?),
                   branch_type = COALESCE(branch_type, ?)
             WHERE session_id = ?
            """,
            (parent_id, branch_type, src_session_id),
        )
        resolved += 1
    return resolved


async def list_session_links_for_session(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[dict[str, object]]:
    cursor = await conn.execute(
        """
        SELECT src_session_id, dst_origin, dst_native_id, link_type,
               resolved_dst_session_id, status, method, confidence,
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
    "TopologyEdgeStatus",
    "count_quarantined_session_links",
    "list_session_links_for_session",
    "resolve_session_links_for_session",
    "resolve_unresolved_links_for_child",
    "upsert_session_links",
]
