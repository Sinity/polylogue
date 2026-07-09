"""Read queries for messages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import aiosqlite

from polylogue.archive.message.roles import MessageRoleFilter, message_role_sql_values
from polylogue.archive.message.types import validate_message_type_filter
from polylogue.core.enums import MaterialOrigin
from polylogue.storage.runtime import MessageRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_message

MessageTypeName = Literal["message", "summary", "tool_use", "tool_result", "thinking", "context", "protocol"]
MaterialOriginFilter = MaterialOrigin | str | tuple[MaterialOrigin | str, ...] | list[MaterialOrigin | str]

_MESSAGE_RECORD_SELECT = """
    m.message_id,
    m.session_id,
    m.native_id AS provider_message_id,
    m.role,
    COALESCE((
        SELECT group_concat(b.text, char(10))
        FROM blocks b
        WHERE b.message_id = m.message_id
          AND b.text IS NOT NULL
    ), '') AS text,
    m.occurred_at_ms / 1000.0 AS sort_key,
    lower(hex(m.content_hash)) AS content_hash,
    1 AS version,
    m.parent_message_id,
    m.variant_index AS branch_index,
    s.origin AS source_name,
    m.word_count,
    m.has_tool_use,
    m.has_thinking,
    m.has_paste,
    m.paste_boundary AS paste_boundary_state,
    m.message_type,
    m.material_origin,
    m.model_name,
    m.input_tokens,
    m.output_tokens,
    m.cache_read_tokens,
    m.cache_write_tokens
"""


async def _resolve_session_id(conn: aiosqlite.Connection, session_id: str) -> str:
    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? OR native_id = ? LIMIT 1", (session_id, session_id)
    )
    row = await cursor.fetchone()
    return str(row["session_id"]) if row is not None else session_id


# Cycle/runaway guard only. Composition is iterative (not recursive), so this is
# NOT a Python-stack limit — a deep acompact/fork chain composes fine. Kept large
# so realistic lineages never truncate; a `visited` set is the real cycle guard.
_MAX_LINEAGE_DEPTH = 1024


async def _prefix_sharing_edge(conn: aiosqlite.Connection, session_id: str) -> tuple[str, str] | None:
    """Return ``(parent_session_id, branch_point_message_id)`` if this session
    inherits a parent's leading prefix (fork / resume / spawned subagent /
    auto-compaction copy), else ``None``. See the lineage model (#2467)."""
    cursor = await conn.execute(
        """
        SELECT resolved_dst_session_id, branch_point_message_id
        FROM session_links
        WHERE src_session_id = ?
          AND inheritance = 'prefix-sharing'
          AND resolved_dst_session_id IS NOT NULL
          AND branch_point_message_id IS NOT NULL
        LIMIT 1
        """,
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return (str(row["resolved_dst_session_id"]), str(row["branch_point_message_id"]))


async def _own_messages(conn: aiosqlite.Connection, session_id: str, *, position_order: bool) -> list[MessageRecord]:
    # A non-lineage session keeps the historical sort-key ordering, which the
    # keyset streamer (iter_messages) mirrors. Lineage composition needs strict
    # position order so the parent prefix is cut at the right message regardless
    # of timestamp gaps or sort-key ties.
    order_by = (
        "m.position, m.variant_index"
        if position_order
        else "(m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id"
    )
    cursor = await conn.execute(
        f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE m.session_id = ?
        ORDER BY {order_by}
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_message(row) for row in rows]


async def get_messages(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    _compose_in_position_order: bool = False,
) -> list[MessageRecord]:
    """Compose a session's full message transcript, holding one read snapshot.

    Composition issues multiple autocommit SELECTs while walking the lineage
    chain (edge reads, then a read per ancestor/descendant). Without a held
    transaction, a concurrent parent re-ingest between those reads can yield a
    torn transcript (4ts.4). If ``conn`` is not already inside a transaction
    (e.g. a caller-held write transaction), this wraps the whole composition
    in one deferred read transaction so every SELECT sees the same snapshot.
    """
    if not conn.in_transaction:
        await conn.execute("BEGIN DEFERRED")
        try:
            return await get_messages(conn, session_id, _compose_in_position_order=_compose_in_position_order)
        finally:
            await conn.execute("ROLLBACK")
    session_id = await _resolve_session_id(conn, session_id)

    # Lineage composition (#2467): a prefix-sharing child stores only its own
    # divergent tail. Walk UP the parent chain collecting (child, branch_point)
    # links to the root, then compose DOWN. This is ITERATIVE (not recursive) so
    # deep acompact/fork chains cannot hit Python's recursion limit; the composed
    # view is position-ordered end to end, while a plain session keeps the
    # sort-key order the keyset streamer relies on. A `visited` set stops a cyclic
    # session_link; _MAX_LINEAGE_DEPTH is only a runaway backstop.
    chain: list[tuple[str, str]] = []  # (child_session_id, branch_point_message_id), leaf-first
    visited: set[str] = {session_id}
    cursor_session = session_id
    for _ in range(_MAX_LINEAGE_DEPTH):
        edge = await _prefix_sharing_edge(conn, cursor_session)
        if edge is None:
            break
        parent_session_id, branch_point_message_id = edge
        parent_session_id = await _resolve_session_id(conn, parent_session_id)
        if parent_session_id in visited:  # cyclic lineage: stop and compose what we have
            break
        chain.append((cursor_session, branch_point_message_id))
        visited.add(parent_session_id)
        cursor_session = parent_session_id

    if not chain:
        return await _own_messages(conn, session_id, position_order=_compose_in_position_order)

    # Compose from the root down: root's full transcript, then splice each
    # descendant's own tail at its branch point in the running composed view.
    composed = await _own_messages(conn, cursor_session, position_order=True)
    for child_session_id, branch_point_message_id in reversed(chain):
        own = await _own_messages(conn, child_session_id, position_order=True)
        prefix: list[MessageRecord] = []
        found = False
        for record in composed:
            prefix.append(record)
            if record.message_id == branch_point_message_id:
                found = True
                break
        # Dangling branch point (e.g. the parent message was hard-deleted): return
        # this child's own tail rather than an over-long transcript (#2467 audit).
        composed = prefix + own if found else own
    return composed


def _filter_composed(
    records: list[MessageRecord],
    *,
    message_role: MessageRoleFilter = (),
    message_type: MessageTypeName | None = None,
    material_origin: MaterialOriginFilter | None = None,
    sort_key_since: float | None = None,
    sort_key_until: float | None = None,
) -> list[MessageRecord]:
    """Apply the SQL-level read filters in Python over an already-composed
    lineage transcript.

    Composition (#2467) spans two sessions, so these filters cannot be pushed
    into the per-session SQL. Forks are a minority of sessions, so filtering the
    composed list in memory keeps the common (non-fork) read paths on their fast
    SQL/keyset queries while making fork reads return the full logical transcript
    instead of a tail-only truncation (#2470). The predicates mirror the SQL:
    a NULL ``sort_key`` is excluded whenever a ``since``/``until`` bound is set,
    exactly as ``m.occurred_at_ms >= ?`` drops NULL rows.
    """
    role_set = set(message_role) if message_role else None
    type_match = validate_message_type_filter(message_type) if message_type else None
    material_origin_set = set(_material_origin_values(material_origin))
    out: list[MessageRecord] = []
    for record in records:
        if role_set is not None and record.role not in role_set:
            continue
        if type_match is not None and record.message_type != type_match:
            continue
        if material_origin_set and record.material_origin.value not in material_origin_set:
            continue
        if sort_key_since is not None and (record.sort_key is None or record.sort_key < sort_key_since):
            continue
        if sort_key_until is not None and (record.sort_key is None or record.sort_key > sort_key_until):
            continue
        out.append(record)
    return out


def _material_origin_values(values: MaterialOriginFilter | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (MaterialOrigin, str)):
        raw_values: tuple[MaterialOrigin | str, ...] = (values,)
    else:
        raw_values = tuple(values)
    return tuple(MaterialOrigin.validate_filter_token(value).value for value in raw_values)


async def get_messages_batch(
    conn: aiosqlite.Connection,
    session_ids: list[str],
    *,
    sort_key_since: float | None = None,
    sort_key_until: float | None = None,
    message_role: MessageRoleFilter = (),
) -> tuple[dict[str, list[MessageRecord]], list[MessageRecord]]:
    if not session_ids:
        return {}, []
    resolved_pairs = [(session_id, await _resolve_session_id(conn, session_id)) for session_id in session_ids]
    resolved_ids = [resolved for _requested, resolved in resolved_pairs]

    result: dict[str, list[MessageRecord]] = {cid: [] for cid in session_ids}
    result.update({cid: [] for cid in resolved_ids})
    all_messages: list[MessageRecord] = []

    # Prefix-sharing children store only their divergent tail; the plain SQL
    # IN-query below would return that truncated tail. Compose them per-session
    # instead so a batched fork read carries its full logical transcript (#2470).
    fork_ids = {rid for rid in dict.fromkeys(resolved_ids) if await _prefix_sharing_edge(conn, rid) is not None}
    sql_ids = [rid for rid in resolved_ids if rid not in fork_ids]

    if sql_ids:
        placeholders = ",".join("?" for _ in sql_ids)
        query = f"""
            SELECT {_MESSAGE_RECORD_SELECT}
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE m.session_id IN ({placeholders})
        """
        params: list[str | float] = list(sql_ids)

        role_values = message_role_sql_values(message_role)
        if role_values:
            role_placeholders = ",".join("?" for _ in role_values)
            query += f" AND m.role IN ({role_placeholders})"
            params.extend(role_values)

        if sort_key_since is not None:
            query += " AND m.occurred_at_ms >= ?"
            params.append(sort_key_since * 1000.0)

        if sort_key_until is not None:
            query += " AND m.occurred_at_ms <= ?"
            params.append(sort_key_until * 1000.0)

        query += " ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id"
        cursor = await conn.execute(
            query,
            tuple(params),
        )
        rows = await cursor.fetchall()

        for row in rows:
            cid = row["session_id"]
            msg = _row_to_message(row)
            if cid in result:
                result[cid].append(msg)
            for requested, resolved in resolved_pairs:
                if requested != cid and resolved == cid and requested in result:
                    result[requested].append(msg)
            all_messages.append(msg)

    for fork_id in fork_ids:
        composed = _filter_composed(
            await get_messages(conn, fork_id),
            message_role=message_role,
            sort_key_since=sort_key_since,
            sort_key_until=sort_key_until,
        )
        for requested, resolved in resolved_pairs:
            if resolved == fork_id and requested in result:
                result[requested] = list(composed)
        if fork_id in result:
            result[fork_id] = list(composed)
        all_messages.extend(composed)

    return result, all_messages


async def get_messages_paginated(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    message_role: MessageRoleFilter = (),
    message_type: MessageTypeName | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[MessageRecord], int]:
    """Return paginated messages for a session with optional filters.

    Returns (messages, total_count) where total_count is the unfiltered
    count of messages matching the SQL-level filters (before limit/offset).
    """
    session_id = await _resolve_session_id(conn, session_id)

    # A prefix-sharing child stores only its divergent tail, so paginating its
    # own ``messages`` rows returns a truncated transcript. Compose the full
    # lineage view, filter in Python, and slice it for offset/limit (#2470).
    # ``total`` is the filtered composed length so page math stays consistent.
    if await _prefix_sharing_edge(conn, session_id) is not None:
        composed = _filter_composed(
            await get_messages(conn, session_id),
            message_role=message_role,
            message_type=message_type,
        )
        return composed[offset : offset + limit], len(composed)

    query = f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE m.session_id = ?
    """
    count_query = "SELECT COUNT(*) FROM messages WHERE session_id = ?"
    params: list[str | int] = [session_id]

    role_values = message_role_sql_values(message_role)
    if role_values:
        placeholders = ",".join("?" for _ in role_values)
        query += f" AND m.role IN ({placeholders})"
        count_query += f" AND role IN ({placeholders})"
        params.extend(role_values)

    if message_type:
        normalized_type = validate_message_type_filter(message_type).value
        query += " AND m.message_type = ?"
        count_query += " AND message_type = ?"
        params.append(normalized_type)

    # Get total count before pagination
    count_cursor = await conn.execute(count_query, tuple(params))
    count_row = await count_cursor.fetchone()
    total = count_row[0] if count_row else 0

    query += " ORDER BY m.position, m.variant_index, m.message_id"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    messages = [_row_to_message(row) for row in rows]

    return messages, total


async def get_message_edge_windows(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    message_role: MessageRoleFilter = (),
    message_type: MessageTypeName | None = None,
    material_origin: MaterialOriginFilter | None = None,
    edge_limit: int = 8,
) -> tuple[list[MessageRecord], list[MessageRecord], int]:
    """Return first/last transcript-order message windows for one session.

    This is for bounded export/review projections: transcript order is the
    provider position, while timestamps remain evidence displayed on rows.
    """

    session_id = await _resolve_session_id(conn, session_id)
    edge_limit = max(edge_limit, 1)

    if await _prefix_sharing_edge(conn, session_id) is not None:
        composed = _filter_composed(
            await get_messages(conn, session_id),
            message_role=message_role,
            message_type=message_type,
            material_origin=material_origin,
        )
        total = len(composed)
        first = composed[:edge_limit]
        first_ids = {record.message_id for record in first}
        last = [record for record in composed[-edge_limit:] if record.message_id not in first_ids]
        return first, last, total

    where = "WHERE m.session_id = ?"
    count_where = "WHERE session_id = ?"
    params: list[str | int] = [session_id]
    count_params: list[str | int] = [session_id]

    role_values = message_role_sql_values(message_role)
    if role_values:
        role_placeholders = ",".join("?" for _ in role_values)
        where += f" AND m.role IN ({role_placeholders})"
        count_where += f" AND role IN ({role_placeholders})"
        params.extend(role_values)
        count_params.extend(role_values)

    if message_type:
        normalized_type = validate_message_type_filter(message_type).value
        where += " AND m.message_type = ?"
        count_where += " AND message_type = ?"
        params.append(normalized_type)
        count_params.append(normalized_type)

    material_origin_values = _material_origin_values(material_origin)
    if material_origin_values:
        origin_placeholders = ",".join("?" for _ in material_origin_values)
        where += f" AND m.material_origin IN ({origin_placeholders})"
        count_where += f" AND material_origin IN ({origin_placeholders})"
        params.extend(material_origin_values)
        count_params.extend(material_origin_values)

    count_cursor = await conn.execute(
        f"SELECT COUNT(*) FROM messages INDEXED BY idx_messages_session_position {count_where}",
        tuple(count_params),
    )
    count_row = await count_cursor.fetchone()
    total = int(count_row[0]) if count_row is not None else 0

    first_cursor = await conn.execute(
        f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        {where}
        ORDER BY m.position, m.variant_index, m.message_id
        LIMIT ?
        """,
        (*params, edge_limit),
    )
    first = [_row_to_message(row) for row in await first_cursor.fetchall()]
    first_ids = {record.message_id for record in first}

    last_cursor = await conn.execute(
        f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        {where}
        ORDER BY m.position DESC, m.variant_index DESC, m.message_id DESC
        LIMIT ?
        """,
        (*params, edge_limit),
    )
    last_desc = [_row_to_message(row) for row in await last_cursor.fetchall()]
    last = [record for record in reversed(last_desc) if record.message_id not in first_ids]
    return first, last, total


async def iter_messages(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    chunk_size: int = 100,
    message_roles: MessageRoleFilter = (),
    limit: int | None = None,
) -> AsyncIterator[MessageRecord]:
    """Stream a session's messages in deterministic order, chunked.

    Pagination is keyset, not ``LIMIT/OFFSET``: each chunk is seeded by the
    previous chunk's last ``(sort_key, message_id)`` so a single session's
    stream stays linear instead of re-scanning and discarding all prior rows
    (O(M^2)) on every chunk. The ordering
    ``(sort_key IS NULL), sort_key, message_id`` is unchanged and served by
    ``idx_messages_session_sortkey``. ``message_id`` is the table primary
    key (globally unique), so the keyset cursor is a total order with no skipped
    or duplicated rows across chunk boundaries. NULL-``sort_key`` rows form the
    ordering tail and are advanced by a distinct cursor branch.
    """
    session_id = await _resolve_session_id(conn, session_id)
    yielded = 0
    effective_roles = message_roles

    # Keyset streaming is per-session, but a prefix-sharing child's logical
    # transcript spans its parent, so the cursor cannot stream across the
    # lineage boundary. Compose the full transcript and yield from it for forks
    # (a minority of sessions); plain sessions keep the linear keyset stream
    # below (#2470).
    if await _prefix_sharing_edge(conn, session_id) is not None:
        composed = _filter_composed(await get_messages(conn, session_id), message_role=effective_roles)
        for record in composed:
            if limit is not None and yielded >= limit:
                return
            yield record
            yielded += 1
        return

    role_values = message_role_sql_values(effective_roles)

    # Keyset cursor of the previous chunk's final row. ``have_cursor``
    # distinguishes the first chunk from a genuine NULL-sort_key boundary
    # (``last_sort`` is None in both cases).
    last_sort: float | None = None
    last_id: str = ""
    have_cursor = False

    while True:
        query = f"""
            SELECT {_MESSAGE_RECORD_SELECT}
            FROM messages m
            JOIN sessions s ON s.session_id = m.session_id
            WHERE m.session_id = ?
        """
        params: list[str | float] = [session_id]

        if role_values:
            placeholders = ",".join("?" for _ in role_values)
            query += f" AND m.role IN ({placeholders})"
            params.extend(role_values)

        if have_cursor:
            if last_sort is not None:
                # Cursor in the non-NULL sort_key group: advance within it and
                # always include the NULL group that follows in the ordering.
                query += (
                    " AND ("
                    "(m.occurred_at_ms IS NOT NULL"
                    " AND (m.occurred_at_ms > ? OR (m.occurred_at_ms = ? AND m.message_id > ?)))"
                    " OR m.occurred_at_ms IS NULL"
                    ")"
                )
                last_sort_ms = last_sort * 1000.0
                params.extend([last_sort_ms, last_sort_ms, last_id])
            else:
                # Cursor in the NULL sort_key group (ordering tail): only
                # NULL-sort_key rows with a greater message_id remain.
                query += " AND m.occurred_at_ms IS NULL AND m.message_id > ?"
                params.append(last_id)

        query += " ORDER BY (m.occurred_at_ms IS NULL), m.occurred_at_ms, m.message_id"

        fetch_limit = chunk_size
        if limit is not None:
            remaining = limit - yielded
            if remaining <= 0:
                break
            fetch_limit = min(chunk_size, remaining)

        query += " LIMIT ?"
        params.append(fetch_limit)

        cursor = await conn.execute(query, tuple(params))
        rows = list(await cursor.fetchall())

        if not rows:
            break

        last_row = rows[-1]
        last_sort = last_row["sort_key"]
        last_id = str(last_row["message_id"])
        have_cursor = True

        for row in rows:
            yield _row_to_message(row)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

        if len(rows) < fetch_limit:
            break


__all__ = ["get_messages", "get_messages_batch", "get_messages_paginated", "iter_messages"]
