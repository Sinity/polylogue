"""Read queries for messages."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal, get_args

import aiosqlite

from polylogue.archive.message.roles import MessageRoleFilter, message_role_sql_values
from polylogue.archive.message.types import validate_message_type_filter
from polylogue.archive.topology.edge import invalidated_prefix_sql, topology_status_composes_sql
from polylogue.core.enums import MaterialOrigin, MessageType
from polylogue.core.identity_law import transcript_order_sql
from polylogue.logging import get_logger
from polylogue.storage.runtime import (
    LINEAGE_TRUNCATION_CYCLE,
    LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT,
    LineageCompleteness,
    MessageRecord,
)
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import MESSAGES_SPEC
from polylogue.storage.sqlite.queries.mappers_archive import bind_message_row_mapper

logger = get_logger(__name__)

#: polylogue-jglh: the same seven members as :class:`MessageType`, which
#: owns them. Kept as a name because callers pass the plain strings.
MessageTypeName = Literal["message", "summary", "tool_use", "tool_result", "thinking", "context", "protocol"]

if frozenset(get_args(MessageTypeName)) != frozenset(member.value for member in MessageType):
    raise RuntimeError("MessageTypeName drifted from MessageType")
MaterialOriginFilter = MaterialOrigin | str | tuple[MaterialOrigin | str, ...] | list[MaterialOrigin | str]

_MESSAGE_RECORD_SELECT = MESSAGES_SPEC.record_select_column_names("m")

_TRANSCRIPT_ORDER = transcript_order_sql("m")
_TRANSCRIPT_ORDER_DESC = transcript_order_sql("m", descending=True)


async def _resolve_session_id(conn: aiosqlite.Connection, session_id: str) -> str:
    cursor = await conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ? OR native_id = ? LIMIT 1", (session_id, session_id)
    )
    row = await cursor.fetchone()
    return str(row["session_id"]) if row is not None else session_id


async def _prefix_sharing_edge(conn: aiosqlite.Connection, session_id: str) -> tuple[str, str] | None:
    """Return ``(parent_session_id, branch_point_message_id)`` if this session
    inherits a parent's leading prefix (fork / resume / spawned subagent /
    auto-compaction copy), else ``None``. See the lineage model (#2467)."""
    cursor = await conn.execute(
        f"""
        SELECT resolved_dst_session_id, branch_point_message_id
        FROM session_links
        WHERE src_session_id = ?
          AND inheritance = 'prefix-sharing'
          AND resolved_dst_session_id IS NOT NULL
          AND branch_point_message_id IS NOT NULL
          AND {topology_status_composes_sql()}
        ORDER BY link_type, dst_origin, dst_native_id
        LIMIT 1
        """,
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return (str(row["resolved_dst_session_id"]), str(row["branch_point_message_id"]))


async def _has_invalidated_prefix(conn: aiosqlite.Connection, session_id: str) -> bool:
    async with conn.execute(
        f"SELECT 1 FROM session_links WHERE src_session_id = ? AND {invalidated_prefix_sql()} LIMIT 1",
        (session_id,),
    ) as cursor:
        return await cursor.fetchone() is not None


async def _branch_point_content_address_matches(
    conn: aiosqlite.Connection,
    child_session_id: str,
    parent_session_id: str,
    branch_point_message_id: str,
) -> bool:
    cursor = await conn.execute(
        """
        SELECT l.branch_point_content_address, m.content_address
        FROM session_links AS l
        LEFT JOIN messages AS m ON m.message_id = l.branch_point_message_id
        WHERE l.src_session_id = ?
          AND l.resolved_dst_session_id = ?
          AND l.branch_point_message_id = ?
          AND l.inheritance = 'prefix-sharing'
        LIMIT 1
        """,
        (child_session_id, parent_session_id, branch_point_message_id),
    )
    row = await cursor.fetchone()
    if row is None or row[0] is None:
        return True
    return row[1] is not None and bytes(row[0]) == bytes(row[1])


async def _own_messages(conn: aiosqlite.Connection, session_id: str) -> list[MessageRecord]:
    cursor = await conn.execute(
        f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        WHERE m.session_id = ?
        ORDER BY {_TRANSCRIPT_ORDER}
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
    return [decode(row) for row in rows]


@dataclass(frozen=True, slots=True)
class _Segment:
    session_id: str
    end: tuple[int, int] | None = None


async def _segment_count(
    conn: aiosqlite.Connection,
    segment: _Segment,
    *,
    role_values: tuple[str, ...] = (),
    message_type: str | None = None,
) -> int:
    where, params = _segment_predicate(segment, role_values=role_values, message_type=message_type)
    row = await (await conn.execute(f"SELECT COUNT(*) FROM messages m WHERE {where}", params)).fetchone()
    assert row is not None
    return int(row[0])


def _segment_predicate(
    segment: _Segment, *, role_values: tuple[str, ...] = (), message_type: str | None = None
) -> tuple[str, tuple[str | int, ...]]:
    where = "m.session_id = ?"
    params: list[str | int] = [segment.session_id]
    if segment.end is not None:
        where += " AND (m.position, m.variant_index) <= (?, ?)"
        params.extend(segment.end)
    if role_values:
        where += f" AND m.role IN ({','.join('?' for _ in role_values)})"
        params.extend(role_values)
    if message_type is not None:
        where += " AND m.message_type = ?"
        params.append(message_type)
    return where, tuple(params)


async def _lineage_segments(
    conn: aiosqlite.Connection, session_id: str
) -> tuple[tuple[_Segment, ...], LineageCompleteness]:
    """Resolve the logical transcript from edge and coordinate metadata in one snapshot."""
    chain: list[tuple[str, str, str]] = []
    visited = {session_id}
    cursor_session = session_id
    reason = None
    # ``visited`` is the whole termination argument: every step adds a new
    # session, and an archive holds finitely many.
    while (edge := await _prefix_sharing_edge(conn, cursor_session)) is not None:
        parent, branch_point = edge
        parent = await _resolve_session_id(conn, parent)
        if parent in visited:
            reason = LINEAGE_TRUNCATION_CYCLE
            break
        chain.append((cursor_session, parent, branch_point))
        visited.add(parent)
        cursor_session = parent

    if reason is None and await _has_invalidated_prefix(conn, cursor_session):
        reason = LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT
    segments: tuple[_Segment, ...] = (_Segment(cursor_session),)
    for child, parent, branch_point in reversed(chain):
        if reason is None and await _has_invalidated_prefix(conn, child):
            reason = LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT
        coordinates = await (
            await conn.execute(
                "SELECT session_id, position, variant_index FROM messages WHERE message_id = ?",
                (branch_point,),
            )
        ).fetchone()
        witness_matches = await _branch_point_content_address_matches(conn, child, parent, branch_point)
        prefix: tuple[_Segment, ...] | None = None
        if coordinates is not None and witness_matches:
            owner = str(coordinates["session_id"])
            end = (int(coordinates["position"]), int(coordinates["variant_index"]))
            for index, segment in enumerate(segments):
                if segment.session_id == owner and (segment.end is None or end <= segment.end):
                    prefix = (*segments[:index], _Segment(owner, end))
                    break
        if prefix is None:
            if reason is None:
                reason = LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT
            segments = (_Segment(child),)
        else:
            segments = (*prefix, _Segment(child))
    return segments, LineageCompleteness(complete=reason is None, truncation_reason=reason)


async def get_messages(conn: aiosqlite.Connection, session_id: str) -> list[MessageRecord]:
    """Compose a session's full message transcript, holding one read snapshot.

    Thin wrapper over ``get_messages_with_lineage_completeness`` that drops
    the completeness signal for callers that don't need it (single source of
    truth for the composition logic -- see that function's docstring).
    """
    messages, _completeness = await get_messages_with_lineage_completeness(conn, session_id)
    return messages


#: The newest compaction that ends before ``at_position``. A row with no
#: recorded end is still the newest boundary: it is selected so that its
#: incompleteness refuses the summary, rather than being filtered out so an
#: older, complete boundary silently stands in for it.
_EFFECTIVE_CONTEXT_BOUNDARY_SQL = """
    SELECT boundary_start_position, boundary_end_position, boundary_message_id
    FROM session_events
    WHERE session_id = ? AND event_type = 'compaction'
      AND (boundary_end_position IS NULL OR boundary_end_position < ?)
    ORDER BY position DESC
    LIMIT 1
"""


def effective_context_window(
    messages: Sequence[MessageRecord],
    boundary: Sequence[object] | None,
    at_position: int | None,
) -> list[MessageRecord]:
    """Decide the messages visible to the model at ``at_position``.

    The one owner of this decision for every effective-context route.
    ``messages`` is the lineage-composed transcript and ``at_position`` an
    index into it: a prefix-sharing child's own rows restart at position zero,
    while a compaction's recorded range counts the whole replayed transcript.
    ``boundary`` is the row :data:`_EFFECTIVE_CONTEXT_BOUNDARY_SQL` selected.
    Any incomplete or inconsistent boundary yields the plain prefix.
    """
    position = len(messages) - 1 if at_position is None else at_position
    prefix = list(messages[: max(0, position + 1)])
    if boundary is None:
        return prefix
    start, end, summary_id = boundary[0], boundary[1], boundary[2]
    if not isinstance(start, int) or not isinstance(end, int) or summary_id is None:
        return prefix
    if not 0 <= start <= end < len(prefix):
        return prefix
    summary_index = next(
        (index for index, message in enumerate(prefix) if str(message.message_id) == str(summary_id)),
        None,
    )
    if summary_index is None or summary_index <= end:
        return prefix
    summary = prefix[summary_index]
    return [summary] + [message for message in prefix[end + 1 :] if message is not summary]


async def get_effective_context(
    conn: aiosqlite.Connection,
    session_id: str,
    at_position: int | None = None,
) -> list[MessageRecord]:
    """Apply a local compaction to the lineage-composed transcript prefix.

    Messages and the boundary are read in one snapshot; see
    :func:`effective_context_window` for the decision.
    """
    if not conn.in_transaction:
        await conn.execute("BEGIN DEFERRED")
        try:
            return await get_effective_context(conn, session_id, at_position)
        finally:
            await conn.execute("ROLLBACK")
    resolved = await _resolve_session_id(conn, session_id)
    messages = await get_messages(conn, resolved)
    position = len(messages) - 1 if at_position is None else at_position
    cursor = await conn.execute(_EFFECTIVE_CONTEXT_BOUNDARY_SQL, (resolved, position))
    boundary = await cursor.fetchone()
    return effective_context_window(messages, None if boundary is None else tuple(boundary), position)


def get_effective_context_sync(
    conn: sqlite3.Connection,
    session_id: str,
    at_position: int | None = None,
) -> list[MessageRecord]:
    """Synchronous twin of :func:`get_effective_context` over a pinned snapshot.

    Composes the transcript from the same plan the envelope reads use, then
    applies the shared :func:`effective_context_window` decision.
    """
    from polylogue.storage.sqlite.archive_tiers.write import _composed_transcript_plan

    messages: list[MessageRecord] = []
    for segment in _composed_transcript_plan(conn, session_id).segments:
        bound = ""
        params: tuple[object, ...] = (segment.session_id,)
        if segment.upto_position is not None and segment.upto_variant_index is not None:
            bound = " AND (m.position, m.variant_index) <= (?, ?)"
            params = (segment.session_id, segment.upto_position, segment.upto_variant_index)
        cursor = conn.execute(
            f"SELECT {_MESSAGE_RECORD_SELECT} FROM messages m JOIN sessions s ON s.session_id = m.session_id "
            f"WHERE m.session_id = ?{bound} ORDER BY {_TRANSCRIPT_ORDER}",
            params,
        )
        decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
        messages.extend(decode(row) for row in cursor.fetchall())
    position = len(messages) - 1 if at_position is None else at_position
    boundary = conn.execute(_EFFECTIVE_CONTEXT_BOUNDARY_SQL, (session_id, position)).fetchone()
    return effective_context_window(messages, None if boundary is None else tuple(boundary), position)


async def get_messages_with_lineage_completeness(
    conn: aiosqlite.Connection,
    session_id: str,
) -> tuple[list[MessageRecord], LineageCompleteness]:
    """Compose a session's full message transcript, holding one read snapshot,
    and report whether the composed transcript is complete (4ts.6).

    Composition issues multiple autocommit SELECTs while walking the lineage
    chain (edge reads, then a read per ancestor/descendant). Without a held
    transaction, a concurrent parent re-ingest between those reads can yield a
    torn transcript (4ts.4). If ``conn`` is not already inside a transaction
    (e.g. a caller-held write transaction), this wraps the whole composition
    in one deferred read transaction so every SELECT sees the same snapshot.

    Two paths return an INCOMPLETE transcript: a cycle, or a dangling branch
    point (the parent message was hard-deleted, so only this session's own
    divergent tail is returned starting mid-conversation).
    Consumers that care (MCP get_messages, context-image) can distinguish a
    complete logical transcript from a truncated one via the returned
    ``LineageCompleteness``.
    """
    if not conn.in_transaction:
        await conn.execute("BEGIN DEFERRED")
        try:
            return await get_messages_with_lineage_completeness(conn, session_id)
        finally:
            await conn.execute("ROLLBACK")
    session_id = await _resolve_session_id(conn, session_id)

    # Lineage composition (#2467): a prefix-sharing child stores only its own
    # divergent tail. Walk UP the parent chain collecting (child, branch_point)
    # links to the root, then compose DOWN. This is ITERATIVE (not recursive) so
    # deep acompact/fork chains cannot hit Python's recursion limit. The
    # `visited` set stops a cyclic session_link and bounds the walk: every step
    # adds a new session. No depth cap drops a valid ancestor.
    chain: list[tuple[str, str]] = []  # (child_session_id, branch_point_message_id), leaf-first
    visited: set[str] = {session_id}
    cursor_session = session_id
    cycle = False
    while (edge := await _prefix_sharing_edge(conn, cursor_session)) is not None:
        parent_session_id, branch_point_message_id = edge
        parent_session_id = await _resolve_session_id(conn, parent_session_id)
        if parent_session_id in visited:  # cyclic lineage: stop and compose what we have
            cycle = True
            break
        chain.append((cursor_session, branch_point_message_id))
        visited.add(parent_session_id)
        cursor_session = parent_session_id

    if not chain:
        # A self-parent cycle also leaves the chain empty.
        lost = await _has_invalidated_prefix(conn, session_id)
        return await _own_messages(conn, session_id), LineageCompleteness(
            complete=not cycle and not lost,
            truncation_reason=(
                LINEAGE_TRUNCATION_CYCLE if cycle else LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT if lost else None
            ),
        )

    # Compose from the root down: root's full transcript, then splice each
    # descendant's own tail at its branch point in the running composed view.
    # One list is cut at each branch point and extended by each tail, with a
    # first-position index, so a deep chain composes in linear time rather
    # than rebuilding every intermediate transcript.
    composed = await _own_messages(conn, cursor_session)
    position: dict[str, int] = {}
    for index, record in enumerate(composed):
        position.setdefault(record.message_id, index)
    dangling = await _has_invalidated_prefix(conn, cursor_session)
    for child_session_id, branch_point_message_id in reversed(chain):
        dangling = dangling or await _has_invalidated_prefix(conn, child_session_id)
        own = await _own_messages(conn, child_session_id)
        edge = await _prefix_sharing_edge(conn, child_session_id)
        witness_matches = edge is None or await _branch_point_content_address_matches(
            conn, child_session_id, edge[0], branch_point_message_id
        )
        at = position.get(branch_point_message_id)
        if at is not None and witness_matches:
            for record in composed[at + 1 :]:
                if position.get(record.message_id, -1) > at:
                    del position[record.message_id]
            del composed[at + 1 :]
        else:
            # Dangling branch point (e.g. the parent message was hard-deleted):
            # return this child's own tail rather than an over-long transcript
            # (#2467 audit).
            composed, position = [], {}
            dangling = True
        for record in own:
            position.setdefault(record.message_id, len(composed))
            composed.append(record)
    reason = LINEAGE_TRUNCATION_CYCLE if cycle else LINEAGE_TRUNCATION_DANGLING_BRANCH_POINT if dangling else None
    return composed, LineageCompleteness(complete=reason is None, truncation_reason=reason)


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

        query += f" ORDER BY m.session_id, {_TRANSCRIPT_ORDER}"
        cursor = await conn.execute(
            query,
            tuple(params),
        )
        rows = await cursor.fetchall()
        decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
        for row in rows:
            cid = row["session_id"]
            msg = decode(row)
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
) -> tuple[list[MessageRecord], int, LineageCompleteness]:
    """Return paginated messages for a session with optional filters.

    Returns ``(messages, total_count, lineage_completeness)`` where
    ``total_count`` is the count of messages matching the SQL-level
    filters (before limit/offset), and ``lineage_completeness`` reports
    whether ``messages`` is a page of the FULL composed transcript or was
    truncated (dangling branch point, cycle, or depth limit -- polylogue-ppkj).
    A non-lineage session (the ``else`` branch below) can never be truncated
    this way, so it always reports ``LineageCompleteness()`` (complete=True).
    """
    if not conn.in_transaction:
        await conn.execute("BEGIN DEFERRED")
        try:
            return await get_messages_paginated(
                conn, session_id, message_role=message_role, message_type=message_type, limit=limit, offset=offset
            )
        finally:
            await conn.execute("ROLLBACK")
    session_id = await _resolve_session_id(conn, session_id)

    # A prefix-sharing child stores only its divergent tail. Plan its logical
    # segments and fetch the requested window under the same read snapshot.
    if await _prefix_sharing_edge(conn, session_id) is not None:
        segments, completeness = await _lineage_segments(conn, session_id)
        role_values = message_role_sql_values(message_role)
        type_value = validate_message_type_filter(message_type).value if message_type else None
        counts = [
            await _segment_count(conn, segment, role_values=role_values, message_type=type_value)
            for segment in segments
        ]
        total = sum(counts)
        page: list[MessageRecord] = []
        skip = max(offset, 0)
        remaining = max(limit, 0)
        for segment, count in zip(segments, counts, strict=True):
            if skip >= count:
                skip -= count
                continue
            if remaining <= 0:
                break
            where, segment_params = _segment_predicate(segment, role_values=role_values, message_type=type_value)
            cursor = await conn.execute(
                f"SELECT {_MESSAGE_RECORD_SELECT} FROM messages m JOIN sessions s ON s.session_id = m.session_id "
                f"WHERE {where} ORDER BY {_TRANSCRIPT_ORDER} LIMIT ? OFFSET ?",
                (*segment_params, remaining, skip),
            )
            decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
            page.extend(decode(row) for row in await cursor.fetchall())
            remaining = max(limit, 0) - len(page)
            skip = 0
        return page, total, completeness

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

    query += f" ORDER BY {_TRANSCRIPT_ORDER}"
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
    messages = [decode(row) for row in rows]

    return messages, total, LineageCompleteness()


async def get_lineage_completeness(conn: aiosqlite.Connection, session_id: str) -> LineageCompleteness:
    """Standalone probe for the read-time lineage-completeness signal.

    For callers that need to know whether ``session_id``'s composed
    transcript is truncated but build their own message list a different way
    (e.g. a material-origin-filtered read going through the full ``Session``
    domain object) and so cannot receive the signal from
    ``get_messages_paginated`` directly.
    """
    if not conn.in_transaction:
        await conn.execute("BEGIN DEFERRED")
        try:
            return await get_lineage_completeness(conn, session_id)
        finally:
            await conn.execute("ROLLBACK")
    _segments, completeness = await _lineage_segments(conn, await _resolve_session_id(conn, session_id))
    return completeness


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
        ORDER BY {_TRANSCRIPT_ORDER}
        LIMIT ?
        """,
        (*params, edge_limit),
    )
    first_decode = bind_message_row_mapper(tuple(column[0] for column in first_cursor.description or ()))
    first = [first_decode(row) for row in await first_cursor.fetchall()]
    first_ids = {record.message_id for record in first}

    last_cursor = await conn.execute(
        f"""
        SELECT {_MESSAGE_RECORD_SELECT}
        FROM messages m
        JOIN sessions s ON s.session_id = m.session_id
        {where}
        ORDER BY {_TRANSCRIPT_ORDER_DESC}
        LIMIT ?
        """,
        (*params, edge_limit),
    )
    last_decode = bind_message_row_mapper(tuple(column[0] for column in last_cursor.description or ()))
    last_desc = [last_decode(row) for row in await last_cursor.fetchall()]
    last = [record for record in reversed(last_desc) if record.message_id not in first_ids]
    return first, last, total


async def iter_messages(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    chunk_size: int = 100,
    message_roles: MessageRoleFilter = (),
    material_origin: MaterialOriginFilter | None = None,
    limit: int | None = None,
) -> AsyncGenerator[MessageRecord, None]:
    """Stream a session's messages in transcript order, chunked.

    Pagination is keyset, not ``LIMIT/OFFSET``: each chunk is seeded by the
    previous chunk's last ``(position, variant_index)`` so a single session's
    stream stays linear instead of re-scanning and discarding all prior rows
    (O(M^2)) on every chunk. That pair is the messages primary key under
    ``session_id``, so the cursor is a total order with no skipped or duplicated
    rows across chunk boundaries, and ``idx_messages_session_position`` serves
    the ordering without a temp sort. The stream and the batch read
    (``get_messages_paginated``) share ``_TRANSCRIPT_ORDER``: a caller may take
    a total from one and slice the other.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not conn.in_transaction:
        async with conn.execute("BEGIN DEFERRED"):
            pass
        try:
            async with aclosing(
                iter_messages(
                    conn,
                    session_id,
                    chunk_size=chunk_size,
                    message_roles=message_roles,
                    material_origin=material_origin,
                    limit=limit,
                )
            ) as records:
                async for record in records:
                    yield record
        finally:
            async with conn.execute("ROLLBACK"):
                pass
        return

    session_id = await _resolve_session_id(conn, session_id)
    segments, _completeness = await _lineage_segments(conn, session_id)
    role_values = message_role_sql_values(message_roles)
    material_values = _material_origin_values(material_origin)
    yielded = 0
    for segment in segments:
        where, bound_params = _segment_predicate(segment, role_values=role_values)
        if material_values:
            where += f" AND m.material_origin IN ({','.join('?' for _ in material_values)})"
            bound_params = (*bound_params, *material_values)
        after = ""
        cursor_params: tuple[int, ...] = ()
        while limit is None or yielded < limit:
            fetch_limit = chunk_size if limit is None else min(chunk_size, limit - yielded)
            async with conn.execute(
                f"SELECT {_MESSAGE_RECORD_SELECT} FROM messages m JOIN sessions s ON s.session_id = m.session_id "
                f"WHERE {where}{after} ORDER BY {_TRANSCRIPT_ORDER} LIMIT ?",
                (*bound_params, *cursor_params, fetch_limit),
            ) as cursor:
                rows = list(await cursor.fetchall())
                decode = bind_message_row_mapper(tuple(column[0] for column in cursor.description or ()))
            for row in rows:
                yield decode(row)
                yielded += 1
            if len(rows) < fetch_limit:
                break
            last = rows[-1]
            after = " AND (m.position, m.variant_index) > (?, ?)"
            cursor_params = (int(last["position"]), int(last["branch_index"]))


__all__ = [
    "effective_context_window",
    "get_effective_context",
    "get_effective_context_sync",
    "get_lineage_completeness",
    "get_messages",
    "get_messages_batch",
    "get_messages_paginated",
    "get_messages_with_lineage_completeness",
    "iter_messages",
]
