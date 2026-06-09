"""Session read/query families."""

from __future__ import annotations

import aiosqlite

from polylogue.storage.runtime import SessionRecord
from polylogue.storage.sqlite.queries.filter_builder import _build_session_filters, _needs_stats_join
from polylogue.storage.sqlite.queries.mappers import _row_to_session
from polylogue.types import Provider

_SESSION_RECORD_SELECT = """
    session_id,
    native_id,
    origin,
    title,
    datetime(created_at_ms / 1000, 'unixepoch') AS created_at,
    datetime(updated_at_ms / 1000, 'unixepoch') AS updated_at,
    sort_key_ms / 1000.0 AS sort_key,
    lower(hex(content_hash)) AS content_hash,
    '{}' AS metadata,
    1 AS version,
    parent_session_id,
    branch_type,
    raw_id,
    (SELECT json_group_array(path) FROM session_working_dirs swd WHERE swd.session_id = sessions.session_id ORDER BY position) AS working_directories_json,
    git_branch,
    git_repository_url
"""


def _session_record_select(alias: str | None = None) -> str:
    prefix = f"{alias}." if alias else ""
    session_id_expr = f"{prefix}session_id" if prefix else "sessions.session_id"
    cwd_expr = (
        "SELECT json_group_array(path) FROM session_working_dirs swd "
        f"WHERE swd.session_id = {session_id_expr} ORDER BY position"
    )
    return f"""
    {prefix}session_id AS session_id,
    {prefix}native_id AS native_id,
    {prefix}origin AS origin,
    {prefix}title AS title,
    datetime({prefix}created_at_ms / 1000, 'unixepoch') AS created_at,
    datetime({prefix}updated_at_ms / 1000, 'unixepoch') AS updated_at,
    {prefix}sort_key_ms / 1000.0 AS sort_key,
    lower(hex({prefix}content_hash)) AS content_hash,
    '{{}}' AS metadata,
    1 AS version,
    {prefix}parent_session_id AS parent_session_id,
    {prefix}branch_type AS branch_type,
    {prefix}raw_id AS raw_id,
    ({cwd_expr}) AS working_directories_json,
    {prefix}git_branch AS git_branch,
    {prefix}git_repository_url AS git_repository_url
    """


def _native_id_candidates(session_id: str) -> tuple[str, ...]:
    if ":" not in session_id:
        return (session_id,)
    prefix, native = session_id.split(":", 1)
    if Provider.from_string(prefix).value == prefix and native:
        return (session_id, native)
    return (session_id,)


async def get_session(conn: aiosqlite.Connection, session_id: str) -> SessionRecord | None:
    cursor = await conn.execute(
        f"SELECT {_SESSION_RECORD_SELECT} FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        candidates = _native_id_candidates(session_id)
        placeholders = ",".join("?" for _ in candidates)
        cursor = await conn.execute(
            f"""
            SELECT {_SESSION_RECORD_SELECT}
            FROM sessions
            WHERE native_id IN ({placeholders})
            ORDER BY sort_key_ms DESC, session_id DESC
            LIMIT 1
            """,
            candidates,
        )
        row = await cursor.fetchone()
    return _row_to_session(row) if row is not None else None


async def get_sessions_batch(conn: aiosqlite.Connection, ids: list[str]) -> list[SessionRecord]:
    if not ids:
        return []
    native_candidates: list[str] = []
    for session_id in ids:
        native_candidates.extend(_native_id_candidates(session_id))
    placeholders = ",".join("?" for _ in ids)
    native_placeholders = ",".join("?" for _ in native_candidates)
    cursor = await conn.execute(
        f"""
        SELECT {_SESSION_RECORD_SELECT}
        FROM sessions
        WHERE session_id IN ({placeholders})
           OR native_id IN ({native_placeholders})
        """,
        [*ids, *native_candidates],
    )
    rows = await cursor.fetchall()
    by_id = {row["session_id"]: _row_to_session(row) for row in rows}
    by_id.update({row["native_id"]: _row_to_session(row) for row in rows})
    result: list[SessionRecord] = []
    for cid in ids:
        record = by_id.get(cid)
        if record is None:
            for candidate in _native_id_candidates(cid):
                record = by_id.get(candidate)
                if record is not None:
                    break
        if record is not None:
            result.append(record)
    return result


async def list_sessions(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    message_type: str | None = None,
) -> list[SessionRecord]:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_session_filters(
        source=source,
        provider=provider,
        providers=providers,
        parent_id=parent_id,
        since=since,
        until=until,
        title_contains=title_contains,
        referenced_path=referenced_path,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        message_type=message_type,
    )

    if use_stats_join:
        from_clause = "FROM sessions c"
        select_clause = f"SELECT {_session_record_select('c')}"
        order_clause = "ORDER BY (c.sort_key_ms IS NULL) ASC, c.sort_key_ms DESC, c.session_id DESC"
    else:
        from_clause = "FROM sessions"
        select_clause = f"SELECT {_session_record_select()}"
        order_clause = "ORDER BY (sort_key_ms IS NULL) ASC, sort_key_ms DESC, session_id DESC"

    query = f"""
        {select_clause} {from_clause}
        {where_sql}
        {order_clause}
    """
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    elif offset > 0:
        query += " LIMIT -1"

    if offset > 0:
        query += " OFFSET ?"
        params.append(offset)

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session(row) for row in rows]


async def list_session_summaries(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    parent_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    limit: int | None = None,
    offset: int = 0,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    message_type: str | None = None,
) -> list[SessionRecord]:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_session_filters(
        source=source,
        provider=provider,
        providers=providers,
        parent_id=parent_id,
        since=since,
        until=until,
        title_contains=title_contains,
        referenced_path=referenced_path,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        message_type=message_type,
    )

    if use_stats_join:
        from_clause = "FROM sessions c"
        select_clause = f"SELECT {_session_record_select('c')}"
        order_clause = "ORDER BY (c.sort_key_ms IS NULL) ASC, c.sort_key_ms DESC, c.session_id DESC"
    else:
        from_clause = "FROM sessions"
        select_clause = f"SELECT {_session_record_select()}"
        order_clause = "ORDER BY (sort_key_ms IS NULL) ASC, sort_key_ms DESC, session_id DESC"

    query = f"""
        {select_clause} {from_clause}
        {where_sql}
        {order_clause}
    """
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    elif offset > 0:
        query += " LIMIT -1"

    if offset > 0:
        query += " OFFSET ?"
        params.append(offset)

    cursor = await conn.execute(query, tuple(params))
    rows = await cursor.fetchall()
    return [_row_to_session(row) for row in rows]


async def count_sessions(
    conn: aiosqlite.Connection,
    *,
    source: str | None = None,
    provider: str | None = None,
    providers: list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    title_contains: str | None = None,
    referenced_path: list[str] | None = None,
    cwd_prefix: str | None = None,
    action_terms: list[str] | None = None,
    excluded_action_terms: list[str] | None = None,
    tool_terms: list[str] | None = None,
    excluded_tool_terms: list[str] | None = None,
    repo_names: list[str] | None = None,
    has_tool_use: bool = False,
    has_thinking: bool = False,
    has_paste: bool = False,
    typed_only: bool = False,
    min_messages: int | None = None,
    max_messages: int | None = None,
    min_words: int | None = None,
    message_type: str | None = None,
) -> int:
    use_stats_join = _needs_stats_join(
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
    )
    where_sql, params = _build_session_filters(
        source=source,
        provider=provider,
        providers=providers,
        since=since,
        until=until,
        title_contains=title_contains,
        referenced_path=referenced_path,
        cwd_prefix=cwd_prefix,
        action_terms=action_terms,
        excluded_action_terms=excluded_action_terms,
        tool_terms=tool_terms,
        excluded_tool_terms=excluded_tool_terms,
        repo_names=repo_names,
        has_tool_use=has_tool_use,
        has_thinking=has_thinking,
        has_paste=has_paste,
        typed_only=typed_only,
        min_messages=min_messages,
        max_messages=max_messages,
        min_words=min_words,
        message_type=message_type,
    )
    if use_stats_join:
        sql = f"SELECT COUNT(*) as cnt FROM sessions c {where_sql}"
    else:
        sql = f"SELECT COUNT(*) as cnt FROM sessions {where_sql}"
    cursor = await conn.execute(sql, tuple(params))
    row = await cursor.fetchone()
    return int(row["cnt"]) if row else 0


async def list_sessions_by_parent(conn: aiosqlite.Connection, parent_id: str) -> list[SessionRecord]:
    cursor = await conn.execute(
        f"""
        SELECT {_SESSION_RECORD_SELECT}
        FROM sessions
        WHERE parent_session_id = ?
        ORDER BY created_at_ms ASC
        """,
        (parent_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_session(row) for row in rows]


__all__ = [
    "count_sessions",
    "get_session",
    "get_sessions_batch",
    "list_session_summaries",
    "list_sessions",
    "list_sessions_by_parent",
]
