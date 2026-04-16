"""Raw conversation read and selection helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import aiosqlite

from polylogue.storage.backends.connection import _build_source_scope_filter
from polylogue.storage.backends.queries.mappers import _row_to_raw_conversation
from polylogue.storage.backends.queries.raw_state import EFFECTIVE_RAW_PROVIDER_SQL
from polylogue.storage.state_views import RawConversationState
from polylogue.storage.store import RawConversationRecord


def _raw_select_query(
    select_columns: str,
    *,
    source_names: list[str] | None = None,
    provider_name: str | None = None,
    require_unparsed: bool = False,
    require_unvalidated: bool = False,
    validation_statuses: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    where_clauses: list[str] = []
    params: list[str] = []

    if require_unparsed:
        where_clauses.append("parsed_at IS NULL")
    if require_unvalidated:
        where_clauses.append("validated_at IS NULL")
    if provider_name:
        where_clauses.append(f"{EFFECTIVE_RAW_PROVIDER_SQL} = ?")
        params.append(provider_name)
    if validation_statuses:
        placeholders = ",".join("?" for _ in validation_statuses)
        where_clauses.append(f"validation_status IN ({placeholders})")
        params.extend(validation_statuses)

    predicate, scope_params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    if predicate:
        where_clauses.append(predicate)
        params.extend(scope_params)

    sql = f"SELECT {select_columns} FROM raw_conversations"
    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"
    sql += " ORDER BY acquired_at DESC, raw_id ASC"
    return sql, tuple(params)


def raw_id_query(
    *,
    source_names: list[str] | None = None,
    provider_name: str | None = None,
    require_unparsed: bool = False,
    require_unvalidated: bool = False,
    validation_statuses: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    return _raw_select_query(
        "raw_id",
        source_names=source_names,
        provider_name=provider_name,
        require_unparsed=require_unparsed,
        require_unvalidated=require_unvalidated,
        validation_statuses=validation_statuses,
    )


def raw_header_query(
    *,
    source_names: list[str] | None = None,
    provider_name: str | None = None,
    require_unparsed: bool = False,
    require_unvalidated: bool = False,
    validation_statuses: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    return _raw_select_query(
        "raw_id, blob_size",
        source_names=source_names,
        provider_name=provider_name,
        require_unparsed=require_unparsed,
        require_unvalidated=require_unvalidated,
        validation_statuses=validation_statuses,
    )


async def iter_raw_ids(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
    provider_name: str | None = None,
    require_unparsed: bool = False,
    require_unvalidated: bool = False,
    validation_statuses: list[str] | None = None,
    page_size: int = 1000,
) -> AsyncIterator[str]:
    sql, params = raw_id_query(
        source_names=source_names,
        provider_name=provider_name,
        require_unparsed=require_unparsed,
        require_unvalidated=require_unvalidated,
        validation_statuses=validation_statuses,
    )
    cursor = await conn.execute(sql, params)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield str(row["raw_id"])


async def iter_raw_headers(
    conn: aiosqlite.Connection,
    *,
    source_names: list[str] | None = None,
    provider_name: str | None = None,
    require_unparsed: bool = False,
    require_unvalidated: bool = False,
    validation_statuses: list[str] | None = None,
    page_size: int = 1000,
) -> AsyncIterator[tuple[str, int]]:
    sql, params = raw_header_query(
        source_names=source_names,
        provider_name=provider_name,
        require_unparsed=require_unparsed,
        require_unvalidated=require_unvalidated,
        validation_statuses=validation_statuses,
    )
    cursor = await conn.execute(sql, params)
    while True:
        rows = await cursor.fetchmany(page_size)
        if not rows:
            break
        for row in rows:
            yield (str(row["raw_id"]), int(row["blob_size"]))


async def get_raw_conversation(conn: aiosqlite.Connection, raw_id: str) -> RawConversationRecord | None:
    cursor = await conn.execute(
        "SELECT * FROM raw_conversations WHERE raw_id = ?",
        (raw_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_raw_conversation(row)


async def get_known_source_mtimes(conn: aiosqlite.Connection) -> dict[str, str]:
    result: dict[str, str] = {}
    cursor = await conn.execute("SELECT source_path, file_mtime FROM raw_conversations WHERE file_mtime IS NOT NULL")
    while True:
        rows = await cursor.fetchmany(1000)
        if not rows:
            break
        for row in rows:
            result[row["source_path"]] = row["file_mtime"]
    return result


async def get_raw_conversations_batch(conn: aiosqlite.Connection, raw_ids: list[str]) -> list[RawConversationRecord]:
    if not raw_ids:
        return []
    placeholders = ",".join("?" * len(raw_ids))
    cursor = await conn.execute(
        f"SELECT * FROM raw_conversations WHERE raw_id IN ({placeholders})",
        raw_ids,
    )
    records: list[RawConversationRecord] = []
    while True:
        rows = await cursor.fetchmany(200)
        if not rows:
            break
        records.extend(_row_to_raw_conversation(row) for row in rows)
    return records


async def get_raw_blob_sizes(
    conn: aiosqlite.Connection,
    raw_ids: Sequence[str],
) -> list[tuple[str, int]]:
    if not raw_ids:
        return []
    placeholders = ",".join("?" * len(raw_ids))
    cursor = await conn.execute(
        f"SELECT raw_id, blob_size FROM raw_conversations WHERE raw_id IN ({placeholders})",
        tuple(raw_ids),
    )
    sizes = {str(row["raw_id"]): int(row["blob_size"]) for row in await cursor.fetchall()}
    return [(raw_id, sizes[raw_id]) for raw_id in raw_ids if raw_id in sizes]


async def get_raw_conversation_states(
    conn: aiosqlite.Connection, raw_ids: list[str]
) -> dict[str, RawConversationState]:
    if not raw_ids:
        return {}
    placeholders = ",".join("?" * len(raw_ids))
    cursor = await conn.execute(
        f"""
        SELECT
            raw_id,
            source_name,
            source_path,
            parsed_at,
            parse_error,
            payload_provider,
            validation_status,
            validation_provider
        FROM raw_conversations
        WHERE raw_id IN ({placeholders})
        """,
        raw_ids,
    )
    rows = await cursor.fetchall()
    return {
        row["raw_id"]: RawConversationState(
            raw_id=row["raw_id"],
            source_name=row["source_name"],
            source_path=row["source_path"],
            parsed_at=row["parsed_at"],
            parse_error=row["parse_error"],
            payload_provider=row["payload_provider"],
            validation_status=row["validation_status"],
            validation_provider=row["validation_provider"],
        )
        for row in rows
    }


async def iter_raw_conversations(
    conn: aiosqlite.Connection,
    provider: str | None = None,
    limit: int | None = None,
) -> AsyncIterator[RawConversationRecord]:
    offset = 0
    yielded = 0

    while True:
        query = "SELECT * FROM raw_conversations"
        params: list[str | int] = []
        if provider is not None:
            query += f" WHERE {EFFECTIVE_RAW_PROVIDER_SQL} = ?"
            params.append(provider)
        query += " ORDER BY acquired_at DESC"
        chunk_size = 100
        query_with_limit = query + " LIMIT ? OFFSET ?"
        params.extend([chunk_size, offset])

        cursor = await conn.execute(query_with_limit, tuple(params))
        rows = await cursor.fetchall()
        if not rows:
            break
        for row in rows:
            yield _row_to_raw_conversation(row)
            yielded += 1
            if limit is not None and yielded >= limit:
                return
        offset += len(rows)
        if len(rows) < chunk_size:
            break


async def get_raw_conversation_count(conn: aiosqlite.Connection, provider: str | None = None) -> int:
    query = "SELECT COUNT(*) as cnt FROM raw_conversations"
    params: tuple[str, ...] = ()
    if provider is not None:
        query += f" WHERE {EFFECTIVE_RAW_PROVIDER_SQL} = ?"
        params = (provider,)
    cursor = await conn.execute(query, params)
    row = await cursor.fetchone()
    return int(row["cnt"])


__all__ = [
    "get_known_source_mtimes",
    "get_raw_blob_sizes",
    "get_raw_conversation",
    "get_raw_conversation_count",
    "get_raw_conversation_states",
    "get_raw_conversations_batch",
    "iter_raw_headers",
    "iter_raw_conversations",
    "iter_raw_ids",
    "raw_header_query",
    "raw_id_query",
]
