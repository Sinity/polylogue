"""Raw conversation pipeline state queries."""

from __future__ import annotations

from collections.abc import AsyncIterator

import aiosqlite

from polylogue.storage.backends.connection import _build_source_scope_filter
from polylogue.storage.backends.queries.mappers import _row_to_raw_conversation
from polylogue.storage.store import (
    RawConversationRecord,
    RawConversationState,
)
from polylogue.types import Provider, ValidationMode, ValidationStatus

__all__ = [
    "raw_id_query",
    "iter_raw_ids",
    "save_raw_conversation",
    "get_raw_conversation",
    "mark_raw_parsed",
    "mark_raw_validated",
    "get_known_source_mtimes",
    "reset_parse_status",
    "reset_validation_status",
    "get_raw_conversations_batch",
    "get_raw_conversation_states",
    "iter_raw_conversations",
    "get_raw_conversation_count",
]


_EFFECTIVE_RAW_PROVIDER_SQL = "COALESCE(payload_provider, provider_name)"


def raw_id_query(
    *,
    source_names: list[str] | None = None,
    provider_name: str | None = None,
    require_unparsed: bool = False,
    require_unvalidated: bool = False,
    validation_statuses: list[str] | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Build the canonical scoped raw-ID query."""
    where_clauses: list[str] = []
    params: list[str] = []

    if require_unparsed:
        where_clauses.append("parsed_at IS NULL")
    if require_unvalidated:
        where_clauses.append("validated_at IS NULL")
    if provider_name:
        where_clauses.append(f"{_EFFECTIVE_RAW_PROVIDER_SQL} = ?")
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

    sql = "SELECT raw_id FROM raw_conversations"
    if where_clauses:
        sql += f" WHERE {' AND '.join(where_clauses)}"
    sql += " ORDER BY acquired_at DESC, raw_id ASC"
    return sql, tuple(params)


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
    """Iterate raw conversation IDs for a pipeline state slice."""
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


async def save_raw_conversation(
    conn: aiosqlite.Connection,
    record: RawConversationRecord,
    transaction_depth: int,
) -> bool:
    """Save a raw conversation record. Returns True if inserted."""
    cursor = await conn.execute(
        """
        INSERT OR IGNORE INTO raw_conversations (
            raw_id,
            provider_name,
            payload_provider,
            source_name,
            source_path,
            source_index,
            raw_content,
            acquired_at,
            file_mtime,
            parsed_at,
            parse_error,
            validated_at,
            validation_status,
            validation_error,
            validation_drift_count,
            validation_provider,
            validation_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.raw_id,
            record.provider_name,
            record.payload_provider,
            record.source_name,
            record.source_path,
            record.source_index,
            record.raw_content,
            record.acquired_at,
            record.file_mtime,
            record.parsed_at,
            record.parse_error,
            record.validated_at,
            record.validation_status,
            record.validation_error,
            record.validation_drift_count,
            record.validation_provider,
            record.validation_mode,
        ),
    )
    inserted = bool(cursor.rowcount > 0)

    if not inserted and record.file_mtime is not None:
        await conn.execute(
            "UPDATE raw_conversations SET file_mtime = ?, source_path = ? "
            "WHERE raw_id = ? AND (file_mtime IS NOT ? OR source_path IS NOT ?)",
            (record.file_mtime, record.source_path,
             record.raw_id, record.file_mtime, record.source_path),
        )

    if transaction_depth == 0:
        await conn.commit()
    return inserted


async def get_raw_conversation(
    conn: aiosqlite.Connection, raw_id: str
) -> RawConversationRecord | None:
    """Retrieve a raw conversation by ID."""
    cursor = await conn.execute(
        "SELECT * FROM raw_conversations WHERE raw_id = ?",
        (raw_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_raw_conversation(row)


async def mark_raw_parsed(
    conn: aiosqlite.Connection,
    raw_id: str,
    *,
    error: str | None = None,
    payload_provider: str | Provider | None = None,
    transaction_depth: int,
) -> None:
    """Mark a raw conversation as parsed (or record a parse error)."""
    from datetime import datetime, timezone

    provider_token = str(Provider.from_string(payload_provider)) if payload_provider is not None else None

    if error is None:
        await conn.execute(
            "UPDATE raw_conversations "
            "SET parsed_at = ?, parse_error = NULL, payload_provider = COALESCE(?, payload_provider) "
            "WHERE raw_id = ?",
            (datetime.now(timezone.utc).isoformat(), provider_token, raw_id),
        )
    else:
        await conn.execute(
            "UPDATE raw_conversations "
            "SET parse_error = ?, payload_provider = COALESCE(?, payload_provider) "
            "WHERE raw_id = ?",
            (error[:2000], provider_token, raw_id),
        )
    if transaction_depth == 0:
        await conn.commit()


async def mark_raw_validated(
    conn: aiosqlite.Connection,
    raw_id: str,
    *,
    status: ValidationStatus | str,
    error: str | None = None,
    drift_count: int = 0,
    provider: Provider | str | None = None,
    mode: ValidationMode | str | None = None,
    payload_provider: Provider | str | None = None,
    transaction_depth: int,
) -> None:
    """Persist validation status for a raw conversation record."""
    from datetime import datetime, timezone

    try:
        validation_status = ValidationStatus.from_string(str(status))
    except ValueError as exc:
        raise ValueError(f"Invalid validation status: {status}") from exc

    validation_provider = Provider.from_string(provider) if provider is not None else None

    try:
        validation_mode = ValidationMode.from_string(str(mode)) if mode is not None else None
    except ValueError as exc:
        raise ValueError(f"Invalid validation mode: {mode}") from exc

    payload_token = Provider.from_string(payload_provider) if payload_provider is not None else None

    await conn.execute(
        """
        UPDATE raw_conversations
        SET validated_at = ?,
            validation_status = ?,
            validation_error = ?,
            validation_drift_count = ?,
            validation_provider = ?,
            validation_mode = ?,
            payload_provider = COALESCE(?, payload_provider)
        WHERE raw_id = ?
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            validation_status,
            (error[:2000] if error else None),
            max(0, int(drift_count)),
            validation_provider,
            validation_mode,
            payload_token,
            raw_id,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


async def get_known_source_mtimes(conn: aiosqlite.Connection) -> dict[str, str]:
    """Return {source_path: file_mtime} for all raw records with an mtime."""
    result: dict[str, str] = {}
    cursor = await conn.execute(
        "SELECT source_path, file_mtime FROM raw_conversations WHERE file_mtime IS NOT NULL"
    )
    while True:
        rows = await cursor.fetchmany(1000)
        if not rows:
            break
        for row in rows:
            result[row["source_path"]] = row["file_mtime"]
    return result


async def reset_parse_status(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    transaction_depth: int,
) -> int:
    """Clear parsed_at/parse_error to force re-parsing on next run."""
    if provider is not None:
        cursor = await conn.execute(
            "UPDATE raw_conversations SET parsed_at = NULL, parse_error = NULL "
            f"WHERE {_EFFECTIVE_RAW_PROVIDER_SQL} = ? "
            "AND (parsed_at IS NOT NULL OR parse_error IS NOT NULL)",
            (provider,),
        )
    else:
        cursor = await conn.execute(
            "UPDATE raw_conversations SET parsed_at = NULL, parse_error = NULL "
            "WHERE parsed_at IS NOT NULL OR parse_error IS NOT NULL"
        )
    if transaction_depth == 0:
        await conn.commit()
    return cursor.rowcount


async def reset_validation_status(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    transaction_depth: int,
) -> int:
    """Clear validation tracking to force re-validation on next run."""
    if provider is not None:
        cursor = await conn.execute(
            "UPDATE raw_conversations "
            "SET validated_at = NULL, validation_status = NULL, validation_error = NULL, "
            "validation_drift_count = NULL, validation_provider = NULL, validation_mode = NULL "
            f"WHERE {_EFFECTIVE_RAW_PROVIDER_SQL} = ? "
            "AND (validated_at IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL)",
            (provider,),
        )
    else:
        cursor = await conn.execute(
            "UPDATE raw_conversations "
            "SET validated_at = NULL, validation_status = NULL, validation_error = NULL, "
            "validation_drift_count = NULL, validation_provider = NULL, validation_mode = NULL "
            "WHERE validated_at IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL"
        )
    if transaction_depth == 0:
        await conn.commit()
    return cursor.rowcount


async def get_raw_conversations_batch(
    conn: aiosqlite.Connection, raw_ids: list[str]
) -> list[RawConversationRecord]:
    """Fetch multiple raw conversations in a single query."""
    if not raw_ids:
        return []
    placeholders = ",".join("?" * len(raw_ids))
    cursor = await conn.execute(
        f"SELECT * FROM raw_conversations WHERE raw_id IN ({placeholders})",  # noqa: S608
        raw_ids,
    )
    records: list[RawConversationRecord] = []
    while True:
        rows = await cursor.fetchmany(200)
        if not rows:
            break
        records.extend(_row_to_raw_conversation(row) for row in rows)
    return records


async def get_raw_conversation_states(
    conn: aiosqlite.Connection, raw_ids: list[str]
) -> dict[str, RawConversationState]:
    """Fetch persisted processing state for raw conversation IDs."""
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
    """Iterate over raw conversation records."""
    offset = 0
    yielded = 0

    while True:
        query = "SELECT * FROM raw_conversations"
        params: list[str | int] = []

        if provider is not None:
            query += f" WHERE {_EFFECTIVE_RAW_PROVIDER_SQL} = ?"
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


async def get_raw_conversation_count(
    conn: aiosqlite.Connection, provider: str | None = None
) -> int:
    """Get count of raw conversations."""
    query = "SELECT COUNT(*) as cnt FROM raw_conversations"
    params: tuple[str, ...] = ()
    if provider is not None:
        query += f" WHERE {_EFFECTIVE_RAW_PROVIDER_SQL} = ?"
        params = (provider,)
    cursor = await conn.execute(query, params)
    row = await cursor.fetchone()
    return int(row["cnt"])
