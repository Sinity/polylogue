"""Session write/delete helpers."""

from __future__ import annotations

import json as _json

import aiosqlite

from polylogue.core.common import SQL_SESSION_UPSERT as _SESSION_UPSERT_SQL
from polylogue.storage.runtime import SessionRecord, _json_or_none


async def session_exists_by_hash(conn: aiosqlite.Connection, content_hash: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM sessions WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    )
    row = await cursor.fetchone()
    return row is not None


def _derive_working_directories_json(provider_meta: dict[str, object] | None) -> str | None:
    """Promote ``provider_meta.working_directories`` (or ``cwd``) into the
    typed ``working_directories_json`` column when the canonical pipeline
    derivation in ``pipeline.prepare_enrichment`` was bypassed.

    Mirrors the same fallback pattern as ``source_name`` so direct write
    callers (test helpers, ad-hoc backfills) get a column the
    ``cwd_prefix`` filter can read.
    """
    if not provider_meta:
        return None
    wds = provider_meta.get("working_directories")
    if isinstance(wds, list):
        cleaned = [item for item in wds if isinstance(item, str)]
        return _json.dumps(cleaned) if cleaned else None
    cwd = provider_meta.get("cwd")
    if isinstance(cwd, str):
        return _json.dumps([cwd])
    return None


async def save_session_record(
    conn: aiosqlite.Connection,
    record: SessionRecord,
    transaction_depth: int,
) -> None:
    # Derive source_name from provider_meta.source when the field was not
    # explicitly set by the caller (test helpers, make_session, etc.).
    source_name = record.source_name
    if not source_name and record.provider_meta:
        raw = record.provider_meta.get("source")
        source_name = raw if isinstance(raw, str) else ""

    working_directories_json = (
        record.working_directories_json
        if record.working_directories_json is not None
        else _derive_working_directories_json(record.provider_meta)
    )

    await conn.execute(
        _SESSION_UPSERT_SQL,
        (
            record.session_id,
            record.source_name,
            record.provider_session_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.sort_key,
            record.content_hash,
            _json_or_none(record.provider_meta),
            _json_or_none(record.metadata) or "{}",
            record.version,
            record.parent_session_id,
            record.branch_type,
            record.raw_id,
            source_name,
            working_directories_json,
            record.git_branch,
            record.git_repository_url,
        ),
    )
    if transaction_depth == 0:
        await conn.commit()


async def delete_session_sql(
    conn: aiosqlite.Connection,
    session_id: str,
    transaction_depth: int,
) -> bool:
    cursor = await conn.execute(
        "SELECT parent_session_id FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    parent_session_id = row[0]

    await conn.execute(
        """
        UPDATE sessions
        SET parent_session_id = ?
        WHERE parent_session_id = ?
        """,
        (parent_session_id, session_id),
    )

    cursor = await conn.execute(
        """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
           JOIN messages m ON ar.message_id = m.message_id
           WHERE m.session_id = ?""",
        (session_id,),
    )
    affected_attachments = [r[0] for r in await cursor.fetchall()]

    await conn.execute(
        "DELETE FROM sessions WHERE session_id = ?",
        (session_id,),
    )

    if affected_attachments:
        placeholders = ",".join("?" * len(affected_attachments))
        await conn.execute(
            f"""UPDATE attachments SET ref_count = (
                    SELECT COUNT(*) FROM attachment_refs
                    WHERE attachment_refs.attachment_id = attachments.attachment_id
                ) WHERE attachment_id IN ({placeholders})""",
            affected_attachments,
        )
        await conn.execute(
            f"DELETE FROM attachments WHERE attachment_id IN ({placeholders}) AND ref_count <= 0",
            affected_attachments,
        )

    if transaction_depth == 0:
        await conn.commit()
    return True


__all__ = [
    "session_exists_by_hash",
    "delete_session_sql",
    "save_session_record",
]
