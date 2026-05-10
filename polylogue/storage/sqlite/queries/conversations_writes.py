"""Conversation write/delete helpers."""

from __future__ import annotations

import json as _json

import aiosqlite

from polylogue.core.common import SQL_CONVERSATION_UPSERT as _CONVERSATION_UPSERT_SQL
from polylogue.storage.runtime import ConversationRecord, _json_or_none


async def conversation_exists_by_hash(conn: aiosqlite.Connection, content_hash: str) -> bool:
    cursor = await conn.execute(
        "SELECT 1 FROM conversations WHERE content_hash = ? LIMIT 1",
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
        return _json.dumps(wds)
    cwd = provider_meta.get("cwd")
    if isinstance(cwd, str):
        return _json.dumps([cwd])
    return None


async def save_conversation_record(
    conn: aiosqlite.Connection,
    record: ConversationRecord,
    transaction_depth: int,
) -> None:
    # Derive source_name from provider_meta.source when the field was not
    # explicitly set by the caller (test helpers, make_conversation, etc.).
    source_name = record.source_name
    if not source_name and record.provider_meta:
        raw = record.provider_meta.get("source")
        source_name = raw if isinstance(raw, str) else ""

    working_directories_json = record.working_directories_json or _derive_working_directories_json(record.provider_meta)

    await conn.execute(
        _CONVERSATION_UPSERT_SQL,
        (
            record.conversation_id,
            record.provider_name,
            record.provider_conversation_id,
            record.title,
            record.created_at,
            record.updated_at,
            record.sort_key,
            record.content_hash,
            _json_or_none(record.provider_meta),
            _json_or_none(record.metadata) or "{}",
            record.version,
            record.parent_conversation_id,
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


async def delete_conversation_sql(
    conn: aiosqlite.Connection,
    conversation_id: str,
    transaction_depth: int,
) -> bool:
    cursor = await conn.execute(
        "SELECT parent_conversation_id FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return False

    parent_conversation_id = row[0]

    await conn.execute(
        """
        UPDATE conversations
        SET parent_conversation_id = ?
        WHERE parent_conversation_id = ?
        """,
        (parent_conversation_id, conversation_id),
    )

    cursor = await conn.execute(
        """SELECT DISTINCT ar.attachment_id FROM attachment_refs ar
           JOIN messages m ON ar.message_id = m.message_id
           WHERE m.conversation_id = ?""",
        (conversation_id,),
    )
    affected_attachments = [r[0] for r in await cursor.fetchall()]

    await conn.execute(
        "DELETE FROM conversations WHERE conversation_id = ?",
        (conversation_id,),
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
    "conversation_exists_by_hash",
    "delete_conversation_sql",
    "save_conversation_record",
]
