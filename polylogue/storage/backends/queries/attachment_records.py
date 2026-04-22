"""Attachment read-query helpers."""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from polylogue.storage.backends.connection import _build_provider_scope_filter
from polylogue.storage.backends.queries.mappers import _json_object, _parse_json
from polylogue.storage.search_models import ConversationSearchEvidenceHit
from polylogue.storage.store import AttachmentRecord
from polylogue.types import ConversationId


def _build_attachment_record(row: aiosqlite.Row, *, conversation_id: str) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=row["attachment_id"],
        conversation_id=ConversationId(conversation_id),
        message_id=row["message_id"],
        mime_type=row["mime_type"],
        size_bytes=row["size_bytes"],
        path=row["path"],
        provider_meta=_json_object(
            _parse_json(
                row["provider_meta"],
                field="provider_meta",
                record_id=row["attachment_id"],
            )
        ),
    )


async def get_attachments(
    conn: aiosqlite.Connection,
    conversation_id: str,
) -> list[AttachmentRecord]:
    """Get all attachments for a conversation."""
    cursor = await conn.execute(
        """
        SELECT a.*, r.message_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.conversation_id = ?
        """,
        (conversation_id,),
    )
    rows = await cursor.fetchall()
    return [_build_attachment_record(row, conversation_id=conversation_id) for row in rows]


async def get_attachments_batch(
    conn: aiosqlite.Connection,
    conversation_ids: list[str],
) -> dict[str, list[AttachmentRecord]]:
    """Get attachments for multiple conversations in a single query."""
    if not conversation_ids:
        return {}
    result: dict[str, list[AttachmentRecord]] = {cid: [] for cid in conversation_ids}
    placeholders = ",".join("?" for _ in conversation_ids)
    cursor = await conn.execute(
        f"""
        SELECT a.*, r.message_id, r.conversation_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.conversation_id IN ({placeholders})
        """,
        conversation_ids,
    )
    rows = await cursor.fetchall()
    for row in rows:
        cid = row["conversation_id"]
        if cid in result:
            result[cid].append(_build_attachment_record(row, conversation_id=cid))
    return result


def _parse_since_timestamp(since: str) -> float:
    try:
        return datetime.fromisoformat(since).timestamp()
    except ValueError as exc:
        raise ValueError(f"Invalid --since date '{since}': {exc}. Use ISO format (e.g., 2023-01-01)") from exc


def _compact_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text if len(text) <= 120 else f"{text[:117]}..."


def _attachment_identity_snippet(row: aiosqlite.Row) -> str:
    field = _compact_text(row["identity_field"]) or "attachment"
    value = _compact_text(row["identity_value"]) or ""
    parts = [f"{field}={value}"]

    attachment_id = _compact_text(row["attachment_id"])
    if attachment_id and attachment_id != value:
        parts.append(f"attachment_id={attachment_id}")
    name = _compact_text(row["attachment_name"])
    if name:
        parts.append(f'name="{name}"')
    mime_type = _compact_text(row["mime_type"])
    if mime_type:
        parts.append(f"mime={mime_type}")
    path = _compact_text(row["path"])
    if path:
        parts.append(f"path={path}")
    return "attachment identity " + " ".join(parts)


async def search_attachment_identity_evidence_hits(
    conn: aiosqlite.Connection,
    query: str,
    limit: int = 100,
    providers: list[str] | None = None,
    since: str | None = None,
) -> list[ConversationSearchEvidenceHit]:
    """Search selected attachment identity fields and return evidence-bearing hits."""
    identity = query.strip()
    if not identity or limit <= 0:
        return []

    sql = """
        WITH base_attachments AS (
            SELECT
                r.conversation_id,
                r.message_id,
                a.attachment_id,
                a.mime_type,
                a.path,
                a.provider_meta AS attachment_meta,
                r.provider_meta AS ref_meta,
                c.provider_name,
                COALESCE(m.sort_key, c.sort_key, 0) AS sort_key,
                COALESCE(
                    json_extract(a.provider_meta, '$.name'),
                    json_extract(a.provider_meta, '$.title'),
                    json_extract(r.provider_meta, '$.name'),
                    json_extract(r.provider_meta, '$.title')
                ) AS attachment_name
            FROM attachments a
            JOIN attachment_refs r ON r.attachment_id = a.attachment_id
            JOIN conversations c ON c.conversation_id = r.conversation_id
            LEFT JOIN messages m ON m.message_id = r.message_id
            WHERE 1 = 1
    """
    params: list[str | int | float] = []

    if providers:
        scope_sql, scope_params = _build_provider_scope_filter(providers, provider_column="c.provider_name")
        sql += f" AND {scope_sql}"
        params.extend(scope_params)

    if since:
        sql += " AND COALESCE(m.sort_key, c.sort_key, 0) >= ?"
        params.append(_parse_since_timestamp(since))

    sql += """
        ),
        identity_candidates AS (
            SELECT *, 'attachment_id' AS identity_field, attachment_id AS identity_value, 0 AS identity_rank
            FROM base_attachments
            UNION ALL
            SELECT *, 'provider_meta.provider_id', CAST(json_extract(attachment_meta, '$.provider_id') AS TEXT), 1
            FROM base_attachments
            UNION ALL
            SELECT *, 'provider_meta.id', CAST(json_extract(attachment_meta, '$.id') AS TEXT), 2
            FROM base_attachments
            UNION ALL
            SELECT *, 'provider_meta.fileId', CAST(json_extract(attachment_meta, '$.fileId') AS TEXT), 3
            FROM base_attachments
            UNION ALL
            SELECT *, 'provider_meta.driveId', CAST(json_extract(attachment_meta, '$.driveId') AS TEXT), 4
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref_meta.provider_id', CAST(json_extract(ref_meta, '$.provider_id') AS TEXT), 5
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref_meta.id', CAST(json_extract(ref_meta, '$.id') AS TEXT), 6
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref_meta.fileId', CAST(json_extract(ref_meta, '$.fileId') AS TEXT), 7
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref_meta.driveId', CAST(json_extract(ref_meta, '$.driveId') AS TEXT), 8
            FROM base_attachments
        ),
        matched AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY identity_rank ASC, sort_key DESC, attachment_id ASC, COALESCE(message_id, '') ASC
                ) AS conversation_rank
            FROM identity_candidates
            WHERE identity_value = ?
        )
        SELECT
            conversation_id,
            message_id,
            attachment_id,
            mime_type,
            path,
            identity_field,
            identity_value,
            attachment_name
        FROM matched
        WHERE conversation_rank = 1
        ORDER BY identity_rank ASC, sort_key DESC, attachment_id ASC, COALESCE(message_id, '') ASC
        LIMIT ?
    """
    params.extend((identity, limit))
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [
        ConversationSearchEvidenceHit(
            conversation_id=str(row["conversation_id"]),
            rank=rank,
            score=None,
            message_id=str(row["message_id"]) if row["message_id"] is not None else None,
            snippet=_attachment_identity_snippet(row),
            match_surface="attachment",
            retrieval_lane="attachment",
        )
        for rank, row in enumerate(rows, start=1)
    ]


__all__ = [
    "get_attachments",
    "get_attachments_batch",
    "search_attachment_identity_evidence_hits",
]
