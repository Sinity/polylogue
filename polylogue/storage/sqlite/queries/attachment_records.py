"""Attachment read-query helpers."""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from polylogue.storage.runtime import AttachmentRecord
from polylogue.storage.search.models import SessionSearchEvidenceRow
from polylogue.storage.sqlite.connection import _build_provider_scope_filter
from polylogue.storage.sqlite.queries.mappers import _json_object, _parse_json
from polylogue.types import SessionId


def _row_value(row: aiosqlite.Row, key: str) -> object | None:
    """Read an optional column from a row, returning None if the column is absent."""
    try:
        value: object = row[key]
    except (IndexError, KeyError):
        return None
    return value


def _build_attachment_record(row: aiosqlite.Row, *, session_id: str) -> AttachmentRecord:
    return AttachmentRecord(
        attachment_id=row["attachment_id"],
        session_id=SessionId(session_id),
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
        provider_attachment_id=(
            value if isinstance((value := _row_value(row, "provider_attachment_id")), str) else None
        ),
        provider_file_id=(value if isinstance((value := _row_value(row, "provider_file_id")), str) else None),
        provider_drive_id=(value if isinstance((value := _row_value(row, "provider_drive_id")), str) else None),
        upload_origin=(value if isinstance((value := _row_value(row, "upload_origin")), str) else None),
    )


async def get_attachments(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[AttachmentRecord]:
    """Get all attachments for a session."""
    cursor = await conn.execute(
        """
        SELECT a.*, r.message_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ?
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_build_attachment_record(row, session_id=session_id) for row in rows]


async def get_attachments_batch(
    conn: aiosqlite.Connection,
    session_ids: list[str],
) -> dict[str, list[AttachmentRecord]]:
    """Get attachments for multiple sessions in a single query."""
    if not session_ids:
        return {}
    result: dict[str, list[AttachmentRecord]] = {cid: [] for cid in session_ids}
    placeholders = ",".join("?" for _ in session_ids)
    cursor = await conn.execute(
        f"""
        SELECT a.*, r.message_id, r.session_id
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.session_id IN ({placeholders})
        """,
        session_ids,
    )
    rows = await cursor.fetchall()
    for row in rows:
        cid = row["session_id"]
        if cid in result:
            result[cid].append(_build_attachment_record(row, session_id=cid))
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
) -> list[SessionSearchEvidenceRow]:
    """Search selected attachment identity fields and return evidence-bearing hits."""
    identity = query.strip()
    if not identity or limit <= 0:
        return []

    # #1252: attachment identity lookup resolves against typed columns
    # (provider_attachment_id, provider_file_id, provider_drive_id) on both
    # `attachments` and `attachment_refs`. The previous implementation read
    # the same identifiers through json_extract on `provider_meta`; the typed
    # surface keeps the lookup on the hot path index-backed and removes the
    # JSON parse per probed row.
    sql = """
        WITH base_attachments AS (
            SELECT
                r.session_id,
                r.message_id,
                a.attachment_id,
                a.mime_type,
                a.path,
                a.provider_meta AS attachment_meta,
                r.provider_meta AS ref_meta,
                a.provider_attachment_id AS a_provider_attachment_id,
                a.provider_file_id AS a_provider_file_id,
                a.provider_drive_id AS a_provider_drive_id,
                r.provider_attachment_id AS r_provider_attachment_id,
                r.provider_file_id AS r_provider_file_id,
                r.provider_drive_id AS r_provider_drive_id,
                c.source_name,
                COALESCE(m.sort_key, c.sort_key, 0) AS sort_key,
                COALESCE(
                    json_extract(a.provider_meta, '$.name'),
                    json_extract(a.provider_meta, '$.title'),
                    json_extract(r.provider_meta, '$.name'),
                    json_extract(r.provider_meta, '$.title')
                ) AS attachment_name
            FROM attachments a
            JOIN attachment_refs r ON r.attachment_id = a.attachment_id
            JOIN sessions c ON c.session_id = r.session_id
            LEFT JOIN messages m ON m.message_id = r.message_id
            WHERE 1 = 1
    """
    params: list[str | int | float] = []

    if providers:
        scope_sql, scope_params = _build_provider_scope_filter(providers, provider_column="c.source_name")
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
            SELECT *, 'attachment.provider_attachment_id', a_provider_attachment_id, 1
            FROM base_attachments
            UNION ALL
            SELECT *, 'attachment.provider_file_id', a_provider_file_id, 2
            FROM base_attachments
            UNION ALL
            SELECT *, 'attachment.provider_drive_id', a_provider_drive_id, 3
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref.provider_attachment_id', r_provider_attachment_id, 4
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref.provider_file_id', r_provider_file_id, 5
            FROM base_attachments
            UNION ALL
            SELECT *, 'ref.provider_drive_id', r_provider_drive_id, 6
            FROM base_attachments
        ),
        matched AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY session_id
                    ORDER BY identity_rank ASC, sort_key DESC, attachment_id ASC, COALESCE(message_id, '') ASC
                ) AS session_rank
            FROM identity_candidates
            WHERE identity_value = ?
        )
        SELECT
            session_id,
            message_id,
            attachment_id,
            mime_type,
            path,
            identity_field,
            identity_value,
            attachment_name
        FROM matched
        WHERE session_rank = 1
        ORDER BY identity_rank ASC, sort_key DESC, attachment_id ASC, COALESCE(message_id, '') ASC
        LIMIT ?
    """
    params.extend((identity, limit))
    cursor = await conn.execute(sql, params)
    rows = await cursor.fetchall()
    return [
        SessionSearchEvidenceRow(
            session_id=str(row["session_id"]),
            rank=rank,
            score=None,
            message_id=str(row["message_id"]) if row["message_id"] is not None else None,
            snippet=_attachment_identity_snippet(row),
            match_surface="attachment",
            retrieval_lane="attachment",
            matched_terms=(identity.lower(),),
            score_kind=None,
            lane_rank=rank,
        )
        for rank, row in enumerate(rows, start=1)
    ]


__all__ = [
    "get_attachments",
    "get_attachments_batch",
    "search_attachment_identity_evidence_hits",
]
