"""Attachment read-query helpers."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite

from polylogue.core.enums import Origin
from polylogue.core.types import SessionId
from polylogue.storage.runtime import AttachmentRecord
from polylogue.storage.search.models import SessionSearchEvidenceRow

#: Kinds of provider-native attachment identity that a read or an acquisition
#: resolves to exactly one value.
ATTACHMENT_NATIVE_ID_KINDS: tuple[str, ...] = ("attachment", "file", "drive")


def unambiguous_native_id_sql(id_kind: str, *, ref_alias: str = "r") -> str:
    """Correlated selection of one reference's native id for ``id_kind``.

    ``attachment_native_ids`` is keyed ``(ref_id, id_kind, native_id)``, so the
    schema admits two observations of one kind for one reference even though
    every current writer produces at most one
    (``archive_tiers/write.py:_attachment_native_id_values`` reads one
    ``ParsedAttachment`` field per kind, and ``_write_attachments`` clears a
    reference's rows before rewriting them). The two readers here and the
    Drive downloader used to answer that with ``ORDER BY native_id LIMIT 1``:
    lexical order, which is neither provider authority nor a revision. On
    contested identity it picked a winner silently, and every consumer then
    displayed, downloaded and re-bound under it.

    Nothing in this table carries revision or provider-currency evidence, so
    the only authority available is that the observation is unique.
    Ambiguity therefore resolves to ``NULL`` -- an explicitly unresolved
    identity -- while every stored alias stays in the table for history and
    for :func:`search_attachment_identity_evidence_hits`, which deliberately
    joins all of them.

    ``GROUP BY``/``HAVING`` rather than a counting subquery keeps the scan on
    the ``(ref_id, id_kind)`` prefix of the primary key.
    """
    if id_kind not in ATTACHMENT_NATIVE_ID_KINDS:
        raise ValueError(f"unsupported attachment native id kind: {id_kind!r}")
    return (
        "SELECT ani.native_id FROM attachment_native_ids ani "
        f"WHERE ani.ref_id = {ref_alias}.ref_id AND ani.id_kind = '{id_kind}' "
        "GROUP BY ani.ref_id HAVING COUNT(*) = 1"
    )


def contested_native_id_predicate(*, ref_alias: str = "r") -> str:
    """SQL predicate: this reference carries a contested native identity.

    The complement of :func:`unambiguous_native_id_sql` over the kinds an
    acquisition resolves. A candidate matching it has no single downloadable
    identity, so it must stay an explicit unresolved target rather than
    becoming a lexical guess or a terminal "unavailable" claim.
    """
    kinds = ", ".join(f"'{kind}'" for kind in ATTACHMENT_NATIVE_ID_KINDS)
    return (
        "EXISTS (SELECT 1 FROM attachment_native_ids ani "
        f"WHERE ani.ref_id = {ref_alias}.ref_id AND ani.id_kind IN ({kinds}) "
        "GROUP BY ani.ref_id, ani.id_kind HAVING COUNT(*) > 1)"
    )


#: The three identity columns both session reads project, composed once.
_NATIVE_ID_COLUMNS = ",\n".join(
    f"            ({unambiguous_native_id_sql(kind)}) AS {kind}_native_id" for kind in ATTACHMENT_NATIVE_ID_KINDS
)


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
        display_name=(value if isinstance((value := _row_value(row, "display_name")), str) else None),
        source_url=(value if isinstance((value := _row_value(row, "source_url")), str) else None),
        caption=(value if isinstance((value := _row_value(row, "caption")), str) else None),
        attachment_native_id=(value if isinstance((value := _row_value(row, "attachment_native_id")), str) else None),
        file_native_id=(value if isinstance((value := _row_value(row, "file_native_id")), str) else None),
        drive_native_id=(value if isinstance((value := _row_value(row, "drive_native_id")), str) else None),
        upload_origin=(value if isinstance((value := _row_value(row, "upload_origin")), str) else None),
        direction=(value if isinstance((value := _row_value(row, "direction")), str) else None),
        producer_ref=(value if isinstance((value := _row_value(row, "producer_ref")), str) else None),
        reference_id=(value if isinstance((value := _row_value(row, "reference_id")), str) else None),
        supplying_raw_id=(value if isinstance((value := _row_value(row, "supplying_raw_id")), str) else None),
        blob_hash=(bytes(value) if isinstance((value := _row_value(row, "blob_hash")), (bytes, bytearray)) else None),
        acquisition_status=(value if isinstance((value := _row_value(row, "acquisition_status")), str) else None),
        generation_id=(value if isinstance((value := _row_value(row, "generation_id")), str) else None),
    )


async def get_attachments(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[AttachmentRecord]:
    """Get all attachments for a session."""
    cursor = await conn.execute(
        f"""
        SELECT
            a.attachment_id,
            a.media_type AS mime_type,
            a.byte_count AS size_bytes,
            NULL AS path,
            a.blob_hash,
            a.acquisition_status,
            a.display_name,
            r.source_url,
            r.caption,
            r.message_id,
            r.upload_origin,
            r.direction,
            r.producer_ref,
            r.ref_id AS reference_id,
            r.supplying_raw_id,
{_NATIVE_ID_COLUMNS}
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        WHERE r.session_id = ?
        """,
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_build_attachment_record(row, session_id=session_id) for row in rows]


async def get_message_attachments(
    conn: aiosqlite.Connection, message_ids: list[str]
) -> dict[str, list[AttachmentRecord]]:
    """Read each physical message owner's references in bounded SQL batches.

    A child's logical transcript can contain parent messages. Reference
    ownership follows those physical rows, not the requested child's session.
    """
    result: dict[str, list[AttachmentRecord]] = {}
    for start in range(0, len(message_ids), 900):
        batch = message_ids[start : start + 900]
        placeholders = ",".join("?" for _ in batch)
        async with conn.execute(
            f"""
        SELECT
            a.attachment_id,
            a.media_type AS mime_type,
            a.byte_count AS size_bytes,
            NULL AS path,
            a.blob_hash,
            a.acquisition_status,
            a.display_name,
            r.source_url,
            r.caption,
            r.message_id,
            r.session_id,
            r.upload_origin,
            r.direction,
            r.producer_ref,
            r.ref_id AS reference_id,
            r.supplying_raw_id,
{_NATIVE_ID_COLUMNS}
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        JOIN messages m ON m.message_id = r.message_id AND m.session_id = r.session_id
        WHERE r.message_id IN ({placeholders})
        ORDER BY r.message_id, a.attachment_id, r.ref_id
        """,
            tuple(batch),
        ) as cursor:
            rows = await cursor.fetchall()
        for row in rows:
            result.setdefault(row["message_id"], []).append(_build_attachment_record(row, session_id=row["session_id"]))
    return result


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
        SELECT
            a.attachment_id,
            a.media_type AS mime_type,
            a.byte_count AS size_bytes,
            NULL AS path,
            a.blob_hash,
            a.acquisition_status,
            a.display_name,
            r.source_url,
            r.caption,
            r.message_id,
            r.session_id,
            r.upload_origin,
            r.direction,
            r.producer_ref,
            r.ref_id AS reference_id,
            r.supplying_raw_id,
{_NATIVE_ID_COLUMNS}
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


def attachment_library_page_sql(
    *,
    limit: int,
    offset: int,
    mime_filter: str,
    session_filter: str,
    state_filter: str,
    segments: tuple[tuple[str, int | None, int | None], ...] = (),
) -> tuple[str, tuple[object, ...]]:
    """Lower one library window after canonical transcript membership."""
    clauses = ["r.session_id = s.session_id"]
    args: list[object] = []
    if session_filter:
        # Each admitted reference has a physical message owner. Drive the
        # scoped relation from canonical segments instead of walking all refs.
        relation = """
        FROM json_each(?) AS segment
        JOIN messages m ON m.session_id = json_extract(segment.value, '$[0]')
          AND (json_extract(segment.value, '$[1]') IS NULL
               OR (m.position, m.variant_index) <=
                  (json_extract(segment.value, '$[1]'), json_extract(segment.value, '$[2]')))
        JOIN attachment_refs r ON r.message_id = m.message_id AND r.session_id = m.session_id
        JOIN attachments a ON a.attachment_id = r.attachment_id
        JOIN sessions s ON s.session_id = r.session_id
        """
        args.append(json.dumps(segments))
    else:
        relation = """
        FROM attachments a
        JOIN attachment_refs r ON a.attachment_id = r.attachment_id
        JOIN sessions s ON s.session_id = r.session_id
        LEFT JOIN messages m ON m.message_id = r.message_id AND m.session_id = r.session_id
        """
    if mime_filter:
        clauses.append("instr(COALESCE(a.media_type, ''), ?) > 0")
        args.append(mime_filter)
    if state_filter:
        clauses.append(
            "(CASE WHEN a.blob_hash IS NULL THEN 'missing-blob' "
            "WHEN lower(COALESCE(a.media_type, '')) IN "
            "('application/x-tar','application/zip','application/x-7z-compressed','application/x-rar-compressed') "
            "OR lower(COALESCE(a.media_type, '')) LIKE 'application/x-executable%' "
            "OR lower(COALESCE(a.media_type, '')) LIKE 'application/x-msdownload%' "
            "OR lower(COALESCE(a.media_type, '')) LIKE 'application/x-msdos-program%' "
            "OR lower(COALESCE(a.media_type, '')) LIKE 'application/x-sharedlib%' THEN 'unsupported-kind' "
            "WHEN a.byte_count > 8388608 THEN 'too-large' ELSE 'available' END) = ?"
        )
        args.append(state_filter)
    # Newest session first, then transcript order. Within one message the
    # session read orders by attachment ID; ``attachment_refs.position`` is an
    # identity coordinate, not a display ordinal.
    sql = f"""
        SELECT a.attachment_id, a.media_type AS mime_type, a.byte_count AS size_bytes,
               NULL AS path, a.blob_hash, a.acquisition_status, a.display_name,
               r.source_url, r.caption, r.message_id, r.session_id,
               r.upload_origin, r.direction, r.producer_ref,
               r.ref_id AS reference_id, r.supplying_raw_id,
               s.title, s.origin,
               {_NATIVE_ID_COLUMNS}
        {relation}
        WHERE {" AND ".join(clauses)}
        ORDER BY s.sort_key_ms DESC, s.session_id,
                 m.position, m.variant_index, a.attachment_id, r.ref_id
        LIMIT ? OFFSET ?
        """
    return sql, (*args, max(0, limit), max(0, offset))


async def get_attachment_library_page(
    conn: aiosqlite.Connection,
    *,
    limit: int,
    offset: int,
    mime_filter: str = "",
    session_filter: str = "",
    state_filter: str = "",
) -> list[tuple[AttachmentRecord, str, str | None]]:
    """Read one bounded library page under the caller's held read snapshot."""
    segments: tuple[tuple[str, int | None, int | None], ...] = ()
    if session_filter:
        from polylogue.core.errors import DatabaseError
        from polylogue.storage.sqlite.queries.message_query_reads import _lineage_segments

        plan, completeness = await _lineage_segments(conn, session_filter)
        if not completeness.complete:
            raise DatabaseError(
                f"attachment library membership has incomplete lineage: {completeness.truncation_reason}"
            )
        segments = tuple(
            (segment.session_id, segment.end[0], segment.end[1])
            if segment.end is not None
            else (segment.session_id, None, None)
            for segment in plan
        )
    sql, args = attachment_library_page_sql(
        limit=limit,
        offset=offset,
        mime_filter=mime_filter,
        session_filter=session_filter,
        state_filter=state_filter,
        segments=segments,
    )
    async with conn.execute(sql, args) as cursor:
        rows = await cursor.fetchall()
    return [
        (
            _build_attachment_record(row, session_id=str(row["session_id"])),
            str(row["title"] or row["session_id"]),
            row["origin"],
        )
        for row in rows
    ]


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
    origins: list[str] | None = None,
    since: str | None = None,
) -> list[SessionSearchEvidenceRow]:
    """Search selected attachment identity fields and return evidence-bearing hits."""
    identity = query.strip()
    if not identity or limit <= 0:
        return []

    sql = """
        WITH base_attachments AS (
            SELECT
                r.session_id,
                r.message_id,
                a.attachment_id,
                a.media_type AS mime_type,
                COALESCE(r.source_url, a.display_name) AS path,
                ani.id_kind,
                ani.native_id,
                c.origin AS source_name,
                COALESCE(m.occurred_at_ms, c.sort_key_ms) / 1000.0 AS sort_key,
                a.display_name AS attachment_name
            FROM attachments a
            JOIN attachment_refs r ON r.attachment_id = a.attachment_id
            LEFT JOIN attachment_native_ids ani ON ani.ref_id = r.ref_id
            JOIN sessions c ON c.session_id = r.session_id
            LEFT JOIN messages m ON m.message_id = r.message_id
            WHERE 1 = 1
    """
    params: list[str | int | float] = []

    if origins:
        scope_params = [Origin(origin).value for origin in origins]
        placeholders = ",".join("?" for _ in scope_params)
        sql += f" AND c.origin IN ({placeholders})"
        params.extend(scope_params)

    if since:
        # A row with no reliable timestamp anywhere in its fallback chain is
        # not evidence it falls outside a since window -- include it rather
        # than let SQL's NULL propagation silently exclude it
        # (polylogue-s5mm, sort_key_ms COALESCE audit).
        sql += (
            " AND (COALESCE(m.occurred_at_ms, c.sort_key_ms) IS NULL OR COALESCE(m.occurred_at_ms, c.sort_key_ms) >= ?)"
        )
        params.append(_parse_since_timestamp(since) * 1000.0)

    sql += """
        ),
        identity_candidates AS (
            SELECT *, 'attachment_id' AS identity_field, attachment_id AS identity_value, 0 AS identity_rank
            FROM base_attachments
            UNION ALL
            SELECT *, 'native.' || id_kind, native_id, 1
            FROM base_attachments
            WHERE native_id IS NOT NULL
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
