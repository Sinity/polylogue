"""Async reads for ``web_content_constructs`` (polylogue-kktg).

``web_content_constructs`` is written every ingest from the ChatGPT/Claude
parsers (search queries/results, canvas documents, content references,
image results, async tasks, selected sources, token budgets, voice notes --
``core.enums.WebConstructType``, ``archive_tiers/write.py::
_write_web_constructs``), 155k+ live rows, but had no reader above the
storage layer: every production SELECT against it existed only to DELETE
orphans (an integrity sweep) or as a demo smoke-probe COUNT(*). Follows the
same shape as ``queries/file_edits.py``/``queries/session_refs.py``
(single-session + session-batch reads over the dedicated
``idx_web_constructs_session_type`` index).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.storage.runtime import WebContentConstructRecord
from polylogue.storage.sqlite.queries.mappers_archive import _row_to_web_content_construct

__all__ = [
    "get_web_content_constructs_for_session",
    "get_web_content_constructs_for_session_batch",
]

_SELECT_COLUMNS = (
    "construct_id, session_id, message_id, block_id, position, provider, construct_type, "
    "provider_key, title, url, text, source_id, group_id, group_title, query, asset_pointer, "
    "mime_type, status, task_id, task_type, rank, start_index, end_index"
)


async def get_web_content_constructs_for_session(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    construct_type: str | None = None,
) -> list[WebContentConstructRecord]:
    """Return all web-content-construct rows for one session, ordered by message/block/position.

    ``construct_type`` optionally narrows to a single
    ``core.enums.WebConstructType`` value (e.g. ``"search_result"``),
    pushing the filter down onto the dedicated
    ``idx_web_constructs_session_type`` index instead of filtering in
    Python.
    """
    if construct_type is not None:
        rows = await (
            await conn.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM web_content_constructs
                WHERE session_id = ? AND construct_type = ?
                ORDER BY message_id, block_id, position
                """,
                (session_id, construct_type),
            )
        ).fetchall()
    else:
        rows = await (
            await conn.execute(
                f"""
                SELECT {_SELECT_COLUMNS}
                FROM web_content_constructs
                WHERE session_id = ?
                ORDER BY message_id, block_id, position
                """,
                (session_id,),
            )
        ).fetchall()
    return [_row_to_web_content_construct(row) for row in rows]


async def get_web_content_constructs_for_session_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[WebContentConstructRecord]]:
    """Return web-content-construct rows for many sessions, grouped by session_id."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM web_content_constructs
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, message_id, block_id, position
            """,
            tuple(session_ids),
        )
    ).fetchall()
    result: dict[str, list[WebContentConstructRecord]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        record = _row_to_web_content_construct(row)
        result[str(record.session_id)].append(record)
    return dict(result)
