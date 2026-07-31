"""Async reads for ``file_edits`` (polylogue-2qx.4).

``file_edits`` is a dedicated typed table the writer
(``archive_tiers/write.py._write_file_edits``) populates from
``ParsedFileEdit`` evidence attached to a TOOL_RESULT block, keyed by the
paired TOOL_USE block's ``block_id`` -- structuredPatch/originalFile/
oldString/newString/replaceAll/filePath/userModified, previously discarded
entirely (polylogue-cgfy). Follows the same shape as
``queries/session_agent_policies.py`` (single-tool-use + session-batch).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import aiosqlite

from polylogue.storage.runtime import FileEditRecord
from polylogue.storage.sqlite.queries.mappers import _row_to_file_edit

__all__ = [
    "get_file_edits_for_session",
    "get_file_edits_for_session_batch",
]

_SELECT_COLUMNS = (
    "tool_use_block_id, session_id, message_id, file_path, structured_patch_json, "
    "original_file, old_string, new_string, replace_all, user_modified, observed_at_ms"
)


async def get_file_edits_for_session(
    conn: aiosqlite.Connection,
    session_id: str,
) -> list[FileEditRecord]:
    """Return all file-edit rows for one session, ordered by message/block."""
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM file_edits
            WHERE session_id = ?
            ORDER BY message_id, tool_use_block_id
            """,
            (session_id,),
        )
    ).fetchall()
    return [_row_to_file_edit(row) for row in rows]


async def get_file_edits_for_session_batch(
    conn: aiosqlite.Connection,
    session_ids: Sequence[str],
) -> dict[str, list[FileEditRecord]]:
    """Return file-edit rows for many sessions, grouped by session_id."""
    if not session_ids:
        return {}
    placeholders = ", ".join("?" for _ in session_ids)
    rows = await (
        await conn.execute(
            f"""
            SELECT {_SELECT_COLUMNS}
            FROM file_edits
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, message_id, tool_use_block_id
            """,
            tuple(session_ids),
        )
    ).fetchall()
    result: dict[str, list[FileEditRecord]] = defaultdict(list)
    for session_id in session_ids:
        result.setdefault(session_id, [])
    for row in rows:
        record = _row_to_file_edit(row)
        result[str(record.session_id)].append(record)
    return dict(result)
