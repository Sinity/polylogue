from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

from .storage.backends.connection import open_connection

# Columns stored as JSON TEXT that need parsing to avoid double-encoding
_JSON_COLUMNS = frozenset({"provider_meta", "metadata", "ref_meta"})


def _row_to_dict(row: object) -> dict[str, object]:
    """Convert a sqlite3.Row to a dict, parsing JSON TEXT columns."""
    d = dict(row)  # type: ignore[arg-type]
    for col in _JSON_COLUMNS:
        if col in d and isinstance(d[col], str):
            with suppress(json.JSONDecodeError, ValueError):
                d[col] = json.loads(d[col])
    return d


def export_jsonl(*, archive_root: Path, output_path: Path | None = None) -> Path:
    """Export all conversations to newline-delimited JSON.

    Each line is a JSON object with keys ``conversation``, ``messages``,
    and ``attachments``, suitable for bulk processing or backup.

    Streams one conversation at a time so peak memory stays proportional to the
    largest single conversation, not the entire database.

    Args:
        archive_root: Root directory of the Polylogue archive.
        output_path: Destination file. Defaults to
            ``archive_root/exports/conversations.jsonl``.

    Returns:
        Path to the created JSONL file.
    """
    target = output_path or (archive_root / "exports" / "conversations.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open_connection(None) as conn, target.open("w", encoding="utf-8") as handle:
        convo_cursor = conn.execute(
            "SELECT * FROM conversations ORDER BY conversation_id"
        )
        while True:
            convo_batch = convo_cursor.fetchmany(500)
            if not convo_batch:
                break
            for convo in convo_batch:
                convo_id = convo["conversation_id"]

                msg_rows = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY
                        (timestamp IS NULL),
                        CASE
                            WHEN timestamp IS NULL THEN NULL
                            WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS INTEGER)
                            ELSE CAST(timestamp AS REAL)
                        END,
                        message_id
                    """,
                    (convo_id,),
                ).fetchall()

                att_rows = conn.execute(
                    """
                    SELECT
                        attachment_refs.ref_id,
                        attachment_refs.conversation_id,
                        attachment_refs.message_id,
                        attachment_refs.attachment_id,
                        attachment_refs.provider_meta AS ref_meta,
                        attachments.mime_type,
                        attachments.size_bytes,
                        attachments.path,
                        attachments.provider_meta
                    FROM attachment_refs
                    JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
                    WHERE attachment_refs.conversation_id = ?
                    ORDER BY attachment_refs.ref_id
                    """,
                    (convo_id,),
                ).fetchall()

                payload = {
                    "conversation": _row_to_dict(convo),
                    "messages": [_row_to_dict(m) for m in msg_rows],
                    "attachments": [_row_to_dict(a) for a in att_rows],
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


__all__ = ["export_jsonl"]
