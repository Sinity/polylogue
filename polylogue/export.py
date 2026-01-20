from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from .db import open_connection


def export_jsonl(*, archive_root: Path, output_path: Path | None = None) -> Path:
    target = output_path or (archive_root / "exports" / "conversations.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open_connection(None) as conn, target.open("w", encoding="utf-8") as handle:
        conversations = conn.execute("SELECT * FROM conversations").fetchall()
        messages = conn.execute(
            """
            SELECT * FROM messages
            ORDER BY
                conversation_id,
                (timestamp IS NULL),
                CASE
                    WHEN timestamp IS NULL THEN NULL
                    WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS INTEGER)
                    ELSE CAST(timestamp AS REAL)
                END,
                message_id
            """
        ).fetchall()
        attachments = conn.execute(
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
            ORDER BY attachment_refs.conversation_id, attachment_refs.ref_id
            """
        ).fetchall()
        messages_by_convo = defaultdict(list)
        for msg in messages:
            messages_by_convo[msg["conversation_id"]].append(dict(msg))
        attachments_by_convo = defaultdict(list)
        for att in attachments:
            attachments_by_convo[att["conversation_id"]].append(dict(att))
        for convo in conversations:
            convo_id = convo["conversation_id"]
            payload = {
                "conversation": dict(convo),
                "messages": messages_by_convo.get(convo_id, []),
                "attachments": attachments_by_convo.get(convo_id, []),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


__all__ = ["export_jsonl"]
