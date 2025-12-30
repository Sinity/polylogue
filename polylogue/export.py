from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .db import open_connection


def export_jsonl(*, archive_root: Path, output_path: Optional[Path] = None) -> Path:
    target = output_path or (archive_root / "exports" / "conversations.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open_connection(None) as conn, target.open("w", encoding="utf-8") as handle:
        conversations = conn.execute("SELECT * FROM conversations").fetchall()
        for convo in conversations:
            messages = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp, message_id",
                (convo["conversation_id"],),
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
                WHERE attachment_refs.conversation_id = ?
                """,
                (convo["conversation_id"],),
            ).fetchall()
            payload = {
                "conversation": dict(convo),
                "messages": [dict(msg) for msg in messages],
                "attachments": [dict(att) for att in attachments],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return target


__all__ = ["export_jsonl"]
