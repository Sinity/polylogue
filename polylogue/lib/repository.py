from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from polylogue.db import open_connection
from polylogue.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.lib.models import Conversation


class ConversationRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        # open_connection context manager handles setup
        # But here we need a raw connection or a context manager we can use?
        # polylogue.db.open_connection returns a context manager that strictly yields conn.
        # We can't reuse it easily across calls if we close it immediately.
        # For a Repository, usually passing a connection or managing one is best.
        # For simplicity in this MVP, let's open per-call using existing helper.
        return open_connection(self._db_path)

    def get(self, conversation_id: str) -> Conversation | None:
        with self._get_conn() as conn:
            # 1. Fetch Conversation
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if not row:
                return None

            # Map row to Record (Pydantic models need dicts or kwargs)
            # RowObkect is dict-like
            conv_record = ConversationRecord(**dict(row))

            # 2. Fetch Messages
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
                (conversation_id,),
            ).fetchall()
            msg_records = [MessageRecord(**dict(r)) for r in msg_rows]

            # 3. Fetch Attachments
            # Attachments are linked via attachment_refs
            att_rows = conn.execute(
                """
                SELECT 
                    attachment_refs.message_id, 
                    attachments.*
                FROM attachment_refs
                JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
                WHERE attachment_refs.conversation_id = ?
                """,
                (conversation_id,),
            ).fetchall()

            att_records = []
            for r in att_rows:
                # We need to correctly map the joined row to AttachmentRecord
                # The join result has message_id from refs, but AttachmentRecord expects it?
                # AttachmentRecord has message_id field.
                data = dict(r)
                # Ensure message_id is communicated. attachment_refs.message_id
                # overrides whatever might be in attachments table (which doesn't link to msg)
                # Actually AttachmentRecord definition in store.py HAS message_id.
                # In store.ts: upsert_attachment sets ref.
                # Wait, AttachmentRecord in store.py is the structure of the INSERT payload.
                # In DB, 'attachments' table doesn't have message_id (it's in refs).
                # So we synthesize the record.
                data["message_id"] = r["message_id"]
                att_records.append(AttachmentRecord(**data))

            return Conversation.from_records(conv_record, msg_records, att_records)

    def list(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
        # This is expensive if we do full fetch for lists.
        # Ideally we return a lightweight ConversationSummary model for lists.
        # For MVP, full fetch is safest (but slow).
        # Let's optimize: List usually needs Title, Date, ID, Provider.
        # But our return type says List[Conversation].
        # We'll fetch basic data and empty messages list?
        # Or just IDs and call get()?
        # Let's fetch IDs first.
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id 
                FROM conversations 
                ORDER BY updated_at DESC 
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
            ids = [r["conversation_id"] for r in rows]

        # This N+1 is bad for 50 items.
        # But for an MVP "Library" it works.
        # Optimization: Bulk fetch messages?
        results = []
        for cid in ids:
            c = self.get(cid)
            if c:
                results.append(c)
        return results

    def search(self, query: str) -> list[Conversation]:
        # FTS search
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id 
                FROM conversations_fts 
                WHERE conversations_fts MATCH ? 
                ORDER BY rank 
                LIMIT 20
                """,
                (query,),
            ).fetchall()
            ids = [r["conversation_id"] for r in rows]

        results = []
        for cid in ids:
            c = self.get(cid)
            if c:
                results.append(c)
        return results
