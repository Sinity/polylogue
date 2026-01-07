from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager as ContextManager
from pathlib import Path

from polylogue.db import open_connection
from polylogue.lib.models import Conversation
from polylogue.store import AttachmentRecord, ConversationRecord, MessageRecord


class ConversationRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def _get_conn(self) -> ContextManager[sqlite3.Connection]:
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

    def _get_many(self, conversation_ids: list[str]) -> list[Conversation]:
        """Bulk fetch full conversation objects."""
        if not conversation_ids:
            return []

        with self._get_conn() as conn:
            # 1. Fetch Conversations
            placeholders = ", ".join("?" for _ in conversation_ids)
            conv_rows = conn.execute(
                f"SELECT * FROM conversations WHERE conversation_id IN ({placeholders})",
                tuple(conversation_ids),
            ).fetchall()

            # Map by ID to maintain order or easier lookup
            conv_map = {r["conversation_id"]: ConversationRecord(**dict(r)) for r in conv_rows}

            # 2. Fetch Messages
            msg_rows = conn.execute(
                f"""
                SELECT * FROM messages 
                WHERE conversation_id IN ({placeholders})
                ORDER BY 
                    (timestamp IS NULL),
                    CASE 
                        WHEN timestamp IS NULL THEN NULL
                        WHEN timestamp GLOB '*[^0-9.]*' THEN CAST(strftime('%s', timestamp) AS INTEGER)
                        ELSE CAST(timestamp AS REAL)
                    END,
                    message_id
                """,
                tuple(conversation_ids),
            ).fetchall()

            msg_map: dict[str, list[MessageRecord]] = {}
            for r in msg_rows:
                cid = r["conversation_id"]
                msg_map.setdefault(cid, []).append(MessageRecord(**dict(r)))

            # 3. Fetch Attachments
            att_rows = conn.execute(
                f"""
                SELECT 
                    attachment_refs.message_id, 
                    attachment_refs.conversation_id,
                    attachments.*
                FROM attachment_refs
                JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
                WHERE attachment_refs.conversation_id IN ({placeholders})
                """,
                tuple(conversation_ids),
            ).fetchall()

            att_map: dict[str, list[AttachmentRecord]] = {}
            for r in att_rows:
                cid = r["conversation_id"]
                data = dict(r)
                data["message_id"] = r["message_id"]
                att_map.setdefault(cid, []).append(AttachmentRecord(**data))

        results = []
        for cid in conversation_ids:
            if cid in conv_map:
                results.append(
                    Conversation.from_records(
                        conv_map[cid],
                        msg_map.get(cid, []),
                        att_map.get(cid, []),
                    )
                )
        return results

    def list(self, limit: int = 50, offset: int = 0) -> list[Conversation]:
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

        return self._get_many(ids)

    def search(self, query: str) -> list[Conversation]:
        # FTS search using messages_fts (the actual FTS index)
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT conversation_id 
                FROM messages_fts 
                WHERE messages_fts MATCH ? 
                LIMIT 20
                """,
                (query,),
            ).fetchall()
            ids = [r["conversation_id"] for r in rows]

        return self._get_many(ids)
