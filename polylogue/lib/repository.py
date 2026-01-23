"""Repository layer for conversation access.

This module provides the `ConversationRepository` class, which is the primary
interface for querying and retrieving conversations from the database.

The repository returns `Conversation` objects that support semantic projections
like `substantive_only()`, `iter_pairs()`, and `without_noise()`.
"""

from __future__ import annotations

import builtins
import logging
import sqlite3
from contextlib import AbstractContextManager as ContextManager
from pathlib import Path

from polylogue.core import json as json_utils
from polylogue.storage.db import DatabaseError, open_connection
from polylogue.lib.models import Conversation
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import ConversationId

logger = logging.getLogger(__name__)


def _decode_meta(payload: dict[str, object]) -> None:
    raw = payload.get("provider_meta")
    if isinstance(raw, str) and raw:
        try:
            payload["provider_meta"] = json_utils.loads(raw)
        except (json_utils.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse provider_meta JSON for conversation %s: %s",
                payload.get("conversation_id", "unknown"),
                exc,
            )
            payload["provider_meta"] = None
    elif raw is None:
        payload["provider_meta"] = None


class ConversationRepository:
    """Repository for querying and retrieving conversations.

    This is the primary interface for accessing conversation data. It returns
    `Conversation` objects that support semantic projections like:

    - `substantive_only()` - Filter to substantive dialogue
    - `iter_pairs()` - Iterate user/assistant turn pairs
    - `without_noise()` - Remove tool calls, context dumps
    - `to_clean_text()` - Render as clean dialogue text

    Example:
        repo = ConversationRepository()
        conv = repo.get("claude:abc123")
        if conv:
            for pair in conv.iter_pairs():
                print(pair.exchange)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def _get_conn(self) -> ContextManager[sqlite3.Connection]:
        return open_connection(self._db_path)

    def resolve_id(self, id_prefix: str) -> ConversationId | None:
        """Resolve a partial ID to a full conversation ID.

        Supports both exact matches and prefix matches. If multiple
        conversations match the prefix, returns None (ambiguous).

        Args:
            id_prefix: Full or partial conversation ID.

        Returns:
            The full conversation ID if exactly one match, None otherwise.
        """
        with self._get_conn() as conn:
            # Try exact match first
            row = conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
                (id_prefix,),
            ).fetchone()
            if row:
                return ConversationId(str(row["conversation_id"]))

            # Try prefix match
            rows = conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id LIKE ? LIMIT 2",
                (f"{id_prefix}%",),
            ).fetchall()
            if len(rows) == 1:
                return ConversationId(str(rows[0]["conversation_id"]))
            return None  # No match or ambiguous

    def view(self, conversation_id: str) -> Conversation | None:
        """Get a conversation with full semantic projection support.

        This is the primary API for consumers. The returned Conversation
        has methods like `substantive_only()`, `iter_pairs()`, `without_noise()`.

        Supports partial ID resolution - if a unique prefix is provided,
        it will be resolved to the full ID.

        Args:
            conversation_id: Full or partial conversation ID.

        Returns:
            A Conversation with projection methods, or None if not found.
        """
        full_id = self.resolve_id(conversation_id)
        if not full_id:
            return None
        return self.get(full_id)

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
            conv_data = dict(row)
            _decode_meta(conv_data)
            conv_record = ConversationRecord(**conv_data)

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
            msg_records = []
            for record in msg_rows:
                data = dict(record)
                _decode_meta(data)
                msg_records.append(MessageRecord(**data))

            # 3. Fetch Attachments
            # Attachments are linked via attachment_refs
            att_rows = conn.execute(
                """
                SELECT
                    attachment_refs.message_id,
                    attachment_refs.conversation_id,
                    attachments.*
                FROM attachment_refs
                JOIN attachments ON attachments.attachment_id = attachment_refs.attachment_id
                WHERE attachment_refs.conversation_id = ?
                """,
                (conversation_id,),
            ).fetchall()

            att_records = []
            for r in att_rows:
                # Build AttachmentRecord from joined row. The attachments table doesn't
                # have message_id/conversation_id columns (those live in attachment_refs),
                # so we pull them from the SELECT explicitly.
                data = dict(r)
                data["message_id"] = r["message_id"]
                data["conversation_id"] = r["conversation_id"]
                _decode_meta(data)
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
            conv_map = {}
            for conv in conv_rows:
                conv_data = dict(conv)
                _decode_meta(conv_data)
                conv_map[conv_data["conversation_id"]] = ConversationRecord(**conv_data)

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
                data = dict(r)
                _decode_meta(data)
                msg_map.setdefault(cid, []).append(MessageRecord(**data))

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
                _decode_meta(data)
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

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
    ) -> list[Conversation]:
        with self._get_conn() as conn:
            if provider:
                rows = conn.execute(
                    """
                    SELECT conversation_id
                    FROM conversations
                    WHERE provider_name = ?
                    ORDER BY
                        CASE WHEN updated_at IS NULL OR updated_at = '' THEN 1 ELSE 0 END,
                        updated_at DESC,
                        conversation_id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (provider, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT conversation_id
                    FROM conversations
                    ORDER BY
                        CASE WHEN updated_at IS NULL OR updated_at = '' THEN 1 ELSE 0 END,
                        updated_at DESC,
                        conversation_id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            ids = [r["conversation_id"] for r in rows]

        return self._get_many(ids)

    def search(self, query: str) -> "builtins.list[Conversation]":
        # FTS search using messages_fts (the actual FTS index)
        with self._get_conn() as conn:
            # Check if FTS table exists before querying
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            if not exists:
                raise DatabaseError(
                    "Search index not built. Run `polylogue run` with index enabled or `polylogue index`."
                )
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
