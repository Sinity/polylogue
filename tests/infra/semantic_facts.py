"""Semantic fact extraction for cross-surface agreement testing.

Normalizes conversations, query results, and products into stable
semantic fact tuples so tests can compare meaning rather than formatting.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass

from polylogue.lib.json import JSONDocument, json_document_list
from polylogue.lib.models import Conversation
from polylogue.storage.store import ConversationRecord, MessageRecord


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _content_blocks(message: JSONDocument) -> list[JSONDocument]:
    return json_document_list(message.get("content_blocks"))


@dataclass(frozen=True)
class ConversationFacts:
    """Stable semantic facts about a conversation, independent of surface."""

    conversation_id: str
    provider: str
    title: str | None
    message_count: int
    role_multiset: dict[str, int]
    has_attachments: bool
    has_tool_use: bool
    has_thinking: bool

    @classmethod
    def from_domain_conversation(cls, conv: Conversation) -> ConversationFacts:
        """Extract facts from a domain Conversation object."""
        messages = list(conv.messages)
        roles = Counter(str(m.role) for m in messages)
        has_tool_use = any(
            m.content_blocks and any(block.get("type") in ("tool_use", "tool_result") for block in m.content_blocks)
            for m in messages
        )
        has_thinking = any(
            m.content_blocks and any(block.get("type") == "thinking" for block in m.content_blocks) for m in messages
        )
        has_attachments = any(m.attachments for m in messages)
        return cls(
            conversation_id=str(conv.id),
            provider=str(conv.provider),
            title=conv.title,
            message_count=len(messages),
            role_multiset=dict(roles),
            has_attachments=has_attachments,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
        )

    @classmethod
    def from_json_payload(cls, payload: JSONDocument) -> ConversationFacts:
        """Extract facts from a CLI JSON output payload."""
        messages = json_document_list(payload.get("messages"))
        roles = Counter(_string_value(m.get("role")) or "unknown" for m in messages)
        has_tool_use = any(
            any(block.get("type") in ("tool_use", "tool_result") for block in _content_blocks(message))
            for message in messages
        )
        has_thinking = any(
            any(block.get("type") == "thinking" for block in _content_blocks(message)) for message in messages
        )
        has_attachments = any(bool(message.get("attachments")) for message in messages)
        return cls(
            conversation_id=_string_value(payload.get("id")) or _string_value(payload.get("conversation_id")),
            provider=_string_value(payload.get("provider")) or _string_value(payload.get("provider_name")),
            title=_optional_string(payload.get("title")),
            message_count=len(messages),
            role_multiset=dict(roles),
            has_attachments=has_attachments,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
        )

    @classmethod
    def from_records(cls, conv_record: ConversationRecord, msg_records: list[MessageRecord]) -> ConversationFacts:
        """Extract facts from storage records (ConversationRecord + MessageRecords)."""
        roles = Counter(str(m.role) for m in msg_records)
        has_tool_use = any(m.has_tool_use for m in msg_records)
        has_thinking = any(m.has_thinking for m in msg_records)
        return cls(
            conversation_id=str(conv_record.conversation_id),
            provider=str(conv_record.provider_name),
            title=conv_record.title,
            message_count=len(msg_records),
            role_multiset=dict(roles),
            has_attachments=False,
            has_tool_use=bool(has_tool_use),
            has_thinking=bool(has_thinking),
        )


@dataclass(frozen=True)
class ArchiveFacts:
    """Aggregate facts about the archive, independent of surface."""

    total_conversations: int
    provider_counts: dict[str, int]
    total_messages: int

    @classmethod
    def from_db_connection(cls, conn: sqlite3.Connection) -> ArchiveFacts:
        total_convs = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
        provider_rows = conn.execute(
            "SELECT provider_name, COUNT(*) as cnt FROM conversations GROUP BY provider_name"
        ).fetchall()
        provider_counts = {str(row["provider_name"]): int(row["cnt"]) for row in provider_rows}
        total_msgs = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        return cls(
            total_conversations=total_convs,
            provider_counts=provider_counts,
            total_messages=total_msgs,
        )
