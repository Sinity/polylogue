"""Semantic fact extraction for cross-surface agreement testing.

Normalizes conversations, query results, and products into stable
semantic fact tuples so tests can compare meaning rather than formatting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


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
    def from_domain_conversation(cls, conv: Any) -> ConversationFacts:
        """Extract facts from a domain Conversation object."""
        messages = list(conv.messages)
        roles = Counter(str(m.role) for m in messages)
        has_tool_use = any(
            getattr(m, "content_blocks", None)
            and any(b.get("type") in ("tool_use", "tool_result") for b in m.content_blocks if isinstance(b, dict))
            for m in messages
        )
        has_thinking = any(
            getattr(m, "content_blocks", None)
            and any(b.get("type") == "thinking" for b in m.content_blocks if isinstance(b, dict))
            for m in messages
        )
        has_attachments = any(getattr(m, "attachments", None) for m in messages)
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
    def from_json_payload(cls, payload: dict[str, Any]) -> ConversationFacts:
        """Extract facts from a CLI JSON output payload."""
        messages = payload.get("messages", [])
        roles = Counter(m.get("role", "unknown") for m in messages)
        has_tool_use = any(
            any(b.get("type") in ("tool_use", "tool_result") for b in m.get("content_blocks", [])) for m in messages
        )
        has_thinking = any(any(b.get("type") == "thinking" for b in m.get("content_blocks", [])) for m in messages)
        has_attachments = any(m.get("attachments") for m in messages)
        return cls(
            conversation_id=payload.get("id", payload.get("conversation_id", "")),
            provider=payload.get("provider", payload.get("provider_name", "")),
            title=payload.get("title"),
            message_count=len(messages),
            role_multiset=dict(roles),
            has_attachments=has_attachments,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
        )

    @classmethod
    def from_records(cls, conv_record: Any, msg_records: list) -> ConversationFacts:
        """Extract facts from storage records (ConversationRecord + MessageRecords)."""
        roles = Counter(str(m.role) for m in msg_records)
        has_tool_use = any(getattr(m, "has_tool_use", 0) for m in msg_records)
        has_thinking = any(getattr(m, "has_thinking", 0) for m in msg_records)
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
    def from_db_connection(cls, conn) -> ArchiveFacts:
        total_convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        provider_rows = conn.execute(
            "SELECT provider_name, COUNT(*) as cnt FROM conversations GROUP BY provider_name"
        ).fetchall()
        provider_counts = {r["provider_name"]: r["cnt"] for r in provider_rows}
        total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return cls(
            total_conversations=total_convs,
            provider_counts=provider_counts,
            total_messages=total_msgs,
        )
