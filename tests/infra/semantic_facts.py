"""Semantic fact extraction for cross-surface agreement testing.

Normalizes conversations, query results, and products into stable
semantic fact tuples so tests can compare meaning rather than formatting.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from polylogue.lib.json import JSONDocument, json_document_list
from polylogue.lib.models import Conversation
from polylogue.lib.semantic_facts import build_conversation_semantic_facts
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord


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
    message_ids: tuple[str, ...] = ()
    text_message_ids: tuple[str, ...] = ()
    text_role_counts: dict[str, int] | None = None
    attachment_count: int = 0
    word_count: int = 0

    @classmethod
    def from_domain_conversation(cls, conv: Conversation) -> ConversationFacts:
        """Extract facts from a domain Conversation object."""
        messages = list(conv.messages)
        semantic = build_conversation_semantic_facts(conv)
        roles = Counter(str(m.role) for m in messages)
        return cls(
            conversation_id=semantic.conversation_id,
            provider=semantic.provider,
            title=conv.title,
            message_count=semantic.total_messages,
            role_multiset=dict(roles),
            message_ids=semantic.message_ids,
            text_message_ids=semantic.text_message_ids,
            text_role_counts=semantic.text_role_counts,
            attachment_count=semantic.attachment_count,
            word_count=semantic.word_count,
            has_attachments=semantic.attachment_count > 0,
            has_tool_use=semantic.tool_messages > 0,
            has_thinking=semantic.thinking_messages > 0,
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
            message_ids=tuple(_string_value(message.get("id")) for message in messages),
            text_message_ids=tuple(
                _string_value(message.get("id")) for message in messages if _string_value(message.get("text")).strip()
            ),
            text_role_counts=dict(roles),
            attachment_count=sum(1 for message in messages if bool(message.get("attachments"))),
            has_attachments=has_attachments,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
        )

    @classmethod
    def from_records(
        cls,
        conv_record: ConversationRecord,
        msg_records: list[MessageRecord],
        attachment_records: Sequence[AttachmentRecord] = (),
    ) -> ConversationFacts:
        """Extract facts from storage records (ConversationRecord + MessageRecords)."""
        roles = Counter(str(m.role) for m in msg_records)
        text_roles = Counter(str(m.role) for m in msg_records if (m.text or "").strip())
        has_tool_use = any(m.has_tool_use for m in msg_records)
        has_thinking = any(m.has_thinking for m in msg_records)
        return cls(
            conversation_id=str(conv_record.conversation_id),
            provider=str(conv_record.provider_name),
            title=conv_record.title,
            message_count=len(msg_records),
            role_multiset=dict(roles),
            message_ids=tuple(str(message.message_id) for message in msg_records),
            text_message_ids=tuple(str(message.message_id) for message in msg_records if (message.text or "").strip()),
            text_role_counts=dict(text_roles),
            attachment_count=len(attachment_records),
            word_count=sum(int(message.word_count or 0) for message in msg_records),
            has_attachments=bool(attachment_records),
            has_tool_use=bool(has_tool_use),
            has_thinking=bool(has_thinking),
        )

    def comparable_core(self) -> tuple[object, ...]:
        """Facts stable across record, hydrated, and surface representations."""
        return (
            self.conversation_id,
            self.provider,
            self.title,
            self.message_count,
            self.role_multiset,
            self.message_ids,
            self.text_message_ids,
            self.text_role_counts or {},
            self.attachment_count,
            self.has_attachments,
            self.has_tool_use,
            self.has_thinking,
        )


def assert_same_conversation_facts(*facts: ConversationFacts) -> None:
    """Assert all supplied conversation facts agree on archive semantics."""
    if len(facts) < 2:
        return
    expected = facts[0].comparable_core()
    mismatches = [fact for fact in facts[1:] if fact.comparable_core() != expected]
    assert not mismatches, f"Conversation facts disagree: expected={facts[0]!r} mismatches={mismatches!r}"


@dataclass(frozen=True)
class ArchiveFacts:
    """Aggregate facts about the archive, independent of surface."""

    total_conversations: int
    provider_counts: dict[str, int]
    total_messages: int
    conversation_ids: tuple[str, ...] = ()

    @classmethod
    def from_db_connection(cls, conn: sqlite3.Connection) -> ArchiveFacts:
        total_convs = int(conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0])
        provider_rows = conn.execute(
            "SELECT provider_name, COUNT(*) as cnt FROM conversations GROUP BY provider_name"
        ).fetchall()
        provider_counts = {str(row["provider_name"]): int(row["cnt"]) for row in provider_rows}
        total_msgs = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        conversation_ids = tuple(
            str(row["conversation_id"])
            for row in conn.execute("SELECT conversation_id FROM conversations ORDER BY conversation_id").fetchall()
        )
        return cls(
            total_conversations=total_convs,
            provider_counts=provider_counts,
            total_messages=total_msgs,
            conversation_ids=conversation_ids,
        )

    @classmethod
    def from_conversations(cls, conversations: Iterable[Conversation]) -> ArchiveFacts:
        conversations = tuple(conversations)
        provider_counts = Counter(str(conversation.provider) for conversation in conversations)
        return cls(
            total_conversations=len(conversations),
            provider_counts=dict(provider_counts),
            total_messages=sum(len(conversation.messages) for conversation in conversations),
            conversation_ids=tuple(sorted(str(conversation.id) for conversation in conversations)),
        )

    def comparable_core(self) -> tuple[object, ...]:
        return (
            self.total_conversations,
            self.provider_counts,
            self.total_messages,
            self.conversation_ids,
        )


def assert_same_archive_facts(*facts: ArchiveFacts) -> None:
    """Assert all supplied archive facts agree on aggregate archive semantics."""
    if len(facts) < 2:
        return
    expected = facts[0].comparable_core()
    mismatches = [fact for fact in facts[1:] if fact.comparable_core() != expected]
    assert not mismatches, f"Archive facts disagree: expected={facts[0]!r} mismatches={mismatches!r}"


__all__ = [
    "ArchiveFacts",
    "ConversationFacts",
    "assert_same_archive_facts",
    "assert_same_conversation_facts",
]
