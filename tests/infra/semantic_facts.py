"""Semantic fact extraction for cross-surface agreement testing.

Normalizes sessions, query results, and insights into stable
semantic fact tuples so tests can compare meaning rather than formatting.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from polylogue.archive.models import Session
from polylogue.archive.semantic.facts import build_session_semantic_facts
from polylogue.core.enums import Origin
from polylogue.core.json import JSONDocument, json_document_list
from polylogue.core.sources import provider_from_origin
from polylogue.storage.runtime import AttachmentRecord, MessageRecord, SessionRecord
from polylogue.types import Provider


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _provider_token(token: str) -> str:
    """Canonicalize a provider-or-origin token to its provider token.

    The cross-surface oracle compares provenance in provider-token vocabulary
    (matching storage ``source_name`` and the archive reverse projection). The
    domain model and semantic facts now expose origin tokens, so origin
    inputs are reversed to their canonical provider (#1743 transition).
    """
    if not token:
        return ""
    try:
        origin = Origin(token)
    except ValueError:
        return str(Provider.from_string(token))
    return str(provider_from_origin(origin))


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _content_blocks(message: JSONDocument) -> list[JSONDocument]:
    blocks = message.get("content_blocks")
    return json_document_list(blocks)


@dataclass(frozen=True)
class SessionFacts:
    """Stable semantic facts about a session, independent of surface."""

    session_id: str
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
    def from_domain_session(cls, conv: Session) -> SessionFacts:
        """Extract facts from a domain Session object."""
        messages = list(conv.messages)
        semantic = build_session_semantic_facts(conv)
        roles = Counter(str(m.role) for m in messages)
        return cls(
            session_id=semantic.session_id,
            provider=_provider_token(str(semantic.origin)),
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
    def from_json_payload(cls, payload: JSONDocument) -> SessionFacts:
        """Extract facts from a CLI JSON output payload."""
        messages = json_document_list(payload.get("messages"))
        roles = Counter(_string_value(message.get("role")) or "unknown" for message in messages)
        has_tool_use = any(
            any(block.get("type") in ("tool_use", "tool_result") for block in _content_blocks(message))
            for message in messages
        )
        has_thinking = any(
            any(block.get("type") == "thinking" for block in _content_blocks(message)) for message in messages
        )
        has_attachments = any(bool(message.get("attachments")) for message in messages)
        return cls(
            session_id=_string_value(payload.get("id")) or _string_value(payload.get("session_id")),
            provider=_provider_token(
                _string_value(payload.get("provider"))
                or _string_value(payload.get("origin"))
                or _string_value(payload.get("source_name"))
            ),
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
        conv_record: SessionRecord,
        msg_records: list[MessageRecord],
        attachment_records: Sequence[AttachmentRecord] = (),
    ) -> SessionFacts:
        """Extract facts from storage records (SessionRecord + MessageRecords)."""
        roles = Counter(str(m.role) for m in msg_records)
        text_roles = Counter(str(m.role) for m in msg_records if (m.text or "").strip())
        has_tool_use = any(m.has_tool_use for m in msg_records)
        has_thinking = any(m.has_thinking for m in msg_records)
        return cls(
            session_id=str(conv_record.session_id),
            provider=_provider_token(str(conv_record.source_name)),
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
            self.session_id,
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


def assert_same_session_facts(*facts: SessionFacts) -> None:
    """Assert all supplied session facts agree on archive semantics."""
    if len(facts) < 2:
        return
    expected = facts[0].comparable_core()
    mismatches = [fact for fact in facts[1:] if fact.comparable_core() != expected]
    assert not mismatches, f"Session facts disagree: expected={facts[0]!r} mismatches={mismatches!r}"


@dataclass(frozen=True)
class ArchiveFacts:
    """Aggregate facts about the archive, independent of surface."""

    total_sessions: int
    provider_counts: dict[str, int]
    total_messages: int
    session_ids: tuple[str, ...] = ()

    @classmethod
    def from_db_connection(cls, conn: sqlite3.Connection) -> ArchiveFacts:
        """Aggregate archive facts from a ``index.db``.

        Reads archive `sessions` / ``messages``. The provider is recovered
        from the session ``origin`` so the per-provider counts agree with the
        domain ``Session.provider`` projection used by the facade surface.
        """
        from polylogue.api.archive import _provider_for_archive_origin

        total_convs = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        provider_rows = conn.execute("SELECT origin, COUNT(*) as cnt FROM sessions GROUP BY origin").fetchall()
        provider_counts: dict[str, int] = {}
        for row in provider_rows:
            provider = str(_provider_for_archive_origin(str(row["origin"])))
            provider_counts[provider] = provider_counts.get(provider, 0) + int(row["cnt"])
        total_msgs = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
        session_ids = tuple(
            str(row["session_id"])
            for row in conn.execute("SELECT session_id FROM sessions ORDER BY session_id").fetchall()
        )
        return cls(
            total_sessions=total_convs,
            provider_counts=provider_counts,
            total_messages=total_msgs,
            session_ids=session_ids,
        )

    @classmethod
    def from_sessions(cls, sessions: Iterable[Session]) -> ArchiveFacts:
        sessions = tuple(sessions)
        provider_counts = Counter(_provider_token(str(session.origin)) for session in sessions)
        return cls(
            total_sessions=len(sessions),
            provider_counts=dict(provider_counts),
            total_messages=sum(len(session.messages) for session in sessions),
            session_ids=tuple(sorted(str(session.id) for session in sessions)),
        )

    def comparable_core(self) -> tuple[object, ...]:
        return (
            self.total_sessions,
            self.provider_counts,
            self.total_messages,
            self.session_ids,
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
    "SessionFacts",
    "assert_same_archive_facts",
    "assert_same_session_facts",
]
