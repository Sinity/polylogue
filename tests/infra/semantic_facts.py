"""Semantic fact extraction for cross-surface agreement testing.

Normalizes sessions, query results, and insights into stable
semantic fact tuples so tests can compare meaning rather than formatting.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from polylogue.archive.models import Session
from polylogue.archive.semantic.facts import build_session_semantic_facts
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument, json_document_list
from polylogue.core.sources import provider_from_origin
from polylogue.storage.runtime import AttachmentRecord, MessageRecord, SessionRecord


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


def _object_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        if isinstance(dumped, Mapping):
            return dumped
    raise AssertionError(f"{context} is not a mapping or model payload: {type(value).__name__}")


def _required_string(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{context} is not a non-empty string")
    return value


def _required_int(value: object, *, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"{context} is not an integer")
    return value


def _optional_number(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AssertionError(f"{context} is not numeric or null")
    return float(value)


def _timestamp_millis(value: object, *, context: str, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise AssertionError(f"{context} is required")
        return None
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{context} is not an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AssertionError(f"{context} is not an ISO timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


@dataclass(frozen=True)
class MaterializationProvenanceFacts:
    """Public materialization provenance normalized by meaning, not encoding."""

    materializer_version: int
    materialized_at_ms: int
    source_updated_at_ms: int | None
    source_sort_key: float | None
    input_high_water_mark_ms: int | None
    input_high_water_mark_source: str | None
    time_confidence: str

    @classmethod
    def from_value(cls, value: object, *, context: str = "provenance") -> MaterializationProvenanceFacts:
        payload = _object_mapping(value, context=context)
        materialized_at_ms = _timestamp_millis(
            payload.get("materialized_at"),
            context=f"{context}.materialized_at",
            required=True,
        )
        assert materialized_at_ms is not None
        source = payload.get("input_high_water_mark_source")
        if source is not None and not isinstance(source, str):
            raise AssertionError(f"{context}.input_high_water_mark_source is not a string or null")
        return cls(
            materializer_version=_required_int(
                payload.get("materializer_version"), context=f"{context}.materializer_version"
            ),
            materialized_at_ms=materialized_at_ms,
            source_updated_at_ms=_timestamp_millis(
                payload.get("source_updated_at"), context=f"{context}.source_updated_at"
            ),
            source_sort_key=_optional_number(payload.get("source_sort_key"), context=f"{context}.source_sort_key"),
            input_high_water_mark_ms=_timestamp_millis(
                payload.get("input_high_water_mark"), context=f"{context}.input_high_water_mark"
            ),
            input_high_water_mark_source=source,
            time_confidence=_required_string(payload.get("time_confidence"), context=f"{context}.time_confidence"),
        )


@dataclass(frozen=True)
class SessionProfileFacts:
    """One public session-profile fact set shared by stable read surfaces.

    The selected algebra deliberately keeps the public ``origin`` token and
    normalizes temporal strings to epoch milliseconds. Surface-local envelope
    names and JSON formatting are not part of the comparison.
    """

    session_id: str
    logical_session_id: str
    origin: str
    title: str | None
    message_count: int
    substantive_count: int
    attachment_count: int
    tool_use_count: int
    thinking_count: int
    word_count: int
    provenance: MaterializationProvenanceFacts

    @classmethod
    def _from_payloads(
        cls,
        identity: object,
        evidence: object,
        provenance: object,
        *,
        context: str,
    ) -> SessionProfileFacts:
        identity_payload = _object_mapping(identity, context=f"{context}.identity")
        evidence_payload = _object_mapping(evidence, context=f"{context}.evidence")
        title = identity_payload.get("title")
        if title is not None and not isinstance(title, str):
            raise AssertionError(f"{context}.title is not a string or null")
        return cls(
            session_id=_required_string(identity_payload.get("session_id"), context=f"{context}.session_id"),
            logical_session_id=_required_string(
                identity_payload.get("logical_session_id"), context=f"{context}.logical_session_id"
            ),
            origin=_required_string(identity_payload.get("origin"), context=f"{context}.origin"),
            title=title,
            message_count=_required_int(evidence_payload.get("message_count"), context=f"{context}.message_count"),
            substantive_count=_required_int(
                evidence_payload.get("substantive_count"), context=f"{context}.substantive_count"
            ),
            attachment_count=_required_int(
                evidence_payload.get("attachment_count"), context=f"{context}.attachment_count"
            ),
            tool_use_count=_required_int(evidence_payload.get("tool_use_count"), context=f"{context}.tool_use_count"),
            thinking_count=_required_int(evidence_payload.get("thinking_count"), context=f"{context}.thinking_count"),
            word_count=_required_int(evidence_payload.get("word_count"), context=f"{context}.word_count"),
            provenance=MaterializationProvenanceFacts.from_value(provenance, context=f"{context}.provenance"),
        )

    @classmethod
    def from_insight(cls, insight: object, *, context: str = "session profile insight") -> SessionProfileFacts:
        evidence = getattr(insight, "evidence", None)
        provenance = getattr(insight, "provenance", None)
        if evidence is None:
            raise AssertionError(f"{context} omitted evidence")
        if provenance is None:
            raise AssertionError(f"{context} omitted provenance")
        return cls._from_payloads(insight, evidence, provenance, context=context)

    @classmethod
    def from_insight_payload(
        cls,
        payload: Mapping[str, object],
        *,
        context: str = "session profile JSON",
    ) -> SessionProfileFacts:
        evidence = payload.get("evidence")
        provenance = payload.get("provenance")
        if evidence is None:
            raise AssertionError(f"{context} omitted evidence")
        if provenance is None:
            raise AssertionError(f"{context} omitted provenance")
        return cls._from_payloads(payload, evidence, provenance, context=context)

    @classmethod
    def from_daemon_payload(
        cls,
        payload: Mapping[str, object],
        *,
        context: str = "daemon profile response",
    ) -> SessionProfileFacts:
        kinds = _object_mapping(payload.get("kinds"), context=f"{context}.kinds")
        panel = _object_mapping(kinds.get("profile"), context=f"{context}.kinds.profile")
        if panel.get("materialized") is not True or panel.get("readiness_tag") != "q-ready":
            raise AssertionError(f"{context} did not return a materialized q-ready profile")
        profile = panel.get("profile")
        provenance = panel.get("provenance")
        if profile is None:
            raise AssertionError(f"{context} omitted profile")
        if provenance is None:
            raise AssertionError(f"{context} omitted provenance")
        return cls._from_payloads(profile, profile, provenance, context=context)


def assert_same_session_profile_facts(*facts: SessionProfileFacts) -> None:
    """Assert all supplied profile projections preserve the selected algebra."""
    if len(facts) < 2:
        return
    expected = facts[0]
    mismatches = [fact for fact in facts[1:] if fact != expected]
    assert not mismatches, f"Session profile facts disagree: expected={expected!r} mismatches={mismatches!r}"


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
            provider=_provider_token(conv_record.origin.value),
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

        Reads archive `sessions` / ``messages``. Legacy scenario accounting
        uses the provider-wire token recovered from the canonical origin.
        """
        from polylogue.core.enums import Origin
        from polylogue.core.sources import provider_from_origin

        total_convs = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
        provider_rows = conn.execute("SELECT origin, COUNT(*) as cnt FROM sessions GROUP BY origin").fetchall()
        provider_counts: dict[str, int] = {}
        for row in provider_rows:
            provider = str(provider_from_origin(Origin.from_string(str(row["origin"]))))
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
    "MaterializationProvenanceFacts",
    "SessionFacts",
    "SessionProfileFacts",
    "assert_same_archive_facts",
    "assert_same_session_facts",
    "assert_same_session_profile_facts",
]
