"""Round-trip laws for domain-derived session surface envelopes."""

from __future__ import annotations

from datetime import UTC, datetime

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.session.domain_models import Session
from polylogue.core.enums import Origin, TitleSource
from polylogue.core.types import SessionId
from polylogue.surfaces.payloads import (
    _SESSION_LIST_MASK,
    _SESSION_SUMMARY_MASK,
    SessionDetailEnvelope,
    SessionListEnvelope,
    SessionSummaryEnvelope,
    session_detail_envelope_from_domain,
    session_list_envelope_from_domain,
    session_summary_envelope_from_domain,
)


def _build_session() -> Session:
    message = Message(
        id="m1",
        role=Role.USER,
        text="hello",
        timestamp=datetime(2026, 5, 27, 10, 0, tzinfo=UTC),
        duration_ms=17,
        stop_reason="end_turn",
    )
    return Session(
        id=SessionId("codex-session:c1"),
        origin=Origin.CODEX_SESSION,
        title="A canonical session",
        title_source=TitleSource.HEURISTIC,
        messages=MessageCollection(messages=[message]),
    )


def test_session_envelopes_are_pydantic_models_with_compatible_wire_fields() -> None:
    session = _build_session()
    summary = session_summary_envelope_from_domain(session)
    detail = session_detail_envelope_from_domain(session)
    row = session_list_envelope_from_domain(session)

    assert isinstance(summary, SessionSummaryEnvelope)
    assert isinstance(detail, SessionDetailEnvelope)
    assert isinstance(row, SessionListEnvelope)
    assert summary.model_dump(mode="json")["origin"] == "codex-session"
    assert detail.messages[0].duration_ms == 17
    assert detail.messages[0].stop_reason == "end_turn"
    assert row.selected({"id", "title"}) == {"id": "codex-session:c1", "title": "A canonical session"}


def test_session_masks_cover_the_domain_fields_used_by_each_surface() -> None:
    session = _build_session()
    dumped = session.model_dump(mode="python")
    summary = session_summary_envelope_from_domain(session)
    row = session_list_envelope_from_domain(session)

    for domain_name, surface_name in _SESSION_SUMMARY_MASK:
        if domain_name == "message_count":
            assert summary.message_count == len(session.messages)
        else:
            expected = dumped[domain_name]
            if domain_name == "origin":
                expected = expected.value
            elif domain_name == "title_source":
                expected = expected.value if expected else None
            assert getattr(summary, surface_name) == expected

    for domain_name, surface_name in _SESSION_LIST_MASK:
        if domain_name == "title":
            continue
        expected = dumped[domain_name]
        if domain_name == "origin":
            expected = expected.value
        elif domain_name in {"created_at", "updated_at"}:
            expected = expected.isoformat() if expected else None
        elif domain_name == "title_source":
            expected = expected.value if expected else None
        assert getattr(row, surface_name) == expected
