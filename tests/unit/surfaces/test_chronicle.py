from __future__ import annotations

from datetime import UTC, datetime

from polylogue.archive.message.models import Message
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.domain_models import SessionSummary
from polylogue.core.enums import MaterialOrigin
from polylogue.surfaces.chronicle import (
    build_chronicle_projection_payload,
    build_chronicle_session_payload,
    chronicle_json_document,
    render_chronicle_markdown,
)


def _message(
    message_id: str,
    text: str,
    *,
    role: str = "assistant",
    minute: int = 0,
    material_origin: MaterialOrigin | None = None,
) -> Message:
    if material_origin is None:
        material_origin = MaterialOrigin.HUMAN_AUTHORED if role == "user" else MaterialOrigin.ASSISTANT_AUTHORED
    return Message(
        id=message_id,
        role=Role.normalize(role),
        text=text,
        timestamp=datetime(2026, 6, 30, 8, minute, tzinfo=UTC),
        message_type=MessageType.MESSAGE,
        material_origin=material_origin,
    )


def test_chronicle_session_payload_dedupes_edges_and_counts_omissions() -> None:
    summary = SessionSummary.model_validate(
        {
            "id": "chatgpt-export:abc",
            "origin": "chatgpt-export",
            "title": "Large project session",
        }
    )

    payload = build_chronicle_session_payload(
        summary,
        first_messages=[_message("m1", "first", role="user"), _message("m2", "second", minute=1)],
        last_messages=[_message("m2", "second", minute=1), _message("m5", "last", minute=5)],
        total_matching_messages=5,
        edge_limit=2,
    )

    assert payload.session_id == "chatgpt-export:abc"
    assert payload.title == "Large project session"
    assert [message.message_id for message in payload.first_messages] == ["m1", "m2"]
    assert [message.message_id for message in payload.last_messages] == ["m5"]
    assert payload.included_count == 3
    assert payload.omitted_count == 2
    assert "middle_messages_omitted" in payload.caveats


def test_chronicle_session_payload_omits_non_authored_rows() -> None:
    summary = SessionSummary.model_validate(
        {
            "id": "chatgpt-export:abc",
            "origin": "chatgpt-export",
            "title": "Large project session",
        }
    )

    payload = build_chronicle_session_payload(
        summary,
        first_messages=[
            _message("m1", '{"queries":["find context"]}', material_origin=MaterialOrigin.RUNTIME_PROTOCOL),
            _message("m2", "human-readable assistant prose"),
        ],
        last_messages=[
            _message("m3", "bash -lc ls -lah", material_origin=MaterialOrigin.OPERATOR_COMMAND),
            _message("m4", "final prose"),
        ],
        total_matching_messages=4,
        edge_limit=2,
    )

    assert [message.message_id for message in payload.first_messages] == ["m2"]
    assert [message.message_id for message in payload.last_messages] == ["m4"]
    assert "non_authored_or_empty_messages_omitted" in payload.caveats


def test_chronicle_markdown_and_json_report_body_policy_and_counts() -> None:
    summary = SessionSummary.model_validate(
        {
            "id": "codex-session:def",
            "origin": "codex-session",
            "title": "Devloop",
        }
    )
    session = build_chronicle_session_payload(
        summary,
        first_messages=[_message("m1", "start", role="user")],
        last_messages=[_message("m4", "finish", minute=4)],
        total_matching_messages=4,
        edge_limit=1,
    )
    projection = build_chronicle_projection_payload([session], edge_limit=1)

    document = chronicle_json_document(projection)
    assert document["chronicle"]["body_policy"] == "authored-dialogue"
    assert document["chronicle"]["sessions"][0]["omitted_count"] == 2

    rendered = render_chronicle_markdown(projection)
    assert "# Session Chronicle" in rendered
    assert "- Body policy: authored-dialogue" in rendered
    assert "- Omitted middle messages: 2" in rendered
    assert "2026-06-30T08:00:00+00:00 - user / message" in rendered
