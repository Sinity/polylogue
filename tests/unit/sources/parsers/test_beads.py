from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MaterialOrigin, Provider
from polylogue.core.json import JSONValue
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers import beads


def _fixture_payload() -> list[JSONValue]:
    path = Path(__file__).parents[3] / "fixtures" / "beads" / "issue-interactions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_parse_groups_and_sorts_issue_timeline_with_structured_events() -> None:
    payload = _fixture_payload()

    sessions = beads.parse(payload, "ignored")

    assert len(sessions) == 1
    session = sessions[0]
    assert session.source_name is Provider.BEADS
    assert session.provider_session_id == "polylogue-7fj"
    assert session.title == "Beads issue polylogue-7fj"
    assert session.created_at == "2026-07-08T20:14:35.726797203Z"
    assert session.updated_at == "2026-07-08T20:14:36.726797203Z"
    assert [message.provider_message_id for message in session.messages] == ["int-priority", "int-close"]
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.role for message in session.messages] == [Role.USER, Role.USER]
    assert all(message.material_origin is MaterialOrigin.RUNTIME_PROTOCOL for message in session.messages)
    assert all(message.blocks[0].type is BlockType.TEXT for message in session.messages)
    assert session.messages[-1].is_active_leaf is True
    assert session.active_leaf_message_provider_id == "int-close"
    assert [event.event_type for event in session.session_events] == ["beads_field_change", "beads_field_change"]
    assert session.session_events[-1].payload["extra"] == {
        "field": "status",
        "old_value": "in_progress",
        "new_value": "closed",
        "reason": "Focused parser checks passed.",
    }
    message_text = session.messages[-1].text
    assert message_text is not None
    assert 'status from "in_progress" to "closed"' in message_text


def test_parse_rejects_mixed_issue_timeline() -> None:
    payload = _fixture_payload()
    second = payload[1]
    assert isinstance(second, dict)
    payload[1] = {**second, "issue_id": "polylogue-other"}

    with pytest.raises(ValueError, match="exactly one issue_id"):
        beads.parse_issue_timeline(payload, "polylogue-7fj")


def test_dispatch_detects_and_parses_interaction_ledger() -> None:
    payload = _fixture_payload()

    detected = detect_provider(payload)
    sessions = parse_payload(Provider.BEADS, payload, "ledger")

    assert detected is Provider.BEADS
    assert [session.provider_session_id for session in sessions] == ["polylogue-7fj"]


def test_workspace_namespace_prevents_cross_repository_issue_collisions(tmp_path: Path) -> None:
    payload = _fixture_payload()
    first_path = tmp_path / "first" / ".beads" / "interactions.jsonl"
    second_path = tmp_path / "second" / ".beads" / "interactions.jsonl"
    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)

    [first] = beads.parse(payload, "ignored", source_path=str(first_path))
    [first_replay] = beads.parse(payload, "ignored", source_path=str(first_path))
    [second] = beads.parse(payload, "ignored", source_path=str(second_path))

    assert first.provider_session_id == first_replay.provider_session_id
    assert first.provider_session_id != second.provider_session_id
    assert first.working_directories == [str(first_path.parent.parent.resolve())]
