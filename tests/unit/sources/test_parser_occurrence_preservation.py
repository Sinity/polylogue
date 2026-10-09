"""Schema-admitted synthetic parser laws for native identity and occurrence survival."""

from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from polylogue.core.enums import BlockType, Provider
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import parse_payload, parse_stream_payload
from polylogue.sources.parsers.codex import parse
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.index_writer import write_fixture_index_session
from tests.infra.storage_records import db_setup

_ROOT = Path(__file__).parents[3]


def _records(provider: str, name: str) -> list[object]:
    fixture = _ROOT / "tests" / "fixtures" / provider / name
    schema = (
        _ROOT
        / "polylogue"
        / "schemas"
        / "providers"
        / provider
        / "versions"
        / "v5"
        / "elements"
        / "session_record_stream.schema.json.gz"
    )
    validator = Draft202012Validator(json.loads(gzip.decompress(schema.read_bytes())))
    records = [json.loads(line) for line in fixture.read_text().splitlines()]
    for record in records:
        validator.validate(record)
    return records


@pytest.mark.parametrize("streamed", [False, True])
def test_codex_survives_occurrences_native_reasoning_and_code(streamed: bool) -> None:
    records = _records("codex", "occurrence-preservation.jsonl")
    sessions = (
        parse_stream_payload(Provider.CODEX, iter(records), "neutral")
        if streamed
        else parse_payload(Provider.CODEX, records, "neutral")
    )
    session = sessions[0]
    assert [m.provider_message_id for m in session.messages] == [
        "message-one",
        "prompt-two",
        "reasoning-one",
        "code-one",
    ]
    assert [m.text for m in session.messages[:2]] == ["continue", "continue"]
    assert session.messages[1].timestamp == "2026-01-01T01:00:00Z"
    assert session.messages[2].blocks[0].type is BlockType.THINKING
    assert session.messages[3].blocks[0].type is BlockType.CODE
    assert session.messages[3].blocks[0].text == "print(1)"
    assert session.updated_at == "2026-01-01T01:00:02Z"


@pytest.mark.parametrize("streamed", [False, True])
def test_claude_extrema_compare_instants_and_retain_wire_offsets(streamed: bool) -> None:
    records = _records("claude-code", "timezone-extrema.jsonl")
    sessions = (
        parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "neutral")
        if streamed
        else parse_payload(Provider.CLAUDE_CODE, records, "neutral")
    )
    session = sessions[0]
    assert session.created_at == "2026-01-01T10:00:00+02:00"
    assert session.updated_at == "2026-01-01T09:00:00+00:00"
    assert [m.timestamp for m in session.messages] == [session.created_at, session.updated_at]


def test_codex_native_reasoning_event_links_to_persisted_message(workspace_env: Mapping[str, Path]) -> None:
    session = parse(_records("codex", "occurrence-preservation.jsonl"), "neutral")
    with open_connection(db_setup(workspace_env)) as conn:
        write_fixture_index_session(conn, session, content_hash=session_content_hash(session))
        row = conn.execute(
            "SELECT e.source_message_id, m.native_id FROM session_events e LEFT JOIN messages m ON m.message_id = e.source_message_id WHERE e.event_type = 'reasoning'"
        ).fetchone()
        assert row is not None
        assert row["native_id"] == "reasoning-one"
        assert row["source_message_id"] is not None


@pytest.mark.parametrize("evidence", ["instant", "turn", "native_id", "none"])
def test_codex_mirror_matches_only_one_proven_occurrence(evidence: str) -> None:
    event = {"type": "user_message", "client_id": "prompt-one", "message": "continue"}
    response = {
        "type": "message",
        "id": "message-one",
        "role": "user",
        "content": [{"type": "input_text", "text": "continue"}],
    }
    if evidence == "instant":
        event["timestamp"] = "2026-01-01T01:00:00+01:00"
        response["timestamp"] = "2026-01-01T00:00:00Z"
    elif evidence == "turn":
        event["turn_id"] = response["turn_id"] = "turn-one"
    elif evidence == "native_id":
        response["id"] = "prompt-one"
    records = [
        {"type": "event_msg", "payload": event},
        {"type": "event_msg", "payload": {**event, "client_id": "prompt-two"}},
        {"type": "response_item", "payload": response},
    ]
    session = parse(records, "neutral")
    assert len(session.messages) == (3 if evidence == "none" else 2)
    assert any(m.provider_message_id == "prompt-two" for m in session.messages)


def test_codex_different_formatting_does_not_prove_a_mirror() -> None:
    records = [
        {
            "type": "event_msg",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"type": "user_message", "message": "x\n  y"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x y"}]},
        },
    ]
    assert [m.text for m in parse(records, "neutral").messages] == ["x\n  y", "x y"]
