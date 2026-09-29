"""Usage counters, Codex reference evidence, the semantic oracle and state chunk coordinates."""

import json
from pathlib import Path

import pytest

from polylogue.archive.message.types import MessageType
from polylogue.schemas.synthetic.wire_formats import (
    _parser_artifact_node_material_origin,
    _parser_artifact_node_message_type,
    _parser_artifact_node_role,
)
from polylogue.sources.parsers.claude import parse_code
from polylogue.sources.parsers.codex import parse as parse_codex
from polylogue.sources.parsers.codex_state import iter_codex_state_parts
from tests.infra.source_parser_cases import case, counter_payload, goals_db


@pytest.mark.parametrize(
    "wire,event_key",
    [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cached_input_tokens"),
        ("cache_creation_input_tokens", "cache_write_tokens"),
    ],
)
@pytest.mark.parametrize("value", [None, "unknown", "NaN", -1, True, 1.5])
def test_claude_invalid_counter_is_not_measured_zero(wire: str, event_key: str, value: object) -> None:
    """Invalid usage formerly became a measured zero/negative in usage events."""
    session = parse_code(counter_payload("claude_code", wire, value), "fallback")
    event = next(event for event in session.session_events if event.event_type == "message_usage")
    usage = event.payload["last_token_usage"]
    assert isinstance(usage, dict)
    assert event_key not in usage


def test_claude_numeric_zero_remains_measured() -> None:
    """Positive control: rejecting malformed counters must not erase a real zero."""
    session = parse_code(counter_payload("claude_code", "input_tokens", 0), "fallback")
    event = next(event for event in session.session_events if event.event_type == "message_usage")
    usage = event.payload["last_token_usage"]
    assert isinstance(usage, dict)
    assert usage["input_tokens"] == 0


@pytest.mark.parametrize(
    "wire,attribute",
    [
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("cache_read_input_tokens", "cache_read_tokens"),
        ("cache_creation_input_tokens", "cache_write_tokens"),
    ],
)
@pytest.mark.parametrize("value", [None, "unknown", "NaN", -1, True, 1.5])
def test_codex_sparse_counter_remains_unknown(wire: str, attribute: str, value: object) -> None:
    """A present null/nonnumeric field formerly flowed through the zero coercer."""
    session = parse_codex(counter_payload("codex", wire, value), "fallback")
    assert getattr(session.messages[0], attribute) is None


def test_codex_numeric_zero_remains_measured() -> None:
    """Positive control: an explicit native zero is not missing telemetry."""
    session = parse_codex(counter_payload("codex", "input_tokens", 0), "fallback")
    assert session.messages[0].input_tokens == 0


def test_local_image_reference_cannot_copy_inline_bytes_to_events() -> None:
    """String and mapping local_images references formerly copied the data URL."""
    session = parse_codex([*case("codex"), case("codex_image_event")], "fallback")
    rendered = json.dumps([event.model_dump(mode="json") for event in session.session_events])
    assert "data:image/png;base64,YWJjZA==" not in rendered
    assert "sha256_base64=" in rendered
    assert "acquired_bytes" in rendered


def test_codex_semantic_oracle_uses_protocol_precedence() -> None:
    """The oracle formerly called every developer node CONTEXT regardless of prose."""
    node = case("protocol_node")
    payload = case("codex")
    payload[1]["payload"] = node
    [message] = parse_codex(payload, "fallback").messages
    role = _parser_artifact_node_role("codex", node)
    assert message.message_type is MessageType.PROTOCOL
    assert _parser_artifact_node_message_type("codex", node) is message.message_type
    assert _parser_artifact_node_material_origin("codex", node, role) is message.material_origin


def test_state_chunk_coordinates_ignore_physical_rowid(tmp_path: Path) -> None:
    """Physically rebuilding an identical native row formerly renamed every chunk."""
    first = list(iter_codex_state_parts(goals_db(tmp_path / "first.db", 1), state_kind="goals", text_chars=4))
    second = list(iter_codex_state_parts(goals_db(tmp_path / "second.db", 99), state_kind="goals", text_chars=4))
    assert any(part.part_kind == "text_chunk" for part in first)
    assert [(part.thread_id, part.item_id, part.part_kind, part.payload) for part in first] == [
        (part.thread_id, part.item_id, part.part_kind, part.payload) for part in second
    ]
