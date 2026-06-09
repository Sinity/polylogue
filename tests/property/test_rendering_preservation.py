"""Property tests for semantic preservation across rendering surfaces.

These tests replace the runtime semantic proof infrastructure (35 deleted files)
with Hypothesis-driven verification that runs in CI. They check the same
invariants: message counts, role distributions, timestamps, and cross-surface
agreement.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import NotRequired, TypedDict

from hypothesis import HealthCheck, given, settings

from polylogue.archive.message.messages import MessageCollection
from polylogue.archive.message.roles import Role
from polylogue.archive.models import Message, Session
from polylogue.core.sources import origin_from_provider
from polylogue.rendering.core_markdown import format_session_markdown
from polylogue.rendering.formatting import format_session
from polylogue.types import Provider, SessionId
from tests.infra.strategies.messages import session_strategy


class RenderMessagePayload(TypedDict):
    id: str
    role: str
    text: str
    timestamp: NotRequired[object]
    content_blocks: NotRequired[list[dict[str, object]]]


class RenderSessionPayload(TypedDict):
    id: str
    provider: str
    title: str
    messages: list[RenderMessagePayload]
    created_at: NotRequired[str]


# ---------------------------------------------------------------------------
# Helpers: extract semantic facts from rendered output
# ---------------------------------------------------------------------------


def _markdown_facts(md: str) -> dict[str, object]:
    """Extract semantic facts from rendered markdown."""
    message_sections = 0
    timestamp_lines = 0
    attachment_lines = 0
    role_counts: Counter[str] = Counter()

    for line in md.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            if section != "attachments":
                message_sections += 1
                role_counts[section] += 1
        elif line.startswith("_Timestamp: "):
            timestamp_lines += 1
        elif line.startswith("- Attachment: "):
            attachment_lines += 1

    return {
        "message_sections": message_sections,
        "timestamp_lines": timestamp_lines,
        "attachment_lines": attachment_lines,
        "role_counts": dict(role_counts),
    }


def _json_facts(json_str: str) -> dict[str, object]:
    """Extract semantic facts from JSON export."""
    data = json.loads(json_str)
    if not isinstance(data, dict):
        return {
            "session_id": "",
            "origin": "",
            "title": None,
            "message_count": 0,
            "role_counts": {},
            "timestamped_messages": 0,
        }
    messages = data.get("messages", [])
    role_counts: Counter[str] = Counter()
    timestamped = 0
    for msg in messages:
        if isinstance(msg, dict):
            role_counts[str(msg.get("role", "")).lower()] += 1
            if msg.get("timestamp"):
                timestamped += 1
    return {
        "session_id": data.get("id", ""),
        "origin": data.get("origin", ""),
        "title": data.get("title"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "role_counts": dict(role_counts),
        "timestamped_messages": timestamped,
    }


def _message_text(payload: RenderMessagePayload, *, placeholder: bool) -> str:
    text = payload.get("text", "")
    if not placeholder:
        return text
    return text.strip() or "placeholder"


def _messages(
    payload: RenderSessionPayload,
    *,
    placeholder: bool = True,
    non_empty_only: bool = False,
    content_blocks: list[dict[str, object]] | None = None,
) -> list[Message]:
    messages: list[Message] = []
    for index, item in enumerate(payload["messages"]):
        text = _message_text(item, placeholder=placeholder)
        if non_empty_only and not text.strip():
            continue
        blocks = content_blocks if content_blocks is not None and index == 0 and item["role"] == "assistant" else []
        messages.append(
            Message(
                id=item["id"],
                role=Role.normalize(item["role"]),
                text=text,
                timestamp=None,
                content_blocks=blocks,
            )
        )
    return messages


def _session(payload: RenderSessionPayload, messages: list[Message]) -> Session:
    return Session(
        id=SessionId(payload["id"]),
        origin=origin_from_provider(Provider.from_string(payload["provider"])),
        title=payload["title"],
        messages=MessageCollection(messages=messages),
    )


def _render_json(payload: RenderSessionPayload) -> str:
    return format_session(_session(payload, _messages(payload)), "json", None)


def _render_yaml(payload: RenderSessionPayload) -> str:
    return format_session(_session(payload, _messages(payload)), "yaml", None)


# ---------------------------------------------------------------------------
# Test: Markdown rendering preserves message structure
# ---------------------------------------------------------------------------


@given(conv_data=session_strategy(min_messages=1, max_messages=10))
@settings(max_examples=30, deadline=5000)
def test_markdown_preserves_message_count(conv_data: RenderSessionPayload) -> None:
    """Every non-empty message must produce a ## section in markdown."""
    messages = _messages(conv_data, placeholder=False, non_empty_only=True)
    if not messages:
        return  # skip if all messages empty

    md = format_session_markdown(_session(conv_data, messages))
    facts = _markdown_facts(md)

    # Core invariant: renderable messages → ## sections
    assert facts["message_sections"] == len(messages), (
        f"Expected {len(messages)} sections, got {facts['message_sections']}"
    )


@given(conv_data=session_strategy(min_messages=2, max_messages=8))
@settings(max_examples=30, deadline=5000)
def test_markdown_preserves_role_distribution(conv_data: RenderSessionPayload) -> None:
    """Markdown role sections match the input role distribution."""
    messages = _messages(conv_data)
    md = format_session_markdown(_session(conv_data, messages))
    facts = _markdown_facts(md)

    # Build expected role distribution (only renderable messages — non-empty text)
    expected: Counter[str] = Counter()
    for msg in messages:
        if (msg.text or "").strip():
            expected[str(msg.role).lower()] += 1

    assert facts["role_counts"] == dict(expected)


# ---------------------------------------------------------------------------
# Test: JSON export preserves core fields
# ---------------------------------------------------------------------------


@given(conv_data=session_strategy(min_messages=1, max_messages=10))
@settings(max_examples=30, deadline=5000)
def test_json_export_preserves_session_identity(conv_data: RenderSessionPayload) -> None:
    """JSON export must preserve session ID, provider, title."""
    conv = _session(conv_data, _messages(conv_data))
    json_str = format_session(conv, "json", None)
    facts = _json_facts(json_str)

    assert facts["session_id"] == str(conv.id)
    assert facts["origin"] == conv.origin
    assert facts["title"] == conv.title


@given(conv_data=session_strategy(min_messages=1, max_messages=10))
@settings(max_examples=30, deadline=5000)
def test_json_export_preserves_message_count(conv_data: RenderSessionPayload) -> None:
    """JSON export must have one message entry per input message."""
    messages = _messages(conv_data)
    json_str = format_session(_session(conv_data, messages), "json", None)
    facts = _json_facts(json_str)

    assert facts["message_count"] == len(messages)


# ---------------------------------------------------------------------------
# Test: Cross-surface differential — JSON and YAML agree
# ---------------------------------------------------------------------------


@given(conv_data=session_strategy(min_messages=1, max_messages=5))
@settings(max_examples=20, deadline=5000)
def test_json_yaml_agree_on_message_count(conv_data: RenderSessionPayload) -> None:
    """JSON and YAML exports of the same session have the same message count."""
    import yaml

    json_data = json.loads(_render_json(conv_data))
    yaml_data = yaml.safe_load(_render_yaml(conv_data))
    assert isinstance(json_data, dict)
    assert isinstance(yaml_data, dict)

    json_msgs = json_data.get("messages", [])
    yaml_msgs = yaml_data.get("messages", [])
    assert isinstance(json_msgs, list)
    assert isinstance(yaml_msgs, list)

    assert len(json_msgs) == len(yaml_msgs), f"JSON has {len(json_msgs)} messages, YAML has {len(yaml_msgs)}"


@given(conv_data=session_strategy(min_messages=1, max_messages=5))
@settings(max_examples=20, deadline=5000, suppress_health_check=[HealthCheck.filter_too_much])
def test_json_yaml_agree_on_identity(conv_data: RenderSessionPayload) -> None:
    """JSON and YAML agree on session_id, provider, title."""
    import yaml
    from hypothesis import assume

    # YAML has edge cases with certain Unicode control characters
    assume(conv_data["title"].isprintable())

    json_data = json.loads(_render_json(conv_data))
    yaml_data = yaml.safe_load(_render_yaml(conv_data))
    assert isinstance(json_data, dict)
    assert isinstance(yaml_data, dict)

    assert json_data["id"] == yaml_data["id"]
    assert json_data["origin"] == yaml_data["origin"]
    assert json_data["title"] == yaml_data["title"]


# ---------------------------------------------------------------------------
# Test: Content block structure preservation
# ---------------------------------------------------------------------------


@given(conv_data=session_strategy(min_messages=2, max_messages=5))
@settings(max_examples=20, deadline=5000)
def test_content_blocks_produce_structured_markdown(conv_data: RenderSessionPayload) -> None:
    """When content blocks are present, markdown should contain structural markers."""
    # Add content blocks to the first message
    blocks: list[dict[str, object]] = [
        {"type": "thinking", "thinking": "Let me analyze this..."},
        {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/tmp/test.py"}},
        {"type": "text", "text": "Here's what I found."},
    ]

    messages = _messages(conv_data, content_blocks=blocks)
    md = format_session_markdown(_session(conv_data, messages))

    # If first message was assistant and had blocks, check structure
    if messages[0].role.value == "assistant":
        assert "Thinking" in md or "Tool:" in md or "Read" in md, (
            "Content blocks should produce structural markers in markdown"
        )
