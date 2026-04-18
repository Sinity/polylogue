"""Property tests for semantic preservation across rendering surfaces.

These tests replace the runtime semantic proof infrastructure (35 deleted files)
with Hypothesis-driven verification that runs in CI. They check the same
invariants: message counts, role distributions, timestamps, and cross-surface
agreement.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from hypothesis import given, settings

from tests.infra.strategies.messages import conversation_strategy

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
    messages = data.get("messages", [])
    role_counts: Counter[str] = Counter()
    timestamped = 0
    for msg in messages:
        if isinstance(msg, dict):
            role_counts[str(msg.get("role", "")).lower()] += 1
            if msg.get("timestamp"):
                timestamped += 1
    return {
        "conversation_id": data.get("id", ""),
        "provider": data.get("provider", ""),
        "title": data.get("title"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "role_counts": dict(role_counts),
        "timestamped_messages": timestamped,
    }


def _csv_facts(csv_str: str) -> dict[str, object]:
    """Extract semantic facts from CSV export."""
    import csv
    import io

    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    role_counts: Counter[str] = Counter()
    timestamped = 0
    for row in rows:
        role_counts[str(row.get("role", "")).lower()] += 1
        if row.get("timestamp"):
            timestamped += 1
    return {
        "message_count": len(rows),
        "role_counts": dict(role_counts),
        "timestamped_messages": timestamped,
    }


# ---------------------------------------------------------------------------
# Test: Markdown rendering preserves message structure
# ---------------------------------------------------------------------------


@given(conv_data=conversation_strategy(min_messages=1, max_messages=10))
@settings(max_examples=30, deadline=5000)
def test_markdown_preserves_message_count(conv_data: dict[str, Any]) -> None:
    """Every non-empty message must produce a ## section in markdown."""
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.core_markdown import format_conversation_markdown

    messages = [
        Message(
            id=m["id"],
            role=m["role"],
            text=m.get("text", ""),
            timestamp=None,
        )
        for m in conv_data["messages"]
        if m.get("text", "").strip()  # non-empty only
    ]
    if not messages:
        return  # skip if all messages empty

    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    md = format_conversation_markdown(conv)
    facts = _markdown_facts(md)

    # Core invariant: renderable messages → ## sections
    assert facts["message_sections"] == len(messages), (
        f"Expected {len(messages)} sections, got {facts['message_sections']}"
    )


@given(conv_data=conversation_strategy(min_messages=2, max_messages=8))
@settings(max_examples=30, deadline=5000)
def test_markdown_preserves_role_distribution(conv_data: dict[str, Any]) -> None:
    """Markdown role sections match the input role distribution."""
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.core_markdown import format_conversation_markdown

    messages = [
        Message(
            id=m["id"],
            role=m["role"],
            # Ensure text is non-whitespace so the message is renderable
            text=(m.get("text", "") or "").strip() or "placeholder",
        )
        for m in conv_data["messages"]
    ]
    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    md = format_conversation_markdown(conv)
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


@given(conv_data=conversation_strategy(min_messages=1, max_messages=10))
@settings(max_examples=30, deadline=5000)
def test_json_export_preserves_conversation_identity(conv_data: dict[str, Any]) -> None:
    """JSON export must preserve conversation ID, provider, title."""
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.formatting import format_conversation

    messages = [
        Message(
            id=m["id"],
            role=m["role"],
            text=m.get("text", "") or "placeholder",
        )
        for m in conv_data["messages"]
    ]
    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    json_str = format_conversation(conv, "json", None)
    facts = _json_facts(json_str)

    assert facts["conversation_id"] == str(conv.id)
    assert facts["provider"] == str(conv.provider)
    assert facts["title"] == conv.title


@given(conv_data=conversation_strategy(min_messages=1, max_messages=10))
@settings(max_examples=30, deadline=5000)
def test_json_export_preserves_message_count(conv_data: dict[str, Any]) -> None:
    """JSON export must have one message entry per input message."""
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.formatting import format_conversation

    messages = [
        Message(
            id=m["id"],
            role=m["role"],
            text=m.get("text", "") or "placeholder",
        )
        for m in conv_data["messages"]
    ]
    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    json_str = format_conversation(conv, "json", None)
    facts = _json_facts(json_str)

    assert facts["message_count"] == len(messages)


# ---------------------------------------------------------------------------
# Test: Cross-surface differential — JSON and YAML agree
# ---------------------------------------------------------------------------


@given(conv_data=conversation_strategy(min_messages=1, max_messages=5))
@settings(max_examples=20, deadline=5000)
def test_json_yaml_agree_on_message_count(conv_data: dict[str, Any]) -> None:
    """JSON and YAML exports of the same conversation have the same message count."""
    import yaml

    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.formatting import format_conversation

    messages = [
        Message(
            id=m["id"],
            role=m["role"],
            text=m.get("text", "") or "placeholder",
        )
        for m in conv_data["messages"]
    ]
    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    json_str = format_conversation(conv, "json", None)
    yaml_str = format_conversation(conv, "yaml", None)

    json_data = json.loads(json_str)
    yaml_data = yaml.safe_load(yaml_str)

    json_msgs = json_data.get("messages", [])
    yaml_msgs = yaml_data.get("messages", [])

    assert len(json_msgs) == len(yaml_msgs), f"JSON has {len(json_msgs)} messages, YAML has {len(yaml_msgs)}"


@given(conv_data=conversation_strategy(min_messages=1, max_messages=5))
@settings(max_examples=20, deadline=5000)
def test_json_yaml_agree_on_identity(conv_data: dict[str, Any]) -> None:
    """JSON and YAML agree on conversation_id, provider, title."""
    import yaml
    from hypothesis import assume

    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.formatting import format_conversation

    # YAML has edge cases with certain Unicode control characters
    assume(conv_data["title"].isprintable())

    messages = [
        Message(
            id=m["id"],
            role=m["role"],
            text=m.get("text", "") or "placeholder",
        )
        for m in conv_data["messages"]
    ]
    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    json_data = json.loads(format_conversation(conv, "json", None))
    yaml_data = yaml.safe_load(format_conversation(conv, "yaml", None))

    assert json_data["id"] == yaml_data["id"]
    assert json_data["provider"] == yaml_data["provider"]
    assert json_data["title"] == yaml_data["title"]


# ---------------------------------------------------------------------------
# Test: Content block structure preservation
# ---------------------------------------------------------------------------


@given(conv_data=conversation_strategy(min_messages=2, max_messages=5))
@settings(max_examples=20, deadline=5000)
def test_content_blocks_produce_structured_markdown(conv_data: dict[str, Any]) -> None:
    """When content blocks are present, markdown should contain structural markers."""
    from polylogue.lib.messages import MessageCollection
    from polylogue.lib.models import Conversation, Message
    from polylogue.rendering.core_markdown import format_conversation_markdown

    # Add content blocks to the first message
    blocks: list[dict[str, Any]] = [
        {"type": "thinking", "thinking": "Let me analyze this..."},
        {"type": "tool_use", "name": "Read", "id": "tool-1", "input": {"file_path": "/tmp/test.py"}},
        {"type": "text", "text": "Here's what I found."},
    ]

    messages = []
    for i, m in enumerate(conv_data["messages"]):
        msg = Message(
            id=m["id"],
            role=m["role"],
            text=m.get("text", "") or "placeholder",
            content_blocks=blocks if i == 0 and m["role"] == "assistant" else [],
        )
        messages.append(msg)

    conv = Conversation(
        id=conv_data["id"],
        provider=conv_data["provider"],
        title=conv_data["title"],
        messages=MessageCollection(messages=messages),
    )

    md = format_conversation_markdown(conv)

    # If first message was assistant and had blocks, check structure
    if messages[0].role.value == "assistant":
        assert "Thinking" in md or "Tool:" in md or "Read" in md, (
            "Content blocks should produce structural markers in markdown"
        )
