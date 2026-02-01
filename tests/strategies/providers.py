"""Hypothesis strategies for provider-specific export formats.

These strategies generate valid export structures that match the format
each importer expects, enabling property-based testing of parsing logic.
"""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st


# =============================================================================
# ChatGPT Strategies
# =============================================================================


@st.composite
def chatgpt_message_node_strategy(
    draw: st.DrawFn,
    with_children: bool = True,
) -> dict[str, Any]:
    """Generate a ChatGPT mapping node.

    ChatGPT exports have a `mapping` object where each key is a UUID
    and values are message nodes with nested content.
    """
    node_id = draw(st.uuids()).hex

    # Content parts can be strings (text) or dicts (image refs, etc)
    content_parts = draw(st.lists(
        st.text(min_size=1, max_size=500),
        min_size=1,
        max_size=3,
    ))

    node: dict[str, Any] = {
        "id": node_id,
        "message": {
            "id": node_id,
            "author": {
                "role": draw(st.sampled_from(["user", "assistant", "system", "tool"])),
                "name": draw(st.none() | st.text(min_size=1, max_size=20)),
                "metadata": {},
            },
            "content": {
                "content_type": draw(st.sampled_from(["text", "thoughts", "code"])),
                "parts": content_parts,
            },
            "create_time": draw(st.floats(min_value=1577836800, max_value=1893456000)),
            "metadata": {},
        },
    }

    if with_children:
        # Children are UUIDs of other nodes
        num_children = draw(st.integers(min_value=0, max_value=2))
        node["children"] = [draw(st.uuids()).hex for _ in range(num_children)]

    return node


@st.composite
def chatgpt_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> dict[str, Any]:
    """Generate a complete ChatGPT export structure.

    The export has:
    - id: conversation UUID
    - title: conversation title
    - mapping: dict of UUID -> node
    - create_time/update_time: epoch timestamps
    """
    nodes = draw(st.lists(
        chatgpt_message_node_strategy(with_children=False),
        min_size=min_messages,
        max_size=max_messages,
    ))

    # Build mapping from nodes
    mapping = {node["id"]: node for node in nodes}

    # Wire up children references (linear chain)
    node_ids = list(mapping.keys())
    for i, node_id in enumerate(node_ids[:-1]):
        mapping[node_id]["children"] = [node_ids[i + 1]]

    return {
        "id": draw(st.uuids()).hex,
        "title": draw(st.text(min_size=1, max_size=100)),
        "mapping": mapping,
        "create_time": draw(st.floats(min_value=1577836800, max_value=1893456000)),
        "update_time": draw(st.floats(min_value=1577836800, max_value=1893456000)),
        "current_node": node_ids[-1] if node_ids else "",
    }


# =============================================================================
# Claude AI Strategies
# =============================================================================


@st.composite
def claude_ai_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a Claude AI chat_messages entry."""
    return {
        "uuid": draw(st.uuids()).hex,
        "sender": draw(st.sampled_from(["human", "assistant"])),
        "text": draw(st.text(min_size=1, max_size=500)),
        "created_at": draw(st.datetimes()).isoformat(),
    }


@st.composite
def claude_ai_export_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> dict[str, Any]:
    """Generate a Claude AI export structure."""
    messages = draw(st.lists(
        claude_ai_message_strategy(),
        min_size=min_messages,
        max_size=max_messages,
    ))

    # Alternate roles for realism
    for i, msg in enumerate(messages):
        msg["sender"] = "human" if i % 2 == 0 else "assistant"

    return {
        "id": draw(st.uuids()).hex,
        "name": draw(st.text(min_size=1, max_size=100)),
        "chat_messages": messages,
        "created_at": draw(st.datetimes()).isoformat(),
        "updated_at": draw(st.datetimes()).isoformat(),
    }


# =============================================================================
# Claude Code Strategies
# =============================================================================


@st.composite
def claude_code_content_block_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a Claude Code content block."""
    block_type = draw(st.sampled_from(["text", "thinking", "tool_use", "tool_result"]))

    if block_type == "text":
        return {"type": "text", "text": draw(st.text(min_size=1, max_size=500))}
    elif block_type == "thinking":
        return {"type": "thinking", "thinking": draw(st.text(min_size=10, max_size=1000))}
    elif block_type == "tool_use":
        return {
            "type": "tool_use",
            "name": draw(st.sampled_from(["Read", "Write", "Bash", "Glob"])),
            "id": f"toolu_{draw(st.uuids()).hex[:20]}",
            "input": {"path": draw(st.text(min_size=1, max_size=50))},
        }
    else:  # tool_result
        return {
            "type": "tool_result",
            "tool_use_id": f"toolu_{draw(st.uuids()).hex[:20]}",
            "content": draw(st.text(max_size=500)),
        }


@st.composite
def claude_code_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a Claude Code JSONL message entry."""
    msg_type = draw(st.sampled_from(["user", "assistant"]))

    content_blocks = draw(st.lists(
        claude_code_content_block_strategy(),
        min_size=1,
        max_size=5,
    ))

    msg: dict[str, Any] = {
        "type": msg_type,
        "uuid": draw(st.uuids()).hex,
        "timestamp": int(draw(st.floats(min_value=1577836800000, max_value=1893456000000))),
        "message": {
            "role": msg_type,
            "content": content_blocks,
        },
    }

    # Add metadata for assistant messages
    if msg_type == "assistant":
        if draw(st.booleans()):
            msg["costUSD"] = draw(st.floats(min_value=0, max_value=1.0))
        if draw(st.booleans()):
            msg["durationMs"] = draw(st.integers(min_value=100, max_value=30000))

    return msg


@st.composite
def claude_code_session_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
) -> list[dict[str, Any]]:
    """Generate a complete Claude Code JSONL session (list of messages)."""
    messages = draw(st.lists(
        claude_code_message_strategy(),
        min_size=min_messages,
        max_size=max_messages,
    ))

    # Alternate types for realism
    for i, msg in enumerate(messages):
        msg["type"] = "user" if i % 2 == 0 else "assistant"
        msg["message"]["role"] = msg["type"]

    # Add session ID to all
    session_id = draw(st.uuids()).hex
    for msg in messages:
        msg["sessionId"] = session_id

    return messages


# =============================================================================
# Codex Strategies
# =============================================================================


@st.composite
def codex_content_item_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a Codex content item."""
    return {
        "type": "input_text",
        "text": draw(st.text(min_size=1, max_size=500)),
    }


@st.composite
def codex_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a Codex message entry."""
    return {
        "type": "message",
        "role": draw(st.sampled_from(["user", "assistant"])),
        "id": draw(st.uuids()).hex[:12],
        "content": draw(st.lists(
            codex_content_item_strategy(),
            min_size=1,
            max_size=3,
        )),
    }


@st.composite
def codex_session_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 10,
    use_envelope: bool = True,
) -> list[dict[str, Any]]:
    """Generate a Codex JSONL session.

    Args:
        use_envelope: Use newer envelope format (session_meta + response_item)
    """
    session_id = draw(st.uuids()).hex
    timestamp = draw(st.datetimes()).isoformat()

    entries: list[dict[str, Any]] = []

    if use_envelope:
        # Session metadata
        entries.append({
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": timestamp,
            },
        })

        # Messages wrapped in response_item
        messages = draw(st.lists(
            codex_message_strategy(),
            min_size=min_messages,
            max_size=max_messages,
        ))
        for i, msg in enumerate(messages):
            msg["role"] = "user" if i % 2 == 0 else "assistant"
            entries.append({
                "type": "response_item",
                "payload": msg,
            })
    else:
        # Legacy format - direct records
        entries.append({
            "id": session_id,
            "timestamp": timestamp,
        })

        messages = draw(st.lists(
            codex_message_strategy(),
            min_size=min_messages,
            max_size=max_messages,
        ))
        for i, msg in enumerate(messages):
            msg["role"] = "user" if i % 2 == 0 else "assistant"
            entries.append(msg)

    return entries
