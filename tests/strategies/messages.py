"""Hypothesis strategies for message and conversation generation.

These strategies generate valid message structures for testing importers
and the semantic models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hypothesis import strategies as st

# =============================================================================
# Content Block Strategies
# =============================================================================


@st.composite
def text_content_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a text content block."""
    return {
        "type": "text",
        "text": draw(st.text(min_size=1, max_size=500)),
    }


@st.composite
def thinking_block_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a thinking/reasoning block."""
    thinking_text = draw(st.text(min_size=10, max_size=1000))
    return {
        "type": "thinking",
        "thinking": thinking_text,
    }


@st.composite
def tool_use_block_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a tool_use block."""
    tool_names = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebFetch"]
    return {
        "type": "tool_use",
        "name": draw(st.sampled_from(tool_names)),
        "id": draw(st.uuids()).hex[:24],
        "input": draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
            values=st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
            max_size=5,
        )),
    }


@st.composite
def tool_result_block_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a tool_result block."""
    return {
        "type": "tool_result",
        "tool_use_id": draw(st.uuids()).hex[:24],
        "content": draw(st.text(max_size=500)),
    }


@st.composite
def content_block_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate any type of content block."""
    return draw(st.one_of(
        text_content_strategy(),
        thinking_block_strategy(),
        tool_use_block_strategy(),
        tool_result_block_strategy(),
    ))


# =============================================================================
# Message Strategies
# =============================================================================


@st.composite
def message_strategy(
    draw: st.DrawFn,
    roles: list[str] | None = None,
    with_timestamp: bool = True,
    with_content_blocks: bool = False,
) -> dict[str, Any]:
    """Generate a generic message dict.

    Args:
        roles: Allowed roles (default: user, assistant)
        with_timestamp: Include timestamp field
        with_content_blocks: Include structured content_blocks
    """
    if roles is None:
        roles = ["user", "assistant"]

    msg: dict[str, Any] = {
        "id": draw(st.uuids()).hex[:12],
        "role": draw(st.sampled_from(roles)),
        "text": draw(st.text(min_size=1, max_size=500)),
    }

    if with_timestamp:
        # Generate timestamp as ISO string or epoch
        ts_format = draw(st.sampled_from(["iso", "epoch"]))
        if ts_format == "iso":
            dt = draw(st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 1, 1),
                timezones=st.just(timezone.utc),
            ))
            msg["timestamp"] = dt.isoformat()
        else:
            msg["timestamp"] = draw(st.floats(
                min_value=1577836800,  # 2020-01-01
                max_value=1893456000,  # 2030-01-01
            ))

    if with_content_blocks:
        msg["content_blocks"] = draw(st.lists(
            content_block_strategy(),
            min_size=1,
            max_size=5,
        ))

    return msg


@st.composite
def parsed_message_strategy(draw: st.DrawFn) -> dict[str, Any]:
    """Generate a ParsedMessage-compatible dict."""
    return {
        "provider_message_id": draw(st.uuids()).hex[:12],
        "role": draw(st.sampled_from(["user", "assistant", "system"])),
        "text": draw(st.text(min_size=1, max_size=500)),
        "timestamp": draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 1, 1),
            timezones=st.just(timezone.utc),
        )).isoformat(),
        "provider_meta": draw(st.none() | st.fixed_dictionaries({
            "raw": st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.text(max_size=50),
                max_size=3,
            ),
        })),
    }


# =============================================================================
# Typed Strategies (return actual model instances)
# =============================================================================


@st.composite
def parsed_message_model_strategy(draw: st.DrawFn):
    """Generate a ParsedMessage model instance for property testing."""
    from polylogue.sources.parsers.base import ParsedMessage

    return ParsedMessage(
        provider_message_id=draw(st.uuids()).hex[:12],
        role=draw(st.sampled_from(["user", "assistant", "system"])),
        text=draw(st.text(min_size=0, max_size=500)),
        timestamp=draw(st.one_of(
            st.none(),
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 1, 1),
                timezones=st.just(timezone.utc),
            ).map(lambda dt: dt.isoformat()),
        )),
    )


@st.composite
def parsed_attachment_model_strategy(draw: st.DrawFn):
    """Generate a ParsedAttachment model instance for property testing."""
    from polylogue.sources.parsers.base import ParsedAttachment

    return ParsedAttachment(
        provider_attachment_id=draw(st.uuids()).hex[:12],
        message_provider_id=draw(st.uuids()).hex[:12],
        name=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        mime_type=draw(st.one_of(
            st.none(),
            st.sampled_from([
                "text/plain",
                "application/pdf",
                "image/png",
                "image/jpeg",
                "application/json",
            ]),
        )),
        size_bytes=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100_000_000))),
    )


@st.composite
def parsed_conversation_model_strategy(
    draw: st.DrawFn,
    min_messages: int = 0,
    max_messages: int = 10,
    with_attachments: bool = False,
):
    """Generate a ParsedConversation model instance for property testing.

    This strategy creates actual ParsedConversation objects (not dicts),
    enabling property-based testing of pipeline hashing functions.

    Args:
        min_messages: Minimum number of messages
        max_messages: Maximum number of messages
        with_attachments: Whether to include attachments
    """
    from polylogue.sources.parsers.base import ParsedConversation

    providers = ["chatgpt", "claude", "claude-code", "codex", "gemini"]

    messages = draw(st.lists(
        parsed_message_model_strategy(),
        min_size=min_messages,
        max_size=max_messages,
    ))

    # Ensure alternating user/assistant for realism
    for i, msg in enumerate(messages):
        object.__setattr__(msg, "role", "user" if i % 2 == 0 else "assistant")

    attachments = []
    if with_attachments:
        attachments = draw(st.lists(
            parsed_attachment_model_strategy(),
            min_size=0,
            max_size=3,
        ))

    return ParsedConversation(
        provider_name=draw(st.sampled_from(providers)),
        provider_conversation_id=draw(st.uuids()).hex,
        title=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
        created_at=draw(st.one_of(
            st.none(),
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 1, 1),
                timezones=st.just(timezone.utc),
            ).map(lambda dt: dt.isoformat()),
        )),
        updated_at=draw(st.one_of(
            st.none(),
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 1, 1),
                timezones=st.just(timezone.utc),
            ).map(lambda dt: dt.isoformat()),
        )),
        messages=messages,
        attachments=attachments,
    )


# =============================================================================
# Conversation Strategies
# =============================================================================


@st.composite
def conversation_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 20,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a conversation dict.

    Args:
        min_messages: Minimum number of messages
        max_messages: Maximum number of messages
        providers: Allowed provider names
    """
    if providers is None:
        providers = ["chatgpt", "claude", "claude-code", "codex"]

    messages = draw(st.lists(
        message_strategy(),
        min_size=min_messages,
        max_size=max_messages,
    ))

    # Ensure alternating user/assistant for realism
    for i, msg in enumerate(messages):
        msg["role"] = "user" if i % 2 == 0 else "assistant"

    return {
        "id": draw(st.uuids()).hex,
        "title": draw(st.text(min_size=1, max_size=100)),
        "provider": draw(st.sampled_from(providers)),
        "messages": messages,
        "created_at": draw(st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2030, 1, 1),
            timezones=st.just(timezone.utc),
        )).isoformat(),
    }
