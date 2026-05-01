"""Hypothesis strategies for message and conversation generation.

These strategies generate valid message structures for testing parsers
and the semantic models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeAlias

from hypothesis import strategies as st

from polylogue.archive.message.roles import Role
from polylogue.types import ConversationId, Provider

if TYPE_CHECKING:
    from polylogue.archive.models import Conversation, Message

JSONRecord: TypeAlias = dict[str, object]

# =============================================================================
# Content Block Strategies
# =============================================================================


@st.composite
def text_content_strategy(draw: st.DrawFn) -> JSONRecord:
    """Generate a text content block."""
    return {
        "type": "text",
        "text": draw(st.text(min_size=1, max_size=500)),
    }


@st.composite
def thinking_block_strategy(draw: st.DrawFn) -> JSONRecord:
    """Generate a thinking/reasoning block."""
    thinking_text = draw(st.text(min_size=10, max_size=1000))
    return {
        "type": "thinking",
        "thinking": thinking_text,
    }


@st.composite
def tool_use_block_strategy(draw: st.DrawFn) -> JSONRecord:
    """Generate a tool_use block."""
    tool_names = ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "Task", "WebFetch"]
    return {
        "type": "tool_use",
        "name": draw(st.sampled_from(tool_names)),
        "id": draw(st.uuids()).hex[:24],
        "input": draw(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N"))),
                values=st.one_of(st.text(max_size=100), st.integers(), st.booleans()),
                max_size=5,
            )
        ),
    }


@st.composite
def tool_result_block_strategy(draw: st.DrawFn) -> JSONRecord:
    """Generate a tool_result block."""
    return {
        "type": "tool_result",
        "tool_use_id": draw(st.uuids()).hex[:24],
        "content": draw(st.text(max_size=500)),
    }


@st.composite
def code_block_strategy(draw: st.DrawFn) -> JSONRecord:
    """Generate a code content block."""
    return {
        "type": "code",
        "text": draw(st.text(min_size=1, max_size=500)),
        "language": draw(st.one_of(st.none(), st.sampled_from(["python", "bash", "json"]))),
    }


@st.composite
def content_block_strategy(draw: st.DrawFn) -> JSONRecord:
    """Generate any type of content block."""
    return draw(
        st.one_of(
            text_content_strategy(),
            thinking_block_strategy(),
            tool_use_block_strategy(),
            tool_result_block_strategy(),
            code_block_strategy(),
        )
    )


# =============================================================================
# Message Strategies
# =============================================================================


@st.composite
def message_strategy(
    draw: st.DrawFn,
    roles: list[str] | None = None,
    with_timestamp: bool = True,
    with_content_blocks: bool = False,
) -> JSONRecord:
    """Generate a generic message dict.

    Args:
        roles: Allowed roles (default: user, assistant)
        with_timestamp: Include timestamp field
        with_content_blocks: Include structured content_blocks
    """
    if roles is None:
        roles = ["user", "assistant"]

    msg: JSONRecord = {
        "id": draw(st.uuids()).hex[:12],
        "role": draw(st.sampled_from(roles)),
        "text": draw(st.text(min_size=1, max_size=500)),
    }

    if with_timestamp:
        # Generate timestamp as ISO string or epoch
        ts_format = draw(st.sampled_from(["iso", "epoch"]))
        if ts_format == "iso":
            dt = draw(
                st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2030, 1, 1),
                    timezones=st.just(timezone.utc),
                )
            )
            msg["timestamp"] = dt.isoformat()
        else:
            msg["timestamp"] = draw(
                st.floats(
                    min_value=1577836800,  # 2020-01-01
                    max_value=1893456000,  # 2030-01-01
                )
            )

    if with_content_blocks:
        msg["content_blocks"] = draw(
            st.lists(
                content_block_strategy(),
                min_size=1,
                max_size=5,
            )
        )

    return msg


# =============================================================================
# Conversation Strategies
# =============================================================================


@st.composite
def conversation_strategy(
    draw: st.DrawFn,
    min_messages: int = 1,
    max_messages: int = 20,
    providers: list[str] | None = None,
) -> JSONRecord:
    """Generate a conversation dict.

    Args:
        min_messages: Minimum number of messages
        max_messages: Maximum number of messages
        providers: Allowed provider names
    """
    if providers is None:
        providers = ["chatgpt", "claude-ai", "claude-code", "codex"]

    messages = draw(
        st.lists(
            message_strategy(),
            min_size=min_messages,
            max_size=max_messages,
        )
    )

    # Ensure alternating user/assistant for realism
    for i, msg in enumerate(messages):
        msg["role"] = "user" if i % 2 == 0 else "assistant"

    return {
        "id": draw(st.uuids()).hex,
        "title": draw(st.text(min_size=1, max_size=100)),
        "provider": draw(st.sampled_from(providers)),
        "messages": messages,
        "created_at": draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2030, 1, 1),
                timezones=st.just(timezone.utc),
            )
        ).isoformat(),
    }


# =============================================================================
# Model Instance Strategies (return actual Message/Conversation model objects)
# =============================================================================
# Dict-based strategies above: raw JSON structures for parser/wire-format testing.
# Model-based strategies below: typed domain objects for business-logic testing.


@st.composite
def message_model_strategy(draw: st.DrawFn, *, role: str | None = None) -> Message:
    """Generate a Message model instance with arbitrary content."""
    from polylogue.archive.models import Message as MessageModel

    role_val = role or draw(st.sampled_from(["user", "assistant", "system", "tool"]))
    text = draw(st.one_of(st.none(), st.text(max_size=200)))

    # Optionally add content_blocks in provider_meta
    provider_meta: JSONRecord | None = None
    block_type = draw(st.sampled_from(["text", "thinking", "tool_use", "none"]))
    if block_type != "none":
        block_text = draw(st.text(max_size=100))
        provider_meta = {"content_blocks": [{"type": block_type, "text": block_text}]}

    # Optionally add cost/duration via raw in provider_meta
    cost = draw(
        st.one_of(st.none(), st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False))
    )
    duration = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=60000)))
    if cost is not None or duration is not None:
        raw: JSONRecord = {}
        if cost is not None:
            raw["costUSD"] = cost
        if duration is not None:
            raw["durationMs"] = duration
        if provider_meta is None:
            provider_meta = {}
        provider_meta["raw"] = raw

    return MessageModel(
        id=draw(st.text(min_size=1, max_size=40, alphabet=st.characters(whitelist_categories=("L", "N")))),
        role=Role.normalize(role_val),
        text=text,
        provider_meta=provider_meta,
    )


@st.composite
def parsed_attachment_model_strategy(draw: st.DrawFn) -> object:
    """Generate a ParsedAttachment model instance for property testing."""
    from polylogue.sources.parsers.base import ParsedAttachment

    return ParsedAttachment(
        provider_attachment_id=draw(st.uuids()).hex[:12],
        message_provider_id=draw(st.one_of(st.none(), st.uuids().map(lambda u: u.hex[:12]))),
        name=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50))),
        mime_type=draw(
            st.one_of(
                st.none(),
                st.sampled_from(
                    [
                        "text/plain",
                        "application/pdf",
                        "image/png",
                        "image/jpeg",
                        "application/json",
                    ]
                ),
            )
        ),
        size_bytes=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=100_000_000))),
        path=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
    )


@st.composite
def conversation_model_strategy(draw: st.DrawFn, *, min_messages: int = 0, max_messages: int = 10) -> Conversation:
    """Generate a Conversation model instance with arbitrary messages."""
    from polylogue.archive.message.messages import MessageCollection
    from polylogue.archive.models import Conversation as ConversationModel

    messages = draw(st.lists(message_model_strategy(), min_size=min_messages, max_size=max_messages))
    return ConversationModel(
        id=ConversationId(
            draw(st.text(min_size=1, max_size=40, alphabet=st.characters(whitelist_categories=("L", "N"))))
        ),
        provider=Provider.from_string(
            draw(st.sampled_from(["chatgpt", "claude-ai", "claude-code", "codex", "gemini"]))
        ),
        messages=MessageCollection(messages=messages),
    )
