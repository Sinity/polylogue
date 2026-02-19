"""Shared test builders for Conversation and Message domain objects.

These factories build polylogue.lib.models instances with sensible defaults,
suitable for unit tests that operate on domain models rather than storage records.

For storage-layer helpers (ConversationRecord, MessageRecord), see helpers.py.

Usage:
    from tests.infra.builders import make_msg, make_conv

    msg = make_msg(role="assistant", text="Hello")
    conv = make_conv(title="My Conversation", messages=[msg])
    conv_no_title = make_conv(title=None)
    msg_no_text = make_msg(text=None)
"""

from __future__ import annotations

from polylogue.lib.messages import MessageCollection
from polylogue.lib.models import Conversation, Message


def make_msg(
    id: str = "m1",
    role: str = "user",
    text: str | None = "hello",
    **kwargs,
) -> Message:
    """Build a Message with sensible defaults for testing.

    Args:
        id: Message ID (default "m1").
        role: Message role — "user", "assistant", "system", etc. (default "user").
        text: Message text; may be None to exercise None-guard code paths.
        **kwargs: Any additional Message fields (timestamp, attachments, etc.).

    Returns:
        A fully constructed Message domain object.
    """
    return Message(id=id, role=role, text=text, **kwargs)


def make_conv(
    messages: list[Message] | None = None,
    title: str | None = "Test",
    provider: str = "test",
    id: str = "test-conv",
    **kwargs,
) -> Conversation:
    """Build a Conversation with sensible defaults for testing.

    Args:
        messages: List of Message objects; defaults to [make_msg()] when None.
        title: Conversation title; may be None to exercise None-guard code paths.
        provider: Provider name (default "test").
        id: Conversation ID (default "test-conv").
        **kwargs: Any additional Conversation fields (created_at, metadata, etc.).

    Returns:
        A fully constructed Conversation domain object with an eager MessageCollection.
    """
    msgs = messages if messages is not None else [make_msg()]
    return Conversation(
        id=id,
        provider=provider,
        title=title,
        messages=MessageCollection(messages=msgs),
        **kwargs,
    )
