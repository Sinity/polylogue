"""Shared test builders for Conversation and Message domain objects.

These factories build polylogue.lib.models instances with sensible defaults,
suitable for unit tests that operate on domain models rather than storage records.

For storage-layer helpers (ConversationRecord, MessageRecord), see
tests.infra.storage_records.

Usage:
    from tests.infra.builders import make_msg, make_conv

    msg = make_msg(role="assistant", text="Hello")
    conv = make_conv(title="My Conversation", messages=[msg])
    conv_no_title = make_conv(title=None)
    msg_no_text = make_msg(text=None)
"""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.lib.message.messages import MessageCollection
from polylogue.lib.models import Conversation, Message
from polylogue.lib.roles import Role
from polylogue.types import ConversationId, Provider


def make_msg(
    id: str = "m1",
    role: Role | str = Role.USER,
    text: str | None = "hello",
    **kwargs: object,
) -> Message:
    """Build a Message with sensible defaults for testing.

    Args:
        id: Message ID (default "m1").
        role: Message role — "user", "assistant", "system", etc. (default "user").
        text: Message text; may be None to exercise None-guard code paths.
        **kwargs: Additional Message fields (timestamp, attachments, etc.).

    Returns:
        A fully constructed Message domain object.
    """
    role_value = role if isinstance(role, Role) else (Role.normalize(role.strip()) if role.strip() else Role.UNKNOWN)
    payload: dict[str, object] = {"id": id, "role": role_value, "text": text}
    payload.update(kwargs)
    return Message.model_validate(payload)


def make_conv(
    messages: Sequence[Message] | MessageCollection | None = None,
    title: str | None = "Test",
    provider: Provider | str = Provider.UNKNOWN,
    id: str = "test-conv",
    **kwargs: object,
) -> Conversation:
    """Build a Conversation with sensible defaults for testing.

    Args:
        messages: List of Message objects; defaults to [make_msg()] when None.
        title: Conversation title; may be None to exercise None-guard code paths.
        provider: Provider name (default "test").
        id: Conversation ID (default "test-conv").
        **kwargs: Additional Conversation fields (created_at, metadata, etc.).

    Returns:
        A fully constructed Conversation domain object with an eager MessageCollection.
    """
    provider_value = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    if messages is None:
        message_collection = MessageCollection(messages=[make_msg()])
    elif isinstance(messages, MessageCollection):
        message_collection = messages
    else:
        message_collection = MessageCollection(messages=list(messages))
    payload: dict[str, object] = {
        "id": ConversationId(id),
        "provider": provider_value,
        "title": title,
        "messages": message_collection,
    }
    payload.update(kwargs)
    return Conversation.model_validate(payload)
