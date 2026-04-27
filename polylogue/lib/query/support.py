"""Shared support helpers for immutable conversation query plans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.types import Provider

if TYPE_CHECKING:
    from polylogue.lib.models import Conversation, ConversationSummary


def provider_values(values: tuple[Provider | str, ...]) -> tuple[str, ...]:
    return tuple(str(Provider.from_string(value)) for value in values)


def conversation_has_branches(conversation: Conversation) -> bool:
    return any(message.branch_index > 0 for message in conversation.messages)


def conversation_to_summary(conversation: Conversation) -> ConversationSummary:
    from polylogue.lib.models import ConversationSummary

    return ConversationSummary(
        id=conversation.id,
        provider=conversation.provider,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        provider_meta=conversation.provider_meta,
        metadata=conversation.metadata,
        parent_id=conversation.parent_id,
        branch_type=conversation.branch_type,
        message_count=len(conversation.messages),
        dialogue_count=sum(1 for message in conversation.messages if message.is_dialogue),
    )


__all__ = [
    "conversation_has_branches",
    "conversation_to_summary",
    "provider_values",
]
