"""Shared helpers for the conversation repository."""

from __future__ import annotations

from polylogue.lib.models import Conversation
from polylogue.protocols import VectorProvider
from polylogue.storage.hydrators import conversation_from_records
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord


def records_to_conversation(
    conversation: ConversationRecord,
    messages: list[MessageRecord],
    attachments: list[AttachmentRecord],
) -> Conversation:
    """Convert records to a Conversation model."""
    return conversation_from_records(conversation, messages, attachments)


def provider_conversation_id(conversation_id: str, provider: str | None) -> str:
    """Strip only the canonical provider prefix from conversation IDs."""
    if not provider:
        return conversation_id
    prefix = f"{provider}:"
    return conversation_id[len(prefix) :] if conversation_id.startswith(prefix) else conversation_id


def resolve_optional_vector_provider(
    vector_provider: VectorProvider | None,
) -> VectorProvider | None:
    """Resolve the explicitly supplied provider or create the default one."""
    if vector_provider is not None:
        return vector_provider

    from polylogue.storage.search_providers import create_vector_provider

    return create_vector_provider()
