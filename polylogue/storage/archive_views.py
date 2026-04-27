"""Archive-facing projections and lightweight identity views."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from polylogue.storage.runtime import AttachmentRecord, ConversationRecord, MessageRecord


class ExistingConversation(BaseModel):
    conversation_id: str
    content_hash: str


@dataclass(frozen=True)
class ConversationRenderProjection:
    """Repository-owned render projection preserving raw attachment layout."""

    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]


__all__ = ["ConversationRenderProjection", "ExistingConversation"]
