"""Conversation and summary domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.archive.conversation.runtime import ConversationRuntimeMixin
from polylogue.archive.conversation.summary_runtime import ConversationSummaryRuntimeMixin
from polylogue.archive.message.messages import MessageCollection
from polylogue.types import ConversationId, Provider


class ConversationSummary(ConversationSummaryRuntimeMixin, BaseModel):
    """Lightweight conversation metadata without messages."""

    id: ConversationId
    provider: Provider
    title: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    parent_id: ConversationId | None = None
    branch_type: BranchType | None = None
    message_count: int | None = None
    dialogue_count: int | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")


class Conversation(ConversationRuntimeMixin, BaseModel):
    """Conversation with eagerly or lazily materialized message collection."""

    id: ConversationId
    provider: Provider
    title: str | None = None
    messages: MessageCollection
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    parent_id: ConversationId | None = None
    branch_type: BranchType | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["Conversation", "ConversationSummary"]
