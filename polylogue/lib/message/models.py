"""Message and dialogue-pair domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from polylogue.lib.attachment.models import Attachment
from polylogue.lib.message.model_runtime import MessageRuntimeMixin
from polylogue.lib.message.types import MessageType
from polylogue.lib.roles import Role
from polylogue.types import Provider


class Message(MessageRuntimeMixin, BaseModel):
    id: str
    role: Role
    text: str | None = None
    timestamp: datetime | None = None
    provider: Provider | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
    content_blocks: list[dict[str, object]] = Field(default_factory=list)
    message_type: MessageType = MessageType.MESSAGE
    parent_id: str | None = None
    branch_index: int = 0

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        raw = (str(v) if v is not None else "").strip() or "unknown"
        return Role.normalize(raw)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("message_type", mode="before")
    @classmethod
    def coerce_message_type(cls, v: object) -> MessageType:
        return MessageType.normalize(v)


class DialoguePair(BaseModel):
    """A user message followed by assistant response."""

    user: Message
    assistant: Message

    @model_validator(mode="after")
    def validate_roles(self) -> DialoguePair:
        if not self.user.is_user:
            raise ValueError(f"user message must have user role, got {self.user.role}")
        if not self.assistant.is_assistant:
            raise ValueError(f"assistant message must have assistant role, got {self.assistant.role}")
        return self

    @property
    def exchange(self) -> str:
        return f"User: {self.user.text or ''}\n\nAssistant: {self.assistant.text or ''}"


__all__ = ["DialoguePair", "Message"]
