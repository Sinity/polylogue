"""Message and dialogue-pair domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.model_runtime import MessageRuntimeMixin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin, Provider


class Message(MessageRuntimeMixin, BaseModel):
    id: str
    role: Role
    text: str | None = None
    timestamp: datetime | None = None
    provider: Provider | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    blocks: list[dict[str, object]] = Field(default_factory=list)
    message_type: MessageType = MessageType.MESSAGE
    material_origin: MaterialOrigin = MaterialOrigin.UNKNOWN
    parent_id: str | None = None
    branch_index: int = 0
    # Stats projected from the storage layer so reader surfaces can
    # render fold/paste indicators without re-deriving them. See #1201
    # (paste rendering) and the session-level flags in
    # ``polylogue.surfaces.payloads.SessionFlagsPayload``. Word
    # count remains derived (``MessageRuntimeMixin.word_count``).
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: int = 0
    model_name: str | None = None

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

    @field_validator("material_origin", mode="before")
    @classmethod
    def coerce_material_origin(cls, v: object) -> MaterialOrigin:
        return MaterialOrigin.normalize(v)

    @model_validator(mode="after")
    def derive_material_origin(self) -> Message:
        if self.material_origin is MaterialOrigin.UNKNOWN:
            from polylogue.archive.message.artifacts import classify_material_origin

            block_types: list[BlockType] = []
            for block in self.blocks:
                raw_type = block.get("type")
                if raw_type is None:
                    continue
                try:
                    block_types.append(BlockType.from_string(str(raw_type)))
                except ValueError:
                    continue
            self.material_origin = classify_material_origin(
                role=self.role,
                message_type=self.message_type,
                text=self.text,
                block_types=tuple(block_types),
            )
        return self


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
