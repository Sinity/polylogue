"""Message and dialogue-pair domain models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator

from polylogue.archive.attachment.models import Attachment
from polylogue.archive.message.model_runtime import MessageRuntimeMixin
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin, Origin


class Message(MessageRuntimeMixin, BaseModel):
    id: str
    role: Role
    text: str | None = None
    timestamp: datetime | None = None
    origin: Origin | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    blocks: list[dict[str, object]] = Field(default_factory=list)
    message_type: MessageType = MessageType.MESSAGE
    material_origin: MaterialOrigin = MaterialOrigin.UNKNOWN
    parent_id: str | None = None
    # branch_index (aka variant_index) is CREATION ORDER among siblings -- for
    # ChatGPT it's ``children.index(current_node_id)`` at ingest time. It is
    # NOT display state: an edited/regenerated turn's first attempt is
    # variant_index 0, but the accepted sibling can be any index. Do not use
    # this to decide what renders as "the" conversation -- see
    # ``is_active_path`` (polylogue-9qq7).
    branch_index: int = 0
    # The provider-reported "this sibling is the currently-accepted one"
    # signal (storage: messages.is_active_path). This, not branch_index, is
    # authoritative for mainline/display selection. ``None`` means unknown --
    # e.g. a read path that has not threaded the column through yet, or a
    # provider with no branch concept -- and must NOT be treated as "not
    # active": collapsing unknown to hidden would silently empty transcripts.
    is_active_path: bool | None = None
    # Stats projected from the storage layer so reader surfaces can
    # render fold/paste indicators without re-deriving them. See #1201
    # (paste rendering) and the session-level flags in
    # ``polylogue.surfaces.payloads.SessionFlagsPayload``. Word
    # count remains derived (``MessageRuntimeMixin.word_count``).
    has_tool_use: bool = False
    has_thinking: bool = False
    has_paste: bool = False
    paste_boundary_state: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: int = 0
    model_name: str | None = None
    # Provider-reported terminal signal for this assistant turn (storage:
    # messages.stop_reason, e.g. Anthropic's ``end_turn``/``tool_use``/
    # ``max_tokens``/``stop_sequence``/``pause_turn``). ``None`` means
    # unreported/not-applicable/not-yet-threaded by the read path, not
    # "ended normally" -- do not default it to a happy-path value.
    stop_reason: str | None = None

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        raw = (str(v) if v is not None else "").strip() or "unknown"
        return Role.normalize(raw)

    @field_validator("origin", mode="before")
    @classmethod
    def coerce_origin(cls, v: object) -> Origin | None:
        if v is None:
            return None
        if isinstance(v, Origin):
            return v
        return Origin.from_string(str(v))

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
