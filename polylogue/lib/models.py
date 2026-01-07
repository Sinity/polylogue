from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from polylogue.core.timestamps import parse_timestamp
from polylogue.store import AttachmentRecord, ConversationRecord, MessageRecord


class Attachment(BaseModel):
    id: str
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict | None = None

    @classmethod
    def from_record(cls, record: AttachmentRecord) -> Attachment:
        # Extract name from meta if possible
        name = record.provider_meta.get("name") if record.provider_meta else None
        return cls(
            id=record.attachment_id,
            name=name or record.attachment_id,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            path=record.path,
            provider_meta=record.provider_meta,
        )


class Message(BaseModel):
    id: str
    role: str
    text: str | None = None
    timestamp: datetime | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    provider_meta: dict | None = None

    @classmethod
    def from_record(cls, record: MessageRecord, attachments: list[AttachmentRecord]) -> Message:
        ts = parse_timestamp(record.timestamp)

        return cls(
            id=record.message_id,
            role=record.role or "unknown",
            text=record.text,
            timestamp=ts,
            attachments=[Attachment.from_record(a) for a in attachments],
            provider_meta=record.provider_meta,
        )

    @property
    def is_user(self) -> bool:
        return self.role.lower() == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role.lower() in ("assistant", "model")


class Conversation(BaseModel):
    id: str
    provider: str
    title: str | None = None
    messages: list[Message]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider_meta: dict | None = None

    @classmethod
    def from_records(
        cls,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> Conversation:
        # Group attachments by message_id
        att_map = {}
        for att in attachments:
            if att.message_id:
                att_map.setdefault(att.message_id, []).append(att)
            # What about conversation-level attachments? Polylogue logic attaches them to messages or generic?
            # Store logic: attachment_refs has message_id. If None, it's orphan.

        rich_messages = []
        for msg in messages:
            msg_atts = att_map.get(msg.message_id, [])
            rich_messages.append(Message.from_record(msg, msg_atts))

        # Sort messages
        # Ideally DB provides order, but we can ensure sort here if needed

        return cls(
            id=conversation.conversation_id,
            provider=conversation.provider_name,
            title=conversation.title,
            messages=rich_messages,
            provider_meta=conversation.provider_meta,
        )

    def user_only(self) -> Conversation:
        """Return a view of the conversation with only user messages."""
        return self.model_copy(update={"messages": [m for m in self.messages if m.is_user]})

    def without_files(self) -> Conversation:
        """Return a view with file attachments stripped."""
        new_msgs = []
        for m in self.messages:
            # Logic: if message IS just a file wrapper?
            # Or just strip attachments list?
            new_msgs.append(m.model_copy(update={"attachments": []}))
        return self.model_copy(update={"messages": new_msgs})

    def text_only(self) -> str:
        """Render conversation to plain text."""
        return "\n\n".join(f"{m.role}: {m.text}" for m in self.messages if m.text)
