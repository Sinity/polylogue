from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from typing import Callable

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
        name = record.provider_meta.get("name") if record.provider_meta else None
        return cls(
            id=record.attachment_id,
            name=name or record.attachment_id,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            path=record.path,
            provider_meta=record.provider_meta,
        )


# Patterns for content classification
_TOOL_MARKERS = [
    "tool_use", "tool_result", "function_calls", "function_result",
    "antml:invoke", "antml:function_calls",
]
_THINKING_MARKERS = ["antml:thinking", "thinking", "isThought"]
_CONTEXT_PATTERNS = [
    r"^Contents of .+:",
    r"^<file path=",
]


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

    # --- Role classification ---

    @property
    def is_user(self) -> bool:
        return self.role.lower() == "user"

    @property
    def is_assistant(self) -> bool:
        return self.role.lower() in ("assistant", "model")

    @property
    def is_system(self) -> bool:
        return self.role.lower() == "system"

    @property
    def is_dialogue(self) -> bool:
        """True if this is a user or assistant message (not system/tool)."""
        return self.is_user or self.is_assistant

    # --- Content classification (dynamic, heuristic-based) ---

    @property
    def is_tool_use(self) -> bool:
        """Message contains tool/function calls or results."""
        if not self.text:
            return False
        text_lower = self.text.lower()
        return any(marker in text_lower for marker in _TOOL_MARKERS)

    @property
    def is_thinking(self) -> bool:
        """Message contains reasoning/thinking traces."""
        if self.provider_meta and self.provider_meta.get("isThought"):
            return True
        if not self.text:
            return False
        text_lower = self.text.lower()
        return any(marker in text_lower for marker in _THINKING_MARKERS)

    @property
    def is_context_dump(self) -> bool:
        """Message is primarily context/file content, not dialogue."""
        if not self.text:
            return False
        if len(self.attachments) > 0 and (not self.text or len(self.text) < 100):
            return True
        for pattern in _CONTEXT_PATTERNS:
            if re.search(pattern, self.text, re.MULTILINE):
                return True
        return False

    @property
    def is_noise(self) -> bool:
        """Message is noise (tool results, context dumps, system)."""
        return self.is_tool_use or self.is_context_dump or self.is_system

    @property
    def is_substantive(self) -> bool:
        """Message has substantive dialogue content."""
        return self.is_dialogue and not self.is_noise and bool(self.text and len(self.text.strip()) > 10)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        if not self.text:
            return 0
        return len(self.text.split())

    def extract_thinking(self) -> str | None:
        """Extract thinking content if present."""
        if not self.text:
            return None
        match = re.search(r"<(?:antml:)?thinking>(.*?)</(?:antml:)?thinking>", self.text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None


class DialoguePair(BaseModel):
    """A user message followed by assistant response."""
    user: Message
    assistant: Message
    
    @property
    def exchange(self) -> str:
        """Render as text exchange."""
        return f"User: {self.user.text}\n\nAssistant: {self.assistant.text}"


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
        att_map: dict[str, list[AttachmentRecord]] = {}
        for att in attachments:
            if att.message_id:
                att_map.setdefault(att.message_id, []).append(att)

        rich_messages = [
            Message.from_record(msg, att_map.get(msg.message_id, []))
            for msg in messages
        ]

        return cls(
            id=conversation.conversation_id,
            provider=conversation.provider_name,
            title=conversation.title,
            messages=rich_messages,
            created_at=parse_timestamp(conversation.created_at),
            updated_at=parse_timestamp(conversation.updated_at),
            provider_meta=conversation.provider_meta,
        )

    # --- Filtering views ---

    def filter(self, predicate: Callable[[Message], bool]) -> Conversation:
        """Return a view with messages matching predicate."""
        return self.model_copy(update={"messages": [m for m in self.messages if predicate(m)]})

    def user_only(self) -> Conversation:
        """Return a view with only user messages."""
        return self.filter(lambda m: m.is_user)

    def assistant_only(self) -> Conversation:
        """Return a view with only assistant messages."""
        return self.filter(lambda m: m.is_assistant)

    def dialogue_only(self) -> Conversation:
        """Return a view with only user/assistant dialogue (no system)."""
        return self.filter(lambda m: m.is_dialogue)

    def without_noise(self) -> Conversation:
        """Return a view excluding tool calls, context dumps, system messages."""
        return self.filter(lambda m: not m.is_noise)

    def substantive_only(self) -> Conversation:
        """Return a view with only substantive dialogue."""
        return self.filter(lambda m: m.is_substantive)

    def without_attachments(self) -> Conversation:
        """Return a view with attachments stripped from messages."""
        new_msgs = [m.model_copy(update={"attachments": []}) for m in self.messages]
        return self.model_copy(update={"messages": new_msgs})

    # --- Iteration helpers ---

    def iter_dialogue(self) -> Iterator[Message]:
        """Iterate over dialogue messages only."""
        for m in self.messages:
            if m.is_dialogue:
                yield m

    def iter_substantive(self) -> Iterator[Message]:
        """Iterate over substantive messages only."""
        for m in self.messages:
            if m.is_substantive:
                yield m

    def iter_pairs(self) -> Iterator[DialoguePair]:
        """Iterate over user/assistant dialogue pairs."""
        substantive = [m for m in self.messages if m.is_substantive]
        i = 0
        while i < len(substantive) - 1:
            if substantive[i].is_user and substantive[i + 1].is_assistant:
                yield DialoguePair(user=substantive[i], assistant=substantive[i + 1])
                i += 2
            else:
                i += 1

    def iter_thinking(self) -> Iterator[str]:
        """Iterate over thinking/reasoning traces."""
        for m in self.messages:
            if m.is_thinking:
                thinking = m.extract_thinking()
                if thinking:
                    yield thinking

    # --- Rendering ---

    def to_text(self, include_role: bool = True) -> str:
        """Render conversation to plain text."""
        lines = []
        for m in self.messages:
            if not m.text:
                continue
            if include_role:
                lines.append(f"{m.role}: {m.text}")
            else:
                lines.append(m.text)
        return "\n\n".join(lines)

    def to_clean_text(self) -> str:
        """Render only substantive dialogue to text."""
        return self.substantive_only().to_text()

    # --- Statistics ---

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_message_count(self) -> int:
        return sum(1 for m in self.messages if m.is_user)

    @property
    def assistant_message_count(self) -> int:
        return sum(1 for m in self.messages if m.is_assistant)

    @property
    def word_count(self) -> int:
        return sum(m.word_count for m in self.messages)

    @property
    def substantive_word_count(self) -> int:
        return sum(m.word_count for m in self.messages if m.is_substantive)
