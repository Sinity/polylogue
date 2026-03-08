"""Semantic models for conversations, messages, and attachments.

This module defines the core domain models used throughout polylogue.
The key types are:

- `Message`: A single message with classification properties (`is_user`,
  `is_assistant`, `is_tool_use`, `is_thinking`, `is_substantive`)

- `Conversation`: A conversation with filtering views (`substantive_only()`,
  `without_noise()`) and iteration helpers (`iter_pairs()`, `iter_thinking()`)

- `DialoguePair`: A user message paired with its assistant response

- `Attachment`: File attachments associated with messages

These models support "semantic projections" - views over the data that
filter and transform based on semantic meaning rather than raw structure.

Example:
    # Get only substantive dialogue (no tool calls, context dumps)
    clean = conversation.substantive_only()

    # Iterate over user/assistant turn pairs
    for pair in conversation.iter_pairs():
        print(f"Q: {pair.user.text[:50]}")
        print(f"A: {pair.assistant.text[:50]}")

    # Check message classification
    if message.is_thinking:
        print("This is a reasoning trace")
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from polylogue.lib.branch_type import BranchType
from polylogue.lib.log import get_logger
from polylogue.lib.messages import MessageCollection
from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import ConversationId, MessageId, Provider

if TYPE_CHECKING:
    from polylogue.lib.projections import ConversationProjection

logger = get_logger(__name__)


class Attachment(BaseModel):
    id: str
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    provider_meta: dict[str, object] | None = None

    @classmethod
    def from_record(cls, record: AttachmentRecord) -> Attachment:
        name = record.provider_meta.get("name") if record.provider_meta else None
        return cls(
            id=record.attachment_id,
            name=name if isinstance(name, str) else record.attachment_id,
            mime_type=record.mime_type,
            size_bytes=record.size_bytes,
            path=record.path,
            provider_meta=record.provider_meta,
        )


# Patterns for context dump detection (kept for is_context_dump).
# Only patterns NOT already handled by fast inline checks in is_context_dump.
_CONTEXT_PATTERNS = [
    r"^Contents of .+:",
    r"^<file path=",
]


class Message(BaseModel):
    id: str
    role: Role
    text: str | None = None
    timestamp: datetime | None = None
    provider: Provider | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
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

    @classmethod
    def from_record(
        cls,
        record: MessageRecord,
        attachments: list[AttachmentRecord],
        *,
        provider: Provider | str | None = None,
    ) -> Message:
        ts = parse_timestamp(record.timestamp)
        return cls(
            id=record.message_id,
            role=(record.role or "").strip() or "unknown",
            text=record.text,
            timestamp=ts,
            attachments=[Attachment.from_record(a) for a in attachments],
            provider_meta=record.provider_meta,
            parent_id=record.parent_message_id,
            branch_index=record.branch_index,
        )

    # --- Branching properties ---

    @property
    def is_branch(self) -> bool:
        """True if this message is a branch (not mainline)."""
        return self.branch_index > 0

    # --- Role classification ---

    @property
    def is_user(self) -> bool:
        """Message is from the user."""
        return self.role == Role.USER

    @property
    def is_assistant(self) -> bool:
        """Message is from the assistant."""
        return self.role == Role.ASSISTANT

    @property
    def is_system(self) -> bool:
        """Message is a system prompt."""
        return self.role == Role.SYSTEM

    @property
    def is_dialogue(self) -> bool:
        """True if this is a user or assistant message (not system/tool)."""
        return self.is_user or self.is_assistant

    # --- Content classification (dynamic, heuristic-based) ---

    @cached_property
    def harmonized(self):
        """Harmonized viewports extracted from provider_meta when provider is known."""
        if not self.provider or not self.provider_meta:
            return None
        try:
            return extract_from_provider_meta(
                self.provider,
                self.provider_meta,
                message_id=self.id,
                role=self.role,
                text=self.text,
                timestamp=self.timestamp,
            )
        except Exception:
            logger.warning(
                "Failed to extract harmonized viewports for message %s (provider=%s)",
                self.id,
                self.provider,
                exc_info=True,
            )
            return None

    def _is_chatgpt_thinking(self) -> bool:
        """Check if this is a ChatGPT reasoning model thinking trace."""
        if not self.provider_meta:
            return False
        raw = self.provider_meta.get("raw", {})
        if not isinstance(raw, dict):
            return False

        # Check content_type for thoughts/reasoning_recap (any role)
        content = raw.get("content", {})
        if isinstance(content, dict):
            content_type = content.get("content_type", "")
            if content_type in ("thoughts", "reasoning_recap"):
                return True

        # Legacy: role=tool with finished_text in metadata
        if self.role == Role.TOOL:
            metadata = raw.get("metadata", {})
            if isinstance(metadata, dict) and "finished_text" in metadata:
                return True

        return False

    @property
    def is_tool_use(self) -> bool:
        """Message contains tool/function calls or results."""
        # Check structured content_blocks (populated at parse time)
        if self.provider_meta:
            blocks = self.provider_meta.get("content_blocks", [])
            if isinstance(blocks, list) and any(
                isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in blocks
            ):
                return True

        # ChatGPT role=tool: distinguish thinking from actual tools
        if self.role == Role.TOOL:
            return not self._is_chatgpt_thinking()

        # Claude-code sidechain/meta markers
        meta = self.provider_meta or {}
        raw = meta.get("raw", meta)
        return bool(raw.get("isSidechain") or raw.get("isMeta"))

    @property
    def is_thinking(self) -> bool:
        """Message contains reasoning/thinking traces."""
        # Check structured content_blocks (populated at parse time)
        if self.provider_meta:
            blocks = self.provider_meta.get("content_blocks", [])
            if isinstance(blocks, list) and any(isinstance(b, dict) and b.get("type") == "thinking" for b in blocks):
                return True

        # Gemini isThought marker (from raw data)
        if self.provider_meta:
            if self.provider_meta.get("isThought"):
                return True
            raw = self.provider_meta.get("raw", {})
            if isinstance(raw, dict) and raw.get("isThought"):
                return True

        # ChatGPT content_type check (from raw data)
        return bool(self._is_chatgpt_thinking())

    @property
    def is_context_dump(self) -> bool:
        """Message is primarily context/file content, not dialogue."""
        if not self.text:
            return False
        if len(self.attachments) > 0 and len(self.text) < 100:
            return True
        # System prompt content
        if "<system>" in self.text and "</system>" in self.text:
            return True
        # Multiple code fences (3+) suggest pasted file content
        code_fence_count = self.text.count("```")
        if code_fence_count >= 6:  # 3 complete fences = 6 backtick blocks
            return True
        return any(re.search(pattern, self.text, re.MULTILINE) for pattern in _CONTEXT_PATTERNS)

    @property
    def is_noise(self) -> bool:
        """Message is noise (tool results, context dumps, system)."""
        return self.is_tool_use or self.is_context_dump or self.is_system

    @property
    def is_substantive(self) -> bool:
        """Message has substantive dialogue content."""
        if not self.is_dialogue or self.is_noise or self.is_thinking:
            return False
        return bool(self.text and len(self.text.strip()) > 10)

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        if not self.text:
            return 0
        return len(self.text.split())

    # --- Cost/performance metadata (claude-code) ---

    @property
    def cost_usd(self) -> float | None:
        """Cost in USD (claude-code messages)."""
        if not self.provider_meta:
            return None
        raw = self.provider_meta.get("raw", self.provider_meta)
        cost = raw.get("costUSD")
        return float(cost) if isinstance(cost, (int, float)) else None

    @property
    def duration_ms(self) -> int | None:
        """Response duration in milliseconds (claude-code messages)."""
        if not self.provider_meta:
            return None
        raw = self.provider_meta.get("raw", self.provider_meta)
        duration = raw.get("durationMs")
        return int(duration) if isinstance(duration, (int, float)) else None

    def extract_thinking(self) -> str | None:
        """Extract thinking content if present.

        Checks (in order):
        1. Harmonized reasoning traces (Claude Code raw format)
        2. Structured content_blocks with type "thinking" (legacy format)
        3. XML <thinking> tags in message text (legacy/antml format)
        4. Full message text for Gemini isThought or ChatGPT thinking messages
        """
        # 1. Harmonized reasoning traces (Claude Code with raw provider_meta)
        harmonized = self.harmonized
        if harmonized and harmonized.reasoning_traces:
            texts = [t.text for t in harmonized.reasoning_traces if t.text]
            if texts:
                return "\n\n".join(texts).strip() or None

        # 2. Structured content_blocks (legacy format)
        if self.provider_meta:
            blocks = self.provider_meta.get("content_blocks", [])
            if isinstance(blocks, list):
                thinking_texts = [
                    b["text"]
                    for b in blocks
                    if isinstance(b, dict) and b.get("type") == "thinking" and isinstance(b.get("text"), str)
                ]
                if thinking_texts:
                    return "\n\n".join(thinking_texts).strip() or None

        # 3. XML tags in text (legacy/antml format)
        if self.text:
            match = re.search(r"<(?:antml:)?thinking>(.*?)</(?:antml:)?thinking>", self.text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # 4. Gemini/ChatGPT thinking: the message text IS the thinking content
        if self.text and (self._is_chatgpt_thinking() or (self.provider_meta and self.provider_meta.get("isThought"))):
            return self.text.strip() or None

        return None


class DialoguePair(BaseModel):
    """A user message followed by assistant response."""
    user: Message
    assistant: Message

    @model_validator(mode="after")
    def validate_roles(self) -> DialoguePair:
        """Ensure user message has user role and assistant message has assistant role."""
        if not self.user.is_user:
            raise ValueError(f"user message must have user role, got {self.user.role}")
        if not self.assistant.is_assistant:
            raise ValueError(f"assistant message must have assistant role, got {self.assistant.role}")
        return self

    @property
    def exchange(self) -> str:
        """Render as text exchange."""
        return f"User: {self.user.text or ''}\n\nAssistant: {self.assistant.text or ''}"


class ConversationSummary(BaseModel):
    """Lightweight conversation metadata without messages.

    Use this for listing/filtering operations that don't need message content.
    Much more memory efficient than loading full Conversation objects.
    """

    id: ConversationId
    provider: Provider
    title: str | None = None
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
    # Cached stats (populated from get_conversation_stats if available)
    message_count: int | None = None
    dialogue_count: int | None = None

    @classmethod
    def from_record(cls, record: ConversationRecord) -> ConversationSummary:
        """Create summary from ConversationRecord without loading messages."""
        return cls(
            id=record.conversation_id,
            provider=record.provider_name,
            title=record.title,
            created_at=parse_timestamp(record.created_at),
            updated_at=parse_timestamp(record.updated_at),
            provider_meta=record.provider_meta,
            metadata=record.metadata or {},
            parent_id=record.parent_conversation_id,
            branch_type=record.branch_type,
        )

    @property
    def display_date(self) -> datetime | None:
        """Best available date: updated_at > created_at > None."""
        return self.updated_at or self.created_at

    @property
    def display_title(self) -> str:
        """Display title with precedence: user_title > title > truncated ID."""
        user_title = self.metadata.get("title")
        if user_title:
            return str(user_title)
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def tags(self) -> list[str]:
        """List of tags from metadata."""
        tags = self.metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(t) for t in tags]
        return []

    @property
    def summary(self) -> str | None:
        """User-defined summary from metadata."""
        summary = self.metadata.get("summary")
        return str(summary) if summary is not None else None

    @property
    def is_continuation(self) -> bool:
        return self.branch_type == BranchType.CONTINUATION

    @property
    def is_sidechain(self) -> bool:
        return self.branch_type == BranchType.SIDECHAIN

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


class Conversation(BaseModel):
    """A conversation with messages and metadata.

    The `messages` field is a `MessageCollection` which supports both lazy
    eager loading. Messages are pre-loaded in memory via `Conversation.from_records()`
    or filter operations.

    You can also pass a list of Message objects directly — Pydantic coercion
    will auto-wrap them in an eager MessageCollection.
    """

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

    # Allow MessageCollection which is not a standard Pydantic type
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def display_date(self) -> datetime | None:
        """Best available date: updated_at > created_at > None."""
        return self.updated_at or self.created_at

    @classmethod
    def from_records(
        cls,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> Conversation:
        """Create a Conversation with eager-loaded messages.

        This is the traditional constructor that loads all messages into memory.
        Used for filtered views, tests, and when full message access is needed.

        Args:
            conversation: Conversation metadata record
            messages: List of message records
            attachments: List of attachment records

        Returns:
            Conversation with messages in eager mode
        """
        att_map: dict[MessageId, list[AttachmentRecord]] = {}
        for att in attachments:
            if att.message_id:
                att_map.setdefault(att.message_id, []).append(att)

        conv_provider = Provider.from_string(conversation.provider_name)
        rich_messages = [
            Message.from_record(
                msg,
                att_map.get(msg.message_id, []),
                provider=conv_provider,
            )
            for msg in messages
        ]

        return cls(
            id=conversation.conversation_id,
            provider=conv_provider,
            title=conversation.title,
            messages=MessageCollection(messages=rich_messages),
            created_at=parse_timestamp(conversation.created_at),
            updated_at=parse_timestamp(conversation.updated_at),
            provider_meta=conversation.provider_meta,
            metadata=conversation.metadata or {},
            parent_id=conversation.parent_conversation_id,
            branch_type=conversation.branch_type,
        )

    # --- Branching properties ---

    @property
    def is_continuation(self) -> bool:
        """True if this is a continuation of another session."""
        return self.branch_type == BranchType.CONTINUATION

    @property
    def is_sidechain(self) -> bool:
        """True if this is a sidechain conversation."""
        return self.branch_type == BranchType.SIDECHAIN

    @property
    def is_root(self) -> bool:
        """True if this conversation has no parent (is a root)."""
        return self.parent_id is None

    # --- Metadata properties ---

    @property
    def user_title(self) -> str | None:
        """User-defined title override from metadata."""
        title = self.metadata.get("title")
        return str(title) if title is not None else None

    @property
    def display_title(self) -> str:
        """Display title with precedence: user_title > title > truncated ID."""
        if self.user_title:
            return self.user_title
        if self.title:
            return self.title
        return self.id[:8]

    @property
    def summary(self) -> str | None:
        """User-defined summary from metadata."""
        summary = self.metadata.get("summary")
        return str(summary) if summary is not None else None

    @property
    def tags(self) -> list[str]:
        """List of tags from metadata."""
        tags = self.metadata.get("tags", [])
        if isinstance(tags, list):
            return [str(t) for t in tags]
        return []

    # --- Filtering views ---

    def filter(self, predicate: Callable[[Message], bool]) -> Conversation:
        """Return a view with messages matching predicate.

        Note: This materializes messages to apply the filter, returning
        a new Conversation with an eager MessageCollection.
        """
        filtered = [m for m in self.messages if predicate(m)]
        return self.model_copy(update={"messages": MessageCollection(messages=filtered)})

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

    def mainline_messages(self) -> list[Message]:
        """Return only mainline messages (branch_index == 0)."""
        return [m for m in self.messages if m.branch_index == 0]

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

    def iter_branches(self) -> Iterator[tuple[str, list[Message]]]:
        """Iterate over branch groups (messages sharing the same parent).

        Yields tuples of (parent_id, branch_messages) for each parent that has
        multiple children (branches). Only includes actual branches where there
        are 2+ messages with the same parent_id.

        Example:
            for parent_id, branches in conv.iter_branches():
                print(f"Parent {parent_id} has {len(branches)} branches")
                for msg in branches:
                    print(f"  Branch {msg.branch_index}: {msg.text[:50]}")
        """
        from collections import defaultdict

        # Group messages by parent_id
        by_parent: dict[str, list[Message]] = defaultdict(list)
        for m in self.messages:
            if m.parent_id:
                by_parent[m.parent_id].append(m)

        # Yield only parents with multiple children (actual branches)
        for parent_id, children in by_parent.items():
            if len(children) > 1:
                # Sort by branch_index for consistent ordering
                sorted_children = sorted(children, key=lambda m: m.branch_index)
                yield parent_id, sorted_children

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
    def total_cost_usd(self) -> float:
        """Total cost in USD (sum of all message costs)."""
        return sum(m.cost_usd or 0.0 for m in self.messages)

    @property
    def total_duration_ms(self) -> int:
        """Total response duration in milliseconds."""
        return sum(m.duration_ms or 0 for m in self.messages)

    # --- Projection API ---

    def project(self) -> ConversationProjection:
        """Create a projection builder for lazy, composable filtering.

        Example:
            conv.project().substantive().min_words(50).execute()
        """
        from polylogue.lib.projections import ConversationProjection
        return ConversationProjection(self)


__all__ = [
    "Attachment",
    "Conversation",
    "ConversationSummary",
    "DialoguePair",
    "Message",
]
