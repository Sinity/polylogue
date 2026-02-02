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
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

from polylogue.core.timestamps import parse_timestamp
from polylogue.storage.store import AttachmentRecord, ConversationRecord, MessageRecord
from polylogue.types import ConversationId, MessageId

if TYPE_CHECKING:
    from polylogue.lib.projections import ConversationProjection


class Role(str, Enum):
    """Canonical message roles across all providers."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, role: str) -> Role:
        """Convert role string to Role enum."""
        role_lower = role.lower() if role else ""
        mapping = {
            "user": cls.USER,
            "human": cls.USER,
            "assistant": cls.ASSISTANT,
            "model": cls.ASSISTANT,
            "system": cls.SYSTEM,
            "tool": cls.TOOL,
        }
        return mapping.get(role_lower, cls.UNKNOWN)


# =============================================================================
# Claude Code Semantic Models
# =============================================================================


class ThinkingTrace(BaseModel):
    """Structured thinking block with metadata.

    Represents Claude's internal reasoning as captured in <thinking> blocks.
    """

    text: str
    """The thinking/reasoning content."""

    duration_ms: int | None = None
    """Duration of the thinking phase in milliseconds (if available)."""

    token_count: int | None = None
    """Approximate token count of the thinking content."""


class ToolInvocation(BaseModel):
    """Structured tool use with parsed semantics.

    Represents a single tool call from Claude Code sessions,
    with derived semantic properties.
    """

    tool_name: str
    """Tool name (e.g., Bash, Read, Write, Edit, Glob, Grep, Task)."""

    tool_id: str
    """Unique identifier for this tool invocation."""

    input: dict[str, object]
    """Input parameters passed to the tool."""

    result: str | None = None
    """Result returned by the tool (if captured)."""

    success: bool | None = None
    """Whether the tool invocation succeeded."""

    @property
    def is_file_operation(self) -> bool:
        """True if this is a file read/write/edit operation."""
        return self.tool_name in {"Read", "Write", "Edit", "NotebookEdit"}

    @property
    def is_git_operation(self) -> bool:
        """True if this is a git command (Bash with git)."""
        if self.tool_name != "Bash":
            return False
        command = self.input.get("command", "")
        return isinstance(command, str) and command.strip().startswith("git ")

    @property
    def is_search_operation(self) -> bool:
        """True if this is a search operation."""
        return self.tool_name in {"Glob", "Grep", "WebSearch"}

    @property
    def is_subagent(self) -> bool:
        """True if this spawns a subagent."""
        return self.tool_name == "Task"

    @property
    def affected_paths(self) -> list[str]:
        """Extract file paths affected by this operation."""
        paths: list[str] = []

        if self.tool_name in {"Read", "Write", "Edit"}:
            path = self.input.get("file_path") or self.input.get("path")
            if isinstance(path, str):
                paths.append(path)

        elif self.tool_name == "Glob":
            pattern = self.input.get("pattern")
            if isinstance(pattern, str):
                paths.append(pattern)

        elif self.tool_name == "Bash":
            # Try to extract paths from common commands
            command = self.input.get("command", "")
            if isinstance(command, str):
                # Very basic extraction - could be enhanced
                for token in command.split():
                    if "/" in token and not token.startswith("-"):
                        paths.append(token)

        return paths


class GitOperation(BaseModel):
    """Parsed git operation from Bash tool use."""

    command: str
    """Git subcommand (commit, push, checkout, etc.)."""

    branch: str | None = None
    """Branch name if applicable."""

    commit_hash: str | None = None
    """Commit hash if applicable."""

    files: list[str] = Field(default_factory=list)
    """Files affected by the operation."""

    message: str | None = None
    """Commit message if this is a commit operation."""


class FileChange(BaseModel):
    """File modification extracted from Edit/Write tools."""

    path: str
    """Absolute path to the file."""

    operation: str
    """Operation type: read, write, edit, delete."""

    old_content: str | None = None
    """Previous content (for edits)."""

    new_content: str | None = None
    """New content (for writes/edits)."""


class SubagentSpawn(BaseModel):
    """Subagent spawn extracted from Task tool invocations."""

    agent_type: str
    """Type of subagent (e.g., Explore, Plan, Bash)."""

    prompt: str
    """Prompt/task given to the subagent."""

    description: str | None = None
    """Short description of what the agent will do."""

    run_in_background: bool = False
    """Whether this agent runs in background."""


class ContextCompaction(BaseModel):
    """Context compaction event."""

    timestamp: datetime | None = None
    """When the compaction occurred."""

    summary: str
    """Summary of compacted content."""

    messages_compacted: int | None = None
    """Number of messages that were compacted."""
    def from_string(cls, value: str | None) -> Role:
        """Normalize various role strings to canonical Role."""
        if not value:
            return cls.UNKNOWN
        normalized = value.lower().strip()
        # Handle provider variations
        if normalized in ("assistant", "model"):
            return cls.ASSISTANT
        if normalized in ("user", "human"):
            return cls.USER
        # Direct match
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


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


# Patterns for context dump detection (kept for is_context_dump)
_CONTEXT_PATTERNS = [
    r"^Contents of .+:",
    r"^<file path=",
    r"<system>.*</system>",  # System prompts pasted as context
    r"(```\w*\n.*?\n```.*?){3,}",  # 3+ code fences = context dump
]


class Message(BaseModel):
    id: str
    role: str
    text: str | None = None
    timestamp: datetime | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    provider_meta: dict[str, object] | None = None
    parent_id: str | None = None
    branch_index: int = 0

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
            parent_id=record.parent_message_id,
            branch_index=record.branch_index,
        )

    # --- Branching properties ---

    @property
    def is_branch(self) -> bool:
        """True if this message is a branch (not mainline)."""
        return self.branch_index > 0

    # --- Role classification ---

    @cached_property
    def role_enum(self) -> Role:
        """Get the normalized Role enum for this message."""
        return Role.from_string(self.role)

    @property
    def is_user(self) -> bool:
        """Message is from the user."""
        return self.role_enum == Role.USER

    @property
    def is_assistant(self) -> bool:
        """Message is from the assistant."""
        return self.role_enum == Role.ASSISTANT

    @property
    def is_system(self) -> bool:
        """Message is a system prompt."""
        return self.role_enum == Role.SYSTEM

    @property
    def is_dialogue(self) -> bool:
        """True if this is a user or assistant message (not system/tool)."""
        return self.is_user or self.is_assistant

    # --- Content classification (dynamic, heuristic-based) ---

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
        if self.role.lower() == "tool":
            metadata = raw.get("metadata", {})
            if isinstance(metadata, dict) and "finished_text" in metadata:
                return True

        return False

    @property
    def is_tool_use(self) -> bool:
        """Message contains tool/function calls or results."""
        # Check structured content_blocks (populated at ingest time)
        if self.provider_meta:
            blocks = self.provider_meta.get("content_blocks", [])
            if isinstance(blocks, list) and any(
                isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in blocks
            ):
                return True

        # ChatGPT role=tool: distinguish thinking from actual tools
        if self.role.lower() == "tool":
            return not self._is_chatgpt_thinking()

        # Claude-code sidechain/meta markers (from raw data)
        if self.provider_meta:
            raw = self.provider_meta.get("raw", {})
            if isinstance(raw, dict):
                if raw.get("isSidechain") or raw.get("isMeta"):
                    return True

        return False

    @property
    def is_thinking(self) -> bool:
        """Message contains reasoning/thinking traces."""
        # Check structured content_blocks (populated at ingest time)
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
        if self._is_chatgpt_thinking():
            return True

        return False

    @property
    def is_context_dump(self) -> bool:
        """Message is primarily context/file content, not dialogue."""
        if not self.text:
            return False
        if len(self.attachments) > 0 and (not self.text or len(self.text) < 100):
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
        cost = self.provider_meta.get("costUSD")
        return float(cost) if isinstance(cost, (int, float)) else None

    @property
    def duration_ms(self) -> int | None:
        """Response duration in milliseconds (claude-code messages)."""
        if not self.provider_meta:
            return None
        duration = self.provider_meta.get("durationMs")
        return int(duration) if isinstance(duration, (int, float)) else None

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
        return f"User: {self.user.text}\n\nAssistant: {self.assistant.text}"


class Conversation(BaseModel):
    id: ConversationId
    provider: str
    title: str | None = None
    messages: list[Message]
    created_at: datetime | None = None
    updated_at: datetime | None = None
    provider_meta: dict[str, object] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    parent_id: ConversationId | None = None
    branch_type: str | None = None  # "continuation", "sidechain", "fork"

    @classmethod
    def from_records(
        cls,
        conversation: ConversationRecord,
        messages: list[MessageRecord],
        attachments: list[AttachmentRecord],
    ) -> Conversation:
        att_map: dict[MessageId, list[AttachmentRecord]] = {}
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
            metadata=conversation.metadata or {},
            parent_id=conversation.parent_conversation_id,
            branch_type=conversation.branch_type,
        )

    # --- Branching properties ---

    @property
    def is_continuation(self) -> bool:
        """True if this is a continuation of another session."""
        return self.branch_type == "continuation"

    @property
    def is_sidechain(self) -> bool:
        """True if this is a sidechain conversation."""
        return self.branch_type == "sidechain"

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
