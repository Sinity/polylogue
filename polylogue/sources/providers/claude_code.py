"""Claude Code provider-specific typed models.

These models match the Claude Code JSONL session format exactly.
Derived from schema: polylogue/schemas/providers/claude-code.schema.json

Claude Code sessions are JSONL files where each line is a record.
Records have a `type` field that determines their structure.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    CostInfo,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
    classify_tool,
)
from polylogue.schemas.unified import (
    extract_content_blocks as _extract_content_blocks,
)
from polylogue.schemas.unified import (
    extract_reasoning_traces as _extract_reasoning_traces,
)
from polylogue.schemas.unified import (
    extract_tool_calls as _extract_tool_calls,
)


class ClaudeCodeToolUse(BaseModel):
    """A tool_use content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)

    def to_tool_call(self) -> ToolCall:
        """Convert to harmonized ToolCall."""
        return ToolCall(
            name=self.name,
            id=self.id,
            input=self.input,
            category=classify_tool(self.name, self.input),
            provider="claude-code",
            raw=self.model_dump(),
        )


class ClaudeCodeToolResult(BaseModel):
    """A tool_result content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[Any] = ""
    is_error: bool = False


class ClaudeCodeTextBlock(BaseModel):
    """A text content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["text"] = "text"
    text: str


class ClaudeCodeThinkingBlock(BaseModel):
    """A thinking content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["thinking"] = "thinking"
    thinking: str

    def to_reasoning_trace(self) -> ReasoningTrace:
        """Convert to harmonized ReasoningTrace."""
        return ReasoningTrace(
            text=self.thinking,
            provider="claude-code",
            raw=self.model_dump(),
        )


class ClaudeCodeUsage(BaseModel):
    """Token usage from Claude Code response."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None

    def to_token_usage(self) -> TokenUsage:
        """Convert to harmonized TokenUsage."""
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_input_tokens,
            cache_write_tokens=self.cache_creation_input_tokens,
        )


class ClaudeCodeMessageContent(BaseModel):
    """Message content from Claude Code (assistant turn)."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    type: str = "message"
    role: str
    model: str | None = None
    content: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: ClaudeCodeUsage | None = None


class ClaudeCodeUserMessage(BaseModel):
    """User message content from Claude Code."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user"] = "user"
    content: str | list[dict[str, Any]] = ""


class ClaudeCodeRecord(BaseModel):
    """A single record from a Claude Code JSONL session.

    Records have different structures based on their `type`:
    - "user": User message
    - "assistant": Assistant response
    - "progress": Tool execution progress
    - "summary": Context compaction summary
    - "init": Session initialization
    - "file-history-snapshot": File backup snapshot
    """

    model_config = ConfigDict(extra="allow")

    # Common fields
    type: str
    """Record type: user, assistant, progress, summary, init, etc."""

    uuid: str | None = None
    """Unique record identifier."""

    parentUuid: str | None = None
    """Parent record UUID (for tree structure)."""

    timestamp: str | int | float | None = None
    """Timestamp - can be ISO string or Unix milliseconds."""

    sessionId: str | None = None
    """Session identifier."""

    # Message content (for user/assistant types)
    message: ClaudeCodeMessageContent | ClaudeCodeUserMessage | dict[str, Any] | None = None
    """Message content."""

    # Session metadata
    cwd: str | None = None
    """Current working directory."""

    gitBranch: str | None = None
    """Git branch name."""

    version: str | None = None
    """Claude Code version."""

    # Cost/performance
    costUSD: float | None = None
    """API cost in USD."""

    durationMs: int | None = None
    """Response time in milliseconds."""

    # Flags
    isSidechain: bool = False
    """Whether this is a sidechain (branch) conversation."""

    isMeta: bool = False
    """Whether this is a meta/system message."""

    # =========================================================================
    # Viewport extraction methods
    # =========================================================================

    @property
    def parsed_timestamp(self) -> datetime | None:
        """Parse timestamp to datetime."""
        if self.timestamp is None:
            return None
        # Handle millisecond timestamps (Claude Code uses Unix ms)
        if isinstance(self.timestamp, (int, float)) and float(self.timestamp) > 1e11:
            return parse_timestamp(float(self.timestamp) / 1000.0)
        return parse_timestamp(self.timestamp)

    @property
    def role(self) -> str:
        """Extract role from record.

        Claude Code JSONL record types map to roles:
        - user → user
        - assistant → assistant
        - summary, system, file-history-snapshot, queue-operation → system
        - progress, result → tool
        """
        if self.type == "user":
            return "user"
        if self.type == "assistant":
            return "assistant"
        if self.type in {"summary", "system", "file-history-snapshot", "queue-operation"}:
            return "system"
        if self.type in {"progress", "result"}:
            return "tool"
        return "unknown"

    @property
    def text_content(self) -> str:
        """Extract plain text content.

        For user/assistant records, text lives in self.message.content.
        For system records (compact_boundary, local_command), text lives
        in a top-level 'content' field stored as Pydantic extra.
        """
        if not self.message:
            # Fall back to top-level content field (system records)
            top_content = getattr(self, "content", None)
            if isinstance(top_content, str):
                return top_content
            return ""

        if isinstance(self.message, dict):
            content = self.message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "thinking":
                            texts.append(f"[Thinking: {block.get('thinking', '')[:100]}...]")
                return "\n".join(texts)
            return ""

        # Typed message content
        if hasattr(self.message, "content"):
            content = self.message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                return "\n".join(texts)

        return ""

    @property
    def content_blocks_raw(self) -> list[dict[str, Any]]:
        """Get raw content blocks from message."""
        if not self.message:
            return []

        if isinstance(self.message, dict):
            content = self.message.get("content", [])
            if isinstance(content, list):
                return content
            return []

        if hasattr(self.message, "content"):
            content = self.message.content
            if isinstance(content, list):
                return content

        return []

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        """Extract all thinking/reasoning traces from this record."""
        return _extract_reasoning_traces(self.content_blocks_raw, "claude-code")

    def extract_tool_calls(self) -> list[ToolCall]:
        """Extract all tool invocations from this record."""
        return _extract_tool_calls(self.content_blocks_raw, "claude-code")

    def extract_content_blocks(self) -> list[ContentBlock]:
        """Extract harmonized content blocks."""
        return _extract_content_blocks(self.content_blocks_raw)

    def to_meta(self) -> MessageMeta:
        """Convert to harmonized MessageMeta."""
        # Extract token usage
        tokens = None
        if isinstance(self.message, ClaudeCodeMessageContent) and self.message.usage:
            tokens = self.message.usage.to_token_usage()
        elif isinstance(self.message, dict):
            usage_raw = self.message.get("usage", {})
            if usage_raw:
                tokens = TokenUsage(
                    input_tokens=usage_raw.get("input_tokens"),
                    output_tokens=usage_raw.get("output_tokens"),
                    cache_read_tokens=usage_raw.get("cache_read_input_tokens"),
                    cache_write_tokens=usage_raw.get("cache_creation_input_tokens"),
                )

        # Extract cost
        cost = None
        if self.costUSD is not None:
            cost = CostInfo(total_usd=self.costUSD)

        # Extract model
        model = None
        if isinstance(self.message, ClaudeCodeMessageContent):
            model = self.message.model
        elif isinstance(self.message, dict):
            model = self.message.get("model")

        return MessageMeta(
            id=self.uuid,
            timestamp=self.parsed_timestamp,
            role=self.role,
            model=model,
            tokens=tokens,
            cost=cost,
            duration_ms=self.durationMs,
            provider="claude-code",
        )

    @property
    def is_context_compaction(self) -> bool:
        """Check if this is a context compaction summary."""
        return self.type == "summary"

    @property
    def is_tool_progress(self) -> bool:
        """Check if this is a tool execution progress event."""
        return self.type == "progress"

    @property
    def is_actual_message(self) -> bool:
        """Check if this is an actual user/assistant message."""
        return self.type in ("user", "assistant")
