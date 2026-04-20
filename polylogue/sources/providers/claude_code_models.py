"""Claude Code typed block and message support models."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from polylogue.lib.json import json_document
from polylogue.lib.viewports import ReasoningTrace, TokenUsage, ToolCall, classify_tool
from polylogue.types import Provider

ClaudeCodeContentBlockRecord: TypeAlias = dict[str, object]
ClaudeCodeContentBlocks: TypeAlias = list[ClaudeCodeContentBlockRecord]


class ClaudeCodeToolUse(BaseModel):
    """A tool_use content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: ClaudeCodeContentBlockRecord = Field(default_factory=dict)

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            name=self.name,
            id=self.id,
            input=self.input,
            category=classify_tool(self.name, json_document(self.input)),
            provider=Provider.CLAUDE_CODE,
            raw=self.model_dump(),
        )


class ClaudeCodeToolResult(BaseModel):
    """A tool_result content block from Claude Code."""

    model_config = ConfigDict(extra="allow")

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[object] = ""
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
        return ReasoningTrace(
            text=self.thinking,
            provider=Provider.CLAUDE_CODE,
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
    content: ClaudeCodeContentBlocks = Field(default_factory=list)
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: ClaudeCodeUsage | None = None


class ClaudeCodeUserMessage(BaseModel):
    """User message content from Claude Code."""

    model_config = ConfigDict(extra="allow")

    role: Literal["user"] = "user"
    content: str | ClaudeCodeContentBlocks = ""
