"""Harmonized viewport types that abstract across provider formats.

This module implements the "parse don't validate" philosophy:
1. Provider-specific data is parsed into typed structures
2. Viewports expose common semantics across all providers
3. The harmonized types enable provider-agnostic analysis

Architecture:
    ┌─────────────────────────────────────────────────┐
    │                  Viewports                       │
    │  (ReasoningTrace, ToolCall, ContentBlock, etc.) │
    └─────────────────────────────────────────────────┘
                          ▲
                          │ extract
    ┌─────────────────────┴─────────────────────────┐
    │              Provider Raw Types                │
    │  (ChatGPTMessage, ClaudeCodeRecord, etc.)     │
    └───────────────────────────────────────────────┘
                          ▲
                          │ parse
    ┌─────────────────────┴─────────────────────────┐
    │                 Raw JSON                       │
    │  (dict from JSON parsing)                      │
    └───────────────────────────────────────────────┘

The viewports are designed to answer questions like:
- "Show me all reasoning traces across all my Claude and ChatGPT conversations"
- "Find all file operations regardless of provider"
- "What's my total token usage across all providers?"

Example:
    # Extract viewports from any message
    for trace in message.reasoning_traces:
        print(f"Thinking: {trace.text[:100]}")

    for tool in message.tool_calls:
        if tool.is_file_operation:
            print(f"File op: {tool.operation_type} on {tool.affected_paths}")
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

_PATH_PATTERN = re.compile(r'(?:^|[\s"\'])(/[^\s"\']+|[./][^\s"\']+)')

# =============================================================================
# Enums for harmonized classification
# =============================================================================


class ContentType(str, Enum):
    """Unified content type classification across providers."""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    SYSTEM = "system"
    ERROR = "error"
    UNKNOWN = "unknown"


class ToolCategory(str, Enum):
    """High-level tool operation categories."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    SHELL = "shell"
    GIT = "git"
    SEARCH = "search"
    WEB = "web"
    AGENT = "agent"
    OTHER = "other"


# =============================================================================
# Core Viewport Types
# =============================================================================


class ReasoningTrace(BaseModel):
    """Harmonized reasoning/thinking trace across providers.

    Maps to:
    - Claude Code: <thinking> blocks
    - ChatGPT: (potential future chain-of-thought markers)
    - Other providers: Internal reasoning if exposed
    """

    text: str
    """The reasoning content."""

    duration_ms: int | None = None
    """Time spent on this reasoning phase."""

    token_count: int | None = None
    """Tokens used for this reasoning."""

    provider: str | None = None
    """Source provider (claude-code, chatgpt, etc.)"""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Original provider-specific data."""


class ToolCall(BaseModel):
    """Cross-provider harmonized tool invocation for rendering.

    This is a viewport type for uniform display across providers.
    For Claude Code-specific semantic analysis (is_file_operation,
    affected_paths, etc.), use lib.models.ToolInvocation instead.

    Maps to:
    - Claude Code: tool_use blocks (Bash, Read, Write, Edit, etc.)
    - ChatGPT: function_call / tool_calls
    - Codex: tool invocations
    """

    name: str
    """Tool name (Bash, Read, file_search, etc.)"""

    id: str | None = None
    """Unique tool call ID."""

    input: dict[str, Any] = Field(default_factory=dict)
    """Tool input parameters."""

    output: str | None = None
    """Tool result/output."""

    success: bool | None = None
    """Whether the tool call succeeded."""

    category: ToolCategory = ToolCategory.OTHER
    """High-level category for grouping."""

    provider: str | None = None
    """Source provider."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Original provider-specific data."""

    @property
    def is_file_operation(self) -> bool:
        """Check if this is a file-related operation."""
        return self.category in (
            ToolCategory.FILE_READ,
            ToolCategory.FILE_WRITE,
            ToolCategory.FILE_EDIT,
        )

    @property
    def is_git_operation(self) -> bool:
        """Check if this is a git operation."""
        return self.category == ToolCategory.GIT

    @property
    def affected_paths(self) -> list[str]:
        """Extract file paths affected by this operation."""
        paths = []

        # Common field names for paths
        for field in ("file_path", "path", "file", "filename", "command"):
            if field in self.input:
                val = self.input[field]
                if isinstance(val, str):
                    # For commands, try to extract paths
                    if field == "command":
                        # Simple heuristic: look for path-like strings
                        paths.extend(_PATH_PATTERN.findall(val)[:5])
                    else:
                        paths.append(val)

        return paths


class ContentBlock(BaseModel):
    """Harmonized content block across providers.

    Represents a single piece of content within a message.
    Messages may contain multiple blocks of different types.
    """

    type: ContentType
    """Content type classification."""

    text: str | None = None
    """Text content (for text, code, thinking types)."""

    language: str | None = None
    """Programming language (for code blocks)."""

    url: str | None = None
    """URL (for images, files, links)."""

    mime_type: str | None = None
    """MIME type (for files, images)."""

    tool_call: ToolCall | None = None
    """Nested tool call (for tool_use type)."""

    raw: dict[str, Any] = Field(default_factory=dict)
    """Original provider-specific data."""


class TokenUsage(BaseModel):
    """Harmonized token usage across providers."""

    input_tokens: int | None = None
    """Input/prompt tokens."""

    output_tokens: int | None = None
    """Output/completion tokens."""

    cache_read_tokens: int | None = None
    """Tokens read from cache (Claude)."""

    cache_write_tokens: int | None = None
    """Tokens written to cache (Claude)."""

    total_tokens: int | None = None
    """Total tokens (computed if not provided)."""

class CostInfo(BaseModel):
    """Harmonized cost information across providers."""

    total_usd: float | None = None
    """Total cost in USD."""

    input_cost_usd: float | None = None
    """Cost for input tokens."""

    output_cost_usd: float | None = None
    """Cost for output tokens."""

    model: str | None = None
    """Model used (for cost attribution)."""


class MessageMeta(BaseModel):
    """Harmonized message metadata across providers."""

    id: str | None = None
    """Provider message ID."""

    timestamp: datetime | None = None
    """Message timestamp."""

    role: Literal["user", "assistant", "system", "tool", "unknown"] = "unknown"
    """Normalized role."""

    model: str | None = None
    """Model used for this message."""

    tokens: TokenUsage | None = None
    """Token usage for this message."""

    cost: CostInfo | None = None
    """Cost information."""

    duration_ms: int | None = None
    """Response generation time."""

    provider: str | None = None
    """Source provider."""


# =============================================================================
# Extraction helpers
# =============================================================================


def classify_tool(name: str, input_data: dict[str, Any]) -> ToolCategory:
    """Classify a tool call into a high-level category.

    Args:
        name: Tool name (e.g., "Bash", "Read", "file_search")
        input_data: Tool input parameters

    Returns:
        ToolCategory for the operation
    """
    name_lower = name.lower()

    # File operations
    if name_lower in ("read", "view", "cat"):
        return ToolCategory.FILE_READ
    if name_lower in ("write", "create"):
        return ToolCategory.FILE_WRITE
    if name_lower in ("edit", "patch", "sed"):
        return ToolCategory.FILE_EDIT

    # Search operations
    if name_lower in ("glob", "grep", "search", "find", "file_search"):
        return ToolCategory.SEARCH

    # Shell operations
    if name_lower in ("bash", "shell", "terminal", "run"):
        # Check if it's a git command
        cmd = input_data.get("command", "")
        if isinstance(cmd, str) and cmd.strip().startswith("git "):
            return ToolCategory.GIT
        return ToolCategory.SHELL

    # Agent operations
    if name_lower in ("task", "agent", "subagent"):
        return ToolCategory.AGENT

    # Web operations
    if name_lower in ("web", "fetch", "browse", "webfetch", "websearch"):
        return ToolCategory.WEB

    return ToolCategory.OTHER
