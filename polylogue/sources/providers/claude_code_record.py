"""Claude Code session record model and viewport extraction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from polylogue.lib.provider_semantics import (
    extract_claude_code_text,
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.lib.roles import normalize_role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    CostInfo,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
)

from .claude_code_models import ClaudeCodeMessageContent, ClaudeCodeUserMessage


class ClaudeCodeRecord(BaseModel):
    """A single record from a Claude Code JSONL session."""

    model_config = ConfigDict(extra="allow")

    type: str
    subtype: str | None = None
    uuid: str | None = None
    parentUuid: str | None = None
    timestamp: str | int | float | None = None
    sessionId: str | None = None
    message: ClaudeCodeMessageContent | ClaudeCodeUserMessage | dict[str, Any] | None = None
    cwd: str | None = None
    gitBranch: str | None = None
    version: str | None = None
    costUSD: float | None = None
    durationMs: int | None = None
    isSidechain: bool = False
    isMeta: bool = False

    @property
    def parsed_timestamp(self) -> datetime | None:
        if self.timestamp is None:
            return None
        if isinstance(self.timestamp, (int, float)) and float(self.timestamp) > 1e11:
            return parse_timestamp(float(self.timestamp) / 1000.0)
        return parse_timestamp(self.timestamp)

    @property
    def role(self) -> str:
        message_role = None
        if isinstance(self.message, dict):
            message_role = self.message.get("role")
        elif self.message is not None:
            message_role = getattr(self.message, "role", None)
        if isinstance(message_role, str) and message_role:
            try:
                normalized_role = normalize_role(message_role)
            except ValueError:
                pass
            else:
                if normalized_role != "unknown":
                    return normalized_role

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
    def role_normalized(self) -> str:
        return self.role

    @property
    def text_content(self) -> str:
        if not self.message:
            top_content = getattr(self, "content", None)
            return top_content if isinstance(top_content, str) else ""

        msg = self.message
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", None)

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return extract_claude_code_text(content)
        return ""

    @property
    def content_blocks_raw(self) -> list[dict[str, Any]]:
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
        return extract_reasoning_traces(self.content_blocks_raw, "claude-code")

    def extract_tool_calls(self) -> list[ToolCall]:
        return extract_tool_calls(self.content_blocks_raw, "claude-code")

    def extract_content_blocks(self) -> list[ContentBlock]:
        return extract_content_blocks(self.content_blocks_raw)

    def to_meta(self) -> MessageMeta:
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

        cost = None
        if self.costUSD is not None:
            cost = CostInfo(total_usd=self.costUSD)

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
        return self.type == "summary" or (
            self.type == "system" and self.subtype == "compact_boundary"
        )

    @property
    def is_tool_progress(self) -> bool:
        return self.type == "progress"

    @property
    def is_actual_message(self) -> bool:
        return self.type in ("user", "assistant")
