"""Claude Code session record model and viewport extraction."""

from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from polylogue.lib.json import JSONDocument, JSONDocumentList, json_document, json_document_list
from polylogue.lib.provider_semantics import (
    extract_claude_code_text,
    extract_content_blocks,
    extract_reasoning_traces,
    extract_tool_calls,
)
from polylogue.lib.roles import Role, normalize_role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    CostInfo,
    MessageMeta,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
)
from polylogue.types import Provider

from .claude_code_models import ClaudeCodeMessageContent, ClaudeCodeUsage, ClaudeCodeUserMessage

ClaudeCodeMessagePayload: TypeAlias = ClaudeCodeMessageContent | ClaudeCodeUserMessage | dict[str, object] | None


def _usage_int(record: JSONDocument, key: str) -> int | None:
    value = record.get(key)
    return value if isinstance(value, int) else None


def _message_record(message: ClaudeCodeMessagePayload) -> JSONDocument:
    if isinstance(message, BaseModel):
        return json_document(message.model_dump())
    return json_document(message)


def _message_role(message: ClaudeCodeMessagePayload) -> str | None:
    role = _message_record(message).get("role")
    return role if isinstance(role, str) else None


def _message_content_value(message: ClaudeCodeMessagePayload) -> object:
    if isinstance(message, BaseModel):
        return getattr(message, "content", None)
    return _message_record(message).get("content")


def _message_content_blocks(message: ClaudeCodeMessagePayload) -> JSONDocumentList:
    return json_document_list(_message_content_value(message))


def _message_usage(message: ClaudeCodeMessagePayload) -> ClaudeCodeUsage | JSONDocument | None:
    if isinstance(message, ClaudeCodeMessageContent):
        return message.usage
    if isinstance(message, BaseModel):
        usage = _message_record(message).get("usage")
        return usage if isinstance(usage, dict) else None
    usage = _message_record(message).get("usage")
    return usage if isinstance(usage, dict) else None


def _message_model(message: ClaudeCodeMessagePayload) -> str | None:
    value = _message_record(message).get("model")
    return value if isinstance(value, str) else None


class ClaudeCodeRecord(BaseModel):
    """A single record from a Claude Code JSONL session."""

    model_config = ConfigDict(extra="allow")

    type: str
    subtype: str | None = None
    uuid: str | None = None
    parentUuid: str | None = None
    timestamp: str | int | float | None = None
    sessionId: str | None = None
    message: ClaudeCodeMessagePayload = None
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
        message_role = _message_role(self.message)
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

        content = _message_content_value(self.message)

        if isinstance(content, str):
            return content
        blocks = json_document_list(content)
        if blocks:
            return extract_claude_code_text(blocks)
        return ""

    @property
    def content_blocks_raw(self) -> JSONDocumentList:
        return _message_content_blocks(self.message)

    def extract_reasoning_traces(self) -> list[ReasoningTrace]:
        return extract_reasoning_traces(self.content_blocks_raw, "claude-code")

    def extract_tool_calls(self) -> list[ToolCall]:
        return extract_tool_calls(self.content_blocks_raw, "claude-code")

    def extract_content_blocks(self) -> list[ContentBlock]:
        return extract_content_blocks(self.content_blocks_raw)

    def to_meta(self) -> MessageMeta:
        tokens = None
        usage = _message_usage(self.message)
        if isinstance(usage, ClaudeCodeUsage):
            tokens = usage.to_token_usage()
        elif isinstance(usage, dict):
            tokens = TokenUsage(
                input_tokens=_usage_int(usage, "input_tokens"),
                output_tokens=_usage_int(usage, "output_tokens"),
                cache_read_tokens=_usage_int(usage, "cache_read_input_tokens"),
                cache_write_tokens=_usage_int(usage, "cache_creation_input_tokens"),
            )

        cost = None
        if self.costUSD is not None:
            cost = CostInfo(total_usd=self.costUSD)

        model = _message_model(self.message)

        return MessageMeta(
            id=self.uuid,
            timestamp=self.parsed_timestamp,
            role=Role.normalize(self.role) if self.role else Role.UNKNOWN,
            model=model,
            tokens=tokens,
            cost=cost,
            duration_ms=self.durationMs,
            provider=Provider.CLAUDE_CODE,
        )

    @property
    def is_context_compaction(self) -> bool:
        return self.type == "summary" or (self.type == "system" and self.subtype == "compact_boundary")

    @property
    def is_tool_progress(self) -> bool:
        return self.type == "progress"

    @property
    def is_actual_message(self) -> bool:
        return self.type in ("user", "assistant")
