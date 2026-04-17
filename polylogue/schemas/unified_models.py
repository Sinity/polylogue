"""Core harmonized semantic models and low-level token helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, NoReturn

from pydantic import BaseModel, Field, field_validator

from polylogue.lib.roles import Role
from polylogue.lib.viewports import ContentBlock, CostInfo, ReasoningTrace, TokenUsage, ToolCall
from polylogue.types import Provider


def _missing_role() -> NoReturn:
    """Called when role is missing - raises error to surface data quality issues."""
    raise ValueError("Message has no role. Data should be validated at import time.")


class HarmonizedMessage(BaseModel):
    """Unified message representation with viewport extractions."""

    id: str | None = None
    role: Role
    text: str
    timestamp: datetime | None = None
    reasoning_traces: list[ReasoningTrace] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    model: str | None = None
    tokens: TokenUsage | None = None
    cost: CostInfo | None = None
    duration_ms: int | None = None
    provider: Provider
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        raw = (str(v) if v is not None else "").strip() or "unknown"
        return Role.normalize(raw)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider:
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v) if v is not None else "unknown")

    @property
    def has_reasoning(self) -> bool:
        return len(self.reasoning_traces) > 0

    @property
    def has_tool_use(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def file_operations(self) -> list[ToolCall]:
        return [call for call in self.tool_calls if call.is_file_operation]

    @property
    def git_operations(self) -> list[ToolCall]:
        return [call for call in self.tool_calls if call.is_git_operation]


def extract_token_usage(usage: dict[str, Any] | None) -> TokenUsage | None:
    """Extract token usage from usage dict."""
    if not usage:
        return None

    return TokenUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cache_read_tokens=usage.get("cache_read_input_tokens"),
        cache_write_tokens=usage.get("cache_creation_input_tokens"),
        total_tokens=usage.get("total_tokens"),
    )


__all__ = ["HarmonizedMessage", "_missing_role", "extract_token_usage"]
