"""Typed viewport models that abstract across provider formats."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from polylogue.lib.roles import Role
from polylogue.lib.viewport.enums import ContentType, ToolCategory
from polylogue.lib.viewport.tools import (
    PATH_PATTERN,
    clean_metadata_path_candidate,
    clean_path_candidate,
    clean_shell_path_candidate,
)
from polylogue.types import Provider


class ReasoningTrace(BaseModel):
    text: str
    duration_ms: int | None = None
    token_count: int | None = None
    provider: Provider | None = None
    raw: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, value: object) -> Provider | None:
        if value is None:
            return None
        if isinstance(value, Provider):
            return value
        return Provider.from_string(str(value))


class ToolCall(BaseModel):
    name: str
    id: str | None = None
    input: Mapping[str, object] = Field(default_factory=dict)
    output: str | None = None
    success: bool | None = None
    category: ToolCategory = ToolCategory.OTHER
    provider: Provider | None = None
    raw: Mapping[str, object] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, value: object) -> Provider | None:
        if value is None:
            return None
        if isinstance(value, Provider):
            return value
        return Provider.from_string(str(value))

    @property
    def is_file_operation(self) -> bool:
        return self.category in (
            ToolCategory.FILE_READ,
            ToolCategory.FILE_WRITE,
            ToolCategory.FILE_EDIT,
        )

    @property
    def is_git_operation(self) -> bool:
        return self.category == ToolCategory.GIT

    @property
    def is_subagent(self) -> bool:
        return self.category == ToolCategory.SUBAGENT

    @property
    def affected_paths(self) -> list[str]:
        paths: list[str] = []
        for field in ("file_path", "path", "file", "filename"):
            value = clean_path_candidate(self.input.get(field))
            if value:
                paths.append(value)

        metadata = self.raw.get("metadata") if isinstance(self.raw, dict) else None
        if isinstance(metadata, dict):
            for field in ("path", "file_path"):
                value = clean_metadata_path_candidate(metadata.get(field))
                if value:
                    paths.append(value)
            files = metadata.get("files")
            if isinstance(files, list):
                for item in files:
                    value = clean_metadata_path_candidate(item)
                    if value:
                        paths.append(value)

        command = self.input.get("command")
        if isinstance(command, str):
            for candidate in PATH_PATTERN.findall(command)[:8]:
                value = clean_shell_path_candidate(candidate)
                if value:
                    paths.append(value)

        return list(dict.fromkeys(paths))


class ContentBlock(BaseModel):
    type: ContentType
    text: str | None = None
    language: str | None = None
    url: str | None = None
    mime_type: str | None = None
    tool_call: ToolCall | None = None
    raw: Mapping[str, object] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None


class CostInfo(BaseModel):
    total_usd: float | None = None
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    model: str | None = None


class MessageMeta(BaseModel):
    id: str | None = None
    timestamp: datetime | None = None
    role: Role = Role.UNKNOWN
    model: str | None = None
    tokens: TokenUsage | None = None
    cost: CostInfo | None = None
    duration_ms: int | None = None
    provider: Provider | None = None

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, value: object) -> Role:
        if isinstance(value, Role):
            return value
        raw = str(value).strip() if value is not None else ""
        return Role.normalize(raw) if raw else Role.UNKNOWN

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, value: object) -> Provider | None:
        if value is None:
            return None
        if isinstance(value, Provider):
            return value
        return Provider.from_string(str(value))


__all__ = [
    "ContentBlock",
    "CostInfo",
    "MessageMeta",
    "ReasoningTrace",
    "TokenUsage",
    "ToolCall",
]
