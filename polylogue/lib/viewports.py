"""Harmonized viewport types that abstract across provider formats.

This module implements the "parse don't validate" philosophy:
1. Provider-specific data is parsed into typed structures
2. Viewports expose common semantics across all providers
3. The harmonized types enable provider-agnostic analysis
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from polylogue.lib.roles import Role
from polylogue.types import Provider

_PATH_PATTERN = re.compile(r'(?:^|[\s"\'])(/[^\s"\']+|[./][^\s"\']+)')
_NOISE_PATH_TOKENS = frozenset({"...", "//", "/dev/null", "|", "||", "&&", ";"})


class ContentType(str, Enum):
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
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    SHELL = "shell"
    GIT = "git"
    SEARCH = "search"
    WEB = "web"
    AGENT = "agent"
    SUBAGENT = "subagent"
    OTHER = "other"


class ReasoningTrace(BaseModel):
    text: str
    duration_ms: int | None = None
    token_count: int | None = None
    provider: Provider | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))


class ToolCall(BaseModel):
    name: str
    id: str | None = None
    input: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    success: bool | None = None
    category: ToolCategory = ToolCategory.OTHER
    provider: Provider | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

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
            value = _clean_path_candidate(self.input.get(field))
            if value:
                paths.append(value)

        metadata = self.raw.get("metadata") if isinstance(self.raw, dict) else None
        if isinstance(metadata, dict):
            for field in ("path", "file_path"):
                value = _clean_path_candidate(metadata.get(field))
                if value:
                    paths.append(value)
            files = metadata.get("files")
            if isinstance(files, list):
                for item in files:
                    value = _clean_path_candidate(item)
                    if value:
                        paths.append(value)

        cmd = self.input.get("command")
        if isinstance(cmd, str):
            for candidate in _PATH_PATTERN.findall(cmd)[:8]:
                value = _clean_shell_path_candidate(candidate)
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
    raw: dict[str, Any] = Field(default_factory=dict)


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
    def coerce_role(cls, v: object) -> Role:
        if isinstance(v, Role):
            return v
        raw = str(v).strip() if v is not None else ""
        return Role.normalize(raw) if raw else Role.UNKNOWN

    @field_validator("provider", mode="before")
    @classmethod
    def coerce_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))


def classify_tool(name: str, input_data: dict[str, Any]) -> ToolCategory:
    name_lower = name.lower()

    if name_lower in ("read", "view", "cat"):
        return ToolCategory.FILE_READ
    if "__get_file_contents" in name_lower:
        return ToolCategory.FILE_READ
    if name_lower in ("write", "create"):
        return ToolCategory.FILE_WRITE
    if name_lower in ("edit", "patch", "sed", "notebookedit", "multiedit") or "__edit_file" in name_lower or "__create_or_update_file" in name_lower:
        return ToolCategory.FILE_EDIT
    if name_lower in ("glob", "grep", "search", "find", "file_search", "ls", "toolsearch") or any(
        marker in name_lower
        for marker in (
            "__query-docs",
            "__get-library-docs",
            "__resolve-library-id",
            "__search_code",
            "__search_tool",
            "__search_repositories",
            "__search_users",
            "__search_issues",
            "__search_pull_requests",
        )
    ):
        return ToolCategory.SEARCH
    if name_lower in ("bash", "shell", "terminal", "run"):
        cmd = input_data.get("command", "")
        if isinstance(cmd, str) and cmd.strip().startswith("git "):
            return ToolCategory.GIT
        return ToolCategory.SHELL
    if name_lower in ("task", "subagent"):
        return ToolCategory.SUBAGENT
    if name_lower == "agent" or name_lower.startswith(("todo", "task")) or name_lower in (
        "askuserquestion",
        "exitplanmode",
        "batch",
        "mcp__sequential-thinking__sequentialthinking",
    ):
        return ToolCategory.AGENT
    if name_lower in ("web", "fetch", "browse", "webfetch", "websearch") or any(
        marker in name_lower for marker in ("__web_search", "__webfetch")
    ):
        return ToolCategory.WEB
    return ToolCategory.OTHER


def _clean_path_candidate(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().strip('"\'`')
    candidate = candidate.rstrip(",;:")
    if not candidate:
        return None
    if candidate in _NOISE_PATH_TOKENS:
        return None
    if set(candidate) <= {".", "/"}:
        return None
    if candidate.startswith("-"):
        return None
    if any(ch in candidate for ch in "*?[]{}"):
        return None
    return candidate


def _clean_shell_path_candidate(value: object) -> str | None:
    candidate = _clean_path_candidate(value)
    if candidate is None:
        return None
    if candidate.startswith(".") and "/" not in candidate:
        return None
    return candidate
