"""Shared enums for harmonized provider viewports."""

from __future__ import annotations

from enum import Enum


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


__all__ = ["ContentType", "ToolCategory"]
