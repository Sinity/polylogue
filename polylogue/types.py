"""Type aliases and enums for polylogue."""
from __future__ import annotations

from enum import Enum
from typing import NewType

from polylogue.lib.provider_identity import canonical_runtime_provider

# Semantic ID types - provides compile-time distinction
ConversationId = NewType("ConversationId", str)
MessageId = NewType("MessageId", str)
AttachmentId = NewType("AttachmentId", str)
ContentHash = NewType("ContentHash", str)


class Provider(str, Enum):
    """Known conversation providers."""

    CHATGPT = "chatgpt"
    CLAUDE_AI = "claude-ai"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    GEMINI = "gemini"
    DRIVE = "drive"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | Provider | None) -> Provider:
        """Normalize provider string to enum, defaulting to UNKNOWN."""
        normalized = canonical_runtime_provider(str(value) if value is not None else None)
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN

    def __str__(self) -> str:
        return self.value


class ContentBlockType(str, Enum):
    """Canonical stored and parsed content block kinds."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    CODE = "code"
    DOCUMENT = "document"

    @classmethod
    def from_string(cls, value: str | ContentBlockType) -> ContentBlockType:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class SemanticBlockType(str, Enum):
    """Canonical semantic classifications for stored content blocks."""

    OTHER = "other"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_EDIT = "file_edit"
    SHELL = "shell"
    GIT = "git"
    SEARCH = "search"
    WEB = "web"
    AGENT = "agent"
    SUBAGENT = "subagent"
    THINKING = "thinking"

    @classmethod
    def from_string(cls, value: str | SemanticBlockType) -> SemanticBlockType:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class ValidationStatus(str, Enum):
    """Persisted raw-schema validation outcome."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def from_string(cls, value: str | ValidationStatus) -> ValidationStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class ValidationMode(str, Enum):
    """Configured raw-schema validation strictness."""

    OFF = "off"
    ADVISORY = "advisory"
    STRICT = "strict"

    @classmethod
    def from_string(cls, value: str | ValidationMode) -> ValidationMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class ArtifactSupportStatus(str, Enum):
    """Durable support state for an observed raw artifact."""

    SUPPORTED_PARSEABLE = "supported_parseable"
    RECOGNIZED_UNPARSED = "recognized_unparsed"
    UNSUPPORTED_PARSEABLE = "unsupported_parseable"
    DECODE_FAILED = "decode_failed"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | ArtifactSupportStatus) -> ArtifactSupportStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class PlanStage(str, Enum):
    """Supported ingest/runtime planning stages."""

    ALL = "all"
    ACQUIRE = "acquire"
    CUSTOM = "custom"
    PARSE = "parse"
    MATERIALIZE = "materialize"
    INDEX = "index"
    RENDER = "render"
    SCHEMA = "schema"
    REPROCESS = "reprocess"

    @classmethod
    def from_string(cls, value: str | PlanStage) -> PlanStage:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class SearchProvider(str, Enum):
    """Supported static-site search backends."""

    PAGEFIND = "pagefind"
    LUNR = "lunr"

    @classmethod
    def from_string(cls, value: str | SearchProvider) -> SearchProvider:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


class ExerciseIOMode(str, Enum):
    """Showcase exercise input/output mutability mode."""

    READ = "read"
    WRITE = "write"
    IDEMPOTENT = "idempotent"

    @classmethod
    def from_string(cls, value: str | ExerciseIOMode) -> ExerciseIOMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


__all__ = [
    "AttachmentId",
    "ArtifactSupportStatus",
    "ContentBlockType",
    "ContentHash",
    "ConversationId",
    "ExerciseIOMode",
    "MessageId",
    "PlanStage",
    "Provider",
    "SearchProvider",
    "SemanticBlockType",
    "ValidationMode",
    "ValidationStatus",
]
