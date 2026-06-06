"""Closed vocabularies and SQL CHECK helpers for archive work."""

from __future__ import annotations

from enum import StrEnum

from polylogue.core.provider_identity import canonical_runtime_provider


class PolylogueStrEnum(StrEnum):
    """Base class for persisted string enums."""

    def __str__(self) -> str:
        return self.value


def enum_values(enum_type: type[PolylogueStrEnum]) -> tuple[str, ...]:
    """Return persisted values for a closed enum."""
    return tuple(item.value for item in enum_type)


def sql_string_literal(value: str) -> str:
    """Return a SQLite string literal for an enum value."""
    return "'" + value.replace("'", "''") + "'"


def sql_value_list(enum_type: type[PolylogueStrEnum]) -> str:
    """Return comma-separated SQLite literals for an enum CHECK."""
    return ", ".join(sql_string_literal(value) for value in enum_values(enum_type))


def sql_check_in(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return ``column IN (...)`` for a non-null enum column."""
    return f"{column} IN ({sql_value_list(enum_type)})"


def nullable_sql_check_in(column: str, enum_type: type[PolylogueStrEnum]) -> str:
    """Return a nullable enum CHECK expression."""
    return f"({sql_check_in(column, enum_type)} OR {column} IS NULL)"


class Origin(PolylogueStrEnum):
    """Archive source-origin tokens."""

    CLAUDE_CODE_SESSION = "claude-code-session"
    CODEX_SESSION = "codex-session"
    GEMINI_CLI_SESSION = "gemini-cli-session"
    HERMES_SESSION = "hermes-session"
    ANTIGRAVITY_SESSION = "antigravity-session"
    CHATGPT_EXPORT = "chatgpt-export"
    CLAUDE_AI_EXPORT = "claude-ai-export"
    AISTUDIO_DRIVE = "aistudio-drive"
    UNKNOWN_EXPORT = "unknown-export"

    @classmethod
    def from_string(cls, value: str | Origin | None) -> Origin:
        """Normalize an origin token to the enum, defaulting to UNKNOWN_EXPORT."""
        if value is None:
            return cls.UNKNOWN_EXPORT
        try:
            return cls(str(value))
        except ValueError:
            return cls.UNKNOWN_EXPORT


class Provider(PolylogueStrEnum):
    """Legacy runtime provider tokens retained during the archive transition."""

    CHATGPT = "chatgpt"
    CLAUDE_AI = "claude-ai"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    GEMINI = "gemini"
    GEMINI_CLI = "gemini-cli"
    HERMES = "hermes"
    ANTIGRAVITY = "antigravity"
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


class Role(PolylogueStrEnum):
    """Canonical session roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, raw: str) -> Role:
        """Normalize a provider role string to a canonical role."""
        lowered = raw.strip().lower()
        if not lowered:
            raise ValueError("Role cannot be empty. Handle missing roles at parse time.")

        if lowered in {"user", "human"}:
            return cls.USER
        if lowered in {"assistant", "model", "ai"}:
            return cls.ASSISTANT
        if lowered in {"system", "developer"}:
            return cls.SYSTEM
        if lowered in {"tool", "function", "tool_use", "tool_result", "progress", "result"}:
            return cls.TOOL
        return cls.UNKNOWN


class MessageType(PolylogueStrEnum):
    """Normalized message type for filtering and read surfaces."""

    MESSAGE = "message"
    SUMMARY = "summary"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    CONTEXT = "context"
    PROTOCOL = "protocol"

    @classmethod
    def normalize(cls, value: object) -> MessageType:
        """Coerce provider/parser message-type values to a canonical type."""
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        if not candidate:
            return cls.MESSAGE
        for item in cls:
            if item.value == candidate:
                return item
        return cls.MESSAGE

    @classmethod
    def validate_filter_token(cls, value: object) -> MessageType:
        """Validate one user-supplied message-type filter token."""
        if isinstance(value, MessageType):
            return value
        candidate = (str(value) if value is not None else "").strip().lower().replace("-", "_")
        for item in cls:
            if item.value == candidate:
                return item
        valid = ", ".join(item.value for item in cls)
        msg = f"Unknown message type {str(value)!r}. Valid message types: {valid}"
        raise ValueError(msg)


class ContentBlockType(PolylogueStrEnum):
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


class BlockType(PolylogueStrEnum):
    """Archive block kind vocabulary."""

    TEXT = "text"
    THINKING = "thinking"
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    CODE = "code"
    DOCUMENT = "document"


class SemanticBlockType(PolylogueStrEnum):
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


class BranchType(PolylogueStrEnum):
    """Classification for how a session relates to its parent."""

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    FORK = "fork"
    SUBAGENT = "subagent"


class LinkType(PolylogueStrEnum):
    """Archive cross-session edge vocabulary."""

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    SUBAGENT = "subagent"
    BRANCH = "branch"
    FORK = "fork"
    RESUME = "resume"
    REPAIRED = "repaired"


class TopologyEdgeStatus(PolylogueStrEnum):
    """Closed lifecycle vocabulary for topology/session-link edges."""

    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    REPAIRED = "repaired"
    QUARANTINED = "quarantined"


class PasteBoundary(PolylogueStrEnum):
    """Boundary quality for detected paste spans."""

    EXACT = "exact"
    PROJECTED = "projected"
    WHOLE_MESSAGE_FALLBACK = "whole_message_fallback"
    HASH_ONLY = "hash_only"


class ValidationStatus(PolylogueStrEnum):
    """Persisted raw-schema validation outcome."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @classmethod
    def from_string(cls, value: str | ValidationStatus) -> ValidationStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class ValidationMode(PolylogueStrEnum):
    """Configured raw-schema validation strictness."""

    OFF = "off"
    ADVISORY = "advisory"
    STRICT = "strict"

    @classmethod
    def from_string(cls, value: str | ValidationMode) -> ValidationMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class ArtifactSupportStatus(PolylogueStrEnum):
    """Durable support state for an observed raw artifact."""

    SUPPORTED_PARSEABLE = "supported_parseable"
    RECOGNIZED_UNPARSED = "recognized_unparsed"
    UNSUPPORTED_PARSEABLE = "unsupported_parseable"
    DECODE_FAILED = "decode_failed"
    PARTIAL_DECODE = "partial_decode"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | ArtifactSupportStatus) -> ArtifactSupportStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class PlanStage(PolylogueStrEnum):
    """Supported ingest/runtime planning stages."""

    ALL = "all"
    ACQUIRE = "acquire"
    CUSTOM = "custom"
    PARSE = "parse"
    MATERIALIZE = "materialize"
    RENDER = "render"
    SITE = "site"
    INDEX = "index"
    SCHEMA = "schema"
    REPROCESS = "reprocess"
    PUBLISH = "publish"

    @classmethod
    def from_string(cls, value: str | PlanStage) -> PlanStage:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


class ExerciseIOMode(PolylogueStrEnum):
    """Showcase exercise input/output mutability mode."""

    READ = "read"
    WRITE = "write"
    IDEMPOTENT = "idempotent"

    @classmethod
    def from_string(cls, value: str | ExerciseIOMode) -> ExerciseIOMode:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


__all__ = [
    "ArtifactSupportStatus",
    "BlockType",
    "BranchType",
    "ContentBlockType",
    "ExerciseIOMode",
    "LinkType",
    "MessageType",
    "Origin",
    "PasteBoundary",
    "PlanStage",
    "PolylogueStrEnum",
    "Provider",
    "Role",
    "SemanticBlockType",
    "TopologyEdgeStatus",
    "ValidationMode",
    "ValidationStatus",
    "enum_values",
    "nullable_sql_check_in",
    "sql_check_in",
    "sql_string_literal",
    "sql_value_list",
]
