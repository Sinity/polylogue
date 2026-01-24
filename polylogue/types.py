"""Type aliases and enums for polylogue."""
from __future__ import annotations

from enum import Enum
from typing import NewType

# Semantic ID types - provides compile-time distinction
ConversationId = NewType("ConversationId", str)
MessageId = NewType("MessageId", str)
AttachmentId = NewType("AttachmentId", str)
ContentHash = NewType("ContentHash", str)


class Provider(str, Enum):
    """Known conversation providers."""
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    GEMINI = "gemini"
    DRIVE = "drive"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str | None) -> Provider:
        """Normalize provider string to enum, defaulting to UNKNOWN."""
        if not value:
            return cls.UNKNOWN
        normalized = value.lower().strip()
        # Handle aliases
        if normalized in ("gpt", "openai"):
            return cls.CHATGPT
        if normalized in ("claude-ai", "anthropic"):
            return cls.CLAUDE
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN

    def __str__(self) -> str:
        return self.value
