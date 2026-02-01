"""Common semantic types that all providers map to.

This module defines the canonical types that represent the "common subset"
across all providers. Each provider implements extraction to these types.

Design:
    Instead of declarative YAML mappings, we use Python protocols + dataclasses.
    This gives us:
    1. Type checking at development time
    2. IDE autocompletion
    3. Runtime validation via Pydantic
    4. Clear "interface" that providers must implement

Usage:
    # Define what we want to extract
    class CommonMessage(Protocol):
        role: str
        text: str
        timestamp: datetime | None

    # Each provider implements extraction
    def extract_from_chatgpt(raw: dict) -> CommonMessage: ...
    def extract_from_claude(raw: dict) -> CommonMessage: ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    """Canonical roles across all providers."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

    @classmethod
    def normalize(cls, raw: str) -> "Role":
        """Normalize any provider's role string."""
        mapping = {
            "user": cls.USER,
            "human": cls.USER,
            "assistant": cls.ASSISTANT,
            "model": cls.ASSISTANT,
            "ai": cls.ASSISTANT,
            "system": cls.SYSTEM,
            "tool": cls.TOOL,
            "function": cls.TOOL,
        }
        return mapping.get(raw.lower(), cls.USER)


@dataclass
class CommonMessage:
    """The common subset all providers can provide."""

    role: Role
    text: str
    timestamp: datetime | None = None

    # Optional enrichments (not all providers have these)
    id: str | None = None
    model: str | None = None
    tokens: int | None = None
    cost_usd: float | None = None
    is_thinking: bool = False

    # Preserve original for debugging
    provider: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class CommonToolCall:
    """Tool invocation common subset."""

    name: str
    input: dict
    output: str | None = None
    success: bool | None = None

    provider: str = ""
    raw: dict = field(default_factory=dict)


# =============================================================================
# Provider Extractors - Each provider implements these
# =============================================================================


def extract_chatgpt_message(raw: dict) -> CommonMessage:
    """Extract CommonMessage from ChatGPT message."""
    author = raw.get("author", {})
    content = raw.get("content", {})

    # Get text from parts
    text = ""
    parts = content.get("parts", [])
    if parts:
        text = "\n".join(str(p) for p in parts if isinstance(p, str))

    # Timestamp
    ts = raw.get("create_time")
    timestamp = datetime.fromtimestamp(ts) if ts else None

    return CommonMessage(
        role=Role.normalize(author.get("role", "user")),
        text=text,
        timestamp=timestamp,
        id=raw.get("id"),
        model=raw.get("metadata", {}).get("model_slug"),
        provider="chatgpt",
        raw=raw,
    )


def extract_claude_ai_message(raw: dict) -> CommonMessage:
    """Extract CommonMessage from Claude AI message."""
    # Timestamp
    ts_str = raw.get("created_at")
    timestamp = None
    if ts_str:
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    return CommonMessage(
        role=Role.normalize(raw.get("sender", "user")),
        text=raw.get("text", ""),
        timestamp=timestamp,
        id=raw.get("uuid"),
        provider="claude-ai",
        raw=raw,
    )


def extract_claude_code_message(raw: dict) -> CommonMessage:
    """Extract CommonMessage from Claude Code record."""
    msg = raw.get("message", {})

    # Get text from content blocks
    text_parts = []
    is_thinking = False
    for block in msg.get("content", []):
        if isinstance(block, dict):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                text_parts.append(block.get("thinking", ""))
                is_thinking = True

    # Timestamp
    ts_str = raw.get("timestamp")
    timestamp = None
    if ts_str:
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    # Usage
    usage = msg.get("usage", {})
    tokens = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)

    return CommonMessage(
        role=Role.normalize(msg.get("role", raw.get("type", "user"))),
        text="\n".join(text_parts),
        timestamp=timestamp,
        id=raw.get("uuid"),
        model=msg.get("model"),
        tokens=tokens if tokens else None,
        cost_usd=raw.get("costUSD"),
        is_thinking=is_thinking,
        provider="claude-code",
        raw=raw,
    )


def extract_gemini_message(raw: dict) -> CommonMessage:
    """Extract CommonMessage from Gemini message."""
    return CommonMessage(
        role=Role.normalize(raw.get("role", "user")),
        text=raw.get("text", ""),
        tokens=raw.get("tokenCount"),
        is_thinking=raw.get("isThought", False),
        provider="gemini",
        raw=raw,
    )


def extract_codex_message(raw: dict) -> CommonMessage:
    """Extract CommonMessage from Codex record."""
    # Handle envelope vs direct format
    if "payload" in raw:
        payload = raw["payload"]
        role = payload.get("role", "user")
        content = payload.get("content", [])
    else:
        role = raw.get("role", "user")
        content = raw.get("content", [])

    # Get text
    text_parts = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict):
            text_parts.append(block.get("text", "") or block.get("input_text", "") or block.get("output_text", ""))

    # Timestamp
    ts_str = raw.get("timestamp")
    timestamp = None
    if ts_str:
        try:
            timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    return CommonMessage(
        role=Role.normalize(role),
        text="\n".join(text_parts),
        timestamp=timestamp,
        id=raw.get("id"),
        provider="codex",
        raw=raw,
    )


# =============================================================================
# Dispatcher
# =============================================================================


EXTRACTORS = {
    "chatgpt": extract_chatgpt_message,
    "claude": extract_claude_ai_message,
    "claude-ai": extract_claude_ai_message,
    "claude-code": extract_claude_code_message,
    "gemini": extract_gemini_message,
    "codex": extract_codex_message,
}


def extract_message(provider: str, raw: dict) -> CommonMessage:
    """Extract CommonMessage from any provider's raw data."""
    extractor = EXTRACTORS.get(provider)
    if not extractor:
        raise ValueError(f"Unknown provider: {provider}")
    return extractor(raw)
