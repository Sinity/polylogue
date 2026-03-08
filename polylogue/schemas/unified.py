"""Unified extraction layer combining glom specs with viewport types.

This module provides a single interface for extracting harmonized data from
any provider format, whether from raw exports or the polylogue database.

Architecture:
    Raw Provider Data
           │
           ▼ [glom spec]
    ExtractedMessage (flat dict)
           │
           ▼ [Pydantic]
    HarmonizedMessage (with viewports)

The HarmonizedMessage contains:
    - Core fields (role, text, timestamp, id)
    - Viewport extractions (tool_calls, reasoning_traces, content_blocks)
    - Metadata (tokens, cost, model)
    - Original raw data for debugging
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

try:
    from glom import glom
except ImportError:

    def glom(target: Any, spec: Any) -> Any: ...


from polylogue.lib.roles import Role
from polylogue.lib.timestamps import parse_timestamp
from polylogue.lib.viewports import (
    ContentBlock,
    ContentType,
    CostInfo,
    ReasoningTrace,
    TokenUsage,
    ToolCall,
    classify_tool,
)
from polylogue.types import Provider


def _missing_role() -> str:
    """Called when role is missing - raises error to surface data quality issues."""
    raise ValueError("Message has no role. Data should be validated at import time.")


# =============================================================================
# Unified Message Type
# =============================================================================


class HarmonizedMessage(BaseModel):
    """Unified message representation with viewport extractions.

    Combines core message data with rich semantic extractions that enable
    cross-provider analysis.
    """

    # Core fields
    id: str | None = None
    role: Role
    text: str
    timestamp: datetime | None = None

    # Viewport extractions
    reasoning_traces: list[ReasoningTrace] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(default_factory=list)

    # Metadata
    model: str | None = None
    tokens: TokenUsage | None = None
    cost: CostInfo | None = None
    duration_ms: int | None = None

    # Provider info
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
        """Check if message contains reasoning/thinking."""
        return len(self.reasoning_traces) > 0

    @property
    def has_tool_use(self) -> bool:
        """Check if message contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def file_operations(self) -> list[ToolCall]:
        """Get all file-related tool calls."""
        return [t for t in self.tool_calls if t.is_file_operation]

    @property
    def git_operations(self) -> list[ToolCall]:
        """Get all git-related tool calls."""
        return [t for t in self.tool_calls if t.is_git_operation]


# Transform functions now imported from core modules:
# - parse_timestamp from polylogue.lib.timestamps
# - normalize_role, ROLE_MAP from polylogue.lib.roles


# =============================================================================
# Viewport Extraction Functions
# =============================================================================


def extract_reasoning_traces(content: list[dict[str, Any]] | None, provider: Provider | str) -> list[ReasoningTrace]:
    """Extract reasoning traces from content blocks."""
    if not content:
        return []

    traces = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        text = None

        if block_type == "thinking":
            text = block.get("thinking") or block.get("text")
        elif block.get("isThought"):  # Gemini
            text = block.get("text")

        if text:
            traces.append(
                ReasoningTrace(
                    text=text,
                    provider=provider,
                    raw=block,
                )
            )

    return traces


def extract_tool_calls(content: list[dict[str, Any]] | None, provider: Provider | str) -> list[ToolCall]:
    """Extract tool calls from content blocks."""
    if not content:
        return []

    calls = []
    for block in content:
        if not isinstance(block, dict):
            continue

        if block.get("type") != "tool_use":
            continue

        name = block.get("name", "")
        input_data = block.get("input", {})

        calls.append(
            ToolCall(
                name=name,
                id=block.get("id"),
                input=input_data if isinstance(input_data, dict) else {},
                category=classify_tool(name, input_data if isinstance(input_data, dict) else {}),
                provider=provider,
                raw=block,
            )
        )

    return calls


def extract_content_blocks(content: list[dict[str, Any]] | None) -> list[ContentBlock]:
    """Extract content blocks with type classification."""
    if not content:
        return []

    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "text")

        if block_type == "text":
            blocks.append(
                ContentBlock(
                    type=ContentType.TEXT,
                    text=block.get("text"),
                    raw=block,
                )
            )
        elif block_type == "thinking":
            blocks.append(
                ContentBlock(
                    type=ContentType.THINKING,
                    text=block.get("thinking") or block.get("text"),
                    raw=block,
                )
            )
        elif block_type == "tool_use":
            name = block.get("name", "")
            input_data = block.get("input", {})
            blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_USE,
                    tool_call=ToolCall(
                        name=name,
                        id=block.get("id"),
                        input=input_data if isinstance(input_data, dict) else {},
                        category=classify_tool(name, input_data if isinstance(input_data, dict) else {}),
                    ),
                    raw=block,
                )
            )
        elif block_type == "tool_result":
            blocks.append(
                ContentBlock(
                    type=ContentType.TOOL_RESULT,
                    text=str(block.get("content", "")),
                    raw=block,
                )
            )
        elif block_type == "code":
            blocks.append(
                ContentBlock(
                    type=ContentType.CODE,
                    text=block.get("text") or block.get("code"),
                    language=block.get("language"),
                    raw=block,
                )
            )

    return blocks


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


# =============================================================================
# Text Extraction Helpers
# =============================================================================


def extract_claude_code_text(content: list[dict[str, Any]] | None) -> str:
    """Extract text from Claude Code content blocks.

    Only extracts ``type: "text"`` blocks. Thinking/reasoning traces are
    surfaced via reasoning_traces, not mixed into the main text content.
    """
    if not content:
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))

    return "\n".join(filter(None, parts))


def extract_chatgpt_text(content: dict[str, Any] | None) -> str:
    """Extract text from ChatGPT content structure."""
    if not content:
        return ""
    parts = content.get("parts", [])
    if not isinstance(parts, list):
        return str(parts) if parts else ""
    return "\n".join(str(p) for p in parts if isinstance(p, str))


def extract_codex_text(content: list[dict[str, Any]] | None) -> str:
    """Extract text from Codex content blocks."""
    if not content or not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # Codex has multiple text field names
        text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
        if text:
            parts.append(text)

    return "\n".join(parts)


# =============================================================================
# Provider Extraction
# =============================================================================


def extract_harmonized_message(provider: Provider | str, raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract HarmonizedMessage from raw provider data.

    Args:
        provider: Provider enum (or string for backward compat)
        raw: Raw message data in provider's native format

    Returns:
        HarmonizedMessage with core fields and viewport extractions
    """
    p = provider if isinstance(provider, Provider) else Provider.from_string(provider)
    if p == Provider.CLAUDE_CODE:
        return _extract_claude_code(raw)
    elif p == Provider.CLAUDE:
        return _extract_claude_ai(raw)
    elif p == Provider.CHATGPT:
        return _extract_chatgpt(raw)
    elif p == Provider.GEMINI:
        return _extract_gemini(raw)
    elif p == Provider.CODEX:
        return _extract_codex(raw)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _extract_claude_code(raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract from Claude Code format."""
    msg = raw.get("message", {})
    content = msg.get("content", []) if isinstance(msg, dict) else []

    return HarmonizedMessage(
        id=raw.get("uuid"),
        role=Role.normalize((msg.get("role") if isinstance(msg, dict) else raw.get("type")) or _missing_role()),
        text=extract_claude_code_text(content),
        timestamp=parse_timestamp(raw.get("timestamp")),
        reasoning_traces=extract_reasoning_traces(content, Provider.CLAUDE_CODE),
        tool_calls=extract_tool_calls(content, Provider.CLAUDE_CODE),
        content_blocks=extract_content_blocks(content),
        model=msg.get("model") if isinstance(msg, dict) else None,
        tokens=extract_token_usage(msg.get("usage") if isinstance(msg, dict) else None),
        cost=CostInfo(total_usd=raw.get("costUSD")) if raw.get("costUSD") else None,
        duration_ms=raw.get("durationMs"),
        provider=Provider.CLAUDE_CODE,
        raw=raw,
    )


def _extract_claude_ai(raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract from Claude AI (web) format."""
    return HarmonizedMessage(
        id=raw.get("uuid"),
        role=Role.normalize(raw.get("sender") or _missing_role()),
        text=raw.get("text", ""),
        timestamp=parse_timestamp(raw.get("created_at")),
        provider=Provider.CLAUDE,
        raw=raw,
    )


def _extract_chatgpt(raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract from ChatGPT format."""
    author = raw.get("author", {})
    content = raw.get("content", {})
    metadata = raw.get("metadata", {})

    return HarmonizedMessage(
        id=raw.get("id"),
        role=Role.normalize((author.get("role") if isinstance(author, dict) else None) or _missing_role()),
        text=extract_chatgpt_text(content) if isinstance(content, dict) else "",
        timestamp=parse_timestamp(raw.get("create_time")),
        model=metadata.get("model_slug") if isinstance(metadata, dict) else None,
        provider=Provider.CHATGPT,
        raw=raw,
    )


def _extract_gemini(raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract from Gemini format."""
    is_thinking = raw.get("isThought", False)

    return HarmonizedMessage(
        id=None,  # Gemini doesn't have message IDs in export
        role=Role.normalize(raw.get("role") or _missing_role()),
        text=raw.get("text", ""),
        timestamp=None,  # Gemini doesn't have timestamps in export
        reasoning_traces=[
            ReasoningTrace(
                text=raw.get("text", ""),
                token_count=raw.get("thinkingBudget"),
                provider=Provider.GEMINI,
                raw=raw,
            )
        ]
        if is_thinking
        else [],
        tokens=TokenUsage(output_tokens=raw.get("tokenCount")) if raw.get("tokenCount") else None,
        provider=Provider.GEMINI,
        raw=raw,
    )


def _extract_codex(raw: dict[str, Any]) -> HarmonizedMessage:
    """Extract from Codex format."""
    # Handle envelope vs direct format
    if "payload" in raw:
        payload = raw["payload"]
        role = payload.get("role", "user")
        content = payload.get("content", [])
    else:
        role = raw.get("role", "user")
        content = raw.get("content", [])

    # Extract text from content blocks
    text_parts = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict):
            text = block.get("text", "") or block.get("input_text", "") or block.get("output_text", "")
            if text:
                text_parts.append(text)

    return HarmonizedMessage(
        id=raw.get("id"),
        role=Role.normalize(role),
        text="\n".join(text_parts),
        timestamp=parse_timestamp(raw.get("timestamp")),
        provider=Provider.CODEX,
        raw=raw,
    )


# =============================================================================
# Database Integration
# =============================================================================


def extract_from_provider_meta(
    provider: Provider | str,
    provider_meta: dict[str, Any],
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage:
    """Extract HarmonizedMessage from polylogue database format.

    Providers store a ``raw`` key in provider_meta containing the original
    record; this passes through to full re-extraction via the provider
    dispatcher. Falls back to treating provider_meta itself as the raw record.

    Args:
        provider: Provider name
        provider_meta: The provider_meta JSON from messages table

    Returns:
        HarmonizedMessage with full viewport extractions
    """
    raw = provider_meta.get("raw")
    if raw is not None:
        return extract_harmonized_message(provider, raw)
    return extract_harmonized_message(provider, provider_meta)


def is_message_record(provider: str, raw: dict[str, Any]) -> bool:
    """Check if a record is an actual message (vs metadata).

    Some providers (like Claude Code) include metadata records
    mixed with messages.  When the ``raw`` original record is not
    available (claude-code stores extracted fields instead), we
    assume it's a message record since non-messages are filtered
    during parsing.
    """
    if provider in ("claude-code", "claude_code"):
        record_type = raw.get("type")
        if record_type is None:
            # No type field → extracted provider_meta, already filtered
            return True
        return record_type in ("user", "assistant", "system")
    return True  # Other providers only have message records


# =============================================================================
# Parser Integration
# =============================================================================


def harmonize_parsed_message(
    provider: str,
    provider_meta: dict[str, Any] | None,
    *,
    message_id: str | None = None,
    role: str | None = None,
    text: str | None = None,
    timestamp: datetime | str | float | int | None = None,
) -> HarmonizedMessage | None:
    """Convert ParsedMessage.provider_meta to HarmonizedMessage.

    This bridges the existing parser infrastructure with the unified
    extraction layer. Parsers produce ParsedMessage with provider_meta
    containing the raw data; this function extracts rich viewports.

    Args:
        provider: Provider name
        provider_meta: The provider_meta dict from ParsedMessage

    Returns:
        HarmonizedMessage with viewport extractions, or None if not extractable
    """
    if not provider_meta:
        return None

    raw = provider_meta.get("raw", provider_meta)

    if not is_message_record(provider, raw):
        return None

    return extract_from_provider_meta(
        provider,
        provider_meta,
        message_id=message_id,
        role=role,
        text=text,
        timestamp=timestamp,
    )


def bulk_harmonize(
    provider: str,
    parsed_messages: list[Any],
) -> list[HarmonizedMessage]:
    """Bulk convert ParsedMessages to HarmonizedMessages.

    Args:
        provider: Provider name
        parsed_messages: List of ParsedMessage objects

    Returns:
        List of HarmonizedMessage (skipping non-message records)
    """
    results = []
    for pm in parsed_messages:
        meta = getattr(pm, "provider_meta", None)
        if meta:
            harmonized = harmonize_parsed_message(
                provider,
                meta,
                message_id=getattr(pm, "provider_message_id", None),
                role=getattr(pm, "role", None),
                text=getattr(pm, "text", None),
                timestamp=getattr(pm, "timestamp", None),
            )
            if harmonized:
                results.append(harmonized)
    return results
