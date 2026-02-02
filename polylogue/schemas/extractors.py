"""Declarative provider-to-common extraction using glom.

This module uses glom for declarative data reshaping, composing with
Pydantic for validation. The pattern:

    Raw JSON → [glom spec] → flat dict → [Pydantic] → Typed Model

Each provider's spec is a glom "spec" - a Python dict that declares
what to extract and how to transform it. This replaces imperative
get-chains with a readable, maintainable mapping.

Design rationale:
    - glom handles path traversal, fallbacks, iteration
    - Custom functions handle provider-specific logic (e.g., block filtering)
    - Pydantic validates the final shape
    - Specs ARE documentation (inspectable, testable)

Example:
    >>> raw = {"uuid": "abc", "message": {"role": "assistant", ...}}
    >>> data = glom(raw, CLAUDE_CODE_SPEC)
    >>> msg = CommonMessage(**data)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

# glom import - will be available after adding to devshell
try:
    from glom import glom, Coalesce, T, SKIP, Iter, Check
    GLOM_AVAILABLE = True
except ImportError:
    GLOM_AVAILABLE = False
    # Stub for type hints
    def glom(target: Any, spec: Any) -> Any: ...  # noqa: E704

from polylogue.schemas.common import Role, CommonMessage, CommonToolCall


# =============================================================================
# Reusable Transform Functions
# =============================================================================


def parse_iso_timestamp(ts: str | None) -> datetime | None:
    """Parse ISO 8601 timestamp with Z suffix handling."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def parse_unix_timestamp(ts: float | int | None) -> datetime | None:
    """Parse Unix timestamp (seconds since epoch)."""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        return None


def normalize_role(raw: str | None) -> Role:
    """Normalize provider role string to canonical Role."""
    if not raw:
        return Role.USER
    return Role.normalize(raw)


def sum_usage_tokens(usage: dict | None) -> int | None:
    """Sum input + output tokens from usage dict."""
    if not usage:
        return None
    total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
    return total if total > 0 else None


def const(value: Any) -> Callable[[Any], Any]:
    """Create a function that returns a constant value."""
    return lambda _: value


# =============================================================================
# Provider-Specific Block Extractors
# =============================================================================


def extract_claude_code_text(content: list[dict] | None) -> str:
    """Extract text from Claude Code content blocks.

    Handles: text blocks, thinking blocks (concatenated).
    """
    if not content:
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "thinking":
            parts.append(block.get("thinking", ""))

    return "\n".join(filter(None, parts))


def extract_claude_code_is_thinking(content: list[dict] | None) -> bool:
    """Check if any block is a thinking block."""
    if not content:
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "thinking"
        for b in content
    )


def extract_chatgpt_text(content: dict | None) -> str:
    """Extract text from ChatGPT content structure."""
    if not content:
        return ""
    parts = content.get("parts", [])
    return "\n".join(str(p) for p in parts if isinstance(p, str))


def is_claude_code_message(raw: dict) -> bool:
    """Check if a Claude Code record is an actual message (vs metadata).

    Claude Code exports contain multiple record types:
    - user, assistant: Actual conversation messages
    - file-history-snapshot, progress, etc.: Metadata records

    Only user/assistant should be treated as messages.
    """
    record_type = raw.get("type")
    return record_type in ("user", "assistant")


def extract_codex_text(content: list[dict] | None) -> str:
    """Extract text from Codex content blocks."""
    if not content or not isinstance(content, list):
        return ""

    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # Codex has multiple text field names
        text = (
            block.get("text", "") or
            block.get("input_text", "") or
            block.get("output_text", "")
        )
        if text:
            parts.append(text)

    return "\n".join(parts)


# =============================================================================
# Provider Extraction Specs
# =============================================================================


if GLOM_AVAILABLE:

    # Claude Code spec
    # Note: Export contains multiple record types (user, assistant, file-history-snapshot, progress).
    # Only user/assistant have message.content. Others fallback to empty/defaults.
    CLAUDE_CODE_SPEC = {
        "role": (Coalesce("message.role", "type", default="user"), normalize_role),
        "text": (Coalesce("message.content", default=[]), extract_claude_code_text),
        "timestamp": (Coalesce("timestamp", default=None), parse_iso_timestamp),
        "id": Coalesce("uuid", default=None),
        "model": Coalesce("message.model", default=None),
        "tokens": (Coalesce("message.usage", default=None), sum_usage_tokens),
        "cost_usd": Coalesce("costUSD", default=None),
        "is_thinking": (Coalesce("message.content", default=[]), extract_claude_code_is_thinking),
        "provider": const("claude-code"),
    }

    # Claude AI (web) spec
    CLAUDE_AI_SPEC = {
        "role": ("sender", normalize_role),
        "text": Coalesce("text", default=""),
        "timestamp": ("created_at", parse_iso_timestamp),
        "id": Coalesce("uuid", default=None),
        "model": const(None),
        "tokens": const(None),
        "cost_usd": const(None),
        "is_thinking": const(False),
        "provider": const("claude-ai"),
    }

    # ChatGPT spec
    CHATGPT_SPEC = {
        "role": ("author.role", normalize_role),
        "text": ("content", extract_chatgpt_text),
        "timestamp": ("create_time", parse_unix_timestamp),
        "id": Coalesce("id", default=None),
        "model": Coalesce("metadata.model_slug", default=None),
        "tokens": const(None),
        "cost_usd": const(None),
        "is_thinking": const(False),
        "provider": const("chatgpt"),
    }

    # Gemini spec
    GEMINI_SPEC = {
        "role": ("role", normalize_role),
        "text": Coalesce("text", default=""),
        "timestamp": const(None),  # Gemini doesn't have timestamps in export
        "id": const(None),
        "model": const(None),
        "tokens": Coalesce("tokenCount", default=None),
        "cost_usd": const(None),
        "is_thinking": Coalesce("isThought", default=False),
        "provider": const("gemini"),
    }

    # Codex spec - handles envelope vs direct format
    def _get_codex_role(raw: dict) -> str:
        """Get role from Codex record (envelope or direct)."""
        if "payload" in raw:
            return raw["payload"].get("role", "user")
        return raw.get("role", "user")

    def _get_codex_content(raw: dict) -> list:
        """Get content from Codex record (envelope or direct)."""
        if "payload" in raw:
            return raw["payload"].get("content", [])
        return raw.get("content", [])

    CODEX_SPEC = {
        "role": (_get_codex_role, normalize_role),
        "text": (_get_codex_content, extract_codex_text),
        "timestamp": Coalesce(("timestamp", parse_iso_timestamp), default=None),
        "id": Coalesce("id", default=None),
        "model": const(None),
        "tokens": const(None),
        "cost_usd": const(None),
        "is_thinking": const(False),
        "provider": const("codex"),
    }

    # Registry
    PROVIDER_SPECS: dict[str, dict] = {
        "claude-code": CLAUDE_CODE_SPEC,
        "claude-ai": CLAUDE_AI_SPEC,
        "claude": CLAUDE_AI_SPEC,  # alias
        "chatgpt": CHATGPT_SPEC,
        "gemini": GEMINI_SPEC,
        "codex": CODEX_SPEC,
    }

else:
    # Fallback: empty registry when glom not available
    PROVIDER_SPECS = {}


# =============================================================================
# Public API
# =============================================================================


def extract_message(provider: str, raw: dict) -> CommonMessage:
    """Extract CommonMessage from raw provider data.

    This works on the ORIGINAL provider format (e.g., from exports).
    For data from polylogue's database, use extract_message_from_db().

    Args:
        provider: Provider name (e.g., "claude-code", "chatgpt")
        raw: Raw message data in original provider format

    Returns:
        CommonMessage with harmonized fields

    Raises:
        ValueError: If provider is unknown
        ImportError: If glom is not available
    """
    if not GLOM_AVAILABLE:
        raise ImportError("glom is required for extraction. Install with: pip install glom")

    spec = PROVIDER_SPECS.get(provider)
    if not spec:
        raise ValueError(f"Unknown provider: {provider}. Known: {list(PROVIDER_SPECS.keys())}")

    # Apply the spec to extract data
    data = glom(raw, spec)

    # Preserve raw for debugging
    data["raw"] = raw

    # Construct typed message
    return CommonMessage(**data)


def extract_message_from_db(provider: str, provider_meta: dict) -> CommonMessage:
    """Extract CommonMessage from polylogue database format.

    The database stores pre-processed data with original format in 'raw' key.

    Args:
        provider: Provider name (e.g., "claude-code", "chatgpt")
        provider_meta: The provider_meta JSON from messages table

    Returns:
        CommonMessage with harmonized fields
    """
    # Database format wraps original in 'raw' key
    raw = provider_meta.get("raw", provider_meta)
    return extract_message(provider, raw)


def extract_all(provider: str, messages: list[dict]) -> list[CommonMessage]:
    """Extract CommonMessage from a list of raw messages.

    Args:
        provider: Provider name
        messages: List of raw message dicts

    Returns:
        List of CommonMessage instances
    """
    return [extract_message(provider, msg) for msg in messages]


# =============================================================================
# Tool Call Extraction (separate from messages)
# =============================================================================


if GLOM_AVAILABLE:

    def extract_claude_code_tool_calls(raw: dict) -> list[dict]:
        """Extract tool calls from Claude Code message."""
        content = raw.get("message", {}).get("content", [])
        calls = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append({
                    "name": block.get("name", ""),
                    "id": block.get("id"),
                    "input": block.get("input", {}),
                    "output": None,  # Matched by subsequent tool_result
                    "success": None,
                    "provider": "claude-code",
                    "raw": block,
                })

        return calls


def extract_tool_calls(provider: str, raw: dict) -> list[CommonToolCall]:
    """Extract tool calls from a message.

    Args:
        provider: Provider name
        raw: Raw message data

    Returns:
        List of CommonToolCall instances (empty if no tool calls)
    """
    if not GLOM_AVAILABLE:
        raise ImportError("glom is required")

    if provider == "claude-code":
        calls_data = extract_claude_code_tool_calls(raw)
        return [CommonToolCall(**c) for c in calls_data]

    # Other providers: return empty (extend as needed)
    return []
