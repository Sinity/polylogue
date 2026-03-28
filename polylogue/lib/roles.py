"""Unified role normalization for Polylogue.

Provides canonical mapping of provider-specific role names to standard roles.
"""

from __future__ import annotations

from enum import Enum


class Role(str, Enum):
    """Canonical conversation roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, raw: str) -> Role:
        """Normalize a provider role string to a canonical Role.

        Args:
            raw: Provider-specific role string (e.g., "human", "model", "ai").
                 Must be non-empty. Missing roles should be handled at parse time.

        Returns:
            Canonical Role enum value. Returns UNKNOWN for unrecognized roles.

        Raises:
            ValueError: If raw is empty or whitespace-only.
        """
        lowered = raw.strip().lower()
        if not lowered:
            raise ValueError("Role cannot be empty. Handle missing roles at parse time.")

        # User variants
        if lowered in {"user", "human"}:
            return cls.USER

        # Assistant variants
        if lowered in {"assistant", "model", "ai"}:
            return cls.ASSISTANT

        # System
        if lowered == "system":
            return cls.SYSTEM

        # Tool/function (includes claude-code progress/result types)
        if lowered in {"tool", "function", "tool_use", "tool_result", "progress", "result"}:
            return cls.TOOL

        return cls.UNKNOWN


# Role mapping for compatibility with existing code
ROLE_MAP = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "model": "assistant",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "tool_use": "tool",
    "tool_result": "tool",
    "progress": "tool",
    "result": "tool",
}


def normalize_role(raw: str) -> str:
    """Normalize a provider role string to a canonical role string.

    This is the primary function for role normalization throughout the codebase.
    Returns string values compatible with database storage and API interfaces.

    Args:
        raw: Provider-specific role string (e.g., "human", "model", "ai").
             Must be non-empty. Missing roles should be handled at parse time.

    Returns:
        Canonical role string: "user", "assistant", "system", "tool", or
        "unknown" if not recognized.

    Raises:
        ValueError: If raw is empty or whitespace-only.
    """
    lowered = raw.strip().lower()
    if not lowered:
        raise ValueError("Role cannot be empty. Handle missing roles at parse time.")

    return ROLE_MAP.get(lowered, "unknown")


__all__ = ["Role", "ROLE_MAP", "normalize_role"]
