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

    def __str__(self) -> str:
        return self.value

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


def normalize_role(raw: str) -> str:
    """Normalize a provider role string to a canonical role string.

    Thin shim — delegates to Role.normalize() and returns the string value.
    Prefer Role.normalize() directly in new code.

    Args:
        raw: Provider-specific role string (e.g., "human", "model", "ai").
             Must be non-empty. Missing roles should be handled at parse time.

    Returns:
        Canonical role string: "user", "assistant", "system", "tool", or
        "unknown" if not recognized.

    Raises:
        ValueError: If raw is empty or whitespace-only.
    """
    return Role.normalize(raw).value


__all__ = ["Role", "normalize_role"]
