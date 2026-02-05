"""Environment variable utilities with POLYLOGUE_* precedence.

This module provides helpers for reading environment variables with
Polylogue-specific prefixed versions taking precedence over global ones.
"""

from __future__ import annotations

import os


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with POLYLOGUE_* precedence.

    Checks POLYLOGUE_{KEY} first, then {KEY}, then returns default.

    This allows project-specific configuration without affecting global tools.
    For example, POLYLOGUE_QDRANT_URL will be used instead of QDRANT_URL if set.

    Args:
        key: Environment variable name (without POLYLOGUE_ prefix)
        default: Default value if neither variable is set

    Returns:
        Value from POLYLOGUE_{KEY}, {KEY}, or default (in that order)

    Examples:
        >>> os.environ["QDRANT_URL"] = "http://global:6333"
        >>> get_env("QDRANT_URL")
        'http://global:6333'

        >>> os.environ["POLYLOGUE_QDRANT_URL"] = "http://local:6333"
        >>> get_env("QDRANT_URL")
        'http://local:6333'

        >>> get_env("MISSING_VAR", "fallback")
        'fallback'
    """
    prefixed_key = f"POLYLOGUE_{key}"
    return os.environ.get(prefixed_key) or os.environ.get(key) or default


def get_env_multi(key: str, *alternatives: str, default: str | None = None) -> str | None:
    """Get environment variable with multiple fallback names.

    Checks POLYLOGUE_{KEY} first, then {KEY}, then alternative names.

    Useful for variables that have multiple common names (e.g., GOOGLE_API_KEY vs GEMINI_API_KEY).

    Args:
        key: Primary environment variable name
        *alternatives: Alternative unprefixed names to check
        default: Default value if no variables are set

    Returns:
        First non-None value found, or default

    Examples:
        >>> os.environ["GEMINI_API_KEY"] = "key123"
        >>> get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        'key123'

        >>> os.environ["POLYLOGUE_GOOGLE_API_KEY"] = "key456"
        >>> get_env_multi("GOOGLE_API_KEY", "GEMINI_API_KEY")
        'key456'
    """
    # Check prefixed version first
    prefixed_key = f"POLYLOGUE_{key}"
    value = os.environ.get(prefixed_key)
    if value:
        return value

    # Check unprefixed primary
    value = os.environ.get(key)
    if value:
        return value

    # Check alternatives
    for alt in alternatives:
        value = os.environ.get(alt)
        if value:
            return value

    return default


__all__ = ["get_env", "get_env_multi"]
