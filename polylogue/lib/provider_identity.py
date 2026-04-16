"""Provider identity normalization shared across runtime and schema flows."""

from __future__ import annotations

from typing import Final

CORE_RUNTIME_PROVIDERS: Final[tuple[str, ...]] = (
    "chatgpt",
    "claude-ai",
    "claude-code",
    "codex",
    "gemini",
    "drive",
    "unknown",
)

CORE_SCHEMA_PROVIDERS: Final[tuple[str, ...]] = (
    "chatgpt",
    "claude-ai",
    "claude-code",
    "codex",
    "gemini",
)

_RUNTIME_PROVIDER_ALIASES: Final[dict[str, str]] = {}


def normalize_provider_token(value: str | None) -> str:
    """Normalize provider token casing/separators."""
    if not value:
        return ""
    return value.strip().lower().replace("_", "-").replace(" ", "-")


def canonical_runtime_provider(
    value: str | None,
    *,
    default: str = "unknown",
) -> str:
    """Normalize provider names used by runtime parser/storage flows."""
    normalized = normalize_provider_token(value)
    if not normalized:
        return default

    canonical = _RUNTIME_PROVIDER_ALIASES.get(normalized, normalized)
    if canonical in CORE_RUNTIME_PROVIDERS:
        return canonical
    return default


def canonical_schema_provider(
    value: str | None,
    *,
    default: str = "unknown",
) -> str:
    """Normalize provider names to canonical schema identifiers."""
    canonical = canonical_runtime_provider(value, default=default)
    if canonical in CORE_SCHEMA_PROVIDERS:
        return canonical
    return default


__all__ = [
    "CORE_RUNTIME_PROVIDERS",
    "CORE_SCHEMA_PROVIDERS",
    "canonical_acquisition_provider",
    "canonical_runtime_provider",
    "canonical_schema_provider",
    "normalize_provider_token",
]


def canonical_acquisition_provider(
    provider_hint: str | None,
    *,
    source_name: str | None = None,
) -> str:
    """Resolve the raw acquisition provider hint stored with raw payloads.

    Acquisition-time provider identity must stay distinct from the configured
    source name. Source names like ``inbox`` or ``seeded`` are operator scope,
    not provider truth. Only store a provider token when either the scanned
    payload hint or the source family itself resolves to a known runtime
    provider; otherwise store ``unknown``.
    """

    provider = canonical_runtime_provider(provider_hint, default="")
    if provider:
        return provider
    return canonical_runtime_provider(source_name)
