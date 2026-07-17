"""Provider identity normalization shared across runtime and schema flows.

Vocabulary boundary
-------------------

Polylogue distinguishes four concepts that the legacy ``provider`` term
conflated.  This module defines the canonical tokens and alias rules for
each layer; other code should reference this table when choosing a name
for a public field, flag, or doc string.

===================== ====================== ===========================================
Concept                Examples                Where used
===================== ====================== ===========================================
**source family**      claude-code, codex,     Public filters, CLI --provider completions,
                       gemini-cli, aistudio,   daemon source discovery, watcher roots,
                       hermes, antigravity     MCP list/search params, docs + help text.

**provider / lab**     OpenAI, Google,         Provider-wire schemas, lab/model pricing
                       Anthropic              lookups, schema identity classification.

**model identity**     gpt-5, claude-opus-4-7, provider_meta payloads, cost rollups,
                       gemini-2.5-pro         stats by model.

**configured source    ~/.claude/projects/,    Config TOML ``[sources] roots``, daemon
   root**              ~/.codex/sessions/,     watcher file system discovery, CLI --root.
                       /inbox
===================== ====================== ===========================================

Surviving ``provider`` uses and their classification
-----------------------------------------------------

=================================== ===================================================
Location                             Classification
=================================== ===================================================
DB column ``source_name``             Canonical source-family column — renamed from
                                     ``provider_name`` per ADR 001 (#1022).
``Provider`` enum in types.py        Backend canonical form; maps source-family tokens
                                     to storage values via ``canonical_runtime_provider``.
CLI ``--provider`` / ``--exclude-provider``
                                     Public source-family filter (alias-aware).
MCP ``provider`` param               Same as CLI — source-family filter.
``SessionSummaryPayload.provider``
                                     Public surface field; carries source-family token.
Provider-wire schemas under          Lab/provider scope (OpenAI/Google/Anthropic) —
  ``schemas/providers/``             describes raw export shapes, not ingestion sources.
Daemon ``/api/facets`` providers     Source-family counts.
=================================== ===================================================

Key alias rules
---------------

- ``aistudio`` canonicalizes to ``gemini`` at the storage/enum layer
  (Google AI Studio exports are Gemini-provider sessions) but
  remains a distinct *source family* for watcher discovery and source
  name display.
- ``gemini-cli`` is a separate runtime/schema token from ``gemini``
  (different raw export shape, different watcher root).
- ``hermes`` and ``antigravity`` are distinct source-family tokens with
  their own watcher roots and schema providers.
- ``claude`` and ``anthropic`` are aliases for ``claude-ai`` (Claude web).
"""

from __future__ import annotations

from typing import Final

CORE_RUNTIME_PROVIDERS: Final[tuple[str, ...]] = (
    "chatgpt",
    "claude-ai",
    "claude-code",
    "codex",
    "gemini",
    "gemini-cli",
    "hermes",
    "antigravity",
    "beads",
    "grok",
    "drive",
    "unknown",
)

CORE_SCHEMA_PROVIDERS: Final[tuple[str, ...]] = (
    "chatgpt",
    "claude-ai",
    "claude-code",
    "codex",
    "gemini",
    "gemini-cli",
    "hermes",
    "antigravity",
    "grok",
)

_RUNTIME_PROVIDER_ALIASES: Final[dict[str, str]] = {
    "claude": "claude-ai",
    "anthropic": "claude-ai",
    "openai": "chatgpt",
    "google": "gemini",
    "google-gemini": "gemini",
    "aistudio": "gemini",
    "ai-studio": "gemini",
    "hermes-agent": "hermes",
    "xai": "grok",
    "x-ai": "grok",
    "twitter-grok": "grok",
    "cursor": "codex",
}


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
    if provider and provider != "unknown":
        return provider
    source_provider = canonical_runtime_provider(source_name, default="")
    if source_provider:
        return source_provider
    source_token = normalize_provider_token(source_name)
    source_prefix = source_token.split(":", 1)[0]
    return canonical_runtime_provider(source_prefix)
