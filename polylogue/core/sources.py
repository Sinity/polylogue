"""Source-centered identity for conversation origins.

This module introduces the ``Source`` family alongside the legacy
``Provider`` enum (``polylogue/types.py``).  It does not rename any
existing field, column, CLI flag, or MCP parameter — the two
vocabularies coexist for a transition period documented in
``docs/architecture.md`` ("Dual Vocabulary Period: Provider and Source").

The shape encodes three distinct concepts that the historical
``provider`` token conflated:

* ``family`` — the source-family token (e.g. ``claude-code-session``,
  ``codex-session``).  This is the unit users select on the CLI/MCP
  filter surface and the daemon's source discovery layer.
* ``runtime_root`` — the canonical on-disk root from which the source
  family is typically acquired (e.g. ``~/.claude/projects``).  Stored
  as a string hint; ``None`` for source families that have no fixed
  filesystem origin (e.g. ``unknown``).
* ``originating_lab`` — the AI lab whose product produced the
  transcripts (``anthropic``, ``openai``, ``google``, ``nous``,
  ``unknown``).  Distinct from the runtime so that, for example,
  Codex (OpenAI tooling) and ChatGPT exports both attribute to OpenAI
  while remaining separate source families.

A canonical ``Source`` exists for every ``Provider`` enum member and
the mapping is exhaustive (verified by tests in
``tests/unit/core/test_sources.py``).  Helpers ``provider_to_source``
and ``source_to_provider`` bridge the two vocabularies for code that
needs to cross the boundary during the migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polylogue.types import Provider

__all__ = [
    "ALL_SOURCES",
    "Lab",
    "Source",
    "SourceFamily",
    "provider_to_source",
    "source_for_family",
    "source_to_provider",
]


# Source-family tokens.  These are intentionally distinct from the
# Provider enum values: the family name describes the *runtime origin*
# of the transcript rather than the storage-layer provider key.
SourceFamily = str
Lab = str


@dataclass(frozen=True, slots=True)
class Source:
    """A typed description of a conversation source.

    ``Source`` is immutable and hashable so it can be used as a dict
    key, set element, or cache lookup token.  It is intentionally a
    plain dataclass rather than an ``Enum`` so that future source
    families (e.g. Antigravity variants, new lab products) can be
    added without breaking enum-membership checks.
    """

    family: SourceFamily
    """Canonical source-family token.  Stable identifier."""

    runtime_root: str | None
    """Default on-disk root for this source family, or ``None`` if the
    source family has no canonical filesystem origin."""

    originating_lab: Lab
    """The lab whose product produced the transcripts."""


# Canonical Source instances.  One per Provider enum value.  The
# mapping is exhaustive — see test_sources::test_every_provider_has_source.
_SOURCE_CHATGPT: Final[Source] = Source(
    family="chatgpt-export",
    runtime_root=None,  # delivered via OpenAI data export ZIPs
    originating_lab="openai",
)
_SOURCE_CLAUDE_AI: Final[Source] = Source(
    family="claude-ai-export",
    runtime_root=None,  # delivered via Anthropic data export ZIPs
    originating_lab="anthropic",
)
_SOURCE_CLAUDE_CODE: Final[Source] = Source(
    family="claude-code-session",
    runtime_root="~/.claude/projects",
    originating_lab="anthropic",
)
_SOURCE_CODEX: Final[Source] = Source(
    family="codex-session",
    runtime_root="~/.codex/sessions",
    originating_lab="openai",
)
_SOURCE_GEMINI: Final[Source] = Source(
    family="gemini-export",
    runtime_root=None,  # Google AI Studio exports via Drive / Takeout
    originating_lab="google",
)
_SOURCE_GEMINI_CLI: Final[Source] = Source(
    family="gemini-cli-session",
    runtime_root="~/.gemini/tmp",
    originating_lab="google",
)
_SOURCE_HERMES: Final[Source] = Source(
    family="hermes-session",
    runtime_root="~/.hermes",
    originating_lab="nous",
)
_SOURCE_ANTIGRAVITY: Final[Source] = Source(
    family="antigravity-session",
    runtime_root="~/.antigravity",
    originating_lab="google",
)
_SOURCE_DRIVE: Final[Source] = Source(
    family="drive-takeout",
    runtime_root=None,  # acquired via Google Drive / Takeout APIs
    originating_lab="google",
)
_SOURCE_UNKNOWN: Final[Source] = Source(
    family="unknown",
    runtime_root=None,
    originating_lab="unknown",
)


# Provider -> Source mapping.  Exhaustive over Provider enum.
_PROVIDER_TO_SOURCE: Final[dict[Provider, Source]] = {
    Provider.CHATGPT: _SOURCE_CHATGPT,
    Provider.CLAUDE_AI: _SOURCE_CLAUDE_AI,
    Provider.CLAUDE_CODE: _SOURCE_CLAUDE_CODE,
    Provider.CODEX: _SOURCE_CODEX,
    Provider.GEMINI: _SOURCE_GEMINI,
    Provider.GEMINI_CLI: _SOURCE_GEMINI_CLI,
    Provider.HERMES: _SOURCE_HERMES,
    Provider.ANTIGRAVITY: _SOURCE_ANTIGRAVITY,
    Provider.DRIVE: _SOURCE_DRIVE,
    Provider.UNKNOWN: _SOURCE_UNKNOWN,
}


# Reverse lookup table keyed by family token.  Built once at import
# time so ``source_to_provider`` is O(1).
_FAMILY_TO_PROVIDER: Final[dict[SourceFamily, Provider]] = {
    source.family: provider for provider, source in _PROVIDER_TO_SOURCE.items()
}


ALL_SOURCES: Final[tuple[Source, ...]] = tuple(_PROVIDER_TO_SOURCE.values())
"""Every canonical Source defined by this module, in Provider enum order."""


def provider_to_source(provider: Provider) -> Source:
    """Return the canonical ``Source`` for a ``Provider`` enum value.

    The mapping is total over ``Provider`` — every enum member has a
    canonical Source, including ``Provider.UNKNOWN``.
    """

    return _PROVIDER_TO_SOURCE[provider]


def source_to_provider(source: Source) -> Provider | None:
    """Return the ``Provider`` enum value for a ``Source``, if any.

    Returns ``None`` for ``Source`` instances whose ``family`` is not
    one of the canonical families defined here.  This is the failure
    mode for future source families that are introduced before being
    given a backing ``Provider`` enum value (the parallel-vocabulary
    period is asymmetric: Source can grow without immediately growing
    Provider).
    """

    return _FAMILY_TO_PROVIDER.get(source.family)


def source_for_family(family: SourceFamily) -> Source | None:
    """Look up a canonical ``Source`` by its family token.

    Returns ``None`` when the family is not one of the canonical
    families.  Use this in surfaces that accept a string token (CLI
    flags, MCP parameters) and need to recover the typed ``Source``.
    """

    provider = _FAMILY_TO_PROVIDER.get(family)
    if provider is None:
        return None
    return _PROVIDER_TO_SOURCE[provider]
