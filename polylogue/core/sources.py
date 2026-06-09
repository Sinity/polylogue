"""Source-centered identity for session origins.

This module carries the ``Source`` family alongside the storage/provider
schema vocabulary and the public ``Origin`` tokens.  The vocabulary split is
documented in ``docs/architecture.md`` ("Provider and Origin Vocabulary").

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
needs to cross a provider-wire boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from polylogue.core.enums import Origin
from polylogue.core.provider_identity import CORE_SCHEMA_PROVIDERS
from polylogue.types import Provider

__all__ = [
    "ALL_SOURCES",
    "CORE_SCHEMA_ORIGINS",
    "Lab",
    "Source",
    "SourceFamily",
    "lab_of_origin",
    "origin_from_provider",
    "provider_from_origin",
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
    """A typed description of a session source.

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


# ---------------------------------------------------------------------------
# Provider -> Origin bridge (#1743 axis 2: provider purge → origin)
#
# The archive ``Origin`` StrEnum (core/enums.py) is the canonical vocabulary
# for "which product/surface produced this session" (``[product]-[kind]`` family
# tokens). ``Provider`` is the provider-wire token retained for parser/schema
# boundaries.
# This is a SEMANTIC bridge, not a 1:1 rename: ``Provider`` conflates lab and
# product identity, and three ``Source.family`` tokens predate the authoritative
# ``Origin`` set (``gemini-export``/``drive-takeout``/``unknown`` vs Origin's
# ``aistudio-drive``/``unknown-export``). Per #1743, ``Provider.GEMINI`` and
# ``Provider.DRIVE`` both collapse to ``Origin.AISTUDIO_DRIVE`` (both are Google
# AI-Studio/Drive exports routed through ``parsers/drive.py``).
_PROVIDER_TO_ORIGIN: Final[dict[Provider, Origin]] = {
    Provider.CHATGPT: Origin.CHATGPT_EXPORT,
    Provider.CLAUDE_AI: Origin.CLAUDE_AI_EXPORT,
    Provider.CLAUDE_CODE: Origin.CLAUDE_CODE_SESSION,
    Provider.CODEX: Origin.CODEX_SESSION,
    Provider.GEMINI: Origin.AISTUDIO_DRIVE,
    Provider.GEMINI_CLI: Origin.GEMINI_CLI_SESSION,
    Provider.HERMES: Origin.HERMES_SESSION,
    Provider.ANTIGRAVITY: Origin.ANTIGRAVITY_SESSION,
    Provider.DRIVE: Origin.AISTUDIO_DRIVE,
    Provider.UNKNOWN: Origin.UNKNOWN_EXPORT,
}


# Origin -> lab. Lab is *derived* from origin, never stored (#1743 terminology).
# Values mirror the ``originating_lab`` of each Provider's canonical Source so
# the two vocabularies cannot drift (asserted in test_sources).
_ORIGIN_TO_LAB: Final[dict[Origin, Lab]] = {
    Origin.CLAUDE_CODE_SESSION: "anthropic",
    Origin.CLAUDE_AI_EXPORT: "anthropic",
    Origin.CHATGPT_EXPORT: "openai",
    Origin.CODEX_SESSION: "openai",
    Origin.GEMINI_CLI_SESSION: "google",
    Origin.AISTUDIO_DRIVE: "google",
    Origin.ANTIGRAVITY_SESSION: "google",
    Origin.HERMES_SESSION: "nous",
    Origin.UNKNOWN_EXPORT: "unknown",
}


def origin_from_provider(provider: Provider) -> Origin:
    """Return the archive ``Origin`` for a provider-wire ``Provider`` enum value.

    Total over ``Provider``. ``Provider.GEMINI`` and ``Provider.DRIVE`` both map
    to ``Origin.AISTUDIO_DRIVE`` (see module note). Use this at boundaries that
    still carry a ``Provider`` while their storage/wire side speaks
    ``origin``; new code should take ``Origin`` directly.
    """

    return _PROVIDER_TO_ORIGIN[provider]


def lab_of_origin(origin: Origin) -> Lab:
    """Return the AI lab that produced sessions of a given ``Origin``.

    Lab is derived, not stored (#1743). Total over ``Origin``.
    """

    return _ORIGIN_TO_LAB[origin]


# Origin -> canonical Provider. The reverse of ``origin_from_provider`` is not
# injective: ``Origin.AISTUDIO_DRIVE`` is produced by both ``Provider.GEMINI``
# and ``Provider.DRIVE``; ``Provider.GEMINI`` is chosen as canonical so that a
# Gemini session round-trips provider -> origin -> provider. Use this only at
# boundaries that must project an origin back onto provider-wire vocabulary.
_ORIGIN_TO_PROVIDER: Final[dict[Origin, Provider]] = {
    Origin.CLAUDE_CODE_SESSION: Provider.CLAUDE_CODE,
    Origin.CODEX_SESSION: Provider.CODEX,
    Origin.GEMINI_CLI_SESSION: Provider.GEMINI_CLI,
    Origin.HERMES_SESSION: Provider.HERMES,
    Origin.ANTIGRAVITY_SESSION: Provider.ANTIGRAVITY,
    Origin.CHATGPT_EXPORT: Provider.CHATGPT,
    Origin.CLAUDE_AI_EXPORT: Provider.CLAUDE_AI,
    Origin.AISTUDIO_DRIVE: Provider.GEMINI,
    Origin.UNKNOWN_EXPORT: Provider.UNKNOWN,
}


def provider_from_origin(origin: Origin) -> Provider:
    """Return the canonical provider-wire ``Provider`` for an archive ``Origin``.

    Total over ``Origin``. Inverse of :func:`origin_from_provider` with a
    canonical choice for the non-injective ``Origin.AISTUDIO_DRIVE``
    (``Provider.GEMINI``). New code should consume ``Origin`` directly; this
    exists only for boundaries that still speak provider tokens.
    """

    return _ORIGIN_TO_PROVIDER[origin]


# User-facing origin tokens for CLI/MCP/completion choices. Derived from the
# core schema providers so the surfaced set tracks the same products, excluding
# the internal ``unknown-export`` origin. Order-preserving and deduplicated
# (``gemini`` and ``drive`` both collapse onto ``aistudio-drive``).
CORE_SCHEMA_ORIGINS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(origin_from_provider(Provider.from_string(token)).value for token in CORE_SCHEMA_PROVIDERS)
)
