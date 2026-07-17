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

from polylogue.core.enums import Origin, Provider
from polylogue.core.provider_identity import CORE_SCHEMA_PROVIDERS

__all__ = [
    "ALL_SOURCES",
    "CORE_SCHEMA_ORIGINS",
    "Lab",
    "Source",
    "SourceFamily",
    "lab_of_origin",
    "origin_from_provider",
    "origin_provider_fiber",
    "provider_from_origin",
    "provider_to_source",
    "source_for_family",
    "source_name_to_origin",
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
_SOURCE_BEADS: Final[Source] = Source(
    family="beads-issue",
    runtime_root=None,  # repository-local .beads workspaces, not one user-global root
    originating_lab="beads",
)
_SOURCE_GROK: Final[Source] = Source(
    family="grok-export",
    runtime_root=None,
    originating_lab="xai",
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
    Provider.BEADS: _SOURCE_BEADS,
    Provider.GROK: _SOURCE_GROK,
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
    Provider.BEADS: Origin.BEADS_ISSUE,
    Provider.GROK: Origin.GROK_EXPORT,
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
    Origin.BEADS_ISSUE: "beads",
    Origin.HERMES_SESSION: "nous",
    Origin.GROK_EXPORT: "xai",
    Origin.UNKNOWN_EXPORT: "unknown",
}


def origin_from_provider(provider: Provider | Origin | str) -> Origin:
    """Return the archive ``Origin`` for provider-wire or origin input.

    Total over ``Provider``. ``Provider.GEMINI`` and ``Provider.DRIVE`` both map
    to ``Origin.AISTUDIO_DRIVE`` (see module note). Use this at boundaries that
    still carry a ``Provider`` while their storage/wire side speaks
    ``origin``; callers that already carry ``Origin`` or a canonical origin
    token pass through unchanged.
    """

    if isinstance(provider, Origin):
        return provider
    if isinstance(provider, str):
        if provider in _CANONICAL_ORIGIN_VALUES:
            return Origin.from_string(provider)
        provider = Provider.from_string(provider)
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
    Origin.BEADS_ISSUE: Provider.BEADS,
    Origin.GROK_EXPORT: Provider.GROK,
    Origin.CHATGPT_EXPORT: Provider.CHATGPT,
    Origin.CLAUDE_AI_EXPORT: Provider.CLAUDE_AI,
    Origin.AISTUDIO_DRIVE: Provider.GEMINI,
    Origin.UNKNOWN_EXPORT: Provider.UNKNOWN,
}


def _build_origin_provider_fiber() -> dict[Origin, tuple[Provider, ...]]:
    """Group every ``Provider`` by the ``Origin`` it collapses onto.

    Built from :data:`_PROVIDER_TO_ORIGIN` (the total, authoritative forward
    map) rather than re-declared by hand, so a newly added ``Provider`` can
    never silently omit itself from its origin's fiber the way the three
    independently hand-copied ``_provider_for_origin`` dicts drifted before
    polylogue-9e5.8 Step 1 deduplicated them.
    """

    fiber: dict[Origin, list[Provider]] = {}
    for provider in Provider:
        fiber.setdefault(_PROVIDER_TO_ORIGIN[provider], []).append(provider)
    return {origin: tuple(providers) for origin, providers in fiber.items()}


_ORIGIN_PROVIDER_FIBER: Final[dict[Origin, tuple[Provider, ...]]] = _build_origin_provider_fiber()


def origin_provider_fiber(origin: Origin) -> tuple[Provider, ...]:
    """Return every ``Provider`` that collapses onto ``origin``.

    Most origins have exactly one member and round-trip losslessly through
    :func:`provider_from_origin`. ``Origin.AISTUDIO_DRIVE`` is the sole
    multi-member fiber today: ``(Provider.GEMINI, Provider.DRIVE)`` --
    Gemini/AI-Studio export bundles and Google-Drive-live-acquired AI-Studio
    captures are the same public source-origin but different provider-wire
    acquisition mechanisms (see ``docs/provider-origin-identity.md``'s
    "Capture mode" row). Use this to test whether an origin is ambiguous
    before trusting :func:`provider_from_origin`'s canonical fallback, or to
    validate that a ``family_hint`` actually belongs to the fiber it claims
    to disambiguate.
    """

    return _ORIGIN_PROVIDER_FIBER.get(origin, ())


def _resolve_family_hint(hint: Provider | SourceFamily) -> Provider | None:
    """Resolve a disambiguating hint to a ``Provider``, or ``None`` if it
    does not name a recognized provider or source family."""

    if isinstance(hint, Provider):
        return hint
    family_provider = _FAMILY_TO_PROVIDER.get(hint)
    if family_provider is not None:
        return family_provider
    try:
        return Provider.from_string(hint)
    except ValueError:
        return None


def provider_from_origin(origin: Origin, *, family_hint: Provider | SourceFamily | None = None) -> Provider:
    """Return the canonical provider-wire ``Provider`` for an archive ``Origin``.

    Total over ``Origin``. Inverse of :func:`origin_from_provider` with a
    canonical choice for the non-injective ``Origin.AISTUDIO_DRIVE``
    (``Provider.GEMINI``). New code should consume ``Origin`` directly; this
    exists only for boundaries that still speak provider tokens.

    ``family_hint`` disambiguates origins whose :func:`origin_provider_fiber`
    has more than one member -- today only ``Origin.AISTUDIO_DRIVE``
    (``{Provider.GEMINI, Provider.DRIVE}``). Pass a ``Source.family`` token
    (e.g. ``"drive-takeout"``), a provider-wire string, or a ``Provider``
    value that the caller already knows independently of the stored
    ``Origin`` -- typically an explicit filter parameter or other
    acquisition-time context -- to recover the correct fiber member instead
    of the canonical default.

    A hint that fails to resolve, or resolves to a ``Provider`` outside
    ``origin``'s fiber, is ignored (falls back to the canonical choice)
    rather than raising: disambiguation here is advisory, not authoritative.
    The durable ``source.db`` ``raw_sessions.capture_mode`` field preserves
    acquisition-time provenance for new raw captures, so source-tier readers
    can pass it here to recover a persisted fiber member.  ``index.db``
    sessions still carry only the collapsed ``origin`` and historical source
    rows retain ``NULL`` for unknown provenance; those callers still need
    independent context or receive the canonical fallback.
    """

    if family_hint is not None:
        resolved = _resolve_family_hint(family_hint)
        if resolved is not None and resolved in _ORIGIN_PROVIDER_FIBER.get(origin, ()):
            return resolved
    return _ORIGIN_TO_PROVIDER[origin]


_CANONICAL_ORIGIN_VALUES: Final[frozenset[str]] = frozenset(origin.value for origin in Origin)


def source_name_to_origin(source_name: object) -> str:
    """Project a stored ``source_name`` token onto a public ``Origin`` value.

    Insight rows persist a ``source_name`` that may already be a canonical
    ``Origin`` token or a provider-wire / source-family token. This bridges
    such a stored value onto the public origin vocabulary for read payloads,
    so surface modules (daemon HTTP, MCP) project insight rows to origin
    without importing the provider-wire ``Provider`` enum themselves.

    Returns ``Origin.UNKNOWN_EXPORT`` for empty or unrecognized input. A token that is already a
    canonical origin passes through unchanged; otherwise it is interpreted
    as a provider-wire token and mapped via :func:`origin_from_provider`.
    """

    value = str(source_name or "")
    if not value:
        return Origin.UNKNOWN_EXPORT.value
    if value in _CANONICAL_ORIGIN_VALUES:
        return value
    # Source-family tokens (e.g. ``gemini-export``, ``drive-takeout``) are
    # canonical families, NOT provider-wire values — ``Provider.from_string``
    # would normalize them to ``unknown`` and mis-group those sessions under
    # ``unknown-export``. Map families first via the family table, then fall
    # back to provider-wire parsing (#1810).
    family_provider = _FAMILY_TO_PROVIDER.get(value)
    if family_provider is not None:
        return origin_from_provider(family_provider).value
    try:
        return origin_from_provider(Provider.from_string(value)).value
    except ValueError:
        return Origin.UNKNOWN_EXPORT.value


# User-facing origin tokens for CLI/MCP/completion choices. Derived from the
# core schema providers so the surfaced set tracks the same products, excluding
# the internal ``unknown-export`` origin. Order-preserving and deduplicated
# (``gemini`` and ``drive`` both collapse onto ``aistudio-drive``).
CORE_SCHEMA_ORIGINS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(origin_from_provider(Provider.from_string(token)).value for token in CORE_SCHEMA_PROVIDERS)
)
