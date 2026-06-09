"""Tests for the Source dataclass and Provider<->Source mapping."""

from __future__ import annotations

import pytest

from polylogue.core.enums import Origin
from polylogue.core.sources import (
    ALL_SOURCES,
    Source,
    lab_of_origin,
    origin_from_provider,
    provider_to_source,
    source_for_family,
    source_to_provider,
)
from polylogue.types import Provider


def test_every_provider_has_source() -> None:
    """Every Provider enum member maps to a canonical Source.

    The parallel-vocabulary period requires the Provider->Source
    mapping to be total; partial coverage would silently degrade to
    Source-less providers in any code that crosses the boundary.
    """

    for provider in Provider:
        source = provider_to_source(provider)
        assert isinstance(source, Source)
        assert source.family, f"Provider {provider!r} maps to a Source with empty family"


def test_every_source_round_trips_to_its_provider() -> None:
    """Every canonical Source resolves back to its originating Provider."""

    for provider in Provider:
        source = provider_to_source(provider)
        assert source_to_provider(source) is provider


def test_all_sources_matches_provider_enum_size() -> None:
    """ALL_SOURCES covers every Provider, no more and no fewer."""

    assert len(ALL_SOURCES) == len(list(Provider))
    families = {source.family for source in ALL_SOURCES}
    assert len(families) == len(ALL_SOURCES), "duplicate family token in ALL_SOURCES"


def test_source_to_provider_returns_none_for_unknown_family() -> None:
    """A Source with an unrecognized family does not resolve to a Provider."""

    stray = Source(family="future-source-family", runtime_root=None, originating_lab="unknown")
    assert source_to_provider(stray) is None


def test_source_for_family_round_trips() -> None:
    """source_for_family recovers a Source by its family token."""

    for provider in Provider:
        source = provider_to_source(provider)
        assert source_for_family(source.family) is source


def test_source_for_family_returns_none_for_unknown_token() -> None:
    assert source_for_family("not-a-real-family") is None


def test_source_is_immutable_and_hashable() -> None:
    """Source is frozen + slotted, suitable for dict keys and set members."""

    source = provider_to_source(Provider.CLAUDE_CODE)
    with pytest.raises((AttributeError, Exception)):
        source.family = "mutated"  # type: ignore[misc]
    # Hashable: usable as dict key and set member.
    mapping = {source: 1}
    membership = {source}
    assert mapping[source] == 1
    assert source in membership


def test_runtime_root_present_for_local_session_sources() -> None:
    """Source families whose primary acquisition is a local filesystem
    root should declare that root.  This pins the wiring between the
    dataclass and the daemon's source-discovery layer that future
    renames will operate against.
    """

    expectations = {
        Provider.CLAUDE_CODE: "~/.claude/projects",
        Provider.CODEX: "~/.codex/sessions",
        Provider.GEMINI_CLI: "~/.gemini/tmp",
        Provider.HERMES: "~/.hermes",
        Provider.ANTIGRAVITY: "~/.antigravity",
    }
    for provider, expected_root in expectations.items():
        assert provider_to_source(provider).runtime_root == expected_root


def test_originating_lab_attribution() -> None:
    """Lab attribution is distinct from the source family.

    Codex (OpenAI tooling) and ChatGPT exports both attribute to OpenAI;
    Gemini CLI and Drive exports both attribute to Google; Claude Code
    and Claude.ai both attribute to Anthropic.
    """

    expected_labs = {
        Provider.CHATGPT: "openai",
        Provider.CODEX: "openai",
        Provider.CLAUDE_AI: "anthropic",
        Provider.CLAUDE_CODE: "anthropic",
        Provider.GEMINI: "google",
        Provider.GEMINI_CLI: "google",
        Provider.ANTIGRAVITY: "google",
        Provider.DRIVE: "google",
        Provider.HERMES: "nous",
        Provider.UNKNOWN: "unknown",
    }
    for provider, lab in expected_labs.items():
        assert provider_to_source(provider).originating_lab == lab


# ── source family separability (#1022) ──────────────────────────────


def test_gemini_and_gemini_cli_are_distinct_families() -> None:
    """Gemini (AI Studio exports) and Gemini CLI are separate source families.

    They share the same originating lab (Google) but have different
    runtime roots and acquisition paths.
    """
    gemini = provider_to_source(Provider.GEMINI)
    gemini_cli = provider_to_source(Provider.GEMINI_CLI)

    assert gemini.family != gemini_cli.family
    assert gemini.originating_lab == gemini_cli.originating_lab == "google"
    assert gemini.runtime_root is None  # AI Studio exports come via Drive/Takeout
    assert gemini_cli.runtime_root == "~/.gemini/tmp"


def test_aistudio_alias_resolves_to_gemini_not_gemini_cli() -> None:
    """The 'aistudio' token canonicalizes to gemini (AI Studio), not gemini-cli."""
    from polylogue.core.provider_identity import canonical_runtime_provider

    assert canonical_runtime_provider("aistudio") == "gemini"
    assert canonical_runtime_provider("aistudio") != "gemini-cli"


def test_hermes_is_distinct_source_family() -> None:
    """Hermes is a separate source family with its own lab attribution."""
    hermes = provider_to_source(Provider.HERMES)

    assert hermes.family == "hermes-session"
    assert hermes.originating_lab == "nous"
    assert source_for_family("hermes-session") is hermes
    # Not colliding with any other family
    for provider in Provider:
        if provider is not Provider.HERMES:
            assert provider_to_source(provider).family != "hermes-session"


def test_antigravity_is_distinct_source_family() -> None:
    """Antigravity is a separate source family despite Google lab attribution."""
    antigravity = provider_to_source(Provider.ANTIGRAVITY)

    assert antigravity.family == "antigravity-session"
    assert antigravity.originating_lab == "google"
    assert source_for_family("antigravity-session") is antigravity
    # Not colliding with any other family
    for provider in Provider:
        if provider is not Provider.ANTIGRAVITY:
            assert provider_to_source(provider).family != "antigravity-session"


def test_all_source_families_are_unique() -> None:
    """No two Providers share the same source family token."""
    families = [provider_to_source(p).family for p in Provider]
    assert len(families) == len(set(families))


def test_source_family_separability_in_filters() -> None:
    """Each source family can be independently selected via source_for_family.

    This is the programmatic equivalent of CLI --provider / MCP provider
    filter separability: each family token resolves to exactly one Source,
    and no two families produce the same Source.
    """
    seen: set[str] = set()
    for provider in Provider:
        source = provider_to_source(provider)
        family = source.family
        # Each family must produce a unique Source
        assert family not in seen, f"Duplicate family token: {family}"
        seen.add(family)
        # source_for_family must round-trip
        assert source_for_family(family) is source
        # The reverse lookup must match
        assert source_to_provider(source) is provider


# ---------------------------------------------------------------------------
# Provider -> Origin bridge (#1743 axis 2)
# ---------------------------------------------------------------------------


def test_origin_from_provider_is_total() -> None:
    """Every Provider maps to a valid archive Origin."""
    for provider in Provider:
        origin = origin_from_provider(provider)
        assert isinstance(origin, Origin)


def test_gemini_and_drive_collapse_to_aistudio_drive() -> None:
    """Per #1743 both Gemini and Drive providers map to the single AISTUDIO_DRIVE origin."""
    assert origin_from_provider(Provider.GEMINI) is Origin.AISTUDIO_DRIVE
    assert origin_from_provider(Provider.DRIVE) is Origin.AISTUDIO_DRIVE


def test_gemini_cli_origin_is_distinct_from_aistudio() -> None:
    """The Gemini CLI session origin stays distinct from AI-Studio/Drive exports."""
    assert origin_from_provider(Provider.GEMINI_CLI) is Origin.GEMINI_CLI_SESSION
    assert origin_from_provider(Provider.GEMINI_CLI) is not origin_from_provider(Provider.GEMINI)


def test_lab_of_origin_is_total() -> None:
    """Every Origin has a derived lab."""
    for origin in Origin:
        assert lab_of_origin(origin), f"Origin {origin!r} has no lab"


def test_lab_derivation_agrees_with_source_attribution() -> None:
    """lab_of_origin(origin_from_provider(p)) must equal the provider's Source lab.

    This is the anti-drift contract: the derived origin→lab map cannot diverge
    from the canonical Source.originating_lab attribution.
    """
    for provider in Provider:
        derived = lab_of_origin(origin_from_provider(provider))
        assert derived == provider_to_source(provider).originating_lab
