"""Tests for the Source dataclass and Provider<->Source mapping."""

from __future__ import annotations

import pytest

from polylogue.core.sources import (
    ALL_SOURCES,
    Source,
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
