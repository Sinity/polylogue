"""Origin projection behavior for public read surfaces (#1810)."""

from __future__ import annotations

from polylogue.core.sources import source_name_to_origin


def test_source_name_to_origin_projects_provider_tokens_to_origin() -> None:
    # Provider-wire tokens are projected onto the public origin vocabulary.
    assert source_name_to_origin("claude-code") == "claude-code-session"
    assert source_name_to_origin("codex") == "codex-session"
    # A value that is already a canonical origin passes through unchanged.
    assert source_name_to_origin("chatgpt-export") == "chatgpt-export"
    # Empty input degrades to the neutral "unknown" token.
    assert source_name_to_origin("") == "unknown"
    assert source_name_to_origin(None) == "unknown"


def test_source_name_to_origin_maps_source_family_tokens_not_just_providers() -> None:
    # Source-family tokens are NOT provider-wire values; they must map to their
    # real origin, not degrade to unknown-export (#1810 / Codex on #1901).
    gemini_origin = source_name_to_origin("gemini-export")
    drive_origin = source_name_to_origin("drive-takeout")
    assert gemini_origin != "unknown"
    assert drive_origin != "unknown"
    # The family token resolves to the same origin as its provider equivalent.
    assert gemini_origin == source_name_to_origin("gemini")
    assert drive_origin == source_name_to_origin("drive")


def test_archive_query_origin_resolution_is_origin_only() -> None:
    from polylogue.cli.archive_query import _resolve_excluded_origins, _resolve_origins

    assert _resolve_origins({"origin": "codex-session,claude-code-session"}) == (
        "codex-session",
        "claude-code-session",
    )
    assert _resolve_excluded_origins({"exclude_origin": "chatgpt-export"}) == ("chatgpt-export",)
