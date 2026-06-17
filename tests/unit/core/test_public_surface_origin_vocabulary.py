"""Guard: public CLI/MCP/daemon read surfaces speak origin, not provider (#1810).

The terminology sweep requires that public read surfaces expose the
``origin`` vocabulary only. Provider-wire identity is bridged behind
``core.sources.source_name_to_origin`` at the boundary, so these surface
modules must not bind the provider-wire ``Provider`` enum at all. Absence
of provider vocabulary on these surfaces is itself the user-visible
contract, which is why this guard asserts it directly.
"""

from __future__ import annotations

import importlib

import pytest

from polylogue.core.sources import source_name_to_origin

_PUBLIC_SURFACE_MODULES = (
    "polylogue.cli.archive_query",
    "polylogue.mcp.server_insight_tools",
    "polylogue.daemon.http",
)


@pytest.mark.parametrize("module_name", _PUBLIC_SURFACE_MODULES)
def test_public_surface_does_not_bind_provider_enum(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert getattr(module, "Provider", None) is None, (
        f"{module_name} must not import the provider-wire Provider enum (#1810); "
        "bridge source tokens through core.sources.source_name_to_origin instead."
    )


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
