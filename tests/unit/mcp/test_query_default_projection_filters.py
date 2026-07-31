"""Regression tests for polylogue-hnl7: ``query()``'s default (unit-source)
projection silently dropped origin/tag/repo/since/until/min_messages/
max_messages/min_words -- only expression/limit/continuation reached
``query_units``. Live-archive evidence: filtering ``role:user`` messages by
``origin="claude-code-session"`` returned the ALL-origin total, and an
unrecognised origin was accepted silently instead of rejected.

These tests seed two sessions with different origins and assert the default
projection actually scopes to the requested origin, and that an unknown
origin / an unsupported ``sort`` on the default projection fail loudly
instead of being ignored.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.mcp.declarations.models import MCPCapabilities
from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


@contextmanager
def _installed_runtime_services(archive_root: Path) -> Iterator[None]:
    """Install real RuntimeServices for ``archive_root``, restoring whatever was active before."""
    from polylogue.config import Config
    from polylogue.mcp import server_support
    from polylogue.services import RuntimeServices

    services = RuntimeServices(
        config=Config(archive_root=archive_root, render_root=archive_root.parent / "render", sources=[]),
    )
    try:
        original: RuntimeServices | None = server_support._get_runtime_services()
    except RuntimeError:
        original = None
    server_support._set_runtime_services(services)
    try:
        yield
    finally:
        server_support._set_runtime_services(original)


def _build_tools(
    capabilities: MCPCapabilities = MCPCapabilities(),
) -> dict[str, Callable[..., str | Awaitable[str]]]:
    from polylogue.mcp.server import build_server

    server = cast(MCPServerUnderTest, build_server(capabilities=capabilities))
    return {name: tool.fn for name, tool in server._tool_manager._tools.items()}


def _seed_two_origin_sessions(archive_root: Path) -> None:
    """Write one claude-code-session and one chatgpt-export session, each
    with a single ``role:user`` message, so an origin-scoped count can be
    told apart from the archive-wide count."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        for provider, native_id in ((Provider.CLAUDE_CODE, "cc-1"), (Provider.CHATGPT, "gpt-1")):
            archive.write_parsed(
                ParsedSession(
                    source_name=provider,
                    provider_session_id=native_id,
                    title=f"origin filter probe ({provider.value})",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="origin filter probe message",
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="origin filter probe message")],
                        )
                    ],
                )
            )


class TestDefaultProjectionFilters:
    @pytest.mark.asyncio
    async def test_origin_filter_scopes_default_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_two_origin_sessions(archive_root)
        query_fn = _build_tools()["query"]

        with _installed_runtime_services(archive_root):
            unfiltered = json.loads(await invoke_surface_async(query_fn, expression="messages where role:user | count"))
            scoped = json.loads(
                await invoke_surface_async(
                    query_fn,
                    expression="messages where role:user | count",
                    origin="claude-code-session",
                )
            )

        assert unfiltered["items"][0]["count"] == 2
        # Before the fix, ``origin`` never reached ``query_units`` for the
        # default projection, so this would also read 2 (the whole-archive
        # total) instead of the origin-scoped 1.
        assert scoped["items"][0]["count"] == 1

    @pytest.mark.asyncio
    async def test_unknown_origin_rejected_loudly(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_two_origin_sessions(archive_root)
        query_fn = _build_tools()["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    query_fn,
                    expression="messages where role:user | count",
                    origin="bogus-origin",
                )
            )

        assert result.get("is_error") is True
        assert result.get("code") == "invalid_argument"
        assert "bogus-origin" in result.get("message", "")

    @pytest.mark.asyncio
    async def test_sort_rejected_on_default_projection(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        _seed_two_origin_sessions(archive_root)
        query_fn = _build_tools()["query"]

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    query_fn,
                    expression="messages where role:user | count",
                    sort="recency",
                )
            )

        assert result.get("is_error") is True
        assert result.get("code") == "invalid_argument"
