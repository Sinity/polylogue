"""Real-repository MCP integration tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.mcp import invoke_surface, invoke_surface_async
from tests.infra.storage_records import make_message, make_session


async def _insert_session(
    repo: SessionRepository,
    *,
    session_id: str,
    provider: str,
    provider_session_id: str,
    text: str,
) -> None:
    session = make_session(
        session_id=session_id,
        source_name=provider,
        provider_session_id=provider_session_id,
        title=f"{provider} session",
    )
    message = make_message(
        message_id=f"{session_id}:m1",
        session_id=session_id,
        role="user",
        text=text,
        source_name=provider,
    )
    await repo.save_session(session, [message], [])


class TestMCPRealRepositoryPaths:
    """Exercise MCP server tools against a real temporary repository."""

    @pytest.mark.asyncio
    async def test_search_uses_real_repository_and_filter_stack(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-search.db")
        repo = SessionRepository(backend=backend)

        try:
            await _insert_session(
                repo,
                session_id="chatgpt:needle",
                provider="chatgpt",
                provider_session_id="needle",
                text="finding a needle in a haystack",
            )
            await _insert_session(
                repo,
                session_id="claude-ai:other",
                provider="claude-ai",
                provider_session_id="other",
                text="something unrelated",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server()
                result = await invoke_surface_async(server._tool_manager._tools["search"].fn, query="needle", limit=10)

            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["session"]["id"] == "chatgpt:needle"
            assert parsed[0]["session"]["provider"] == "chatgpt"
            assert parsed[0]["match"]["match_surface"] == "message"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_applies_provider_filter_on_real_repository(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-list.db")
        repo = SessionRepository(backend=backend)

        try:
            await _insert_session(
                repo,
                session_id="chatgpt:one",
                provider="chatgpt",
                provider_session_id="one",
                text="chatgpt content",
            )
            await _insert_session(
                repo,
                session_id="claude-ai:one",
                provider="claude-ai",
                provider_session_id="one",
                text="claude content",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server()
                result = await invoke_surface_async(
                    server._tool_manager._tools["list_sessions"].fn,
                    provider="claude-ai",
                    limit=10,
                )

            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["id"] == "claude-ai:one"
            assert parsed[0]["provider"] == "claude-ai"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_with_invalid_limit_clamps_on_real_repository(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-invalid-limit.db")
        repo = SessionRepository(backend=backend)

        try:
            await _insert_session(
                repo,
                session_id="chatgpt:one",
                provider="chatgpt",
                provider_session_id="one",
                text="first result",
            )
            await _insert_session(
                repo,
                session_id="chatgpt:two",
                provider="chatgpt",
                provider_session_id="two",
                text="second result",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server()
                result = await invoke_surface_async(server._tool_manager._tools["list_sessions"].fn, limit=-1)

            parsed = json.loads(result)
            assert isinstance(parsed, list)
            assert len(parsed) == 1
        finally:
            await backend.close()

    def test_add_list_remove_tag_roundtrip(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-mutations-real.db")
        repo = SessionRepository(backend=backend)

        try:
            conv_id = "chatgpt:real-tag"
            asyncio.run(
                _insert_session(
                    repo,
                    session_id=conv_id,
                    provider="chatgpt",
                    provider_session_id="real-tag",
                    text="tag me",
                )
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server(role="write")

                initial_tags = json.loads(
                    invoke_surface(server._tool_manager._tools["list_tags"].fn, provider="chatgpt")
                )
                assert initial_tags.get("important", 0) == 0

                add_payload = json.loads(
                    invoke_surface(
                        server._tool_manager._tools["add_tag"].fn,
                        session_id=conv_id,
                        tag="important",
                    )
                )
                assert add_payload["status"] == "ok"

                list_payload = json.loads(
                    invoke_surface(server._tool_manager._tools["list_tags"].fn, provider="chatgpt")
                )
                assert list_payload.get("important", 0) == 1

                remove_payload = json.loads(
                    invoke_surface(
                        server._tool_manager._tools["remove_tag"].fn,
                        session_id=conv_id,
                        tag="important",
                    )
                )
                assert remove_payload["status"] == "ok"

                list_after = json.loads(invoke_surface(server._tool_manager._tools["list_tags"].fn, provider="chatgpt"))
                assert list_after.get("important", 0) == 0
        finally:
            asyncio.run(backend.close())
