"""Real-repository MCP integration tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from polylogue.storage.repository import ConversationRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.mcp import invoke_surface, invoke_surface_async
from tests.infra.storage_records import make_conversation, make_message


async def _insert_conversation(
    repo: ConversationRepository,
    *,
    conversation_id: str,
    provider: str,
    provider_conversation_id: str,
    text: str,
) -> None:
    conversation = make_conversation(
        conversation_id=conversation_id,
        provider_name=provider,
        provider_conversation_id=provider_conversation_id,
        title=f"{provider} conversation",
    )
    message = make_message(
        message_id=f"{conversation_id}:m1",
        conversation_id=conversation_id,
        role="user",
        text=text,
        provider_name=provider,
    )
    await repo.save_conversation(conversation, [message], [])


class TestMCPRealRepositoryPaths:
    """Exercise MCP server tools against a real temporary repository."""

    @pytest.mark.asyncio
    async def test_search_uses_real_repository_and_filter_stack(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-search.db")
        repo = ConversationRepository(backend=backend)

        try:
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:needle",
                provider="chatgpt",
                provider_conversation_id="needle",
                text="finding a needle in a haystack",
            )
            await _insert_conversation(
                repo,
                conversation_id="claude-ai:other",
                provider="claude-ai",
                provider_conversation_id="other",
                text="something unrelated",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server()
                result = await invoke_surface_async(server._tool_manager._tools["search"].fn, query="needle", limit=10)

            parsed = json.loads(result)
            assert len(parsed) == 1
            assert parsed[0]["conversation"]["id"] == "chatgpt:needle"
            assert parsed[0]["conversation"]["provider"] == "chatgpt"
            assert parsed[0]["match"]["match_surface"] == "message"
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_list_applies_provider_filter_on_real_repository(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-list.db")
        repo = ConversationRepository(backend=backend)

        try:
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:one",
                provider="chatgpt",
                provider_conversation_id="one",
                text="chatgpt content",
            )
            await _insert_conversation(
                repo,
                conversation_id="claude-ai:one",
                provider="claude-ai",
                provider_conversation_id="one",
                text="claude content",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server()
                result = await invoke_surface_async(
                    server._tool_manager._tools["list_conversations"].fn,
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
        repo = ConversationRepository(backend=backend)

        try:
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:one",
                provider="chatgpt",
                provider_conversation_id="one",
                text="first result",
            )
            await _insert_conversation(
                repo,
                conversation_id="chatgpt:two",
                provider="chatgpt",
                provider_conversation_id="two",
                text="second result",
            )

            with patch("polylogue.mcp.server._get_repo", return_value=repo):
                server = build_server()
                result = await invoke_surface_async(server._tool_manager._tools["list_conversations"].fn, limit=-1)

            parsed = json.loads(result)
            assert isinstance(parsed, list)
            assert len(parsed) == 1
        finally:
            await backend.close()

    def test_add_list_remove_tag_roundtrip(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        backend = SQLiteBackend(db_path=tmp_path / "mcp-mutations-real.db")
        repo = ConversationRepository(backend=backend)

        try:
            conv_id = "chatgpt:real-tag"
            asyncio.run(
                _insert_conversation(
                    repo,
                    conversation_id=conv_id,
                    provider="chatgpt",
                    provider_conversation_id="real-tag",
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
                        conversation_id=conv_id,
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
                        conversation_id=conv_id,
                        tag="important",
                    )
                )
                assert remove_payload["status"] == "ok"

                list_after = json.loads(invoke_surface(server._tool_manager._tools["list_tags"].fn, provider="chatgpt"))
                assert list_after.get("important", 0) == 0
        finally:
            asyncio.run(backend.close())
