"""Real-repository MCP integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from polylogue.mcp.server_support import _set_runtime_services
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.mcp import invoke_surface, invoke_surface_async


def _content_hash(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


def _seed_session(
    archive_root: Path,
    *,
    session_id: str,
    origin: str,
    native_id: str,
    text: str,
) -> None:
    with ArchiveStore(archive_root):
        pass
    message_id = f"{session_id}:m1"
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        conn = archive._conn
        conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
            (native_id, origin, f"{origin} session", _content_hash(f"session:{session_id}")),
        )
        conn.execute(
            "INSERT INTO messages (session_id, native_id, position, role, content_hash) VALUES (?, ?, ?, ?, ?)",
            (session_id, "m1", 0, "user", _content_hash(f"message:{message_id}")),
        )
        conn.execute(
            "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, ?, ?, ?)",
            (message_id, session_id, 0, "text", text),
        )
        conn.commit()


def _prepare_mcp_archive(monkeypatch: pytest.MonkeyPatch, archive_root: Path) -> None:
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    _set_runtime_services(None)


class TestMCPRealRepositoryPaths:
    """Exercise MCP server tools against a real temporary repository."""

    @pytest.mark.asyncio
    async def test_search_uses_real_repository_and_filter_stack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _prepare_mcp_archive(monkeypatch, archive_root)
        _seed_session(
            archive_root,
            session_id="chatgpt-export:needle",
            origin="chatgpt-export",
            native_id="needle",
            text="finding a needle in a haystack",
        )
        _seed_session(
            archive_root,
            session_id="claude-ai-export:other",
            origin="claude-ai-export",
            native_id="other",
            text="something unrelated",
        )

        server = build_server()
        result = await invoke_surface_async(server._tool_manager._tools["search"].fn, query="needle", limit=10)

        parsed = json.loads(result)
        assert parsed["total"] == 1
        assert parsed["hits"][0]["session"]["id"] == "chatgpt-export:needle"
        assert parsed["hits"][0]["session"]["origin"] == "chatgpt-export"
        assert parsed["hits"][0]["match"]["match_surface"] == "message"

    @pytest.mark.asyncio
    async def test_list_applies_provider_filter_on_real_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _prepare_mcp_archive(monkeypatch, archive_root)
        _seed_session(
            archive_root,
            session_id="chatgpt-export:one",
            origin="chatgpt-export",
            native_id="one",
            text="chatgpt content",
        )
        _seed_session(
            archive_root,
            session_id="claude-ai-export:one",
            origin="claude-ai-export",
            native_id="one",
            text="claude content",
        )

        server = build_server()
        result = await invoke_surface_async(
            server._tool_manager._tools["list_sessions"].fn,
            origin="claude-ai-export",
            limit=10,
        )

        parsed = json.loads(result)
        assert parsed["total"] == 1
        assert parsed["items"][0]["id"] == "claude-ai-export:one"
        assert parsed["items"][0]["origin"] == "claude-ai-export"

    @pytest.mark.asyncio
    async def test_list_with_invalid_limit_clamps_on_real_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _prepare_mcp_archive(monkeypatch, archive_root)
        _seed_session(
            archive_root,
            session_id="chatgpt-export:one",
            origin="chatgpt-export",
            native_id="one",
            text="first result",
        )
        _seed_session(
            archive_root,
            session_id="chatgpt-export:two",
            origin="chatgpt-export",
            native_id="two",
            text="second result",
        )

        server = build_server()
        result = await invoke_surface_async(server._tool_manager._tools["list_sessions"].fn, limit=-1)

        parsed = json.loads(result)
        assert parsed["total"] == 2
        assert len(parsed["items"]) == 1

    def test_add_list_remove_tag_roundtrip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _prepare_mcp_archive(monkeypatch, archive_root)
        conv_id = "chatgpt-export:real-tag"
        _seed_session(
            archive_root,
            session_id=conv_id,
            origin="chatgpt-export",
            native_id="real-tag",
            text="tag me",
        )
        server = build_server(role="write")

        initial_tags = json.loads(invoke_surface(server._tool_manager._tools["list_tags"].fn, origin="chatgpt-export"))
        assert initial_tags.get("important", 0) == 0

        add_payload = json.loads(
            invoke_surface(
                server._tool_manager._tools["add_tag"].fn,
                session_id=conv_id,
                tag="important",
            )
        )
        assert add_payload["status"] == "ok"

        list_payload = json.loads(invoke_surface(server._tool_manager._tools["list_tags"].fn, origin="chatgpt-export"))
        assert list_payload.get("important", 0) == 1

        remove_payload = json.loads(
            invoke_surface(
                server._tool_manager._tools["remove_tag"].fn,
                session_id=conv_id,
                tag="important",
            )
        )
        assert remove_payload["status"] == "ok"

        list_after = json.loads(invoke_surface(server._tool_manager._tools["list_tags"].fn, origin="chatgpt-export"))
        assert list_after.get("important", 0) == 0
