"""Unit tests for the build_context_pack MCP tool."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Session
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.mcp.context_pack import select_context_pack_sessions
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.types import ContentBlockType, Provider
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import (
    EXPECTED_TOOL_NAMES,
    MCPServerUnderTest,
    invoke_surface,
)


def _seed_context_pack_archive(tmp_path: Path, *, provider_session_id: str) -> Path:
    """Seed an archive with one session and return its root."""
    archive_root = tmp_path / "archive"
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=provider_session_id,
                title="Context pack fixture",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="context pack body",
                        content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="context pack body")],
                    )
                ],
            )
        )
    return archive_root


class TestBuildContextPackRegistration:
    """Verify the build_context_pack tool is registered and callable."""

    def test_tool_name_is_in_expected_set(self) -> None:
        """The tool is listed in EXPECTED_TOOL_NAMES."""
        assert "build_context_pack" in EXPECTED_TOOL_NAMES

    def test_tool_is_registered_on_server(self, mcp_server: MCPServerUnderTest) -> None:
        """The tool is present in the server tool manager."""
        tools = mcp_server._tool_manager._tools
        assert "build_context_pack" in tools
        assert callable(tools["build_context_pack"].fn)

    @pytest.mark.asyncio
    async def test_context_pack_broadens_zero_result_archaeology_query(self) -> None:
        """A pasted multi-token archaeology query falls back to recall terms."""
        conv = make_conv(
            id="test:conv-1",
            provider=Provider.CLAUDE_AI,
            title="Replay Replacement History",
            messages=[make_msg(id="m1", role=Role.USER, text="event_replacements")],
        )
        seen_queries: list[tuple[str, ...]] = []

        async def query_sessions(spec: SessionQuerySpec) -> list[Session]:
            seen_queries.append(spec.query_terms)
            if spec.query_terms == ("event_replacements",):
                return [conv]
            return []

        def coerce_limit(value: object) -> int:
            return int(str(value))

        selection = await select_context_pack_sessions(
            query_sessions,
            coerce_limit,
            project_path=None,
            project_repo=None,
            since="2026-04-01",
            until=None,
            origin=None,
            query="supersedes_event_id event_replacements equivalence_key",
            limit=5,
        )

        assert selection.sessions == [conv]
        assert selection.match_strategy == "term_recall"
        assert ("supersedes_event_id event_replacements equivalence_key",) in seen_queries
        assert ("event_replacements",) in seen_queries

    @pytest.mark.asyncio
    async def test_context_pack_relaxes_project_filter_after_recall_miss(self) -> None:
        """Project filters are relaxed only after strict and in-project recall miss."""
        conv = make_conv(
            id="test:conv-2",
            provider=Provider.CLAUDE_AI,
            title="Target Vision Archaeology",
            messages=[make_msg(id="m1", role=Role.USER, text="source_material_id")],
        )

        async def query_sessions(spec: SessionQuerySpec) -> list[Session]:
            if spec.cwd_prefix is None and spec.query_terms == ("source_material_id",):
                return [conv]
            return []

        def coerce_limit(value: object) -> int:
            return int(str(value))

        selection = await select_context_pack_sessions(
            query_sessions,
            coerce_limit,
            project_path="/realm/project/sinex",
            project_repo=None,
            since="2026-04-01",
            until=None,
            origin=None,
            query="source_material_id anchor_byte",
            limit=5,
        )

        assert selection.sessions == [conv]
        assert selection.match_strategy == "relaxed_project_term_recall"
        assert selection.relaxed_filters == ("project_path",)

    def test_summary_detail_omits_messages(self, mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
        """Summary detail level omits message bodies on the archive path."""
        from polylogue.mcp.server_support import _set_runtime_services

        archive_root = _seed_context_pack_archive(tmp_path, provider_session_id="summary-detail")

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn
        mock_services = MagicMock()
        mock_services.get_config.return_value = SimpleNamespace(
            archive_root=archive_root,
            db_path=archive_root / "polylogue.db",
        )
        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, max_sessions=1, detail_level="summary")
            parsed = json.loads(result)
            convs = parsed.get("sessions", [])
            assert convs
            assert len(convs[0].get("messages", [])) == 0
        finally:
            _set_runtime_services(None)

    def test_redact_paths_defaults_to_true(self, mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
        """Provenance.redacted is True by default on the archive path."""
        from polylogue.mcp.server_support import _set_runtime_services

        archive_root = _seed_context_pack_archive(tmp_path, provider_session_id="redact-default")

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn
        mock_services = MagicMock()
        mock_services.get_config.return_value = SimpleNamespace(
            archive_root=archive_root,
            db_path=archive_root / "polylogue.db",
        )
        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, max_sessions=1, detail_level="summary")
            parsed = json.loads(result)
            assert parsed["provenance"]["redacted"] is True
        finally:
            _set_runtime_services(None)

    def test_tool_reads_archive_file_set_without_polylogue_db(
        self, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="mcp-context-v1",
                    title="MCP archive context pack",
                    created_at="2026-01-01T00:00:00+00:00",
                    updated_at="2026-01-01T00:01:00+00:00",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="mcp context needle",
                            content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="mcp context needle")],
                        )
                    ],
                )
            )

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn
        mock_services = MagicMock()
        mock_services.get_config.return_value = SimpleNamespace(
            archive_root=archive_root,
            db_path=archive_root / "polylogue.db",
        )
        mock_services.get_repository.side_effect = AssertionError("context pack must not open monolithic storage")
        mock_services.get_archive_ops.side_effect = AssertionError("context pack must not open monolithic ops")
        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, query="needle", max_sessions=1, max_messages_per_session=1)
            parsed = json.loads(result)
            assert parsed["total_sessions"] == 1
            assert parsed["provenance"]["archive_runtime"] == "archive_file_set"
            assert parsed["provenance"]["archive_root"] == str(archive_root)
            assert parsed["provenance"]["active_db_path"] == str(archive_root / "index.db")
            session = parsed["sessions"][0]
            assert session["session_id"] == "codex-session:mcp-context-v1"
            assert session["origin"] == "codex-session"
            assert session["messages"][0]["text"] == "mcp context needle"
        finally:
            _set_runtime_services(None)

    def test_tool_reads_archive_file_set_when_polylogue_db_exists(
        self, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="mcp-context-v1-mixed",
                    title="MCP mixed archive context pack",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="mixed context needle",
                            content_blocks=[
                                ParsedContentBlock(type=ContentBlockType.TEXT, text="mixed context needle")
                            ],
                        )
                    ],
                )
            )
        db_anchor = archive_root / "polylogue.db"
        db_anchor.touch()

        tools = mcp_server._tool_manager._tools
        fn = tools["build_context_pack"].fn
        mock_services = MagicMock()
        mock_services.get_config.return_value = SimpleNamespace(
            archive_root=archive_root,
            db_path=db_anchor,
        )
        mock_services.get_repository.side_effect = AssertionError("context pack must not open monolithic storage")
        mock_services.get_archive_ops.side_effect = AssertionError("context pack must not open monolithic ops")
        _set_runtime_services(mock_services)

        try:
            result = invoke_surface(fn, query="mixed", max_sessions=1, max_messages_per_session=1)
            parsed = json.loads(result)
            assert parsed["total_sessions"] == 1
            assert parsed["provenance"]["archive_runtime"] == "archive_file_set"
            assert parsed["provenance"]["active_db_path"] == str(archive_root / "index.db")
            assert parsed["sessions"][0]["session_id"] == "codex-session:mcp-context-v1-mixed"
        finally:
            _set_runtime_services(None)


class TestContextPackModels:
    """Smoke test the Pydantic models directly."""

    def test_payload_default_construction(self) -> None:
        from polylogue.mcp.context_pack import ContextPackPayload

        payload = ContextPackPayload()
        assert payload.provenance.source == "polylogue"
        assert payload.provenance.redacted is True
        assert payload.provenance.archive_runtime == "archive_file_set"
        assert isinstance(payload.sessions, list)
        assert len(payload.sessions) == 0

    def test_redact_path_home_directory(self) -> None:
        import os

        from polylogue.mcp.context_pack import redact_path

        home = os.path.expanduser("~")
        result = redact_path(home + "/projects/foo")
        assert result == "~/projects/foo"
        assert not result.startswith(home)

    def test_redact_path_non_home_unchanged(self) -> None:
        from polylogue.mcp.context_pack import redact_path

        result = redact_path("/tmp/scratch")
        assert result == "/tmp/scratch"

    def test_redact_path_exact_home(self) -> None:
        import os

        from polylogue.mcp.context_pack import redact_path

        home = os.path.expanduser("~")
        result = redact_path(home)
        assert result == "~"
