"""Unit contracts for MCP server resources, prompts, exports, and payload surfaces."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Session, SessionSummary
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.types import ContentBlockType, Provider, SessionId
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import (
    EXPECTED_PROMPT_NAMES,
    EXPECTED_RESOURCE_TEMPLATE_URIS,
    EXPECTED_RESOURCE_URIS,
    EXPECTED_TOOL_NAMES,
    MCPServerUnderTest,
    invoke_surface,
    invoke_surface_async,
    make_mock_filter,
    make_polylogue_mock,
    make_simple_session,
)

SerializationCase = tuple[str, Session, dict[str, object], str]


def _write_archive_session(
    archive: ArchiveStore,
    *,
    provider: Provider = Provider.CODEX,
    native_id: str,
    title: str = "Native session",
    text: str = "archive message",
) -> str:
    return archive.write_parsed(
        ParsedSession(
            source_name=provider,
            provider_session_id=native_id,
            title=title,
            messages=[
                ParsedMessage(
                    provider_message_id=f"{native_id}-m1",
                    role=Role.USER,
                    text=text,
                    content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=text)],
                )
            ],
        )
    )


def _summary_for(session: Session) -> SessionSummary:
    return SessionSummary(
        id=SessionId(str(session.id)),
        origin=session.origin,
        title=session.display_title,
        message_count=len(session.messages),
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


SERIALIZATION_CASES: list[SerializationCase] = [
    (
        "no_timestamps",
        make_conv(id="t1", provider=Provider.UNKNOWN, title="No Times", messages=[]),
        {"created_at": None, "updated_at": None, "message_count": 0},
        "summary",
    ),
    (
        "empty_messages",
        make_conv(id="t2", provider=Provider.UNKNOWN, title="Empty", messages=[]),
        {"messages": []},
        "detail",
    ),
    (
        "empty_role",
        make_conv(
            id="t3",
            provider=Provider.UNKNOWN,
            title="Empty Role",
            messages=[make_msg(id="m1", role=Role.UNKNOWN, text="test")],
        ),
        {"messages": [{"role": "unknown"}]},
        "detail",
    ),
    (
        "null_text",
        make_conv(
            id="t4",
            provider=Provider.UNKNOWN,
            title="Null Text",
            messages=[make_msg(id="m1", role=Role.USER, text=None)],
        ),
        {"messages": [{"text": ""}]},
        "detail",
    ),
    (
        "null_timestamp",
        make_conv(
            id="t5",
            provider=Provider.UNKNOWN,
            title="No TS",
            messages=[make_msg(id="m1", role=Role.USER, text="hi")],
        ),
        {"messages": [{"timestamp": None}]},
        "detail",
    ),
    (
        "unusual_role",
        make_conv(
            id="t6",
            provider=Provider.UNKNOWN,
            title="Unusual Role",
            messages=[make_msg(id="m1", role=Role.TOOL, text="test")],
        ),
        {"messages": [{"role": "tool"}]},
        "detail",
    ),
]


@pytest.fixture
def simple_session() -> Session:
    return make_simple_session()


class TestServerSurfaceRegistration:
    """Server registration should expose the documented MCP surfaces."""

    def testbuild_server_exposes_managers(self: object, mcp_server: MCPServerUnderTest) -> None:
        assert mcp_server is not None
        assert hasattr(mcp_server, "_tool_manager")
        assert hasattr(mcp_server, "_resource_manager")
        assert hasattr(mcp_server, "_prompt_manager")

    def test_dynamic_product_tools_publish_explicit_signatures(self: object, mcp_server: MCPServerUnderTest) -> None:
        signature = inspect.signature(mcp_server._tool_manager._tools["session_profiles"].fn)

        assert "origin" in signature.parameters
        assert "provider" not in signature.parameters
        assert "workflow_shape" in signature.parameters
        assert "terminal_state" in signature.parameters
        assert "limit" in signature.parameters
        assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())

    @pytest.mark.parametrize(
        ("surface_attr", "actual_getter", "expected"),
        [
            ("tools", lambda server: set(server._tool_manager._tools.keys()), EXPECTED_TOOL_NAMES),
            ("resources", lambda server: set(server._resource_manager._resources.keys()), EXPECTED_RESOURCE_URIS),
            (
                "resource_templates",
                lambda server: set(server._resource_manager._templates.keys()),
                EXPECTED_RESOURCE_TEMPLATE_URIS,
            ),
            ("prompts", lambda server: set(server._prompt_manager._prompts.keys()), EXPECTED_PROMPT_NAMES),
        ],
    )
    @pytest.mark.contract
    def test_server_surface_contract(
        self: object,
        surface_attr: str,
        actual_getter: Callable[[MCPServerUnderTest], set[str]],
        expected: set[str],
        mcp_server: MCPServerUnderTest,
    ) -> None:
        actual = actual_getter(mcp_server)
        missing = expected - actual
        assert not missing, f"Missing {surface_attr}: {sorted(missing)}"

    def test_read_role_omits_mutation_and_maintenance_tools(self: object) -> None:
        from polylogue.mcp.server import build_server

        server = build_server(role="read")
        tools = set(server._tool_manager._tools.keys())

        assert "search" in tools
        assert "get_messages" in tools
        assert "session_profiles" in tools
        assert "add_tag" not in tools
        assert "set_metadata" not in tools
        assert "rebuild_index" not in tools
        assert "rebuild_session_insights" not in tools

    def test_write_role_omits_admin_maintenance_tools(self: object) -> None:
        from polylogue.mcp.server import build_server

        server = build_server(role="write")
        tools = set(server._tool_manager._tools.keys())

        assert "add_tag" in tools
        assert "set_metadata" in tools
        assert "rebuild_index" not in tools
        assert "rebuild_session_insights" not in tools


class TestResourceSurfaces:
    def test_stats_returns_archive_statistics(self: object, mcp_server: MCPServerUnderTest, tmp_path: Path) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            _write_archive_session(archive, provider=Provider.CHATGPT, native_id="stats-a", text="alpha beta")
            _write_archive_session(archive, provider=Provider.CHATGPT, native_id="stats-b", text="gamma delta")

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://stats"].fn)

        stats = json.loads(result)
        assert stats["total_sessions"] == 2
        assert stats["total_messages"] == 2
        assert stats["origins"] == {"chatgpt-export": 2}

    def test_stats_resource_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="resource-stats-v1",
                    messages=[
                        ParsedMessage(
                            provider_message_id="m1",
                            role=Role.USER,
                            text="resource stats v1",
                            content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text="resource stats v1")],
                        )
                    ],
                )
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("stats resource must not open monolithic ops")
            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://stats"].fn)

        stats = json.loads(result)
        assert stats["total_sessions"] == 1
        assert stats["total_messages"] == 1
        assert stats["origins"] == {"codex-session": 1}

    @pytest.mark.asyncio
    async def test_sessions_resource_returns_list(
        self: object,
        mcp_server: MCPServerUnderTest,
        tmp_path: Path,
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(archive, native_id="resource-list", title="Listed session")

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            result = await invoke_surface_async(mcp_server._resource_manager._resources["polylogue://sessions"].fn)

        payload = json.loads(result)
        assert "items" in payload
        assert payload["total"] == 1
        assert len(payload["items"]) == 1
        assert payload["items"][0]["id"] == session_id

    @pytest.mark.asyncio
    async def test_sessions_resource_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="resource-list-v1",
                title="Native list resource",
                text="listed from schema archive",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_query_store.side_effect = AssertionError("sessions resource must not open monolithic storage")
            result = await invoke_surface_async(mcp_server._resource_manager._resources["polylogue://sessions"].fn)

        payload = json.loads(result)
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == session_id
        assert payload["items"][0]["origin"] == "codex-session"

    def test_single_session_resource(
        self: object,
        mcp_server: MCPServerUnderTest,
        tmp_path: Path,
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="resource-single",
                    title="Single resource",
                    messages=[
                        ParsedMessage(
                            provider_message_id=f"resource-single-m{i}",
                            role=Role.USER,
                            text=f"line {i}",
                            content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=f"line {i}")],
                        )
                        for i in range(2)
                    ],
                )
            )

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://session/{conv_id}"].fn,
                conv_id=session_id,
            )

        conv = json.loads(result)
        assert conv["id"] == session_id
        assert conv["message_count"] == 2
        assert "messages" not in conv

    def test_single_session_resource_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="resource-single-v1",
                title="Native single resource",
                text="single from schema archive",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("session resource must not open monolithic ops")
            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://session/{conv_id}"].fn,
                conv_id=session_id[:12],
            )

        conv = json.loads(result)
        assert conv["id"] == session_id
        assert conv["title"] == "Native single resource"
        assert conv["message_count"] == 1

    def test_session_resource_not_found(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = MagicMock()
            mock_poly.get_session_summary = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://session/{conv_id}"].fn,
                conv_id="nonexistent",
            )

        result_dict = json.loads(result)
        assert "error" in result_dict

    def test_messages_resource_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="resource-messages-v1",
                title="Native messages resource",
                text="message body from schema archive",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("messages resource must not open monolithic ops")
            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://messages/{conv_id}"].fn,
                conv_id=session_id,
            )

        payload = json.loads(result)
        assert payload["session_id"] == session_id
        assert payload["total"] == 1
        assert payload["messages"][0]["text"] == "message body from schema archive"

    @pytest.mark.asyncio
    async def test_origin_recent_resource_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            codex_session_id = _write_archive_session(
                archive,
                provider=Provider.CODEX,
                native_id="origin-recent-codex-v1",
                title="Recent codex v1",
            )
            _write_archive_session(
                archive,
                provider=Provider.CHATGPT,
                native_id="origin-recent-chatgpt-v1",
                title="Recent chatgpt v1",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_query_store.side_effect = AssertionError("origin resource must not open monolithic storage")
            result = await invoke_surface_async(
                mcp_server._resource_manager._templates["polylogue://origin/{name}/recent"].fn,
                name="codex-session",
            )

        payload = json.loads(result)
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == codex_session_id
        assert payload["items"][0]["origin"] == "codex-session"

    def test_tags_resource(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.list_tags.return_value = {"feature": 10, "bug": 5}
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://tags"].fn)

        parsed = json.loads(result)
        assert parsed == {"feature": 10, "bug": 5}

    def test_readiness_resource(self: object, mcp_server: MCPServerUnderTest) -> None:
        mock_check = MagicMock()
        mock_check.name = "database"
        mock_check.status.value = "ok"

        mock_report = MagicMock()
        mock_report.checks = [mock_check]
        mock_report.summary = "All systems operational"

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.readiness.get_readiness") as mock_get_readiness,
        ):
            mock_get_config.return_value = MagicMock()
            mock_get_readiness.return_value = mock_report

            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://readiness"].fn)

        parsed = json.loads(result)
        assert len(parsed["checks"]) == 1
        assert parsed["summary"] == "All systems operational"


class TestArchiveGenericToolSurfaces:
    @pytest.mark.asyncio
    async def test_list_sessions_tool_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(archive, native_id="tool-list-v1", title="Tool list v1")

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("list tool must not open monolithic ops")
            result = await invoke_surface_async(mcp_server._tool_manager._tools["list_sessions"].fn)

        payload = json.loads(result)
        assert payload["total"] == 1
        assert payload["items"][0]["id"] == session_id

    @pytest.mark.asyncio
    async def test_search_tool_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-search-v1",
                title="Tool search v1",
                text="needle appears in schema archive",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("search tool must not open monolithic ops")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="needle",
            )

        payload = json.loads(result)
        assert payload["hits"][0]["session"]["id"] == session_id
        assert payload["hits"][0]["match"]["message_id"]

    def test_get_session_tools_read_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-get-v1",
                title="Tool get v1",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("get tools must not open monolithic Polylogue")
            result = invoke_surface(mcp_server._tool_manager._tools["get_session"].fn, id=session_id[:12])
            summary_result = invoke_surface(
                mcp_server._tool_manager._tools["get_session_summary"].fn,
                id=session_id,
            )

        payload = json.loads(result)
        summary_payload = json.loads(summary_result)
        assert payload["id"] == session_id
        assert summary_payload["id"] == session_id

    def test_get_messages_tool_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-messages-v1",
                title="Tool messages v1",
                text="tool message from schema archive",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_polylogue.side_effect = AssertionError("get_messages must not open monolithic Polylogue")
            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                session_id=session_id,
            )

        payload = json.loads(result)
        assert payload["session_id"] == session_id
        assert payload["messages"][0]["text"] == "tool message from schema archive"


class TestPromptSurfaces:
    @pytest.mark.asyncio
    async def test_analyze_errors_prompt_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            _write_archive_session(
                archive,
                native_id="prompt-errors-v1",
                title="Prompt errors v1",
                text="error from schema archive prompt path",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_query_store.side_effect = AssertionError("prompt must not open monolithic storage")
            result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert "1 sessions" in result
        assert "error from schema archive" in result
        assert '"origin": "codex-session"' in result
        assert '"provider"' not in result

    @pytest.mark.asyncio
    async def test_compare_sessions_prompt_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            first_id = _write_archive_session(
                archive,
                native_id="prompt-compare-a-v1",
                title="Prompt compare A",
                text="first archive prompt session",
            )
            second_id = _write_archive_session(
                archive,
                native_id="prompt-compare-b-v1",
                title="Prompt compare B",
                text="second archive prompt session",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_query_store.side_effect = AssertionError("compare prompt must not open monolithic storage")
            result = await invoke_surface_async(
                mcp_server._prompt_manager._prompts["compare_sessions"].fn,
                id1=first_id,
                id2=second_id,
            )

        assert "Prompt compare A" in result
        assert "Prompt compare B" in result
        assert '"origin": "codex-session"' in result
        assert '"provider"' not in result

    @pytest.mark.asyncio
    async def test_extract_code_prompt_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            _write_archive_session(
                archive,
                native_id="prompt-code-v1",
                title="Prompt code v1",
                text="```python\nprint('schema archive')\n```",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            mock_get_query_store.side_effect = AssertionError("code prompt must not open monolithic storage")
            result = await invoke_surface_async(
                mcp_server._prompt_manager._prompts["extract_code"].fn,
                language="python",
            )

        assert "Found 1 code blocks" in result
        assert "schema archive" in result

    @pytest.mark.asyncio
    async def test_analyze_errors_with_sessions(
        self: object,
        simple_session: Session,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        simple_session.messages.to_list()[0].text = "Got an error while running"

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[simple_session])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert isinstance(result, str)
        assert "error" in result.lower()

    @pytest.mark.asyncio
    async def test_analyze_errors_limits_error_contexts_to_20(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id="big-errors",
                    title="Errors",
                    messages=[
                        ParsedMessage(
                            provider_message_id=f"m{i}",
                            role=Role.USER,
                            text=f"error number {i} occurred",
                            content_blocks=[
                                ParsedContentBlock(type=ContentBlockType.TEXT, text=f"error number {i} occurred")
                            ],
                        )
                        for i in range(30)
                    ],
                )
            )

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert "20 error instances" in result

    @pytest.mark.asyncio
    async def test_analyze_errors_no_matches(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert "0 sessions" in result

    @pytest.mark.asyncio
    async def test_summarize_week_empty(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["summarize_week"].fn)

        assert "0 sessions" in result
        assert "0 messages" in result

    @pytest.mark.asyncio
    async def test_extract_code_no_code_blocks(self: object, mcp_server: MCPServerUnderTest) -> None:
        conv = make_conv(
            id="nocode",
            provider=Provider.UNKNOWN,
            title="No Code",
            messages=[make_msg(id="m1", role=Role.USER, text="Just text, no code")],
        )

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["extract_code"].fn)

        assert "0 code blocks" in result

    @pytest.mark.asyncio
    async def test_extract_code_with_language_filter(self: object, mcp_server: MCPServerUnderTest) -> None:
        conv = make_conv(
            id="code",
            provider=Provider.UNKNOWN,
            title="Code",
            messages=[
                make_msg(
                    id="m1",
                    role=Role.ASSISTANT,
                    text="```python\nprint('hi')\n```\n```javascript\nconsole.log('hi')\n```",
                )
            ],
        )

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                result = await invoke_surface_async(
                    mcp_server._prompt_manager._prompts["extract_code"].fn,
                    language="python",
                )

        assert "python" in result.lower()

    @pytest.mark.asyncio
    async def test_extract_code_null_message_text(self: object, mcp_server: MCPServerUnderTest) -> None:
        conv = make_conv(
            id="nulltext",
            provider=Provider.UNKNOWN,
            title="Null",
            messages=[make_msg(id="m1", role=Role.ASSISTANT, text=None)],
        )

        with patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store:
            mock_get_query_store.return_value = MagicMock()
            with patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls:
                mock_filter_cls.return_value = make_mock_filter(results=[conv])

                result = await invoke_surface_async(mcp_server._prompt_manager._prompts["extract_code"].fn)

        assert isinstance(result, str)

    def test_compare_sessions_prompt(
        self: object,
        simple_session: Session,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_eager = AsyncMock(side_effect=[simple_session, simple_session])
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(
                mcp_server._prompt_manager._prompts["compare_sessions"].fn,
                id1="test:conv-1",
                id2="test:conv-2",
            )

        assert "Compare" in result
        assert "Session 1" in result
        assert "Session 2" in result

    @pytest.mark.asyncio
    async def test_extract_patterns_prompt(
        self: object,
        simple_session: Session,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
            patch("polylogue.archive.filter.filters.SessionFilter") as mock_filter_cls,
        ):
            mock_get_query_store.return_value = MagicMock()
            mock_filter_cls.return_value = make_mock_filter(results=[simple_session])

            result = await invoke_surface_async(mcp_server._prompt_manager._prompts["extract_patterns"].fn)

        assert isinstance(result, str)
        assert "patterns" in result.lower()


class TestExportSessionTool:
    def test_get_messages_tool_native_content_projection_is_not_applied(
        self: object,
        mcp_server: MCPServerUnderTest,
        tmp_path: Path,
    ) -> None:
        """PROD GAP: the archive ``get_messages`` path builds the content
        projection request but never applies it — ``archive_messages_payload``
        joins all block text verbatim. So ``no_code_blocks=True`` does NOT strip
        fenced code on the archive store. This test pins the *current* archive
        contract (text returned verbatim) so the divergence is visible and the
        gap is tracked; the monolithic behaviour stripped code via
        ``content_projection``.
        """
        body = "Alpha\n\n```python\nprint('x')\n```\n\nOmega"
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = archive.write_parsed(
                ParsedSession(
                    source_name=Provider.CHATGPT,
                    provider_session_id="projection",
                    title="Projected Session",
                    messages=[
                        ParsedMessage(
                            provider_message_id="msg-1",
                            role=Role.ASSISTANT,
                            text=body,
                            content_blocks=[ParsedContentBlock(type=ContentBlockType.TEXT, text=body)],
                        )
                    ],
                )
            )

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "polylogue.db",
            )
            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                session_id=session_id,
                no_code_blocks=True,
            )

        payload = json.loads(result)
        # Native path returns the code fence intact (projection not applied).
        assert payload["messages"][0]["text"] == body

    def test_export_markdown(self: object, simple_session: Session, mcp_server: MCPServerUnderTest) -> None:
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
            patch("polylogue.rendering.formatting.format_session") as mock_format,
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.get_session = AsyncMock(return_value=simple_session)
            mock_get_polylogue.return_value = mock_poly
            mock_format.return_value = "# Test Session\n\nFormatted content"

            result = invoke_surface(
                mcp_server._tool_manager._tools["export_session"].fn,
                id="test:conv-123",
                format="markdown",
            )

        assert "Test Session" in result
        mock_format.assert_called_once()
        call_args = mock_format.call_args
        assert call_args[0][0] == simple_session
        assert call_args[0][1] == "markdown"

    def test_export_not_found(self: object, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.get_session = AsyncMock(return_value=None)
            mock_get_polylogue.return_value = mock_poly

            result = invoke_surface(mcp_server._tool_manager._tools["export_session"].fn, id="nonexistent")

        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"].lower()

    def test_export_invalid_format_falls_back_to_markdown(
        self: object,
        simple_session: Session,
        mcp_server: MCPServerUnderTest,
    ) -> None:
        with (
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
            patch("polylogue.rendering.formatting.format_session") as mock_format,
        ):
            mock_poly = make_polylogue_mock()
            mock_poly.get_session = AsyncMock(return_value=simple_session)
            mock_get_polylogue.return_value = mock_poly
            mock_format.return_value = "# Content"

            invoke_surface(
                mcp_server._tool_manager._tools["export_session"].fn,
                id="test:conv-123",
                format="invalid_format",
            )

        assert mock_format.call_args.args[1] == "markdown"


class TestTypedPayloads:
    def test_session_to_summary_dict(self: object, simple_session: Session) -> None:
        from polylogue.mcp.payloads import MCPSessionSummaryPayload

        result = MCPSessionSummaryPayload.from_session(simple_session).model_dump(mode="json")

        assert result["id"] == "test:conv-123"
        assert result["origin"] == "chatgpt-export"
        assert result["message_count"] == 2
        assert "created_at" in result
        assert "updated_at" in result

    def test_session_to_full_dict(self: object, simple_session: Session) -> None:
        from polylogue.mcp.payloads import MCPSessionDetailPayload

        result = MCPSessionDetailPayload.from_session(simple_session).model_dump(mode="json")

        assert "messages" in result
        assert len(result["messages"]) == 2
        msg = result["messages"][0]
        assert {"id", "role", "text", "timestamp"} <= msg.keys()

    def test_session_to_full_dict_applies_projection(self: object) -> None:
        from polylogue.mcp.payloads import MCPSessionDetailPayload

        session = make_conv(
            id="projected",
            provider=Provider.CHATGPT,
            title="Projected",
            messages=[
                make_msg(
                    id="m1",
                    role="assistant",
                    text="Alpha\n\n```python\nprint('x')\n```\n\nOmega",
                )
            ],
        )

        result = MCPSessionDetailPayload.from_session(
            session,
            content_projection=ContentProjectionSpec.prose_only(),
        ).model_dump(mode="json")

        assert result["messages"][0]["text"] == "Alpha\n\nOmega"

    @pytest.mark.parametrize(("case_id", "conv", "expected_fields", "payload_kind"), SERIALIZATION_CASES)
    def test_serialization_edge_cases(
        self: object,
        case_id: str,
        conv: Session,
        expected_fields: dict[str, object],
        payload_kind: str,
    ) -> None:
        if payload_kind == "summary":
            from polylogue.mcp.payloads import MCPSessionSummaryPayload

            result = MCPSessionSummaryPayload.from_session(conv).model_dump(mode="json")
        else:
            from polylogue.mcp.payloads import MCPSessionDetailPayload

            result = MCPSessionDetailPayload.from_session(conv).model_dump(mode="json")

        for key, expected_value in expected_fields.items():
            if key == "messages":
                assert key in result
                if expected_value:
                    assert isinstance(expected_value, list)
                    for index, expected_msg in enumerate(expected_value):
                        assert isinstance(expected_msg, dict)
                        for msg_key, msg_val in expected_msg.items():
                            assert result[key][index][msg_key] == msg_val, f"Failed for case {case_id}: {msg_key}"
                else:
                    assert result[key] == expected_value
            else:
                assert result[key] == expected_value, f"Failed for case {case_id}: {key}"


class TestClampLimit:
    @pytest.mark.parametrize(
        ("raw_limit", "expected"),
        [
            (10, 10),
            (1, 1),
            # Values above the 1000 cap clamp to 1000.
            (5000, 1000),
            (99999, 1000),
            (10001, 1000),
            (1000, 1000),
            (0, 1),
            (-5, 1),
            ("10", 10),
            ("not-a-number", 10),
            ([1, 2], 10),
        ],
    )
    def test_clamp_limit_contract(self: object, raw_limit: object, expected: int) -> None:
        from polylogue.mcp.server_support import _clamp_limit

        assert _clamp_limit(raw_limit) == expected
