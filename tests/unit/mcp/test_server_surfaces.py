"""Unit contracts for MCP server resources, prompts, exports, and payload surfaces."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polylogue.api import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.archive.models import Session, SessionSummary
from polylogue.archive.semantic.content_projection import ContentProjectionSpec
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility, BlockType, Provider
from polylogue.core.refs import EvidenceRef
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import ArchiveAssertionEnvelope
from polylogue.surfaces.payloads import (
    ArchiveDebtListPayload,
    ArchiveDebtTotalsPayload,
    AssertionClaimPayload,
    ImportExplainEntryPayload,
    ImportExplainPayload,
    ImportProducedRowsPayload,
    PublicRefResolutionPayload,
)
from polylogue.types import SessionId
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
    working_directories: list[str] | None = None,
    blocks: list[dict[str, Any]] | None = None,
) -> str:
    parsed_blocks = (
        [ParsedContentBlock.model_validate(block) for block in blocks]
        if blocks is not None
        else [ParsedContentBlock(type=BlockType.TEXT, text=text)]
    )
    return archive.write_parsed(
        ParsedSession(
            source_name=provider,
            provider_session_id=native_id,
            title=title,
            working_directories=working_directories or [],
            messages=[
                ParsedMessage(
                    provider_message_id=f"{native_id}-m1",
                    role=Role.USER,
                    text=text,
                    blocks=parsed_blocks,
                )
            ],
        )
    )


def _materialize_run_projection(index_db: Path) -> None:
    """Rebuild session insights for richer digest-derived run-projection rows."""
    from polylogue.storage.insights.session.rebuild import rebuild_session_insights_sync
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(index_db) as conn:
        rebuild_session_insights_sync(conn)


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
        if surface_attr == "tools":
            extra = actual - expected
            assert not extra, f"Unexpected {surface_attr}: {sorted(extra)}"

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
                db_path=archive_root / "index.db",
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
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text="resource stats v1")],
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
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("stats resource must not open archive operations")
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
                db_path=archive_root / "index.db",
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
                title="Archive list resource",
                text="listed from archive index",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_query_store.side_effect = AssertionError("sessions resource must not open archive query store")
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
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"line {i}")],
                        )
                        for i in range(2)
                    ],
                )
            )

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
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
                text="single from archive index",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("session resource must not open archive operations")
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
        assert "message" in result_dict

    def test_messages_resource_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="resource-messages-v1",
                title="Native messages resource",
                text="message body from archive index",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("messages resource must not open archive operations")
            result = invoke_surface(
                mcp_server._resource_manager._templates["polylogue://messages/{conv_id}"].fn,
                conv_id=session_id,
            )

        payload = json.loads(result)
        assert payload["session_id"] == session_id
        assert payload["total"] == 1
        assert payload["messages"][0]["text"] == "message body from archive index"

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
                db_path=archive_root / "index.db",
            )
            mock_get_query_store.side_effect = AssertionError("origin resource must not open archive query store")
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
        from polylogue.core.outcomes import OutcomeCheck, OutcomeStatus

        mock_report = MagicMock()
        mock_report.checks = [
            OutcomeCheck("database", OutcomeStatus.OK, summary="DB reachable", count=1),
            OutcomeCheck(
                "index",
                OutcomeStatus.WARNING,
                summary="messages FTS stale",
                count=2,
                details=["messages_fts_row_mismatch"],
            ),
        ]
        mock_report.summary = {"ok": 1, "warning": 1, "error": 0}

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.readiness.get_readiness") as mock_get_readiness,
        ):
            mock_get_config.return_value = MagicMock()
            mock_get_readiness.return_value = mock_report

            result = invoke_surface(mcp_server._resource_manager._resources["polylogue://readiness"].fn)

        parsed = json.loads(result)
        assert len(parsed["checks"]) == 2
        assert parsed["summary"] == {"ok": 1, "warning": 1, "error": 0}
        assert parsed["checks"][0] == {"name": "database", "status": "ok"}
        component_readiness = parsed["component_readiness"]
        assert component_readiness["database"]["component"] == "database"
        assert component_readiness["database"]["scope"] == "mcp_readiness"
        assert component_readiness["database"]["state"] == "ready"
        assert component_readiness["database"]["counts"] == {"count": 1}
        assert component_readiness["index"]["state"] == "degraded"
        assert component_readiness["index"]["caveats"] == ["messages_fts_row_mismatch"]


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
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("list tool must not open archive operations")
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
                text="needle appears in archive index",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("search tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["search"].fn,
                query="needle",
            )

        payload = json.loads(result)
        assert payload["hits"][0]["session"]["id"] == session_id
        assert payload["hits"][0]["match"]["message_id"]

    @pytest.mark.asyncio
    async def test_query_units_tool_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-query-units-v1",
                title="Tool query units v1",
                text="needle terminal row from archive index",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="messages where text:needle",
            )

        payload = json.loads(result)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "message"
        assert payload["items"][0]["session_id"] == session_id
        assert payload["items"][0]["unit"] == "message"

    @pytest.mark.asyncio
    async def test_query_units_tool_applies_session_scope_filters(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            kept_id = _write_archive_session(
                archive,
                native_id="tool-query-units-repo-kept",
                text="shared terminal row",
                working_directories=["/realm/project/polylogue"],
            )
            _write_archive_session(
                archive,
                native_id="tool-query-units-repo-dropped",
                text="shared terminal row",
                working_directories=["/realm/project/sinex"],
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="messages where text:terminal",
                repo="polylogue",
            )

        payload = json.loads(result)
        assert [item["session_id"] for item in payload["items"]] == [kept_id]

    @pytest.mark.asyncio
    async def test_query_units_tool_returns_bounded_delegation_rows(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        instruction = "inspect delegation routing " + "carefully " * 40
        with ArchiveStore(archive_root) as archive:
            parent_id = _write_archive_session(
                archive,
                native_id="tool-query-units-delegation",
                title="Delegation query parent",
                text="dispatch a bounded subagent task",
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_id": "task-delegation",
                        "tool_name": "Task",
                        "tool_input": {"prompt": instruction, "model": "haiku"},
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "task-delegation",
                        "text": "dispatch failed before child creation",
                        "tool_result_is_error": 1,
                    },
                ],
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="delegations where mapping_state:unresolved AND instruction:routing",
            )

        payload = json.loads(result)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "delegation"
        [item] = payload["items"]
        assert item["parent_session_id"] == parent_id
        assert item["mapping_state"] == "unresolved"
        assert item["instruction_preview"] == instruction[:240]
        assert item["instruction_truncated"] is True
        assert "instruction_payload" not in item

        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as facade:
            with patch("polylogue.mcp.server._get_polylogue", return_value=facade):
                resolved = await invoke_surface_async(
                    mcp_server._tool_manager._tools["resolve_ref"].fn,
                    ref=item["delegation_ref"],
                )
        card = json.loads(resolved)
        assert card["payload_kind"] == "delegation-card"
        assert card["summary"] == instruction[:240]
        assert card["payload"]["instruction"] == instruction
        assert card["payload"]["attempt"]["parent_session_id"] == parent_id

    @pytest.mark.asyncio
    async def test_query_units_tool_returns_run_rows_without_archive_operations(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-query-units-run",
                title="Tool query units run",
                text="subagent run terminal row",
                working_directories=["/realm/project/polylogue"],
                blocks=[
                    {
                        "type": "tool_use",
                        "tool_id": "task-use",
                        "tool_name": "Task",
                        "tool_input": {
                            "subagent_type": "Explore",
                            "taskId": "task-run",
                            "child_session_id": "codex-session:tool-query-units-run-child",
                            "prompt": "Map run query unit wiring.",
                        },
                    },
                    {
                        "type": "tool_result",
                        "tool_id": "task-use",
                        "text": "Subagent done: run query unit wired.\n3 passed in 0.12s",
                    },
                ],
            )
        _materialize_run_projection(archive_root / "index.db")
        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="runs where role:subagent AND status:completed AND agent:Explore",
            )

        payload = json.loads(result)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "run"
        [item] = payload["items"]
        assert item["unit"] == "run"
        assert item["session_id"] == session_id
        assert item["role"] == "subagent"
        assert item["status"] == "completed"
        assert item["agent_ref"] == "agent:codex/Explore"

    @pytest.mark.asyncio
    async def test_query_units_tool_returns_observed_event_rows_without_archive_operations(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-query-units-event",
                title="Tool query units event",
                text="observed event terminal row",
                working_directories=["/realm/project/polylogue"],
            )
        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="observed-events where session.repo:polylogue AND kind:session_started",
            )

        payload = json.loads(result)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "observed-event"
        [item] = payload["items"]
        assert item["unit"] == "observed-event"
        assert item["session_id"] == session_id
        assert item["kind"] == "session_started"
        assert item["subject_ref"] == f"session:{session_id}"

    @pytest.mark.asyncio
    async def test_query_units_tool_returns_context_snapshot_rows_without_archive_operations(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-query-units-context",
                title="Tool query units context",
                text="review injection context for PR #2100",
                working_directories=["/realm/project/polylogue"],
            )
        _materialize_run_projection(archive_root / "index.db")

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="context-snapshots where session.repo:polylogue AND boundary:session_start AND text:context",
            )

        payload = json.loads(result)
        assert payload["mode"] == "query-unit"
        assert payload["unit"] == "context-snapshot"
        [item] = payload["items"]
        assert item["unit"] == "context-snapshot"
        assert item["session_id"] == session_id
        assert item["boundary"] == "session_start"
        assert item["evidence_refs"] == [session_id]

    @pytest.mark.asyncio
    async def test_query_units_tool_accepts_inline_session_scope(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            kept_id = _write_archive_session(
                archive,
                provider=Provider.CODEX,
                native_id="tool-query-units-inline-kept",
                text="shared inline terminal row",
            )
            _write_archive_session(
                archive,
                provider=Provider.CHATGPT,
                native_id="tool-query-units-inline-dropped",
                text="shared inline terminal row",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("query_units tool must not open archive operations")
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["query_units"].fn,
                expression="messages where session.origin:codex-session AND text:terminal",
            )

        payload = json.loads(result)
        assert [item["session_id"] for item in payload["items"]] == [kept_id]

    @pytest.mark.asyncio
    async def test_query_units_tool_rejects_session_expression(self: object, mcp_server: MCPServerUnderTest) -> None:
        result = await invoke_surface_async(
            mcp_server._tool_manager._tools["query_units"].fn,
            expression="repo:polylogue",
        )

        payload = json.loads(result)
        assert payload["is_error"] is True
        assert payload["tool"] == "query_units"
        assert payload["code"] == "invalid_query"

    def test_get_session_summary_tool_reads_archive_file_set(
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
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("get_session_summary must not open Polylogue facade")
            result = invoke_surface(
                mcp_server._tool_manager._tools["get_session_summary"].fn,
                id=session_id[:12],
            )

        payload = json.loads(result)
        assert payload["id"] == session_id

    def test_get_messages_tool_reads_archive_file_set(
        self: object, mcp_server: MCPServerUnderTest, tmp_path: Path
    ) -> None:
        archive_root = tmp_path / "archive"
        with ArchiveStore(archive_root) as archive:
            session_id = _write_archive_session(
                archive,
                native_id="tool-messages-v1",
                title="Tool messages v1",
                text="tool message from archive index",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_polylogue.side_effect = AssertionError("get_messages must not open Polylogue facade")
            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                session_id=session_id,
            )

        payload = json.loads(result)
        assert payload["session_id"] == session_id
        assert payload["messages"][0]["text"] == "tool message from archive index"

    @pytest.mark.asyncio
    async def test_archive_debt_tool_reads_shared_facade_payload(self: object, mcp_server: MCPServerUnderTest) -> None:
        mock_poly = make_polylogue_mock()
        mock_poly.archive_debt.return_value = ArchiveDebtListPayload(
            generated_at="2026-06-20T00:00:00+00:00",
            archive_root="/tmp/polylogue-test",
            rows=(),
            totals=ArchiveDebtTotalsPayload(),
        )

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["archive_debt"].fn,
                kind="embedding,fts",
                only_actionable=True,
                limit=3,
            )

        payload = json.loads(result)
        assert payload["mode"] == "archive-debt-list"
        assert payload["rows"] == []
        mock_poly.archive_debt.assert_awaited_once_with(
            kinds=("embedding", "fts"),
            only_actionable=True,
            limit=3,
            exact_fts=False,
        )

    @pytest.mark.asyncio
    async def test_explain_import_reads_shared_facade_payload(self: object, mcp_server: MCPServerUnderTest) -> None:
        mock_poly = make_polylogue_mock()
        mock_poly.explain_import.return_value = ImportExplainPayload(
            source_path="archive:raw:raw-1",
            entries=(
                ImportExplainEntryPayload(
                    raw_ref="raw:raw-1",
                    source_path="~/.codex/sessions/a.jsonl",
                    artifact_kind="session_record_stream",
                    detected_origin="codex-session",
                    detector="source.raw_sessions",
                    parser="archive source/index evidence",
                    parser_mode="archived_raw_session",
                    produced=ImportProducedRowsPayload(sessions=1, raw_records=1),
                ),
            ),
            produced=ImportProducedRowsPayload(sessions=1, raw_records=1),
        )

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["explain_import"].fn,
                raw_ref="raw:raw-1",
                limit=7,
            )

        payload = json.loads(result)
        assert payload["mode"] == "import-explain"
        assert payload["entries"][0]["raw_ref"] == "raw:raw-1"
        mock_poly.explain_import.assert_awaited_once_with(
            None,
            raw_ref="raw:raw-1",
            source_path=None,
            source_name="unknown",
            limit=7,
            redact_paths=True,
        )

    @pytest.mark.asyncio
    async def test_compile_context_reads_shared_context_image(self: object, mcp_server: MCPServerUnderTest) -> None:
        from polylogue.context.compiler import ContextImage, ContextSegment, ContextSpec

        image = ContextImage(
            spec=ContextSpec(seed_refs=("session:session-1",), read_views=("messages",)),
            segments=(
                ContextSegment(
                    segment_id="read-view:session-1:messages",
                    kind="read_view",
                    title="Messages",
                    markdown="Message context with explicit evidence.",
                    payload_kind="messages",
                    evidence_refs=(EvidenceRef(session_id="session-1"),),
                    token_estimate=4,
                ),
            ),
            evidence_refs=(EvidenceRef(session_id="session-1"),),
            token_estimate=4,
        )
        mock_poly = make_polylogue_mock()
        mock_poly.compile_context.return_value = image

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["compile_context"].fn,
                seed_ref="session:session-1",
                read_views="messages",
                max_tokens=1000,
            )

        payload = json.loads(result)
        assert payload["spec"]["seed_refs"] == ["session:session-1"]
        assert payload["spec"]["read_views"] == ["messages"]
        assert payload["segments"][0]["payload_kind"] == "messages"
        assert payload["evidence_refs"][0]["session_id"] == "session-1"
        called_spec = mock_poly.compile_context.await_args.args[0]
        assert called_spec.seed_refs == ("session:session-1",)
        assert called_spec.read_views == ("messages",)
        assert called_spec.max_tokens == 1000

    def test_context_delivery_payload_mapper_annotations_resolve_at_runtime(self: object) -> None:
        from polylogue.mcp.payloads import MCPContextDeliveryPayload
        from polylogue.storage.sqlite.archive_tiers.context_delivery_write import ArchiveContextDeliveryEnvelope

        hints = get_type_hints(MCPContextDeliveryPayload.from_envelope)

        assert hints["envelope"] is ArchiveContextDeliveryEnvelope
        assert hints["return"] is MCPContextDeliveryPayload

    @pytest.mark.asyncio
    async def test_get_context_delivery_reads_recipient_scoped_facade_receipt(
        self: object, mcp_server: MCPServerUnderTest
    ) -> None:
        from polylogue.context.compiler import ContextImage, ContextSegment, ContextSpec, context_image_sha256
        from polylogue.core.refs import EvidenceRef
        from polylogue.storage.sqlite.archive_tiers.context_delivery_write import ArchiveContextDeliveryEnvelope

        image = ContextImage(
            spec=ContextSpec(seed_refs=("session:session-1",), read_views=()),
            segments=(
                ContextSegment(
                    segment_id="read-view:session-1:messages",
                    kind="read_view",
                    title="Delivered context",
                    markdown="exact bytes",
                    evidence_refs=(EvidenceRef(session_id="session-1"),),
                    token_estimate=2,
                ),
            ),
            evidence_refs=(EvidenceRef(session_id="session-1"),),
            token_estimate=2,
        )
        receipt = ArchiveContextDeliveryEnvelope(
            snapshot_ref="context-snapshot:session-1:explicit-recall",
            recipient_ref="agent:hermes-main",
            run_ref=None,
            boundary="explicit-recall",
            inheritance_mode="explicit",
            context_image_sha256=context_image_sha256(image),
            context_image=image,
            segment_refs=("read-view:session-1:messages",),
            evidence_refs=("session-1",),
            assertion_refs=(),
            omissions=(),
            caveats=("quoted evidence",),
            metadata={"context_image_sha256": context_image_sha256(image)},
            delivered_by_ref="user:local",
            delivered_at_ms=123,
        )
        mock_poly = make_polylogue_mock()
        mock_poly.get_context_delivery = AsyncMock(return_value=receipt)

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["get_context_delivery"].fn,
                snapshot_ref=receipt.snapshot_ref,
                recipient_ref=receipt.recipient_ref,
            )

        payload = json.loads(result)
        assert payload["snapshot_ref"] == receipt.snapshot_ref
        assert payload["context_image_sha256"] == receipt.context_image_sha256
        assert payload["context_image"]["segments"][0]["markdown"] == "exact bytes"
        mock_poly.get_context_delivery.assert_awaited_once_with(
            receipt.snapshot_ref,
            recipient_ref=receipt.recipient_ref,
        )

    @pytest.mark.asyncio
    async def test_get_context_delivery_hides_missing_receipt_or_recipient_mismatch(
        self: object, mcp_server: MCPServerUnderTest
    ) -> None:
        mock_poly = make_polylogue_mock()
        mock_poly.get_context_delivery = AsyncMock(return_value=None)

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["get_context_delivery"].fn,
                snapshot_ref="context-snapshot:session-1:explicit-recall",
                recipient_ref="agent:other-recipient",
            )

        payload = json.loads(result)
        assert payload["is_error"] is True
        assert payload["code"] == "not_found"
        assert payload["tool"] == "get_context_delivery"
        mock_poly.get_context_delivery.assert_awaited_once_with(
            "context-snapshot:session-1:explicit-recall",
            recipient_ref="agent:other-recipient",
        )

    @pytest.mark.asyncio
    async def test_list_assertion_claims_reads_shared_facade_claims(
        self: object, mcp_server: MCPServerUnderTest
    ) -> None:
        claim = ArchiveAssertionEnvelope(
            assertion_id="claim-1",
            scope_ref="repo:polylogue",
            target_ref="session:session-1",
            key=None,
            kind=AssertionKind.DECISION,
            value=None,
            body_text="Use shared assertion claim reads.",
            author_ref="agent:codex",
            author_kind="agent",
            evidence_refs=["session-1::m1"],
            status=AssertionStatus.ACTIVE,
            visibility=AssertionVisibility.PRIVATE,
            confidence=0.8,
            staleness=None,
            context_policy={"inject": True},
            supersedes=[],
            created_at_ms=1_700_000_000_000,
            updated_at_ms=1_700_000_000_100,
        )
        mock_poly = make_polylogue_mock()
        mock_poly.list_assertion_claim_payloads.return_value = [AssertionClaimPayload.from_envelope(claim)]

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["list_assertion_claims"].fn,
                kinds="decision,caveat",
                target_ref="session:session-1",
                statuses="active",
                context_inject=True,
                limit=5,
            )

        payload = json.loads(result)
        assert payload["total"] == 1
        assert payload["items"][0]["assertion_id"] == "claim-1"
        assert payload["items"][0]["kind"] == "decision"
        assert payload["items"][0]["context_policy"] == {"inject": True}
        assert payload["statuses"] == ["active"]
        assert payload["kinds"] == ["decision", "caveat"]
        mock_poly.list_assertion_claim_payloads.assert_awaited_once_with(
            kinds=("decision", "caveat"),
            target_ref="session:session-1",
            scope_ref=None,
            statuses=("active",),
            context_inject=True,
            limit=5,
        )

    @pytest.mark.asyncio
    async def test_resolve_ref_reads_shared_facade_payload(self: object, mcp_server: MCPServerUnderTest) -> None:
        mock_poly = make_polylogue_mock()
        mock_poly.resolve_ref = AsyncMock(
            return_value=PublicRefResolutionPayload(
                ref="session:session-1",
                normalized_ref="session:session-1",
                kind="session",
                resolved=True,
                payload_kind="session-summary",
                payload={"id": "session-1"},
            )
        )

        with patch("polylogue.mcp.server._get_polylogue", return_value=mock_poly):
            result = await invoke_surface_async(
                mcp_server._tool_manager._tools["resolve_ref"].fn,
                ref="session:session-1",
            )

        payload = json.loads(result)
        assert payload["mode"] == "ref-resolution"
        assert payload["resolved"] is True
        assert payload["payload_kind"] == "session-summary"
        assert payload["payload"]["id"] == "session-1"
        mock_poly.resolve_ref.assert_awaited_once_with("session:session-1")


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
                text="error from archive index prompt path",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_query_store.side_effect = AssertionError("prompt must not open archive query store")
            result = await invoke_surface_async(mcp_server._prompt_manager._prompts["analyze_errors"].fn)

        assert "1 sessions" in result
        assert "error from archive index" in result
        assert '"origin": "codex-session"' in result
        assert '"provider"' not in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt_name", "kwargs", "expected_tools"),
        [
            (
                "resume_context",
                {},
                ("find_resume_candidates", "get_resume_brief", "agent_coordination_brief", "blackboard_list"),
            ),
            (
                "postmortem_last",
                {},
                ("find_abandoned_sessions", "find_stuck_sessions", "get_postmortem_bundle", "get_pathologies"),
            ),
            (
                "decisions_about",
                {"topic": "schema versioning"},
                ("list_assertion_claims", "query_units", "search"),
            ),
            (
                "unacknowledged_failures",
                {},
                ("query_units", "find_stuck_sessions", "list_marks"),
            ),
            (
                "sessions_touching_file",
                {"path": "polylogue/mcp/server.py"},
                ("query_units", "search", "get_session_summary"),
            ),
            (
                "cost_of",
                {},
                ("cost_rollups", "session_costs", "provider_usage"),
            ),
        ],
    )
    async def test_cookbook_prompts_render_tool_recipes(
        self: object,
        mcp_server: MCPServerUnderTest,
        prompt_name: str,
        kwargs: dict[str, object],
        expected_tools: tuple[str, ...],
    ) -> None:
        result = await invoke_surface_async(mcp_server._prompt_manager._prompts[prompt_name].fn, **kwargs)
        for tool in expected_tools:
            assert tool in result, f"{prompt_name} recipe must name tool {tool}"

    @pytest.mark.asyncio
    async def test_cookbook_prompts_prefill_repo_context(self: object, mcp_server: MCPServerUnderTest) -> None:
        prompts = mcp_server._prompt_manager._prompts
        cwd = Path.cwd()

        resume = await invoke_surface_async(prompts["resume_context"].fn)
        assert str(cwd) in resume
        assert cwd.name in resume
        assert "POLYLOGUE_ARCHIVE_ROOT" in resume
        assert "refs" in resume.lower()

        override = await invoke_surface_async(prompts["resume_context"].fn, repo="sinex")
        assert "'sinex'" in override

        touching = await invoke_surface_async(
            prompts["sessions_touching_file"].fn, path="polylogue/cli/click_app.py", repo="polylogue"
        )
        assert "path:polylogue/cli/click_app.py" in touching
        assert "repo:polylogue" in touching

        decisions = await invoke_surface_async(prompts["decisions_about"].fn, topic="lineage")
        assert 'near:"lineage"' in decisions
        assert "candidate" in decisions

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
                db_path=archive_root / "index.db",
            )
            mock_get_query_store.side_effect = AssertionError("compare prompt must not open archive query store")
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
                text="```python\nprint('archive index')\n```",
            )

        with (
            patch("polylogue.mcp.server._get_config") as mock_get_config,
            patch("polylogue.mcp.server._get_polylogue") as mock_get_query_store,
        ):
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            mock_get_query_store.side_effect = AssertionError("code prompt must not open archive query store")
            result = await invoke_surface_async(
                mcp_server._prompt_manager._prompts["extract_code"].fn,
                language="python",
            )

        assert "Found 1 code blocks" in result
        assert "archive index" in result

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
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"error number {i} occurred")],
                        )
                        for i in range(30)
                    ],
                )
            )

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
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
    async def test_extract_code_handles_plain_text_session(self: object, mcp_server: MCPServerUnderTest) -> None:
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
    def test_get_messages_tool_returns_full_archive_messages(
        self: object,
        mcp_server: MCPServerUnderTest,
        tmp_path: Path,
    ) -> None:
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
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=body)],
                        )
                    ],
                )
            )

        with patch("polylogue.mcp.server._get_config") as mock_get_config:
            mock_get_config.return_value = SimpleNamespace(
                archive_root=archive_root,
                db_path=archive_root / "index.db",
            )
            result = invoke_surface(
                mcp_server._tool_manager._tools["get_messages"].fn,
                session_id=session_id,
            )

        payload = json.loads(result)
        assert payload["messages"][0]["text"] == body
        assert payload["messages"][0]["content_blocks"][0]["text"] == body


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
