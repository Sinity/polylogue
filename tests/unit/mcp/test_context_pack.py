"""Unit tests for the build_context_pack MCP tool and context-pack selection.

build_context_pack collapsed onto the shared ``compile_context`` engine: it is a
thin lens that selects sessions through the query algebra and returns the
``ContextImage`` payload. These tests cover the selection helper directly and the
tool's delegation/contract; ContextImage compilation itself is covered by the API
and compiler test suites.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Session
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.context.compiler import ContextImage, ContextOmission, ContextSegment, ContextSpec
from polylogue.core.enums import Provider
from polylogue.mcp.context_pack import select_context_pack_sessions
from tests.infra.builders import make_conv, make_msg
from tests.infra.mcp import (
    EXPECTED_TOOL_NAMES,
    MCPServerUnderTest,
    invoke_surface_async,
    make_polylogue_mock,
)


def _context_image(*, with_messages: bool = True) -> ContextImage:
    """Build a representative ContextImage as compile_context would emit."""
    segments: tuple[ContextSegment, ...] = ()
    if with_messages:
        segments = (
            ContextSegment(
                segment_id="read-view:codex-session:ctx-1:messages",
                kind="read_view",
                title="Messages",
                markdown="# Messages\n\nuser: needle\n",
                payload_kind="messages",
                token_estimate=2,
            ),
        )
    spec = ContextSpec(purpose="handoff", seed_refs=("session:codex-session:ctx-1",), read_views=("messages",))
    return ContextImage(
        spec=spec,
        segments=segments,
        evidence_refs=(),
        omitted=()
        if with_messages
        else (ContextOmission(ref="session:codex-session:ctx-1", view="messages", reason="budget", detail="over"),),
        token_estimate=2 if with_messages else 0,
    )


class TestBuildContextPackRegistration:
    """Verify the build_context_pack tool is registered and callable."""

    def test_tool_name_is_in_expected_set(self) -> None:
        assert "build_context_pack" in EXPECTED_TOOL_NAMES

    def test_tool_is_registered_on_server(self, mcp_server: MCPServerUnderTest) -> None:
        tools = mcp_server._tool_manager._tools
        assert "build_context_pack" in tools
        assert callable(tools["build_context_pack"].fn)


class TestContextPackSelection:
    """The recall-oriented query-algebra selection retained after the collapse."""

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
        assert ("supersedes_event_id", "event_replacements", "equivalence_key") in seen_queries
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


class TestBuildContextPackDelegation:
    """The tool delegates to context_pack_payload and serializes the ContextImage."""

    @pytest.mark.asyncio
    async def test_returns_context_image_payload(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.context_pack_payload = AsyncMock(return_value=_context_image())
            mock_get_polylogue.return_value = mock_poly

            raw = await invoke_surface_async(
                mcp_server._tool_manager._tools["build_context_pack"].fn,
                query="needle",
                max_sessions=1,
            )

        payload = json.loads(raw)
        assert "segments" in payload
        assert payload["segments"][0]["payload_kind"] == "messages"
        assert payload["token_estimate"] == 2
        # Delegated to the shared engine with messages included by default.
        kwargs = mock_poly.context_pack_payload.call_args.kwargs
        assert kwargs["query"] == "needle"
        assert kwargs["include_messages"] is True
        assert kwargs["redact_paths"] is True

    @pytest.mark.asyncio
    async def test_summary_detail_omits_messages(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.context_pack_payload = AsyncMock(return_value=_context_image(with_messages=False))
            mock_get_polylogue.return_value = mock_poly

            await invoke_surface_async(
                mcp_server._tool_manager._tools["build_context_pack"].fn,
                max_sessions=1,
                detail_level="summary",
            )

        kwargs = mock_poly.context_pack_payload.call_args.kwargs
        assert kwargs["include_messages"] is False

    @pytest.mark.asyncio
    async def test_redact_paths_passthrough(self, mcp_server: MCPServerUnderTest) -> None:
        with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
            mock_poly = make_polylogue_mock()
            mock_poly.context_pack_payload = AsyncMock(return_value=_context_image())
            mock_get_polylogue.return_value = mock_poly

            await invoke_surface_async(
                mcp_server._tool_manager._tools["build_context_pack"].fn,
                max_sessions=1,
                redact_paths=False,
            )

        assert mock_poly.context_pack_payload.call_args.kwargs["redact_paths"] is False
