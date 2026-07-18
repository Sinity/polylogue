"""Unit tests for the privileged transaction tools (write/judge/run/maintenance, t46.8.3).

These are thin adapters over the same typed owners the retired per-operation
MCP tools used (write, run, maintenance) or already used (judge) -- see
``register_cutover_privileged_tools`` in ``polylogue/mcp/server_cutover.py``.
Each is verified against a real seeded archive via ``RuntimeServices``, not
mocks, matching the pattern established in ``test_envelope_contracts.py`` and
``test_contract_evidence.py`` (query/context/explain route through the cached
``_get_polylogue()`` facade, so a real runtime service scope is required).

``build_server()`` must be called *before* entering ``_installed_runtime_services``
-- it always resolves and installs its own default runtime services when not
given one explicitly, which would otherwise clobber the seeded ones.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from tests.infra.mcp import MCPServerUnderTest, invoke_surface_async


def _seed_archive(archive_root: Path) -> str:
    """Write one session with searchable text; returns its canonical id."""
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        return archive.write_parsed(
            ParsedSession(
                source_name=Provider.CHATGPT,
                provider_session_id="privileged-contract",
                title="Privileged tool contract probe",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text="needle privileged contract evidence",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="needle privileged contract evidence")],
                    )
                ],
            )
        )


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


class TestRoleGating:
    def test_read_role_has_no_privileged_tools(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(role="read"))
        tools = set(server._tool_manager._tools)
        assert tools.isdisjoint({"write", "judge", "run", "maintenance"})

    def test_write_role_has_write_and_run_but_not_judge_or_maintenance(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(role="write"))
        tools = set(server._tool_manager._tools)
        assert {"write", "run"} <= tools
        assert tools.isdisjoint({"judge", "maintenance"})

    def test_review_role_adds_judge(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(role="review"))
        tools = set(server._tool_manager._tools)
        assert {"write", "run", "judge"} <= tools
        assert "maintenance" not in tools

    def test_admin_role_has_all_four_privileged_tools(self) -> None:
        from polylogue.mcp.server import build_server

        server = cast(MCPServerUnderTest, build_server(role="admin"))
        tools = set(server._tool_manager._tools)
        assert {"write", "run", "judge", "maintenance"} <= tools


class TestWriteTool:
    @pytest.mark.asyncio
    async def test_add_tag_then_remove_tag_round_trips_against_real_archive(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            added = json.loads(
                await invoke_surface_async(write_fn, operation="add_tag", session_id=session_id, tag="reviewed")
            )
            assert added.get("is_error") is not True, added
            assert added["outcome"] == "added"

            removed = json.loads(
                await invoke_surface_async(write_fn, operation="remove_tag", session_id=session_id, tag="reviewed")
            )
            assert removed.get("is_error") is not True, removed
            assert removed["outcome"] == "removed"

    @pytest.mark.asyncio
    async def test_missing_required_argument_returns_invalid_argument_envelope(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="add_tag", tag="reviewed"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_operation_specific_field_is_read_from_fields_dict(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="add_mark",
                    session_id=session_id,
                    fields={"mark_type": "star"},
                )
            )
            assert result.get("is_error") is not True, result
            assert result["outcome"] == "added"

    @pytest.mark.asyncio
    async def test_add_mark_without_mark_type_field_returns_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="add_mark", session_id=session_id))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_invalid_argument_envelope(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="not_a_real_operation"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_delete_session_without_confirm_is_refused(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(write_fn, operation="delete_session", session_id=session_id))
            assert result.get("is_error") is True
            assert "confirm" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_save_and_delete_saved_view_round_trips(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn

        with _installed_runtime_services(archive_root):
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_saved_view",
                    fields={"name": "needle sessions", "query_json": json.dumps({"query": "needle"})},
                )
            )
            assert saved.get("is_error") is not True, saved
            view_id = saved["key"]

            deleted = json.loads(
                await invoke_surface_async(write_fn, operation="delete_saved_view", fields={"view_id": view_id})
            )
            assert deleted.get("is_error") is not True, deleted
            assert deleted["status"] == "deleted"


class TestJudgeTool:
    @pytest.mark.asyncio
    async def test_single_candidate_shorthand_builds_a_one_item_bulk_call(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock, patch

        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="review"))
        judge_fn = server._tool_manager._tools["judge"].fn

        with _installed_runtime_services(archive_root):
            with patch("polylogue.mcp.server._get_polylogue") as mock_get_polylogue:
                from polylogue.api import Polylogue
                from polylogue.surfaces.payloads import AssertionBulkJudgmentPayload

                real_poly = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
                real_poly.judge_assertion_candidates = AsyncMock(  # type: ignore[method-assign]
                    return_value=AssertionBulkJudgmentPayload(
                        items=(), applied_count=0, idempotent_count=0, failed_count=0
                    )
                )
                mock_get_polylogue.return_value = real_poly

                single = json.loads(
                    await invoke_surface_async(
                        judge_fn, candidate_ref="assertion:contract-candidate", decision="accept"
                    )
                )
                assert single.get("is_error") is not True, single
                real_poly.judge_assertion_candidates.assert_awaited_once()
                await_args = real_poly.judge_assertion_candidates.await_args
                assert await_args is not None
                items = await_args.kwargs["items"]
                assert len(items) == 1
                assert items[0].candidate_ref == "assertion:contract-candidate"
                assert items[0].decision == "accept"

    @pytest.mark.asyncio
    async def test_neither_items_nor_candidate_ref_returns_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="review"))
        judge_fn = server._tool_manager._tools["judge"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(judge_fn))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"


class TestRunTool:
    @pytest.mark.asyncio
    async def test_run_executes_a_saved_query_ref_and_returns_matching_sessions(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        write_fn = server._tool_manager._tools["write"].fn
        run_fn = server._tool_manager._tools["run"].fn

        with _installed_runtime_services(archive_root):
            saved = json.loads(
                await invoke_surface_async(
                    write_fn,
                    operation="save_saved_view",
                    fields={"name": "needle sessions", "query_json": json.dumps({"query": "needle"})},
                )
            )
            assert saved.get("is_error") is not True, saved
            view_id = saved["key"]

            result = json.loads(await invoke_surface_async(run_fn, ref=f"saved-query:{view_id}"))
            assert result.get("is_error") is not True, result
            assert "hits" in result or "items" in result

    @pytest.mark.asyncio
    async def test_unknown_saved_view_ref_returns_not_found(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        run_fn = server._tool_manager._tools["run"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(run_fn, ref="saved-query:does-not-exist"))
            assert result.get("is_error") is True
            assert result.get("code") == "not_found"

    @pytest.mark.asyncio
    async def test_non_saved_query_ref_kind_is_rejected(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="write"))
        run_fn = server._tool_manager._tools["run"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(run_fn, ref="session:not-a-saved-query"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"


class TestMaintenanceTool:
    @pytest.mark.asyncio
    async def test_list_returns_empty_envelope_on_a_fresh_archive(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="admin"))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="list"))
            assert result.get("is_error") is not True, result
            assert result["items"] == []
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_status_without_operation_id_returns_invalid_argument(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="admin"))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(maintenance_fn, operation="status"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_argument"

    @pytest.mark.asyncio
    async def test_status_for_missing_operation_id_returns_not_found(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="admin"))
        maintenance_fn = server._tool_manager._tools["maintenance"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(maintenance_fn, operation="status", operation_id="does-not-exist")
            )
            assert result.get("is_error") is True
            assert result.get("code") == "not_found"


class TestQuerySessionsProjection:
    @pytest.mark.asyncio
    async def test_ranked_search_finds_the_seeded_session(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        session_id = _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="read"))
        query_fn = server._tool_manager._tools["query"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(
                await invoke_surface_async(query_fn, expression="needle", projection="sessions", limit=10)
            )
            assert result.get("is_error") is not True, result
            assert "hits" in result
            assert result["total"] >= 1
            hit_session_ids = {hit["session"]["id"] for hit in result["hits"]}
            assert session_id in hit_session_ids

    @pytest.mark.asyncio
    async def test_exhaustive_listing_without_expression_returns_items(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="read"))
        query_fn = server._tool_manager._tools["query"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="sessions", limit=10))
            assert result.get("is_error") is not True, result
            assert "items" in result
            assert result["total"] >= 1

    @pytest.mark.asyncio
    async def test_sessions_projection_rejects_continuation(self, tmp_path: Path) -> None:
        from polylogue.mcp.server import build_server

        archive_root = tmp_path / "archive"
        _seed_archive(archive_root)
        server = cast(MCPServerUnderTest, build_server(role="read"))
        query_fn = server._tool_manager._tools["query"].fn

        with _installed_runtime_services(archive_root):
            result = json.loads(await invoke_surface_async(query_fn, projection="sessions", continuation="bogus"))
            assert result.get("is_error") is True
            assert result.get("code") == "invalid_continuation"
