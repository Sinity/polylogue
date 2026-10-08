from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from polylogue.analysis.archive import ThreadInsightQuery
from polylogue.api import Polylogue
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_archive_fixture_write
from tests.infra.mcp import MCPServerUnderTest, installed_runtime_services, invoke_surface_async
from tests.infra.storage_records import seed_thread_search_archive


@pytest.mark.asyncio
async def test_public_thread_search_filters_support_before_api_and_mcp_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = await run_archive_fixture_write(tmp_path, lambda: seed_thread_search_archive(tmp_path))
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        assert (
            archive._conn.execute("SELECT 1 FROM session_profiles WHERE session_id = ?", (ids["newer"],)).fetchone()
            is None
        )
        assert (
            archive._conn.execute("SELECT 1 FROM session_profiles WHERE session_id = ?", (ids["older"],)).fetchone()
            is not None
        )
        newer = archive.get_thread_insight(ids["newer"])
        assert newer is not None
        assert newer.thread.session_ids == (ids["newer"], ids["child"])
        assert newer.thread.support_level == "strong"
        terms = {
            "strong": [ids["newer"]],
            "moderate": [ids["older"]],
            "explicit_lineage": [ids["newer"]],
            "parent_session_id": [ids["newer"]],
            "archive_threads": [ids["newer"], ids["older"]],
            "archive_thread_sessions": [ids["newer"], ids["older"]],
            "old-repo": [ids["older"]],
            "old-branch": [ids["older"]],
            "older lookup": [ids["older"]],
            "absent-term": [],
        }
        for term, expected in terms.items():
            assert [row.thread_id for row in archive.list_thread_insights(query=term, limit=None)] == expected

    async with Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db") as poly:
        with installed_runtime_services(tmp_path):
            from polylogue.mcp.server import build_server
            from polylogue.services import RuntimeServices

            server = cast(MCPServerUnderTest, build_server(services=RuntimeServices(config=poly.config)))
            tool = server._tool_manager._tools["query"].fn
            for term, expected in terms.items():
                api_rows = await poly.list_thread_insights(ThreadInsightQuery(query=term, limit=1))
                result = json.loads(
                    cast(
                        str, await invoke_surface_async(tool, expression=term, projection="threads", limit=1, offset=0)
                    )
                )
                assert [row.thread_id for row in api_rows] == expected[:1]
                assert "threads" in result, result
                assert [row["thread_id"] for row in result["threads"]] == expected[:1]
            for offset, expected in enumerate(([ids["newer"]], [ids["older"]], [])):
                api_rows = await poly.list_thread_insights(ThreadInsightQuery(limit=1, offset=offset))
                result = json.loads(
                    cast(
                        str,
                        await invoke_surface_async(
                            tool, expression="archive_threads", projection="threads", limit=1, offset=offset
                        ),
                    )
                )
                assert [row.thread_id for row in api_rows] == expected
                assert [row["thread_id"] for row in result["threads"]] == expected

            # Force the original transport boundary after real registry selection.
            # The direct continuation must retain the search and page position.
            from polylogue.mcp import server_support

            full_page = cast(
                str,
                await invoke_surface_async(tool, expression="archive_threads", projection="threads", limit=2, offset=0),
            )
            monkeypatch.setattr(server_support, "MCP_RESPONSE_ENVELOPE_HEADROOM_BYTES", 0)
            monkeypatch.setattr(server_support, "MCP_RESPONSE_BUDGET_BYTES", len(full_page.encode("utf-8")) - 1)
            narrowed = json.loads(
                cast(
                    str,
                    await invoke_surface_async(
                        tool, expression="archive_threads", projection="threads", limit=2, offset=0
                    ),
                )
            )
            assert narrowed["status"] == "response_budget_exceeded"
            arguments = narrowed["continuation"]["arguments"]
            assert arguments["expression"] == "archive_threads"
            assert arguments["projection"] == "threads"
            assert narrowed["returned_items"] == 1
            assert arguments["limit"] == 2
            assert arguments["offset"] == 1
