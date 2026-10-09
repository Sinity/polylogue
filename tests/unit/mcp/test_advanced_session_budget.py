"""Advanced session pages retain their answering frame through byte paging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import run_off_event_loop
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import build_tools, installed_runtime_services, invoke_surface_async


def _seed(root: Path, count: int = 100) -> set[str]:
    with ArchiveStore(root) as archive:
        return {
            write_index_session(
                archive,
                ParsedSession(
                    source_name=Provider.CODEX,
                    provider_session_id=f"budget-neutral-{index}",
                    title=f"Neutral {index}",
                    messages=[
                        ParsedMessage(
                            provider_message_id=f"neutral-{index}",
                            role=Role.USER,
                            blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"needle neutral {index}")],
                        )
                    ],
                ),
            )
            for index in range(count)
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", [None, "needle", "-secret"])
async def test_advanced_query_budget_continuations_walk_every_row_once(
    workspace_env: dict[str, Path], expression: str | None
) -> None:
    """Dropping the frame or resuming through sessions.list loses the 100-row walk."""
    root = workspace_env["archive_root"]
    expected = run_off_event_loop(lambda: _seed(root))
    with installed_runtime_services(root):
        query = build_tools()["query"]
        arguments: dict[str, object] = {
            "projection": "sessions",
            "origin": "codex-session,claude-code-session",
            "limit": 100,
            "expression": expression,
        }
        seen: list[str] = []
        budget_pages = 0
        while True:
            body = json.loads(await invoke_surface_async(query, **arguments))
            assert body.get("status") != "error", body
            page = body.get("page", body)
            if "hits" in page:
                seen.extend(hit["session"]["id"] for hit in page["hits"])
                if body.get("status") == "response_budget_exceeded":
                    assert page["next_cursor"] is None
            else:
                seen.extend(item["id"] for item in page["items"])
            if body.get("status") != "response_budget_exceeded":
                assert page["next_offset"] is None
                break
            budget_pages += 1
            resume = body["continuation"]
            assert resume is not None, body
            assert resume["tool"] == "query"
            assert resume["arguments"]["projection"] == "sessions"
            arguments = resume["arguments"]
        assert budget_pages > 0
        assert set(seen) == expected
        assert len(seen) == len(expected)


@pytest.mark.asyncio
async def test_advanced_budget_continuation_refuses_changed_selection_frame(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    run_off_event_loop(lambda: _seed(root))
    with installed_runtime_services(root):
        query = build_tools()["query"]
        first = json.loads(
            await invoke_surface_async(
                query, projection="sessions", origin="codex-session,claude-code-session", limit=100
            )
        )
        resume = first["continuation"]
        for conflicting in ({"repo": "other"}, {"limit": 1000}, {"offset": 0}):
            refused = json.loads(await invoke_surface_async(query, **{**resume["arguments"], **conflicting}))
            assert refused["code"] == "invalid_continuation"
        run_off_event_loop(lambda: _seed(root, count=101))
        stale = json.loads(await invoke_surface_async(query, **resume["arguments"]))
        assert stale["code"] == "query_continuation_stale"
