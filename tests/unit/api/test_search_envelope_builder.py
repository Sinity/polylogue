"""Live cursor controls use the production query, SQLite, and API route."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.api import Polylogue
from polylogue.api.search_envelope_builder import build_search_envelope_for_spec
from polylogue.archive.query.search_cursor import InvalidSearchCursorError, decode_search_cursor
from polylogue.archive.query.spec import SessionQuerySpec
from tests.infra.storage_records import SessionBuilder


def _add(root: Path, name: str, order: int, *, actions: bool = False, blocks: int = 1) -> str:
    builder = SessionBuilder(root / "index.db", name).provider("codex").title(name)
    for i in range(blocks):
        content = (
            [{"type": "tool_use", "tool_name": "shell", "tool_id": f"call-{i}", "input": {"command": "needle"}}]
            if actions
            else None
        )
        builder.add_message(f"m-{i}", role="assistant" if actions else "user", text="needle", blocks=content)
    builder.save()
    sid = builder.native_session_id()
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("UPDATE sessions SET updated_at_ms = ? WHERE session_id = ?", (order, sid))
    return sid


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["auto", "actions"])
@pytest.mark.parametrize("reverse", [False, True])
async def test_present_anchor_relocated_after_arbitrary_growth(
    workspace_env: dict[str, Path], lane: str, reverse: bool
) -> None:
    root = workspace_env["archive_root"]
    for i in range(1, 5):
        _add(root, f"old-{i}", i, actions=lane == "actions")
    spec = SessionQuerySpec(query_terms=("needle",), retrieval_lane=lane, sort="date", reverse=reverse, limit=1)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert first.hits[0].session.title == ("old-1" if reverse else "old-4")
        assert first.next_cursor is not None
        # Well beyond the predecessor's two-page fetch window.
        for i in range(1, 12):
            _add(root, f"new-{i}", -i if reverse else 100 + i, actions=lane == "actions")
        second = await build_search_envelope_for_spec(facade, replace(spec, cursor=first.next_cursor))
    assert second.hits[0].session.title == ("old-2" if reverse else "old-3")
    assert second.offset == 12
    assert second.next_cursor is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_removed_anchor_uses_original_complete_key(workspace_env: dict[str, Path], reverse: bool) -> None:
    root = workspace_env["archive_root"]
    ids = [_add(root, f"old-{i}", i) for i in range(1, 5)]
    spec = SessionQuerySpec(query_terms=("needle",), sort="date", reverse=reverse, limit=1)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert first.next_cursor
        # Remove the anchor from the qualifying relation, not from archive custody.
        with sqlite3.connect(root / "index.db") as conn:
            conn.execute(
                "UPDATE sessions SET origin = 'claude-code-session' WHERE session_id = ?",
                (ids[0] if reverse else ids[-1],),
            )
        filtered = replace(spec, origins=("codex-session",))
        # Start under the filter so the token binds the same logical relation.
        first = await build_search_envelope_for_spec(facade, filtered)
        assert first.next_cursor
        anchor = first.hits[0].session.id
        with sqlite3.connect(root / "index.db") as conn:
            conn.execute("UPDATE sessions SET origin = 'claude-code-session' WHERE session_id = ?", (anchor,))
        for i in range(5):
            _add(root, f"ahead-{i}", -100 - i if reverse else 100 + i)
        second = await build_search_envelope_for_spec(facade, replace(filtered, cursor=first.next_cursor))
    assert second.hits[0].session.title == ("old-3" if reverse else "old-2")


@pytest.mark.asyncio
async def test_block_grain_lookahead_ignores_session_total(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    _add(root, "many", 1, blocks=4)
    spec = SessionQuerySpec(query_terms=("needle",), sort="date", limit=1)
    seen: list[str | None] = []
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        while True:
            page = await build_search_envelope_for_spec(facade, spec)
            assert page.total == 1
            seen.extend(hit.match.block_id for hit in page.hits)
            if page.next_cursor is None:
                assert page.next_offset is None
                break
            cursor = decode_search_cursor(page.next_cursor)
            assert cursor.position.grain == "block"
            spec = replace(spec, cursor=page.next_cursor)
    assert len(seen) == len(set(seen)) == 4


@pytest.mark.asyncio
async def test_structural_fallback_owned_by_producer(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    for i in range(3):
        _add(root, f"session-{i}", i)
    spec = SessionQuerySpec(origins=("codex-session",), sort="date", limit=1)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert first.next_cursor
        assert decode_search_cursor(first.next_cursor).position.grain == "session"
        assert first.hits[0].match.block_id is None
        assert first.hits[0].match.score is None and first.hits[0].match.score_kind is None
        for i in range(8):
            _add(root, f"new-{i}", 100 + i)
        second = await build_search_envelope_for_spec(facade, replace(spec, cursor=first.next_cursor))
    assert second.hits[0].session.title == "session-1"
    assert second.hits[0].match.match_surface == "session"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        lambda spec: replace(spec, sort="words"),
        lambda spec: replace(spec, reverse=True),
        lambda spec: replace(spec, origins=("claude-code-session",)),
        lambda spec: replace(spec, query_terms=("other",)),
        lambda spec: replace(spec, retrieval_lane="actions"),
    ],
)
async def test_cursor_rejects_selection_lane_or_order_change(
    workspace_env: dict[str, Path], change: Callable[[SessionQuerySpec], SessionQuerySpec]
) -> None:
    root = workspace_env["archive_root"]
    for i in range(3):
        _add(root, str(i), i)
    spec = SessionQuerySpec(query_terms=("needle",), sort="date", limit=1)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
        assert first.next_cursor
        with pytest.raises(InvalidSearchCursorError):
            await build_search_envelope_for_spec(facade, replace(change(spec), cursor=first.next_cursor))


@pytest.mark.asyncio
async def test_random_has_no_cursor(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    for i in range(3):
        _add(root, str(i), i)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        result = await build_search_envelope_for_spec(
            facade, SessionQuerySpec(query_terms=("needle",), sort="random", limit=1)
        )
    assert len(result.hits) == 1
    assert result.next_cursor is None
    assert result.next_offset == 1


@pytest.mark.asyncio
async def test_surface_adapters_share_producer_continuation(workspace_env: dict[str, Path]) -> None:
    import json

    from polylogue.archive.query.archive_execution import archive_search_hits
    from polylogue.archive.query.search_hits import project_search_hits
    from polylogue.cli.query_output import format_search_envelope
    from polylogue.mcp.archive_support import archive_search_payload
    from polylogue.operations.daemon_reads import execute_read_operation
    from polylogue.operations.operation_context import open_operation_read

    root = workspace_env["archive_root"]
    for i in range(4):
        _add(root, f"old-{i}", i)
    spec = SessionQuerySpec(query_terms=("needle",), sort="date", limit=1)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        first = await build_search_envelope_for_spec(facade, spec)
    assert first.next_cursor
    for i in range(7):
        _add(root, f"new-{i}", 100 + i)
    spec = replace(spec, cursor=first.next_cursor)
    with open_operation_read(root) as pinned:
        result = archive_search_hits(spec.to_plan(), archive_root=root, config=None, archive=pinned.archive)
        projected = project_search_hits(spec.to_plan(), result)
        cli = json.loads(
            format_search_envelope(
                projected, query="needle", retrieval_lane="auto", limit=1, offset=0, sort="date", total=11
            )
        )
        mcp = archive_search_payload(
            pinned.archive, spec, query="needle", limit=1, offset=0, sort="date", archive_root=root
        ).model_dump(mode="json")
        daemon = execute_read_operation(
            "cli.query",
            {"params": {"query": ("needle",), "sort": "date", "limit": 1, "cursor": first.next_cursor}},
            archive=pinned.archive,
            serving_identity="daemon",
        )
    for raw_page in (cli, mcp, daemon):
        page = cast(dict[str, Any], raw_page)
        assert page["hits"][0]["session"]["title"] == "old-2"
        assert page["offset"] == 8
        assert page["next_offset"] == 9
        assert page["next_cursor"] == result.execution.next_cursor


def test_callback_reads_keep_offsets_without_minting_cursor(workspace_env: dict[str, Path]) -> None:
    from polylogue.archive.query.archive_execution import archive_search_hits
    from polylogue.archive.query.plan import SessionQueryPlan

    root = workspace_env["archive_root"]
    for i in range(3):
        _add(root, f"session-{i}", i)
    plan = SessionQueryPlan(sort="date", limit=1, predicates=(lambda session: True,))
    first = archive_search_hits(plan, archive_root=root, config=None)
    assert first.hits
    assert first.execution.has_more
    assert first.execution.next_cursor is None
    with pytest.raises(InvalidSearchCursorError):
        archive_search_hits(replace(plan, cursor="opaque"), archive_root=root, config=None)


@pytest.mark.asyncio
async def test_empty_query_and_latest_preserve_cursor_scope(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    for i in range(3):
        _add(root, str(i), i)
    async with Polylogue(archive_root=root, db_path=root / "index.db") as facade:
        empty = await build_search_envelope_for_spec(facade, SessionQuerySpec(limit=1))
        latest = await build_search_envelope_for_spec(
            facade, SessionQuerySpec(query_terms=("needle",), latest=True, limit=1)
        )
    assert not empty.hits and empty.next_cursor is None
    assert len(latest.hits) == 1
    assert latest.next_cursor is None and latest.next_offset is None
