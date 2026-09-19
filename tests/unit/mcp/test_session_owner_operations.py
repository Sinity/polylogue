"""Owner contracts exercise real archive reads and synthetic original files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from jsonschema import validate

from polylogue import Polylogue
from polylogue.archive.query.transaction import QueryContinuationInvalidError, QueryContinuationStaleError
from polylogue.core.enums import BlockType, Origin, Provider, Role
from polylogue.mcp.server import build_server
from polylogue.operations.raw_sessions.sessions import SessionError, SessionSource
from polylogue.operations.session_contracts import (
    SESSION_OPERATION_ADAPTER,
    RawList,
    RawMemorySearch,
    RawRead,
    RawSearch,
    RawTimeline,
    SessionList,
    SessionOperation,
    SessionOperationError,
    SessionRead,
    SessionSearch,
    SessionTimeline,
    session_operation_contracts,
)
from polylogue.operations.session_reads import execute_session_operation, raw_operation
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession, ParsedSessionEvent
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.frozen_clock import FrozenClock
from tests.infra.live_ingest import write_index_session
from tests.infra.mcp import invoke_surface_async


def seed(root: Path, count: int = 3) -> list[str]:
    ids = []
    with ArchiveStore(root) as archive:
        for index in range(count):
            ids.append(
                write_index_session(
                    archive,
                    ParsedSession(
                        source_name=Provider.CODEX,
                        provider_session_id=f"session-{index}",
                        title=f"Session {index}",
                        messages=[
                            ParsedMessage(
                                provider_message_id=f"m{m}",
                                role=Role.USER,
                                timestamp=f"2026-01-0{index + 1}T12:0{m}:00Z" if m == 0 else None,
                                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"needle {index} {m}")],
                            )
                            for m in range(2)
                        ],
                        session_events=[
                            ParsedSessionEvent(
                                event_type="turn_context",
                                timestamp=f"2026-01-0{index + 1}T13:00:00Z",
                                payload={"model": "synthetic"},
                            )
                        ],
                    ),
                )
            )
    return ids


@pytest.mark.asyncio
async def test_session_pages_resume_without_rebuilding_filters_and_reject_changed_scope(tmp_path: Path) -> None:
    """Ignoring continuation repeats page one; dropping origin changes its scope."""
    root = tmp_path / "archive"
    ids = seed(root)
    async with Polylogue(archive_root=root) as api:
        request = SessionList(origin=Origin.CODEX_SESSION, limit=1)
        first = await execute_session_operation(api, request)
        result = first
        seen: list[str] = []
        while True:
            seen.extend(item.id for item in result.items)
            if result.continuation is None:
                break
            result = await execute_session_operation(api, SessionList(continuation=result.continuation))
        assert set(seen) == set(ids)
        assert len(seen) == len(ids)
        with pytest.raises(QueryContinuationInvalidError):
            await execute_session_operation(
                api, SessionList(continuation=first.continuation, origin=Origin.CHATGPT_EXPORT)
            )
        with pytest.raises(QueryContinuationInvalidError):
            await execute_session_operation(api, SessionSearch(continuation=first.continuation))
        seed(root, count=4)
        with pytest.raises(QueryContinuationStaleError):
            await execute_session_operation(api, SessionList(continuation=first.continuation))


@pytest.mark.asyncio
async def test_search_read_timeline_and_mcp_use_owner(tmp_path: Path) -> None:
    """Owner dispatch must preserve timeline order, missing timestamps, and page advancement."""
    root = tmp_path / "archive"
    ids = seed(root)
    async with Polylogue(archive_root=root) as api:
        search = await execute_session_operation(api, SessionSearch(expression="needle", limit=1))
        assert search.items
        next_search = await execute_session_operation(api, SessionSearch(continuation=search.continuation))
        assert next_search.items[0].session.id != search.items[0].session.id
        read = await execute_session_operation(api, SessionRead(ref=f"session:{ids[0]}", limit=1))
        again = await execute_session_operation(
            api, SessionRead(ref=f"session:{ids[0]}", continuation=read.continuation)
        )
        assert again.offset == 1
        timeline = await execute_session_operation(api, SessionTimeline(limit=2))
        assert timeline.coverage.time_basis == "event-timestamp"
        assert timeline.outcome == "degraded"
        assert timeline.coverage.gaps
        assert timeline.items[0].kind == "session-event"
        assert timeline.items[0].timestamp_ms > timeline.items[1].timestamp_ms
        timeline_next = await execute_session_operation(api, SessionTimeline(continuation=timeline.continuation))
        assert {row.reference for row in timeline.items}.isdisjoint(row.reference for row in timeline_next.items)
        with patch("polylogue.mcp.server._get_polylogue", return_value=api):
            server = build_server()
            query = server._tool_manager._tools["query"].fn
            first = json.loads(await invoke_surface_async(query, projection="sessions", limit=1))
            second = json.loads(
                await invoke_surface_async(query, projection="sessions", continuation=first["continuation"])
            )
            assert first["items"][0]["id"] != second["items"][0]["id"]
            exact = json.loads(
                await invoke_surface_async(
                    query,
                    projection="session-operations",
                    session_operation={"operation": "sessions.timeline", "limit": 2},
                )
            )
            from polylogue.archive.query.transaction import QueryContinuation

            token = exact.pop("continuation")
            assert exact == timeline.model_dump(mode="json", exclude={"continuation"})
            assert timeline.continuation is not None
            assert QueryContinuation.decode(token).request == QueryContinuation.decode(timeline.continuation).request
            missing = json.loads(await invoke_surface_async(query, projection="session-operations"))
            assert missing["code"] == "invalid_argument"


def raw_sources(tmp_path: Path) -> tuple[SessionSource, ...]:
    root = tmp_path / "raw"
    root.mkdir()
    for index in range(3):
        path = root / f"original-{index}.jsonl"
        path.write_text(json.dumps({"text": f"needle {index} café"}) + "\n")
        os.utime(path, ns=(1_000_000_000 + index, 1_000_000_000 + index))
    return (SessionSource("codex", root), SessionSource("claude-code", tmp_path / "missing"))


def test_raw_fallback_preserves_original_identity_and_file_clock(tmp_path: Path) -> None:
    """The fallback must never mint an indexed session id or label mtime as event time."""
    sources = raw_sources(tmp_path)
    first = raw_operation(RawList(origin="codex-session", limit=1), sources=sources)
    assert first.items[0].reference == "codex:original-2.jsonl"
    assert first.items[0].indexed_session_id is None
    assert first.coverage.time_basis == "session-file-mtime"
    assert first.continuation
    second = raw_operation(RawList(origin="codex-session", limit=1, continuation=first.continuation), sources=sources)
    assert second.items[0].reference == "codex:original-1.jsonl"
    content = raw_operation(RawRead(reference=first.items[0].reference, max_bytes=8), sources=sources)
    assert content.next_offset == 8
    assert content.mtime_ns == first.items[0].mtime_ns
    assert content.coverage.time_basis == "none"
    with pytest.raises(SessionError):
        raw_operation(RawRead(reference="codex:../escape.jsonl"), sources=sources)
    source = sources[0].root / "original-0.jsonl"
    source.write_text(source.read_text() + "changed\n")
    with pytest.raises(SessionError, match="changed"):
        raw_operation(RawList(origin="codex-session", limit=1, continuation=first.continuation), sources=sources)


def test_raw_scan_budget_advances_and_fanout_reports_unavailable(tmp_path: Path) -> None:
    """A limited scan is not an empty scope; continuation reaches later byte matches."""
    sources = raw_sources(tmp_path)
    result = raw_operation(RawSearch(origin="codex-session", query="needle", scan_bytes=4), sources=sources)
    assert result.coverage.scanned_bytes <= 4
    assert not result.coverage.complete
    matches = list(result.items)
    for _ in range(100):
        if result.continuation is None:
            break
        result = raw_operation(
            RawSearch(origin="codex-session", query="needle", scan_bytes=4, continuation=result.continuation),
            sources=sources,
        )
        matches.extend(result.items)
    assert len(matches) == 3
    memory = raw_operation(RawMemorySearch(query="needle"), sources=sources)
    assert memory.outcome == "degraded"
    assert memory.sources[0].availability == "unavailable"
    assert len(memory.items) == 3
    timeline = raw_operation(RawTimeline(), sources=sources)
    assert timeline.coverage.time_basis == "session-file-mtime"
    assert timeline.outcome == "degraded"
    assert [row.mtime_ns for row in timeline.items] == sorted([row.mtime_ns for row in timeline.items], reverse=True)


def test_operation_contracts_validate_real_results(tmp_path: Path) -> None:
    """Generating declarations from unrelated tool signatures loses per-operation bounds."""
    contracts = session_operation_contracts()["operations"]
    assert set(contracts) == {
        "sessions.list",
        "sessions.search",
        "sessions.read",
        "sessions.orchestration",
        "context.resume",
        "sessions.timeline",
        "sessions.raw.list",
        "sessions.raw.search",
        "sessions.raw.read",
        "sessions.raw.timeline",
        "memory.raw.search",
        "memory.raw.get",
    }
    request = RawList(origin="codex-session", limit=1)
    result = raw_operation(request, sources=raw_sources(tmp_path))
    validate(request.model_dump(mode="json"), contracts[request.operation]["request_schema"])
    validate(result.model_dump(mode="json"), contracts[request.operation]["result_schema"])
    with pytest.raises(ValueError):
        SESSION_OPERATION_ADAPTER.validate_python(
            {"operation": "sessions.raw.search", "origin": "codex-session", "query": "x", "scan_bytes": 8_388_609}
        )
    with pytest.raises(ValueError):
        SESSION_OPERATION_ADAPTER.validate_python(
            {"operation": "sessions.raw.read", "reference": "codex:a.jsonl", "root": "/arbitrary"}
        )


def test_raw_fanout_continuation_keeps_pending_source_and_global_timeline_order(tmp_path: Path) -> None:
    """Finishing the first provider must not mark pending providers covered."""
    roots = [tmp_path / name for name in ("claude", "codex")]
    sources = tuple(
        SessionSource(provider, root) for provider, root in zip(("claude-code", "codex"), roots, strict=True)
    )
    for index, root in enumerate(roots):
        root.mkdir()
        for file_index in range(2):
            path = root / f"{file_index}.jsonl"
            path.write_text('{"text":"needle"}\n')
            stamp = (index * 2 + file_index + 1) * 1_000_000_000
            os.utime(path, ns=(stamp, stamp))
    request = RawMemorySearch(query="needle", limit=1)
    references: list[str] = []
    result = raw_operation(request, sources=sources)
    for _ in range(10):
        references.extend(row.reference for row in result.items)
        if result.coverage.complete:
            break
        result = raw_operation(request.model_copy(update={"source_cursors": result.source_cursors}), sources=sources)
    assert len(references) == len(set(references)) == 4
    timeline = raw_operation(RawTimeline(limit=1), sources=sources)
    stamps: list[int] = []
    for _ in range(10):
        stamps.extend(row.mtime_ns for row in timeline.items)
        if timeline.continuation is None:
            break
        timeline = raw_operation(RawTimeline(limit=1, continuation=timeline.continuation), sources=sources)
    assert stamps == [4_000_000_000, 3_000_000_000, 2_000_000_000, 1_000_000_000]


def test_raw_timeline_observes_each_provider_once_per_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = [tmp_path / name for name in ("claude", "codex")]
    sources = tuple(
        SessionSource(provider, root) for provider, root in zip(("claude-code", "codex"), roots, strict=True)
    )
    for index, root in enumerate(roots):
        root.mkdir()
        for file_index in range(3):
            path = root / f"{file_index}.jsonl"
            path.write_text("{}\n")
            stamp = (index * 3 + file_index + 1) * 1_000_000_000
            os.utime(path, ns=(stamp, stamp))

    from polylogue.operations.raw_sessions.sessions import SessionLogService

    observed: list[str] = []
    original = SessionLogService._files

    def record(_self: SessionLogService, source: SessionSource) -> list[tuple[Path, os.stat_result]]:
        observed.append(source.provider)
        return original(source)

    monkeypatch.setattr(SessionLogService, "_files", record)

    result = raw_operation(RawTimeline(limit=5), sources=sources)

    assert result.items
    assert observed == ["claude-code", "codex"]


def test_filtered_raw_timeline_waits_for_a_sparse_source_frontier(tmp_path: Path) -> None:
    newer_root = tmp_path / "claude"
    older_root = tmp_path / "codex"
    newer_root.mkdir()
    older_root.mkdir()
    newer = newer_root / "newer.jsonl"
    older = older_root / "older.jsonl"
    newer.write_text("x" * 24 + " needle\n")
    older.write_text("needle\n")
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    sources = (SessionSource("claude-code", newer_root), SessionSource("codex", older_root))

    request = RawTimeline(limit=1, query="needle", scan_bytes=8)
    page = raw_operation(request, sources=sources)
    assert page.items == []
    assert page.continuation is not None
    assert page.coverage.complete is False

    for _ in range(64):
        if page.items:
            break
        page = raw_operation(request.model_copy(update={"continuation": page.continuation}), sources=sources)

    assert page.items
    assert [row.reference for row in page.items] == ["claude-code:newer.jsonl"]


@pytest.mark.uses_real_clock("Cancellation must interrupt a running SQLite statement through the owner transaction.")
@pytest.mark.asyncio
async def test_session_owner_cancellation_drains_controlled_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bypassing QueryTransaction leaves the expensive statement running after disconnect."""
    import asyncio
    import threading

    from polylogue.archive.query.transaction import QueryTransaction

    root = tmp_path / "archive"
    seed(root, count=1)
    started = threading.Event()
    contexts: list[Any] = []
    original_init = QueryTransaction.__init__

    def capture(self: QueryTransaction, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        contexts.append(self.context)

    def expensive(store: ArchiveStore, **kwargs: object) -> list[object]:
        started.set()
        store._conn.execute(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<2000000000) SELECT count(*) FROM n"
        ).fetchone()
        return []

    monkeypatch.setattr(QueryTransaction, "__init__", capture)
    monkeypatch.setattr(ArchiveStore, "list_summaries", expensive)
    async with Polylogue(archive_root=root) as api:
        task = asyncio.create_task(execute_session_operation(api, SessionList()))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    assert contexts[0].cancelled
    assert contexts[0].receipt.cleanup_complete


@pytest.mark.asyncio
async def test_all_indexed_operation_outputs_match_their_generated_contracts(tmp_path: Path) -> None:
    """Adapters cannot substitute one tool-wide envelope for distinct operation outputs."""
    from polylogue.operations.session_contracts import ResumeContext, SessionOrchestration
    from polylogue.operations.session_reads import session_operation_response

    root = tmp_path / "archive"
    ids = seed(root, count=1)
    contracts = session_operation_contracts()["operations"]
    requests: list[SessionOperation] = [
        SessionList(),
        SessionSearch(expression="needle"),
        SessionRead(ref=ids[0]),
        SessionTimeline(),
        SessionOrchestration(ref=ids[0]),
        ResumeContext(),
    ]
    async with Polylogue(archive_root=root) as api:
        for request in requests:
            result = await execute_session_operation(api, request)
            validate(result.model_dump(mode="json"), contracts[request.operation]["result_schema"])
        error = await session_operation_response(api, SessionOrchestration(ref="codex-session:missing"))
        validate(error.model_dump(mode="json"), contracts["sessions.orchestration"]["error_schema"])
        assert isinstance(error, SessionOperationError)
        assert error.outcome == "error"


@pytest.mark.asyncio
async def test_timeline_filters_complete_event_text_before_bounding_excerpt(tmp_path: Path) -> None:
    """Filtering the display excerpt loses matching events with later evidence."""
    root = tmp_path / "archive"
    with ArchiveStore(root) as archive:
        write_index_session(
            archive,
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="long-event",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        timestamp="2026-01-01T12:00:00Z",
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="x" * 3000 + "needle")],
                    )
                ],
            ),
        )
    async with Polylogue(archive_root=root) as api:
        result = await execute_session_operation(api, SessionTimeline(expression="needle"))
    assert result.total == 1
    assert len(result.items[0].text) == 2000


@pytest.mark.frozen_clock_modules("polylogue.core.dates")
@pytest.mark.asyncio
async def test_session_continuation_freezes_relative_date_scope(tmp_path: Path, frozen_clock: FrozenClock) -> None:
    """Reparsing a relative date on later pages silently drops the oldest session."""
    from datetime import UTC, datetime

    root = tmp_path / "archive"
    ids = seed(root)
    frozen_clock.set_time(datetime(2026, 1, 4, 11, 59, tzinfo=UTC).timestamp())
    async with Polylogue(archive_root=root) as api:
        result = await execute_session_operation(api, SessionList(since="3d", limit=1))
        assert result.total == 3
        found = [result.items[0].id]
        frozen_clock.advance(120)
        while result.continuation:
            result = await execute_session_operation(api, SessionList(continuation=result.continuation))
            found.extend(row.id for row in result.items)
    assert set(found) == set(ids)


@pytest.mark.parametrize("operation", ["search", "memory", "list", "timeline"])
def test_raw_owner_rejects_source_change_between_scan_and_emission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """A first-page result cannot combine an old snippet with a newer file observation."""
    from polylogue.operations.raw_sessions.sessions import SessionLogService

    sources = raw_sources(tmp_path)
    from polylogue.operations.raw_sessions.timeline import TimelineService

    owner = TimelineService if operation == "timeline" else SessionLogService
    method_name = "query" if operation == "timeline" else "search" if operation in {"search", "memory"} else "timeline"
    original = getattr(owner, method_name)

    def change_after_scan(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result: dict[str, Any] = original(self, *args, **kwargs)
        path = sources[0].root / "original-2.jsonl"
        path.write_text(path.read_text() + "changed after scan\n")
        return result

    monkeypatch.setattr(owner, method_name, change_after_scan)
    requests: dict[str, RawSearch | RawMemorySearch | RawList | RawTimeline] = {
        "search": RawSearch(origin="codex-session", query="needle"),
        "memory": RawMemorySearch(origins=["codex-session"], query="needle"),
        "list": RawList(origin="codex-session"),
        "timeline": RawTimeline(origins=["codex-session"]),
    }
    with pytest.raises(SessionError, match="changed"):
        raw_operation(requests[operation], sources=sources)


@pytest.mark.asyncio
async def test_session_contract_discovery_passes_mcp_argument_validation() -> None:
    """Discovery must accept the operation subject through the registered MCP transport."""
    result = await build_server().call_tool("explain", {"subject": "session-operations", "expression": "sessions.list"})
    payload = result.model_dump(mode="json")
    assert payload["is_error"] is False
    contract = json.loads(payload["content"][0]["text"])
    assert contract == session_operation_contracts()["operations"]["sessions.list"]


@pytest.mark.asyncio
async def test_old_archive_refuses_indexed_reads_without_migration_and_keeps_raw_available(tmp_path: Path) -> None:
    """Read admission must not upgrade an old archive to make a query succeed."""
    import hashlib
    import sqlite3
    from contextlib import closing

    from polylogue.core.errors import SchemaVersionMismatchError

    root = tmp_path / "archive"
    ids = seed(root, count=1)
    paths = [root / f"{tier}.db" for tier in ("source", "audit", "index")]
    for path, version in zip(paths, (30, 2, 67), strict=True):
        with closing(sqlite3.connect(path)) as conn:
            conn.execute(f"PRAGMA user_version = {version}")
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    sources = raw_sources(tmp_path)
    requests: list[SessionOperation] = [
        SessionList(),
        SessionSearch(expression="needle"),
        SessionRead(ref=ids[0]),
        SessionTimeline(),
    ]
    async with Polylogue(archive_root=root) as api:
        for request in requests:
            with pytest.raises(SchemaVersionMismatchError) as failure:
                await execute_session_operation(api, request)
            assert failure.value.current_version == 67
            assert failure.value.lifecycle_action == "rebuild_index"
        raw = await execute_session_operation(
            api, RawSearch(origin="codex-session", query="needle"), raw_sources=sources
        )
        assert raw.model_dump()["items"]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == before
