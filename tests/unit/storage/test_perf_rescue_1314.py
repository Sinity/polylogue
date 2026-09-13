"""Regression guards for the #1314 perf rescue cluster.

These tests pin the four perf invariants from issue #1314 by counting
SQL operations (not wall-clock time), so they stay portable across
hosts and CI shapes.

1. ``_SESSION_INSIGHT_REBUILD_PAGE_SIZE`` must be at least 50; the
   page-size-1 regression produced ~17K SQL round-trips for ~4K
   sessions.
2. ``search_session_hits`` must return current FTS membership through its
   authoritative relation check.
3. ``get_origin_metrics_rows`` must read the per-session aggregates on
   ``sessions`` instead of scanning ``messages``.
4. The hydration path (``get_messages*``) must not call pydantic
   ``model_copy`` per message — in-place attachment of content blocks
   is the load-bearing invariant.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from polylogue.storage.derived.session.rebuild import _SESSION_INSIGHT_REBUILD_PAGE_SIZE
from polylogue.storage.sqlite.queries.sessions_search import search_session_hits
from polylogue.storage.sqlite.queries.stats import get_origin_metrics_rows
from tests.benchmarks.helpers import open_bench_store

pytest_plugins = ("tests.benchmarks.conftest",)


@pytest.fixture
def isolated_bench_db_1k(tmp_path: Path, bench_db_1k: Path) -> Path:
    """Give each test a private copy of the session-scoped benchmark input."""
    index_db = tmp_path / "index.db"
    shutil.copy2(bench_db_1k.parent / "index.db", index_db)
    return index_db


@contextmanager
def _capture_aiosqlite_sql() -> Iterator[list[str]]:
    """Capture every SQL string passed to ``aiosqlite.Connection.execute``."""
    statements: list[str] = []
    original = aiosqlite.Connection.execute

    async def _spy(self: aiosqlite.Connection, sql: str, *args: Any, **kwargs: Any) -> Any:
        statements.append(sql)
        return await original(self, sql, *args, **kwargs)

    aiosqlite.Connection.execute = _spy  # type: ignore[method-assign,assignment]
    try:
        yield statements
    finally:
        aiosqlite.Connection.execute = original  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Item 1: rebuild page size.
# ---------------------------------------------------------------------------


def test_session_insight_rebuild_page_size_is_at_least_50() -> None:
    """The page-size-1 regression caused ~17K SQL round-trips for ~4K convs.

    The message-budget chunker (#1314) is the actual safety net for large
    sessions; the page size only controls per-session SQL
    round-trips. Anything below 50 reintroduces the round-trip storm.
    """
    assert _SESSION_INSIGHT_REBUILD_PAGE_SIZE >= 50, (
        "Page size dropped back below the #1314 floor — full rebuilds will "
        "spend most of their wall-clock in per-session round-trips."
    )


# ---------------------------------------------------------------------------
# Item 2: authoritative FTS membership before retrieval.
# ---------------------------------------------------------------------------


def test_search_session_hits_returns_current_fts_membership(isolated_bench_db_1k: Path) -> None:
    """Search returns indexed sessions after checking the canonical FTS relation."""
    with open_bench_store(isolated_bench_db_1k) as store:
        backend = store.backend

        async def _run() -> list[str]:
            async with backend.connection() as conn:
                return (await search_session_hits(conn, "analysis", limit=5)).session_ids()

        assert store.run(_run()), "benchmark fixture must retain at least one searchable analysis session"


# ---------------------------------------------------------------------------
# Item 3: provider metrics reads sessions aggregates.
# ---------------------------------------------------------------------------


def test_origin_metrics_reads_sessions_aggregates(isolated_bench_db_1k: Path) -> None:
    """Origin metrics must source the per-session pre-aggregates from
    ``sessions`` rather than scanning ``messages``.
    """
    with open_bench_store(isolated_bench_db_1k) as store:
        backend = store.backend

        async def _run() -> list[dict[str, object]]:
            async with backend.connection() as conn:
                rows = await get_origin_metrics_rows(conn)
                return [dict(row) for row in rows]

        with _capture_aiosqlite_sql() as statements:
            rows = store.run(_run())

        assert rows, "1k-message benchmark fixture should produce at least one origin row"
        joined = "\n".join(statements).lower()
        assert "from sessions" in joined
        assert "session_stats" not in joined
        assert "from messages" not in joined

        # Result envelope must keep the contract intact.
        first = rows[0]
        for key in (
            "origin",
            "session_count",
            "message_count",
            "user_message_count",
            "assistant_message_count",
            "user_word_sum",
            "assistant_word_sum",
            "tool_use_count",
            "thinking_count",
            "sessions_with_tools",
            "sessions_with_thinking",
        ):
            assert key in first, f"OriginMetricsRow contract dropped {key!r}"


# ---------------------------------------------------------------------------
# Item 4: hydration avoids per-message model_copy.
# ---------------------------------------------------------------------------


def test_get_messages_hydration_does_not_call_model_copy(isolated_bench_db_1k: Path) -> None:
    """``get_messages`` must mutate the freshly-constructed MessageRecord
    instances in place rather than calling pydantic's ``model_copy``.

    The hot hydration path used ``model_copy`` to attach content blocks —
    every call built a second BaseModel instance per message. In-place
    attachment is the #1314 invariant.
    """
    from polylogue.storage.runtime import MessageRecord

    calls: list[None] = []
    original = MessageRecord.model_copy

    def _spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        return original(self, *args, **kwargs)

    MessageRecord.model_copy = _spy  # type: ignore[method-assign]
    try:
        with open_bench_store(isolated_bench_db_1k) as store:

            async def _run() -> int:
                summaries = await store.repository.list_summaries(limit=5)
                target = next(iter(summaries), None)
                if target is None:
                    return 0
                messages = await store.repository.get_messages(str(target.id))
                assert isinstance(messages, list)
                return len(messages)

            store.run(_run())
    finally:
        MessageRecord.model_copy = original  # type: ignore[method-assign]

    assert not calls, (
        "get_messages still calls MessageRecord.model_copy — the #1314 in-place attachment invariant regressed."
    )
