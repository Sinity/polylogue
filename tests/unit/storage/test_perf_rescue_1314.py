"""Regression guards for the #1314 perf rescue cluster.

These tests pin the four perf invariants from issue #1314 by counting
SQL operations (not wall-clock time), so they stay portable across
hosts and CI shapes.

1. ``_SESSION_INSIGHT_REBUILD_PAGE_SIZE`` must be at least 50; the
   page-size-1 regression produced ~17K SQL round-trips for ~4K
   sessions.
2. ``search_session_hits`` must skip archive-scale FTS COUNT(*) probes
   when the daemon-maintained freshness ledger says the message FTS surface is
   ready, and must fall back to exact verification when that row is absent.
3. ``get_origin_metrics_rows`` must read the per-session aggregates on
   ``sessions`` instead of scanning ``messages``.
4. The hydration path (``get_messages*``) must not call pydantic
   ``model_copy`` per message — in-place attachment of content blocks
   is the load-bearing invariant.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from polylogue.storage.insights.session.rebuild import _SESSION_INSIGHT_REBUILD_PAGE_SIZE
from polylogue.storage.sqlite.queries.sessions_search import search_session_hits
from polylogue.storage.sqlite.queries.stats import get_origin_metrics_rows
from tests.benchmarks.helpers import open_bench_store


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
# Item 2: exact FTS freshness before retrieval.
# ---------------------------------------------------------------------------


@pytest.mark.scale_small
def test_search_session_hits_uses_freshness_ledger_before_match(tier_small_db: Path) -> None:
    """Search should not pay archive-scale COUNT(*) probes after daemon readiness."""
    from polylogue.storage.fts.freshness import READY, record_fts_surface_state_async

    with open_bench_store(tier_small_db) as store:
        backend = store.backend

        async def _run(statements: list[str]) -> None:
            async with backend.connection() as conn:
                await record_fts_surface_state_async(
                    conn,
                    surface="messages_fts",
                    state=READY,
                    source_rows=1,
                    indexed_rows=1,
                )
                await conn.commit()
                await search_session_hits(conn, "analysis", limit=5)

        with _capture_aiosqlite_sql() as statements:
            store.run(_run(statements))

        lowered = [" ".join(sql.lower().split()) for sql in statements]
        match_index = next(i for i, sql in enumerate(lowered) if "messages_fts match" in sql)
        assert all("count(*) from messages_fts_docsize" not in sql for sql in lowered[:match_index])
        assert all("count(*) from messages where text is not null" not in sql for sql in lowered[:match_index])


@pytest.mark.scale_small
def test_search_session_hits_falls_back_to_exact_freshness(tier_small_db: Path) -> None:
    """Absent ledger rows fall back to exact FTS verification before MATCH."""
    with open_bench_store(tier_small_db) as store:
        backend = store.backend

        async def _run(statements: list[str]) -> None:
            async with backend.connection() as conn:
                await conn.execute("DELETE FROM fts_freshness_state WHERE surface = 'messages_fts'")
                await conn.commit()
                await search_session_hits(conn, "analysis", limit=5)

        with _capture_aiosqlite_sql() as statements:
            store.run(_run(statements))

        lowered = [sql.lower() for sql in statements]
        docsize_count_index = next(i for i, sql in enumerate(lowered) if "count(*) from messages_fts_docsize" in sql)
        block_probe_index = next(
            i for i, sql in enumerate(lowered) if "from blocks" in sql and ("count(*)" in sql or "limit 1" in sql)
        )
        match_index = next(i for i, sql in enumerate(lowered) if "messages_fts match" in sql)
        assert docsize_count_index < match_index
        assert block_probe_index < match_index


# ---------------------------------------------------------------------------
# Item 3: provider metrics reads sessions aggregates.
# ---------------------------------------------------------------------------


@pytest.mark.scale_small
def test_origin_metrics_reads_sessions_aggregates(tier_small_db: Path) -> None:
    """Origin metrics must source the per-session pre-aggregates from
    ``sessions`` rather than scanning ``messages``.
    """
    with open_bench_store(tier_small_db) as store:
        backend = store.backend

        async def _run() -> list[dict[str, object]]:
            async with backend.connection() as conn:
                rows = await get_origin_metrics_rows(conn)
                return [dict(row) for row in rows]

        with _capture_aiosqlite_sql() as statements:
            rows = store.run(_run())

        assert rows, "scale_small fixture should produce at least one origin row"
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


@pytest.mark.scale_small
def test_get_messages_hydration_does_not_call_model_copy(tier_small_db: Path) -> None:
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
        with open_bench_store(tier_small_db) as store:

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
