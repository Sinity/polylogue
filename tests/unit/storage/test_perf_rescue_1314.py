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

import re
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

from polylogue.storage.derived.session.rebuild import _SESSION_INSIGHT_REBUILD_PAGE_SIZE
from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync, restore_fts_triggers_sync
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


def _fixture_search_term(index_db: Path) -> str:
    """Pick a term the fixture corpus actually indexes, rather than guessing one.

    The benchmark corpus is generated, so hard-coding a word makes the test
    assert the generator's vocabulary instead of search behaviour.
    """
    with sqlite3.connect(index_db) as conn:
        counts: dict[str, int] = {}
        for (text,) in conn.execute("SELECT search_text FROM blocks WHERE search_text IS NOT NULL LIMIT 500"):
            for token in set(re.findall(r"[a-z]{5,}", str(text).lower())):
                counts[token] = counts.get(token, 0) + 1
    # Deterministic: most frequent, ties broken alphabetically.
    term = min(counts, key=lambda token: (-counts[token], token))
    assert counts[term] >= 2, "benchmark fixture produced no repeated searchable term"
    return term


def test_search_session_hits_returns_current_fts_membership(isolated_bench_db_1k: Path) -> None:
    """Search reports exactly the sessions the message FTS relation currently holds.

    Anti-vacuity: serve results from a cached or ledger-declared membership
    instead of the live relation, drop the session scope so every session is
    returned, or let a repaired-away session keep matching, and this goes red.
    """
    term = _fixture_search_term(isolated_bench_db_1k)

    with sqlite3.connect(isolated_bench_db_1k) as conn:
        expected = {
            str(row[0])
            for row in conn.execute(
                # messages_fts is contentless, so its own columns read back
                # NULL; the FTS rowid is the canonical block rowid, which is
                # what names the session behind a match.
                """
                SELECT DISTINCT b.session_id
                FROM messages_fts AS f
                JOIN blocks AS b ON b.rowid = f.rowid
                WHERE messages_fts MATCH ?
                """,
                (term,),
            )
        }
    assert expected, f"fixture term {term!r} matched no indexed session"

    with open_bench_store(isolated_bench_db_1k) as store:
        backend = store.backend

        async def _search() -> list[str]:
            async with backend.connection() as conn:
                return (await search_session_hits(conn, term, limit=1000)).session_ids()

        assert set(store.run(_search())) == expected

    # Remove one session's content through the canonical repair route; search
    # must follow the relation's new membership on the very next read.
    retired = sorted(expected)[0]
    with sqlite3.connect(isolated_bench_db_1k) as conn:
        restore_fts_triggers_sync(conn)
        conn.execute("DELETE FROM blocks WHERE session_id = ?", (retired,))
        repair_message_fts_index_sync(conn, [retired], record_exact_snapshot=False)
        conn.commit()

    with open_bench_store(isolated_bench_db_1k) as store:
        backend = store.backend

        async def _search_again() -> list[str]:
            async with backend.connection() as conn:
                return (await search_session_hits(conn, term, limit=1000)).session_ids()

        assert set(store.run(_search_again())) == expected - {retired}


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
