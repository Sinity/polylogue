"""Retrieval reads readiness and postings under one snapshot (polylogue-5guoo).

``message_fts_search_readiness_*`` may fall back to a live measurement that
compares two independently-counted relations (``messages_fts_docsize`` against
the indexable ``blocks`` denominator). On an autocommit connection those two
counts are two separate read snapshots, so an unrelated commit landing between
them makes the counts disagree and ``check_fts_readiness`` refuses a perfectly
healthy archive. The daemon commits continuously, so this is the ordinary live
condition, not an exotic race.

The search entry points therefore open their own deferred read transaction
when -- and only when -- the caller is not already inside one, and release it
with ``rollback`` so retrieval never becomes a writer.
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import aiosqlite
import pytest

import polylogue.storage.fts.fts_lifecycle as fts_lifecycle
from polylogue.storage.fts.freshness import record_fts_invariant_snapshot_sync
from polylogue.storage.fts.fts_lifecycle import fts_invariant_snapshot_sync
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.queries.sessions_search import (
    search_action_session_hits,
    search_session_evidence_hits,
    search_session_hits,
)
from tests.infra.identity import archive_message_id

_ORIGIN = "unknown-export"
_TERM = "snapshot"

SearchEntryPoint = Callable[..., Awaitable[Any]]

# Every retrieval entry point in sessions_search that measures readiness and
# then matches; each must take the same snapshot decision.
_ENTRY_POINTS: tuple[SearchEntryPoint, ...] = (
    search_session_hits,
    search_session_evidence_hits,
    search_action_session_hits,
)


def _seed_session(conn: sqlite3.Connection, native_session_id: str, text: str) -> None:
    """Insert one minimal searchable session/message/block."""
    session_id = f"{_ORIGIN}:{native_session_id}"
    content_hash = b"e" * 32
    message_id = archive_message_id(session_id, "m0", position=0)
    conn.execute(
        "INSERT OR IGNORE INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
        (native_session_id, _ORIGIN, "snapshot ownership", content_hash),
    )
    conn.execute(
        """
        INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash)
        VALUES (?, 'm0', 0, 'user', 'message', ?)
        """,
        (session_id, content_hash),
    )
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text, content_hash)
        VALUES (?, ?, 0, 'text', ?, ?)
        """,
        (message_id, session_id, text, content_hash),
    )


@pytest.fixture
def searchable_db(tmp_path: Path) -> Path:
    """An archive whose message FTS is exactly fresh and answers ``_TERM``."""
    db_path = tmp_path / "index.db"
    with open_connection(db_path) as conn:
        _seed_session(conn, "conv-snapshot", f"searchable {_TERM} content")
        record_fts_invariant_snapshot_sync(conn, fts_invariant_snapshot_sync(conn))
        conn.commit()
    return db_path


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=lambda fn: fn.__name__)
async def test_search_releases_the_snapshot_it_opened(searchable_db: Path, entry_point: SearchEntryPoint) -> None:
    """An autocommit caller must be handed back an autocommit connection.

    Mutation that fails this: drop the ``finally``/``rollback`` arm. The
    connection then stays inside the deferred transaction search opened,
    pinning a WAL read snapshot open for the rest of the caller's life.
    """
    async with aiosqlite.connect(searchable_db) as conn:
        conn.row_factory = aiosqlite.Row
        assert conn.in_transaction is False

        await entry_point(conn, _TERM, limit=5)

        assert conn.in_transaction is False


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=lambda fn: fn.__name__)
async def test_search_does_not_discard_a_caller_transaction(searchable_db: Path, entry_point: SearchEntryPoint) -> None:
    """Search must not roll back writes it does not own.

    Mutation that fails this: hardcode ``owns_snapshot = True``. The
    ``finally`` arm then rolls back the caller's still-pending INSERT and both
    assertions below fail -- retrieval would silently destroy an in-flight
    ingest transaction that happened to run a search.
    """
    async with aiosqlite.connect(searchable_db) as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute("BEGIN")
        await conn.execute(
            "INSERT INTO sessions (native_id, origin, title, content_hash) VALUES (?, ?, ?, ?)",
            ("pending-caller-write", _ORIGIN, "uncommitted", b"p" * 32),
        )
        assert conn.in_transaction is True

        await entry_point(conn, _TERM, limit=5)

        assert conn.in_transaction is True, "search took ownership of a transaction it did not open"
        row = await (
            await conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id = 'pending-caller-write'")
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 1, "search rolled back the caller's pending write"


@pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=lambda fn: fn.__name__)
async def test_commit_between_readiness_probes_does_not_refuse(
    searchable_db: Path,
    entry_point: SearchEntryPoint,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A commit landing mid-measurement must not fake an incomplete index.

    The count-only fallback runs only when no trusted freshness record exists,
    so the record is removed first. ``fts_index_status_async`` is then wrapped
    to commit a new indexable block from a *separate* connection at the exact
    point between the indexed-row count and the indexable-row count -- the
    interleaving the daemon produces continuously.

    Mutation that fails this: remove the ``BEGIN``/``owns_snapshot`` arm. The
    second count then reads a newer snapshot than the first, the totals
    disagree by the racing row, and ``check_fts_readiness`` raises
    "Search index is incomplete" against an archive that is in fact complete.
    """
    with open_connection(searchable_db) as setup_conn:
        setup_conn.execute("DELETE FROM fts_freshness_state WHERE surface = 'messages_fts'")
        setup_conn.commit()

    real_status = fts_lifecycle.fts_index_status_async
    counter = itertools.count()
    raced = False

    async def racing_status(probe_conn: aiosqlite.Connection) -> Any:
        nonlocal raced
        status = await real_status(probe_conn)
        with open_connection(searchable_db) as writer:
            _seed_session(writer, f"conv-late-{next(counter)}", "late arriving block")
            writer.commit()
        raced = True
        return status

    monkeypatch.setattr(fts_lifecycle, "fts_index_status_async", racing_status)

    async with aiosqlite.connect(searchable_db) as conn:
        conn.row_factory = aiosqlite.Row
        # No exception: the readiness probes and the MATCH share one snapshot.
        await entry_point(conn, _TERM, limit=5)

    assert raced, "the count-only fallback never ran, so no snapshot straddling was exercised"
