"""The message-FTS readiness binding makes inspection cost archive-independent.

polylogue-crwl6 AC6. Every status request path -- ``/healthz``, ``/api/status``,
``/metrics``, ``health``, ``status_snapshot`` -- reaches FTS readiness through
``fts_readiness_info``, which ran an archive-proportional global inspection on
every call. These tests pin the two halves of the replacement: a standing
binding answers without reading ``blocks`` at all, and a binding can never
report ``ready`` for a surface that actually drifted.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.daemon.fts_status import fts_readiness_info
from polylogue.operations.fts_derivation import (
    archive_fts_surface,
    bound_archive_fts_surface,
    fts_readiness_binding,
    stamp_fts_readiness_binding,
)
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.fts.fts_lifecycle import (
    restore_message_fts_triggers_sync,
    suspend_message_fts_triggers_sync,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _session(native_id: str, *, blocks: int) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=native_id,
        title=f"binding {native_id}",
        messages=[
            ParsedMessage(
                provider_message_id=f"m{index}",
                role=Role.USER,
                text=f"searchable text {index}",
                position=index,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=f"searchable text {index}")],
            )
            for index in range(blocks)
        ],
    )


@pytest.fixture
def seeded(tmp_path: Path) -> Iterator[tuple[Path, sqlite3.Connection]]:
    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(db)
    try:
        write_parsed_session_to_archive(conn, _session("bind-1", blocks=4))
        conn.commit()
        yield db, conn
    finally:
        conn.close()


class _BlockReadCounter:
    """Count SQLite's own authorizer events for reads of ``blocks``.

    This is the cost assertion. Wall-clock timing would be a flaky proxy; what
    the acceptance criterion actually demands is that the answer does not depend
    on the block population, and the only way that can hold is if the query
    never reads a block row.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.block_reads = 0

    def __enter__(self) -> _BlockReadCounter:
        def authorize(action: int, arg1: str | None, arg2: str | None, *_rest: str | None) -> int:
            if action == sqlite3.SQLITE_READ and arg1 == "blocks":
                self.block_reads += 1
            return sqlite3.SQLITE_OK

        self._conn.set_authorizer(authorize)
        return self

    def __exit__(self, *_exc: object) -> None:
        self._conn.set_authorizer(None)


def test_bound_readiness_reads_no_blocks(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """A standing binding answers readiness without touching ``blocks``.

    ANTI-VACUITY: restore the exact computation -- make
    ``bound_archive_fts_surface`` return ``archive_fts_surface(conn)`` instead
    of projecting the stored binding -- and ``bound.block_reads == 0`` goes red
    with a real count, because the answer is once again proportional to the
    block population. ``census.block_reads > 0`` pins the opposite direction: a
    counter that never observed anything, or a readiness path that answered
    without looking at the archive at all, fails on the census leg.
    """
    db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()

    with _BlockReadCounter(conn) as census:
        assert archive_fts_surface(conn)["ready"] is True
    with _BlockReadCounter(conn) as bound:
        surface = bound_archive_fts_surface(conn)

    assert surface is not None
    assert surface["ready"] is True
    assert surface["source_rows"] == archive_fts_surface(conn)["source_rows"]
    assert bound.block_reads == 0
    assert census.block_reads > 0
    assert db.exists()


def test_probe_answers_from_the_binding(
    seeded: tuple[Path, sqlite3.Connection],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one route all five callers share must not census a bound surface.

    ``/healthz``, ``/api/status``, ``/metrics``, ``health`` and
    ``status_snapshot`` all reach FTS readiness through ``fts_readiness_info``,
    so pinning it here pins all five. The census is made to raise rather than
    merely counted: a request path that still reaches it is the defect
    polylogue-crwl6 AC6 names, whatever answer it eventually returns.

    ANTI-VACUITY: drop the ``bound_archive_fts_surface`` consultation from
    ``daemon.fts_status._archive_blocks_surface`` and this raises
    ``readiness must not re-derive a bound FTS surface``. The trailing
    assertions pin the opposite direction -- a probe that refused to answer, or
    answered ``unmeasured``, fails on the verdict.
    """
    db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()

    def _forbidden(_conn: sqlite3.Connection) -> dict[str, object]:
        raise AssertionError("readiness must not re-derive a bound FTS surface")

    monkeypatch.setattr("polylogue.operations.fts_derivation.archive_fts_surface", _forbidden)

    payload = fts_readiness_info(db)
    assert payload["messages_ready"] is True
    assert payload["invariant_ready"] is True
    assert payload["coverage_exact"] is True
    assert payload["coverage_pct"] == 100.0


def test_unbound_fallback_reports_timeout_and_resumes_slow_attempt(
    seeded: tuple[Path, sqlite3.Connection], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An aggregate miss is deadline-bounded and never duplicated on the next poll.

    ANTI-VACUITY: bypassing the persistent component registry makes the first
    call wait for the collector and the second call start a duplicate attempt;
    returning the collector's eventual value immediately fabricates a ready
    verdict before the declared deadline has elapsed.
    """
    import polylogue.daemon.fts_status as fts_status

    db, _conn = seeded
    monkeypatch.setattr(fts_status, "_FTS_READINESS_DEADLINE_S", 0.02)
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    calls = {"n": 0}

    from polylogue.operations import status_protocol

    run_collector = status_protocol._run_collector

    def track_completion(spec: object, attempt: object) -> None:
        run_collector(spec, attempt)  # type: ignore[arg-type]
        completed.set()

    monkeypatch.setattr(status_protocol, "_run_collector", track_completion)

    def slow_collection(_dbf: Path, *, exact: bool = False) -> dict[str, object]:
        del exact
        calls["n"] += 1
        started.set()
        release.wait()
        return {"messages_ready": True, "invariant_ready": True, "coverage_exact": True}

    monkeypatch.setattr(fts_status, "_collect_fts_readiness_info", slow_collection)
    try:
        first = fts_readiness_info(db)
        assert first["inspection_state"] == "timed_out"
        assert first["messages_ready"] is False
        assert first["message_indexed_count"] is None
        assert started.wait(timeout=1.0)
        assert calls["n"] == 1

        second = fts_readiness_info(db)
        assert second["inspection_state"] == "refreshing"
        assert second["messages_ready"] is False
        assert calls["n"] == 1

        release.set()
        assert completed.wait(timeout=1.0)
        third = fts_readiness_info(db)
        assert third["inspection_state"] == "fresh"
        assert third["messages_ready"] is True
        assert calls["n"] == 1
    finally:
        release.set()


def test_block_write_retires_the_binding(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """The canonical writer's own transaction deletes the binding."""
    _db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()
    assert fts_readiness_binding(conn) is not None

    write_parsed_session_to_archive(conn, _session("bind-2", blocks=2))
    conn.commit()

    assert fts_readiness_binding(conn) is None
    assert bound_archive_fts_surface(conn) is None


def test_unindexed_block_retires_the_binding(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """A block written while the FTS maintainer is suspended is not certified.

    This is the case the retirement triggers exist for and the only one the
    stored shadow-relation counts cannot see: the bulk paths drop the FTS
    maintenance triggers by name, so ``messages_fts``/``messages_fts_identity``
    gain no row and both counts still equal what the binding recorded, while
    ``blocks`` has grown a searchable row nothing indexed.

    ANTI-VACUITY: drop ``messages_fts_readiness_binding_blocks_ai`` from
    INDEX_DDL. The binding then survives -- ``bound_archive_fts_surface``
    reports ``ready`` for an archive with an unindexed block, which is the false
    certification polylogue-crwl6 AC3 forbids. Keeping the retirement triggers
    out of ``BLOCKS_FTS_TRIGGER_DDL`` is what makes them survive the suspension.
    """
    _db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()
    before = fts_readiness_binding(conn)
    assert before is not None

    message_id, session_id = conn.execute("SELECT message_id, session_id FROM blocks LIMIT 1").fetchone()
    suspend_message_fts_triggers_sync(conn)
    conn.execute(
        "INSERT INTO blocks (message_id, session_id, position, block_type, text) VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, 900, "text", "never indexed"),
    )
    restore_message_fts_triggers_sync(conn)
    conn.commit()

    # The counts the binding recorded are untouched: only the retirement
    # trigger distinguishes this archive from the one that was inspected.
    assert int(conn.execute("SELECT COUNT(*) FROM messages_fts_docsize").fetchone()[0]) == before.indexed_rows
    assert int(conn.execute("SELECT COUNT(*) FROM messages_fts_identity").fetchone()[0]) == before.identity_rows

    assert fts_readiness_binding(conn) is None
    assert bound_archive_fts_surface(conn) is None
    assert archive_fts_surface(conn)["ready"] is False


def test_block_delete_retires_the_binding(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """A delete is a write too; the ``ad`` arm covers the shrinking direction."""
    _db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()

    conn.execute("DELETE FROM blocks WHERE position = 0")
    conn.commit()

    assert fts_readiness_binding(conn) is None


def test_output_drift_is_never_bound_ready(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """A row removed from ``messages_fts`` alone still classifies not-ready.

    ``messages_fts`` is a virtual table and SQLite cannot carry a trigger on
    one, so this direction has no retirement path -- it is caught by comparing
    the binding's recorded shadow-relation counts.

    ANTI-VACUITY: delete the ``_output_row_counts`` comparison from
    ``fts_readiness_binding`` and this returns ``ready=True`` for a
    surface that is missing a document.
    """
    _db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()

    rowid = int(conn.execute("SELECT rowid FROM blocks LIMIT 1").fetchone()[0])
    conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    conn.commit()

    assert bound_archive_fts_surface(conn) is None
    assert archive_fts_surface(conn)["ready"] is False
    assert fts_readiness_info(_db)["messages_ready"] is False


def test_stamp_refuses_an_invalid_surface(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """A binding states a valid verdict, so a stale surface publishes nothing."""
    _db, conn = seeded
    rowid = int(conn.execute("SELECT rowid FROM blocks LIMIT 1").fetchone()[0])
    conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (rowid,))
    conn.commit()

    assert stamp_fts_readiness_binding(conn) is False
    assert fts_readiness_binding(conn) is None


def test_recipe_change_invalidates_every_binding(seeded: tuple[Path, sqlite3.Connection]) -> None:
    """A fold/tokenizer recipe bump makes a standing binding compare unequal."""
    _db, conn = seeded
    assert stamp_fts_readiness_binding(conn)
    conn.commit()
    conn.execute("UPDATE messages_fts_readiness_binding SET recipe_id = 'messages_fts.v0:other'")
    conn.commit()

    assert fts_readiness_binding(conn) is None


def test_binding_stage_publishes_then_converges(tmp_path: Path) -> None:
    """The daemon stage is the production publisher, and it stops once bound."""
    from polylogue.daemon.convergence_stages import make_fts_readiness_binding_stage

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    db = archive_root / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(db)
    try:
        write_parsed_session_to_archive(conn, _session("stage-1", blocks=3))
        conn.commit()
    finally:
        conn.close()

    stage = make_fts_readiness_binding_stage(db)
    assert stage.check(db) is True
    assert stage.execute(db) is True
    assert stage.check(db) is False

    reader = sqlite3.connect(db)
    try:
        assert fts_readiness_binding(reader) is not None
    finally:
        reader.close()
