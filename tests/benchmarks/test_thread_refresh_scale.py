"""Regression benchmark for polylogue-6wnh: bound thread refresh cost.

Live evidence (2026-07-04, ``/realm/tmp/polylogue-workload-graph-tail-499e26363.json``):
the worst recent ``append.index.graph_resolve`` sample on a 340.8 MB Codex
append spent 3.020976s of a 3.040429s total in
``append.index.graph_resolve.thread_refresh`` (``_refresh_thread`` in
``polylogue/storage/sqlite/archive_tiers/write.py``). The live archive at that
time had 3 threads averaging ~3000 members each (``thread_session_count: 8992``
across ``thread_count: 3`` per the same evidence snapshot).

Root cause (traced via ``_refresh_thread`` source + a granular per-statement
profiling harness against a synthetic fixture of this shape): when a thread
member's ``sort_key_ms`` (``COALESCE(updated_at_ms, created_at_ms)``) moves
past siblings -- e.g. a late append to an older session in a large,
long-lived thread built from many resumes/forks/subagent spawns -- the
previous implementation unconditionally deleted *every* row in
``thread_sessions`` for that thread and reinserted all of them one at a time
via a Python loop, even when only a handful of positions actually moved.
This is real O(thread_size) row-mutation work, not a query-shape or
per-statement-overhead artifact (see ``test_graph_resolve_deferred_tail.py``
for the analogous conclusion on the #2467 deferred-tail path).

The fix (this bead) makes ``_refresh_thread`` refresh only the affected span:
it trims the common leading/trailing run shared between the old and new
orderings (safe because equal-length lists mean index ``i`` names the same
numeric ``position`` in both orderings) and only delete+reinserts the
differing middle span, using ``executemany`` instead of a per-row Python
loop. A full front-to-back reorder (the worst case -- everything moves) is
still O(thread_size), matching the architectural floor for a dense
0..n-1 position invariant; a *localized* reorder (the common real-world
shape: one member's timestamp jumps past a handful of siblings, not the
entire thread) now costs O(affected span + two O(n) SELECTs to compute the
diff), not O(thread_size) row mutations.

Run with:
    pytest tests/benchmarks/test_thread_refresh_scale.py --benchmark-enable -p no:xdist -v
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers import write as archive_tier_write
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_INDEX_DDL = ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX]

_ORIGIN = "codex-session"


def _build_thread_fixture(db_path: Path, *, n_sessions: int) -> tuple[sqlite3.Connection, str]:
    """Seed one root session plus ``n_sessions - 1`` descendants sharing its
    ``root_session_id``, with ``threads``/``thread_sessions`` already
    materialized to match ascending-``updated_at_ms`` order -- the on-disk
    shape of one big long-lived Codex thread built from many resumes/forks.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_INDEX_DDL)

    root_native_id = "root"
    root_session_id = f"{_ORIGIN}:{root_native_id}"
    conn.execute(
        "INSERT INTO sessions (origin, native_id, title, content_hash, root_session_id, "
        "created_at_ms, updated_at_ms) VALUES (?, ?, 't', ?, ?, 0, 0)",
        (_ORIGIN, root_native_id, b"r" * 32, root_session_id),
    )
    rows = [
        (_ORIGIN, f"child{i}", bytes([i % 256]) * 32, root_session_id, i * 1000, i * 1000) for i in range(1, n_sessions)
    ]
    conn.executemany(
        "INSERT INTO sessions (origin, native_id, content_hash, root_session_id, "
        "created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    conn.execute(
        "INSERT INTO threads (thread_id, created_at_ms, session_count, depth) VALUES (?, 0, ?, ?)",
        (root_session_id, n_sessions, n_sessions - 1),
    )
    session_ids = [root_session_id] + [f"{_ORIGIN}:child{i}" for i in range(1, n_sessions)]
    conn.executemany(
        "INSERT INTO thread_sessions (thread_id, session_id, position) VALUES (?, ?, ?)",
        [(root_session_id, sid, pos) for pos, sid in enumerate(session_ids)],
    )
    conn.commit()
    return conn, root_session_id


def _bump_session(conn: sqlite3.Connection, *, native_id: str, n_sessions: int, new_ms: int) -> None:
    conn.execute(
        "UPDATE sessions SET updated_at_ms = ? WHERE origin = ? AND native_id = ?",
        (new_ms, _ORIGIN, native_id),
    )
    conn.commit()


def _time_full_reorder(n_sessions: int, tmp_path: Path) -> float:
    """Worst case: the middle member's timestamp jumps past every later
    sibling, forcing it to the very end -- roughly half the thread's
    positions must shift.
    """
    conn, root_session_id = _build_thread_fixture(tmp_path / f"full_{n_sessions}.db", n_sessions=n_sessions)
    try:
        mid = n_sessions // 2
        native_id = "root" if mid == 0 else f"child{mid}"
        _bump_session(conn, native_id=native_id, n_sessions=n_sessions, new_ms=n_sessions * 1000 + 1)
        started = time.perf_counter()
        archive_tier_write._refresh_thread(conn, root_session_id)
        conn.commit()
        return time.perf_counter() - started
    finally:
        conn.close()


def _time_local_reorder(n_sessions: int, tmp_path: Path, *, span: int = 20) -> float:
    """Common case: a member near the tail jumps past only ``span``
    immediate siblings (e.g. concurrent subagent timestamps interleaving),
    not the entire thread.
    """
    conn, root_session_id = _build_thread_fixture(tmp_path / f"local_{n_sessions}.db", n_sessions=n_sessions)
    try:
        idx = max(n_sessions - span - 1, 1)
        native_id = f"child{idx}"
        _bump_session(conn, native_id=native_id, n_sessions=n_sessions, new_ms=(idx + span) * 1000 + 1)
        started = time.perf_counter()
        archive_tier_write._refresh_thread(conn, root_session_id)
        conn.commit()
        return time.perf_counter() - started
    finally:
        conn.close()


def _touched_row_count_for_local_reorder(n_sessions: int, tmp_path: Path, *, span: int = 20) -> int:
    """Count actual thread_sessions INSERT statements issued for a localized
    reorder -- the direct, host-noise-free measure of "only affected rows
    were touched" (wall-clock time on a small in-process fixture is too
    dominated by fixed per-call overhead and host jitter to reliably show
    the asymptotic win at small N; row counts are exact and deterministic).
    """
    conn, root_session_id = _build_thread_fixture(tmp_path / f"count_{n_sessions}.db", n_sessions=n_sessions)
    try:
        idx = max(n_sessions - span - 1, 1)
        native_id = f"child{idx}"
        _bump_session(conn, native_id=native_id, n_sessions=n_sessions, new_ms=(idx + span) * 1000 + 1)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        archive_tier_write._refresh_thread(conn, root_session_id)
        conn.set_trace_callback(None)
        conn.commit()
        return sum(1 for stmt in statements if "INSERT INTO thread_sessions" in stmt)
    finally:
        conn.close()


@pytest.mark.benchmark
def test_thread_refresh_full_reorder_scales_linearly_not_quadratically(tmp_path: Path) -> None:
    """A full front-to-back reorder is architecturally O(thread_size) under
    the dense 0..n-1 position invariant (this bead does not remove that
    floor). Guard against a *new* accidental quadratic regression.
    """
    small_seconds = _time_full_reorder(500, tmp_path)
    large_seconds = _time_full_reorder(4000, tmp_path)

    ratio = large_seconds / max(small_seconds, 1e-9)
    print(
        f"\nthread_refresh full-reorder scaling: 500={small_seconds:.4f}s, "
        f"4000={large_seconds:.4f}s, ratio={ratio:.1f}x (linear expects ~8x)"
    )
    assert ratio < 25, (
        f"thread_refresh cost grew {ratio:.1f}x for 8x the thread size "
        f"(500={small_seconds:.4f}s, 4000={large_seconds:.4f}s) -- expected roughly "
        "linear (~8x); this smells like a new quadratic regression in "
        "_refresh_thread, reopening the polylogue-3wb-class 260s tail."
    )


@pytest.mark.benchmark
def test_thread_refresh_local_reorder_is_bounded_by_affected_span(tmp_path: Path) -> None:
    """The concrete polylogue-6wnh acceptance criterion: a localized reorder
    in a giant thread must touch only the affected rows, not rebuild the
    whole thread.

    Row-mutation count is the deterministic, host-noise-free signal here --
    on a small in-process synthetic fixture, wall-clock time is dominated by
    fixed per-call overhead rather than the row-count difference this fix
    targets (the live 3.02s/9-thousand-row production evidence lives on a
    multi-GiB disk-resident archive; see test_graph_resolve_deferred_tail.py
    for the sibling benchmark's discussion of the same measurement
    trade-off). Before this bead, a local reorder issued ``n_sessions``
    thread_sessions INSERTs (a full unconditional rebuild) regardless of how
    few positions actually moved; after the fix it issues only
    ``span + 1``.
    """
    n_sessions = 4000
    span = 20
    touched = _touched_row_count_for_local_reorder(n_sessions, tmp_path, span=span)

    print(f"\nthread_refresh n={n_sessions} local reorder (span={span}): {touched} rows reinserted")
    # A handful more than `span` (the moved member plus everyone strictly
    # between its old and new slot) -- nowhere near the full n_sessions a
    # blanket rebuild would touch.
    assert touched <= span + 5, (
        f"a {span}-sibling local reorder in a {n_sessions}-member thread reinserted "
        f"{touched} thread_sessions rows -- expected roughly {span}, not the full "
        f"{n_sessions}-member thread. thread_refresh is no longer bounding its work "
        "to the affected span (polylogue-6wnh regression)."
    )

    # Wall-clock sanity check (informational, generous headroom): the local
    # reorder should still not be dramatically slower than a full reorder of
    # the same thread.
    full_seconds = _time_full_reorder(n_sessions, tmp_path)
    local_seconds = _time_local_reorder(n_sessions, tmp_path, span=span)
    print(f"thread_refresh n={n_sessions}: full-reorder={full_seconds:.4f}s, local-reorder={local_seconds:.4f}s")
