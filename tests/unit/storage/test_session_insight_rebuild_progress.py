"""Progress reporting + bounded-WAL invariants for ``rebuild_session_insights_sync``.

Pins the contracts of the bounded-WAL rebuild model (#1607 heartbeats, #2458
per-chunk commit):

1. ``_delete_tables_with_progress_sync`` still emits one progress event per
   table it clears (the aggregate-table helper, exercised directly below).

2. The full rebuild no longer clears per-session insight tables upfront. It
   upserts per chunk, commits per chunk (bounded WAL), and prunes orphan rows
   after the chunk loop — emitting a per-table "pruned orphans" heartbeat so
   operators still see forward motion.

3. Because per-session insight rows are upserted (not wiped) and the failing
   chunk never commits, an exception mid-rebuild leaves the prior insights
   intact rather than emptying the archive — now achieved by upsert + no
   upfront delete instead of one giant transaction.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.storage.insights.session.rebuild import _delete_tables_with_progress_sync
from tests.infra.storage_records import SessionBuilder


def _rebuild(db_path: Path, *, progress_callback: object = None) -> None:
    async def _run() -> None:
        archive = Polylogue(archive_root=db_path.parent, db_path=db_path)
        try:
            await archive.rebuild_insights(progress_callback=progress_callback)  # type: ignore[arg-type]
        finally:
            await archive.close()

    asyncio.run(_run())


def _count_profiles(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM session_profiles").fetchone()
        return int(row[0])


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


def test_delete_tables_with_progress_emits_one_event_per_table(tmp_path: Path) -> None:
    """The helper that wraps the seven DELETEs must emit a progress event
    per table so the operator sees forward motion at table granularity."""
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE a (id INT)")
        conn.execute("CREATE TABLE b (id INT)")
        conn.execute("CREATE TABLE c (id INT)")
        conn.execute("INSERT INTO a VALUES (1), (2), (3)")
        conn.execute("INSERT INTO b VALUES (1)")
        # c stays empty
        conn.commit()

        events: list[tuple[int, str | None]] = []

        def progress(amount: int, desc: str | None = None) -> None:
            events.append((amount, desc))

        _delete_tables_with_progress_sync(
            conn,
            tables=("a", "b", "c"),
            progress_callback=progress,
        )
    finally:
        conn.close()

    assert [desc for _, desc in events] == [
        "rebuild: cleared a",
        "rebuild: cleared b",
        "rebuild: cleared c",
    ]
    # rowcount per table: 3, 1, 0
    assert [amount for amount, _ in events] == [3, 1, 0]


def test_delete_tables_progress_callback_is_optional(tmp_path: Path) -> None:
    """Missing callback must not raise; the DELETEs still execute."""
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE t (id INT)")
        conn.execute("INSERT INTO t VALUES (1), (2)")
        conn.commit()

        _delete_tables_with_progress_sync(conn, tables=("t",), progress_callback=None)

        remaining = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
        assert remaining == 0
    finally:
        conn.close()


def test_full_rebuild_emits_orphan_prune_progress_per_table(
    cli_workspace: dict[str, Path],
) -> None:
    """The bounded-WAL full rebuild (#2458) does not clear per-session insight
    tables upfront; it prunes orphan rows after the chunk loop and emits a
    per-table heartbeat so the operator still sees forward motion. The old
    upfront "cleared session_*" heartbeats must be gone."""
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-1")
        .provider("claude-code")
        .title("seed")
        .updated_at("2026-03-01T10:10:00+00:00")
        .add_message("u1", role="user", text="hi", timestamp="2026-03-01T10:00:00+00:00")
        .save()
    )

    events: list[str | None] = []

    def progress(amount: int, desc: str | None = None) -> None:
        events.append(desc)

    _rebuild(db_path, progress_callback=progress)

    prune_events = [desc for desc in events if desc and desc.startswith("rebuild: pruned orphans from ")]
    assert prune_events == [
        "rebuild: pruned orphans from session_work_events",
        "rebuild: pruned orphans from session_phases",
        "rebuild: pruned orphans from session_runs",
        "rebuild: pruned orphans from session_observed_events",
        "rebuild: pruned orphans from session_context_snapshots",
        "rebuild: pruned orphans from session_latency_profiles",
        "rebuild: pruned orphans from session_profiles",
    ]
    # The bounded-WAL model removed the upfront per-session table wipe.
    assert not [desc for desc in events if desc and desc.startswith("rebuild: cleared session_")]


# ---------------------------------------------------------------------------
# No-empty-window — exception mid-rebuild leaves prior insights intact
# ---------------------------------------------------------------------------


def test_full_rebuild_failure_preserves_prior_profiles(
    cli_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In the bounded-WAL model (#2458) there is no upfront wipe of per-session
    insight tables: each chunk upserts and commits independently. If the loop
    raises while building the first chunk's records, that chunk never commits
    and no other session's rows were touched, so the prior session_profiles
    survive — not the "seconds-to-minutes of empty archive" #1607 worried about.
    """
    db_path = cli_workspace["db_path"]
    (
        SessionBuilder(db_path, "conv-existing")
        .provider("claude-code")
        .title("prior")
        .updated_at("2026-02-01T10:10:00+00:00")
        .add_message("u1", role="user", text="hello", timestamp="2026-02-01T10:00:00+00:00")
        .save()
    )

    # Establish a baseline of insight rows by running one successful rebuild.
    _rebuild(db_path)
    baseline = _count_profiles(db_path)
    assert baseline >= 1, "baseline rebuild produced no profiles"

    # Now arrange for the next rebuild's per-session loop to fail while building
    # the first chunk's records (before any chunk commit). The archive rebuild
    # imports build_session_insight_records at call time from the rebuild
    # module, so patching it there is observed.
    from polylogue.storage.insights.session import rebuild as rebuild_module

    def _explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated mid-rebuild failure")

    monkeypatch.setattr(rebuild_module, "build_session_insight_records", _explode)

    with pytest.raises(RuntimeError, match="simulated mid-rebuild failure"):
        _rebuild(db_path)

    # The post-failure count must equal the baseline. If the rebuild had wiped
    # the per-session tables upfront, this would be 0.
    surviving = _count_profiles(db_path)
    assert surviving == baseline, f"rebuild failure emptied prior insights: baseline={baseline}, surviving={surviving}"
