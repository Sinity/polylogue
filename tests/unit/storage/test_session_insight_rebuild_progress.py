"""Progress reporting + atomicity invariants for ``rebuild_session_insights_sync`` (#1607).

Pins two contracts:

1. The seven DELETEs that open a full rebuild emit a progress event per
   table (instead of running silently for seconds-to-minutes against
   the production-scale insight tables). Operators tailing the log see
   forward motion at table granularity, not "stuck at start".

2. The full rebuild runs as a single transaction — the implicit
   transaction started by the first DELETE spans every subsequent DML
   through the eventual ``conn.commit()``. A SIGKILL or unraised
   exception before COMMIT must leave the prior insights intact rather
   than emptying the archive.
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
    db_path = tmp_path / "polylogue.db"
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
    db_path = tmp_path / "polylogue.db"
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


def test_full_rebuild_emits_progress_for_each_deleted_table(
    cli_workspace: dict[str, Path],
) -> None:
    """The full native ``rebuild_insights`` path must surface DELETE progress
    through the user-supplied callback so "polylogue ... --rebuild-insights"
    stops hanging silently (#1607 parity over archive)."""
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

    delete_events = [desc for desc in events if desc and desc.startswith("rebuild: cleared ")]
    assert delete_events == [
        "rebuild: cleared session_work_events",
        "rebuild: cleared session_phases",
        "rebuild: cleared session_profiles",
        "rebuild: cleared thread_sessions",
        "rebuild: cleared threads",
        "rebuild: cleared insight_materialization",
    ]


# ---------------------------------------------------------------------------
# Atomicity — SIGKILL/exception mid-rebuild leaves prior insights intact
# ---------------------------------------------------------------------------


def test_full_rebuild_rolls_back_on_exception_keeping_prior_profiles(
    cli_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the rebuild loop raises between the DELETE phase and ``conn.commit()``,
    the implicit transaction must roll back so prior session_profiles
    survive — not the "seconds-to-minutes of empty archive" the #1607
    report worried about.
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

    # Now arrange for the next rebuild's per-session loop to fail after the
    # DELETE phase. The archive rebuild imports build_session_insight_records
    # at call time from the rebuild module, so patching it there is observed.
    from polylogue.storage.insights.session import rebuild as rebuild_module

    def _explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated mid-rebuild failure")

    monkeypatch.setattr(rebuild_module, "build_session_insight_records", _explode)

    with pytest.raises(RuntimeError, match="simulated mid-rebuild failure"):
        _rebuild(db_path)

    # The post-failure count must equal the baseline. If the DELETEs had
    # leaked past their implicit transaction, this would be 0.
    surviving = _count_profiles(db_path)
    assert surviving == baseline, f"rebuild failure emptied prior insights: baseline={baseline}, surviving={surviving}"
