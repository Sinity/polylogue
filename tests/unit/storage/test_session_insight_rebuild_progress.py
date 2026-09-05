"""Progress reporting + bounded-WAL invariants for ``rebuild_session_insights_sync``.

Pins the contracts of the bounded-WAL rebuild model (#1607 heartbeats, #2458
per-chunk commit):

1. The full rebuild no longer clears per-session insight tables upfront. It
   upserts per chunk, commits per chunk (bounded WAL), and prunes orphan rows
   after the chunk loop — emitting a per-table "pruned orphans" heartbeat so
   operators still see forward motion.

2. Because per-session insight rows are upserted (not wiped) and the failing
   chunk never commits, an exception mid-rebuild leaves the prior insights
   intact rather than emptying the archive — now achieved by upsert + no
   upfront delete instead of one giant transaction.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.storage_scale

from polylogue.api import Polylogue
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

    # polylogue-dab/itvd: session_runs/session_observed_events/
    # session_context_snapshots are no longer materialized tables (they are
    # computed on read by run_projection_relations.py's CTEs), so there is
    # nothing to orphan-prune for them anymore.
    prune_events = [desc for desc in events if desc and desc.startswith("rebuild: pruned orphans from ")]
    assert prune_events == [
        "rebuild: pruned orphans from session_work_events",
        "rebuild: pruned orphans from session_phases",
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
    from polylogue.storage.derived.session import rebuild as rebuild_module

    def _explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated mid-rebuild failure")

    monkeypatch.setattr(rebuild_module, "build_session_insight_records", _explode)

    with pytest.raises(RuntimeError, match="simulated mid-rebuild failure"):
        _rebuild(db_path)

    # The post-failure count must equal the baseline. If the rebuild had wiped
    # the per-session tables upfront, this would be 0.
    surviving = _count_profiles(db_path)
    assert surviving == baseline, f"rebuild failure emptied prior insights: baseline={baseline}, surviving={surviving}"
