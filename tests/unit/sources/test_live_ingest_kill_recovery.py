"""A process killed inside an ingest write loses nothing and duplicates nothing.

A child process runs the production intake route (``FileIntakeAdapter`` over
a ``LiveWatcher``) and is SIGKILLed at a chosen point: after the parsed index
write for the session has run but before the pass returns, so whatever that
write left uncommitted dies with the process. A fresh process then admits the
same source and the archive must hold exactly the source's messages, each
once, with no raw row carrying a parse error.

The existing whole-daemon SIGKILL test proves session counts after a kill at a
random moment of a cold build. This one pins the moment to the write itself
and checks message and raw rows, for a first ingest and for an append to an
already-ingested file.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

import polylogue.sources.live.cursor as cursor_module
from polylogue import Polylogue
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.sources.revision_backfill import RetainedPreparationRetryableError
from polylogue.sources.source_layout import export_drop_layout
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationReplacement
from tests.infra.raw_owner_routes import live_owner_set

_SESSION_ID = "kill-recovery"
_MAX_DEFERRED_PAGES = 20

# The child's own admission helper mirrors ``_admit`` below. ``KILL_AT_WRITE``
# names which parsed-index write (1-based) kills the process: the patched
# write runs to completion and then the process receives SIGKILL, before the
# enclosing pass can commit, record cursors or return.
_CHILD = textwrap.dedent(
    """
    import asyncio, os, signal, sys
    from pathlib import Path

    from polylogue import Polylogue
    from polylogue.core.compute import BoundedComputeAdapter
    from polylogue.daemon.intake import AdmissionOutcome
    from polylogue.daemon.raw_observation_owner import RawObservationConvergenceOwner
    from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
    from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
    from polylogue.sources.live import LiveWatcher, WatchSource
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.sqlite_capture import LiveSQLiteCaptureStage
    from polylogue.sources.source_layout import export_drop_layout
    from polylogue.storage.sqlite.archive_tiers import revision_governance

    archive_root = Path(sys.argv[1])
    source_root = Path(sys.argv[2])
    kill_at = int(sys.argv[3])
    original = revision_governance._write_parsed_precedence_result
    calls = 0

    def write_then_die(*args, **kwargs):
        global calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == kill_at:
            sys.stdout.write("killing\\n")
            sys.stdout.flush()
            os.kill(os.getpid(), signal.SIGKILL)
        return result

    revision_governance._write_parsed_precedence_result = write_then_die

    async def main():
        archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
        # The daemon's intake owners; the process is killed, so none settles.
        compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
        coordinator = DaemonWriteCoordinator(archive_root=archive_root)
        raw_owner = RawObservationConvergenceOwner(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            write_coordinator=coordinator,
        )
        watcher = LiveWatcher(
            archive,
            (WatchSource(name="claude-code", root=source_root, layout=export_drop_layout((".jsonl",))),),
            cursor=CursorStore(archive_root / "index.db"),
            write_coordinator=coordinator,
            sqlite_capture_stage=LiveSQLiteCaptureStage(compute_adapter=compute),
            append_runner=raw_owner.ingest_append_plans,
            convergence_runner=raw_owner.run_convergence_sync,
            retained_runner=raw_owner.ingest_retained_raw_ids,
        )
        # A page that only defers pending preparation never reaches the
        # write; offer fresh pages until one does (the kill ends the loop).
        for _ in range(20):
            adapter = FileIntakeAdapter(
                DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=watcher._sources),
                watcher._sources[0],
            )
            outcomes = await adapter.admit_page(await adapter.discover(limit=8))
            if {result.outcome for result in outcomes.values()} != {AdmissionOutcome.DEFERRED}:
                break
        sys.stdout.write("survived\\n")

    asyncio.run(main())
    """
)


def _message(index: int) -> dict[str, object]:
    role = "user" if index % 2 == 0 else "assistant"
    return {
        "type": role,
        "uuid": f"message-{index}",
        "parentUuid": f"message-{index - 1}" if index else None,
        "sessionId": _SESSION_ID,
        "timestamp": f"2026-07-10T00:00:{index:02d}Z",
        "message": {
            "role": role,
            "content": f"turn {index}" if role == "user" else [{"type": "text", "text": f"turn {index}"}],
        },
    }


def _append(path: Path, indexes: range) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for index in indexes:
            handle.write(json.dumps(_message(index)) + "\n")


async def _admit(archive_root: Path, source_root: Path) -> dict[str, Any]:
    """Admit the source through the production intake route until it is attempted.

    A page can defer a path whose off-writer preparation is not ready (the
    dispatcher re-offers it); deferral is not the outcome under test, so a
    fresh page is offered until the item gets a verdict.
    """
    pages = await _admit_pages(archive_root, source_root, pages=_MAX_DEFERRED_PAGES, stop_on_verdict=True)
    outcomes = pages[-1]
    if {result.outcome for result in outcomes.values()} == {AdmissionOutcome.DEFERRED}:
        raise AssertionError(f"the source stayed deferred for {_MAX_DEFERRED_PAGES} pages: {outcomes}")
    return outcomes


async def _admit_pages(
    archive_root: Path,
    source_root: Path,
    *,
    pages: int,
    stop_on_verdict: bool = False,
) -> list[dict[str, Any]]:
    """Offer ``pages`` fresh pages through the production intake route."""
    archive = Polylogue(archive_root=archive_root, db_path=archive_root / "index.db")
    async with live_owner_set(archive_root) as owners:
        watcher = LiveWatcher(
            archive,
            (WatchSource(name="claude-code", root=source_root, layout=export_drop_layout((".jsonl",))),),
            cursor=CursorStore(archive_root / "index.db"),
            **owners.watcher_kwargs(),
        )
        try:
            observed: list[dict[str, Any]] = []
            for _ in range(pages):
                adapter = FileIntakeAdapter(
                    DaemonIntakeContext(archive_root=archive_root, watcher=watcher, sources=watcher._sources),
                    watcher._sources[0],
                )
                observed.append(dict(await adapter.admit_page(await adapter.discover(limit=8))))
                if stop_on_verdict and {result.outcome for result in observed[-1].values()} != {
                    AdmissionOutcome.DEFERRED
                }:
                    break
            return observed
        finally:
            watcher.stop()
            await archive.close()


def _kill_child_during_write(archive_root: Path, source_root: Path, *, kill_at: int, log_dir: Path) -> None:
    # Files, not pipes: any worker process the child starts inherits its
    # descriptors, so a pipe would stay open after the child dies.
    stdout_path = log_dir / "child.stdout"
    stderr_path = log_dir / "child.stderr"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(archive_root), str(source_root), str(kill_at)],
            stdout=stdout,
            stderr=stderr,
            env=dict(os.environ),
            start_new_session=True,
        )
        try:
            child.wait(timeout=240)
        finally:
            # No worker may outlive the killed process and keep writing.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    out = stdout_path.read_text(errors="replace")
    assert child.returncode == -signal.SIGKILL, (
        f"child was not killed at the chosen write (rc={child.returncode}):\n{out}\n"
        f"{stderr_path.read_text(errors='replace')[-4000:]}"
    )
    assert "killing" in out


def _message_rows(archive_root: Path) -> tuple[int, int, int]:
    with sqlite3.connect(archive_root / "index.db") as conn:
        total, distinct = conn.execute(
            "SELECT count(*), count(DISTINCT message_id) FROM messages WHERE session_id = ?",
            (f"claude-code-session:{_SESSION_ID}",),
        ).fetchone()
        sessions = conn.execute(
            "SELECT count(*) FROM sessions WHERE session_id = ?", (f"claude-code-session:{_SESSION_ID}",)
        ).fetchone()[0]
    return int(total), int(distinct), int(sessions)


def _raw_parse_errors(archive_root: Path, source_path: Path) -> list[str]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return [
            str(row[0])
            for row in conn.execute(
                "SELECT parse_error FROM raw_sessions WHERE source_path = ? AND parse_error IS NOT NULL",
                (str(source_path),),
            )
        ]


@pytest.mark.asyncio
async def test_sigkill_inside_the_first_index_write_recovers_exactly(workspace_env: dict[str, Path]) -> None:
    """Anti-vacuity: a recovery that re-applies rows the killed process had
    already committed, without idempotent identity, shows up as ``total >
    distinct`` or ``total > 6``; one that trusts a cursor or raw state
    advanced before the lost commit shows up as ``total < 6`` (the file is
    skipped as already admitted). The child's exit status proves the kill
    happened at the patched write rather than after the pass finished."""
    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    source_path = source_root / "session.jsonl"
    log_dir = workspace_env["state_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    _append(source_path, range(0, 6))
    _kill_child_during_write(archive_root, source_root, kill_at=1, log_dir=log_dir)
    assert _message_rows(archive_root) == (0, 0, 0)

    outcomes = await _admit(archive_root, source_root)
    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.ADMITTED}, outcomes

    assert _message_rows(archive_root) == (6, 6, 1)
    assert _raw_parse_errors(archive_root, source_path) == []


@pytest.mark.asyncio
async def test_sigkill_inside_an_append_index_write_recovers_exactly(
    workspace_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kill inside the append's index write loses no appended message.

    While the restart's tail preparation is deferred, no page acknowledges it
    as DUPLICATE; once the retry is due the tail is admitted exactly, and
    later pages find nothing owed.

    Anti-vacuity (polylogue-b8of0): recovery that trusts the retained,
    never-parsed append raws as already admitted reports the restart page
    DUPLICATE and leaves the index at 3 messages; one that re-applies the
    committed prefix without idempotent identity shows ``total > distinct``.
    """
    archive_root = workspace_env["archive_root"]
    source_root = workspace_env["data_root"] / "claude-projects"
    source_root.mkdir(parents=True)
    source_path = source_root / "session.jsonl"
    log_dir = workspace_env["state_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    _append(source_path, range(0, 3))
    first = await _admit(archive_root, source_root)
    assert {result.outcome for result in first.values()} == {AdmissionOutcome.ADMITTED}, first
    assert _message_rows(archive_root) == (3, 3, 1)

    _append(source_path, range(3, 6))
    _kill_child_during_write(archive_root, source_root, kill_at=1, log_dir=log_dir)
    assert _message_rows(archive_root) == (3, 3, 1)

    # Make the restart's pages fail the tail's preparation. Preparation is
    # never abandoned on a deadline, so the failure comes from the event that
    # still produces one: the raw owner's preparation fails retryably (a
    # worker loss). The restart re-observes the killed file through the full
    # retained route, and every route prepares through the same derivation.
    # Until a retry publishes, the tail is owed work, so no page may
    # acknowledge it as DUPLICATE. Before the fix the second page did, and the
    # appended messages were never materialized.
    original_preparation = RawObservationDerivation.compute

    def failing_preparation(*args: object, **kwargs: object) -> RawObservationReplacement:
        raise RetainedPreparationRetryableError("synthetic preparation worker failure")

    monkeypatch.setattr(RawObservationDerivation, "compute", failing_preparation)
    restart_pages = await _admit_pages(archive_root, source_root, pages=3)
    assert all(
        result.outcome in {AdmissionOutcome.DEFERRED, AdmissionOutcome.RETRYABLE}
        for page in restart_pages
        for result in page.values()
    ), restart_pages
    assert _message_rows(archive_root) == (3, 3, 1)

    # Restore the preparation and make the scheduled retry due at once so the
    # next pages reach it through the owner's own route.
    monkeypatch.setattr(RawObservationDerivation, "compute", original_preparation)
    monkeypatch.setattr(cursor_module, "_FULL_CURSOR_RECONCILIATION_RETRY_DELAY_S", 0)
    CursorStore(archive_root / "index.db").defer_full_cursor_reconciliation(source_path)

    outcomes = await _admit(archive_root, source_root)
    # The due retry publishes the retained tail inside this page; the page's
    # own selection may then find nothing new and acknowledge it. Either way
    # the acknowledgement follows materialization, never precedes it.
    assert {result.outcome for result in outcomes.values()} <= {
        AdmissionOutcome.ADMITTED,
        AdmissionOutcome.DUPLICATE,
    }, outcomes
    assert _message_rows(archive_root) == (6, 6, 1)
    assert _raw_parse_errors(archive_root, source_path) == []

    # A reconciliation retry left owed by the deferral may admit the unchanged
    # file once more; that admission is idempotent, and then nothing is owed.
    settled = await _admit_pages(archive_root, source_root, pages=2)
    assert _message_rows(archive_root) == (6, 6, 1)
    assert {result.outcome for result in settled[-1].values()} <= {AdmissionOutcome.DUPLICATE}, settled
