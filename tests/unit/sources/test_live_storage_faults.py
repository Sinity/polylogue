"""Archive storage faults on the live intake route leave inputs retryable.

A full disk, I/O error or corrupt page fails every write alike. The page must
be refused with a typed reason, and the input must keep no failure mark: no
failed cursor backing off toward quarantine, no ``parse_error`` on its raw,
and no attempt row stuck in ``running``. Once storage recovers the same input
is admitted.
"""

from __future__ import annotations

import errno
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from polylogue import Polylogue
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.logging import capture
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.operations.operation_context import open_operation_read
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveRawParsedWriteResult


def _write_session(path: Path, session_id: str) -> None:
    record = {
        "type": "user",
        "uuid": "message-1",
        "parentUuid": None,
        "sessionId": session_id,
        "timestamp": "2026-07-10T00:00:00Z",
        "message": {"role": "user", "content": "kept through a full disk"},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def _raw_parse_states(archive_root: Path, source_path: Path) -> list[tuple[int | None, str | None]]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT parsed_at_ms, parse_error FROM raw_sessions WHERE source_path = ? ORDER BY source_index",
                (str(source_path),),
            )
        ]


def _latest_attempt_id(watcher: LiveWatcher) -> str:
    with sqlite3.connect(watcher._cursor._ops_db_path) as conn:
        row = conn.execute(
            "SELECT attempt_id FROM ingest_attempts ORDER BY started_at_ms DESC, rowid DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    return str(row[0])


def _latest_attempt(watcher: LiveWatcher) -> tuple[str, str, str | None]:
    with sqlite3.connect(watcher._cursor._ops_db_path) as conn:
        row = conn.execute(
            "SELECT status, outcome_code, evidence_ref FROM ingest_attempts ORDER BY started_at_ms DESC, rowid DESC LIMIT 1"
        ).fetchone()
    assert row is not None
    return str(row[0]), str(row[1]), row[2]


async def _admit(watcher: LiveWatcher) -> dict[str, Any]:
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
    )
    return dict(await adapter.admit_page(await adapter.discover(limit=8)))


def _full_disk() -> sqlite3.OperationalError:
    exc = sqlite3.OperationalError("database or disk is full")
    exc.sqlite_errorcode = sqlite3.SQLITE_FULL
    return exc


def _fail_first_index_write(monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> None:
    original = archive_revision_governance._write_parsed_precedence_result
    calls = 0

    def fail_once(*args: Any, **kwargs: Any) -> ArchiveRawParsedWriteResult:
        # Signature-agnostic on purpose: a drifted stub raising TypeError would
        # be classified as an ordinary per-file failure and hide the guard.
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(archive_revision_governance, "_write_parsed_precedence_result", fail_once)


def _fail_first_blob_copy(monkeypatch: pytest.MonkeyPatch, failure: OSError) -> None:
    original = ArchiveBlobPublisher.write_from_path
    calls = 0

    def fail_once(self: ArchiveBlobPublisher, *args: Any, **kwargs: Any) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_path", fail_once)


@pytest.fixture
def storage_env(workspace_env: dict[str, Path]) -> tuple[Polylogue, LiveWatcher, Path]:
    root = workspace_env["data_root"] / "claude-projects"
    root.mkdir(parents=True)
    source_path = root / "session.jsonl"
    _write_session(source_path, "storage-fault")
    archive = Polylogue(
        archive_root=workspace_env["archive_root"],
        db_path=workspace_env["archive_root"] / "index.db",
    )
    watcher = LiveWatcher(
        archive,
        (WatchSource(name="claude-code", root=root),),
        cursor=CursorStore(archive.archive_root / "index.db"),
        # The daemon's read route; without it off-writer preparation defers
        # every full-route file and nothing reaches the index write.
        read_snapshot=open_operation_read,
    )
    return archive, watcher, source_path


async def _assert_recovers(archive: Polylogue, watcher: LiveWatcher, source_path: Path) -> None:
    outcomes = await _admit(watcher)
    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.ADMITTED}
    session = await archive.get_session("claude-code-session:storage-fault")
    assert session is not None
    assert [message.text for message in session.messages] == ["kept through a full disk"]


@pytest.mark.asyncio
async def test_index_write_storage_fault_refuses_page_without_marking_input(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anti-vacuity: without the storage-fault escape the batch's generic
    handler marks the cursor failed (``failure_count == 1``, backing off toward
    quarantine) and records a durable ``parse_error`` on the raw."""
    archive, watcher, source_path = storage_env
    _fail_first_index_write(monkeypatch, _full_disk())
    try:
        with capture() as events:
            outcomes = await _admit(watcher)

        assert outcomes, "the page must report every item"
        for result in outcomes.values():
            assert result.outcome is AdmissionOutcome.RETRYABLE
            assert "storage fault (capacity)" in str(result.reason)
        cursor = watcher._cursor.get_record(source_path)
        assert cursor is None or (cursor.failure_count == 0 and not cursor.excluded)
        assert all(error is None for _parsed, error in _raw_parse_states(archive.archive_root, source_path))
        assert _latest_attempt(watcher) == ("failed", "transient_error", "archive_write:storage_fault:capacity")
        refusals = [event for event in events if event.get("event") == "daemon.intake.page_refused"]
        assert [(event.get("level"), event.get("reason")) for event in refusals] == [
            ("error", "storage_fault.capacity")
        ]

        await _assert_recovers(archive, watcher, source_path)
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_blob_copy_enospc_refuses_page_without_marking_input(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENOSPC can only come from the archive side of the copy. Anti-vacuity:
    without the escape the copy failure is recorded as this file's failure
    (``failure_count == 1``)."""
    archive, watcher, source_path = storage_env
    _fail_first_blob_copy(monkeypatch, OSError(errno.ENOSPC, "No space left on device"))
    try:
        outcomes = await _admit(watcher)
        assert outcomes
        assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
        cursor = watcher._cursor.get_record(source_path)
        assert cursor is None or cursor.failure_count == 0

        await _assert_recovers(archive, watcher, source_path)
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_blob_copy_eio_remains_the_files_own_failure(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EIO during a copy may be the source read, so it stays per-file.
    Anti-vacuity: dropping the ``ARCHIVE_SIDE_FAULTS`` narrowing refuses the
    page instead and leaves no failure on the cursor."""
    archive, watcher, source_path = storage_env
    _fail_first_blob_copy(monkeypatch, OSError(errno.EIO, "Input/output error"))
    try:
        await _admit(watcher)
        cursor = watcher._cursor.get_record(source_path)
        assert cursor is not None
        assert cursor.failure_count == 1
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_events_inside_an_ingest_attempt_carry_its_attempt_id(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
) -> None:
    """The attempt's chunk event joins to its ``ingest_attempts`` row by
    ``attempt_id``. Anti-vacuity: without the attempt-scoped ``bind`` the chunk
    event carries no ``attempt_id`` (and before the field was registered, the
    validator dropped it)."""
    archive, watcher, _source_path = storage_env
    try:
        with capture() as events:
            outcomes = await _admit(watcher)
        assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.ADMITTED}
        attempt_id = _latest_attempt_id(watcher)
        chunks = [event for event in events if event.get("event") == "live.ingest.chunk"]
        assert chunks
        assert {event.get("attempt_id") for event in chunks} == {attempt_id}
        assert not [
            event
            for event in events
            if event.get("event") == "log.field_rejected" and event.get("source_event") == "live.ingest.chunk"
        ]
    finally:
        watcher.stop()
        await archive.close()
