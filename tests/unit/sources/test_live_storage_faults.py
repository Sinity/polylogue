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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue import Polylogue
from polylogue.core.compute import DaemonBackpressureError
from polylogue.daemon.intake import AdmissionOutcome
from polylogue.logging import capture
from polylogue.operations.intake_adapters import DaemonIntakeContext, FileIntakeAdapter
from polylogue.sources.live import LiveWatcher, WatchSource
from polylogue.sources.live.cursor import CursorStore
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.sqlite.archive_tiers import revision_governance as archive_revision_governance
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveRawParsedWriteResult
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.raw_owner_routes import live_owner_set


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
    # Captures stream through the acquisition boundary into write_from_writer.
    original = ArchiveBlobPublisher.write_from_writer
    calls = 0

    def fail_once(self: ArchiveBlobPublisher, *args: Any, **kwargs: Any) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_writer", fail_once)


@pytest.fixture
async def storage_env(workspace_env: dict[str, Path]) -> AsyncIterator[tuple[Polylogue, LiveWatcher, Path]]:
    root = workspace_env["data_root"] / "claude-projects"
    project = root / "-home-user-repo"
    project.mkdir(parents=True)
    source_path = project / "session.jsonl"
    _write_session(source_path, "storage-fault")
    archive = Polylogue(
        archive_root=workspace_env["archive_root"],
        db_path=workspace_env["archive_root"] / "index.db",
    )
    # The daemon's intake owners; without them full-route preparation never
    # publishes and nothing reaches the index write.
    async with live_owner_set(archive.archive_root) as owners:
        watcher = LiveWatcher(
            archive,
            (WatchSource(name="claude-code", root=root),),
            cursor=CursorStore(archive.archive_root / "index.db"),
            **owners.watcher_kwargs(),
        )
        yield archive, watcher, source_path


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
async def test_index_write_compute_backpressure_stays_retryable_and_recovers(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compute admission refusal leaves the retained source retryable and later indexable."""
    archive, watcher, source_path = storage_env
    _fail_first_index_write(
        monkeypatch,
        DaemonBackpressureError("daemon compute admission is saturated; retry shortly"),
    )
    try:
        outcomes = await _admit(watcher)

        assert outcomes
        assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
        cursor = watcher._cursor.get_record(source_path)
        assert cursor is None or (cursor.failure_count == 0 and not cursor.excluded)
        assert all(error is None for _parsed, error in _raw_parse_states(archive.archive_root, source_path))
        assert _latest_attempt(watcher) == (
            "failed",
            "transient_error",
            "archive_write:DaemonBackpressureError",
        )

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


@pytest.mark.asyncio
async def test_a_cancelled_attempt_is_named_and_cancellation_propagates(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation (shutdown) admits no further ops write, but the attempt
    it interrupted is named at once. Anti-vacuity: without the explicit
    ``CancelledError`` branch no ``live.ingest.attempt_cancelled`` event is
    emitted; swallowing it instead fails ``pytest.raises``."""
    import asyncio

    from polylogue.sources.live.batch import LiveBatchProcessor

    archive, watcher, source_path = storage_env

    async def cancelled_mid_attempt(self: LiveBatchProcessor, paths: list[Path], **kwargs: Any) -> Any:
        kwargs["open_attempt"].opened("attempt-under-cancellation")
        raise asyncio.CancelledError

    monkeypatch.setattr(LiveBatchProcessor, "_ingest_files", cancelled_mid_attempt)
    try:
        with capture() as events, pytest.raises(asyncio.CancelledError):
            await watcher._batch_processor.ingest_files([source_path])
        cancelled = [event for event in events if event.get("event") == "live.ingest.attempt_cancelled"]
        assert [event.get("attempt_id") for event in cancelled] == ["attempt-under-cancellation"]
    finally:
        watcher.stop()
        await archive.close()


def test_zip_member_publication_on_a_full_archive_escapes_instead_of_excluding(tmp_path: Path) -> None:
    """A ZIP whose members stream into a full archive is not "a ZIP with no
    admissible record". Anti-vacuity: without the escape both helpers turn
    ENOSPC into an empty/None extraction, and the caller acknowledges the
    unchanged ZIP as excluded."""
    import zipfile
    from types import SimpleNamespace
    from typing import cast

    from polylogue.core.enums import Provider
    from polylogue.core.storage_faults import ArchiveStorageFaultError
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.storage.blob_store import BlobStore

    bundle = tmp_path / "claude-export.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "projects/project/session.jsonl",
            b'{"parentUuid":null,"type":"user","message":{"role":"user","content":"kept"},'
            b'"uuid":"u1","timestamp":"2025-01-01T00:00:00Z"}\n',
        )
    index_db = tmp_path / "index.db"
    bootstrap_archive_root(tmp_path)
    processor = LiveBatchProcessor(
        cast(Any, SimpleNamespace(archive_root=tmp_path, backend=SimpleNamespace(db_path=index_db))),
        (WatchSource(name="claude-code", root=tmp_path),),
        cursor=CursorStore(index_db),
        parser_fingerprint="test-parser",
    )

    class _FullBlobStore(BlobStore):
        def prepare_from_fileobj(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

        def write_from_fileobj(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

        def prepare_from_writer(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

        def write_from_writer(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError(errno.ENOSPC, "No space left on device")

    full = _FullBlobStore(tmp_path / "blob")
    # The eager member route is retired; source-only extraction is the single
    # remaining ZIP member route and must still type the full-disk refusal.
    with pytest.raises(ArchiveStorageFaultError):
        processor._extract_source_only_zip_member_records(
            bundle,
            blob_store=full,
            fallback_provider=Provider.CLAUDE_CODE,
            file_mtime="2026-09-04T00:00:00+00:00",
            zip_inputs={},
        )


@pytest.mark.asyncio
async def test_sqlite_contention_closes_the_attempt_as_retryable(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real SQLite contention failure closes its attempt as retryable."""
    from polylogue.sources.live.batch import LiveBatchProcessor

    archive, watcher, source_path = storage_env
    original = LiveBatchProcessor._ingest_files

    async def spend_hold(self: LiveBatchProcessor, paths: list[Path], **kwargs: Any) -> Any:
        attempt = kwargs["open_attempt"]
        attempt_id = await self._run_ops_write(
            "attempt_start", self._cursor.begin_ingest_attempt, paths=paths, input_bytes=0, queued_file_count=1
        )
        attempt.opened(attempt_id)
        attempt.started = True
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(LiveBatchProcessor, "_ingest_files", spend_hold)
    try:
        with pytest.raises(sqlite3.OperationalError):
            await watcher._batch_processor.ingest_files([source_path])
        with sqlite3.connect(watcher._cursor._ops_db_path) as conn:
            row = conn.execute(
                "SELECT status, outcome_code, retryable, evidence_ref FROM ingest_attempts "
                "ORDER BY started_at_ms DESC, rowid DESC LIMIT 1"
            ).fetchone()
        assert tuple(row) == ("failed", "transient_error", 1, "archive_write:OperationalError")
    finally:
        monkeypatch.setattr(LiveBatchProcessor, "_ingest_files", original)
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_append_storage_fault_leaves_the_append_raw_unmarked(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The append index write can record a failure state on the raw before
    the fault reaches the append handler; the handler resets it, as it does
    for contention. Anti-vacuity: without the reset the append raw keeps a
    ``parse_error`` naming the full disk."""
    archive, watcher, source_path = storage_env
    try:
        first = await _admit(watcher)
        assert {result.outcome for result in first.values()} == {AdmissionOutcome.ADMITTED}
        with source_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "assistant",
                        "uuid": "message-2",
                        "parentUuid": "message-1",
                        "sessionId": "storage-fault",
                        "timestamp": "2026-07-10T00:00:01Z",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "appended"}]},
                    }
                )
                + "\n"
            )
        _fail_first_index_write(monkeypatch, _full_disk())
        outcomes = await _admit(watcher)
        assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
        assert all(error is None for _parsed, error in _raw_parse_states(archive.archive_root, source_path))
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_an_attempt_close_skipped_under_lock_is_reported(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``finish_ingest_attempt`` gives up quietly (returns ``False``) when the
    ops tier stays locked; the closer must say the row is still running.
    Anti-vacuity: treating every non-raising call as closed emits nothing."""
    from polylogue.sources.live.batch import LiveBatchProcessor

    archive, watcher, source_path = storage_env

    async def spend_hold(self: LiveBatchProcessor, paths: list[Path], **kwargs: Any) -> Any:
        attempt_id = await self._run_ops_write(
            "attempt_start", self._cursor.begin_ingest_attempt, paths=paths, input_bytes=0, queued_file_count=1
        )
        kwargs["open_attempt"].opened(attempt_id)
        kwargs["open_attempt"].started = True
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(LiveBatchProcessor, "_ingest_files", spend_hold)
    monkeypatch.setattr(CursorStore, "finish_ingest_attempt", lambda self, *args, **kwargs: False)
    try:
        with capture() as events, pytest.raises(sqlite3.OperationalError):
            await watcher._batch_processor.ingest_files([source_path])
        skipped = [event for event in events if event.get("event") == "live.ingest.attempt_finish_failed"]
        assert [event.get("reason") for event in skipped] == ["ops_write_skipped"]
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_a_storage_fault_discards_blobs_staged_earlier_in_the_pass(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first file's blob is staged, the second copy hits ENOSPC. The
    staged copy would only be published by the write the fault prevented, so
    it must not stay behind in ``.staging``. Anti-vacuity: without the
    discard the first file's temporary remains after every retry."""
    archive, watcher, source_path = storage_env
    _write_session(source_path.parent / "second.jsonl", "storage-fault-2")
    # Captures stream through the acquisition boundary into write_from_writer.
    original = ArchiveBlobPublisher.write_from_writer
    calls = 0

    def fail_second(self: ArchiveBlobPublisher, *args: Any, **kwargs: Any) -> tuple[str, int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ArchiveBlobPublisher, "write_from_writer", fail_second)
    try:
        outcomes = await _admit(watcher)
        assert calls == 2
        assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
        staging = archive.archive_root / "blob" / ".staging"
        assert not staging.exists() or not [path for path in staging.rglob("*") if path.is_file()]
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_a_cancelled_start_still_names_its_attempt(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation can land while the attempt-start write is detached and
    still committing; the key is chosen first, so the event names the row.
    Anti-vacuity: taking the key from the write's return value leaves
    ``attempt_id`` unset and no event is emitted."""
    import asyncio

    from polylogue.sources.live.batch import LiveBatchProcessor

    archive, watcher, source_path = storage_env
    original = LiveBatchProcessor._run_ops_write

    async def cancel_at_start(self: LiveBatchProcessor, label: str, *args: Any, **kwargs: Any) -> Any:
        if label == "attempt_start":
            raise asyncio.CancelledError
        return await original(self, label, *args, **kwargs)

    monkeypatch.setattr(LiveBatchProcessor, "_run_ops_write", cancel_at_start)
    try:
        with capture() as events, pytest.raises(asyncio.CancelledError):
            await watcher._batch_processor.ingest_files([source_path])
        cancelled = [event for event in events if event.get("event") == "live.ingest.attempt_cancelled"]
        assert len(cancelled) == 1
        assert cancelled[0].get("attempt_id")
        # The start never returned, so the event must not claim the row exists.
        assert cancelled[0].get("reason") == "cancelled_before_start_confirmed"
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_a_raw_fault_from_the_publication_flush_discards_staged_blobs(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flush raises a raw ``SQLITE_FULL`` that is converted only later;
    staged temporaries must still be discarded. Anti-vacuity: cleaning up
    only on ``ArchiveStorageFaultError`` leaves the staged copy behind."""
    archive, watcher, source_path = storage_env

    def full_flush(self: ArchiveBlobPublisher, *args: Any, **kwargs: Any) -> Any:
        raise _full_disk()

    monkeypatch.setattr(ArchiveBlobPublisher, "flush", full_flush)
    try:
        outcomes = await _admit(watcher)
        assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
        staging = archive.archive_root / "blob" / ".staging"
        assert not staging.exists() or not [path for path in staging.rglob("*") if path.is_file()]
    finally:
        watcher.stop()
        await archive.close()


@pytest.mark.asyncio
async def test_degraded_daemon_admits_nothing_and_reads_no_authority(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production intake route short-circuits a structurally degraded daemon.

    Anti-vacuity: drop the degraded check from ``FileIntakeAdapter.discover``
    and degraded discovery returns a page; drop it from ``admit_page`` and the
    page reaches ``require_cursor_authority`` (which reads the archive's
    existence journals) and cursor initialization.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _archive, watcher, _source_path = storage_env

    def authority_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a degraded daemon must not run the source-selection gate")

    monkeypatch.setattr(watcher._batch_processor, "require_cursor_authority", authority_must_not_run)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
    )
    # A page discovered while healthy, then admitted after degradation.
    page = await adapter.discover(limit=8)
    assert page
    set_degraded(DegradedReason(code="schema_version_mismatch", message="v12 vs v9"))
    try:
        # Discovery reads the cursor tier; degraded, it returns no page at all.
        assert list(await adapter.discover(limit=8)) == []
        outcomes = dict(await adapter.admit_page(page))
        # The page stays unattempted: no failed-attempt cooldown after recovery.
        assert not adapter._fresh_attempted_paths
    finally:
        clear_degraded()

    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
    assert all("degraded" in (result.reason or "") for result in outcomes.values())
    # No admission work happened, so none is charged to the class deficit.
    assert {result.actual_cost for result in outcomes.values()} == {0}


@pytest.mark.asyncio
async def test_degraded_admission_keeps_a_due_retry_page_for_re_offer(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
) -> None:
    """Anti-vacuity: leave ``_retry_page_pending`` set on the degraded return and
    the next discovery rotates ``_retry_skip_after`` past the whole page, so its
    unplanned tail waits for the durable retry sweep to wrap."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _archive, watcher, source_path = storage_env
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
    )
    tail = source_path.with_name("tail.jsonl")
    adapter._retry_page = True
    adapter._retry_page_pending = True
    adapter._retry_page_paths = (source_path, tail)
    set_degraded(DegradedReason(code="schema_version_mismatch", message="v12 vs v9"))
    try:
        await adapter.admit_page(())
    finally:
        clear_degraded()

    assert adapter._retry_page_pending is False
    assert not adapter._retry_page_paths
    assert adapter._retry_skip_after is None


@pytest.mark.asyncio
async def test_degraded_admission_restores_due_local_retries(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
) -> None:
    """Anti-vacuity: keep the offer's pushed-forward deadline and the carrier
    reports a cooldown after recovery although nothing was attempted."""
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _archive, watcher, source_path = storage_env
    clock = {"now": 100.0}
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
        clock=lambda: clock["now"],
    )
    adapter._fresh_retry_debt[source_path] = 105.0  # pushed forward by the offer
    adapter._retry_page = True
    adapter._retry_page_pending = True
    adapter._local_retry_page = True
    adapter._retry_page_paths = (source_path,)
    set_degraded(DegradedReason(code="schema_version_mismatch", message="v12 vs v9"))
    try:
        await adapter.admit_page(())
    finally:
        clear_degraded()

    assert adapter._fresh_retry_debt[source_path] == 100.0


@pytest.mark.asyncio
async def test_degradation_during_admission_marks_items_unattempted(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degradation after the entry check still reports the page unattempted.

    Anti-vacuity: drop ``unattempted=True`` from the batch-metrics branches of
    ``admit_page`` and the dispatcher counts these items as failed attempts.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _archive, watcher, _source_path = storage_env
    real_ingest = watcher._ingest_files

    async def degrade_then_ingest(*args: Any, **kwargs: Any) -> Any:
        set_degraded(DegradedReason(code="database_layout_mismatch", message="structural error"))
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(watcher, "_ingest_files", degrade_then_ingest)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
    )
    page = await adapter.discover(limit=8)
    # Offer the page as a durable retry page so the retry position would move.
    adapter._retry_page = True
    adapter._retry_page_paths = tuple(Path(cast(Any, item.payload)) for item in page)
    retry_after_before = adapter._retry_after
    try:
        outcomes = dict(await adapter.admit_page(page))
    finally:
        clear_degraded()
    # Nothing was attempted, so the retry position does not advance.
    assert adapter._retry_after == retry_after_before

    assert outcomes
    assert {result.outcome for result in outcomes.values()} == {AdmissionOutcome.RETRYABLE}
    assert all(result.unattempted and result.actual_cost == 0 for result in outcomes.values())


@pytest.mark.asyncio
async def test_degradation_before_an_all_empty_page_still_marks_it_unattempted(
    storage_env: tuple[Polylogue, LiveWatcher, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded skip of zero-byte files refuses no bytes but attempted nothing.

    Anti-vacuity: derive the degraded skip from ``refused_bytes_by_reason``
    again and this page, whose offered bytes total zero, is reported attempted:
    ``unattempted`` is false and the retry position advances.
    """
    from polylogue.core.degraded import DegradedReason, clear_degraded, set_degraded

    _archive, watcher, source_path = storage_env
    source_path.write_bytes(b"")
    real_ingest = watcher._ingest_files

    async def degrade_then_ingest(*args: Any, **kwargs: Any) -> Any:
        set_degraded(DegradedReason(code="database_layout_mismatch", message="structural error"))
        return await real_ingest(*args, **kwargs)

    monkeypatch.setattr(watcher, "_ingest_files", degrade_then_ingest)
    adapter = FileIntakeAdapter(
        DaemonIntakeContext(
            archive_root=Path(watcher._polylogue.archive_root),
            watcher=watcher,
            sources=watcher._sources,
        ),
        watcher._sources[0],
    )
    page = await adapter.discover(limit=8)
    assert page
    assert all(Path(cast(Any, item.payload)).stat().st_size == 0 for item in page)
    adapter._retry_page = True
    adapter._retry_page_paths = tuple(Path(cast(Any, item.payload)) for item in page)
    retry_after_before = adapter._retry_after
    try:
        outcomes = dict(await adapter.admit_page(page))
    finally:
        clear_degraded()

    assert adapter._retry_after == retry_after_before
    assert outcomes
    assert all(result.unattempted and result.actual_cost == 0 for result in outcomes.values())
