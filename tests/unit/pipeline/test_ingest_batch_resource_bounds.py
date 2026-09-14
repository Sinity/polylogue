"""Resource-boundary tests for ingest worker handoff."""

from __future__ import annotations

from concurrent.futures import Future
from multiprocessing.process import BaseProcess
from types import SimpleNamespace

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_batch import _IngestWorkerRequest, _iter_ingest_results_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
from polylogue.sinex.models import PublicationMode
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.runtime import RawSessionRecord


def _large_raw_record() -> RawSessionRecord:
    return RawSessionRecord(
        raw_id="raw-large",
        source_name="codex",
        source_path="/tmp/raw-large.jsonl",
        blob_size=150 * 1024 * 1024,
        acquired_at="2026-04-02T00:00:00Z",
    )


def _worker_request() -> _IngestWorkerRequest:
    return _IngestWorkerRequest(
        archive_root_str="/tmp/archive",
        blob_root_str="/tmp/blob-store",
        validation_mode="strict",
        measure_ingest_result_size=False,
    )


def _session_data_with_rows(*, session_id: str = "codex:conv-large", messages: int = 0) -> SessionWritePayload:
    parsed_messages = [
        ParsedMessage(provider_message_id=f"msg-{index}", role=Role.USER, text="payload") for index in range(messages)
    ]
    parsed = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id=session_id.split(":", 1)[-1],
        title="Large",
        messages=parsed_messages,
    )
    return SessionWritePayload(
        session_id=session_id,
        content_hash="0" * 64,
        parsed_session=parsed,
        message_count=messages,
        raw_id="raw-large",
    )


def test_iter_ingest_results_sync_can_isolate_single_worker_in_process_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_artifacts = [_large_raw_record()]
    submitted: list[str] = []

    class FakeExecutor:
        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def submit(
            self,
            fn: object,
            raw_record: RawSessionRecord,
            request: _IngestWorkerRequest,
        ) -> Future[IngestRecordResult]:
            del fn, request
            submitted.append(raw_record.raw_id)
            future: Future[IngestRecordResult] = Future()
            future.set_result(IngestRecordResult(raw_id=raw_record.raw_id))
            return future

    def fake_process_pool_executor(*, max_workers: int) -> FakeExecutor:
        assert max_workers == 1
        return FakeExecutor()

    monkeypatch.setattr(ingest_batch_core, "process_pool_executor", fake_process_pool_executor)

    results = list(
        _iter_ingest_results_sync(
            raw_artifacts,
            request=_worker_request(),
            worker_count=1,
            force_process_pool=True,
        )
    )

    assert submitted == ["raw-large"]
    assert [result.raw_id for result in results] == ["raw-large"]


def test_consume_ingest_results_delays_write_transaction_until_parse_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeConnection:
        def execute(self, sql: str) -> object:
            if sql == "BEGIN IMMEDIATE":
                events.append("begin")
            return None

    def fake_iter(*args: object, **kwargs: object) -> list[IngestRecordResult]:
        del args, kwargs
        events.append("parse-drained")
        return [IngestRecordResult(raw_id="raw-large")]

    def fake_drain(*args: object, **kwargs: object) -> None:
        del args
        ensure_transaction = kwargs.get("ensure_index_transaction")
        assert callable(ensure_transaction)
        ensure_transaction()
        events.append("drain")

    monkeypatch.setattr(ingest_batch_core, "_iter_ingest_results_sync", fake_iter)
    monkeypatch.setattr(ingest_batch_core, "_drain_ingest_result", fake_drain)

    summary = SimpleNamespace(result_wait_s=0.0, teardown_elapsed_s=0.0, worker_count=1)
    transaction_started = ingest_batch_core._consume_ingest_results(
        FakeConnection(),  # type: ignore[arg-type]
        [_large_raw_record()],
        worker_request=_worker_request(),
        summary=summary,  # type: ignore[arg-type]
        materialized_ids=set(),
        publication_mode=PublicationMode.OFF,
    )

    assert transaction_started is True
    assert events == ["parse-drained", "begin", "drain"]


def test_consume_ingest_results_releases_large_result_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdata = _session_data_with_rows(messages=1001)
    result = IngestRecordResult(raw_id="raw-large", sessions=[cdata])
    releases: list[str] = []

    class FakeConnection:
        def execute(self, sql: str) -> object:
            assert sql == "BEGIN IMMEDIATE"
            return None

    monkeypatch.setattr(ingest_batch_core, "_iter_ingest_results_sync", lambda *args, **kwargs: [result])

    def fake_drain(*args: object, **kwargs: object) -> None:
        del args
        ensure_transaction = kwargs.get("ensure_index_transaction")
        assert callable(ensure_transaction)
        ensure_transaction()

    monkeypatch.setattr(ingest_batch_core, "_drain_ingest_result", fake_drain)
    monkeypatch.setattr(ingest_batch_core, "release_process_memory", lambda: releases.append("release"))
    monkeypatch.setattr(ingest_batch_core, "read_current_rss_mb", lambda: 42.0)

    summary = SimpleNamespace(
        result_wait_s=0.0,
        teardown_elapsed_s=0.0,
        worker_count=1,
        max_current_rss_mb=None,
    )
    transaction_started = ingest_batch_core._consume_ingest_results(
        FakeConnection(),  # type: ignore[arg-type]
        [_large_raw_record()],
        worker_request=_worker_request(),
        summary=summary,  # type: ignore[arg-type]
        materialized_ids=set(),
        publication_mode=PublicationMode.OFF,
    )

    assert transaction_started is True
    assert result.sessions == []
    assert releases == ["release"]
    assert summary.max_current_rss_mb == 42.0


def test_drain_ready_session_entries_drops_written_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdata = _session_data_with_rows(messages=3)
    writes: list[int] = []

    def fake_write(*args: object, **kwargs: object) -> bool:
        del args, kwargs
        writes.append(cdata.message_count)
        return True

    monkeypatch.setattr(ingest_batch_core, "_write_session_entry", fake_write)
    # The drain prelude removes stale session rows; that is outside this
    # payload-release contract.
    monkeypatch.setattr(ingest_batch_core, "_delete_stale_sessions_for_raw_entries", lambda *_a, **_k: None)

    ingest_batch_core._drain_ready_session_entries(
        object(),  # type: ignore[arg-type]
        [("raw-large", cdata)],
        summary=SimpleNamespace(),  # type: ignore[arg-type]
        materialized_ids=set(),
    )

    assert writes == [3]
    assert cdata.parsed_session.messages == []


def _block_until_killed(*_args: object, **_kwargs: object) -> IngestRecordResult:
    """A worker body that outlives the progress deadline, as a stall does."""
    import time as _time

    _time.sleep(600)
    return IngestRecordResult(raw_id="unreachable")


def test_stalled_ingest_pool_terminates_its_running_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled pass reclaims its worker processes instead of orphaning them.

    ``Future.cancel()`` cannot stop a task already executing in a
    ``ProcessPoolExecutor``, and neither can ``shutdown(cancel_futures=True)``.
    Before the fix every stalled pass left its workers running, and repeated
    passes accumulated them for the life of the daemon.

    Anti-vacuity: the assertion is on real OS process liveness, not on a
    recorded call. Remove the ``terminate_process_pool`` call from the stall
    branch and the worker pids are still alive when the generator is
    exhausted, so ``still_alive`` is non-empty and the test goes red. The
    refusal assertions keep the fix from being "kill the pool and report
    success" -- the stalled items must still surface as retryable refusals.
    """
    from concurrent.futures import ProcessPoolExecutor

    from polylogue.pipeline.services.process_pool import process_pool_context

    monkeypatch.setattr(ingest_batch_core, "_INGEST_RESULT_PROGRESS_DEADLINE_S", 1.0)

    worker_processes: list[BaseProcess] = []

    class BlockingExecutor(ProcessPoolExecutor):
        """A real pool whose submitted work never finishes."""

        def submit(self, fn: object, *args: object, **kwargs: object) -> Future[IngestRecordResult]:  # type: ignore[override]
            del fn, args, kwargs
            future = super().submit(_block_until_killed)
            # Capture the workers now: a terminated pool clears ``_processes``
            # on shutdown, so reading it afterwards would prove nothing.
            worker_processes.extend((getattr(self, "_processes", None) or {}).values())
            return future

    def fake_process_pool_executor(*, max_workers: int) -> BlockingExecutor:
        return BlockingExecutor(max_workers=max_workers, mp_context=process_pool_context())

    monkeypatch.setattr(ingest_batch_core, "process_pool_executor", fake_process_pool_executor)

    results = list(
        _iter_ingest_results_sync(
            [_large_raw_record()],
            request=_worker_request(),
            worker_count=1,
            force_process_pool=True,
        )
    )

    assert [result.raw_id for result in results] == ["raw-large"]
    assert results[0].retryable is True
    assert "progress deadline exceeded" in (results[0].error or "")

    assert worker_processes, "the pool must have started at least one real worker"
    still_alive = [process.pid for process in worker_processes if process.is_alive()]
    assert still_alive == []
