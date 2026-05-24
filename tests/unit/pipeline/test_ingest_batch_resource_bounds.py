"""Resource-boundary tests for ingest worker handoff."""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.pipeline.services.ingest_batch import _IngestWorkerRequest, _iter_ingest_results_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.storage.runtime import RawConversationRecord


def _large_raw_record() -> RawConversationRecord:
    return RawConversationRecord(
        raw_id="raw-large",
        provider_name="codex",
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
            raw_record: RawConversationRecord,
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
        del args, kwargs
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
        pending_by_parent={},
    )

    assert transaction_started is True
    assert events == ["parse-drained", "begin", "drain"]
