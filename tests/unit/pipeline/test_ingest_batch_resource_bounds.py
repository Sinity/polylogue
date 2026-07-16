"""Resource-boundary tests for ingest worker handoff."""

from __future__ import annotations

from concurrent.futures import Future
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

    ingest_batch_core._drain_ready_session_entries(
        object(),  # type: ignore[arg-type]
        [("raw-large", cdata)],
        summary=SimpleNamespace(),  # type: ignore[arg-type]
        materialized_ids=set(),
    )

    assert writes == [3]
    assert cdata.parsed_session.messages == []
