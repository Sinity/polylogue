"""Resource-boundary tests for ingest worker handoff."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_batch import _IngestWorkerRequest, _iter_ingest_results_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload
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


def _raw_records(count: int) -> list[RawSessionRecord]:
    return [
        RawSessionRecord(
            raw_id=f"raw-{index}",
            source_name="codex",
            source_path=f"/tmp/raw-{index}.jsonl",
            blob_size=12,
            acquired_at="2026-04-02T00:00:00Z",
        )
        for index in range(1, count + 1)
    ]


def test_single_record_parse_uses_shared_compute_and_preserves_parser_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from polylogue.core.compute import BoundedComputeAdapter

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=1024)
    workers: list[int] = []

    def parse(record: RawSessionRecord, request: _IngestWorkerRequest) -> IngestRecordResult:
        del request
        workers.append(threading.get_ident())
        if record.raw_id == "raw-2":
            raise ValueError("synthetic parser fault")
        return IngestRecordResult(raw_id=record.raw_id, outcome_code="success")

    monkeypatch.setattr(ingest_batch_core, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(ingest_batch_core, "_run_ingest_record", parse)
    progress = ingest_batch_core._WorkerProgress()
    try:
        results = list(
            _iter_ingest_results_sync(
                _raw_records(3),
                request=_worker_request(),
                worker_count=1,
                progress=progress,
            )
        )
        assert [result.raw_id for result in results] == ["raw-1", "raw-2", "raw-3"]
        assert [result.outcome_code for result in results] == ["success", "parser_defect", "success"]
        assert results[1].retryable is False
        assert results[1].evidence_ref == "worker:ValueError"
        assert len(set(workers)) == 1
        assert workers[0] != threading.get_ident()
        assert progress.completed_raw_count == 3
        assert progress.in_flight_raw_ids == []
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)


def test_shared_admission_refusal_does_not_parse_or_drop_unaccepted_raws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from polylogue.core.compute import BoundedComputeAdapter

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=1024)
    entered = threading.Event()
    release = threading.Event()
    parsed: list[str] = []

    def hold() -> None:
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(ingest_batch_core, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(ingest_batch_core, "_run_ingest_record", lambda record, request: parsed.append(record.raw_id))
    try:
        occupied = adapter.submit(hold, admission_class="incremental-background")
        assert entered.wait(5)
        results = list(_iter_ingest_results_sync(_raw_records(3), request=_worker_request(), worker_count=1))
        assert parsed == []
        assert [result.raw_id for result in results] == ["raw-1", "raw-2", "raw-3"]
        assert all(result.retryable for result in results)
        assert all(result.evidence_ref == "worker:compute_backpressure" for result in results)
        assert adapter.snapshot().used_units == 1
        release.set()
        occupied.future.result(timeout=5)
        assert adapter.snapshot().used_units == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_pending_parse_reports_heartbeat_without_a_completion_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from polylogue.core.compute import BoundedComputeAdapter

    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=1024)
    entered = threading.Event()
    release = threading.Event()
    heartbeats: list[int] = []
    progress = ingest_batch_core._WorkerProgress()

    def parse(record: RawSessionRecord, request: _IngestWorkerRequest) -> IngestRecordResult:
        del request
        entered.set()
        assert release.wait(5)
        return IngestRecordResult(raw_id=record.raw_id, outcome_code="success")

    def heartbeat() -> None:
        assert entered.wait(5)
        if not heartbeats:
            assert adapter.snapshot().used_units == 1
            assert progress.in_flight_raw_ids == ["raw-1"]
            release.set()
        heartbeats.append(progress.completed_raw_count)

    monkeypatch.setattr(ingest_batch_core, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(ingest_batch_core, "_run_ingest_record", parse)
    monkeypatch.setattr(ingest_batch_core, "_INGEST_RESULT_WAIT_HEARTBEAT_S", 0.001)
    try:
        results = list(
            _iter_ingest_results_sync(
                _raw_records(1),
                request=_worker_request(),
                worker_count=1,
                heartbeat=heartbeat,
                progress=progress,
            )
        )
        assert heartbeats and heartbeats[0] == 0
        assert results[0].outcome_code == "success"
        assert progress.completed_raw_count == 1
        assert adapter.snapshot().used_units == 0
    finally:
        release.set()
        adapter.shutdown(wait=True)


def test_ingest_window_uses_shared_class_capacity_and_requested_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.core import compute
    from polylogue.core.compute import BoundedComputeAdapter

    adapter = BoundedComputeAdapter(max_workers=4, queue_units=4, queue_bytes=1024)
    monkeypatch.setattr(compute, "compute_adapter", lambda: adapter)
    try:
        ceiling = adapter.snapshot().by_class("incremental-background").ceiling_units
        assert ingest_batch_core._select_ingest_worker_count(_raw_records(50), None) == ceiling
        assert ingest_batch_core._select_ingest_worker_count(_raw_records(50), 1) == 1
        assert ingest_batch_core._select_ingest_worker_count(_raw_records(1), 50) == 1
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)
