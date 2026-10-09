"""Cancellation at actual decoder record boundaries preserves handle cleanup."""

from __future__ import annotations

import io
import threading

import pytest

from polylogue.core.compute import BoundedComputeAdapter, DaemonOperationCancelled
from tests.infra.json_values import iter_owned_json_values


def test_cancelled_decode_stops_before_publishing_the_next_record() -> None:
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=0, queue_bytes=1024)
    first_record = threading.Event()
    resume = threading.Event()
    closed = threading.Event()
    published: list[object] = []

    def decode() -> None:
        try:
            with io.BytesIO(b'{"marker":1}\n{"marker":2}\n') as source:
                for value in iter_owned_json_values(source, "synthetic.jsonl", fail_on_decode_error=True):
                    published.append(value)
                    first_record.set()
                    assert resume.wait(5)
        finally:
            assert source.closed
            closed.set()

    try:
        operation = adapter.submit(decode, estimated_bytes=1024)
        assert first_record.wait(5)
        operation.cancellation.cancel()
        assert not operation.future.done()
        resume.set()
        with pytest.raises(DaemonOperationCancelled):
            operation.future.result(timeout=5)
        assert published == [{"marker": 1}]
        assert closed.is_set()
        assert adapter.snapshot().used_units == 0
    finally:
        resume.set()
        adapter.shutdown(wait=True)
