"""Coarse compute trace follows submitted work across the executor boundary."""

from __future__ import annotations

import threading

import pytest

from polylogue import logging as plog
from polylogue.core import compute


@pytest.mark.uses_real_clock
def test_compute_trace_is_opt_in_and_wait_is_a_logical_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute, "TRACE_COMPUTE", True)
    previous = plog.set_level("info")
    adapter = compute.BoundedComputeAdapter(max_workers=1, queue_units=2)
    submit_tid = threading.get_native_id()
    try:
        with plog.capture() as records:
            with plog.span("caller") as caller:
                operation = adapter.submit(lambda: 42)
                assert operation.future.result() == 42
                caller.ok()
        queue = next(r for r in records if r["event"] == "compute.queue.ok")
        service = next(r for r in records if r["event"] == "compute.service.ok")
        parent = next(r for r in records if r["event"] == "caller.ok")
        assert queue["task_id"] == service["task_id"]
        assert queue["trace_id"] == service["trace_id"] == parent["trace_id"]
        assert queue["parent_span_id"] == service["parent_span_id"] == parent["span_id"]
        assert queue["span_kind"] == "wait"
        assert queue["span_start_native_thread_id"] == submit_tid
        assert service["native_thread_id"] != submit_tid
        assert service["cpu_scope"] == "thread"
        assert queue["span_start_ns"] <= queue["span_end_ns"] <= service["span_start_ns"]
        monkeypatch.setattr(compute, "TRACE_COMPUTE", False)
        with plog.capture() as disabled:
            assert adapter.submit(lambda: 9).future.result() == 9
        assert not any(str(r["event"]).startswith("compute.") for r in disabled)
    finally:
        adapter.shutdown(wait=True)
        plog.set_level(previous)


@pytest.mark.uses_real_clock
def test_compute_service_failure_retains_exception_and_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compute, "TRACE_COMPUTE", True)
    adapter = compute.BoundedComputeAdapter(max_workers=1)
    original = ValueError("synthetic failure")

    def fail() -> None:
        raise original

    try:
        with plog.capture() as records:
            with pytest.raises(ValueError) as raised:
                adapter.submit(fail).future.result()
        assert raised.value is original
        terminal = next(r for r in records if r["event"] == "compute.service.error")
        assert terminal["outcome"] == "error"
        assert terminal["error_type"] == "ValueError"
        assert terminal["span_start_ns"] <= terminal["span_end_ns"]
    finally:
        adapter.shutdown(wait=True)
