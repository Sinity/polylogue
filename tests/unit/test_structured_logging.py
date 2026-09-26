"""Properties of the structured event layer.

Each test names the mutation that would make it red — a test that cannot fail
is not evidence.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from polylogue import logging as plog
from polylogue.logging_fields import FIELDS, FORBIDDEN_FIELDS, QUARANTINED_FIELDS


@pytest.fixture(autouse=True)
def _isolated_level() -> Iterator[None]:
    previous = plog.set_level("trace")
    try:
        yield
    finally:
        plog.set_level(previous)


# -- correlation ------------------------------------------------------------


def test_correlation_id_survives_a_plain_thread_when_propagated() -> None:
    """Anti-vacuity: drop ``contextvars.copy_context`` from ``propagate`` and the
    worker emits without ``run_id``, so the assertion on the worker record fails."""
    with plog.capture() as records:
        with plog.bind(run_id="thread-run"):

            def worker() -> None:
                plog.emit("worker.tick")

            thread = threading.Thread(target=plog.propagate(worker))
            thread.start()
            thread.join()

    worker_records = [r for r in records if r["event"] == "worker.tick"]
    assert worker_records, "worker emitted nothing"
    assert worker_records[0]["run_id"] == "thread-run"


def test_pooled_thread_loses_context_without_propagate() -> None:
    """``propagate`` earns its existence here, not on plain threads.

    Python 3.14 gives a *newly created* ``threading.Thread`` the creating
    thread's context, so a bare thread correlates for free. A thread taken from
    a pool created *before* the bind does not — and that is precisely the shape
    of ``polylogue.daemon.execution``'s long-lived ThreadPoolExecutor.

    Anti-vacuity: make ``propagate`` the identity function and the second half
    of this test goes red.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda: None).result()  # force thread creation before the bind

        with plog.capture() as records:
            with plog.bind(run_id="pool-run"):
                pool.submit(lambda: plog.emit("pool.bare")).result()
                pool.submit(plog.propagate(lambda: plog.emit("pool.wrapped"))).result()

    by_event = {r["event"]: r for r in records}
    assert "run_id" not in by_event["pool.bare"], "pooled thread unexpectedly inherited context"
    assert by_event["pool.wrapped"]["run_id"] == "pool-run"


def test_plain_thread_inherits_context_on_this_interpreter() -> None:
    """Documents the 3.14 behaviour the conversion recipe relies on.

    Anti-vacuity: on an interpreter that does not propagate context to new
    threads this goes red, which is the signal that every bare
    ``threading.Thread`` in the tree needs a ``propagate`` wrapper after all.
    """
    with plog.capture() as records:
        with plog.bind(run_id="thread-run"):
            thread = threading.Thread(target=lambda: plog.emit("bare.tick"))
            thread.start()
            thread.join()

    bare = [r for r in records if r["event"] == "bare.tick"]
    assert bare and bare[0]["run_id"] == "thread-run"


def test_correlation_id_survives_an_await_and_to_thread_boundary() -> None:
    """Anti-vacuity: replace the ContextVar with a thread-local and the
    ``asyncio.to_thread`` record loses ``run_id``."""

    async def scenario() -> None:
        with plog.bind(run_id="async-run"):
            await asyncio.sleep(0)
            plog.emit("async.after_await")
            await asyncio.to_thread(plog.emit, "async.in_thread")

    with plog.capture() as records:
        asyncio.run(scenario())

    by_event = {r["event"]: r for r in records}
    assert by_event["async.after_await"]["run_id"] == "async-run"
    assert by_event["async.in_thread"]["run_id"] == "async-run"


def test_executor_submit_requires_propagate_and_then_correlates() -> None:
    """Anti-vacuity: ``ThreadPoolExecutor`` reuses threads; if ``propagate``
    captured the context at call time rather than per-wrap, the second submit
    would carry the first submit's run_id and this test would fail."""
    with plog.capture() as records:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with plog.bind(run_id="first"):
                pool.submit(plog.propagate(lambda: plog.emit("pool.tick"))).result()
            with plog.bind(run_id="second"):
                pool.submit(plog.propagate(lambda: plog.emit("pool.tick"))).result()

    ticks = [r["run_id"] for r in records if r["event"] == "pool.tick"]
    assert ticks == ["first", "second"]


def test_nested_spans_share_a_trace_and_chain_parentage() -> None:
    """Anti-vacuity: stop reading ``parent.get("trace_id")`` in ``span`` and the
    child gets a fresh trace, breaking the equality assertion."""
    with plog.capture() as records:
        with plog.span("outer") as outer:
            with plog.span("inner") as inner:
                inner.ok()
            outer.ok()

    starts = {r["event"]: r for r in records if str(r["event"]).endswith(".start")}
    assert starts["inner.start"]["trace_id"] == starts["outer.start"]["trace_id"]
    assert starts["inner.start"]["parent_span_id"] == starts["outer.start"]["span_id"]


# -- honesty ----------------------------------------------------------------


def test_swallowed_exception_still_emits_an_error_event() -> None:
    """The campaign's defect: a broad ``except`` hiding a failure.

    Anti-vacuity: delete the ``except BaseException`` arm of ``span`` and no
    error record exists, so the assertion fails even though the caller's
    ``except`` still runs.
    """
    with plog.capture() as records:
        try:
            with plog.span("risky", stage="parse"):
                raise ValueError("kaboom")
        except ValueError:
            pass  # the caller carries on; the log must not

    errors = [r for r in records if r["event"] == "risky.error"]
    assert errors, "a swallowed exception produced no error event"
    assert errors[0]["outcome"] == "error"
    assert errors[0]["error_type"] == "ValueError"
    assert errors[0]["level"] == "error"


def test_span_without_declared_outcome_is_unmeasured_not_success() -> None:
    """Anti-vacuity: default ``_outcome`` to ``"ok"`` and this goes red — which
    is precisely the "probe timed out, status said ready" defect."""
    with plog.capture() as records:
        with plog.span("probe"):
            pass

    terminal = [r for r in records if str(r["event"]).startswith("probe.") and not str(r["event"]).endswith(".start")]
    assert len(terminal) == 1
    assert terminal[0]["event"] == "probe.unmeasured"
    assert terminal[0]["outcome"] == "unmeasured"
    assert terminal[0]["level"] == "warning"


def test_refusal_is_never_rendered_as_success() -> None:
    """Anti-vacuity: map ``refused`` to INFO/ok and the level assertion fails."""
    with plog.capture() as records:
        with plog.span("op") as s:
            s.refused("write_lease_unavailable")

    terminal = [r for r in records if r["event"] == "op.refused"]
    assert terminal and terminal[0]["outcome"] == "refused"
    assert terminal[0]["level"] == "warning"
    assert terminal[0]["reason"] == "write_lease_unavailable"


def test_caller_fields_cannot_forge_reserved_keys() -> None:
    """Anti-vacuity: move the reserved-key assignment before ``dict(fields)`` in
    ``_emit_raw`` and a caller-supplied ``level`` overwrites the real one."""
    with plog.capture() as records:
        plog.emit("real.event", level=plog.ERROR, reason="x")

    assert records[-1]["event"] == "real.event"
    assert records[-1]["level"] == "error"


# -- PII boundary -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FORBIDDEN_FIELDS))
def test_no_content_field_can_reach_the_log(name: str) -> None:
    """Anti-vacuity: make ``rejection_reason`` return ``None`` for unknown names
    (i.e. switch the allowlist to a denylist-free pass-through) and every
    parameter case leaks its payload into the record."""
    payload = "SECRET-TRANSCRIPT-PAYLOAD"
    with plog.capture() as records:
        plog.emit("attempt", **{name: payload})  # type: ignore[arg-type]

    emitted = [r for r in records if r["event"] == "attempt"]
    assert emitted, "the event itself should still be emitted"
    assert name not in emitted[0]
    assert payload not in json.dumps(records)

    rejections = [r for r in records if r["event"] == "log.field_rejected"]
    assert rejections and rejections[0]["field"] == name
    assert rejections[0]["reason"] == "content_field"


def test_unregistered_field_is_dropped_with_a_reason() -> None:
    """Anti-vacuity: accept unknown names and ``blah`` appears in the record."""
    with plog.capture() as records:
        plog.emit("attempt", blah="whatever")

    assert "blah" not in [r for r in records if r["event"] == "attempt"][0]
    rejection = [r for r in records if r["event"] == "log.field_rejected"][0]
    assert rejection["reason"] == "unregistered_field"


def test_bind_also_enforces_the_allowlist() -> None:
    """Anti-vacuity: skip ``_validate`` in ``bind`` and content bound once would
    ride along on every downstream event — the worst possible leak shape."""
    with plog.capture() as records:
        with plog.bind(content="SECRET"):
            plog.emit("downstream")

    assert "content" not in [r for r in records if r["event"] == "downstream"][0]
    assert "SECRET" not in json.dumps(records)


def test_quarantined_free_text_is_truncated_and_redactable() -> None:
    """Anti-vacuity: remove the truncation branch and the long string survives
    whole; remove the redact filter and it survives rendering."""
    long_detail = "x" * 5000
    with plog.capture() as records:
        plog.emit("boom", error_detail=long_detail)

    record = records[-1]
    assert len(str(record["error_detail"])) < 400
    assert str(record["error_detail"]).endswith("<truncated>")

    assert "error_detail" in plog.render_console(record, redact=False)
    assert "error_detail" not in plog.render_console(record, redact=True)


def test_quarantine_set_stays_minimal() -> None:
    """The redaction story only holds while there is one free-text field.

    Anti-vacuity: register a second ``text``-kind field and this goes red,
    forcing a deliberate decision rather than quiet drift.
    """
    assert set(QUARANTINED_FIELDS) == {"error_detail"}
    assert [name for name, kind in FIELDS.items() if kind == "text"] == ["error_detail"]


def test_non_scalar_values_are_reduced_to_their_type() -> None:
    """Anti-vacuity: stringify via ``repr`` instead and an object's contents
    (which may embed session text) land in the record."""

    class Carrier:
        def __repr__(self) -> str:
            return "Carrier(secret='LEAKED')"

    with plog.capture() as records:
        plog.emit("attempt", origin=Carrier())

    assert records[-1]["origin"] == "<Carrier>"
    assert "LEAKED" not in json.dumps(records)


# -- cost and rendering -----------------------------------------------------


def test_suppressed_events_do_no_work() -> None:
    """Anti-vacuity: move the threshold check below ``_validate`` and the sink
    is consulted (or validation runs) for a suppressed event."""
    calls: list[object] = []
    sink = plog.add_sink(lambda record: calls.append(record))
    try:
        plog.set_level("error")
        plog.emit("noise", level=plog.DEBUG, reason="ignored")
        assert calls == []
        plog.emit("real", level=plog.ERROR, reason="kept")
        assert len(calls) == 1
    finally:
        plog.remove_sink(sink)


def test_json_is_the_storage_form_and_console_is_a_view() -> None:
    """Anti-vacuity: emit pre-rendered strings instead of records and the
    round-trip through ``json.loads`` fails."""
    stream = io.StringIO()
    sink = plog.add_sink(plog.make_stream_sink(stream, fmt="json"))
    try:
        plog.emit("daemon.stage.ok", stage="ingest", sessions=3)
    finally:
        plog.remove_sink(sink)

    parsed = json.loads(stream.getvalue().strip())
    assert parsed["event"] == "daemon.stage.ok"
    assert parsed["sessions"] == 3
    assert "daemon.stage.ok" in plog.render_console(parsed)


def test_stdlib_records_are_bridged_into_the_event_stream() -> None:
    """Unconverted modules must stay visible during migration.

    Anti-vacuity: remove the handler installation and the legacy warning
    produces no event at all.
    """
    import logging as stdlib_logging

    bridge = plog._StdlibBridge()
    stdlib_logging.getLogger("legacy.module").addHandler(bridge)
    try:
        with plog.capture() as records:
            stdlib_logging.getLogger("legacy.module").warning("old style %s", "message")
    finally:
        stdlib_logging.getLogger("legacy.module").removeHandler(bridge)

    bridged = [r for r in records if r["event"] == "stdlib.record"]
    assert bridged and bridged[0]["logger"] == "legacy.module"
    assert bridged[0]["error_detail"] == "old style message"


def test_propagate_preserves_arguments_and_return_value() -> None:
    """A wrapped callable still receives its arguments and returns its result.

    The *static* half of this — that ``propagate`` is generic, so a wrapped
    ``ThreadPoolExecutor.submit`` keeps its result type — cannot be asserted at
    runtime, because ``ParamSpec`` erases to ``*args``/``**kwargs`` in the
    actual signature. ``devtools gate mypy`` is what enforces it: restoring the
    old ``Callable[..., object] -> Callable[..., object]`` fails typechecking on
    both ``asyncio.shield`` calls in ``daemon/convergence.py``.

    That signature mattered because under it the path of least resistance is to
    drop the wrapper rather than cast at the call site — and dropping it loses
    the correlation id silently.

    Anti-vacuity for this test: have ``runner`` swallow ``kwargs`` and the
    keyword-only argument is lost.
    """

    def typed(count: int, *, label: str) -> tuple[int, str]:
        return count, label

    assert plog.propagate(typed)(3, label="x") == (3, "x")


def test_console_rendering_escapes_untrusted_control_characters() -> None:
    """A remote-controlled field cannot forge a log line or drive the terminal.

    ``origin`` carries a browser-supplied ``Origin`` header and ``error_detail``
    carries quarantined free text; both reach the console sink verbatim.
    Anti-vacuity: drop ``_console_safe`` from ``render_console`` and the newline
    below again splits one record into two rendered lines, with the ESC sequence
    delivered raw to the terminal.
    """
    record = {
        "ts": "2026-01-01T00:00:00.000000Z",
        "level": "info",
        "event": "browser_capture.request",
        "origin": "chrome-extension://a\nFAKE forged event\x1b[31m",
    }
    rendered = plog.render_console(record)
    assert "\n" not in rendered
    assert "\x1b" not in rendered
    assert "\\x0a" in rendered
    assert "\\x1b" in rendered
    assert "FAKE forged event" in rendered


def test_console_rendering_leaves_ordinary_values_untouched() -> None:
    """Escaping must not disturb the normal human view.

    Anti-vacuity: escape a character outside the control range and this goes red
    on the plain ``event``/``origin`` spelling below.
    """
    record = {
        "ts": "2026-01-01T00:00:00.000000Z",
        "level": "info",
        "event": "daemon.stage.ok",
        "origin": "chrome-extension://abcdef",
    }
    rendered = plog.render_console(record)
    assert "daemon.stage.ok" in rendered
    assert "origin=chrome-extension://abcdef" in rendered


def test_log_redact_strips_quarantined_fields_from_the_json_storage_form() -> None:
    """POLYLOGUE_LOG_REDACT reaches the JSON sink, not only the console view.

    Anti-vacuity: restoring ``render_json(record)`` (no ``redact`` argument) in
    ``make_stream_sink`` leaves ``error_detail`` in the retained JSON line and
    this goes red. The unredacted assertions keep the fix from degenerating
    into stripping the field unconditionally.
    """
    record = {"ts": "2026-01-01T00:00:00.000000Z", "level": "error", "event": "boom", "error_detail": "secret detail"}

    assert "secret detail" in plog.render_json(record)
    assert "error_detail" not in plog.render_json(record, redact=True)
    assert json.loads(plog.render_json(record, redact=True))["event"] == "boom"

    stream = io.StringIO()
    plog.make_stream_sink(stream, fmt="json", redact=True)(record)
    assert "error_detail" not in stream.getvalue()

    plain = io.StringIO()
    plog.make_stream_sink(plain, fmt="json")(record)
    assert "secret detail" in plain.getvalue()


def test_invalid_measurements_are_rejected_by_real_renderer() -> None:
    """A string count, infinite duration, or invented outcome cannot reach the sink."""
    stream = io.StringIO()
    sink = plog.add_sink(plog.make_stream_sink(stream, fmt="json"))
    try:
        plog.emit(
            "intake.chunk",
            files="7",
            duration_ms=float("inf"),
            outcome="imaginary",
            reason="x" * 100_000,
            stage_timings_ms={"parse": float("nan")},
        )
    finally:
        plog.remove_sink(sink)
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    chunk = next(record for record in records if record["event"] == "intake.chunk")
    assert {record["field"] for record in records if record["event"] == "log.field_rejected"} == {
        "files",
        "duration_ms",
        "outcome",
        "reason",
        "stage_timings_ms",
    }
    assert "reason" not in chunk
    assert len(json.dumps(chunk)) <= plog.EVENT_MAX_BYTES


def test_combined_context_and_event_fields_stay_within_record_limit() -> None:
    """Two individually bounded field sets cannot combine into an oversized record."""
    bound = dict.fromkeys(
        (
            "trace_id",
            "span_id",
            "run_id",
            "pass_id",
            "operation_id",
            "request_id",
            "session_id",
            "message_id",
            "block_id",
            "raw_id",
            "artifact_id",
            "blob_hash",
            "content_hash",
            "source_id",
            "source_name",
            "member_id",
            "cursor_id",
            "tool_id",
            "generation_id",
            "branch_point_message_id",
        ),
        "x",
    )
    emitted = dict.fromkeys(
        (
            "sessions",
            "messages",
            "blocks",
            "raws",
            "files",
            "considered",
            "ingested",
            "skipped",
            "failed",
            "succeeded",
            "refused",
            "deferred",
            "pending",
            "retried",
            "queued",
            "active",
            "bytes",
            "size",
            "rows",
            "attempts",
        ),
        1,
    )
    with plog.capture() as records, plog.bind(**bound):
        plog.emit("bounded.record", **emitted)
    record = next(row for row in records if row["event"] == "bounded.record")
    assert len(record) <= plog.EVENT_MAX_FIELDS
    assert any(row["event"] == "log.field_rejected" and row["reason"] == "event_field_limit" for row in records)


def test_span_terminal_collisions_preserve_original_exception() -> None:
    """A terminal field from a caller cannot cause a second TypeError."""
    original = ValueError("original failure")
    with plog.capture() as records:
        with pytest.raises(ValueError) as raised:
            with plog.span("collision") as active:
                active.set(level="debug", outcome="ok", duration_ms=999, reason="false", error_type="Wrong")
                raise original
    assert raised.value is original
    terminal = [record for record in records if record["event"] == "collision.error"]
    assert len(terminal) == 1
    assert terminal[0]["outcome"] == "error"
    assert terminal[0]["level"] == "error"
    assert terminal[0]["error_type"] == "ValueError"
    assert terminal[0]["duration_ms"] != 999


def test_unprintable_exception_and_malformed_field_cannot_mask_work() -> None:
    class UnprintableError(ValueError):
        def __str__(self) -> str:
            raise RuntimeError("bad formatter")

    class BadPath:
        def __fspath__(self) -> str:
            raise RuntimeError("bad path")

    original = UnprintableError()
    with plog.capture() as records:
        plog.emit("safe.success", path=BadPath())
        with pytest.raises(UnprintableError) as raised:
            with plog.span("safe.failure"):
                raise original
    assert raised.value is original
    assert [record for record in records if record["event"] == "safe.success"]
    terminal = [record for record in records if record["event"] == "safe.failure.error"]
    assert len(terminal) == 1
    assert terminal[0]["error_detail"] == "<unprintable UnprintableError>"


@pytest.mark.uses_real_clock(
    "measures bounded caller latency while a real worker thread is blocked in the configured sink"
)
def test_stalled_configured_sink_bounds_caller_and_reports_loss() -> None:
    """A blocked device cannot block emit or conceal queue overflow."""
    entered = threading.Event()
    release = threading.Event()

    class BlockedStream:
        def write(self, _line: str) -> None:
            entered.set()
            release.wait(timeout=2)

        def flush(self) -> None:
            pass

    plog.reset_events()
    try:
        plog.configure_events(stream=BlockedStream(), fmt="json", bridge_stdlib=False)
        plog.emit("first")
        assert entered.wait(timeout=1)
        started = time.monotonic()
        for _ in range(300):
            plog.emit("more")
        elapsed = time.monotonic() - started
        snapshot = plog.diagnostic_snapshot()
        assert elapsed < 0.5
        assert snapshot["dropped"] > 0
        assert snapshot["queued"] <= 256
        shutdown = plog.shutdown_events(timeout_s=0.01)
        assert shutdown["undrained"] > 0
    finally:
        release.set()
        plog.reset_events()


def test_normal_shutdown_drains_and_flushes_configured_sink() -> None:
    """A daemon exiting just after its terminal event retains that event."""
    stream = io.StringIO()
    plog.reset_events()
    try:
        plog.configure_events(stream=stream, fmt="json", bridge_stdlib=False)
        plog.emit("daemon.run.stop", outcome="ok")
        delivery = plog.shutdown_events(timeout_s=1)
        assert delivery["queued"] == 0
        assert delivery["dropped"] == 0
        assert delivery["delivered"] == 1
        assert [json.loads(line)["event"] for line in stream.getvalue().splitlines()] == ["daemon.run.stop"]
    finally:
        plog.reset_events()


@pytest.mark.uses_real_clock("bounds polling for failure reported asynchronously by the configured sink worker")
def test_failed_configured_sink_reports_loss_without_recursive_logging() -> None:
    class FailedStream:
        def write(self, _line: str) -> None:
            raise OSError("device failed")

        def flush(self) -> None:
            raise OSError("device failed")

    plog.reset_events()
    try:
        plog.configure_events(stream=FailedStream(), fmt="json", bridge_stdlib=False)
        plog.emit("one")
        deadline = time.monotonic() + 1
        while plog.diagnostic_snapshot()["failures"] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert plog.diagnostic_snapshot()["failures"] >= 1
        assert plog.diagnostic_snapshot()["queued"] == 0
    finally:
        plog.reset_events()


@pytest.mark.parametrize("json_logs", [False, True])
@pytest.mark.uses_real_clock("bounds polling for records delivered asynchronously by the logging worker")
def test_pre_and_post_configuration_loggers_share_one_sink(json_logs: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing either bridge makes one of four records disappear."""
    import logging

    plog.reset_events()
    # Another test may have configured structlog already; force this first
    # logger through the pre-configuration stdlib compatibility path.
    monkeypatch.setattr(plog, "_structlog_configured", False)
    before = plog.get_logger("test.before")
    stream = io.StringIO()
    try:
        plog.configure_logging(json_logs=json_logs)
        plog.configure_events(stream=stream, fmt="json")
        before.warning("before record")
        plog.get_logger("test.after").warning("after record")
        logging.getLogger("third.party").warning("third record")
        plog.emit("direct.record")
        deadline = time.monotonic() + 2
        while plog.diagnostic_snapshot()["delivered"] < 4 and time.monotonic() < deadline:
            time.sleep(0.01)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        events = [record["event"] for record in records]
        assert events.count("direct.record") == 1
        assert events.count("stdlib.record") == 2
        assert events.count("structlog.record") == 1
        assert all("error_detail" in record for record in records if record["event"] != "direct.record")
    finally:
        plog.reset_events()
