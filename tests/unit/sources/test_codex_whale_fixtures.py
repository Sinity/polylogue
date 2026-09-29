"""Real Codex streaming-dispatch regression for the shared whale fixture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from polylogue.sources.parsers.base import AdmissionDisposition, AdmissionUnit, ParsedSession
from tests.infra.whale_fixtures import WHALE_FIXTURE_DIMENSIONS, multi_million_codex_stream


def test_multi_million_codex_stream_uses_real_stream_dispatch_without_truncation() -> None:
    """Anti-vacuity: bypassing ``parse_stream_payload`` or shrinking the event boundary fails.

    State records are deliberately reused immutable evidence.  The parser must
    consume the complete fixture through its streaming entry point, while
    materializing only the one authored message in the resulting session.
    """
    from polylogue.sources.dispatch import parse_stream_payload

    class CountingStream:
        def __init__(self) -> None:
            self.yielded = 0
            self._source = multi_million_codex_stream()

        def __iter__(self) -> CountingStream:
            return self

        def __next__(self) -> dict[str, object]:
            value = next(self._source)
            self.yielded += 1
            return value

    stream = CountingStream()
    sessions = parse_stream_payload("codex", stream, "codex-stream-million", source_path="million.jsonl")

    assert stream.yielded == WHALE_FIXTURE_DIMENSIONS.stream_event_count + 2
    assert len(sessions) == 1
    assert sessions[0].provider_session_id == "codex-stream-million"
    assert len(sessions[0].messages) == 1
    assert sessions[0].messages[0].text == "sanitized streaming boundary"
    accounting = sessions[0].unit_accounting
    assert accounting is not None
    assert accounting.expected[AdmissionUnit.OUTER_RECORD] == stream.yielded
    outer = [outcome for outcome in accounting.iter_outcomes() if outcome.unit is AdmissionUnit.OUTER_RECORD]
    assert len(outer) == stream.yielded
    assert outer[0].disposition is AdmissionDisposition.MATERIALIZED
    assert outer[-1].disposition is AdmissionDisposition.MATERIALIZED
    assert outer[WHALE_FIXTURE_DIMENSIONS.stream_event_count].disposition is AdmissionDisposition.MATERIALIZED


#: One ~1 KB state record. 120,000 of them is about 110 MB of input.
_BUDGET_STREAM_RECORD_FILLER = "s" * 900
_BUDGET_STREAM_SMALL_RECORDS = 12_000
_BUDGET_STREAM_LARGE_RECORDS = 120_000
# Measured 2026-09-22 on this fixture. Head: 11.2 MB traced peak at BOTH
# record counts -- both streams exceed the in-memory replay budget, so their
# records and lookahead facts are held in the parser's disk-backed index rather
# than as a second in-memory copy of the source stream. With ``list(records)``
# retained for the lookahead pass, the peak was 137.6 MB at 120,000 records.
_BUDGET_STREAM_PEAK_BYTES_MAX = 40 * 1024 * 1024


def _budget_stream(record_count: int) -> Iterator[dict[str, object]]:
    yield {"type": "session_meta", "payload": {"id": "bounded-stream"}}
    yield {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "bounded-stream-message",
            "role": "user",
            "content": [{"type": "input_text", "text": "bounded"}],
        },
    }
    for sequence in range(record_count):
        yield {
            "record_type": "state",
            "sequence": sequence,
            "filler": f"{_BUDGET_STREAM_RECORD_FILLER}{sequence}",
        }
    yield {"type": "future_whale_record"}


def _parse_budget_stream(record_count: int) -> tuple[list[ParsedSession], int]:
    """Parse ``record_count`` state records and return the sessions and traced peak."""
    import tracemalloc

    from polylogue.sources.dispatch import parse_stream_payload

    tracemalloc.start()
    try:
        sessions = parse_stream_payload(
            "codex",
            _budget_stream(record_count),
            "bounded-stream",
            source_path="bounded.jsonl",
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return list(sessions), peak


def test_stream_dispatch_retention_stays_within_memory_bound() -> None:
    """Lookahead retention is bounded, not proportional to the input size.

    The parser needs lookahead-derived indexes before its materializing pass,
    so it keeps the source records in memory up to a byte budget and spills
    them, with the derived facts, to a disk-backed index above it.

    This replaces an earlier control that asserted *zero* retention by
    counting simultaneously-live decoded records (at most four). That
    condition contradicts the declared budget rather than bounding it: a
    stream whose records fit inside the budget is retained on purpose. What
    has to stay true is the memory bound, and a live-object count cannot see
    it -- so this asserts the traced allocation peak, the same way every
    other bound in this polylogue-ro922 family does.

    Anti-vacuity: replace the budgeted accumulation with ``list(records)``
    and the peak goes to 137.6 MB against a 40 MB bound, and keeps rising
    with the record count instead of staying flat.
    """
    sessions, peak = _parse_budget_stream(_BUDGET_STREAM_LARGE_RECORDS)

    assert peak < _BUDGET_STREAM_PEAK_BYTES_MAX, f"traced peak {peak / 1024 / 1024:.1f} MB"
    assert len(sessions) == 1
    assert sessions[0].messages[0].text == "bounded"
    expected_outer = _BUDGET_STREAM_LARGE_RECORDS + 3
    accounting = sessions[0].unit_accounting
    assert accounting is not None
    assert accounting.expected[AdmissionUnit.OUTER_RECORD] == expected_outer
    outer = [outcome for outcome in accounting.iter_outcomes() if outcome.unit is AdmissionUnit.OUTER_RECORD]
    assert len(outer) == expected_outer
    assert [outcome.disposition for outcome in outer[-2:]] == [
        AdmissionDisposition.MATERIALIZED,
        AdmissionDisposition.TYPED_UNKNOWN,
    ]
    assert outer[-1].ordinal == expected_outer - 1
    assert outer[-1].key == "future_whale_record"


def test_stream_dispatch_retention_does_not_grow_with_the_record_count() -> None:
    """Ten times the input, the same peak: the budget is what bounds it.

    A bound that a larger stream can still satisfy by accident is not a
    bound. This compares two stream sizes an order of magnitude apart and
    requires the traced peak not to track the input. ``list(records)`` fails
    it by construction: 11.3 MB at 12,000
    records and 137.6 MB at 120,000 (measured 2026-09-22).
    """
    _small_sessions, small_peak = _parse_budget_stream(_BUDGET_STREAM_SMALL_RECORDS)
    _large_sessions, large_peak = _parse_budget_stream(_BUDGET_STREAM_LARGE_RECORDS)

    record_ratio = _BUDGET_STREAM_LARGE_RECORDS / _BUDGET_STREAM_SMALL_RECORDS
    assert large_peak < small_peak * 2, (
        f"traced peak grew {large_peak / small_peak:.1f}x for {record_ratio:.0f}x the records "
        f"({small_peak / 1024 / 1024:.1f} MB -> {large_peak / 1024 / 1024:.1f} MB)"
    )


# polylogue-ro922. Codex session files are untrusted input, so the parser's
# per-record lookahead retention is a memory amplifier. The bound below is a
# traced Python allocation peak measured in-test; it names what head measured
# with the fix reverted, which is what makes it non-vacuous.
_REPLACEMENT_CONTEXT_COUNT = 200_000
# Head with no aggregate ceiling: 461.7 MB traced peak and 200_002 session
# events. With the ceiling: 219.3 MB and 515 events.


def _code_mode_item_record(index: int) -> dict[str, object]:
    return {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "CommandExecution",
                "id": f"exec-{index}",
                "command": ["bash", "-lc", f"echo {index}"],
                "cwd": "/w",
                "exit_code": 0,
                "status": "completed",
                "aggregated_output": f"out-{index}",
                "stdout": f"out-{index}",
                "stderr": "",
                "formatted_output": f"$ echo {index}\nout-{index}\n",
                "parsed_cmd": [{"cmd": f"echo {index}", "type": "read", "name": "echo"}],
                "duration_ms": 12,
                "turn_id": f"turn-{index}",
                "started_at": "2025-01-01T00:00:00Z",
                "completed_at": "2025-01-01T00:00:01Z",
                "sandbox_policy": "workspace-write",
                "approval": "on-request",
                "env": {"PATH": "/usr/bin:/bin", "HOME": "/w"},
                "truncated": False,
            },
        },
    }


def test_code_mode_item_lookahead_stores_only_the_reduced_item() -> None:
    """polylogue-ro922: the lookahead keeps the evidence item readers consult, not the whole item.

    Items wait in the parse's disk-backed scratch index, so their memory is
    bounded by construction; what the reduction still decides is what each
    stored item carries. Anti-vacuity: store ``executed`` unreduced and the
    row carries ``env``, ``sandbox_policy`` and every duplicate output text.

    This replaces a 200,000-item traced-peak test whose two arms measured the
    same 234.3 MB once items moved to the scratch index, so restoring the
    whole item could no longer turn it red.
    """
    import pickle
    import sqlite3

    from polylogue.sources.parsers import codex as codex_module

    with sqlite3.connect("") as connection:
        index = codex_module._CodexLookaheadIndex(connection)
        observer = codex_module._CodexLookaheadObserver(index)
        observer.observe(1, _code_mode_item_record(7))
        rows = connection.execute("SELECT item FROM codex_items").fetchall()
        index.close()

    assert len(rows) == 1
    stored = pickle.loads(rows[0][0])
    assert set(stored) <= {*codex_module._CODE_MODE_ITEM_MATCH_KEYS, "paths", "byte_count", "aggregated_output"}
    assert stored["command"] == ["bash", "-lc", "echo 7"]
    assert stored["aggregated_output"] == "out-7"
    assert "env" not in stored and "stdout" not in stored and "formatted_output" not in stored


def _replacement_history_stream() -> Iterator[dict[str, object]]:
    yield {"type": "session_meta", "payload": {"id": "replacement-flood"}}
    yield {
        "type": "response_item",
        "payload": {
            "type": "message",
            "id": "replacement-flood-message",
            "role": "user",
            "content": [{"type": "input_text", "text": "retained"}],
        },
    }
    yield {
        "type": "compacted",
        "payload": {
            "message": "summary",
            "replacement_history": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": f"t{index:08d}"}],
                }
                for index in range(_REPLACEMENT_CONTEXT_COUNT)
            ],
        },
    }


def test_every_distinct_replacement_only_value_is_stored_once_in_bounded_memory(tmp_path: Path) -> None:
    """Replacement-only text is content the session holds nowhere else.

    Every distinct value is stored exactly once, and the candidates wait in the
    parse's scratch index, not in memory.

    The bound is twice the traced size of the input records themselves: the
    parser may hold the compacted record it is reading, but not a second copy
    of every candidate beside it.

    Anti-vacuity: reinstate a count ceiling and fewer than
    ``_REPLACEMENT_CONTEXT_COUNT`` values survive; hold the candidates in a
    Python dict again and the traced peak exceeds the bound (measured
    461.7 MB against a 98 MB input before the scratch index held them).
    """
    import tracemalloc

    tracemalloc.start()
    try:
        input_bytes = tracemalloc.get_traced_memory()[0]
        records = list(_replacement_history_stream())
        input_bytes = tracemalloc.get_traced_memory()[0] - input_bytes
        del records
    finally:
        tracemalloc.stop()

    from polylogue.sources.parsers import codex
    from polylogue.sources.prepared_message_sink import SqliteMessageStore

    store = SqliteMessageStore(tmp_path / "replacement-flood.db")
    try:
        events = store.new_event_sink()
        tracemalloc.start()
        try:
            codex.parse_stream(
                _replacement_history_stream(), "replacement-flood", message_sink=store.new_sink(), event_sink=events
            )
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        stored = 0
        seen: set[str] = set()
        compaction_text_count = None
        for event in events:
            if event.event_type == codex._CODEX_REPLACEMENT_CONTEXT_EVENT_TYPE:
                stored += 1
                seen.add(str(event.payload["content"]))
            elif event.event_type == "compaction":
                compaction_text_count = event.payload["replacement_history_text_count"]
    finally:
        store.close()

    assert peak < 2 * input_bytes, f"traced peak {peak} against input {input_bytes}"
    assert stored == len(seen) == _REPLACEMENT_CONTEXT_COUNT
    assert "retained" not in seen
    assert compaction_text_count == _REPLACEMENT_CONTEXT_COUNT
