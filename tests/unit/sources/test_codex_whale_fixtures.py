"""Real Codex streaming-dispatch regression for the shared whale fixture."""

from __future__ import annotations

from collections.abc import Iterator

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
# record counts -- lookahead facts are held in the parser's disk-backed index,
# not as a second in-memory copy of the source stream. With ``list(records)``
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
    so it stores the source records and derived facts in a disk-backed index.

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
# per-record lookahead retention is a memory amplifier. Bounds below are traced
# Python allocation peaks measured in-test; each names what head measured with
# the fix reverted, which is what makes it non-vacuous.
_CODE_MODE_ITEM_COUNT = 200_000
# Head with ``_CodexExecItemRecord`` retaining the whole ``payload.item``:
# 794.9 MB traced peak for this shape. With the append-time reduction: 367.2 MB.
_CODE_MODE_ITEM_PEAK_BYTES_MAX = 600 * 1024 * 1024
_REPLACEMENT_CONTEXT_COUNT = 200_000
# Head with no aggregate ceiling: 461.7 MB traced peak and 200_002 session
# events. With the ceiling: 219.3 MB and 515 events.
_REPLACEMENT_CONTEXT_PEAK_BYTES_MAX = 330 * 1024 * 1024
_REPLACEMENT_CONTEXT_EVENT_MAX = 1_000


def _code_mode_item_stream() -> Iterator[dict[str, object]]:
    yield {"type": "session_meta", "payload": {"id": "code-mode-flood"}}
    for index in range(_CODE_MODE_ITEM_COUNT):
        yield {
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


def test_code_mode_item_lookahead_does_not_retain_whole_payload_items() -> None:
    """Anti-vacuity: restoring ``item=executed`` (the unreduced mapping) blows the traced-peak bound.

    The reduction keeps every key the item readers consult plus the one
    selected output text, so the parse result is unchanged; only the retained
    bytes per item change.
    """
    import tracemalloc

    from polylogue.sources.dispatch import parse_stream_payload

    tracemalloc.start()
    try:
        sessions = parse_stream_payload(
            "codex",
            _code_mode_item_stream(),
            "code-mode-flood",
            source_path="code-mode-flood.jsonl",
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert len(sessions) == 1
    assert peak < _CODE_MODE_ITEM_PEAK_BYTES_MAX, (
        f"code-mode item lookahead traced peak {peak} exceeds {_CODE_MODE_ITEM_PEAK_BYTES_MAX}"
    )


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


def test_replacement_context_ceiling_degrades_excess_into_the_digest_only_event() -> None:
    """Anti-vacuity: raising the session ceiling constants blows the traced-peak and event-count bounds.

    The per-value cap bounds one context, not their number; without the
    aggregate ceiling every distinct small replacement text becomes its own
    durable session event. The excess must reuse the existing omission
    channel, never disappear.
    """
    import tracemalloc

    from polylogue.sources.dispatch import parse_stream_payload
    from polylogue.sources.parsers import codex

    tracemalloc.start()
    try:
        sessions = parse_stream_payload(
            "codex",
            _replacement_history_stream(),
            "replacement-flood",
            source_path="replacement-flood.jsonl",
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert len(sessions) == 1
    events = sessions[0].session_events
    assert peak < _REPLACEMENT_CONTEXT_PEAK_BYTES_MAX, (
        f"replacement-context traced peak {peak} exceeds {_REPLACEMENT_CONTEXT_PEAK_BYTES_MAX}"
    )
    assert len(events) < _REPLACEMENT_CONTEXT_EVENT_MAX

    contexts = [event for event in events if event.event_type == codex._CODEX_REPLACEMENT_CONTEXT_EVENT_TYPE]
    omissions = [event for event in events if event.event_type == codex._CODEX_REPLACEMENT_CONTEXT_OMITTED_EVENT_TYPE]
    assert len(contexts) == codex._CODEX_REPLACEMENT_CONTEXT_MAX_DISTINCT
    # One digest-only aggregate on the existing channel, not a second channel
    # and not a silent drop: every excess value is counted and hashed.
    assert len(omissions) == 1
    payload = omissions[0].payload
    assert payload["content_policy"] == codex._CODEX_REPLACEMENT_CEILING_POLICY
    assert payload["occurrences"] == _REPLACEMENT_CONTEXT_COUNT - codex._CODEX_REPLACEMENT_CONTEXT_MAX_DISTINCT
    assert payload["reconstruction"] == "source_blob"
    assert "content" not in payload
    assert len(str(payload["content_sha256"])) == 64
    compaction = next(event for event in events if event.event_type == "compaction")
    assert compaction.payload["replacement_history_text_count"] == _REPLACEMENT_CONTEXT_COUNT
