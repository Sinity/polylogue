"""The conservation boundary must read a parser's payload, not its first argument.

``parser_admission`` synthesizes the admission ledger and the
``<provider>_unknown_input`` session events from whatever it believes the wire
payload is. Every decorated parser but one takes that payload first;
``drive.parse_chunked_prompt`` takes ``(provider, payload, fallback_id)``, so
positional binding scanned the provider token. The observable consequence was
that a future-shaped Drive chunk vanished without a typed event while the
ledger claimed one parsed outer record.

Anti-vacuity: restore ``payload = args[0] ...`` in
``polylogue/sources/parsers/base_support.py`` and
``test_drive_future_chunk_is_typed_unknown`` goes red -- the
provider token carries no ``future_`` wire type, so no event and no
``TYPED_UNKNOWN`` outcome is produced. The opposite direction is pinned by
``test_drive_known_chunks_emit_no_unknown``: a blanket "always
emit an unknown" boundary fails it.
"""

from __future__ import annotations

import inspect

import pytest

from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.sources.parsers import base_support
from polylogue.sources.parsers.base_models import AdmissionDisposition, AdmissionUnit
from polylogue.sources.parsers.drive import parse_chunked_prompt


def _payload(*, unknown: bool) -> dict[str, JSONValue]:
    model_chunk: dict[str, JSONValue] = {"role": "model", "text": "conservation"}
    if unknown:
        model_chunk["type"] = "future_reasoning_trace"
    chunks: list[JSONValue] = [
        {"role": "user", "text": "what is the boundary for?"},
        model_chunk,
    ]
    return {"chunkedPrompt": {"chunks": chunks}}


def test_drive_future_chunk_is_typed_unknown() -> None:
    session = parse_chunked_prompt(Provider.DRIVE, _payload(unknown=True), "drive-admission")

    assert [event.event_type for event in session.session_events] == ["drive_unknown_input"]
    assert session.session_events[0].payload["wire_type"] == "future_reasoning_trace"

    accounting = session.unit_accounting
    assert accounting is not None
    unknown_outcomes = [
        outcome for outcome in accounting.outcomes if outcome.disposition is AdmissionDisposition.TYPED_UNKNOWN
    ]
    assert [outcome.key for outcome in unknown_outcomes] == ["future_reasoning_trace"]


def test_drive_known_chunks_emit_no_unknown() -> None:
    session = parse_chunked_prompt(Provider.DRIVE, _payload(unknown=False), "drive-admission")

    assert [event.event_type for event in session.session_events] == []
    accounting = session.unit_accounting
    assert accounting is not None
    assert accounting.expected[AdmissionUnit.OUTER_RECORD] == 1
    assert all(outcome.disposition is not AdmissionDisposition.TYPED_UNKNOWN for outcome in accounting.outcomes)


@pytest.mark.parametrize(
    ("signature_source", "expected"),
    [
        ("def parser(payload, fallback_id): ...", (0, "payload")),
        ("def parser(provider, payload, fallback_id): ...", (1, "payload")),
        ("def parser(task, fallback_id): ...", (0, "task")),
        ("def parser(payload, fallback_id, *, source_path=None): ...", (0, "payload")),
    ],
    ids=["first", "second", "task", "kwonly"],
)
def test_payload_parameter_follows_signature(signature_source: str, expected: tuple[int, str]) -> None:
    namespace: dict[str, object] = {}
    exec(signature_source, namespace)
    parser = namespace["parser"]
    assert callable(parser)
    assert base_support._payload_parameter(parser) == expected


def test_payload_parameter_refuses_no_args() -> None:
    """A parser whose signature hides its payload must fail at decoration."""
    with pytest.raises(TypeError):
        base_support._payload_parameter(lambda **kwargs: None)  # type: ignore[arg-type]

    resolved = base_support._payload_parameter(inspect.unwrap(parse_chunked_prompt))
    assert resolved == (1, "payload")


def _claude_code_records() -> list[object]:
    return [
        {
            "type": "user",
            "sessionId": "cc-admission",
            "uuid": "u-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        },
        {"type": "future_record_kind", "sessionId": "cc-admission", "uuid": "u-2"},
        7,
    ]


def test_claude_code_production_stream_route_is_admitted() -> None:
    """The multi-session stream route carries the same admission proof as the leaf parser.

    Anti-vacuity: remove the per-group observer from
    ``_claude_code_multiway_parse_inner`` and the session reaches the writer
    with ``unit_accounting=None`` and no ``claude_code_unknown_input`` event,
    while the scalar record is silently dropped.
    """
    from polylogue.sources.dispatch import parse_stream_payload

    (session,) = parse_stream_payload(Provider.CLAUDE_CODE, iter(_claude_code_records()), "cc-admission")

    assert "claude_code_unknown_input" in [event.event_type for event in session.session_events]
    accounting = session.unit_accounting
    assert accounting is not None
    assert accounting.expected[AdmissionUnit.OUTER_RECORD] == 3
    dispositions = {outcome.disposition for outcome in accounting.outcomes}
    assert AdmissionDisposition.TYPED_UNKNOWN in dispositions
    assert AdmissionDisposition.TYPED_REFUSAL in dispositions


def test_non_object_record_is_refused_not_materialized() -> None:
    """A skipped scalar record is a typed refusal, never counted as material.

    Anti-vacuity: treat every record without an unknown-type sentinel as
    materialized and the scalar balances the ledger as ``MATERIALIZED``.
    """
    observer = base_support.AdmissionObserver()
    observer.observe({"type": "message"})
    observer.observe(7)
    from polylogue.sources.parsers.base import ParsedSession

    session = observer.apply(ParsedSession(source_name=Provider.CODEX, provider_session_id="s", messages=[]), "codex")

    accounting = session.unit_accounting
    assert accounting is not None
    refusals = [outcome for outcome in accounting.outcomes if outcome.disposition is AdmissionDisposition.TYPED_REFUSAL]
    assert [outcome.ordinal for outcome in refusals] == [1]


def test_disk_backed_event_sink_is_kept_not_copied() -> None:
    """Admission appends to a mutable event sink in place and keeps its identity.

    Anti-vacuity: copy ``session.session_events`` into a fresh list and the
    returned session no longer carries the caller's sink object, which in
    production is the ``SqliteSessionEventSink`` holding a stream's events
    on disk.
    """
    from collections.abc import MutableSequence

    from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent

    class Sink(MutableSequence[ParsedSessionEvent]):
        def __init__(self) -> None:
            self.items: list[ParsedSessionEvent] = []

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index):  # type: ignore[no-untyped-def]
            return self.items[index]

        def __setitem__(self, index, value):  # type: ignore[no-untyped-def]
            self.items[index] = value

        def __delitem__(self, index):  # type: ignore[no-untyped-def]
            del self.items[index]

        def insert(self, index: int, value: ParsedSessionEvent) -> None:
            self.items.insert(index, value)

    sink = Sink()
    sink.append(ParsedSessionEvent(event_type="compaction", payload={}))
    observer = base_support.AdmissionObserver()
    observer.observe({"type": "message"})
    observer.observe({"type": "future_record_kind"})
    session = ParsedSession.model_construct(
        source_name=Provider.CODEX, provider_session_id="s", messages=[], session_events=sink, unit_accounting=None
    )

    admitted = observer.apply(session, "codex")

    kept: object = admitted.session_events
    assert kept is sink
    assert [event.event_type for event in sink.items] == ["compaction", "codex_unknown_input"]


def test_interleaved_claude_code_unknown_keeps_its_file_line() -> None:
    """A per-session admission event names the record's position in the file.

    Anti-vacuity: count positions per session group and the second session's
    unknown record at file line 3 is reported as ``source_index`` 2.
    """
    from polylogue.sources.dispatch import parse_stream_payload

    def user(session_id: str, uuid: str) -> dict[str, object]:
        return {
            "type": "user",
            "sessionId": session_id,
            "uuid": uuid,
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        }

    records: list[object] = [
        user("cc-a", "a-1"),
        user("cc-b", "b-1"),
        {"type": "future_record_kind", "sessionId": "cc-b", "uuid": "b-2"},
    ]

    sessions = parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "cc-a")

    unknowns = [
        event.payload
        for session in sessions
        for event in session.session_events
        if event.event_type == "claude_code_unknown_input"
    ]
    assert unknowns == [{"source_index": 3, "wire_type": "future_record_kind"}]
