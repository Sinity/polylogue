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


def test_claude_code_record_with_missing_type_is_not_materialized() -> None:
    """A dict record ``_fold_code_record`` silently drops must not count as parsed.

    Anti-vacuity (Codex P1, #5711): ``{"sessionId": "s", "uuid": "lost"}``
    has no ``type`` key at all, so ``_fold_code_record`` logs and returns
    without folding any evidence, while the generic nested-sentinel scan in
    ``AdmissionObserver.observe`` sees no specially-prefixed unknown marker
    and would classify the record MATERIALIZED anyway -- claiming complete
    materialization for a record that vanished. Assert the ledger.
    """
    from polylogue.sources.dispatch import parse_stream_payload

    records = [
        {
            "type": "user",
            "sessionId": "cc-missing-type",
            "uuid": "u-1",
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": "hello"},
        },
        {"sessionId": "cc-missing-type", "uuid": "lost"},
    ]

    (session,) = parse_stream_payload(Provider.CLAUDE_CODE, iter(records), "cc-missing-type")

    accounting = session.unit_accounting
    assert accounting is not None
    assert accounting.expected[AdmissionUnit.OUTER_RECORD] == 2
    lost = [outcome for outcome in accounting.outcomes if outcome.ordinal == 1]
    assert len(lost) == 1
    assert lost[0].disposition is AdmissionDisposition.TYPED_UNKNOWN
    assert "claude_code_unknown_input" in [event.event_type for event in session.session_events]


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


def test_claude_code_tool_input_is_not_a_wire_type() -> None:
    """Nested tool arguments are user data, not Claude Code discriminators.

    Anti-vacuity (Codex P2, #5711): scan the whole record and the tool call's
    arbitrary input ``{"type": "unknown"}`` turns a valid assistant record
    into ``TYPED_UNKNOWN`` with a false ``claude_code_unknown_input`` event.
    A block-level future type still classifies, so the scan is not disabled.
    """
    from polylogue.sources.dispatch import parse_stream_payload

    def assistant(uuid: str, block: dict[str, object]) -> dict[str, object]:
        return {
            "type": "assistant",
            "sessionId": "cc-tool-input",
            "uuid": uuid,
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": [block]},
        }

    tool_call: dict[str, object] = {"type": "tool_use", "id": "t-1", "name": "Probe", "input": {"type": "unknown"}}
    (session,) = parse_stream_payload(Provider.CLAUDE_CODE, iter([assistant("a-1", tool_call)]), "cc-tool-input")
    assert "claude_code_unknown_input" not in [event.event_type for event in session.session_events]
    accounting = session.unit_accounting
    assert accounting is not None
    assert all(outcome.disposition is not AdmissionDisposition.TYPED_UNKNOWN for outcome in accounting.outcomes)

    future_block: dict[str, object] = {"type": "future_block_kind", "text": "neutral"}
    (session,) = parse_stream_payload(Provider.CLAUDE_CODE, iter([assistant("a-2", future_block)]), "cc-tool-input")
    admission = [event for event in session.session_events if event.event_type == "claude_code_unknown_input"]
    assert [event.payload["wire_type"] for event in admission] == ["future_block_kind"]


def test_codex_tool_arguments_are_not_wire_types() -> None:
    """Nested MCP invocation arguments are user data, not Codex discriminators.

    Anti-vacuity (Codex P2, #5711): scan the whole record on the Codex stream
    route and an argument ``{"type": "unknown"}`` becomes a false
    ``codex_unknown_input`` event.
    """
    from polylogue.sources.dispatch import parse_stream_payload

    records: list[object] = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "type": "session_meta",
            "payload": {"id": "codex-args", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/w"},
        },
        {
            "timestamp": "2026-01-01T00:00:01Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        },
        {
            "timestamp": "2026-01-01T00:00:02Z",
            "type": "event_msg",
            "payload": {"type": "mcp_tool_call_end", "invocation": {"arguments": {"type": "unknown"}}},
        },
    ]
    (session,) = parse_stream_payload(Provider.CODEX, iter(records), "codex-args")

    assert "codex_unknown_input" not in [event.event_type for event in session.session_events]


@pytest.mark.parametrize("provider", ["hermes", "otel_genai"])
def test_every_unproven_session_of_a_multi_session_result_is_admitted(provider: str) -> None:
    """A parent with its subagent trajectories, or several OTLP conversations, all carry a proof.

    Anti-vacuity (Codex P2, #5711): exempt every non-OTel multi-session result
    (or compare the OTel token un-normalized) and these sessions reach the
    writer with ``unit_accounting=None``.
    """
    from polylogue.sources.parsers.base_models import ParsedSession

    sessions = [
        ParsedSession(source_name=Provider.HERMES, provider_session_id=name, messages=[])
        for name in ("parent", "child")
    ]
    admitted = base_support.admit_parsed_sessions(provider, [{"type": "session"}], sessions)

    assert [session.provider_session_id for session in admitted] == ["parent", "child"]
    assert all(session.unit_accounting is not None for session in admitted)


def test_a_claude_admission_event_takes_its_declared_place() -> None:
    """An untimestamped admission event sorts before timestamped Claude events.

    Anti-vacuity (Codex P2, #5711): append it after ``order_session_events``
    ran and it lands last, violating Claude's missing-timestamps-first order.
    """
    from polylogue.sources.parsers.base import ParsedSession, ParsedSessionEvent

    observer = base_support.AdmissionObserver(scan=lambda record: "future_record_kind")
    observer.observe({"type": "future_record_kind"})
    timed = ParsedSessionEvent(event_type="compaction", payload={}, timestamp="2026-01-01T00:00:00Z")
    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE, provider_session_id="s", messages=[], session_events=[timed]
    )

    admitted = observer.apply(session, "claude_code")

    assert [event.event_type for event in admitted.session_events] == ["claude_code_unknown_input", "compaction"]


def test_repeated_unknown_records_keep_one_pending_event_per_type() -> None:
    """Observing many records of one unknown type retains one pending event, not one per record.

    Anti-vacuity (Codex P1, #5711): keep a tuple per unknown record and the
    pending list grows with the stream.
    """
    observer = base_support.AdmissionObserver(scan=lambda record: "future_record_kind")
    for _ in range(1_000):
        observer.observe({"type": "future_record_kind"})

    assert len(observer._unknowns) == 1


def test_the_default_scan_never_reads_a_wire_type_from_tool_data() -> None:
    """An origin without its own scanner still ignores tool arguments and results.

    Anti-vacuity (#5711, the class Codex found for OTel): recurse into every
    value and a ChatGPT-style tool argument ``{"type": "unknown"}`` marks the
    record unknown, while a real nested future content type must still be
    found.
    """
    from polylogue.sources.parsers.base_support import _unknown_wire_type

    for key in ("arguments", "input", "output", "result", "attributes", "parameters"):
        record = {"type": "message", "tool_call": {key: {"type": "unknown", "nested": [{"kind": "future_x"}]}}}
        assert _unknown_wire_type(record) is None, key
    assert _unknown_wire_type({"type": "message", "content": [{"content_type": "future_part"}]}) == "future_part"
