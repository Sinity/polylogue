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
