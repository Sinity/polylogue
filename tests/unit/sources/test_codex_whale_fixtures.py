"""Real Codex streaming-dispatch regression for the shared whale fixture."""

from __future__ import annotations

from collections.abc import Iterator

from tests.infra.whale_fixtures import WHALE_FIXTURE_DIMENSIONS, multi_million_codex_stream


def test_multi_million_codex_stream_uses_real_stream_dispatch_without_truncation() -> None:
    """Anti-vacuity: bypassing ``parse_stream_payload`` or shrinking the event boundary fails.

    State records are deliberately reused immutable evidence.  The parser must
    consume exactly two million of them through its streaming entry point,
    while materializing only the one authored message in the resulting session.
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


def test_stream_dispatch_does_not_retain_distinct_input_records() -> None:
    """A restored list(records) fails from live-record accumulation before parsing."""
    from polylogue.sources.dispatch import parse_stream_payload

    class TrackedStateRecord(dict[str, object]):
        live = 0

        def __init__(self, sequence: int) -> None:
            super().__init__(record_type="state", sequence=sequence)
            type(self).live += 1

        def __del__(self) -> None:
            type(self).live -= 1

        def __reduce__(self) -> tuple[object, tuple[dict[str, object]]]:
            return dict, (dict(self),)

    def guarded_stream() -> Iterator[dict[str, object]]:
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
        for sequence in range(10_000):
            if TrackedStateRecord.live > 4:
                raise AssertionError("stream parser retained decoded input records")
            yield TrackedStateRecord(sequence)

    sessions = parse_stream_payload("codex", guarded_stream(), "bounded-stream", source_path="bounded.jsonl")

    assert len(sessions) == 1
    assert sessions[0].messages[0].text == "bounded"
