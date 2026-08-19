"""Multi-chunk Claude Code route-equivalence witness (polylogue-4987i residual).

``polylogue-4987i`` fixed eager/streaming ``session_events`` order divergence
for multi-session interleaved Claude Code JSONL, but its 2026-08-06 reopening
note recorded that the closure "explicitly lacked a genuine multi-chunk,
subagent-interleaved Claude Code witness through eager, streaming, alternate
chunk boundaries, and source replay". This module is that witness.

Four production routes are compared against one canonical normalized snapshot
of the same wire bytes:

``eager``
    Whole-file decode into a list, then :func:`parse_payload` -- the route a
    full raw replay/reindex takes.
``streaming``
    ``_iter_json_stream`` over a real binary handle feeding
    :func:`parse_stream_payload` as a one-pass iterator -- the memory-bounded
    route ``sources/live/batch.py`` takes for JSONL above the streaming
    threshold.
``chunked-<n>``
    The same streaming route, but the handle is a buffered reader whose
    underlying raw stream never returns more than ``n`` bytes per read. The
    fixture's records are 175-750 bytes, so every ``n`` below asserts on read
    boundaries that fall *inside* JSONL records and must be reassembled by the
    production decoder rather than by the test.
``source-replay``
    :func:`iter_source_sessions_with_raw` -- filesystem acquisition through
    the real source walk, which is what a live cursor replay drives.

The fixture is deliberately the interleaved family fixture: two sessions whose
records alternate (``main`` at wire rows 0-3, ``other`` at 4, ``main`` at 5-10,
``other`` at 11, ``main`` at 12), so no route can see either session as one
contiguous run.

Anti-vacuity: the comparison is over the full ``model_dump(mode="json")``
serialization of every parsed session, so dropping ``order_session_events``
from either path, resetting per-chunk record indexes, or losing the shared
UUID/sidecar accumulator state changes the canonical snapshot and fails
:func:`test_every_production_route_produces_one_canonical_normalization`.
:func:`test_chunked_reader_actually_splits_records` proves the chunked routes
are not silently degenerating into whole-line reads.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.sources.decoders import _iter_json_stream
from polylogue.sources.dispatch import detect_provider, parse_payload, parse_stream_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.source_parsing import iter_source_sessions_with_raw

_FIXTURE = Path(__file__).parents[2] / "fixtures" / "claude-code" / "claude-normalization-main.jsonl"
_FALLBACK_ID = "claude-normalization-main"

#: Read sizes smaller than the fixture's smallest record (175 bytes), so each
#: one forces at least one buffer boundary inside a JSONL record.
_SPLIT_READ_SIZES = (1, 7, 64, 173)


class _CappedRawStream(io.RawIOBase):
    """A raw byte stream that never returns more than ``limit`` bytes per read.

    ``io.BufferedReader`` wrapping this yields the same lines a plain file
    does, but only by reassembling them across reads -- which is exactly the
    production decoder behaviour under test.
    """

    def __init__(self, data: bytes, limit: int) -> None:
        super().__init__()
        self._data = data
        self._limit = limit
        self._position = 0
        self.read_sizes: list[int] = []

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        span = min(len(buffer), self._limit, len(self._data) - self._position)
        if span <= 0:
            return 0
        buffer[:span] = self._data[self._position : self._position + span]
        self._position += span
        self.read_sizes.append(span)
        return span


def _capped_handle(data: bytes, limit: int) -> tuple[io.BufferedReader, _CappedRawStream]:
    raw = _CappedRawStream(data, limit)
    return io.BufferedReader(raw, buffer_size=max(limit, 1)), raw


def _normalized(sessions: list[ParsedSession]) -> str:
    """Canonical JSON text for a parsed session list -- the comparison unit."""
    return json.dumps(
        [session.model_dump(mode="json") for session in sessions],
        sort_keys=True,
        separators=(",", ":"),
    )


def _eager_route(data: bytes) -> list[ParsedSession]:
    records = list(_iter_json_stream(io.BytesIO(data), _FIXTURE.name))
    return parse_payload(Provider.CLAUDE_CODE, records, _FALLBACK_ID, source_path=str(_FIXTURE))


def _streaming_route(data: bytes) -> list[ParsedSession]:
    with io.BytesIO(data) as handle:
        return parse_stream_payload(
            Provider.CLAUDE_CODE,
            _iter_json_stream(handle, _FIXTURE.name),
            _FALLBACK_ID,
            source_path=str(_FIXTURE),
        )


def _chunked_route(data: bytes, limit: int) -> list[ParsedSession]:
    handle, _raw = _capped_handle(data, limit)
    with handle:
        return parse_stream_payload(
            Provider.CLAUDE_CODE,
            _iter_json_stream(handle, _FIXTURE.name),
            _FALLBACK_ID,
            source_path=str(_FIXTURE),
        )


def _source_replay_route(tmp_path: Path, data: bytes) -> list[ParsedSession]:
    replay_path = tmp_path / _FIXTURE.name
    replay_path.write_bytes(data)
    return [
        session
        for _raw, session in iter_source_sessions_with_raw(
            Source(name="claude-code", path=replay_path),
            capture_raw=False,
        )
    ]


def test_fixture_is_genuinely_multi_chunk_and_claude_code() -> None:
    """Guard the premise: interleaved sessions, records wider than every split."""
    records = [json.loads(line) for line in _FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    assert detect_provider(records) is Provider.CLAUDE_CODE

    session_ids = [record["sessionId"] for record in records]
    assert set(session_ids) == {"claude-normalization-main", "claude-normalization-other"}
    # More runs than distinct sessions == no session is one contiguous chunk.
    runs = [session_ids[0]] + [
        current for previous, current in zip(session_ids, session_ids[1:], strict=False) if current != previous
    ]
    assert len(runs) > len(set(session_ids))
    assert runs.count("claude-normalization-main") >= 3

    line_widths = [len(line) for line in _FIXTURE.read_bytes().splitlines() if line]
    assert min(line_widths) > max(_SPLIT_READ_SIZES)


def test_chunked_reader_actually_splits_records() -> None:
    """Anti-vacuity for the chunked routes: reads land inside records."""
    data = _FIXTURE.read_bytes()
    for limit in _SPLIT_READ_SIZES:
        handle, raw = _capped_handle(data, limit)
        with handle:
            consumed = list(_iter_json_stream(handle, _FIXTURE.name))
        assert consumed, f"limit {limit} decoded no records"
        assert raw.read_sizes, f"limit {limit} performed no reads"
        assert max(raw.read_sizes) <= limit
        # Reassembly happened in the decoder, not the fixture: strictly more
        # reads than records means at least one record spanned a boundary.
        assert len(raw.read_sizes) > len(consumed)


def test_every_production_route_produces_one_canonical_normalization(tmp_path: Path) -> None:
    """Eager, streaming, split-boundary, and replay routes agree byte for byte."""
    data = _FIXTURE.read_bytes()

    canonical = _normalized(_eager_route(data))
    routes: dict[str, str] = {
        "eager": canonical,
        "streaming": _normalized(_streaming_route(data)),
        "source-replay": _normalized(_source_replay_route(tmp_path, data)),
    }
    for limit in _SPLIT_READ_SIZES:
        routes[f"chunked-{limit}"] = _normalized(_chunked_route(data, limit))

    divergent = sorted(name for name, rendered in routes.items() if rendered != canonical)
    assert not divergent, f"routes diverged from the eager normalization: {divergent}"

    # The snapshot itself must be non-trivial, or agreement is vacuous.
    decoded = json.loads(canonical)
    assert [session["provider_session_id"] for session in decoded] == [
        "claude-normalization-main",
        "claude-normalization-other",
    ]
    main = decoded[0]
    assert len(main["messages"]) == 9
    assert [event["event_type"] for event in main["session_events"]] == [
        "message_usage",
        "background_task_completion",
        "message_usage",
        "claude_parse_coverage",
    ]


@pytest.mark.parametrize("limit", _SPLIT_READ_SIZES)
def test_split_boundary_route_preserves_session_event_order(limit: int) -> None:
    """The 4987i invariant specifically, per split size, not just in aggregate."""
    data = _FIXTURE.read_bytes()
    eager_main = next(
        session for session in _eager_route(data) if session.provider_session_id == "claude-normalization-main"
    )
    chunked_main = next(
        session for session in _chunked_route(data, limit) if session.provider_session_id == "claude-normalization-main"
    )
    assert [(event.event_type, event.source_message_provider_id) for event in chunked_main.session_events] == [
        (event.event_type, event.source_message_provider_id) for event in eager_main.session_events
    ]
