"""The exact JSON lexer fed lexeme-aligned chunks: identical events, bounded buffer.

``LexemeAlignedReader`` only changes where the chunks handed to
``ijson.backends.python`` end. These tests hold both halves of that claim:
the event stream is the same as the unaligned stream for every edge case and
chunking, and the lexer's buffer no longer keeps the consumed document.
"""

from __future__ import annotations

import io
import json
import random
import tracemalloc
from collections.abc import Iterator

import pytest
from ijson.backends import python as exact_backend

from polylogue.core.json_envelope import LexemeAlignedReader, _PrefixStringReader

_EDGE_DOCUMENTS: tuple[bytes, ...] = (
    b'{"a": 1}',
    b"[1, 2.5, -0, 1e999, 123456789012345678901234567890, 0.1000000000000000055511151231257827]",
    b'{"big": ' + b"9" * 9000 + b"}",
    b'{"exp": 1e123456789012345678901234567890}',
    b'{"s": "\\ud800 lone", "t": "\\udc00\\ud800", "pair": "\\ud83d\\ude00"}',
    b'{"esc": "quote \\" backslash \\\\ tab \\t uni \\u00e9", "k\\"ey": [true, false, null]}',
    b'{"tail": "ends in a backslash pair \\\\"}',
    b'  [ {"a" :[ ] , "b":{ }} ,"\\\\" ]  ',
    b'{"n": NaN, "i": Infinity, "m": -Infinity}',
    b'{"dup": 1, "dup": 2}',
    b'{"long": "' + b"y\\n" * 40000 + b'"}',
    b'{"wide": "' + b"z" * 200_000 + b'"}',
    b'{"deep": ' + b"[" * 200 + b"]" * 200 + b"}",
    '{"utf": "zażółć ☃ \U0001f600"}'.encode(),
    b'{"a": 1} {"b": 2}',
    b'{"a": "unterminated',
    b'{"a": tru}',
    b'{"a": 1,}',
    b'{"a" "b"}',
    b"",
    b"   ",
)


def _random_documents(count: int) -> Iterator[bytes]:
    rng = random.Random(20261007)
    alphabet = b'{}[],:" \\\n\tabtrue0123456789.eE-+nul'
    for _ in range(count):
        yield bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 48)))


class _Chunked:
    def __init__(self, payload: bytes, size: int) -> None:
        self._source = io.BytesIO(payload)
        self._size = size

    def read(self, size: int = -1) -> bytes:
        del size
        return self._source.read(self._size)


def _events(payload: bytes, *, aligned: bool, chunk: int) -> list[tuple[str, object]]:
    reader: object = _PrefixStringReader(_Chunked(payload, chunk), scalar_values=True)
    if aligned:
        # A small flush bound also drives the no-lexeme-end pass-through.
        reader = LexemeAlignedReader(reader, flush_bytes=max(64, 2 * chunk))  # type: ignore[arg-type]
    try:
        return list(exact_backend.basic_parse(reader))
    except Exception as failure:
        return [("failure", type(failure).__name__)]


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 64, 1024 * 1024])
def test_aligned_chunks_leave_every_event_unchanged(chunk: int) -> None:
    """Edge numbers, lone surrogates, escapes, malformed text: the same events or the same refusal.

    Anti-vacuity: cut a chunk inside a string or after an escape backslash
    (drop the in-string escape skip in ``_scan``) and an escaped quote ends
    the string early, so the events differ.
    """
    for payload in (*_EDGE_DOCUMENTS, *_random_documents(1500)):
        assert _events(payload, aligned=True, chunk=chunk) == _events(payload, aligned=False, chunk=chunk), payload[
            :120
        ]


def _document(messages: int) -> bytes:
    out = io.BytesIO()
    out.write(b'{"session_id":"bounded","messages":[')
    for index in range(messages):
        if index:
            out.write(b",")
        out.write(json.dumps({"role": "user", "content": f"turn {index} " + "x" * 3000}).encode())
    out.write(b"]}")
    return out.getvalue()


def _lexer_peak(payload: bytes) -> int:
    tracemalloc.start()
    try:
        reader = LexemeAlignedReader(_PrefixStringReader(io.BytesIO(payload), scalar_values=True))
        for _event in exact_backend.basic_parse(reader):
            pass
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_exact_lexer_memory_does_not_track_document_size() -> None:
    """Twice the document costs far less extra traced memory than its extra bytes.

    Anti-vacuity: feed ``_PrefixStringReader`` to the lexer directly and its
    buffer keeps every consumed chunk, about two bytes per input byte.
    """
    small, large = _document(2500), _document(5000)
    small_peak, large_peak = _lexer_peak(small), _lexer_peak(large)
    assert large_peak - small_peak < (len(large) - len(small)) // 4, (len(small), small_peak, len(large), large_peak)
