"""Root-field envelopes of JSON documents, read without holding the document.

Structural signatures (source-class recognition, sidecar dispatch identity)
decide from a document's root fields: their presence, type and short leading
text. :func:`top_level_envelopes` streams a document and keeps exactly that,
so a signature gives the same answer for a document of any size. No string is
materialized beyond a bounded prefix, so one huge scalar costs no memory.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import IO, Protocol

#: Characters of a top-level string an envelope keeps. The source-class
#: signatures read only the type and short leading text of root fields.
ENVELOPE_TEXT_PREFIX_CHARS = 4096

#: Raw bytes of any string token passed to the tokenizer. Twelve raw bytes
#: can encode one character (a surrogate-pair escape), so this keeps at least
#: :data:`ENVELOPE_TEXT_PREFIX_CHARS` characters of every string.
_STRING_PREFIX_BYTES = 12 * ENVELOPE_TEXT_PREFIX_CHARS

_READ_BYTES = 1024 * 1024

_ESCAPE_TOKEN = re.compile(rb'\\(?:u([0-9a-fA-F]{4})|["\\/bfnrt])')


class _Readable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


def _prefix_cut(content: bytes) -> int:
    """Largest cut of raw string content that splits no escape, pair or UTF-8 sequence."""
    tokens = list(_ESCAPE_TOKEN.finditer(content))
    last_end = tokens[-1].end() if tokens else 0
    cut = len(content)
    dangling = content.find(b"\\", last_end)
    if dangling >= 0:
        cut = dangling
    for token in reversed(tokens):
        if token.end() < cut:
            break
        unit = token.group(1)
        if token.end() == cut and unit is not None and 0xD800 <= int(unit, 16) <= 0xDBFF:
            cut = token.start()
            break
    index = cut - 1
    while index >= 0 and cut - index <= 4 and 0x80 <= content[index] <= 0xBF:
        index -= 1
    if index >= 0:
        lead = content[index]
        width = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
        if index + width > cut:
            cut = index
    return cut


class _PrefixStringReader:
    """Pass a JSON byte stream through, keeping only a prefix of each string token.

    Quote parity decides string boundaries. A string longer than
    :data:`_STRING_PREFIX_BYTES` raw bytes reaches the tokenizer cut at a
    boundary that splits no escape or character; the rest is skipped unread
    by the tokenizer. The root fields a signature reads keep their full
    leading text.
    """

    def __init__(self, source: _Readable) -> None:
        self._source = source
        self._in_string = False
        self._backslashes = 0
        self._string = bytearray()
        self._skipping = False
        self._eof = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            # The tokenizer probes with an empty read to learn the stream type.
            return b""
        out = bytearray()
        while not out and not self._eof:
            chunk = self._source.read(_READ_BYTES)
            if not chunk:
                self._eof = True
                if self._in_string and not self._skipping:
                    out += self._string
                break
            self._consume(chunk, out)
        return bytes(out)

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def _consume(self, data: bytes, out: bytearray) -> None:
        position = 0
        while position < len(data):
            if not self._in_string:
                quote = data.find(b'"', position)
                if quote < 0:
                    out += data[position:]
                    return
                out += data[position : quote + 1]
                self._in_string = True
                self._backslashes = 0
                self._string = bytearray()
                self._skipping = False
                position = quote + 1
                continue
            end = self._string_end(data, position)
            piece = data[position:end] if end >= 0 else data[position:]
            if not self._skipping:
                self._string += piece
                if len(self._string) > _STRING_PREFIX_BYTES:
                    out += self._string[: _prefix_cut(bytes(self._string[:_STRING_PREFIX_BYTES]))]
                    self._string = bytearray()
                    self._skipping = True
            if end < 0:
                return
            if not self._skipping:
                out += self._string
            out += b'"'
            self._in_string = False
            self._string = bytearray()
            position = end + 1

    def _string_end(self, data: bytes, start: int) -> int:
        """Index of the closing quote of the open string in ``data``, or -1."""
        search = start
        while (quote := data.find(b'"', search)) >= 0:
            run = 0
            index = quote - 1
            while index >= start and data[index] == 0x5C:
                run += 1
                index -= 1
            if index < start:
                run += self._backslashes
            if run % 2 == 0:
                return quote
            search = quote + 1
        segment = data[start:]
        tail = len(segment) - len(segment.rstrip(b"\\"))
        self._backslashes = tail + (self._backslashes if tail == len(segment) else 0)
        return -1


class _LineSource:
    """Physical lines of a byte stream, each readable as its own stream."""

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle
        self._buffer = b""
        self._eof = False

    def next_line(self) -> _Line | None:
        if not self._buffer and not self._eof:
            self._fill()
        if not self._buffer and self._eof:
            return None
        return _Line(self)

    def _fill(self) -> None:
        chunk = self._handle.read(_READ_BYTES)
        if chunk:
            self._buffer += chunk
        else:
            self._eof = True


class _Line:
    """One physical line; reads stop at its newline."""

    def __init__(self, source: _LineSource) -> None:
        self._source = source
        self._done = False

    def read(self, size: int = -1) -> bytes:
        if self._done or size == 0:
            return b""
        source = self._source
        if not source._buffer and not source._eof:
            source._fill()
        if not source._buffer:
            self._done = True
            return b""
        newline = source._buffer.find(b"\n")
        if newline >= 0:
            data, source._buffer = source._buffer[:newline], source._buffer[newline + 1 :]
            self._done = True
            return data
        data, source._buffer = source._buffer, b""
        return data

    def drain(self) -> None:
        while self.read(_READ_BYTES):
            pass


def _envelope_scalar(value: object) -> object:
    if isinstance(value, str) and len(value) > ENVELOPE_TEXT_PREFIX_CHARS:
        return value[:ENVELOPE_TEXT_PREFIX_CHARS]
    return value


def _envelopes(events: Iterator[tuple[str, object]], *, expand_arrays: bool) -> Iterator[object]:
    depth = 0
    root: object = None
    element: object = None
    key: str | None = None
    element_key: str | None = None
    expanding = False
    for event, value in events:
        if event in ("start_map", "start_array"):
            placeholder: object = {} if event == "start_map" else []
            if depth == 0:
                root = placeholder
                expanding = expand_arrays and event == "start_array"
            elif depth == 1 and expanding:
                element = placeholder
            elif depth == 1 and isinstance(root, dict) and key is not None:
                root[key] = placeholder
            elif depth == 2 and expanding and isinstance(element, dict) and element_key is not None:
                element[element_key] = placeholder
            depth += 1
            continue
        if event in ("end_map", "end_array"):
            depth -= 1
            if depth == 0 and not expanding:
                yield root
            elif depth == 1 and expanding:
                yield element
            continue
        if event == "map_key":
            if depth == 1:
                key = str(value)
            elif depth == 2 and expanding:
                element_key = str(value)
            continue
        scalar = _envelope_scalar(value)
        if depth == 0 or (depth == 1 and expanding):
            yield scalar
        elif depth == 1 and isinstance(root, dict) and key is not None:
            root[key] = scalar
        elif depth == 2 and expanding and isinstance(element, dict) and element_key is not None:
            element[element_key] = scalar


def top_level_envelopes(handle: IO[bytes], *, expand_arrays: bool) -> Iterator[object]:
    """Stream one JSON document's envelope, or one per element of an array document.

    An object's envelope keeps its keys with scalar values (strings as a
    leading prefix) and a typed empty placeholder for container values.
    Numbers are read exactly, so no magnitude makes a document unreadable.
    A malformed document raises ``ijson.JSONError``.
    """
    import ijson

    events = ijson.basic_parse(_PrefixStringReader(handle), use_float=False)
    yield from _envelopes(events, expand_arrays=expand_arrays)


def jsonl_record_envelopes(handle: IO[bytes]) -> Iterator[object]:
    """Stream the envelope of each physical line that holds exactly one JSON value.

    A blank or malformed line is skipped, as the JSONL decoder skips it; a
    line holding two values is malformed, not two records.
    """
    import ijson

    lines = _LineSource(handle)
    while (line := lines.next_line()) is not None:
        try:
            values = list(
                _envelopes(ijson.basic_parse(_PrefixStringReader(line), use_float=False), expand_arrays=False)
            )
        except (ijson.JSONError, UnicodeDecodeError, ArithmeticError):
            line.drain()
            continue
        line.drain()
        if len(values) == 1:
            yield values[0]


__all__ = ["ENVELOPE_TEXT_PREFIX_CHARS", "jsonl_record_envelopes", "top_level_envelopes"]
