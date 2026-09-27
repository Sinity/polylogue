"""Root-field envelopes of JSON documents, read without holding the document.

Structural signatures (source-class recognition, sidecar dispatch identity)
decide from a document's root fields: their presence, type and short leading
text. :func:`top_level_envelopes` streams a document and keeps exactly the
root fields its caller declares, so a signature gives the same answer for a
document of any size or width. No string is materialized beyond a bounded
prefix, except a declared identity field, which is read whole or refused at
SQLite's value limit.
"""

from __future__ import annotations

import codecs
import re
import sqlite3
import sys
from collections.abc import Iterator
from functools import cache
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

#: Tokens of a skipped string suffix worth examining: complete valid escapes
#: pass; a lone backslash (an invalid or not-yet-complete escape) or a raw
#: control character is not valid JSON string content.
_SKIPPED_TOKEN = re.compile(rb'\\(?:u[0-9a-fA-F]{4}|["\\/bfnrt])|\\|[\x00-\x1f]')

#: Injected into the tokenizer's view of a string whose skipped suffix is not
#: valid JSON, so the tokenizer rejects the document as the full decoder does.
_INVALID_ESCAPE = b"\\q"

#: A run of bytes that can belong to a JSON number token outside strings.
_NUMBER_RUN = re.compile(rb"[-+0-9.eE]+")

#: Passed in place of an over-long integer token so the tokenizer rejects it
#: as malformed instead of converting it.
_INVALID_NUMBER_END = b"x"

#: Marks the end of a number token's integer part.
_NUMBER_NON_INTEGER = re.compile(rb"[.eE]")

#: Raw bytes of a number token the tokenizer sees exactly; longer tokens are
#: passed as a placeholder of the same JSON type. It exceeds Python's default
#: integer conversion limit, so every integer the decoder accepts is exact.
_NUMBER_VIEW_BYTES = 8192


class EnvelopeValueTooLargeError(ValueError):
    """A declared identity field exceeds SQLite's maximum value length.

    Such a value cannot be stored or joined, so it is refused by name rather
    than shortened into a different identifier.
    """

    def __init__(self, field: str, size: int, limit: int) -> None:
        super().__init__(f"root field {field!r} is {size} bytes, beyond the {limit}-byte SQLite value limit")
        self.field = field
        self.size = size
        self.limit = limit


class _TruncatedText(str):
    """A string field kept as its leading prefix.

    ``raw_bytes`` is the raw length of its token and ``ordinal`` its position
    among the document's string tokens, so it can be re-read whole.
    """

    raw_bytes: int
    ordinal: int


@cache
def _sqlite_value_limit() -> int:
    connection = sqlite3.connect(":memory:")
    try:
        return connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
    finally:
        connection.close()


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

    def __init__(self, source: _Readable, *, whole_ordinals: frozenset[int] = frozenset()) -> None:
        self._source = source
        self._whole_ordinals = whole_ordinals
        self._in_string = False
        self._backslashes = 0
        self._string = bytearray()
        self._skipping = False
        self._eof = False
        #: Ordinal of the current string token, counted from 1 in stream order;
        #: the tokenizer reports keys and string values in the same order.
        self._ordinal = 0
        #: Raw byte length of each string token passed on as a prefix only.
        self.truncated: dict[int, int] = {}
        self._skipped_bytes = 0
        self._skip_carry = b""
        self._skip_invalid = False
        #: State of the number token being passed outside strings, which may
        #: span chunks.
        self._number_open = False
        self._number_view = bytearray()
        self._number_long = False
        self._number_is_integer = True
        self._number_in_integer_part = True
        self._number_digits = 0
        self._skip_decoder: codecs.IncrementalDecoder | None = None

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
                elif self._number_open:
                    self._end_number(out)
                break
            self._consume(chunk, out)
        return bytes(out)

    def _pass_structure(self, segment: bytes, out: bytearray) -> None:
        """Pass bytes outside any string through, number tokens bounded.

        Each number token is held back until it ends. A short token reaches
        the tokenizer exactly. An integer longer than Python's conversion
        limit, which the JSON decoder refuses with ``ValueError``, reaches it
        as a malformed token, so the record is rejected as the decoder rejects
        it and the C tokenizer never converts the value. Any other token
        longer than :data:`_NUMBER_VIEW_BYTES` reaches it as a placeholder of
        the same JSON type: an envelope is a view of presence and type, and a
        signature never reads a number's value, so a number of any length
        costs bounded memory.
        """
        position = 0
        for run in _NUMBER_RUN.finditer(segment):
            if self._number_open and run.start() > 0:
                self._end_number(out)
            out += segment[position : run.start()]
            if not self._number_open:
                self._start_number()
            self._extend_number(run.group())
            position = run.end()
            if run.end() < len(segment):
                self._end_number(out)
        if position < len(segment):
            if self._number_open:
                self._end_number(out)
            out += segment[position:]

    def _start_number(self) -> None:
        self._number_open = True
        self._number_view = bytearray()
        self._number_long = False
        self._number_is_integer = True
        self._number_in_integer_part = True
        self._number_digits = 0

    def _extend_number(self, token: bytes) -> None:
        if self._number_in_integer_part:
            mark = _NUMBER_NON_INTEGER.search(token)
            integer_part = token if mark is None else token[: mark.start()]
            self._number_digits += len(integer_part) - len(integer_part.translate(None, b"0123456789"))
            if mark is not None:
                self._number_in_integer_part = False
                self._number_is_integer = False
        if not self._number_long:
            self._number_view += token
            if len(self._number_view) > _NUMBER_VIEW_BYTES:
                self._number_long = True
                self._number_view = bytearray()

    def _end_number(self, out: bytearray) -> None:
        self._number_open = False
        limit = sys.get_int_max_str_digits()
        if self._number_is_integer and limit and self._number_digits > limit:
            out += _INVALID_NUMBER_END
        elif self._number_long:
            out += b"0" if self._number_is_integer else b"0.0"
        else:
            out += self._number_view
        self._number_view = bytearray()

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
                    self._pass_structure(data[position:], out)
                    return
                self._pass_structure(data[position : quote + 1], out)
                self._in_string = True
                self._backslashes = 0
                self._string = bytearray()
                self._skipping = False
                self._ordinal += 1
                position = quote + 1
                continue
            end = self._string_end(data, position)
            piece = data[position:end] if end >= 0 else data[position:]
            if self._skipping:
                self._skipped_bytes += len(piece)
                self._validate_skipped(piece, out)
            else:
                self._string += piece
                if len(self._string) > _STRING_PREFIX_BYTES and self._ordinal not in self._whole_ordinals:
                    cut = _prefix_cut(bytes(self._string[:_STRING_PREFIX_BYTES]))
                    out += self._string[:cut]
                    self._skipped_bytes = len(self._string)
                    self._skip_carry = b""
                    self._skip_invalid = False
                    self._skip_decoder = codecs.getincrementaldecoder("utf-8")()
                    self._validate_skipped(bytes(self._string[cut:]), out)
                    self._string = bytearray()
                    self._skipping = True
            if end < 0:
                return
            if self._skipping:
                if not self._skip_invalid and self._skip_carry:
                    # An escape left incomplete by the closing quote.
                    self._skip_invalid = True
                    out += _INVALID_ESCAPE
                if not self._skip_invalid and self._skip_decoder is not None:
                    try:
                        self._skip_decoder.decode(b"", final=True)
                    except UnicodeDecodeError:
                        # A UTF-8 sequence left incomplete by the closing quote.
                        self._skip_invalid = True
                        out += _INVALID_ESCAPE
                self.truncated[self._ordinal] = self._skipped_bytes
            else:
                out += self._string
            out += b'"'
            self._in_string = False
            self._string = bytearray()
            position = end + 1

    def _validate_skipped(self, piece: bytes, out: bytearray) -> None:
        """Check that a skipped string suffix is valid JSON string content.

        The tokenizer never sees the suffix, so an invalid UTF-8 sequence,
        invalid escape or raw control character there is injected as an
        invalid escape: the document is then rejected exactly as the full
        decoder rejects it.
        """
        if self._skip_invalid:
            return
        if self._skip_decoder is not None:
            try:
                self._skip_decoder.decode(piece)
            except UnicodeDecodeError:
                self._skip_invalid = True
                out += _INVALID_ESCAPE
                return
        buffer = self._skip_carry + piece
        self._skip_carry = b""
        for match in _SKIPPED_TOKEN.finditer(buffer):
            token = match.group()
            if len(token) > 1:
                continue
            if token == b"\\" and len(buffer) - match.start() < 6:
                # Possibly an escape the next chunk completes.
                self._skip_carry = buffer[match.start() :]
                return
            self._skip_invalid = True
            out += _INVALID_ESCAPE
            return

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


#: Set in an envelope whose object had root keys outside the declared fields,
#: so a caller can still tell an empty object from one with other content,
#: without the envelope holding those keys.
UNDECLARED_FIELDS = "\x00undeclared"


def _envelope_scalar(value: object, raw_bytes: int | None, ordinal: int) -> object:
    if isinstance(value, str) and (raw_bytes is not None or len(value) > ENVELOPE_TEXT_PREFIX_CHARS):
        text = _TruncatedText(value[:ENVELOPE_TEXT_PREFIX_CHARS])
        text.raw_bytes = raw_bytes if raw_bytes is not None else len(value.encode("utf-8", "surrogatepass"))
        text.ordinal = ordinal
        return text
    return value


def _envelopes(
    events: Iterator[tuple[str, object]],
    reader: _PrefixStringReader,
    *,
    expand_arrays: bool,
    fields: frozenset[str],
) -> Iterator[object]:
    depth = 0
    root: object = None
    element: object = None
    key: str | None = None
    element_key: str | None = None
    expanding = False
    ordinal = 0
    for event, value in events:
        if event in ("map_key", "string"):
            ordinal += 1
        if event in ("start_map", "start_array"):
            placeholder: object = {} if event == "start_map" else []
            if depth == 0:
                root = placeholder
                expanding = expand_arrays and event == "start_array"
            elif depth == 1 and expanding:
                element = placeholder
            elif depth == 1 and isinstance(root, dict) and key in fields:
                root[key] = placeholder
            elif depth == 2 and expanding and isinstance(element, dict) and element_key in fields:
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
                if key not in fields and isinstance(root, dict):
                    root[UNDECLARED_FIELDS] = True
            elif depth == 2 and expanding:
                element_key = str(value)
                if element_key not in fields and isinstance(element, dict):
                    element[UNDECLARED_FIELDS] = True
            continue
        scalar = _envelope_scalar(value, reader.truncated.get(ordinal) if event == "string" else None, ordinal)
        if depth == 0 or (depth == 1 and expanding):
            yield scalar
        elif depth == 1 and isinstance(root, dict) and key in fields:
            root[key] = scalar
        elif depth == 2 and expanding and isinstance(element, dict) and element_key in fields:
            element[element_key] = scalar


def _whole_string(handle: IO[bytes], ordinal: int) -> str:
    """Re-read the string token at ``ordinal`` of a seekable document whole.

    Every other string is still passed on as a prefix only, so the re-read
    costs the memory of this one value, however large its neighbours are.
    """
    import ijson

    handle.seek(0)
    reader = _PrefixStringReader(handle, whole_ordinals=frozenset({ordinal}))
    seen = 0
    for event, value in ijson.basic_parse(reader, use_float=False):
        if event in ("map_key", "string"):
            seen += 1
            if seen == ordinal and isinstance(value, str):
                return value
    raise ValueError(f"string token {ordinal} vanished between two reads of the same document")


def top_level_envelopes(
    handle: IO[bytes],
    *,
    expand_arrays: bool,
    fields: frozenset[str],
    whole_fields: frozenset[str] = frozenset(),
) -> Iterator[object]:
    """Stream one JSON document's envelope, or one per element of an array document.

    An object's envelope keeps only the root keys named in ``fields``, with
    scalar values (strings as a leading prefix) and a typed empty placeholder
    for container values, so a document of any width costs the same memory.
    A string in ``whole_fields`` is an identity used as an exact join key: it
    is re-read whole from ``handle`` (which must then be seekable), or refused
    with :class:`EnvelopeValueTooLargeError` beyond SQLite's value limit -- never
    shortened. Numbers are read exactly, so no magnitude makes a document
    unreadable. A malformed document raises ``ijson.JSONError``.
    """
    import ijson

    reader = _PrefixStringReader(handle)
    events = ijson.basic_parse(reader, use_float=False)
    if not whole_fields or expand_arrays:
        # Streamed: an array document's elements are never held together.
        yield from _envelopes(events, reader, expand_arrays=expand_arrays, fields=fields)
        return
    envelopes = list(_envelopes(events, reader, expand_arrays=False, fields=fields))
    if envelopes:
        for envelope in envelopes:
            if not isinstance(envelope, dict):
                continue
            for field in sorted(whole_fields & envelope.keys()):
                value = envelope[field]
                if not isinstance(value, _TruncatedText):
                    continue
                limit = _sqlite_value_limit()
                if value.raw_bytes > limit:
                    raise EnvelopeValueTooLargeError(field, value.raw_bytes, limit)
                envelope[field] = _whole_string(handle, value.ordinal)
    yield from envelopes


def jsonl_record_envelopes(handle: IO[bytes], *, fields: frozenset[str]) -> Iterator[object]:
    """Stream the envelope of each physical line that holds exactly one JSON value.

    A blank or malformed line is skipped, as the JSONL decoder skips it --
    including a line whose integer exceeds Python's conversion limit, which
    the decoder rejects per line with ``ValueError``. A line holding two
    values is malformed, not two records.
    """
    import ijson

    lines = _LineSource(handle)
    while (line := lines.next_line()) is not None:
        reader = _PrefixStringReader(line)
        try:
            values = list(
                _envelopes(ijson.basic_parse(reader, use_float=False), reader, expand_arrays=False, fields=fields)
            )
        except (ijson.JSONError, UnicodeDecodeError, ArithmeticError, ValueError):
            line.drain()
            continue
        line.drain()
        if len(values) == 1:
            yield values[0]


__all__ = [
    "ENVELOPE_TEXT_PREFIX_CHARS",
    "UNDECLARED_FIELDS",
    "EnvelopeValueTooLargeError",
    "jsonl_record_envelopes",
    "top_level_envelopes",
]
