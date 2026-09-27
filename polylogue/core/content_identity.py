"""Structural content identity for decoded provider values.

Byte equality and canonical-JSON digests both answer the wrong question about
an export member: a provider re-serializing the same conversation with
different separators, key order, or ``1`` written as ``1.0`` produces different
bytes for identical content. The identity below is computed over the *decoded*
value under the provider value contract, so it is stable across serialization
while it still separates values JSON itself distinguishes -- ``true`` from
``1``, ``"1"`` from ``1``, an absent key from a null one.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import tempfile
import unicodedata
from collections.abc import Iterator
from decimal import Decimal
from contextlib import closing
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import IO, Protocol

from polylogue.core.text_identity import nfc

_CONTENT_IDENTITY_DOMAIN = b"polylogue:member-content:v2\0"

_UTF8_BOM = b"\xef\xbb\xbf"

#: Bytes read per step of the streaming identity. A pacing window only: the
#: digest is the same for every window size.
_STREAM_READ_BYTES = 1024 * 1024


def _encode(value: object, sink: _Sink) -> None:
    if value is None:
        sink.update(b"z;")
        return
    # Ordered before ``int``: ``bool`` is a subclass of ``int`` and ``True``
    # must never share an identity with ``1``.
    if isinstance(value, bool):
        sink.update(b"b1;" if value else b"b0;")
        return
    if isinstance(value, Decimal):
        _encode_decimal(value, sink)
        return
    if isinstance(value, int):
        sink.update(b"i%d;" % value)
        return
    if isinstance(value, float):
        _encode_float(value, sink)
        return
    if isinstance(value, str):
        _encode_text(b"s", value, sink)
        return
    if isinstance(value, (list, tuple)):
        sink.update(b"a[")
        for item in value:
            _encode(item, sink)
        sink.update(b"]")
        return
    if isinstance(value, dict):
        entries: list[tuple[str, bytes]] = []
        for key, item in value.items():
            child = sha256()
            _encode(item, child)
            entries.append((nfc(str(key)), child.digest()))
        _encode_object_entries(entries, sink)
        return
    raise TypeError(f"value of type {type(value).__name__} has no structural content identity")


class _Sink(Protocol):
    """Anything with ``update(bytes)``: a hashlib object in practice."""

    def update(self, data: bytes, /) -> None: ...


def _encode_object_entries(entries: list[tuple[str, bytes]], sink: _Sink) -> None:
    # An object is its sorted (key, value-digest) entries. Each value is hashed
    # on its own, so an object never has to hold its members' encodings to
    # order them -- the property that lets the byte stream be hashed without
    # decoding the whole document. Two raw keys that normalize to the same
    # text are both kept, ordered by their value digests.
    entries.sort()
    sink.update(b"o%d;" % len(entries))
    for key, digest in entries:
        _encode_text(b"k", key, sink)
        sink.update(digest)


def _encode_text(tag: bytes, value: str, sink: _Sink) -> None:
    encoded = nfc(value).encode("utf-8", errors="surrogatepass")
    sink.update(b"%s%d:" % (tag, len(encoded)))
    sink.update(encoded)
    sink.update(b";")


def _encode_decimal(value: Decimal, sink: _Sink) -> None:
    if not value.is_finite():
        raise ValueError("a non-finite number has no structural content identity")
    if value == value.to_integral_value():
        sink.update(b"i%d;" % int(value))
        return
    _encode_float(float(value), sink)


def _encode_float(value: float, sink: _Sink) -> None:
    if not isfinite(value):
        raise ValueError("a non-finite number has no structural content identity")
    # The provider value contract has one numeric type. An integral float is
    # the same number as the integer it equals, so ``1.0`` and ``1`` share an
    # identity; a fractional value keeps the shortest round-trip form, which
    # is equal exactly when the two floats are equal.
    if value.is_integer():
        sink.update(b"i%d;" % int(value))
        return
    sink.update(b"f%s;" % repr(value).encode("ascii"))


def structural_content_identity(value: object) -> str:
    """Return the content identity digest of one decoded provider value."""
    digest = sha256(_CONTENT_IDENTITY_DOMAIN)
    _encode(value, digest)
    return digest.hexdigest()


class ContentIdentityRefusal(Exception):  # noqa: N818 -- a refusal, not an error in the payload
    """A single token exceeds the archive's physical value limit.

    SQLite cannot store a value longer than its compiled length limit, so a
    member whose single key, number or unbroken character sequence exceeds it
    is refused by name rather than hashed or silently re-identified.
    """

    def __init__(self, token: str, size: int) -> None:
        super().__init__(f"{token} of {size} bytes exceeds the SQLite value limit of {physical_value_limit()} bytes")
        self.token = token
        self.size = size


@lru_cache(maxsize=1)
def physical_value_limit() -> int:
    """SQLite's compiled maximum length of one string or BLOB value."""
    import sqlite3

    with closing(sqlite3.connect(":memory:")) as connection:
        return connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)


class _NotJsonError(Exception):
    """The byte stream is not one JSON document under the decoder contract."""


class _LoneSurrogateEscapeError(Exception):
    """The fast tokenizer would replace a lone ``\\uD8xx`` escape with ``?``."""


# The backslash run is matched possessively from its first backslash, so a
# long run with no following ``u`` is rejected once, not retried per position.
_SURROGATE_ESCAPE = re.compile(rb"(?<!\\)(\\++)u([dD][89a-fA-F][0-9a-fA-F]{2})")

#: A string value longer than this many raw bytes is not handed to the
#: tokenizer: its content is spilled to a scratch file and hashed from there,
#: so no single scalar is ever held whole. A pacing bound only -- the digest is
#: the same whichever side of it a string falls.
_SPILL_STRING_BYTES = 8 * 1024 * 1024

_JSON_WHITESPACE = b" \t\r\n"

#: A run of bytes outside strings that is not structure or whitespace: a
#: number or literal token.
_BARE_TOKEN = re.compile(rb"[^\s\[\]{}:,\"]+")


class _SpilledStrings:
    """String values streamed to scratch files, addressed by a unique marker."""

    def __init__(self) -> None:
        self._nonce = secrets.token_hex(16)
        self._files: dict[str, IO[bytes]] = {}

    def add(self, handle: IO[bytes]) -> bytes:
        marker = f"polylogue-spilled-string-{self._nonce}-{len(self._files)}"
        self._files[marker] = handle
        return marker.encode("ascii")

    def take(self, marker: object) -> IO[bytes] | None:
        if not isinstance(marker, str) or not self._files:
            return None
        return self._files.pop(marker, None)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()


class _TokenReader:
    """Feed a handle to the tokenizer: spill long strings, watch for lone surrogates.

    A leading UTF-8 byte-order mark is dropped, as the in-memory decoder does.
    String tokens are tracked by quote parity; a string value longer than
    :data:`_SPILL_STRING_BYTES` reaches the tokenizer as a unique marker and
    its raw content goes to a scratch file (object keys are passed whole: they
    decide member order). The C tokenizer decodes a lone ``\\uD800`` escape to
    ``?``, which would give two documents one identity, so escapes in the bytes
    it does receive are checked; detection raises and the caller re-reads with
    the exact pure-Python tokenizer.
    """

    #: Longest escape pair plus slack; a tail this long is rescanned so an
    #: escape split across two reads is seen whole.
    _TAIL = 16

    def __init__(self, handle: IO[bytes], spills: _SpilledStrings, *, scan: bool) -> None:
        self._handle = handle
        self._spills = spills
        self._scan_enabled = scan
        self._carry = b""
        self._carry_offset = 0
        self._done_upto = 0
        self._pending_high_end: int | None = None
        self._first = True
        self._eof = False
        self._in_string = False
        self._backslashes = 0
        self._string: bytearray | None = None
        self._spill: IO[bytes] | None = None
        self._awaiting_role = False
        self._bare_run = 0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            # The tokenizer probes with an empty read to learn the stream type.
            return b""
        out = bytearray()
        while not out and not self._eof:
            chunk = self._handle.read(_STREAM_READ_BYTES)
            if self._first:
                self._first = False
                if chunk.startswith(_UTF8_BOM):
                    chunk = chunk[len(_UTF8_BOM) :]
            if not chunk:
                self._eof = True
                if self._awaiting_role:
                    self._finish_spill(is_key=False, out=out)
                elif self._in_string:
                    out += self._flush_string()
            else:
                self._consume(chunk, out)
        data = bytes(out)
        if self._scan_enabled:
            self._scan(data, final=not data)
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def _consume(self, data: bytes, out: bytearray) -> None:
        position = 0
        while position < len(data):
            if self._awaiting_role:
                while position < len(data) and data[position] in _JSON_WHITESPACE:
                    position += 1
                if position == len(data):
                    return
                self._finish_spill(is_key=data[position] == ord(":"), out=out)
                continue
            if not self._in_string:
                quote = data.find(b'"', position)
                segment = data[position:] if quote < 0 else data[position:quote]
                self._check_bare_tokens(segment)
                if quote < 0:
                    out += data[position:]
                    return
                out += data[position : quote + 1]
                self._in_string = True
                self._backslashes = 0
                self._string = bytearray()
                position = quote + 1
                continue
            end = self._string_end(data, position)
            piece = data[position:end] if end >= 0 else data[position:]
            self._append(piece)
            if end < 0:
                return
            self._in_string = False
            position = end + 1
            if self._spill is None:
                assert self._string is not None
                out += self._string
                out += b'"'
                self._string = None
            else:
                self._awaiting_role = True

    def _check_bare_tokens(self, segment: bytes) -> None:
        """Refuse a number or literal longer than the physical value limit."""
        limit = physical_value_limit()
        runs = _BARE_TOKEN.findall(segment)
        if not runs:
            self._bare_run = 0
            return
        first = len(runs[0]) + (self._bare_run if segment[: len(runs[0])] == runs[0] else 0)
        longest = max([first, *(len(run) for run in runs[1:])])
        if longest > limit:
            raise ContentIdentityRefusal("number token", longest)
        last = runs[-1]
        self._bare_run = (first if len(runs) == 1 else len(last)) if segment.endswith(last) else 0

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

    def _append(self, piece: bytes) -> None:
        if self._spill is not None:
            self._spill.write(piece)
            return
        assert self._string is not None
        self._string += piece
        if len(self._string) > _SPILL_STRING_BYTES:
            self._spill = tempfile.TemporaryFile()  # noqa: SIM115 -- closed by its reader or _SpilledStrings
            self._spill.write(self._string)
            self._string = None

    def _flush_string(self) -> bytes:
        """An unterminated string at end of input: hand it on so the tokenizer refuses it."""
        if self._spill is not None:
            self._spill.close()
            self._spill = None
            return b"x"
        return bytes(self._string or b"")

    def _finish_spill(self, *, is_key: bool, out: bytearray) -> None:
        spill = self._spill
        assert spill is not None
        self._spill = None
        self._awaiting_role = False
        if is_key:
            size = spill.seek(0, 2)
            if size > physical_value_limit():
                spill.close()
                raise ContentIdentityRefusal("object key", size)
            spill.seek(0)
            out += spill.read()
            spill.close()
        else:
            spill.seek(0)
            out += self._spills.add(spill)
        out += b'"'

    def _scan(self, chunk: bytes, *, final: bool) -> None:
        window = self._carry + chunk
        base = self._carry_offset
        limit = len(window) if final else max(0, len(window) - self._TAIL)
        # Never end the scanned region inside a run of backslashes: its
        # parity decides whether the next ``u`` starts an escape.
        while not final and limit > 0 and window[limit - 1 : limit] == b"\\":
            limit -= 1
        for match in _SURROGATE_ESCAPE.finditer(window):
            run_start, end = match.start(), match.end()
            if base + end <= self._done_upto:
                continue
            if end > limit:
                break
            run = len(match.group(1))
            if run % 2 == 0:
                continue
            escape_start = base + run_start + run - 1
            unit = int(match.group(2), 16)
            if self._pending_high_end is not None:
                if unit >= 0xDC00 and escape_start == self._pending_high_end:
                    self._pending_high_end = None
                    self._done_upto = base + end
                    continue
                raise _LoneSurrogateEscapeError
            if unit >= 0xDC00:
                raise _LoneSurrogateEscapeError
            self._pending_high_end = base + end
            self._done_upto = base + end
        if self._pending_high_end is not None and self._pending_high_end < base + limit - 6:
            # Six bytes past a high escape and no low one began there.
            raise _LoneSurrogateEscapeError
        if final and self._pending_high_end is not None:
            raise _LoneSurrogateEscapeError
        keep = max(0, limit - self._TAIL)
        # Rescan from a point outside any backslash run so parity is whole.
        while keep > 0 and window[keep - 1 : keep] == b"\\":
            keep -= 1
        self._carry = window[keep:]
        self._carry_offset = base + keep


_ESCAPE_TOKEN = re.compile(rb'\\(?:u([0-9a-fA-F]{4})|["\\/bfnrt])')


@lru_cache(maxsize=1)
def _composition_followers() -> frozenset[str]:
    """Characters that can be the second element of a canonical composition."""
    followers = {chr(code) for code in range(0x1161, 0x1176)} | {chr(code) for code in range(0x11A8, 0x11C3)}
    for code in range(0x110000):
        decomposition = unicodedata.decomposition(chr(code))
        if decomposition and not decomposition.startswith("<"):
            parts = decomposition.split()
            if len(parts) == 2:
                followers.add(chr(int(parts[1], 16)))
    return frozenset(followers)


def _stable_split(text: str) -> int:
    """Largest index before which NFC of ``text`` may be split without effect."""
    followers = _composition_followers()
    for index in range(len(text) - 1, 0, -1):
        char = text[index]
        if unicodedata.combining(char) == 0 and char not in followers and unicodedata.is_normalized("NFC", char):
            return index
    return 0


def _utf8_boundary(data: bytes, cut: int) -> int:
    """Move ``cut`` back to the start of a UTF-8 sequence it would split."""
    index = cut - 1
    while index >= 0 and cut - index <= 4 and 0x80 <= data[index] <= 0xBF:
        index -= 1
    if index < 0:
        return cut
    lead = data[index]
    width = 1 if lead < 0x80 else 2 if lead >= 0xC0 and lead < 0xE0 else 3 if lead < 0xF0 else 4
    return index if index + width > cut else cut


def _encode_spilled_text(raw: IO[bytes], sink: _Sink) -> None:
    """Encode one spilled JSON string exactly as :func:`_encode_text` would.

    The raw content is JSON-unescaped and NFC-normalized in windows cut at
    escape and composition boundaries, written to scratch to learn its length
    for the tag, then streamed into ``sink``.
    """
    with tempfile.TemporaryFile() as normalized:
        length = 0
        pending_raw = b""
        pending_text = ""
        while True:
            block = raw.read(_STREAM_READ_BYTES)
            final = not block
            data = pending_raw + block
            cut = len(data)
            if not final:
                # Cut outside any escape, never between the two escapes of a
                # surrogate pair, and never inside a UTF-8 sequence.
                tokens = list(_ESCAPE_TOKEN.finditer(data))
                last_end = tokens[-1].end() if tokens else 0
                dangling = data.find(b"\\", last_end)
                if dangling >= 0:
                    # Only an escape cut by the window end may wait for more
                    # bytes; a complete invalid escape is malformed now.
                    if len(data) - dangling > 12:
                        raise _NotJsonError
                    cut = dangling
                for token in reversed(tokens):
                    if token.end() < cut:
                        break
                    unit = token.group(1)
                    if token.end() == cut and unit is not None and 0xD800 <= int(unit, 16) <= 0xDBFF:
                        cut = token.start()
                        break
                cut = _utf8_boundary(data, cut)
            pending_raw = data[cut:]
            try:
                text = pending_text + json.loads('"' + data[:cut].decode("utf-8") + '"')
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise _NotJsonError from exc
            split = len(text) if final else _stable_split(text)
            encoded = nfc(text[:split]).encode("utf-8", errors="surrogatepass")
            normalized.write(encoded)
            length += len(encoded)
            pending_text = text[split:]
            if len(pending_text) * 4 > physical_value_limit():
                # One unbroken composition sequence longer than a storable
                # value: no split point exists to normalize it in windows.
                raise ContentIdentityRefusal("combining character sequence", len(pending_text.encode("utf-8", "surrogatepass")))
            if final:
                if pending_raw:
                    raise _NotJsonError
                break
        sink.update(b"s%d:" % length)
        normalized.seek(0)
        while chunk := normalized.read(_STREAM_READ_BYTES):
            sink.update(chunk)
        sink.update(b";")


class _Frame:
    """One open container while the document streams."""

    __slots__ = ("entries", "is_map", "key", "outer", "poisoned", "value")

    def __init__(self, *, is_map: bool, outer: _Sink) -> None:
        self.is_map = is_map
        #: Where this container's own encoding goes.
        self.outer = outer
        #: Raw key -> value digest, or ``None`` for a member holding a value
        #: with no identity. A repeated key replaces the earlier member, as
        #: the in-memory decoder does, so only a surviving one counts.
        self.entries: dict[str, bytes | None] = {}
        self.key: str | None = None
        #: The hasher of the member value being read (objects only).
        self.value: _Sink | None = None
        #: An array element had no identity (arrays only).
        self.poisoned = False


def _stream_identity(events: Iterator[tuple[str, object]], spills: _SpilledStrings) -> str:
    root = sha256(_CONTENT_IDENTITY_DOMAIN)
    stack: list[_Frame] = []
    documents = 0
    root_poisoned = False

    def current() -> _Sink:
        if not stack:
            return root
        top = stack[-1]
        if not top.is_map:
            return top.outer
        if top.value is None:
            raise _NotJsonError
        return top.value

    def finished_value(*, poisoned: bool = False) -> None:
        nonlocal documents, root_poisoned
        if not stack:
            documents += 1
            if documents > 1:
                raise _NotJsonError
            root_poisoned = poisoned
            return
        top = stack[-1]
        if top.is_map:
            assert top.key is not None and isinstance(top.value, type(root))
            top.entries[top.key] = None if poisoned else top.value.digest()
            top.key = None
            top.value = None
        elif poisoned:
            top.poisoned = True

    for event, value in events:
        if event == "map_key":
            top = stack[-1]
            top.key = str(value)
            top.value = sha256()
        elif event == "start_map":
            stack.append(_Frame(is_map=True, outer=current()))
        elif event == "start_array":
            sink = current()
            sink.update(b"a[")
            stack.append(_Frame(is_map=False, outer=sink))
        elif event == "end_map":
            frame = stack.pop()
            if any(digest is None for digest in frame.entries.values()):
                finished_value(poisoned=True)
                continue
            _encode_object_entries(
                [(nfc(key), digest) for key, digest in frame.entries.items() if digest is not None], frame.outer
            )
            finished_value()
        elif event == "end_array":
            frame = stack.pop()
            frame.outer.update(b"]")
            finished_value(poisoned=frame.poisoned)
        else:
            sink = current()
            spilled = spills.take(value) if event == "string" else None
            try:
                if spilled is not None:
                    with spilled:
                        _encode_spilled_text(spilled, sink)
                elif event == "number" and isinstance(value, Decimal):
                    # The decoder contract reads any fraction or exponent as a
                    # binary float and a bare integer exactly.
                    _encode_float(float(value), sink)
                else:
                    _encode(value, sink)
            except ValueError:
                # A non-finite number has no identity. The decoder would still
                # drop it if a later duplicate key replaces this member, so the
                # verdict waits for the enclosing object to close.
                finished_value(poisoned=True)
                continue
            finished_value()
    if stack or documents != 1 or root_poisoned:
        raise _NotJsonError
    return root.hexdigest()


def stream_payload_content_identity(handle: IO[bytes]) -> str:
    """Return :func:`payload_content_identity` of a seekable handle's bytes.

    The document is tokenized in fixed windows and hashed as it streams, so
    memory holds each open object's (key, digest) entries and at most one
    window of any scalar, never the whole document. Every size takes this one
    route.
    """
    import ijson
    from ijson.backends import python as exact_backend

    start = handle.tell()
    spills = _SpilledStrings()
    try:
        try:
            reader = _TokenReader(handle, spills, scan=True)
            events = ijson.basic_parse(reader, use_float=False, buf_size=_STREAM_READ_BYTES)
            return _stream_identity(events, spills)
        except (_LoneSurrogateEscapeError, UnicodeDecodeError):
            # The C tokenizer either met a lone surrogate escape or rejected
            # one while decoding; the exact tokenizer decides.
            spills.close()
            handle.seek(start)
            reader = _TokenReader(handle, spills, scan=False)
            events = exact_backend.basic_parse(reader, use_float=False, buf_size=_STREAM_READ_BYTES)
            return _stream_identity(events, spills)
    except (_NotJsonError, ijson.JSONError, UnicodeDecodeError, TypeError, ValueError, ArithmeticError):
        handle.seek(start)
        opaque = sha256()
        while chunk := handle.read(_STREAM_READ_BYTES):
            opaque.update(chunk)
        return opaque.hexdigest()
    finally:
        spills.close()


def payload_content_identity(payload: bytes) -> str:
    """Return structural identity for JSON bytes, or byte identity otherwise.

    Container members can be provider JSON or opaque/raw evidence.  Both need
    a durable identity, but only decoded JSON has a representation-independent
    identity.  Opaque bytes deliberately fall back to their content hash.
    """
    return stream_payload_content_identity(io.BytesIO(payload))


def structurally_equal(left: object, right: object) -> bool:
    """Report whether two decoded values are the same content."""
    return structural_content_identity(left) == structural_content_identity(right)


__all__ = [
    "ContentIdentityRefusal",
    "payload_content_identity",
    "physical_value_limit",
    "stream_payload_content_identity",
    "structural_content_identity",
    "structurally_equal",
]
