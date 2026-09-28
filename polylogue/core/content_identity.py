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

import codecs
import heapq
import io
import json
import re
import secrets
import sqlite3
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import closing
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import IO, Protocol

from polylogue.core.text_identity import nfc

_CONTENT_IDENTITY_DOMAIN = b"polylogue:member-content:v2\0"

_UTF8_BOM = b"\xef\xbb\xbf"

#: Text encodings the source decoder tries, in order, for a JSON member
#: (``decoder_json.decode_json_bytes``): the first under which the whole
#: payload decodes, with NUL characters and leading byte-order marks removed,
#: is the member's text. The identity reads a member exactly the same way.
JSON_TEXT_ENCODINGS: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
)

#: Open containers the identity holds state for. The source decoder recurses
#: on the C stack and refuses nesting far shallower than this, so no member
#: the archive can decode reaches it; it keeps the per-container state (a
#: frame, and for an object a hasher and member table) of a member no decoder
#: reads well under a GiB. A deeper member is not JSON under the decoder
#: contract and takes its byte identity. A run of directly nested arrays
#: shares one frame and counts once.
_MAX_OPEN_CONTAINERS = 1 << 20

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


class _ByteSource(Protocol):
    """Anything with ``read(size)`` returning bytes, empty only at the end."""

    def read(self, size: int = -1, /) -> bytes: ...


class _Digester(_Sink, Protocol):
    """A sink whose digest names what it was fed."""

    def digest(self) -> bytes: ...


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


# JSON number grammar states, for runs outside strings. Literals (``true``,
# ``false``, ``null``) read as invalid numbers; they are never long.
_NUM_START, _NUM_MINUS, _NUM_ZERO, _NUM_INT, _NUM_DOT, _NUM_FRAC, _NUM_E, _NUM_ESIGN, _NUM_EXP, _NUM_INVALID = range(10)
_NUM_COMPLETE = frozenset({_NUM_ZERO, _NUM_INT, _NUM_FRAC, _NUM_EXP})
_NUM_DIGIT_RUNS = frozenset({_NUM_INT, _NUM_FRAC, _NUM_EXP})


_DIGIT_RUN = re.compile(rb"[0-9]*")


def _number_step(state: int, data: bytes) -> int:
    """Advance the number grammar over ``data``; digit runs are skipped whole."""
    index = 0
    while index < len(data):
        if state == _NUM_INVALID:
            return state
        if state in _NUM_DIGIT_RUNS:
            match = _DIGIT_RUN.match(data, index)
            assert match is not None
            index = match.end()
            if index == len(data):
                return state
        byte = data[index]
        index += 1
        digit = 0x30 <= byte <= 0x39
        if state in (_NUM_START, _NUM_MINUS):
            if state == _NUM_START and byte == 0x2D:
                state = _NUM_MINUS
            else:
                state = _NUM_ZERO if byte == 0x30 else _NUM_INT if digit else _NUM_INVALID
        elif state in (_NUM_ZERO, _NUM_INT, _NUM_FRAC):
            if byte == 0x2E and state != _NUM_FRAC:
                state = _NUM_DOT
            elif byte in (0x45, 0x65):
                state = _NUM_E
            else:
                # A digit after a leading zero, or any other byte.
                state = _NUM_INVALID
        elif state == _NUM_DOT:
            state = _NUM_FRAC if digit else _NUM_INVALID
        elif state == _NUM_E:
            state = _NUM_ESIGN if byte in (0x2B, 0x2D) else _NUM_EXP if digit else _NUM_INVALID
        else:  # _NUM_ESIGN, _NUM_EXP
            state = _NUM_EXP if digit else _NUM_INVALID
    return state


class _SpilledStrings:
    """String values streamed to scratch files, addressed by a unique marker."""

    def __init__(self) -> None:
        self._nonce = secrets.token_hex(16)
        self._files: dict[str, IO[bytes]] = {}
        self._keys: dict[str, str] = {}
        self._placeholders = 0
        #: A token past the physical value limit. Raised only once the whole
        #: document has parsed: until then the bytes may not be JSON at all,
        #: and a non-JSON member takes its byte identity instead.
        self.refusal: ContentIdentityRefusal | None = None

    def add(self, handle: IO[bytes]) -> bytes:
        marker = f"polylogue-spilled-string-{self._nonce}-{len(self._files)}"
        self._files[marker] = handle
        return marker.encode("ascii")

    def add_key(self, key: str) -> bytes:
        """A unique marker standing in for a long key held decoded, never re-escaped."""
        marker = f"polylogue-spilled-key-{self._nonce}-{len(self._keys)}"
        self._keys[marker] = key
        return marker.encode("ascii")

    def take_key(self, marker: str) -> str:
        return self._keys.pop(marker, marker) if self._keys else marker

    def refuse(self, token: str, size: int) -> None:
        if self.refusal is None:
            self.refusal = ContentIdentityRefusal(token, size)

    def placeholder_key(self) -> bytes:
        """A unique stand-in for a refused key, keeping the document parseable."""
        self._placeholders += 1
        return f"polylogue-refused-key-{self._nonce}-{self._placeholders}".encode("ascii")

    def take(self, marker: object) -> IO[bytes] | None:
        if not isinstance(marker, str) or not self._files:
            return None
        return self._files.pop(marker, None)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._keys.clear()
        self.refusal = None


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

    def __init__(self, handle: _ByteSource, spills: _SpilledStrings, *, scan: bool) -> None:
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
        self._bare_open = False
        self._bare_len = 0
        self._bare_state = _NUM_START
        self._bare_suppressed = False

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
                    self._end_bare_run(out)
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
                if quote < 0:
                    self._emit_outside(data[position:], out, open_end=True)
                    return
                self._emit_outside(data[position:quote], out, open_end=False)
                out += b'"'
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

    def _emit_outside(self, segment: bytes, out: bytearray, *, open_end: bool) -> None:
        """Forward bytes outside strings, tracking number and literal runs.

        A run longer than the physical value limit is forwarded only up to a
        point where it is still a complete number; the rest is validated here
        and not handed on. A valid overlong number records a refusal that is
        raised once the document has parsed; an invalid run ends in a byte
        the tokenizer rejects, so the member takes its byte identity.
        """
        position = 0
        for match in _BARE_TOKEN.finditer(segment):
            start, end = match.span()
            if start > position or not self._bare_open:
                self._end_bare_run(out)
                out += segment[position:start]
            self._bare_open = True
            self._feed_bare(segment[start:end], out)
            position = end
        if position < len(segment):
            self._end_bare_run(out)
            out += segment[position:]
        elif not open_end:
            self._end_bare_run(out)

    def _feed_bare(self, piece: bytes, out: bytearray) -> None:
        if self._bare_suppressed:
            self._bare_state = _number_step(self._bare_state, piece)
            self._bare_len += len(piece)
            return
        room = physical_value_limit() - self._bare_len
        head = piece[: max(room, 0)]
        self._bare_state = _number_step(self._bare_state, head)
        self._bare_len += len(head)
        out += head
        rest = piece[len(head) :]
        if not rest:
            return
        index = 0
        while index < len(rest) and self._bare_state not in _NUM_COMPLETE and self._bare_state != _NUM_INVALID:
            self._bare_state = _number_step(self._bare_state, rest[index : index + 1])
            out += rest[index : index + 1]
            index += 1
        self._bare_len += index
        if self._bare_state == _NUM_INVALID:
            # Not a number: the tokenizer rejects it at once.
            out += rest[index:]
            self._bare_len += len(rest) - index
            return
        if self._bare_state in _NUM_COMPLETE:
            self._bare_suppressed = True
            self._bare_state = _number_step(self._bare_state, rest[index:])
            self._bare_len += len(rest) - index

    def _end_bare_run(self, out: bytearray) -> None:
        if not self._bare_open:
            return
        if self._bare_suppressed:
            if self._bare_state in _NUM_COMPLETE:
                self._spills.refuse("number token", self._bare_len)
            else:
                out += b"x"
        self._bare_open = False
        self._bare_len = 0
        self._bare_state = _NUM_START
        self._bare_suppressed = False

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
            # A key decides member order, so it is held whole -- but it is
            # measured decoded, not by its escaped spelling.
            spill.seek(0)
            pieces: list[str] = []
            size = 0
            for piece in _iter_decoded_windows(spill):
                size += len(piece.encode("utf-8", "surrogatepass"))
                if size <= physical_value_limit():
                    pieces.append(piece)
            spill.close()
            if size > physical_value_limit():
                self._spills.refuse("object key", size)
                out += self._spills.placeholder_key()
            else:
                # Held decoded and handed on as a marker: re-escaping it for
                # the tokenizer could multiply its size (a control character
                # is six escaped bytes).
                out += self._spills.add_key("".join(pieces))
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


def _stable_split(text: str, start: int = 0) -> int:
    """Largest index before which NFC of ``text`` may be split without effect.

    ``text[:start]`` holds no split point past index 0, so only the rest is
    searched: an unbroken composition sequence costs one pass, not one per
    window.
    """
    followers = _composition_followers()
    for index in range(len(text) - 1, max(start, 1) - 1, -1):
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


def _iter_decoded_windows(raw: IO[bytes]) -> Iterator[str]:
    """JSON-unescape one spilled string's raw content in windows.

    Windows are cut outside any escape, never between the two escapes of a
    surrogate pair, and never inside a UTF-8 sequence. Malformed content
    raises :class:`_NotJsonError`.
    """
    pending_raw = b""
    while True:
        block = raw.read(_STREAM_READ_BYTES)
        final = not block
        data = pending_raw + block
        cut = len(data)
        if not final:
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
            yield json.loads('"' + data[:cut].decode("utf-8") + '"')
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise _NotJsonError from exc
        if final:
            if pending_raw:
                raise _NotJsonError
            return


def _encode_spilled_text(raw: IO[bytes], sink: _Sink, spills: _SpilledStrings) -> None:
    """Encode one spilled JSON string exactly as :func:`_encode_text` would.

    The raw content is JSON-unescaped and NFC-normalized in windows cut at
    escape and composition boundaries, written to scratch to learn its length
    for the tag, then streamed into ``sink``. An unbroken composition sequence
    longer than a storable value cannot be normalized in windows: it records
    a refusal, and the rest of the string is still read for validity.
    """
    limit = physical_value_limit()
    with tempfile.TemporaryFile() as normalized:
        length = 0
        pending_text = ""
        pending_bytes = 0
        draining = False
        windows = _iter_decoded_windows(raw)
        for piece in windows:
            if draining:
                continue
            text = pending_text + piece
            split = _stable_split(text, len(pending_text))
            encoded = nfc(text[:split]).encode("utf-8", errors="surrogatepass")
            normalized.write(encoded)
            length += len(encoded)
            if split == 0:
                pending_bytes += len(piece.encode("utf-8", "surrogatepass"))
            else:
                pending_bytes = len(text[split:].encode("utf-8", "surrogatepass"))
            pending_text = text[split:]
            if pending_bytes > limit:
                spills.refuse("combining character sequence", pending_bytes)
                pending_text = ""
                draining = True
        if draining:
            sink.update(b"s0:;")
            return
        encoded = nfc(pending_text).encode("utf-8", errors="surrogatepass")
        normalized.write(encoded)
        length += len(encoded)
        sink.update(b"s%d:" % length)
        normalized.seek(0)
        while chunk := normalized.read(_STREAM_READ_BYTES):
            sink.update(chunk)
        sink.update(b";")


#: Estimated bytes of object members held in memory, across every open
#: object of one document, before the object being written moves its members
#: to the shared scratch table. A pacing bound only: the digest is the same
#: whichever side of it a member falls.
_ENTRY_MEMORY_BYTES = 64 * 1024 * 1024

#: Estimated bytes one in-memory member costs beyond its key's characters.
_ENTRY_OVERHEAD_BYTES = 160

#: Bytes reserved in a scratch row beyond its normalized key: the object id,
#: the raw-key hash, the value digest, and SQLite's record header. SQLite
#: bounds a whole row by the same length limit as one value, so a key within
#: this margin of the limit cannot share a row with them.
_SCRATCH_ROW_OVERHEAD = 256


class _EntryStore:
    """Members of one document's objects: an in-memory budget and one scratch table.

    Every open object shares the budget, so nesting cannot multiply it, and
    every spilled object shares one scratch connection, keyed by object id.
    """

    def __init__(self) -> None:
        self.retained = 0
        self._connection: sqlite3.Connection | None = None
        self._next_id = 0

    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect("")
            # Pin the row bound the entries plan against.
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, physical_value_limit())
            connection.execute("PRAGMA journal_mode = OFF")
            connection.execute(
                "CREATE TABLE entries (obj INTEGER NOT NULL, key_hash BLOB NOT NULL, "
                "normalized BLOB NOT NULL, digest BLOB, PRIMARY KEY (obj, key_hash))"
            )
            connection.execute("CREATE INDEX entries_order ON entries (obj, normalized, digest)")
            self._connection = connection
        return self._connection

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class _Entries:
    """One object's members: raw key -> value digest, ``None`` for no identity.

    A repeated key replaces the earlier member, as the in-memory decoder does.
    Members stay in memory until the document's shared budget
    (:data:`_ENTRY_MEMORY_BYTES`) is exceeded; the object being written then
    moves its members to the shared scratch table, which also orders them.
    A row holds the raw key's hash (for replacement), the normalized key (for
    order) and the digest; a key too long to fit a row beside them stays in
    memory, where it already was whole, and is merged in order on encode.
    """

    def __init__(self, store: _EntryStore) -> None:
        self._budget = store
        self._memory: dict[str, bytes | None] = {}
        self._retained = 0
        self._id: int | None = None

    def __setitem__(self, key: str, digest: bytes | None) -> None:
        if self._id is not None:
            self._put(key, digest)
            return
        if key not in self._memory:
            cost = len(key) + _ENTRY_OVERHEAD_BYTES
            self._retained += cost
            self._budget.retained += cost
        self._memory[key] = digest
        if self._budget.retained > _ENTRY_MEMORY_BYTES:
            self._spill()

    def _spill(self) -> None:
        self._id = self._budget.new_id()
        held = self._memory
        self._memory = {}
        self._budget.retained -= self._retained
        self._retained = 0
        for held_key, held_digest in held.items():
            self._put(held_key, held_digest)

    def _put(self, key: str, digest: bytes | None) -> None:
        normalized = nfc(key).encode("utf-8", "surrogatepass")
        if len(normalized) + _SCRATCH_ROW_OVERHEAD > physical_value_limit():
            # A raw key always normalizes to the same text, so its repeats
            # land here too and last-key-wins still holds.
            self._memory[key] = digest
            return
        key_hash = sha256(key.encode("utf-8", "surrogatepass")).digest()
        self._budget.connection().execute(
            "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?)", (self._id, key_hash, normalized, digest)
        )

    def poisoned(self) -> bool:
        if any(digest is None for digest in self._memory.values()):
            return True
        if self._id is None:
            return False
        return (
            self._budget.connection()
            .execute("SELECT 1 FROM entries WHERE obj = ? AND digest IS NULL LIMIT 1", (self._id,))
            .fetchone()
            is not None
        )

    def encode(self, sink: _Sink) -> None:
        if self._id is None:
            _encode_object_entries([(nfc(key), digest) for key, digest in self._memory.items() if digest], sink)
            return
        connection = self._budget.connection()
        count = connection.execute("SELECT COUNT(*) FROM entries WHERE obj = ?", (self._id,)).fetchone()[0]
        sink.update(b"o%d;" % (count + len(self._memory)))
        # Keys are compared as UTF-8 (surrogates passed through), whose byte
        # order is code-point order, so this matches the in-memory sort.
        held = sorted(
            (nfc(key).encode("utf-8", "surrogatepass"), digest) for key, digest in self._memory.items() if digest
        )
        rows = (
            (bytes(normalized), bytes(digest))
            for normalized, digest in connection.execute(
                "SELECT normalized, digest FROM entries WHERE obj = ? ORDER BY normalized, digest", (self._id,)
            )
        )
        for normalized, digest in heapq.merge(rows, held):
            _encode_text(b"k", normalized.decode("utf-8", "surrogatepass"), sink)
            sink.update(digest)

    def close(self) -> None:
        if self._id is None:
            self._budget.retained -= self._retained
            self._retained = 0
        else:
            self._budget.connection().execute("DELETE FROM entries WHERE obj = ?", (self._id,))
        self._memory.clear()


class _Frame:
    """One open container while the document streams.

    Nothing is allocated for a container until it needs it, and a run of
    arrays nested directly in one another shares one frame: arrays encode
    inline into the same sink, and a poisoned element poisons every array
    of the run, so a depth counter and one flag say everything the run's
    frames would. Deep nesting therefore costs a small constant per level,
    not a hasher and a member table each.
    """

    __slots__ = ("depth", "entries", "is_map", "key", "outer", "poisoned", "value")

    def __init__(self, *, is_map: bool, outer: _Sink | None) -> None:
        self.is_map = is_map
        #: Where an array's encoding goes; a map resolves its sink on close.
        self.outer = outer
        self.entries: _Entries | None = None
        self.key: str | None = None
        #: The hasher of the member value being read (objects only), made on
        #: the first write: a member whose value is an object only needs one
        #: once that object closes.
        self.value: _Digester | None = None
        #: An element had no identity (arrays only).
        self.poisoned = False
        #: Arrays in this run (arrays only).
        self.depth = 1


def _stream_identity(events: Iterator[tuple[str, object]], spills: _SpilledStrings) -> str:
    store = _EntryStore()
    try:
        return _stream_identity_into(events, spills, store)
    finally:
        store.close()


def _stream_identity_into(events: Iterator[tuple[str, object]], spills: _SpilledStrings, store: _EntryStore) -> str:
    root = sha256(_CONTENT_IDENTITY_DOMAIN)
    stack: list[_Frame] = []
    documents = 0
    root_poisoned = False

    def current() -> _Sink:
        if not stack:
            return root
        top = stack[-1]
        if not top.is_map:
            assert top.outer is not None
            return top.outer
        if top.key is None:
            raise _NotJsonError
        if top.value is None:
            top.value = sha256()
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
            assert top.key is not None
            if top.entries is None:
                top.entries = _Entries(store)
            if poisoned:
                top.entries[top.key] = None
            else:
                assert top.value is not None
                top.entries[top.key] = top.value.digest()
            top.key = None
            top.value = None
        elif poisoned:
            top.poisoned = True

    for event, value in events:
        if event == "map_key":
            top = stack[-1]
            top.key = spills.take_key(str(value))
            top.value = None
        elif event == "start_map":
            if stack and stack[-1].is_map and stack[-1].key is None:
                raise _NotJsonError
            if len(stack) >= _MAX_OPEN_CONTAINERS:
                raise _NotJsonError
            stack.append(_Frame(is_map=True, outer=None))
        elif event == "start_array":
            sink = current()
            sink.update(b"a[")
            if stack and not stack[-1].is_map:
                stack[-1].depth += 1
            elif len(stack) >= _MAX_OPEN_CONTAINERS:
                raise _NotJsonError
            else:
                stack.append(_Frame(is_map=False, outer=sink))
        elif event == "end_map":
            frame = stack.pop()
            entries = frame.entries
            try:
                if entries is not None and entries.poisoned():
                    finished_value(poisoned=True)
                    continue
                sink = current()
                if entries is None:
                    _encode_object_entries([], sink)
                else:
                    entries.encode(sink)
            finally:
                if entries is not None:
                    entries.close()
            finished_value()
        elif event == "end_array":
            frame = stack[-1]
            assert frame.outer is not None
            frame.outer.update(b"]")
            if frame.depth > 1:
                # The enclosing array of the run: a poisoned element already
                # set the run's shared flag.
                frame.depth -= 1
                continue
            stack.pop()
            finished_value(poisoned=frame.poisoned)
        else:
            sink = current()
            spilled = spills.take(value) if event == "string" else None
            try:
                if spilled is not None:
                    with spilled:
                        _encode_spilled_text(spilled, sink, spills)
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


class _EncodingMismatchError(Exception):
    """The payload does not decode under the encoding being tried."""


class _DecodedText:
    """A member's bytes as the source decoder reads them, re-encoded as UTF-8.

    Decodes with one of :data:`JSON_TEXT_ENCODINGS` one window at a time,
    drops NUL characters and any leading byte-order marks as
    ``decode_json_bytes`` does, and raises :class:`_EncodingMismatchError`
    when a byte does not decode or nothing is left -- the two conditions
    under which the decoder moves on to its next encoding.
    """

    def __init__(self, handle: IO[bytes], start: int, encoding: str, errors: str = "strict") -> None:
        handle.seek(start)
        self._handle = handle
        self._decoder = codecs.getincrementaldecoder(encoding)(errors)
        self._at_start = True
        self._eof = False

    def read(self, size: int = -1) -> bytes:
        while not self._eof:
            chunk = self._handle.read(_STREAM_READ_BYTES)
            self._eof = not chunk
            try:
                text = self._decoder.decode(chunk, final=self._eof)
            except UnicodeDecodeError as exc:
                raise _EncodingMismatchError from exc
            text = text.replace("\x00", "")
            if self._at_start:
                text = text.lstrip("\ufeff")
                if not text:
                    if self._eof:
                        raise _EncodingMismatchError
                    continue
                self._at_start = False
            if text:
                return text.encode("utf-8", "surrogatepass")
        return b""

    def drain(self) -> None:
        """Decode the rest, raising :class:`_EncodingMismatchError` if it does not."""
        while self.read():
            pass


def _identity_as(handle: IO[bytes], start: int, encoding: str, errors: str) -> str:
    """The structural identity of the member read as ``encoding`` text."""
    import ijson
    from ijson.backends import python as exact_backend

    spills = _SpilledStrings()
    text = _DecodedText(handle, start, encoding, errors)
    try:
        try:
            try:
                events = ijson.basic_parse(
                    _TokenReader(text, spills, scan=True), use_float=False, buf_size=_STREAM_READ_BYTES
                )
                digest = _stream_identity(events, spills)
            except (_LoneSurrogateEscapeError, UnicodeDecodeError):
                # The C tokenizer either met a lone surrogate escape or
                # rejected one while decoding; the exact tokenizer decides.
                spills.close()
                text = _DecodedText(handle, start, encoding, errors)
                events = exact_backend.basic_parse(
                    _TokenReader(text, spills, scan=False), use_float=False, buf_size=_STREAM_READ_BYTES
                )
                digest = _stream_identity(events, spills)
        except (_NotJsonError, ijson.JSONError, TypeError, ValueError, ArithmeticError):
            # The decoder picks an encoding by whether the whole payload
            # decodes, not by whether the text parses: a later undecodable
            # byte still moves it to the next encoding.
            text.drain()
            raise
        if spills.refusal is not None:
            # The document is JSON, so an overlong token is a real refusal.
            raise spills.refusal
        return digest
    finally:
        spills.close()


def stream_payload_content_identity(handle: IO[bytes]) -> str:
    """Return :func:`payload_content_identity` of a seekable handle's bytes.

    The member is read as the source decoder reads it (:data:`JSON_TEXT_ENCODINGS`),
    tokenized in fixed windows and hashed as it streams, so memory holds each
    open object's (key, digest) entries and at most one window of any scalar,
    never the whole document. Every size takes this one route.
    """
    import ijson

    start = handle.tell()
    try:
        for encoding in JSON_TEXT_ENCODINGS:
            try:
                return _identity_as(handle, start, encoding, "strict")
            except _EncodingMismatchError:
                continue
        # No encoding decodes the payload whole. The decoder's lossy last
        # resort can turn such bytes into a clean document's text, so they
        # keep their byte identity instead of sharing that document's.
        raise _NotJsonError
    except (_NotJsonError, ijson.JSONError, UnicodeDecodeError, TypeError, ValueError, ArithmeticError):
        handle.seek(start)
        opaque = sha256()
        while chunk := handle.read(_STREAM_READ_BYTES):
            opaque.update(chunk)
        return opaque.hexdigest()


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
    "JSON_TEXT_ENCODINGS",
    "ContentIdentityRefusal",
    "payload_content_identity",
    "physical_value_limit",
    "stream_payload_content_identity",
    "structural_content_identity",
    "structurally_equal",
]
