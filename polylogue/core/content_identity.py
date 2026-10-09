"""Structural content identity for decoded provider values.

Byte equality and canonical-JSON digests both answer the wrong question about
an export member: a provider re-serializing the same conversation with
different separators, key order, or ``1`` written as ``1.0`` produces different
bytes for identical content. The identity below is computed over the *decoded*
value under the provider value contract, so it is stable across serialization
while preserving exact decoded strings and keys, and separating values JSON
itself distinguishes -- ``true`` from
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
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import closing
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from math import isfinite
from typing import IO, Protocol

_CONTENT_IDENTITY_DOMAIN = b"polylogue:member-content:v3\0"

_UTF8_BOM = b"\xef\xbb\xbf"

#: Text encodings the lenient detection decoder tries, in order, for a JSON
#: member (``decoder_json.decode_json_bytes``). Content identity does not use
#: them: it reads a member as the record parser's ``json.load`` does.
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
        _encode_integer(value, sink)
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
            entries.append((str(key), child.digest()))
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
    # decoding the whole document. Decoded keys keep their exact spelling;
    # canonically equivalent Unicode keys remain distinct fields.
    entries.sort()
    sink.update(b"o%d;" % len(entries))
    for key, digest in entries:
        _encode_text(b"k", key, sink)
        sink.update(digest)


def _encode_text(tag: bytes, value: str, sink: _Sink) -> None:
    encoded = value.encode("utf-8", errors="surrogatepass")
    sink.update(b"%s%d:" % (tag, len(encoded)))
    sink.update(encoded)
    sink.update(b";")


def _encode_integer(value: int, sink: _Sink) -> None:
    if value.bit_length() <= 64:
        sink.update(b"i%d;" % value)
    else:
        sink.update(b"i" + format(Decimal(value), "f").encode("ascii") + b";")


def _encode_decimal(value: Decimal, sink: _Sink) -> None:
    if not value.is_finite():
        raise ValueError("a non-finite number has no structural content identity")
    if value == value.to_integral_value():
        _encode_integer(int(value), sink)
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
        _encode_integer(int(value), sink)
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

    def __init__(self, checkpoint: Callable[[], None] | None = None) -> None:
        #: Called before each scratch window is read.
        self.checkpoint = checkpoint
        self._nonce = secrets.token_hex(16)
        self._files: dict[str, IO[bytes]] = {}
        self._keys: dict[str, str | _LongKey] = {}
        self._placeholders = 0
        #: Tokens past the physical value limit. Raised only once the whole
        #: document has parsed: until then the bytes may not be JSON at all,
        #: and a non-JSON member takes its byte identity instead. A refused
        #: key refuses the document; a refused value refuses it only if it
        #: survives (a later duplicate key can replace it), which
        #: ``surviving_refusal`` records. Each refusal travels with the value
        #: it refused, so the one raised is the one that survived.
        self.value_refusals = 0
        self.last_value_refusal: ContentIdentityRefusal | None = None
        self.surviving_refusal: ContentIdentityRefusal | None = None
        self._refused_keys: dict[str, ContentIdentityRefusal] = {}
        #: Complete number tokens seen by the token reader, and the ordinals
        #: (counted from 1) of those past the limit: each is one ``number``
        #: event, so the identity stream can tell which value was refused.
        self.number_tokens = 0
        self.refused_numbers: dict[int, ContentIdentityRefusal] = {}
        #: Integer tokens too long to hand the tokenizer, by ordinal: their
        #: digits in a scratch file, streamed into the identity in place of
        #: the ``0`` the tokenizer was given.
        self.long_integers: dict[int, IO[bytes]] = {}

    def add(self, handle: IO[bytes]) -> bytes:
        marker = f"polylogue-spilled-string-{self._nonce}-{len(self._files)}"
        self._files[marker] = handle
        return marker.encode("ascii")

    def add_key(self, key: str | _LongKey) -> bytes:
        """A unique marker standing in for a long key held decoded, never re-escaped."""
        marker = f"polylogue-spilled-key-{self._nonce}-{len(self._keys)}"
        self._keys[marker] = key
        return marker.encode("ascii")

    def add_key_or_refusal(self, key: str | _LongKey | int) -> bytes:
        """The marker of a decoded key, or of a refusal when ``key`` is its size."""
        if isinstance(key, int):
            return self.refuse_key(key)
        return self.add_key(key)

    def take_key(self, marker: str) -> str | _LongKey:
        return self._keys.pop(marker, marker) if self._keys else marker

    def refuse_key(self, size: int) -> bytes:
        """A placeholder key standing in for a refused key, carrying its refusal."""
        self._placeholders += 1
        marker = f"polylogue-refused-key-{self._nonce}-{self._placeholders}"
        self._refused_keys[marker] = ContentIdentityRefusal("object key", size)
        return marker.encode("ascii")

    def refused_key(self, key: str | _LongKey) -> ContentIdentityRefusal | None:
        return self._refused_keys.get(key) if self._refused_keys and isinstance(key, str) else None

    def refuse_value(self, token: str, size: int) -> ContentIdentityRefusal:
        self.value_refusals += 1
        self.last_value_refusal = ContentIdentityRefusal(token, size)
        return self.last_value_refusal

    def take(self, marker: object) -> IO[bytes] | None:
        if not isinstance(marker, str) or not self._files:
            return None
        return self._files.pop(marker, None)

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        for key in self._keys.values():
            if isinstance(key, _LongKey):
                key.close()
        self._keys.clear()
        for digits in self.long_integers.values():
            digits.close()
        self.long_integers.clear()
        self._refused_keys.clear()
        self.value_refusals = 0
        self.last_value_refusal = None
        self.surviving_refusal = None
        self.number_tokens = 0
        self.refused_numbers.clear()


#: A number or literal run longer than this is not handed to the tokenizer,
#: which would hold the whole token: it is canonicalized as it streams
#: (:class:`_LongNumber`). A pacing bound only -- the digest is the same
#: whichever side of it a token falls.
_HOLD_NUMBER_BYTES = 64 * 1024

_DIGITS = re.compile(rb"[0-9]+")


class _LongNumber:
    """A number token too long to hold, reduced to what decides its value.

    The decoder contract reads a token with a fraction or exponent as the
    nearest binary64 float. That float is decided by the token's leading 800
    significant digits, whether any later digit is nonzero, and its decimal
    exponent: every halfway point between two floats has at most 767
    significant digits. So those are kept, and the rest is only counted. An
    integer token is exact; its digits continue through a scratch file when
    the token crosses the parser's in-memory conversion strategy.
    """

    _SIGNIFICANT = 800
    #: Exponent digits kept; past this the magnitude is far outside any float.
    _EXPONENT_DIGITS = 20

    def __init__(self, *, spool_integer: bool) -> None:
        self.negative = False
        self._section = 0  # 0 integer part, 1 fraction, 2 exponent
        self._significant = bytearray()
        self._sticky = False
        #: Decimal exponent of the last kept significant digit.
        self._scale = 0
        self.integer_digits = 0
        self._exponent = bytearray()
        self._exponent_negative = False
        self._exponent_saturated = False
        self._spool: IO[bytes] | None = tempfile.TemporaryFile() if spool_integer else None  # noqa: SIM115

    @property
    def is_integer(self) -> bool:
        return self._section == 0

    def feed(self, piece: bytes) -> None:
        if self._spool is not None and self._section == 0:
            cut = len(piece)
            for mark in (b".", b"e", b"E"):
                found = piece.find(mark)
                if found >= 0:
                    cut = min(cut, found)
            self._spool.write(piece[:cut])
        position = 0
        while position < len(piece):
            byte = piece[position]
            if 0x30 <= byte <= 0x39:
                match = _DIGITS.match(piece, position)
                assert match is not None
                self._digits(match.group())
                position = match.end()
                continue
            if byte == 0x2D:
                if self._section == 2:
                    self._exponent_negative = True
                else:
                    self.negative = True
            elif byte == 0x2E:
                self._section = 1
            elif byte in (0x45, 0x65):
                self._section = 2
            position += 1

    def _digits(self, run: bytes) -> None:
        if self._section == 2:
            if not self._exponent:
                run = run.lstrip(b"0")
            room = self._EXPONENT_DIGITS - len(self._exponent)
            self._exponent += run[:room]
            self._exponent_saturated = self._exponent_saturated or len(run) > room
            return
        if self._section == 0:
            self.integer_digits += len(run)
        if not self._significant:
            stripped = run.lstrip(b"0")
            if self._section == 1:
                self._scale -= len(run) - len(stripped)
            run = stripped
        take = min(self._SIGNIFICANT - len(self._significant), len(run))
        self._significant += run[:take]
        rest = run[take:]
        if self._section == 1:
            self._scale -= take
        else:
            self._scale += len(rest)
        if rest and rest.count(b"0") != len(rest):
            self._sticky = True

    def float_token(self) -> bytes:
        """A short token the tokenizer reads as the same float."""
        sign = "-" if self.negative else ""
        if not self._significant:
            return f"{sign}0.0".encode("ascii")
        digits = self._significant.decode("ascii") + ("1" if self._sticky else "")
        exponent = int(self._exponent or b"0") * (-1 if self._exponent_negative else 1)
        if self._exponent_saturated:
            exponent = -(10**self._EXPONENT_DIGITS) if self._exponent_negative else 10**self._EXPONENT_DIGITS
        scale = self._scale - (1 if self._sticky else 0) + exponent
        leading = scale + len(digits) - 1
        if leading > 400:
            # Past the largest float: the decoder reads infinity.
            return f"{sign}1e999".encode("ascii")
        if leading < -400:
            return f"{sign}0.0".encode("ascii")
        value = float(f"{sign}{digits}e{scale}")
        if not isfinite(value):
            return f"{sign}1e999".encode("ascii")
        return repr(value).encode("ascii")

    def take_digits(self) -> IO[bytes]:
        spool = self._spool
        assert spool is not None
        self._spool = None
        spool.seek(0)
        return spool

    def close(self) -> None:
        if self._spool is not None:
            self._spool.close()
            self._spool = None


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
        self._bare_held = bytearray()
        self._bare_long: _LongNumber | None = None

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
        """Forward bytes outside strings, holding number and literal runs.

        A run is handed on whole when it ends, if it stayed within
        :data:`_HOLD_NUMBER_BYTES`. A longer valid number is canonicalized as
        it streams and handed on as a short token of the same value; a longer
        invalid run is handed on as a byte the tokenizer rejects, so the
        member takes its byte identity.
        """
        position = 0
        for match in _BARE_TOKEN.finditer(segment):
            start, end = match.span()
            if start > position or not self._bare_open:
                self._end_bare_run(out)
                out += segment[position:start]
            self._bare_open = True
            self._feed_bare(segment[start:end])
            position = end
        if position < len(segment):
            self._end_bare_run(out)
            out += segment[position:]
        elif not open_end:
            self._end_bare_run(out)

    def _feed_bare(self, piece: bytes) -> None:
        self._bare_state = _number_step(self._bare_state, piece)
        self._bare_len += len(piece)
        if self._bare_long is None:
            self._bare_held += piece
            digit_limit = sys.get_int_max_str_digits()
            # The parser's integer conversion setting only selects transport.
            # Longer exact integers continue through the existing digit spool.
            hold_bytes = min(_HOLD_NUMBER_BYTES, digit_limit) if digit_limit else _HOLD_NUMBER_BYTES
            if len(self._bare_held) <= hold_bytes:
                return
            self._bare_long = _LongNumber(spool_integer=True)
            piece, self._bare_held = bytes(self._bare_held), bytearray()
        if self._bare_state != _NUM_INVALID:
            self._bare_long.feed(piece)

    def _end_bare_run(self, out: bytearray) -> None:
        if not self._bare_open:
            return
        complete = self._bare_state in _NUM_COMPLETE
        if complete:
            self._spills.number_tokens += 1
        long = self._bare_long
        if complete and self._bare_len > physical_value_limit():
            # Refused where it stands; raised only if the document parses
            # and no later duplicate key replaces it.
            if long is not None:
                long.close()
            ordinal = self._spills.number_tokens
            self._spills.refused_numbers[ordinal] = self._spills.refuse_value("number token", self._bare_len)
            out += b"0"
        elif long is None:
            out += self._bare_held
        elif not complete:
            long.close()
            out += b"x"
        else:
            out += self._settle_long_number(long)
        self._bare_open = False
        self._bare_len = 0
        self._bare_state = _NUM_START
        self._bare_held = bytearray()
        self._bare_long = None

    def _settle_long_number(self, long: _LongNumber) -> bytes:
        """The short token standing in for a complete long number."""
        if not long.is_integer:
            long.close()
            return long.float_token()
        # Integer digits remain exact independently of runtime conversion limits.
        self._spills.long_integers[self._spills.number_tokens] = long.take_digits()
        return b"0"

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
        # Never above the physical limit: a key past it must reach the
        # spilled route, where it is measured and refused by name.
        if len(self._string) > min(_SPILL_STRING_BYTES, physical_value_limit()):
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
            # A key decides member order, so it is measured as stored:
            # exact decoded bytes, not by its escaped spelling. Two distinct
            # decoded spellings are two keys, so the decoded key also
            # identifies the member; past :data:`_SPILL_STRING_BYTES` it is
            # kept only as its hash beside the encoded key in scratch.
            spill.seek(0)
            out += self._spills.add_key_or_refusal(self._decode_key(spill))
        else:
            spill.seek(0)
            out += self._spills.add(spill)
        out += b'"'

    def _decode_key(self, spill: IO[bytes]) -> str | _LongKey | int:
        """Decode an exact key, spilling its bytes past the in-memory window."""
        limit = physical_value_limit()
        pieces: list[str] = []
        encoded: IO[bytes] | None = None
        key_hash = sha256()
        size = 0
        over = False
        try:
            for piece in _iter_decoded_windows(spill, self._spills.checkpoint):
                if over:
                    continue
                chunk = piece.encode("utf-8", "surrogatepass")
                size += len(chunk)
                if size > limit:
                    over = True
                    continue
                key_hash.update(chunk)
                if encoded is None:
                    pieces.append(piece)
                    if size > _SPILL_STRING_BYTES:
                        encoded = tempfile.TemporaryFile()  # noqa: SIM115 -- owned by the long key
                        encoded.write("".join(pieces).encode("utf-8", "surrogatepass"))
                        pieces.clear()
                else:
                    encoded.write(chunk)
            if over:
                return size
            if encoded is None:
                return "".join(pieces)
            key = _LongKey(key_hash.digest(), _SpooledKey.from_file(encoded, size, self._spills.checkpoint))
            encoded = None
            return key
        finally:
            spill.close()
            if encoded is not None:
                encoded.close()

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


#: Ordered object members written between two cancellation checks.
_CHECKPOINT_MEMBERS = 4096


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


def _iter_decoded_windows(raw: IO[bytes], checkpoint: Callable[[], None] | None = None) -> Iterator[str]:
    """JSON-unescape one spilled string's raw content in windows.

    Windows are cut outside any escape, never between the two escapes of a
    surrogate pair, and never inside a UTF-8 sequence. Malformed content
    raises :class:`_NotJsonError`.
    """
    pending_raw = b""
    while True:
        if checkpoint is not None:
            checkpoint()
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
    """Encode exact JSON-unescaped text in bounded windows, including its length."""
    with tempfile.TemporaryFile() as encoded:
        length = 0
        for piece in _iter_decoded_windows(raw, spills.checkpoint):
            chunk = piece.encode("utf-8", errors="surrogatepass")
            encoded.write(chunk)
            length += len(chunk)
        sink.update(b"s%d:" % length)
        encoded.seek(0)
        while chunk := encoded.read(_STREAM_READ_BYTES):
            if spills.checkpoint is not None:
                spills.checkpoint()
            sink.update(chunk)
        sink.update(b";")


#: Estimated bytes of object members held in memory, across every open
#: object of one document, before the object being written moves its members
#: to the shared scratch table. A pacing bound only: the digest is the same
#: whichever side of it a member falls.
_ENTRY_MEMORY_BYTES = 64 * 1024 * 1024

#: Estimated bytes one in-memory member costs beyond its key's characters.
_ENTRY_OVERHEAD_BYTES = 160

#: Bytes reserved in a scratch row beyond its encoded key: the object id,
#: the raw-key hash, the value digest, and SQLite's record header. SQLite
#: bounds a whole row by the same length limit as one value, so a key within
#: this margin of the limit cannot share a row with them.
_SCRATCH_ROW_OVERHEAD = 256


class _EntryStore:
    """Members of one document's objects: an in-memory budget and one scratch table.

    Every open object shares the budget, so nesting cannot multiply it, and
    every spilled object shares one scratch connection, keyed by object id.
    """

    def __init__(self, checkpoint: Callable[[], None] | None = None) -> None:
        self.retained = 0
        #: Called per scratch window and per page of ordered members.
        self.checkpoint = checkpoint
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
                "encoded BLOB NOT NULL, digest BLOB, PRIMARY KEY (obj, key_hash))"
            )
            connection.execute("CREATE INDEX entries_order ON entries (obj, encoded, digest)")
            self._connection = connection
        return self._connection

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None


#: The digest of a member whose value was refused (a token past the physical
#: value limit): a later duplicate key can still replace it.
_REFUSED_DIGEST = b""


class _SpooledKey:
    """An encoded key too long for a scratch row, held in a scratch file.

    Only a bounded prefix stays in memory; ordering against another key reads
    both from the start, so memory stays bounded however many such keys one
    object holds. Compares with ``bytes`` keys as their byte order does.
    """

    _PREFIX_BYTES = 4096

    def __init__(
        self, handle: IO[bytes], length: int, prefix: bytes, checkpoint: Callable[[], None] | None = None
    ) -> None:
        self._file = handle
        self.length = length
        self._prefix = prefix
        self._checkpoint = checkpoint

    @classmethod
    def of(cls, encoded: bytes, checkpoint: Callable[[], None] | None = None) -> _SpooledKey:
        handle = tempfile.TemporaryFile()  # noqa: SIM115 -- owned by the key, closed with its object
        handle.write(encoded)
        return cls(handle, len(encoded), encoded[: cls._PREFIX_BYTES], checkpoint)

    @classmethod
    def from_file(cls, handle: IO[bytes], length: int, checkpoint: Callable[[], None] | None = None) -> _SpooledKey:
        """Take ownership of a scratch file holding a encoded key."""
        handle.seek(0)
        return cls(handle, length, handle.read(cls._PREFIX_BYTES), checkpoint)

    def chunks(self) -> Iterator[bytes]:
        """The key's bytes in windows, checking for cancellation before each."""
        self._file.seek(0)
        while True:
            if self._checkpoint is not None:
                self._checkpoint()
            chunk = self._file.read(_STREAM_READ_BYTES)
            if not chunk:
                return
            yield chunk

    def _compare(self, other: bytes | _SpooledKey) -> int:
        other_prefix = other._prefix if isinstance(other, _SpooledKey) else other
        head = min(len(self._prefix), len(other_prefix))
        if self._prefix[:head] != other_prefix[:head]:
            return -1 if self._prefix[:head] < other_prefix[:head] else 1
        left = self.chunks()
        right: Iterator[bytes] = other.chunks() if isinstance(other, _SpooledKey) else iter((other,))
        left_buffer = right_buffer = b""
        while True:
            if not left_buffer:
                left_buffer = next(left, b"")
            if not right_buffer:
                right_buffer = next(right, b"")
            if not left_buffer or not right_buffer:
                return (len(left_buffer) > 0) - (len(right_buffer) > 0)
            step = min(len(left_buffer), len(right_buffer))
            if left_buffer[:step] != right_buffer[:step]:
                return -1 if left_buffer[:step] < right_buffer[:step] else 1
            left_buffer, right_buffer = left_buffer[step:], right_buffer[step:]

    def __lt__(self, other: bytes | _SpooledKey) -> bool:
        return self._compare(other) < 0

    def __gt__(self, other: bytes | _SpooledKey) -> bool:
        return self._compare(other) > 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, bytes | _SpooledKey):
            return NotImplemented
        return self._compare(other) == 0

    def __hash__(self) -> int:
        return hash(self._prefix)

    def close(self) -> None:
        self._file.close()


class _LongKey:
    """A key decoded past :data:`_SPILL_STRING_BYTES`: never held whole.

    Its decoded text is known by its hash, which decides replacement by a
    repeated key, and its exact decoded bytes live in scratch, which decides order. A
    key held as ``str`` is never this long, so the two never name one key.
    """

    __slots__ = ("encoded", "raw_hash")

    def __init__(self, raw_hash: bytes, encoded: _SpooledKey) -> None:
        self.raw_hash = raw_hash
        self.encoded = encoded

    def __hash__(self) -> int:
        return hash(self.raw_hash)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _LongKey) and other.raw_hash == self.raw_hash

    def close(self) -> None:
        self.encoded.close()


class _Entries:
    """One object's members: raw key -> value digest, ``None`` for no identity,
    :data:`_REFUSED_DIGEST` for a refused value.

    A repeated key replaces the earlier member, as the in-memory decoder does.
    Members stay in memory until the document's shared budget
    (:data:`_ENTRY_MEMORY_BYTES`) is exceeded; the object being written then
    moves its members to the shared scratch table, which also orders them.
    A row holds the raw key's hash (for replacement), the encoded key (for
    order) and the digest; a key too long to fit a row beside them stays in
    memory, where it already was whole, and is merged in order on encode.
    """

    def __init__(self, store: _EntryStore) -> None:
        self._budget = store
        self._memory: dict[str, bytes | None] = {}
        self._retained = 0
        self._id: int | None = None
        #: Refused members by key; a later duplicate key removes its entry.
        #: Each refused token is past the value limit, so these are few.
        self._refusals: dict[str | _LongKey, ContentIdentityRefusal] = {}
        #: Keys too long for a scratch row after a spill, by raw-key hash:
        #: the encoded key in a scratch file and its digest.
        self._spooled: dict[bytes, tuple[_SpooledKey, bytes | None]] = {}

    def __setitem__(self, key: str | _LongKey, digest: bytes | None) -> None:
        if self._refusals:
            self._refusals.pop(key, None)
        if isinstance(key, _LongKey):
            # Held in scratch already: only its hash and digest stay here.
            previous = self._spooled.pop(key.raw_hash, None)
            if previous is not None and previous[0] is not key.encoded:
                previous[0].close()
            self._spooled[key.raw_hash] = (key.encoded, digest)
            return
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
        encoded = key.encode("utf-8", "surrogatepass")
        key_hash = sha256(key.encode("utf-8", "surrogatepass")).digest()
        if len(encoded) + _SCRATCH_ROW_OVERHEAD > physical_value_limit():
            # Too long for a row beside its hash and digest: the encoded key
            # goes to its own scratch file, keyed by the raw key's hash, so a
            # repeat still replaces it (last-key-wins) and memory holds only
            # a bounded prefix per key.
            previous = self._spooled.pop(key_hash, None)
            if previous is not None:
                previous[0].close()
            self._spooled[key_hash] = (_SpooledKey.of(encoded, self._budget.checkpoint), digest)
            return
        self._budget.connection().execute(
            "INSERT OR REPLACE INTO entries VALUES (?, ?, ?, ?)", (self._id, key_hash, encoded, digest)
        )

    def refuse(self, key: str | _LongKey, refusal: ContentIdentityRefusal) -> None:
        self[key] = _REFUSED_DIGEST
        self._refusals[key] = refusal

    def refused(self) -> ContentIdentityRefusal | None:
        """The first refusal still held by a member, after duplicate keys settled."""
        return next(iter(self._refusals.values()), None)

    def poisoned(self) -> bool:
        if any(digest is None for digest in self._memory.values()):
            return True
        if any(digest is None for _key, digest in self._spooled.values()):
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
        if self._id is None and not self._spooled:
            _encode_object_entries([(key, digest) for key, digest in self._memory.items() if digest], sink)
            return
        connection = self._budget.connection() if self._id is not None else None
        count = (
            connection.execute("SELECT COUNT(*) FROM entries WHERE obj = ?", (self._id,)).fetchone()[0]
            if connection is not None
            else 0
        )
        sink.update(b"o%d;" % (count + len(self._memory) + len(self._spooled)))
        # Keys are compared as UTF-8 (surrogates passed through), whose byte
        # order is code-point order, so this matches the in-memory sort.
        held: list[tuple[bytes | _SpooledKey, bytes]] = sorted(
            (key.encode("utf-8", "surrogatepass"), digest) for key, digest in self._memory.items() if digest
        )
        spooled = sorted(
            ((key, digest) for key, digest in self._spooled.values() if digest), key=lambda item: (item[0], item[1])
        )
        rows = (
            (
                (bytes(encoded), bytes(digest))
                for encoded, digest in connection.execute(
                    "SELECT encoded, digest FROM entries WHERE obj = ? ORDER BY encoded, digest", (self._id,)
                )
            )
            if connection is not None
            else iter(())
        )
        checkpoint = self._budget.checkpoint
        for ordinal, (encoded, digest) in enumerate(
            heapq.merge(rows, held, spooled, key=lambda item: (item[0], item[1]))
        ):
            if checkpoint is not None and not ordinal % _CHECKPOINT_MEMBERS:
                checkpoint()
            if isinstance(encoded, _SpooledKey):
                sink.update(b"k%d:" % encoded.length)
                for chunk in encoded.chunks():
                    sink.update(chunk)
                sink.update(b";")
            else:
                _encode_text(b"k", encoded.decode("utf-8", "surrogatepass"), sink)
            sink.update(digest)

    def close(self) -> None:
        for spooled_key, _digest in self._spooled.values():
            spooled_key.close()
        self._spooled.clear()
        if self._id is None:
            self._budget.retained -= self._retained
        else:
            self._budget.connection().execute("DELETE FROM entries WHERE obj = ?", (self._id,))
            # Oversized keys landed back in ``_memory`` after the spill and
            # are charged to the shared budget in ``_put``; that charge is
            # released here too, or a spilled object's memory would never
            # come back off the document-wide total.
            self._budget.retained -= self._retained
        self._retained = 0
        self._memory.clear()
        self._refusals.clear()


class _Frame:
    """One open container while the document streams.

    Nothing is allocated for a container until it needs it, and a run of
    arrays nested directly in one another shares one frame: arrays encode
    inline into the same sink, and a poisoned element poisons every array
    of the run, so a depth counter and one flag say everything the run's
    frames would. Deep nesting therefore costs a small constant per level,
    not a hasher and a member table each.
    """

    __slots__ = ("depth", "entries", "is_map", "key", "outer", "poisoned", "refused", "refused_key", "value")

    def __init__(self, *, is_map: bool, outer: _Sink | None) -> None:
        self.is_map = is_map
        #: Where an array's encoding goes; a map resolves its sink on close.
        self.outer = outer
        self.entries: _Entries | None = None
        self.key: str | _LongKey | None = None
        #: The hasher of the member value being read (objects only), made on
        #: the first write: a member whose value is an object only needs one
        #: once that object closes.
        self.value: _Digester | None = None
        #: An element had no identity (arrays only).
        self.poisoned = False
        #: The first refused element (arrays only).
        self.refused: ContentIdentityRefusal | None = None
        #: A key past the limit (objects only): the object is refused if it
        #: survives, as a refused value is.
        self.refused_key: ContentIdentityRefusal | None = None
        #: Arrays in this run (arrays only).
        self.depth = 1


def _stream_identity(events: Iterator[tuple[str, object]], spills: _SpilledStrings) -> str:
    store = _EntryStore(spills.checkpoint)
    try:
        return _stream_identity_into(events, spills, store)
    finally:
        store.close()


def _stream_identity_into(events: Iterator[tuple[str, object]], spills: _SpilledStrings, store: _EntryStore) -> str:
    root = sha256(_CONTENT_IDENTITY_DOMAIN)
    stack: list[_Frame] = []
    documents = 0
    root_poisoned = False
    root_refused: ContentIdentityRefusal | None = None
    number_ordinal = 0

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

    def finished_value(*, poisoned: bool = False, refused: ContentIdentityRefusal | None = None) -> None:
        nonlocal documents, root_poisoned, root_refused
        if not stack:
            documents += 1
            if documents > 1:
                raise _NotJsonError
            root_poisoned = poisoned
            root_refused = refused
            return
        top = stack[-1]
        if top.is_map:
            assert top.key is not None
            if top.entries is None:
                top.entries = _Entries(store)
            if poisoned:
                top.entries[top.key] = None
            elif refused is not None:
                top.entries.refuse(top.key, refused)
            else:
                assert top.value is not None
                top.entries[top.key] = top.value.digest()
            top.key = None
            top.value = None
        else:
            top.poisoned = top.poisoned or poisoned
            top.refused = top.refused or refused

    for event, value in events:
        if event == "map_key":
            top = stack[-1]
            top.key = spills.take_key(str(value))
            top.value = None
            if top.refused_key is None:
                top.refused_key = spills.refused_key(top.key)
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
                refusal = frame.refused_key or (entries.refused() if entries is not None else None)
                if refusal is not None:
                    finished_value(refused=refusal)
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
            finished_value(poisoned=frame.poisoned, refused=frame.refused)
        else:
            sink = current()
            spilled = spills.take(value) if event == "string" else None
            refusals_before = spills.value_refusals
            digits = None
            if event == "number":
                number_ordinal += 1
                digits = spills.long_integers.pop(number_ordinal, None)
            try:
                if digits is not None:
                    # An integer is its own canonical digits (JSON has no
                    # leading zeros), streamed from scratch.
                    with digits:
                        sink.update(b"i")
                        while chunk := digits.read(_STREAM_READ_BYTES):
                            if spills.checkpoint is not None:
                                spills.checkpoint()
                            sink.update(chunk)
                        sink.update(b";")
                elif spilled is not None:
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
            # A value past the physical limit is refused where it stands, so
            # a later duplicate key can still replace it.
            finished_value(
                refused=spills.last_value_refusal
                if spills.value_refusals > refusals_before and event == "string"
                else spills.refused_numbers.pop(number_ordinal, None)
                if event == "number"
                else None
            )
    if stack or documents != 1 or root_poisoned:
        raise _NotJsonError
    spills.surviving_refusal = root_refused
    return root.hexdigest()


class _EncodingMismatchError(Exception):
    """The payload does not decode under the encoding being tried."""


_RAW_SURROGATE = re.compile("[\ud800-\udfff]")
#: A high-surrogate escape ending just before a position, and a low one
#: starting just after it.
_HIGH_ESCAPE_BEHIND = re.compile(r"\\u[dD][89abAB][0-9a-fA-F]{2}\Z")
_LOW_ESCAPE_AHEAD = re.compile(r"\\u[dD][c-fC-F][0-9a-fA-F]{2}")


def _backslashes_before(text: str, position: int, carried: int) -> int:
    """The run of backslashes ending just before ``position``, continuing
    into the ``carried`` run that preceded ``text``."""
    index = position - 1
    while index >= 0 and text[index] == "\\":
        index -= 1
    run = position - 1 - index
    return run + carried if index < 0 else run


class _DecodedText:
    """A member's bytes as the source decoder reads them, re-encoded as UTF-8.

    Decodes with the encoding the record parser's ``json.load`` detects
    (``json.detect_encoding``: its codec consumes a byte-order mark) and with
    its ``surrogatepass`` error handler, one window at a time, keeping every
    character, a NUL included, so bytes the parser rejects are rejected here
    too. Raises :class:`_EncodingMismatchError` when a byte does not decode or
    nothing is left.

    An encoded lone surrogate decodes to the same character its ``\\uD800``
    escape does, so it is handed on as that escape and the two spellings
    share an identity. Where re-spelling would change the value, the member
    keeps its byte identity instead (:class:`_NotJsonError`): a raw high
    surrogate before a raw or escaped low one, or a raw low one after an
    escaped high one, would pair; after an odd run of backslashes the
    escape would itself be escaped. Backslash runs are counted across
    windows, so parity is exact.
    """

    def __init__(
        self,
        handle: IO[bytes],
        start: int,
        encoding: str,
        errors: str = "strict",
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        handle.seek(start)
        self._handle = handle
        self._decoder = codecs.getincrementaldecoder(encoding)(errors)
        self._checkpoint = checkpoint
        self._at_start = True
        self._eof = False
        #: Decoded characters held back as lookahead for a raw surrogate.
        self._held = ""
        #: The last characters handed on, as lookbehind for a raw surrogate,
        #: and the backslash run that ended just before them.
        self._behind = ""
        self._behind_run = 0
        self._opaque = False

    def read(self, size: int = -1) -> bytes:
        while not self._eof:
            if self._checkpoint is not None:
                self._checkpoint()
            chunk = self._handle.read(_STREAM_READ_BYTES)
            self._eof = not chunk
            try:
                text = self._decoder.decode(chunk, final=self._eof)
            except UnicodeDecodeError as exc:
                raise _EncodingMismatchError from exc
            if self._at_start:
                if not text:
                    if self._eof:
                        raise _EncodingMismatchError
                    continue
                self._at_start = False
            if not self._opaque:
                text = self._escape_raw_surrogates(text)
            if text:
                return text.encode("utf-8", "surrogatepass")
        return b""

    def _escape_raw_surrogates(self, text: str) -> str:
        text = self._held + text
        self._held = ""
        if not self._eof and _RAW_SURROGATE.search(text, max(0, len(text) - 6)):
            # Six characters of lookahead hold an escape that could pair with it.
            text, self._held = text[:-6], text[-6:]
        context = self._behind + text
        offset = len(self._behind)
        parts: list[str] = []
        last = 0
        for match in _RAW_SURROGATE.finditer(text):
            index = match.start()
            position = offset + index
            unit = ord(match.group())
            ahead = text[index + 1 : index + 7]
            escape_start = position - 6
            pairs = (
                unit <= 0xDBFF
                and ((ahead[:1] and 0xDC00 <= ord(ahead[0]) <= 0xDFFF) or _LOW_ESCAPE_AHEAD.match(ahead) is not None)
            ) or (
                unit >= 0xDC00
                and escape_start >= 0
                and _HIGH_ESCAPE_BEHIND.search(context, escape_start, position) is not None
                and _backslashes_before(context, escape_start + 1, self._behind_run) % 2 == 1
            )
            if pairs or _backslashes_before(context, position, self._behind_run) % 2 == 1:
                self._opaque = True
                raise _NotJsonError
            parts.append(text[last:index])
            parts.append(f"\\u{unit:04x}")
            last = index + 1
        parts.append(text[last:])
        if len(context) > 6:
            self._behind_run = _backslashes_before(context, len(context) - 6, self._behind_run)
            self._behind = context[-6:]
        else:
            self._behind = context
        return "".join(parts)

    def drain(self) -> None:
        """Decode the rest, raising :class:`_EncodingMismatchError` if it does not."""
        self._opaque = True
        while self.read():
            pass


def _identity_as(
    handle: IO[bytes], start: int, encoding: str, errors: str, checkpoint: Callable[[], None] | None
) -> str:
    """The structural identity of the member read as ``encoding`` text."""
    import ijson
    from ijson.backends import python as exact_backend

    spills = _SpilledStrings(checkpoint)
    text = _DecodedText(handle, start, encoding, errors, checkpoint)
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
                text = _DecodedText(handle, start, encoding, errors, checkpoint)
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
        # The document is JSON, so an overlong key or value that no later
        # duplicate key replaced is a real refusal.
        if spills.surviving_refusal is not None:
            raise spills.surviving_refusal
        return digest
    finally:
        spills.close()


def stream_payload_content_identity(handle: IO[bytes], *, checkpoint: Callable[[], None] | None = None) -> str:
    """Return :func:`payload_content_identity` of a seekable handle's bytes.

    The member is read as the record parser's ``json.load`` reads it,
    tokenized in fixed windows and hashed as it streams, so memory holds each
    open object's (key, digest) entries and at most one window of any scalar,
    never the whole document. Every size takes this one route.

    ``checkpoint`` is called before each window is read, from the member and
    from scratch files alike, so a caller can stop a multi-gigabyte pass.
    """
    import json

    import ijson

    start = handle.tell()
    try:
        # The record parser (``decoder_json._iter_json_document_with``) falls
        # back to ``json.load`` on the member's bytes, whose encoding comes
        # from ``json.detect_encoding`` and whose errors are passed through
        # as surrogates (``surrogatepass``), as ``json.loads`` decodes bytes;
        # bytes that do not decode under it, or that it would not parse, keep
        # their byte identity.
        encoding = json.detect_encoding(handle.read(4))
        try:
            return _identity_as(handle, start, encoding, "surrogatepass", checkpoint)
        except _EncodingMismatchError:
            raise _NotJsonError from None
    except (_NotJsonError, ijson.JSONError, UnicodeDecodeError, TypeError, ValueError, ArithmeticError):
        handle.seek(start)
        opaque = sha256()
        while chunk := handle.read(_STREAM_READ_BYTES):
            if checkpoint is not None:
                checkpoint()
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
