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
import re
from collections.abc import Iterator
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from typing import IO

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


class _Sink:
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


class _NotJsonError(Exception):
    """The byte stream is not one JSON document under the decoder contract."""


class _LoneSurrogateEscapeError(Exception):
    """The fast tokenizer would replace a lone ``\\uD8xx`` escape with ``?``."""


_SURROGATE_ESCAPE = re.compile(rb"(\\+)u([dD][89a-fA-F][0-9a-fA-F]{2})")


class _ScanningReader:
    """Feed a handle to the tokenizer while checking for lone surrogate escapes.

    The C tokenizer decodes a lone ``\\uD800`` to ``?``, which would give two
    different documents one identity. A ``\\u`` is an escape only after an
    odd run of backslashes; a high surrogate escape must be followed
    immediately by a low one, and a low one must directly follow a high one.
    Detection raises, and the caller re-reads the document with the exact
    pure-Python tokenizer. A leading UTF-8 byte-order mark is dropped, as the
    in-memory decoder does.
    """

    #: Longest escape pair plus slack; a tail this long is rescanned so an
    #: escape split across two reads is seen whole.
    _TAIL = 16

    def __init__(self, handle: IO[bytes], *, scan: bool) -> None:
        self._handle = handle
        self._scan_enabled = scan
        self._carry = b""
        self._carry_offset = 0
        self._done_upto = 0
        self._pending_high_end: int | None = None
        self._first = True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            # The tokenizer probes with an empty read to learn the stream type.
            return b""
        chunk = self._handle.read(_STREAM_READ_BYTES if size is None or size < 0 else size)
        if self._first:
            self._first = False
            if chunk.startswith(_UTF8_BOM):
                chunk = chunk[len(_UTF8_BOM) :]
        if self._scan_enabled:
            self._scan(chunk, final=not chunk)
        return chunk

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

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
            # A run that began before this window was cut; only whole runs
            # from the rescanned tail reach here, so parity is exact.
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


class _Frame:
    """One open container while the document streams."""

    __slots__ = ("entries", "is_map", "key", "outer", "value")

    def __init__(self, *, is_map: bool, outer: _Sink) -> None:
        self.is_map = is_map
        #: Where this container's own encoding goes.
        self.outer = outer
        #: Raw key -> value digest; a repeated key replaces the earlier member,
        #: as the in-memory decoder does.
        self.entries: dict[str, bytes] = {}
        self.key: str | None = None
        #: The hasher of the member value being read (objects only).
        self.value: _Sink | None = None


def _stream_identity(events: Iterator[tuple[str, object]]) -> str:
    root = sha256(_CONTENT_IDENTITY_DOMAIN)
    stack: list[_Frame] = []
    documents = 0

    def current() -> _Sink:
        if not stack:
            return root
        top = stack[-1]
        if not top.is_map:
            return top.outer
        if top.value is None:
            raise _NotJsonError
        return top.value

    def finished_value() -> None:
        nonlocal documents
        if not stack:
            documents += 1
            if documents > 1:
                raise _NotJsonError
            return
        top = stack[-1]
        if top.is_map:
            assert top.key is not None and isinstance(top.value, type(root))
            top.entries[top.key] = top.value.digest()
            top.key = None
            top.value = None

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
            _encode_object_entries([(nfc(key), digest) for key, digest in frame.entries.items()], frame.outer)
            finished_value()
        elif event == "end_array":
            frame = stack.pop()
            frame.outer.update(b"]")
            finished_value()
        else:
            sink = current()
            if event == "number" and isinstance(value, Decimal):
                # The decoder contract reads any fraction or exponent as a
                # binary float and a bare integer exactly.
                _encode_float(float(value), sink)
            else:
                _encode(value, sink)
            finished_value()
    if stack or documents != 1:
        raise _NotJsonError
    return root.hexdigest()


def stream_payload_content_identity(handle: IO[bytes]) -> str:
    """Return :func:`payload_content_identity` of a seekable handle's bytes.

    The document is tokenized in fixed windows and hashed as it streams, so
    memory holds each open object's (key, digest) entries and the current
    scalar, never the whole document. Every size takes this one route.
    """
    import ijson
    from ijson.backends import python as exact_backend

    start = handle.tell()
    try:
        try:
            reader = _ScanningReader(handle, scan=True)
            return _stream_identity(ijson.basic_parse(reader, use_float=False, buf_size=_STREAM_READ_BYTES))
        except (_LoneSurrogateEscapeError, UnicodeDecodeError):
            # The C tokenizer either met a lone surrogate escape or rejected
            # one while decoding; the exact tokenizer decides.
            handle.seek(start)
            reader = _ScanningReader(handle, scan=False)
            return _stream_identity(exact_backend.basic_parse(reader, use_float=False, buf_size=_STREAM_READ_BYTES))
    except (_NotJsonError, ijson.JSONError, UnicodeDecodeError, TypeError, ValueError):
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
    "payload_content_identity",
    "stream_payload_content_identity",
    "structural_content_identity",
    "structurally_equal",
]
