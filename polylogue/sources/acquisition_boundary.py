"""The acquisition boundary: every byte of a bound source is validated as it is read.

A source location admits only its own origin's material. Rather than each
acquisition route checking what it happens to read (a prefix here, a first
record there), every route obtains source bytes through this module:

- :func:`open_bound_path`, :func:`open_bound_member` and :func:`bind_stream`
  return a stream that feeds each byte, on its first read and in order, to
  :class:`BoundRecordValidator`. A foreign record raises
  :class:`ForeignOriginContentError` from the ``read`` that delivers it,
  before the consumer (a blob capture, a decoder, a hasher) can use it.
- :func:`capture_bound_path` and :func:`capture_bound_stream` are the only
  routes that retain source bytes; the capture copies exactly the bytes it
  validates, so a refusal raises before any publication is queued.
- :func:`open_admitted_blob` reopens a captured blob. Its bytes were
  validated when captured and are content-addressed, so it is not re-read
  through the validator.

``tests/unit/sources/test_acquisition_boundary.py`` enumerates the modules
that read source bytes and fails when a retention or member-open call appears
outside this module without a declared reason.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, BinaryIO

import ijson

from polylogue.archive.zip_admission import open_bounded_zip_entry
from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore, Heartbeat

from .dispatch import (
    ForeignOriginContentError,
    bound_location_provider,
    detect_provider_evidence,
    is_jsonl_source_path,
    same_origin,
)

_READ_CHUNK_BYTES = 1 << 20


def refuse_declared_foreign(name: str, location: Provider | str | None) -> None:
    """Refuse a declared database member of another origin by its declaration.

    A Codex ``state_5.sqlite`` under a Claude Code root is foreign by name,
    whatever its bytes; its content is never JSON-validated.
    """
    bound = bound_location_provider(location)
    if bound is None:
        return
    from .origin_specs import database_member_for_filename

    member = database_member_for_filename(Path(name).name)
    if member is not None and not same_origin(member.provider, bound):
        raise ForeignOriginContentError(expected=bound, found=member.provider, evidence="declared database member")


class BoundRecordValidator:
    """Incremental origin validation of a bound JSON/JSONL byte stream.

    Every record is push-parsed as its bytes arrive, so no record is held as
    bytes or as a second decoded copy:

    - A JSONL stream is validated record by record, each line through its
      own incremental parser.
    - A ``.json`` document goes through one incremental parser. A top-level
      array is validated element by element, and a top-level object is
      validated as the document it is.

    A record (a JSONL line or an array element) is classified both as a
    single record and as a one-record sequence, because some origins declare
    only record detectors and others only sequence detectors. Each record is
    classified from a bounded view (:class:`_EvidenceBuilder`), so memory
    does not grow with a record's size. No byte window truncates the stream:
    every record is parsed to its end and classified. A malformed or
    truncated record is validated from the structure that completed before
    the fault; malformed JSON itself is the parser's typed concern, not a
    foreign-origin claim. Inactive for unbound locations, declared
    ``raw-only`` paths and non-JSON material.
    """

    def __init__(self, name: str, location: Provider | str | None) -> None:
        self._bound = bound_location_provider(location)
        self._is_jsonl = is_jsonl_source_path(name)
        active = self._bound is not None and (self._is_jsonl or name.lower().endswith(".json"))
        if active and self._bound is not None:
            from .origin_specs import path_declaration_refuses_session

            active = not path_declaration_refuses_session(self._bound, Path(name))
        self.active = active
        self._document = _DocumentValidator(self._bound, records=False) if active and not self._is_jsonl else None
        self._line: _DocumentValidator | None = None
        self._finished = False

    def feed(self, chunk: bytes) -> None:
        if not self.active or not chunk:
            return
        if self._document is not None:
            self._document.feed(chunk)
            return
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline == -1 else newline
            if end > start:
                if self._line is None:
                    self._line = _DocumentValidator(self._bound, records=True)
                self._line.feed(chunk[start:end])
            if newline == -1:
                return
            self._end_line()
            start = newline + 1

    def finish(self) -> None:
        if not self.active or self._finished:
            return
        self._finished = True
        if self._document is not None:
            self._document.finish()
        else:
            self._end_line()

    def _end_line(self) -> None:
        line, self._line = self._line, None
        if line is not None:
            line.finish()


#: Characters of a string value kept for origin classification.
_STRING_KEEP_CHARS = 4096
#: Entries of one object or array kept for origin classification.
_CONTAINER_KEEP_ENTRIES = 4096
#: Values (containers and scalars) of one record kept for classification.
_RECORD_KEEP_VALUES = 1 << 18
#: String characters of one record kept for classification.
_RECORD_KEEP_CHARS = 1 << 23


class _EvidenceBuilder:
    """Build the classification view of one record from parser events, in bounded memory.

    Every event is consumed, but only a bounded view is kept: a long string
    keeps its prefix, a container its first entries, and the record as a
    whole a fixed number of values and string characters. Origin
    discriminators are shallow keys and short values that appear early in a
    record; a multi-gigabyte embedded tool result, or a record with millions
    of small values, must not be copied to decide its origin. What is
    dropped is only evidence; the bytes themselves still pass through.
    """

    def __init__(self) -> None:
        self.value: object = None
        self._stack: list[dict[str, object] | list[object]] = []
        self._key: str | None = None
        #: Depth of a dropped container still being consumed.
        self._skip = 0
        self._skip_next = False
        self._values = 0
        self._chars = 0

    def event(self, event: str, value: object) -> None:
        if self._skip:
            if event in ("start_map", "start_array"):
                self._skip += 1
            elif event in ("end_map", "end_array"):
                self._skip -= 1
            return
        if event == "map_key":
            container = self._stack[-1]
            self._skip_next = len(container) >= _CONTAINER_KEEP_ENTRIES or self._values >= _RECORD_KEEP_VALUES
            self._key = str(value)
            return
        if event in ("end_map", "end_array"):
            self._stack.pop()
            return
        if self._skip_next or self._full():
            self._skip_next = False
            if event in ("start_map", "start_array"):
                self._skip = 1
            return
        self._values += 1
        item: object
        if event == "start_map":
            item = {}
        elif event == "start_array":
            item = []
        elif event == "string" and isinstance(value, str):
            keep = max(0, min(_STRING_KEEP_CHARS, _RECORD_KEEP_CHARS - self._chars))
            item = value[:keep]
            self._chars += len(item)
        else:
            item = value
        self._attach(item)
        if isinstance(item, (dict, list)):
            self._stack.append(item)

    def _full(self) -> bool:
        if self._values >= _RECORD_KEEP_VALUES:
            return bool(self._stack)
        container = self._stack[-1] if self._stack else None
        return isinstance(container, list) and len(container) >= _CONTAINER_KEEP_ENTRIES

    def _attach(self, item: object) -> None:
        if not self._stack:
            self.value = item
            return
        container = self._stack[-1]
        if isinstance(container, dict):
            assert self._key is not None
            container[self._key] = item
            self._key = None
        else:
            container.append(item)


class _DocumentValidator:
    """Push-parse one JSON value, validating array elements as they complete.

    ``records`` marks a JSONL line, whose top-level object is a record.
    """

    def __init__(self, bound: Provider | None, *, records: bool) -> None:
        self._bound = bound
        self._records = records
        self._events = ijson.sendable_list()
        self._parser = ijson.basic_parse_coro(self._events, use_float=True)
        self._depth = 0
        self._top: str | None = None
        self._builder: _EvidenceBuilder | None = None
        self._failed = False
        self._seen = False

    def feed(self, chunk: bytes) -> None:
        if self._failed:
            return
        self._seen = self._seen or bool(chunk.strip())
        try:
            self._parser.send(chunk)
        except ijson.JSONError:
            self._failed = True
        self._drain()
        if self._failed:
            self._validate_partial()

    def finish(self) -> None:
        if self._failed or not self._seen:
            return
        try:
            self._parser.close()
        except ijson.JSONError:
            self._failed = True
        self._drain()
        if self._failed or self._depth:
            self._validate_partial()

    def _drain(self) -> None:
        for event, value in self._events:
            self._event(event, value)
        del self._events[:]

    def _event(self, event: str, value: object) -> None:
        opening = event in ("start_map", "start_array")
        closing = event in ("end_map", "end_array")
        if self._depth == 0:
            if opening:
                self._top = event
                if event == "start_map":
                    self._builder = _EvidenceBuilder()
                    self._builder.event(event, value)
                self._depth = 1
            return
        if self._top == "start_array" and self._depth == 1:
            if closing:
                self._depth = 0
                return
            if not opening:
                return  # a scalar array element carries no record shape
            self._builder = _EvidenceBuilder()
        assert self._builder is not None
        self._builder.event(event, value)
        if opening:
            self._depth += 1
        elif closing:
            self._depth -= 1
            if self._depth == 1 and self._top == "start_array":
                self._validate(self._builder.value, record=True)
                self._builder = None
            elif self._depth == 0:
                self._validate(self._builder.value, record=self._records)
                self._builder = None

    def _validate_partial(self) -> None:
        if self._builder is None:
            return
        partial = getattr(self._builder, "value", None)
        self._builder = None
        if isinstance(partial, (dict, list)) and partial:
            self._validate(partial, record=self._records or self._top == "start_array")

    def _validate(self, value: object, *, record: bool) -> None:
        detect_provider_evidence(value, expected=self._bound)
        if record:
            # Origins declare record detectors or sequence detectors; a
            # record is checked against both.
            detect_provider_evidence([value], expected=self._bound)


class BoundStream(io.RawIOBase):
    """A raw reader that validates each source byte on its first read.

    Seeking back (decoders re-read ``.json`` documents) re-delivers bytes
    without re-validating them; seeking forward validates the skipped bytes
    first, so every byte a consumer can reach has been validated, in order.
    End of stream completes validation of a trailing unterminated record.
    """

    def __init__(self, raw: IO[bytes], name: str, location: Provider | str | None, *, admitted: bool = False) -> None:
        super().__init__()
        if not admitted:
            refuse_declared_foreign(name, location)
        self._raw = raw
        self.name = name
        self.location = bound_location_provider(location)
        self._validator = BoundRecordValidator(name, None if admitted else location)
        self._position = 0
        self._validated_to = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return bool(self._raw.seekable())

    def tell(self) -> int:
        return self._position

    def readinto(self, buffer: memoryview | bytearray) -> int:  # type: ignore[override]
        data = self._raw.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        self._observe(self._position, data)
        self._position += size
        return size

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        target = self._raw.seek(offset, whence)
        if target > self._validated_to:
            self._raw.seek(self._validated_to)
            self._position = self._validated_to
            while self._position < target:
                data = self._raw.read(min(_READ_CHUNK_BYTES, target - self._position))
                if not data:
                    break
                self._observe(self._position, data)
                self._position += len(data)
        self._position = target
        return target

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()

    def _observe(self, start: int, data: bytes) -> None:
        if not data:
            if start >= self._validated_to:
                self._validator.finish()
            return
        end = start + len(data)
        if end <= self._validated_to:
            return
        self._validator.feed(data[max(0, self._validated_to - start) :])
        self._validated_to = end


def bind_stream(handle: IO[bytes], name: str, location: Provider | str | None) -> BinaryIO:
    """Route an open source handle through the boundary (idempotent)."""
    if is_bound(handle):
        return handle  # type: ignore[return-value]
    return io.BufferedReader(BoundStream(handle, name, location), buffer_size=_READ_CHUNK_BYTES)


def is_bound(handle: object) -> bool:
    """Whether ``handle`` already reads through the boundary."""
    return isinstance(handle, io.BufferedReader) and isinstance(handle.raw, BoundStream)


@contextmanager
def open_bound_path(path: Path | str, location: Provider | str | None) -> Iterator[BinaryIO]:
    """Open a source file at ``location`` through the boundary."""
    source = Path(path)
    refuse_declared_foreign(source.name, location)
    with bind_stream(source.open("rb"), str(source), location) as stream:
        yield stream


@contextmanager
def open_bound_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    location: Provider | str | None,
    *,
    max_bytes: int | None = None,
) -> Iterator[BinaryIO]:
    """Open an admitted ZIP member through the boundary; the archive's location binds it."""
    refuse_declared_foreign(info.filename, location)
    with bind_stream(open_bounded_zip_entry(zf, info, max_bytes=max_bytes), info.filename, location) as stream:
        yield stream


@contextmanager
def open_admitted_blob(blob_store: BlobStore, blob_hash: str, name: str) -> Iterator[BinaryIO]:
    """Reopen bytes a boundary capture already validated (content-addressed)."""
    raw = BoundStream(blob_store.open(blob_hash), name, None, admitted=True)
    with io.BufferedReader(raw, buffer_size=_READ_CHUNK_BYTES) as stream:
        yield stream


def drain_bound(stream: IO[bytes]) -> None:
    """Read a bound stream to its end, validating every byte and retaining none."""
    if not is_bound(stream):
        raise TypeError("drain_bound needs a stream opened through the acquisition boundary")
    while stream.read(_READ_CHUNK_BYTES):
        pass


def refuse_foreign_path(path: Path | str, location: Provider | str | None) -> None:
    """Validate a whole source file at ``location`` without retaining it."""
    if bound_location_provider(location) is None:
        return
    with open_bound_path(path, location) as stream:
        drain_bound(stream)


def admit_bound_bytes(data: bytes, name: str, location: Provider | str | None) -> None:
    """Validate bytes already in memory (an appended delta) against ``location``."""
    refuse_declared_foreign(name, location)
    validator = BoundRecordValidator(name, location)
    validator.feed(data)
    validator.finish()


def capture_bound_stream(
    blob_store: BlobStore,
    stream: IO[bytes],
    *,
    heartbeat: Heartbeat | None = None,
) -> tuple[str, int]:
    """Retain a bound stream's bytes, validating exactly the bytes copied.

    A refusal (or any read failure) raises inside the copy, so the capture
    never reaches a publication queue.
    """
    if not is_bound(stream):
        raise TypeError("capture_bound_stream needs a stream opened through the acquisition boundary")
    if heartbeat is not None:
        heartbeat()
    return blob_store.write_from_fileobj(stream, heartbeat=heartbeat)


def capture_bound_path(
    blob_store: BlobStore,
    path: Path | str,
    location: Provider | str | None,
    *,
    heartbeat: Heartbeat | None = None,
) -> tuple[str, int]:
    """Retain one source file at ``location`` through the boundary."""
    with open_bound_path(path, location) as stream:
        return capture_bound_stream(blob_store, stream, heartbeat=heartbeat)


def release_refused_capture(blob_store: BlobStore, blob_hash: str, receipt_id: str | None) -> None:
    """Release one capture of a unit refused after that capture was queued or flushed.

    A unit (a file parsed record by record, a split ZIP member) can capture
    admitted parts before the boundary reaches a later foreign record.
    Dropping a pending publication or releasing a flushed reservation hands
    the bytes back to ordinary GC; nothing references them.
    """
    from polylogue.storage.blob_publication import release_refused_publication_receipt

    if receipt_id is None:
        # A store without receipts wrote final bytes; nothing references
        # them, so ordinary GC reclaims them.
        return
    discard_receipt = getattr(blob_store, "discard_pending_receipt", None)
    if callable(discard_receipt) and discard_receipt(receipt_id):
        return
    source_db_path = getattr(blob_store, "source_db_path", None)
    if source_db_path is not None:
        release_refused_publication_receipt(source_db_path, receipt_id, blob_hash)


@contextmanager
def release_captures_on_refusal(
    blob_store: BlobStore,
    *,
    refusals: tuple[type[Exception], ...] = (ForeignOriginContentError,),
) -> Iterator[list[tuple[str, str | None]]]:
    """Scope one admission unit: a refusal releases every capture it made.

    The caller appends each ``(blob_hash, receipt_id)`` it captures; on one of
    ``refusals`` all are released before the exception propagates, so a
    refused unit leaves no retained bytes. A caller whose scope yields
    nothing before it completes widens ``refusals`` to every failure: nothing
    it captured can be referenced then.
    """
    captures: list[tuple[str, str | None]] = []
    try:
        yield captures
    except refusals:
        for blob_hash, receipt_id in captures:
            release_refused_capture(blob_store, blob_hash, receipt_id)
        raise


__all__ = [
    "BoundRecordValidator",
    "BoundStream",
    "admit_bound_bytes",
    "bind_stream",
    "capture_bound_path",
    "capture_bound_stream",
    "drain_bound",
    "is_bound",
    "open_admitted_blob",
    "open_bound_member",
    "open_bound_path",
    "refuse_declared_foreign",
    "refuse_foreign_path",
    "release_captures_on_refusal",
    "release_refused_capture",
]
