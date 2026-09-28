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
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, BinaryIO

import ijson
from ijson.common import ObjectBuilder

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

    - A JSONL stream is validated record by record: each line is buffered
      whole (the parser materializes the same line) and decoded.
    - A ``.json`` document is pushed through an incremental parser. A
      top-level array is validated element by element, each as one record,
      so memory holds one element; a top-level object is validated as the
      document it is.

    No window truncates a record: a discriminator anywhere in a record is
    seen. A malformed or truncated record is validated from the structure
    that completed before the fault; malformed JSON itself is the parser's
    typed concern, not a foreign-origin claim. Inactive for unbound
    locations, declared ``raw-only`` paths and non-JSON material.
    """

    def __init__(self, name: str, location: Provider | str | None) -> None:
        self._bound = bound_location_provider(location)
        self._is_jsonl = is_jsonl_source_path(name)
        active = self._bound is not None and (self._is_jsonl or name.lower().endswith(".json"))
        if active and self._bound is not None:
            from .origin_specs import path_declaration_refuses_session

            active = not path_declaration_refuses_session(self._bound, Path(name))
        self.active = active
        self._line = bytearray()
        self._document = _DocumentValidator(self._bound) if active and not self._is_jsonl else None
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
            if newline == -1:
                self._line += chunk[start:]
                return
            self._line += chunk[start:newline]
            self._validate_line()
            start = newline + 1

    def finish(self) -> None:
        if not self.active or self._finished:
            return
        self._finished = True
        if self._document is not None:
            self._document.finish()
        elif self._line:
            self._validate_line()

    def _validate_line(self) -> None:
        line = bytes(self._line)
        self._line.clear()
        if not line.strip():
            return
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            record = _completed_structure(line)
            if not isinstance(record, dict):
                return
        detect_provider_evidence([record], expected=self._bound)


class _DocumentValidator:
    """Push-parse one JSON document, validating array elements as they complete."""

    def __init__(self, bound: Provider | None) -> None:
        self._bound = bound
        self._events = ijson.sendable_list()
        self._parser = ijson.basic_parse_coro(self._events, use_float=True)
        self._depth = 0
        self._top: str | None = None
        self._builder: ObjectBuilder | None = None
        self._failed = False

    def feed(self, chunk: bytes) -> None:
        if self._failed:
            return
        try:
            self._parser.send(chunk)
        except ijson.JSONError:
            self._failed = True
        self._drain()
        if self._failed:
            self._validate_partial()

    def finish(self) -> None:
        if self._failed:
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
                    self._builder = ObjectBuilder()
                    self._builder.event(event, value)
                self._depth = 1
            return
        if self._top == "start_array" and self._depth == 1:
            if closing:
                self._depth = 0
                return
            if not opening:
                return  # a scalar array element carries no record shape
            self._builder = ObjectBuilder()
        assert self._builder is not None
        self._builder.event(event, value)
        if opening:
            self._depth += 1
        elif closing:
            self._depth -= 1
            if self._depth == 1 and self._top == "start_array":
                self._validate(self._builder.value, element=True)
                self._builder = None
            elif self._depth == 0:
                self._validate(self._builder.value, element=False)
                self._builder = None

    def _validate_partial(self) -> None:
        if self._builder is None:
            return
        partial = getattr(self._builder, "value", None)
        self._builder = None
        if isinstance(partial, (dict, list)) and partial:
            self._validate(partial, element=self._top == "start_array")

    def _validate(self, value: object, *, element: bool) -> None:
        detect_provider_evidence([value] if element else value, expected=self._bound)


def _completed_structure(data: bytes) -> object:
    """The values whose lexical tokens completed inside malformed JSON bytes."""
    builder = ObjectBuilder()
    try:
        for event, value in ijson.basic_parse(io.BytesIO(data), use_float=True):
            builder.event(event, value)
    except ijson.JSONError:
        pass
    return getattr(builder, "value", None)


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
    from polylogue.storage.blob_publication import discard_pending_blob, release_refused_publication_receipt

    discard_receipt = getattr(blob_store, "discard_pending_receipt", None)
    if receipt_id is not None and callable(discard_receipt):
        if discard_receipt(receipt_id):
            return
    elif discard_pending_blob(blob_store, blob_hash):
        return
    source_db_path = getattr(blob_store, "source_db_path", None)
    if source_db_path is not None and receipt_id is not None:
        release_refused_publication_receipt(source_db_path, receipt_id, blob_hash)


@contextmanager
def release_captures_on_refusal(blob_store: BlobStore) -> Iterator[list[tuple[str, str | None]]]:
    """Scope one admission unit: a refusal releases every capture it made.

    The caller appends each ``(blob_hash, receipt_id)`` it captures; on
    :class:`ForeignOriginContentError` all are released before the refusal
    propagates, so a refused unit leaves no retained bytes.
    """
    captures: list[tuple[str, str | None]] = []
    try:
        yield captures
    except ForeignOriginContentError:
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
