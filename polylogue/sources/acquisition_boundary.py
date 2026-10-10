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

import codecs
import errno
import io
import json
import os
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, BinaryIO

import ijson
from ijson.backends import python as ijson_python

from polylogue.archive.zip_admission import open_zip_entry
from polylogue.core.enums import Provider
from polylogue.core.json import JSONDecodeError as FacadeJSONDecodeError
from polylogue.core.json import decode_provider_utf8
from polylogue.core.json import loads as json_loads
from polylogue.core.json_envelope import JSONL_MEMORY_BUFFER_BYTES
from polylogue.storage.blob_store import BlobStore, Heartbeat, PreparedBlob

from .dispatch import (
    ForeignOriginContentError,
    bound_location_provider,
    is_jsonl_source_path,
    same_origin,
)

if TYPE_CHECKING:
    import sqlite3

    from polylogue.schemas.observation_spill import _ScalarTokenStore
    from polylogue.sources.parsers.hermes_identity import CapturedHermesProfile
    from polylogue.sources.source_staging import SourceInputBinding
    from polylogue.sources.sqlite_export import SourceBytePage
    from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

_READ_CHUNK_BYTES = 1 << 20
# UTF-8, lexical transport and the event tokenizer each hold transient copies.
# Bound that work independently of the size requested by a stream consumer.
_VALIDATION_CHUNK_BYTES = 64 << 10


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
    classified through the registry's complete parser-owned projections;
    parser events are spooled privately rather than retaining a whole record.
    No byte window truncates the stream:
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
        #: Small records use the C decoder. Longer records continue through
        #: bounded event transport; this threshold changes memory strategy.
        self._pending = bytearray()
        self._line_limit = JSONL_MEMORY_BUFFER_BYTES if active and self._is_jsonl else 0
        self._line: _DocumentValidator | None = None
        self._spill_lifetime = ExitStack()
        self._spill_connection: sqlite3.Connection | None = None
        self._finished = False
        #: A refusal is sticky: a consumer that swallows it and reads on (an
        #: identity pass falling back to byte identity) meets it again.
        self._refusal: ForeignOriginContentError | None = None

    def feed(self, chunk: bytes) -> None:
        if self._refusal is not None:
            raise self._refusal
        if not self.active or not chunk:
            return
        try:
            self._feed(chunk)
        except ForeignOriginContentError as refusal:
            self._refusal = refusal
            raise

    def _feed(self, chunk: bytes) -> None:
        if self._document is not None:
            self._document.feed(chunk)
            return
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline == -1 else newline
            if end > start:
                self._feed_line(chunk[start:end])
            if newline == -1:
                return
            self._end_line()
            start = newline + 1

    def finish(self) -> None:
        if self._refusal is not None:
            raise self._refusal
        if not self.active or self._finished:
            return
        self._finished = True
        try:
            if self._document is not None:
                self._document.finish()
            else:
                self._end_line()
        except ForeignOriginContentError as refusal:
            self._refusal = refusal
            raise
        finally:
            self.close()

    def close(self) -> None:
        self._pending = bytearray()
        try:
            for document in (self._document, self._line):
                if document is not None:
                    document.close()
        finally:
            self._spill_lifetime.close()
            self._spill_connection = None

    def _feed_line(self, data: bytes) -> None:
        if self._line is None and len(self._pending) + len(data) <= self._line_limit:
            self._pending += data
            return
        if self._line is None:
            if self._spill_connection is None:
                from polylogue.schemas.observation_spill import StreamedJSONDocument

                owner = StreamedJSONDocument(None)
                self._spill_lifetime.enter_context(owner)
                self._spill_connection = owner.connection
            self._line = _DocumentValidator(self._bound, records=True, connection=self._spill_connection)
            held, self._pending = bytes(self._pending), bytearray()
            if held:
                self._line.feed(held)
        self._line.feed(data)

    def _end_line(self) -> None:
        line, self._line = self._line, None
        if line is not None:
            try:
                line.finish()
            finally:
                line.close()
            return
        held, self._pending = bytes(self._pending), bytearray()
        _validate_jsonl_record(held, self._bound)


def _validate_jsonl_record(line: bytes, bound: Provider | None) -> None:
    """Validate one complete JSONL line from its C-decoded value.

    The registry applies each binding's declared projection to the decoded
    record, as the event route does. A line that does not decode keeps the
    event route, which validates the structure completed before the fault.
    """
    raw = line.strip(b" \t\r\n")
    if not raw:
        return
    try:
        value = json_loads(raw)
    except FacadeJSONDecodeError:
        try:
            value = json.loads(decode_provider_utf8(raw), parse_constant=_refuse_constant)
        except (UnicodeDecodeError, ValueError):
            fallback = _DocumentValidator(bound, records=True)
            try:
                fallback.feed(raw)
                fallback.finish()
            finally:
                fallback.close()
            return
    if isinstance(value, dict):
        _validate_record_value(value, bound)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                _validate_record_value(item, bound)


def _refuse_constant(_constant: str) -> object:
    raise ValueError("non-finite JSON constant")


def _validate_record_value(value: object, bound: Provider | None) -> None:
    from .origin_specs import detector_registry

    for provider, evidence in detector_registry().iter_record_detections(value):
        if bound is not None and provider is not None and not same_origin(provider, bound):
            raise ForeignOriginContentError(expected=bound, found=provider, evidence=evidence or "record shape")


@contextmanager
def _record_evidence_file() -> Iterator[IO[bytes]]:
    with tempfile.TemporaryFile(mode="w+b", prefix="polylogue-origin-record-") as handle:
        yield handle


class _RecordEvidence:
    """Spool complete parser events without a second in-memory record."""

    def __init__(self, tokens: _ScalarTokenStore) -> None:
        self._lifetime = ExitStack()
        self._file = self._lifetime.enter_context(_record_evidence_file())
        self._stack: list[str] = []
        self._pending_key: object = None
        self._tokens = tokens

    def _write(self, event: str, value: object) -> None:
        from polylogue.schemas.observation_spill import SpilledKey, _ScalarTokenReference

        kind = None
        if isinstance(value, SpilledKey):
            kind, value = "key", value.token
        elif isinstance(value, _ScalarTokenReference):
            kind, value = value.kind, value.ordinal
        self._file.write(json.dumps((event, value, kind), ensure_ascii=True).encode("ascii") + b"\n")

    def event(self, event: str, value: object) -> None:
        if event == "map_key":
            self._pending_key = value
            return
        if event not in ("end_map", "end_array") and self._pending_key is not None:
            self._write("map_key", self._pending_key)
            self._pending_key = None
        self._write(event, value)
        if event in ("start_map", "start_array"):
            self._stack.append("end_map" if event == "start_map" else "end_array")
        elif event in ("end_map", "end_array"):
            self._stack.pop()

    def events(self) -> Iterator[tuple[str, object]]:
        self._file.seek(0)
        for line in self._file:
            event, value, kind = json.loads(line)
            if kind is not None:
                from polylogue.schemas.observation_spill import SpilledKey, _ScalarTokenReference

                value = (
                    SpilledKey(self._tokens.connection, value, self._tokens)
                    if kind == "key"
                    else _ScalarTokenReference(self._tokens, kind, value)
                )
            yield event, value
        # A syntax fault retains only fields whose values completed. Closing
        # the observed containers reproduces the existing partial-evidence
        # contract; no absent value or dangling key becomes evidence.
        for event in reversed(self._stack):
            yield event, None

    def close(self) -> None:
        self._lifetime.close()

    def validate(self, bound: Provider | None, *, record: bool) -> None:
        from .origin_specs import detector_registry

        try:
            for sequence in (False, True) if record else (False,):
                provider, evidence = detector_registry().detect_record_events(self.events, sequence=sequence)
                if bound is not None and provider is not None and not same_origin(provider, bound):
                    raise ForeignOriginContentError(expected=bound, found=provider, evidence=evidence or "record shape")
        finally:
            self.close()


class _DocumentValidator:
    """Push-parse one JSON value, validating array elements as they complete.

    ``records`` marks a JSONL line, whose top-level object is a record.
    """

    def __init__(self, bound: Provider | None, *, records: bool, connection: sqlite3.Connection | None = None) -> None:
        from polylogue.core.json_envelope import _PrefixStringReader
        from polylogue.schemas.observation_spill import StreamedJSONDocument, _ScalarTokenStore

        self._bound = bound
        self._records = records
        self._events = ijson.sendable_list()
        self._parser = ijson_python.basic_parse_coro(self._events, use_float=True)
        self._depth = 0
        self._top: str | None = None
        self._builder: _RecordEvidence | None = None
        self._failed = False
        self._seen = False
        self._lifetime = ExitStack()
        if connection is None:
            owner = StreamedJSONDocument(None)
            self._lifetime.enter_context(owner)
            connection = owner.connection
        else:
            # The byte stream owns the schema. All scalar and projection rows
            # belong to this record and expire after its evidence folds settle.
            connection.execute("SAVEPOINT origin_record")

            def release_record() -> None:
                connection.execute("ROLLBACK TO origin_record")
                connection.execute("RELEASE origin_record")

            self._lifetime.callback(release_record)
        self._tokens = _ScalarTokenStore(connection)
        self._transport = _PrefixStringReader(
            io.BytesIO(),
            scalar_values=True,
            string_sink=self._tokens.string,
            number_sink=self._tokens.number,
            allow_nonfinite=False,
        )
        self._utf8 = codecs.getincrementaldecoder("utf-8")()
        self._strings = self._numbers = 0

    def feed(self, chunk: bytes) -> None:
        for start in range(0, len(chunk), _VALIDATION_CHUNK_BYTES):
            if self._failed:
                return
            piece = chunk[start : start + _VALIDATION_CHUNK_BYTES]
            self._seen = self._seen or bool(piece.strip())
            try:
                transported = self._transport.feed(self._utf8.decode(piece).encode("utf-8"))
                if transported:
                    self._parser.send(transported)
            except (ijson.JSONError, json.JSONDecodeError, UnicodeError):
                self._failed = True
            self._drain()
            if self._failed:
                self._validate_partial()

    def finish(self) -> None:
        if self._failed or not self._seen:
            return
        try:
            final_text = self._utf8.decode(b"", final=True).encode("utf-8")
            transported = self._transport.feed(final_text) + self._transport.finish()
            if transported:
                self._parser.send(transported)
            self._parser.close()
        except (ijson.JSONError, json.JSONDecodeError, UnicodeError):
            self._failed = True
        self._drain()
        if self._failed or self._depth:
            self._validate_partial()

    def _drain(self) -> None:
        from polylogue.schemas.observation_spill import SpilledKey, _ScalarTokenReference

        for event, value in self._events:
            if event in ("map_key", "string"):
                self._strings += 1
                reference = _ScalarTokenReference(self._tokens, "string", self._strings)
                value = (
                    SpilledKey(self._tokens.connection, self._strings, self._tokens)
                    if event == "map_key"
                    else reference
                )
            elif event == "number":
                self._numbers += 1
                value = _ScalarTokenReference(self._tokens, "number", self._numbers)
            self._event(event, value)
        del self._events[:]

    def _event(self, event: str, value: object) -> None:
        opening = event in ("start_map", "start_array")
        closing = event in ("end_map", "end_array")
        if self._depth == 0:
            if opening:
                self._top = event
                if event == "start_map":
                    self._builder = _RecordEvidence(self._tokens)
                    self._builder.event(event, value)
                self._depth = 1
            return
        if self._top == "start_array" and self._depth == 1:
            if closing:
                self._depth = 0
                return
            if not opening:
                return  # a scalar array element carries no record shape
            self._builder = _RecordEvidence(self._tokens)
        assert self._builder is not None
        self._builder.event(event, value)
        if opening:
            self._depth += 1
        elif closing:
            self._depth -= 1
            if self._depth == 1 and self._top == "start_array":
                self._builder.validate(self._bound, record=True)
                self._builder = None
            elif self._depth == 0:
                self._builder.validate(self._bound, record=self._records)
                self._builder = None

    def close(self) -> None:
        try:
            if self._builder is not None:
                self._builder.close()
                self._builder = None
        finally:
            try:
                with suppress(ijson.JSONError, UnicodeError):
                    self._parser.close()
            finally:
                self._lifetime.close()

    def _validate_partial(self) -> None:
        builder, self._builder = self._builder, None
        if builder is not None:
            builder.validate(self._bound, record=self._records or self._top == "start_array")


class BoundStream(io.RawIOBase):
    """A raw reader that validates each source byte on its first read.

    Seeking back (decoders re-read ``.json`` documents) re-delivers bytes
    without re-validating them; seeking forward validates the skipped bytes
    first, so every byte a consumer can reach has been validated, in order.
    End of stream completes validation of a trailing unterminated record.
    """

    def __init__(
        self,
        raw: IO[bytes],
        name: str,
        location: Provider | str | None,
        *,
        admitted: bool = False,
        canonical_source_path: str | None = None,
        file_observation: tuple[int, int, int, int, int] | None = None,
        profile_identity: CapturedHermesProfile | None = None,
    ) -> None:
        super().__init__()
        if not admitted:
            refuse_declared_foreign(name, location)
        self._raw = raw
        self.canonical_source_path = canonical_source_path
        self.file_observation = file_observation
        self.profile_identity = profile_identity
        self.name = name
        self.location = bound_location_provider(location)
        self._validator = BoundRecordValidator(name, None if admitted else location)
        self._position = 0
        self._validated_to = 0

    def readable(self) -> bool:
        return True

    def fileno(self) -> int:
        """Expose the same opened descriptor; validation remains stream-owned."""
        return self._raw.fileno()

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
            self._validator.close()
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


def bind_stream(
    handle: IO[bytes],
    name: str,
    location: Provider | str | None,
    *,
    canonical_source_path: str | None = None,
    file_observation: tuple[int, int, int, int, int] | None = None,
    profile_identity: CapturedHermesProfile | None = None,
) -> BinaryIO:
    """Route an open source handle through the boundary (idempotent)."""
    if is_bound(handle):
        return handle  # type: ignore[return-value]
    return io.BufferedReader(
        BoundStream(
            handle,
            name,
            location,
            canonical_source_path=canonical_source_path,
            file_observation=file_observation,
            profile_identity=profile_identity,
        ),
        buffer_size=_READ_CHUNK_BYTES,
    )


def is_bound(handle: object) -> bool:
    """Whether ``handle`` already reads through the boundary."""
    return isinstance(handle, io.BufferedReader) and isinstance(handle.raw, BoundStream)


def bound_source_observation(stream: IO[bytes]) -> tuple[str | None, tuple[int, int, int, int, int] | None]:
    """Return only provenance captured by this stream's actual open owner."""
    if not isinstance(stream, io.BufferedReader) or not isinstance(stream.raw, BoundStream):
        raise TypeError("source observation needs an acquisition-bound stream")
    return stream.raw.canonical_source_path, stream.raw.file_observation


def bound_profile_identity(stream: IO[bytes]) -> CapturedHermesProfile | None:
    """Return the namespace captured alongside this actual opened input."""
    if not isinstance(stream, io.BufferedReader) or not isinstance(stream.raw, BoundStream):
        raise TypeError("profile identity needs an acquisition-bound stream")
    return stream.raw.profile_identity


@contextmanager
def open_bound_path(path: Path | str, location: Provider | str | None) -> Iterator[BinaryIO]:
    """Open and bind the source coordinate to the same descriptor as its bytes."""
    source = Path(path).absolute()
    refuse_declared_foreign(source.name, location)
    with ExitStack() as namespace:
        from polylogue.sources.parsers.hermes_identity import capture_profile_namespace

        directory_flags = getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW
        parent = os.open(source.parent.resolve(strict=True), directory_flags)
        namespace.callback(os.close, parent)
        profile = namespace.enter_context(capture_profile_namespace(source, parent))
        physical = source.resolve(strict=True)
        accepted_parent = os.fstat(parent)
        current_parent = source.parent.stat()
        if (accepted_parent.st_dev, accepted_parent.st_ino) != (current_parent.st_dev, current_parent.st_ino):
            raise OSError(errno.ESTALE, "source declared parent changed before opening", str(source))
        physical_parent = os.open(physical.parent, directory_flags)
        namespace.callback(os.close, physical_parent)
        expected = os.stat(physical.name, dir_fd=physical_parent, follow_symlinks=False)
        descriptor = os.open(physical.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=physical_parent)
        raw = namespace.enter_context(os.fdopen(descriptor, "rb"))
        observed = os.fstat(raw.fileno())
        if not stat.S_ISREG(observed.st_mode) or (observed.st_dev, observed.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise OSError(errno.ESTALE, "source differs from its accepted input", str(source))
        canonical = str(physical)
        with bind_stream(
            raw,
            str(source),
            location,
            canonical_source_path=canonical,
            profile_identity=profile,
            file_observation=(
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            ),
        ) as stream:
            yield stream


@contextmanager
def open_bound_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    location: Provider | str | None,
    *,
    profile_identity: CapturedHermesProfile | None = None,
) -> Iterator[BinaryIO]:
    """Open an admitted ZIP member through the boundary; the archive's location binds it."""
    refuse_declared_foreign(info.filename, location)
    with bind_stream(open_zip_entry(zf, info), info.filename, location, profile_identity=profile_identity) as stream:
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
    try:
        validator.feed(data)
        validator.finish()
    finally:
        validator.close()


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


def captured_path_coordinate(path: Path | str, descriptor: int) -> str:
    """Freeze the physical coordinate of the already opened input."""
    source = Path(path)
    observed = os.fstat(descriptor)
    canonical = source.resolve(strict=True)
    named = canonical.stat(follow_symlinks=False)
    if not stat.S_ISREG(observed.st_mode) or (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)) != (
        named.st_dev,
        named.st_ino,
        stat.S_IFMT(named.st_mode),
    ):
        raise OSError(errno.ESTALE, "source coordinate differs from its opened file", str(source))
    return str(canonical)


@dataclass(frozen=True, slots=True)
class BoundPathCapture:
    blob_hash: str
    blob_size: int
    canonical_source_path: str
    file_observation: tuple[int, int, int, int, int]
    captured_profile_key: str | None = None
    captured_profile_source_path: str | None = None


def capture_bound_path(
    blob_store: BlobStore,
    path: Path | str,
    location: Provider | str | None,
    *,
    heartbeat: Heartbeat | None = None,
    source_binding: SourceInputBinding | None = None,
    byte_page: SourceBytePage | None = None,
) -> BoundPathCapture:
    """Retain accepted bytes without closing source descriptors in this process.

    The fresh reader owns the source descriptor through final proof. The
    parent's sink validates bytes before its private blob writer can complete.
    A caller capturing a page of inputs lends its own ``byte_page`` so the
    whole page shares one reader process; otherwise one page serves this
    capture's binding and bytes.
    """
    from polylogue.sources.source_staging import bind_source_input, write_bound_input

    if source_binding is None:
        if byte_page is None:
            from polylogue.sources.sqlite_export import source_byte_page

            # One reader proves the binding and serves the bytes; binding
            # first through a fresh process paid a second interpreter start
            # for every captured file.
            with source_byte_page() as page:
                return capture_bound_path(blob_store, path, location, heartbeat=heartbeat, byte_page=page)
        with bind_source_input(Path(path), byte_page=byte_page) as binding:
            return capture_bound_path(
                blob_store, path, location, heartbeat=heartbeat, source_binding=binding, byte_page=byte_page
            )
    refuse_declared_foreign(str(source_binding.source_path), location)
    validator = BoundRecordValidator(str(source_binding.source_path), location)
    settlement: dict[str, object] = {}

    def produce(destination: IO[bytes]) -> None:
        class ValidatingSink:
            def write(self, data: bytes) -> int:
                if heartbeat is not None:
                    heartbeat()
                validator.feed(data)
                return destination.write(data)

        from polylogue.sources.sqlite_export import source_byte_page

        if byte_page is not None:
            settlement.update(write_bound_input(source_binding, ValidatingSink(), reader=byte_page))
        else:
            with source_byte_page() as reader:
                settlement.update(write_bound_input(source_binding, ValidatingSink(), reader=reader))
        validator.finish()

    try:
        blob_hash, blob_size = blob_store.write_from_writer(produce, heartbeat=heartbeat)
    finally:
        validator.close()
    if (settlement["content_revision"], settlement["size_bytes"]) != (blob_hash, blob_size):
        raise OSError(errno.ESTALE, "captured bytes differ from the accepted reader")
    observed = settlement["file_observation"]
    assert isinstance(observed, list)
    return BoundPathCapture(
        blob_hash,
        blob_size,
        str(source_binding.identity_path),
        (observed[0], observed[1], observed[2], observed[3], observed[4]),
        source_binding.captured_profile_key,
        str(source_binding.captured_profile_source_path),
    )


@dataclass(frozen=True, slots=True)
class BoundContainerCapture:
    """One proved private input and its existing publication ownership handoff."""

    stream: BinaryIO
    blob_hash: str
    blob_size: int
    captured_identity: CapturedSourceInputIdentity
    file_observation: tuple[int, int, int, int, int]
    _prepared: PreparedBlob
    _store: BlobStore
    _transferred: bool = False

    def retain(self) -> tuple[str, int, str | None]:
        """Retain this same prepared input, without reopening the source."""
        from polylogue.storage.blob_publication import ArchiveBlobPublisher, publication_receipt_id

        if self._transferred:
            raise ValueError("container publication was already transferred")
        if isinstance(self._store, ArchiveBlobPublisher):
            result = self._store.queue_prepared(self._prepared)
        else:
            result = self._store.publish_prepared(self._prepared)
        object.__setattr__(self, "_transferred", True)
        return (*result, publication_receipt_id(self._store, result[0]))


@contextmanager
def open_bound_container(
    blob_store: BlobStore,
    source_binding: SourceInputBinding,
    *,
    heartbeat: Heartbeat | None = None,
) -> Iterator[BoundContainerCapture]:
    """Read a private copy of the accepted container after its reader settles.

    The caller may transfer this exact prepared copy into frozen Source input
    custody. Otherwise it is discarded after readers settle. Source bytes are
    never reopened to construct a group identity.
    """
    from polylogue.sources.source_staging import write_bound_input

    settlement: dict[str, object] = {}

    def retain(destination: IO[bytes]) -> None:
        class ProgressSink:
            def write(self, data: bytes) -> int:
                if heartbeat is not None:
                    heartbeat()
                return destination.write(data)

        from polylogue.sources.sqlite_export import source_byte_page

        with source_byte_page() as reader:
            settlement.update(write_bound_input(source_binding, ProgressSink(), reader=reader))

    prepared = blob_store.prepare_from_writer(retain, heartbeat=heartbeat)
    capture = None
    try:
        if (settlement["content_revision"], settlement["size_bytes"]) != (prepared.hash_hex, prepared.size_bytes):
            raise OSError(errno.ESTALE, "container copy differs from its proved source descriptor")
        observed = settlement["file_observation"]
        if not isinstance(observed, list) or len(observed) != 5:
            raise ValueError("container reader did not return its descriptor observation")
        with prepared.temporary_path.open("rb") as stream:
            capture = BoundContainerCapture(
                stream,
                prepared.hash_hex,
                prepared.size_bytes,
                source_binding.captured_identity,
                (observed[0], observed[1], observed[2], observed[3], observed[4]),
                prepared,
                blob_store,
            )
            yield capture
    finally:
        if capture is None or not capture._transferred:
            blob_store.discard_prepared(prepared)


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
