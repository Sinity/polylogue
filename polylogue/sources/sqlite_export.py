"""Canonical logical export of a mutable SQLite source.

An export is the retained material for a live database member: the declared
logical tables, their schema objects and their typed rows, serialized in a
byte-reproducible framing. A page image is not the material -- it changes
after every commit, checkpoint and vacuum, cannot be proven against the live
database, and re-snapshots content the archive already holds.

Framing is one JSON document per line so a multi-gigabyte member streams in
and out in bounded memory:

    {"polylogue_sqlite_export":1,...,"tables":["threads",...]}
    {"table":"threads","columns":["id","title"],"sql":"CREATE TABLE ..."}
    [["t","019f..."],["t","a title"]]
    ...

Values carry their SQLite storage class so text, integers, reals, blobs and
NULL stay distinct: ``["i",5]``, ``["f",1.5]``, ``["t","text"]``, ``["tx",
"<hex>"]`` for TEXT whose bytes are not UTF-8, ``["b","<hex>"]`` for a blob,
and a bare ``null``. The ``sqlite_sequence`` header carries the same typed
name and high-water cells as retained identity evidence. Reconstructions
materialize the declared user tables, not this SQLite-owned header state.

``rowid`` is exported as a column for every rowid table, so a reconstruction
preserves row identity and every ``ORDER BY rowid`` a parser issues answers
exactly as it did against the live database. A table with a user column named
``rowid`` shadows the alias and is the one shape whose row identity cannot be
restored.

Readable generated columns carry their evaluated values from the same
acquisition snapshot as ordinary columns. Their original DDL remains evidence;
reconstruction stores the acquired typed values without replaying expressions.
Hidden virtual-table implementation columns remain outside the readable shape.
"""

from __future__ import annotations

import array
import errno
import hashlib
import json
import os
import resource
import socket
import sqlite3
import stat
import struct
import subprocess
import sys
import tempfile
import threading
from builtins import BaseExceptionGroup
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, closing, contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from polylogue.sources.source_staging import SourceInputBinding
from urllib.parse import quote_from_bytes

from polylogue.core.binary_signatures import SQLITE_MAGIC_HEADER
from polylogue.core.sql_settlement import current_native_sql_lifetimes, retain_native_sql_lifetimes
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    NativeSQLCustodyOwner,
    _close_failed_native_construction,
    retained_native_sql_owners_for_lifetime,
)

EXPORT_VERSION = 1
EXPORT_MAGIC = b'{"polylogue_sqlite_export":1'
#: Enough bytes to decide framing from a blob prefix without reading a row.
EXPORT_PROBE_BYTES = len(EXPORT_MAGIC)


class LogicalExportError(ValueError):
    """A retained export is not readable as one."""


class BinaryWriteSink(Protocol):
    """The streaming export's deliberately small binary output contract."""

    def write(self, payload: bytes) -> int: ...


@dataclass(frozen=True, slots=True)
class MemberExportScope:
    """The export scope and header one declared database member is acquired under."""

    tables: tuple[str, ...] | None = None
    member: str | None = None
    origin: str | None = None
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class LogicalExportHeader:
    """The first line of an export: what was acquired and from which member."""

    version: int
    member: str | None
    origin: str | None
    kind: str | None
    tables: tuple[str, ...]
    #: Declared tables the source did not have. A member whose product is
    #: wholly absent exports nothing, and this is the only record of why.
    missing: tuple[str, ...]
    columns: dict[str, tuple[str, ...]]
    schema: tuple[tuple[str | None, ...], ...]


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _control_bytes(value: object) -> bytes:
    # Protocol paths retain POSIX surrogate-escaped filenames. Canonical
    # export framing keeps its own unchanged UTF-8 serialization above.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")


def _schema_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _encode_value(storage_class: str, value: Any) -> Any:
    if storage_class == "null":
        return None
    if storage_class == "integer":
        return ["i", int(value)]
    if storage_class == "real":
        return ["f", float(value)]
    if storage_class == "blob":
        assert isinstance(value, bytes)
        return ["b", value.hex()]
    assert isinstance(value, bytes)
    try:
        return ["t", value.decode("utf-8")]
    except UnicodeDecodeError:
        return ["tx", value.hex()]


def _decode_value(encoded: Any) -> Any:
    if encoded is None:
        return None
    if not isinstance(encoded, list) or len(encoded) != 2:
        raise LogicalExportError(f"malformed export value: {encoded!r}")
    tag, payload = encoded
    if tag == "i":
        return int(payload)
    if tag == "f":
        return float(payload)
    if tag == "t":
        return str(payload)
    if tag == "b":
        return bytes.fromhex(str(payload))
    if tag == "tx":
        return bytes.fromhex(str(payload))
    raise LogicalExportError(f"unknown export value tag: {tag!r}")


_FileIdentity = tuple[int, int, int]
_SIDECARS = ("-wal", "-shm", "-journal")
_FRAME_HEADER = struct.Struct("!cQ")
_STREAM_CHUNK = 1024 * 1024
_WORKER_COMMAND = "from polylogue.sources.sqlite_export import _source_worker_main; _source_worker_main()"


def _identity(info: os.stat_result) -> _FileIdentity:
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def _named_identity(directory: int, name: str) -> _FileIdentity:
    identity = _identity(os.stat(name, dir_fd=directory, follow_symlinks=False))
    if identity[2] != stat.S_IFREG:
        raise OSError(errno.ELOOP, "SQLite source is not a regular file", name)
    return identity


def _descriptor_census() -> dict[int, _FileIdentity]:
    """Inspect descriptors without opening or closing a database file.

    The finite process descriptor limit is an OS bound, not an input cap.
    /proc enumerates the same set more cheaply where it is available.
    """
    try:
        descriptors: Iterator[int] = iter(int(name) for name in os.listdir("/proc/self/fd"))
    except FileNotFoundError:
        limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        if limit == resource.RLIM_INFINITY:
            limit = os.sysconf("SC_OPEN_MAX")
        if limit < 0:
            raise OSError(errno.ENOTSUP, "the process descriptor bound is unavailable") from None
        descriptors = iter(range(limit))
    result = {}
    for descriptor in descriptors:
        try:
            result[descriptor] = _identity(os.fstat(descriptor))
        except OSError as exc:
            if exc.errno != errno.EBADF:
                raise
    return result


class _SourceDescriptors:
    """Prove the isolated connection uses the accepted main and sidecars."""

    def __init__(self, directory: int, name: str, accepted: dict[str, _FileIdentity | None]) -> None:
        self.directory = directory
        self.name = name
        self.accepted = accepted
        self.baseline = _descriptor_census()
        self.bound: dict[int, _FileIdentity] = {}

    def validate(self) -> None:
        current = _descriptor_census()
        regular = {
            fd: identity
            for fd, identity in current.items()
            if identity[2] == stat.S_IFREG and self.baseline.get(fd) != identity
        }
        main = self.accepted[""]
        if main not in regular.values() or _named_identity(self.directory, self.name) != main:
            raise OSError(errno.ESTALE, "SQLite opened a different source database", self.name)
        for fd, identity in self.bound.items():
            if regular.get(fd) != identity:
                raise OSError(errno.ESTALE, "SQLite source descriptor changed", self.name)
        for fd, identity in regular.items():
            if identity == main:
                self.bound[fd] = identity
                continue
            for suffix in _SIDECARS:
                try:
                    named = _named_identity(self.directory, self.name + suffix)
                except FileNotFoundError:
                    continue
                if named != identity:
                    continue
                expected = self.accepted[suffix]
                if expected is not None and expected != identity:
                    raise OSError(errno.ESTALE, "SQLite opened a different source sidecar", self.name + suffix)
                self.accepted[suffix] = identity
                self.bound[fd] = identity
                break
            else:
                # No exemption for unlinked or temporary regular files: such
                # an exemption could hide an unlinked substituted WAL.
                raise OSError(errno.ESTALE, "SQLite opened an unbound source descriptor", self.name)


def _read_exact(stream: IO[bytes], count: int) -> bytes:
    payload = bytearray()
    while len(payload) < count:
        try:
            chunk = stream.read(count - len(payload))
        except OverflowError as exc:
            raise OSError(errno.EPROTO, "SQLite worker frame exceeds the physical read bound") from exc
        if not chunk:
            raise OSError(errno.EPIPE, "SQLite worker protocol ended before completion")
        payload.extend(chunk)
    return bytes(payload)


def _write_frame(stream: IO[bytes], kind: bytes, payload: bytes = b"") -> None:
    stream.write(_FRAME_HEADER.pack(kind, len(payload)))
    stream.write(payload)
    stream.flush()


class _WorkerSink:
    def write(self, payload: bytes) -> int:
        _write_frame(sys.stdout.buffer, b"D", payload)
        if _read_exact(sys.stdin.buffer, 1) != b"A":
            raise OSError(errno.EPROTO, "SQLite export callback was not acknowledged")
        return len(payload)


def _worker_error(exc: Exception) -> bytes:
    result: dict[str, Any] = {"message": str(exc)}
    if isinstance(exc, sqlite3.Error):
        result.update(
            kind="sqlite",
            type=type(exc).__name__,
            code=getattr(exc, "sqlite_errorcode", None),
            name=getattr(exc, "sqlite_errorname", None),
        )
    elif isinstance(exc, OSError):
        result.update(kind="os", errno=exc.errno, filename=exc.filename)
    elif isinstance(exc, UnicodeDecodeError):
        result.update(
            kind="unicode",
            encoding=exc.encoding,
            payload=exc.object.hex(),
            start=exc.start,
            end=exc.end,
            reason=exc.reason,
        )
    else:
        result.update(kind="value", type=type(exc).__name__)
    return _control_bytes(result)


def _decode_control(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise OSError(errno.EPROTO, "invalid SQLite worker control frame") from exc
    if not isinstance(value, dict):
        raise OSError(errno.EPROTO, "invalid SQLite worker control frame")
    return value


def _decode_shape(payload: bytes) -> dict[str, list[str]]:
    result = _decode_control(payload)
    items: Iterable[tuple[object, object]] = result.items()
    if not all(
        isinstance(key, str) and isinstance(value, list) and all(isinstance(column, str) for column in value)
        for key, value in items
    ):
        raise OSError(errno.EPROTO, "invalid SQLite worker shape")
    return result


def _raise_worker_error(payload: bytes) -> None:
    error = _decode_control(payload)
    try:
        message = error["message"]
        if not isinstance(message, str):
            raise ValueError("invalid error message")
        if error["kind"] == "sqlite":
            exception_type = {
                "DatabaseError": sqlite3.DatabaseError,
                "OperationalError": sqlite3.OperationalError,
                "IntegrityError": sqlite3.IntegrityError,
                "ProgrammingError": sqlite3.ProgrammingError,
                "DataError": sqlite3.DataError,
                "NotSupportedError": sqlite3.NotSupportedError,
                "InterfaceError": sqlite3.InterfaceError,
            }.get(error["type"], sqlite3.Error)
            exc: Exception = exception_type(message)
            if error["code"] is not None:
                assert isinstance(exc, sqlite3.Error)
                exc.sqlite_errorcode = error["code"]
                exc.sqlite_errorname = error["name"]
        elif error["kind"] == "os":
            exc = OSError(error["errno"], message, error["filename"])
        elif error["kind"] == "unicode":
            exc = UnicodeDecodeError(
                error["encoding"], bytes.fromhex(error["payload"]), error["start"], error["end"], error["reason"]
            )
        elif error["kind"] == "value":
            exc = LogicalExportError(message)
        else:
            raise ValueError("invalid error kind")
    except (KeyError, TypeError, ValueError) as malformed:
        raise OSError(errno.EPROTO, "invalid SQLite worker error") from malformed
    raise exc


@contextmanager
def _source_worker_process(pass_fds: tuple[int, ...]) -> Iterator[tuple[subprocess.Popen[bytes], str]]:
    """Own the existing isolated process, pipes and scratch through actual reap."""
    with tempfile.TemporaryDirectory(prefix=".polylogue-sqlite-reader.") as scratch:
        process = subprocess.Popen(
            [sys.executable, "-c", _WORKER_COMMAND],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=pass_fds,
            env={**os.environ, "TMPDIR": scratch},
        )
        try:
            yield process, scratch
        finally:
            # A failed callback, cancellation or malformed frame may leave the
            # child blocked on its ACK. Kill and reap that exact child before
            # releasing any pipe or directory descriptor; no time limit.
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdin is not None:
                with suppress(BrokenPipeError):
                    process.stdin.close()
            if process.stdout is not None:
                try:
                    while process.stdout.read(_STREAM_CHUNK):
                        pass
                finally:
                    process.stdout.close()


def _exchange_worker_request(
    process: subprocess.Popen[bytes],
    request: dict[str, Any],
    handle: BinaryWriteSink | None,
    *,
    final: bool,
    send_request: bool = True,
) -> dict[str, Any]:
    from polylogue.core.compute_cancel import check_compute_cancelled

    operation = request["operation"]
    assert process.stdin is not None and process.stdout is not None
    if send_request:
        _write_frame(process.stdin, b"Q", _control_bytes(request))
    result: dict[str, Any] = {}
    got_result = False
    while True:
        check_compute_cancelled()
        kind, size = _FRAME_HEADER.unpack(_read_exact(process.stdout, _FRAME_HEADER.size))
        if kind == b"D":
            if (
                operation
                not in {
                    "export",
                    "bytes",
                    "preflight_bytes",
                    "staging_receipt",
                    "copy",
                    "backup",
                    "inspect_preflight",
                    "inspect_explain",
                }
                or handle is None
            ):
                raise OSError(errno.EPROTO, "unexpected SQLite export frame")
            if not size:
                handle.write(b"")
            while size:
                check_compute_cancelled()
                chunk = _read_exact(process.stdout, min(size, _STREAM_CHUNK))
                if handle.write(chunk) != len(chunk):
                    raise OSError(errno.EIO, "source sink did not accept its complete frame")
                size -= len(chunk)
            process.stdin.write(b"A")
            process.stdin.flush()
        elif kind == b"R":
            if (
                operation
                not in {
                    "binding",
                    "shape",
                    "inspect_explain",
                    "inspect_preflight",
                    "classify",
                    "backup",
                    "copy",
                    "bytes",
                    "preflight_bytes",
                    "zip_container",
                    "staging_receipt",
                }
                or got_result
            ):
                raise OSError(errno.EPROTO, "unexpected SQLite shape frame")
            result = (_decode_shape if operation == "shape" else _decode_control)(_read_exact(process.stdout, size))
            got_result = True
        elif kind == b"E":
            _raise_worker_error(_read_exact(process.stdout, size))
        elif kind == (b"S" if final else b"C") and size == 0:
            if (
                operation
                in {
                    "binding",
                    "shape",
                    "inspect_explain",
                    "inspect_preflight",
                    "classify",
                    "backup",
                    "copy",
                    "bytes",
                    "preflight_bytes",
                    "zip_container",
                    "staging_receipt",
                }
                and not got_result
            ):
                raise OSError(errno.EPROTO, "SQLite worker omitted its shape result")
            if final and (process.stdout.read(1) or process.wait() != 0):
                raise OSError(errno.EPROTO, "SQLite worker did not settle successfully")
            return result
        else:
            raise OSError(errno.EPROTO, "invalid SQLite worker frame")


#: Operations a creator's live page process serves. A binding opens no
#: source content: it re-proves the input's name and staging provenance
#: through the same received directory capabilities as a byte read.
_BYTE_PAGE_OPERATIONS = frozenset({"bytes", "preflight_bytes", "binding"})


def _exchange_source_worker(request: dict[str, Any], handle: BinaryWriteSink | None = None) -> dict[str, Any]:
    """Exchange declared non-byte source operations with one fresh process."""
    if request["operation"] in {"bytes", "preflight_bytes"}:
        raise OSError(errno.EPROTO, "byte observations require their page owner")
    with _source_worker_process(tuple({request["directory"], request["metadata_directory"]})) as (process, scratch):
        return _exchange_worker_request(process, {**request, "scratch": scratch}, handle, final=True)


class SourceBytePage:
    """One creator-owned byte process, acquired only on the first page input."""

    def __init__(self, stack: ExitStack) -> None:
        self._stack = stack
        self._process: subprocess.Popen[bytes] | None = None
        self._channel: socket.socket | None = None
        self._scratch: str | None = None
        self._failure: BaseException | None = None
        self._finished = False
        self._creator = threading.current_thread()

    def exchange(self, request: dict[str, Any], handle: BinaryWriteSink | None) -> dict[str, Any]:
        from polylogue.core.compute_cancel import check_compute_cancelled

        if threading.current_thread() is not self._creator:
            raise RuntimeError("source byte page belongs to its original creator")
        if self._failure is not None or self._finished:
            raise RuntimeError("source byte page is no longer accepting observations")
        try:
            check_compute_cancelled()
            if request["operation"] not in _BYTE_PAGE_OPERATIONS:
                raise OSError(errno.EPROTO, "undeclared observation on source byte page")
            if self._process is None:
                parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
                self._stack.callback(parent.close)
                self._stack.callback(child.close)
                self._process, self._scratch = self._stack.enter_context(_source_worker_process((child.fileno(),)))
                assert self._process.stdin is not None
                _write_frame(
                    self._process.stdin, b"Q", _control_bytes({"operation": "byte_page", "channel": child.fileno()})
                )
                child.close()
                self._channel = parent
            assert self._channel is not None and self._process.stdin is not None
            directory, metadata = request["directory"], request["metadata_directory"]
            message = {
                **request,
                "scratch": self._scratch,
                "directory_identity": list(_identity(os.fstat(directory))),
                "metadata_identity": list(_identity(os.fstat(metadata))),
            }
            # One outstanding request and one ancillary marker bind the exact
            # two original directory capabilities to that request, in order.
            _write_frame(self._process.stdin, b"Q", _control_bytes(message))
            rights = array.array("i", (directory, metadata))
            if self._channel.sendmsg([b"A"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)]) != 1:
                raise OSError(errno.EPROTO, "source byte directory transfer was incomplete")
            return _exchange_worker_request(self._process, message, handle, final=False, send_request=False)
        except BaseException as error:
            self.reject(error)
            raise

    def reject(self, error: BaseException) -> None:
        """Settle the child before an original binding can release its anchors."""
        if threading.current_thread() is not self._creator:
            raise error
        if self._failure is None:
            self._failure = error
        self._stack.close()

    def finish(self) -> None:
        from polylogue.core.compute_cancel import check_compute_cancelled

        if threading.current_thread() is not self._creator:
            raise RuntimeError("source byte page must settle on its original creator")
        self._finished = True
        if self._failure is not None:
            raise self._failure
        check_compute_cancelled()
        if self._process is None:
            return
        assert self._process.stdin is not None and self._process.stdout is not None
        _write_frame(self._process.stdin, b"F")
        kind, size = _FRAME_HEADER.unpack(_read_exact(self._process.stdout, _FRAME_HEADER.size))
        if kind != b"S" or size or self._process.stdout.read(1) or self._process.wait() != 0:
            raise OSError(errno.EPROTO, "source byte page did not physically settle")


@contextmanager
def source_byte_page() -> Iterator[SourceBytePage]:
    """Lend a bounded caller page; no process or socket is created for empty input."""
    with ExitStack() as stack:
        page = SourceBytePage(stack)
        try:
            yield page
            page.finish()
        finally:
            page._finished = True


class SourceBytePageSequence:
    """Lend one live byte page at a time across a creator's sequential inputs.

    A request failure rejects its page, which kills and reaps that reader; the
    caller has already handled the failure for its own input, so the next
    input receives a fresh page instead of the refusal. A healthy page is
    reused, so a page of inputs pays for one reader process, not one each.
    """

    def __init__(self) -> None:
        self._creator = threading.current_thread()
        self._stack: ExitStack | None = None
        self._page: SourceBytePage | None = None

    def page(self) -> SourceBytePage:
        if threading.current_thread() is not self._creator:
            raise RuntimeError("source byte page sequence belongs to its original creator")
        page = self._page
        if page is not None and (page._failure is not None or page._finished):
            self._discard()
            page = None
        if page is None:
            self._stack = ExitStack()
            page = self._page = SourceBytePage(self._stack)
        return page

    def finish(self) -> None:
        """Settle the healthy current page through its normal final exchange."""
        page = self._page
        if page is None or page._failure is not None:
            self._discard()
            return
        try:
            page.finish()
        finally:
            self._discard()

    def _discard(self) -> None:
        page, stack = self._page, self._stack
        self._page = self._stack = None
        if page is not None:
            page._finished = True
        if stack is not None:
            # Rejection already closed a failed page's stack; closing again is
            # a no-op. An abandoned healthy page has its reader killed and reaped.
            stack.close()


@contextmanager
def source_byte_page_sequence() -> Iterator[SourceBytePageSequence]:
    """Own a sequence's current page; an abandoned sequence kills its reader."""
    sequence = SourceBytePageSequence()
    try:
        yield sequence
        sequence.finish()
    finally:
        sequence._discard()


def _bind_input_in_worker(request: dict[str, Any]) -> dict[str, Any]:
    """Prove one input's name and staging provenance inside a reader process."""
    source = Path(request["source"])
    accepted = request["identities"][""]
    main = None if accepted is None else cast(_FileIdentity, tuple(accepted))
    if main is None or _named_identity(request["directory"], source.name) != main:
        raise OSError(errno.ESTALE, "SQLite binding input changed", str(source))
    if request.get("staged_input") is None:
        original, provenance = None, None
    else:
        from polylogue.sources.source_staging import _verify_staging_metadata_name
        from polylogue.storage.sqlite.archive_tiers.source_items import CapturedSourceInputIdentity

        staged = request["staged_input"]
        if not isinstance(staged, dict) or set(staged) != {"identity", "provenance"}:
            raise OSError(errno.EPROTO, "invalid captured staging input")
        receipt = CapturedSourceInputIdentity.from_dict(staged["identity"])
        provenance = staged["provenance"]
        _verify_staging_metadata_name(request["metadata_directory"], provenance)
        original = {
            "source_path": receipt.semantic_source_path,
            "identity_path": receipt.canonical_source_path,
            "profile_root": receipt.profile_root,
            "profile_key": receipt.profile_key,
            "profile_source_path": receipt.profile_source_path,
        }
    if _named_identity(request["directory"], source.name) != main:
        raise OSError(errno.ESTALE, "SQLite binding input changed", str(source))
    return {
        "source_path": original["source_path"] if original is not None else request["semantic_source"],
        "profile": original,
        "provenance": provenance,
        "staged": original is not None,
    }


def _source_byte_page_main(channel_fd: int) -> None:
    """Each request closes its original file and received directories before C."""
    from polylogue.sources.source_staging import _read_bound_input_in_worker

    with socket.socket(fileno=channel_fd) as channel:
        while True:
            kind, size = _FRAME_HEADER.unpack(_read_exact(sys.stdin.buffer, _FRAME_HEADER.size))
            if kind == b"F" and size == 0:
                _write_frame(sys.stdout.buffer, b"S")
                return
            if kind != b"Q":
                raise OSError(errno.EPROTO, "invalid source byte page request")
            request = _decode_control(_read_exact(sys.stdin.buffer, size))
            descriptors: list[int] = []
            try:
                marker, ancillary, flags, _address = channel.recvmsg(
                    1, socket.CMSG_SPACE(2 * array.array("i").itemsize)
                )
                valid = marker == b"A" and not flags
                for level, kind, payload in ancillary:
                    if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                        valid = False
                        continue
                    rights = array.array("i")
                    complete = len(payload) - len(payload) % rights.itemsize
                    rights.frombytes(payload[:complete])
                    descriptors.extend(rights)
                    if complete != len(payload):
                        valid = False
                if not valid or len(descriptors) != 2:
                    raise OSError(errno.EPROTO, "invalid source byte directory capabilities")
                directory, metadata = descriptors
                for descriptor, expected in zip(
                    descriptors, (request["directory_identity"], request["metadata_identity"]), strict=True
                ):
                    observed = os.fstat(descriptor)
                    if not stat.S_ISDIR(observed.st_mode) or list(_identity(observed)) != expected:
                        raise OSError(errno.ESTALE, "source byte directory differs from its accepted anchor")
                if request["operation"] not in _BYTE_PAGE_OPERATIONS:
                    raise OSError(errno.EPROTO, "invalid source byte page operation")
                request["directory"], request["metadata_directory"] = directory, metadata
                result = (
                    _bind_input_in_worker(request)
                    if request["operation"] == "binding"
                    else _read_bound_input_in_worker(
                        request, _WorkerSink() if request["operation"] == "bytes" else None
                    )
                )
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)
            _write_frame(sys.stdout.buffer, b"R", _control_bytes(result))
            _write_frame(sys.stdout.buffer, b"C")


class _ProgressSink:
    """Cancellation ACKs for private copy operations carry no payload bytes."""

    def __init__(self, heartbeat: Callable[[], None]) -> None:
        self.heartbeat = heartbeat

    def write(self, data: bytes) -> int:
        if data:
            raise OSError(errno.EPROTO, "private copy progress unexpectedly contains payload")
        self.heartbeat()
        return 0


def _run_source_worker(
    source: Path,
    operation: str,
    *,
    handle: BinaryWriteSink | None = None,
    scope: MemberExportScope | None = None,
    immutable: bool = False,
    expected_identity: tuple[int, int] | None = None,
    destination: Path | None = None,
    parent_anchor: int | None = None,
    source_binding: SourceInputBinding | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from polylogue.sources.source_staging import _verify_staging_metadata_name, bind_source_input
    from polylogue.sources.sqlite_snapshot import member_export_scope

    if source_binding is None:
        with bind_source_input(source, parent_anchor=parent_anchor) as binding:
            return _run_source_worker(
                source,
                operation,
                handle=handle,
                scope=scope,
                immutable=immutable,
                expected_identity=expected_identity,
                destination=destination,
                parent_anchor=binding.parent_anchor,
                source_binding=binding,
                heartbeat=heartbeat,
            )
    source = source.absolute()
    if source_binding.source != source:
        raise OSError(errno.ESTALE, "SQLite binding belongs to another coordinate", str(source))
    source = source_binding.physical_path
    parent_identity = _identity(os.fstat(source_binding.parent_anchor))
    if parent_anchor is not None and _identity(os.fstat(parent_anchor)) != parent_identity:
        raise OSError(errno.ESTALE, "SQLite source parent differs from its accepted binding", str(source))
    directory = os.dup(source_binding.parent_anchor)
    parent = source.parent
    try:
        if _identity(os.fstat(directory)) != parent_identity:
            raise OSError(errno.ESTALE, "SQLite source parent changed", str(source))
        accepted: dict[str, _FileIdentity | None] = {"": _named_identity(directory, source.name)}
        main = accepted[""]
        assert main is not None
        if expected_identity is not None and main[:2] != expected_identity:
            raise OSError(errno.ESTALE, "SQLite source identity changed before opening", str(source))
        if main[:2] != source_binding.main_identity:
            raise OSError(errno.ESTALE, "SQLite provenance input changed before opening", str(source))
        _verify_staging_metadata_name(source_binding.metadata_anchor, source_binding.provenance)
        for suffix in _SIDECARS:
            try:
                accepted[suffix] = _named_identity(directory, source.name + suffix)
            except FileNotFoundError:
                accepted[suffix] = None
        existing_roles = [identity for identity in accepted.values() if identity is not None]
        if len(set(existing_roles)) != len(existing_roles):
            raise OSError(errno.ESTALE, "SQLite source roles share a physical file", str(source))
        request = {
            "source": str(parent / source.name),
            "inspection_path": str(source_binding.source_path),
            "profile_identity": source_binding.captured_profile_key,
            "profile_root": str(source_binding.captured_profile_root),
            "profile_source_path": str(source_binding.captured_profile_source_path),
            "directory": directory,
            "metadata_directory": source_binding.metadata_anchor,
            "identities": accepted,
            "operation": operation,
            "progress": heartbeat is not None,
            "scope": asdict(scope or MemberExportScope()),
            "immutable": immutable,
            "destination": None if destination is None else str(destination.absolute()),
            "provenance": source_binding.provenance,
            "expected_content_kind": source_binding.expected_content_kind,
            "expected_content_revision": source_binding.expected_content_revision,
            # The accepted revision was exported under the member's declared
            # scope (``_backup_source_database``); authenticating it under the
            # caller's operation scope compares two different exports.
            "expected_content_scope": asdict(member_export_scope(source_binding.source_path)),
            "accepted_source_path": str(
                source_binding.source_path if source_binding.staged else source_binding.physical_path
            ),
        }
        result = _exchange_source_worker(request, _ProgressSink(heartbeat) if heartbeat is not None else handle)
        if _identity(os.fstat(directory)) != parent_identity or _named_identity(directory, source.name) != main:
            raise OSError(errno.ESTALE, "SQLite source coordinate changed", str(source))
        # The worker request names the database by path, and SQLite resolves
        # its sidecars by that name. A matching main inode cannot prove the
        # read stayed under the accepted directory: an ancestor substituted
        # with a link to a hard link of the same file passes every descriptor
        # check, so the named parent must still be the anchored directory.
        try:
            named_parent = os.stat(parent)
        except OSError as exc:
            raise OSError(errno.ESTALE, "SQLite source parent path disappeared", str(source)) from exc
        if _identity(named_parent) != parent_identity:
            raise OSError(errno.ESTALE, "SQLite source parent path names another directory", str(source))
        _verify_staging_metadata_name(source_binding.metadata_anchor, source_binding.provenance)
        return result
    finally:
        os.close(directory)


def _export_shape_at(directory: int, name: str, expected: _FileIdentity | None) -> dict[str, list[str]] | None:
    # Probe in the fresh process before any SQLite source connection exists.
    # An ordinary DB fd closed in the parent could release another export's
    # process-scoped POSIX locks, even when this operation only wants shape.
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if _identity(before) != expected:
            raise OSError(errno.ESTALE, "source shape descriptor changed", name)
        if not handle.read(EXPORT_PROBE_BYTES).startswith(EXPORT_MAGIC):
            return None
        handle.seek(0)
        header = _parse_header(handle.readline())
        after = os.fstat(handle.fileno())
        if (before.st_ctime_ns, before.st_size) != (after.st_ctime_ns, after.st_size) or _named_identity(
            directory, name
        ) != expected:
            raise OSError(errno.ESTALE, "logical export changed during shape read", name)
    return {table: list(columns) for table, columns in header.columns.items()}


@dataclass(slots=True)
class _InspectionGrouping:
    connection: sqlite3.Connection
    path: Path
    identity: _FileIdentity
    descriptors: dict[int, _FileIdentity]

    def verify(self) -> None:
        current = _descriptor_census()
        if _identity(self.path.lstat()) != self.identity or any(
            current.get(fd) != identity for fd, identity in self.descriptors.items()
        ):
            raise OSError(errno.ESTALE, "private SQLite inspection grouping changed")


def _prepare_inspection_grouping(stack: ExitStack, scratch: Path) -> _InspectionGrouping:
    descriptor, name = tempfile.mkstemp(prefix=".polylogue-inspection.", suffix=".sqlite", dir=scratch)
    os.close(descriptor)
    path = Path(name)
    stack.callback(path.unlink, missing_ok=True)
    before = _descriptor_census()
    connection = stack.enter_context(_source_connection_context(path, readonly=False))
    connection.execute("PRAGMA journal_mode=OFF").close()
    connection.execute("PRAGMA temp_store=FILE").close()
    identity = _identity(path.lstat())
    descriptors = {
        fd: info for fd, info in _descriptor_census().items() if info[2] == stat.S_IFREG and before.get(fd) != info
    }
    if not descriptors or any(info != identity for info in descriptors.values()):
        raise OSError(errno.ESTALE, "private SQLite inspection grouping is not bound")
    return _InspectionGrouping(connection, path, identity, descriptors)


def _inspect_export_at(
    directory: int,
    source: Path,
    expected: _FileIdentity | None,
    *,
    preflight: bool,
    inspection_path: Path,
    profile_identity: str,
    scratch: Path,
    classify: bool = False,
    check_stop: Callable[[], None] | None = None,
) -> dict[str, Any] | None:
    """Reconstruct retained exports from their accepted descriptor, preserving read semantics."""
    from polylogue.sources.parsers.hermes_state import _MESSAGE_READ_INDEXES
    from polylogue.sources.sqlite_inspection import SQLiteInspection, _classify_connection, _inspect_connection

    descriptor = os.open(source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if _identity(before) != expected:
            raise OSError(errno.ESTALE, "source inspection descriptor changed", str(source))
        prefix = handle.read(EXPORT_PROBE_BYTES)
        if prefix.startswith(SQLITE_MAGIC_HEADER):
            return None
        if not prefix.startswith(EXPORT_MAGIC):
            if classify:
                raise sqlite3.DatabaseError("not a SQLite database or logical export")
            return asdict(SQLiteInspection(None, {}))
        handle.seek(0)
        temporary, name = tempfile.mkstemp(prefix=".polylogue-export.", suffix=".sqlite", dir=scratch)
        os.close(temporary)
        reconstruction = Path(name)
        try:
            _materialize_export_records(
                _iter_export_handle(handle), reconstruction, read_indexes=_MESSAGE_READ_INDEXES, check_stop=check_stop
            )
            parent_fd = os.open(reconstruction.parent, getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY)
            try:
                with ExitStack() as stack:
                    grouping = None if classify else _prepare_inspection_grouping(stack, scratch)
                    identities: dict[str, _FileIdentity | None] = {
                        "": _identity(reconstruction.lstat()),
                        **dict.fromkeys(_SIDECARS),
                    }
                    proof = _SourceDescriptors(parent_fd, reconstruction.name, identities)
                    with _source_connection_context(reconstruction, immutable=True, directory=parent_fd) as conn:
                        proof.validate()
                        if check_stop is not None:
                            conn.set_progress_handler(lambda: (check_stop(), 0)[1], 1000)
                            if grouping is not None:
                                grouping.connection.set_progress_handler(lambda: (check_stop(), 0)[1], 1000)
                        conn.execute("BEGIN").close()
                        _source_schema(conn)
                        proof.validate()
                        result = (
                            asdict(_classify_connection(conn))
                            if classify
                            else asdict(
                                _inspect_connection(
                                    conn,
                                    inspection_path,
                                    preflight=preflight,
                                    profile_identity=profile_identity,
                                    grouping=None if grouping is None else grouping.connection,
                                )
                            )
                        )
                        proof.validate()
                        if grouping is not None:
                            grouping.verify()
            finally:
                os.close(parent_fd)
            after = os.fstat(handle.fileno())
            if (before.st_ctime_ns, before.st_size) != (after.st_ctime_ns, after.st_size) or _named_identity(
                directory, source.name
            ) != expected:
                raise OSError(errno.ESTALE, "logical export changed during inspection", str(source))
            return result
        finally:
            reconstruction.unlink(missing_ok=True)


def _source_worker_main() -> None:
    """Private fresh-exec entry; callers only use the existing source APIs."""
    try:
        kind, size = _FRAME_HEADER.unpack(_read_exact(sys.stdin.buffer, _FRAME_HEADER.size))
        if kind != b"Q":
            raise OSError(errno.EPROTO, "invalid SQLite source request")
        request = json.loads(_read_exact(sys.stdin.buffer, size))
        if request["operation"] == "byte_page":
            _source_byte_page_main(request["channel"])
            return
        source = Path(request["source"])
        accepted = {
            name: None if identity is None else cast(_FileIdentity, tuple(identity))
            for name, identity in request["identities"].items()
        }
        if request["operation"] == "staging_receipt":
            from polylogue.sources.source_staging import _read_staging_receipt_in_worker

            result = _read_staging_receipt_in_worker(request, _WorkerSink())
            _write_frame(sys.stdout.buffer, b"R", _control_bytes(result))
            _write_frame(sys.stdout.buffer, b"S")
            return
        if request["operation"] == "zip_container":
            from polylogue.sources.source_staging import _probe_zip_container_in_worker

            result = _probe_zip_container_in_worker(request)
            _write_frame(sys.stdout.buffer, b"R", _control_bytes(result))
            _write_frame(sys.stdout.buffer, b"S")
            return
        if request["operation"] == "binding":
            _write_frame(sys.stdout.buffer, b"R", _control_bytes(_bind_input_in_worker(request)))
            _write_frame(sys.stdout.buffer, b"S")
            return
        if request["operation"] == "copy":
            from polylogue.sources.source_staging import _copy_bound_input_in_worker

            result = _copy_bound_input_in_worker(request)
            _write_frame(sys.stdout.buffer, b"R", _control_bytes(result))
            _write_frame(sys.stdout.buffer, b"S")
            return
        scope = MemberExportScope(**request["scope"])
        from polylogue.sources.source_staging import _verify_staging_metadata_name

        def progress() -> None:
            if request.get("progress"):
                _WorkerSink().write(b"")

        def sql_progress() -> int:
            progress()
            return 0

        _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
        if request["operation"] == "shape":
            export_shape = _export_shape_at(request["directory"], source.name, accepted[""])
            if export_shape is not None:
                _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
                _write_frame(sys.stdout.buffer, b"R", _control_bytes(export_shape))
                _write_frame(sys.stdout.buffer, b"S")
                return
        if request["operation"] in {"inspect_explain", "inspect_preflight", "classify"}:
            export_inspection = _inspect_export_at(
                request["directory"],
                source,
                accepted[""],
                preflight=request["operation"] == "inspect_preflight",
                inspection_path=Path(request["inspection_path"]),
                profile_identity=request["profile_identity"],
                scratch=Path(request["scratch"]),
                classify=request["operation"] == "classify",
                check_stop=progress if request.get("progress") else None,
            )
            if export_inspection is not None:
                _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
                _write_frame(sys.stdout.buffer, b"R", _control_bytes(export_inspection))
                _write_frame(sys.stdout.buffer, b"S")
                return
        with ExitStack() as stack:
            output = None
            output_identity = None
            shape: dict[str, Any] = {}
            output_descriptors: dict[int, _FileIdentity] = {}
            grouping = (
                _prepare_inspection_grouping(stack, Path(request["scratch"]))
                if request["operation"] in {"inspect_explain", "inspect_preflight"}
                else None
            )
            if request["operation"] == "backup":
                before_output = _descriptor_census()
                output = stack.enter_context(_source_connection_context(Path(request["destination"]), readonly=False))
                output_identity = _identity(Path(request["destination"]).lstat())
                output_descriptors = {
                    fd: identity
                    for fd, identity in _descriptor_census().items()
                    if identity[2] == stat.S_IFREG and before_output.get(fd) != identity
                }
                if not output_descriptors or any(
                    identity != output_identity for identity in output_descriptors.values()
                ):
                    raise OSError(errno.ESTALE, "SQLite backup destination is not bound")
            # The existing staged-backup destination belongs to its own SQLite
            # connection and is excluded before the source descriptor baseline.
            proof = _SourceDescriptors(request["directory"], source.name, accepted)
            with _source_connection_context(
                source, immutable=request["immutable"], directory=request["directory"]
            ) as conn:
                proof.validate()
                _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
                if request.get("progress"):
                    conn.set_progress_handler(sql_progress, 1000)
                    if grouping is not None:
                        grouping.connection.set_progress_handler(sql_progress, 1000)
                # Sorting a complete source or preview denominator must spill
                # regardless of the SQLite build's default TEMP policy. Main
                # descriptor proof precedes SQL; no transaction or TEMP object
                # exists yet, so selecting this policy cannot discard state.
                conn.execute("PRAGMA temp_store=FILE").close()
                conn.text_factory = bytes
                conn.execute("BEGIN").close()
                schema = _source_schema(conn)
                proof.validate()
                expected_revision = request.get("expected_content_revision")
                if expected_revision is not None:
                    if request.get("expected_content_kind") != "sqlite":
                        raise OSError(errno.ESTALE, "staged source is not the accepted SQLite input", str(source))
                    accepted_revision = _HashingSink(progress if request.get("progress") else None)
                    _write_export_connection(
                        conn, accepted_revision, MemberExportScope(**request["expected_content_scope"]), schema
                    )
                    proof.validate()
                    if accepted_revision.hexdigest() != expected_revision:
                        raise OSError(errno.ESTALE, "staged SQLite differs from the accepted input", str(source))
                if request["operation"] == "export":
                    _write_export_connection(conn, _WorkerSink(), scope, schema)
                elif request["operation"] == "shape":
                    shape = {}
                    for row in schema:
                        if _schema_text(row[0]) == "table":
                            table = _schema_text(row[1])
                            shape[table] = [_schema_text(item[1]) for item in readable_table_info(conn, table)]
                elif request["operation"] == "classify":
                    from polylogue.sources.sqlite_inspection import _classify_connection

                    shape = asdict(_classify_connection(conn))
                elif request["operation"] in {"inspect_explain", "inspect_preflight"}:
                    from polylogue.sources.sqlite_inspection import _inspect_connection

                    shape = asdict(
                        _inspect_connection(
                            conn,
                            Path(request["inspection_path"]),
                            preflight=request["operation"] == "inspect_preflight",
                            profile_identity=request["profile_identity"],
                            grouping=None if grouping is None else grouping.connection,
                        )
                    )
                elif request["operation"] == "backup" and output is not None:
                    assert output_identity is not None
                    retained_revision = _HashingSink(progress if request.get("progress") else None)
                    _write_export_connection(conn, retained_revision, scope, schema)
                    conn.backup(output, pages=256, progress=lambda *_counts: progress())
                    shape = {
                        "source_path": request["accepted_source_path"],
                        "declared_source_path": request["inspection_path"],
                        "profile_key": request["profile_identity"],
                        "profile_root": request["profile_root"],
                        "profile_source_path": request["profile_source_path"],
                        "database_identity": list(output_identity[:2]),
                        "logical_revision": retained_revision.hexdigest(),
                    }
                else:
                    raise OSError(errno.EPROTO, "invalid SQLite source operation")
                proof.validate()
                _verify_staging_metadata_name(request["metadata_directory"], request["provenance"])
                if grouping is not None:
                    grouping.verify()
                if output_descriptors:
                    current = _descriptor_census()
                    if (
                        any(current.get(fd) != identity for fd, identity in output_descriptors.items())
                        or _identity(Path(request["destination"]).lstat()) != output_identity
                    ):
                        raise OSError(errno.ESTALE, "SQLite backup destination changed")
            if request["operation"] in {"shape", "inspect_explain", "inspect_preflight", "classify", "backup"}:
                _write_frame(sys.stdout.buffer, b"R", _control_bytes(shape))
        _write_frame(sys.stdout.buffer, b"S")
    except Exception as exc:
        _write_frame(sys.stdout.buffer, b"E", _worker_error(exc))
        raise SystemExit(1) from None


def _backup_source_database(
    source: Path,
    destination: Path,
    *,
    source_binding: SourceInputBinding | None = None,
    expected_identity: tuple[int, int] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> dict[str, Any]:
    from polylogue.sources.source_staging import bind_source_input
    from polylogue.sources.sqlite_snapshot import member_export_scope

    if source_binding is None:
        with bind_source_input(source) as binding:
            return _backup_source_database(
                source, destination, source_binding=binding, expected_identity=expected_identity, heartbeat=heartbeat
            )
    result = _run_source_worker(
        source,
        "backup",
        destination=destination,
        source_binding=source_binding,
        expected_identity=expected_identity,
        scope=member_export_scope(source_binding.source_path),
        heartbeat=heartbeat,
    )
    identity = result.get("database_identity")
    if (
        set(result)
        != {
            "source_path",
            "declared_source_path",
            "profile_key",
            "profile_root",
            "profile_source_path",
            "database_identity",
            "logical_revision",
        }
        or not isinstance(result.get("logical_revision"), str)
        or len(result["logical_revision"]) != 64
        or any(character not in "0123456789abcdef" for character in result["logical_revision"])
        or not isinstance(result.get("declared_source_path"), str)
        or not Path(result["declared_source_path"]).is_absolute()
        or not isinstance(result.get("profile_root"), str)
        or not Path(result["profile_root"]).is_absolute()
        or not isinstance(result.get("profile_source_path"), str)
        or not Path(result["profile_source_path"]).is_absolute()
        or not isinstance(result.get("profile_key"), str)
        or len(result["profile_key"]) != 12
        or any(character not in "0123456789abcdef" for character in result["profile_key"])
        or not isinstance(result.get("source_path"), str)
        or not Path(result["source_path"]).is_absolute()
        or not isinstance(identity, list)
        or len(identity) != 2
        or any(type(value) is not int for value in identity)
    ):
        raise OSError(errno.EPROTO, "invalid SQLite backup result")
    return result


@contextmanager
def _source_connection_context(
    path: Path,
    *,
    immutable: bool = False,
    timeout: float = 5.0,
    readonly: bool = True,
    scratch_directory: tempfile.TemporaryDirectory[str] | None = None,
    directory: int | None = None,
) -> Iterator[sqlite3.Connection]:
    if directory is not None:
        # Fresh reader only: VFS sidecar opens retain the accepted parent.
        os.fchdir(directory)
        uri = f"file:{quote_from_bytes(os.fsencode(path.name), safe='')}?mode={'ro' if readonly else 'rwc'}"
    else:
        uri = f"{path.resolve().as_uri()}?mode={'ro' if readonly else 'rwc'}"
    if immutable:
        uri += "&immutable=1"
    connection = connect_measured(uri, uri=True, timeout=timeout)
    owner = NativeSQLCustodyOwner(
        connection,
        scratch_directory=scratch_directory,
        lifetime_dependencies=current_native_sql_lifetimes(),
    )
    try:
        yield owner.require_connection()
    except BaseException as primary:
        _close_failed_native_construction(owner, primary)
        raise
    else:
        owner.close()


def readable_table_info(conn: sqlite3.Connection, table: str) -> list[tuple[Any, ...]]:
    """Return ordered metadata for ordinary and readable generated columns.

    ``table_xinfo`` retains the first six ``table_info`` metadata positions.
    Its final flag distinguishes virtual-table implementation columns (1)
    from ordinary (0), VIRTUAL generated (2), and STORED generated (3) values.
    """
    quoted = '"' + table.replace('"', '""') + '"'
    # Allocate before execute: a failing statement still belongs to this creator.
    from polylogue.storage.io_phase_metrics import _MeasuredConnection, close_connection_cursor

    cursor = conn.cursor()
    primary: BaseException | None = None
    try:
        cursor.execute(f"PRAGMA table_xinfo({quoted})")
        return [tuple(row) for row in cursor if int(row[6]) in (0, 2, 3)]
    except BaseException as failure:
        primary = failure
        raise
    finally:
        try:
            if isinstance(conn, _MeasuredConnection):
                close_connection_cursor(conn, cursor)
            else:
                cursor.close()
        except BaseException as cleanup:
            if primary is not None:
                raise BaseExceptionGroup("Statement and native cursor close failed", [primary, cleanup]) from primary
            raise


def _table_plan(conn: sqlite3.Connection, table: str, table_sql: str) -> tuple[list[str], str, list[str], bool]:
    """Return the exported columns, the row order, the declared columns, and
    whether the first exported column is the synthetic ``rowid``."""
    columns = readable_table_info(conn, table)
    column_names = [_schema_text(row[1]) for row in columns]
    is_without_rowid = "WITHOUT ROWID" in table_sql.upper()
    # A user column literally named ``rowid`` shadows the alias, so the
    # synthetic column would be a duplicate rather than the row's identity.
    # Export the rowid for every rowid table, including one whose INTEGER
    # PRIMARY KEY already carries it: the reconstruction declares columns
    # untyped, so nothing else would restore the row identity a parser reads
    # through ``rowid``. A user column of that name shadows the alias, and
    # then no rowid can be restored at all.
    shadowed = "rowid" in column_names
    synthetic_rowid = not is_without_rowid and not shadowed
    selected = (["rowid"] if synthetic_rowid else []) + column_names
    if not is_without_rowid and not shadowed:
        # ``rowid`` is unique, never NULL, and the physical storage order, so
        # it is a total order that costs no sorter. Every rowid table's rowid
        # is part of the exported content -- either as the INTEGER PRIMARY KEY
        # or as the synthetic column above -- so ordering by it adds no
        # dependency the digest did not already have. A declared PRIMARY KEY
        # is not a substitute: SQLite lets a rowid table's PK columns be NULL,
        # so it is not reliably unique.
        return selected, "rowid", column_names, synthetic_rowid
    ordered = [
        name for _primary_key_position, name in sorted((int(row[5]), _schema_text(row[1])) for row in columns if row[5])
    ]
    order_terms: list[str] = []
    for name in ordered or column_names:
        quoted_column = '"' + name.replace('"', '""') + '"'
        order_terms.extend((f"typeof({quoted_column}) COLLATE BINARY", f"{quoted_column} COLLATE BINARY"))
    return selected, ", ".join(order_terms), column_names, synthetic_rowid


def _source_schema(conn: sqlite3.Connection) -> list[tuple[Any, ...]]:
    with closing(
        conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        )
    ) as cursor:
        return cursor.fetchall()


def _write_export_connection(
    conn: sqlite3.Connection,
    handle: BinaryWriteSink,
    scope: MemberExportScope,
    schema_objects: list[tuple[Any, ...]],
) -> None:
    member, origin, kind = scope.member, scope.origin, scope.kind
    tables = scope.tables
    declared = None if tables is None else set(tables)
    selected_schema = [
        tuple(_schema_text(value) if value is not None else None for value in row)
        for row in schema_objects
        if declared is None or _schema_text(row[2]) in declared
    ]
    table_sql = {
        _schema_text(row[1]): _schema_text(row[3])
        for row in schema_objects
        if _schema_text(row[0]) == "table" and (declared is None or _schema_text(row[1]) in declared)
    }
    exported_tables = [
        name for name, sql in table_sql.items() if not sql.lstrip().upper().startswith("CREATE VIRTUAL TABLE")
    ]
    # Plan every table before the header so a reader can answer a shape
    # question -- "does this export carry these tables with these columns?"
    # -- from the first line, without materializing a single row.
    plans = {table: _table_plan(conn, table, table_sql[table]) for table in exported_tables}
    missing = () if tables is None else tuple(sorted(set(tables) - set(exported_tables)))
    with closing(
        conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'")
    ) as cursor:
        sequence_present = bool(cursor.fetchone())
    sequence_rows: list[list[Any]] | None = None
    if sequence_present:
        # sqlite_sequence is SQLite-owned and excluded from the schema
        # enumeration above with the rest of the sqlite_% names, but its
        # contents are logical state: an insert-then-delete on an
        # AUTOINCREMENT table leaves every user row identical while
        # advancing the stored high-water mark.
        sequence_names = None if declared is None else {name.encode("utf-8") for name in declared}
        with closing(
            conn.execute("SELECT typeof(name), name, typeof(seq), seq FROM sqlite_sequence ORDER BY name")
        ) as cursor:
            sequence_rows = [
                [_encode_value(_schema_text(name_type), name), _encode_value(_schema_text(seq_type), seq)]
                for name_type, name, seq_type, seq in cursor
                if sequence_names is None or (name_type == b"text" and name in sequence_names)
            ]
    header = (
        '{"polylogue_sqlite_export":'
        + str(EXPORT_VERSION)
        + ',"kind":'
        + _dumps(kind)
        + ',"member":'
        + _dumps(member)
        + ',"missing":'
        + _dumps(list(missing))
        + ',"origin":'
        + _dumps(origin)
        + ',"columns":'
        + _dumps({table: plans[table][2] for table in exported_tables})
        + ',"schema":'
        + _dumps(selected_schema)
        + ',"sqlite_sequence":'
        + _dumps(sequence_rows)
        + ',"tables":'
        + _dumps(exported_tables)
        + "}\n"
    )
    handle.write(header.encode("utf-8"))
    for table in exported_tables:
        selected, order, _declared, synthetic_rowid = plans[table]
        handle.write(
            (
                '{"table":'
                + _dumps(table)
                + ',"columns":'
                + _dumps(selected)
                + ',"rowid":'
                + ("true" if synthetic_rowid else "false")
                + ',"sql":'
                + _dumps(table_sql[table])
                + "}\n"
            ).encode("utf-8")
        )
        quoted = '"' + table.replace('"', '""') + '"'
        projection = ", ".join(
            f"typeof({quoted_name}), {quoted_name}"
            for name in selected
            for quoted_name in ('"' + name.replace('"', '""') + '"',)
        )
        statement = f"SELECT {projection} FROM {quoted}" + (f" ORDER BY {order}" if order else "")
        with closing(conn.execute(statement)) as cursor:
            for row in cursor:
                encoded = [
                    _encode_value(_schema_text(storage_class), value)
                    for storage_class, value in zip(row[::2], row[1::2], strict=True)
                ]
                handle.write((_dumps(encoded) + "\n").encode("utf-8"))


def write_logical_export(
    source: Path,
    handle: BinaryWriteSink,
    *,
    scope: MemberExportScope | None = None,
    tables: Sequence[str] | None = None,
    immutable: bool = False,
) -> None:
    """Stream a canonical export from one descriptor-validated SQLite transaction.

    The isolated reader owns SQLite's descriptors through final validation.
    Each sink callback completes before that transaction reads the next frame;
    an exception leaves callers with an unfinished export, never success.
    """
    _write_logical_export_bound(source, handle, scope=scope, tables=tables, immutable=immutable)


def _write_logical_export_bound(
    source: Path,
    handle: BinaryWriteSink,
    *,
    scope: MemberExportScope | None = None,
    tables: Sequence[str] | None = None,
    immutable: bool = False,
    expected_identity: tuple[int, int] | None = None,
    parent_anchor: int | None = None,
    source_binding: SourceInputBinding | None = None,
) -> None:
    scope = scope or MemberExportScope()
    if tables is not None:
        scope = replace(scope, tables=tuple(tables))
    _run_source_worker(
        source,
        "export",
        handle=handle,
        scope=scope,
        immutable=immutable,
        expected_identity=expected_identity,
        parent_anchor=parent_anchor,
        source_binding=source_binding,
    )


def logical_export_bytes(source: Path, **kwargs: Any) -> bytes:
    """Return the canonical export of *source* as bytes."""
    from io import BytesIO

    buffer = BytesIO()
    write_logical_export(source, buffer, **kwargs)
    return buffer.getvalue()


class _HashingSink:
    """A write-only sink that keeps the digest and discards the bytes."""

    def __init__(self, heartbeat: Callable[[], None] | None = None) -> None:
        self._digest = hashlib.sha256()
        self.byte_count = 0
        self._heartbeat = heartbeat
        self._progress_bytes = 0

    def write(self, payload: bytes) -> int:
        self._digest.update(payload)
        self.byte_count += len(payload)
        self._progress_bytes += len(payload)
        if self._heartbeat is not None and self._progress_bytes >= _STREAM_CHUNK:
            self._heartbeat()
            self._progress_bytes = 0
        return len(payload)

    def hexdigest(self) -> str:
        if self._heartbeat is not None:
            self._heartbeat()
        return self._digest.hexdigest()


def logical_export_digest(source: Path, **kwargs: Any) -> str:
    """Digest *source*'s canonical export without materializing it."""
    sink = _HashingSink()
    write_logical_export(source, sink, **kwargs)
    return sink.hexdigest()


def _logical_export_digest_bound(
    source: Path,
    *,
    scope: MemberExportScope,
    expected_identity: tuple[int, int],
    parent_anchor: int | None = None,
    source_binding: SourceInputBinding | None = None,
) -> str:
    sink = _HashingSink()
    _write_logical_export_bound(
        source,
        sink,
        scope=scope,
        expected_identity=expected_identity,
        parent_anchor=parent_anchor,
        source_binding=source_binding,
    )
    return sink.hexdigest()


def logical_export_digest_and_size(source: Path, **kwargs: Any) -> tuple[str, int]:
    """Digest and count the canonical export in the same SQLite read transaction."""
    sink = _HashingSink()
    write_logical_export(source, sink, **kwargs)
    return sink.hexdigest(), sink.byte_count


def looks_like_logical_export_bytes(payload: bytes) -> bool:
    """Return whether *payload* begins an export document."""
    return payload.startswith(EXPORT_MAGIC)


def looks_like_logical_export_path(path: Path) -> bool:
    """Return whether *path* holds an export rather than a SQLite database."""
    try:
        with path.open("rb") as handle:
            return looks_like_logical_export_bytes(handle.read(EXPORT_PROBE_BYTES))
    except OSError:
        return False


def read_export_header(path: Path) -> LogicalExportHeader:
    """Read an export's first line."""
    with path.open("rb") as handle:
        line = handle.readline()
    return _parse_header(line)


def _parse_header(line: bytes) -> LogicalExportHeader:
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LogicalExportError("export header is not a JSON document") from exc
    if not isinstance(payload, dict) or payload.get("polylogue_sqlite_export") != EXPORT_VERSION:
        raise LogicalExportError("not a polylogue SQLite export")
    return LogicalExportHeader(
        version=EXPORT_VERSION,
        member=payload.get("member"),
        origin=payload.get("origin"),
        kind=payload.get("kind"),
        tables=tuple(str(name) for name in payload.get("tables", ())),
        missing=tuple(str(name) for name in payload.get("missing", ())),
        columns={
            str(table): tuple(str(name) for name in names) for table, names in dict(payload.get("columns", {})).items()
        },
        schema=tuple(tuple(row) for row in payload.get("schema", ())),
    )


def logical_source_shape(path: Path, *, immutable: bool = False) -> dict[str, tuple[str, ...]]:
    """Return ``{table: columns}`` for an export or a live SQLite database.

    Detection asks a shape question of every acquired file, so it must not
    cost a reconstruction: an export answers it from its header line. The two
    answer alike, so SQLite-owned ``sqlite_%`` tables are excluded from both.
    """
    result = _run_source_worker(path, "shape", immutable=immutable)
    return {table: tuple(columns) for table, columns in result.items()}


def _iter_export_handle(handle: IO[bytes]) -> Iterator[tuple[LogicalExportHeader | dict[str, Any] | list[Any], str]]:
    yield _parse_header(handle.readline()), "header"
    for line in handle:
        if not line.strip():
            continue
        payload = json.loads(line)
        yield payload, "table" if isinstance(payload, dict) else "row"


def _iter_export(path: Path) -> Iterator[tuple[LogicalExportHeader | dict[str, Any] | list[Any], str]]:
    with path.open("rb") as handle:
        yield from _iter_export_handle(handle)


def _create_statement(table: str, columns: Sequence[str]) -> str:
    """Recreate the table untyped so every stored value round-trips exactly.

    The original DDL is retained in the export as evidence, but replaying it
    would recompute acquired generated values or refuse rows a CHECK owns. An
    untyped table applies no affinity conversion, so an INTEGER stays an
    INTEGER and a TEXT stays a TEXT.

    Declaring no column type also declares no column collation, so the
    reconstruction compares every TEXT value with SQLite's default BINARY
    sequence. A source column declared ``COLLATE NOCASE`` answers ``WHERE
    name = 'ABC'`` with a row storing ``'abc'``; the same query against the
    reconstruction answers with nothing. Exact values round-trip, comparison
    rules do not: a parser that needs source collation must state it in its
    own query (``... COLLATE NOCASE``) rather than inherit it from the
    column. Restoring collation means replaying the declarations, which
    forfeits the exact-value round trip this untyped table exists to give.
    """
    quoted_table = '"' + table.replace('"', '""') + '"'
    declared = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    return f"CREATE TABLE {quoted_table} ({declared})"


def _index_statement(table: str, columns: Sequence[str]) -> str:
    """Build one nonunique read index over the reconstruction's own columns.

    Nonunique deliberately: the hint is a read plan, never a constraint, and
    a UNIQUE index would refuse rows the source itself holds.
    """
    quoted_table = '"' + table.replace('"', '""') + '"'
    quoted_columns = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    name = "polylogue_read_" + "_".join((table, *columns))
    quoted_name = '"' + name.replace('"', '""') + '"'
    return f"CREATE INDEX IF NOT EXISTS {quoted_name} ON {quoted_table} ({quoted_columns})"


def materialize_export(
    path: Path,
    destination: Path,
    *,
    read_indexes: Sequence[tuple[str, tuple[str, ...]]] = (),
) -> None:
    """Rebuild an export's declared tables into a standalone SQLite file.

    ``read_indexes`` names ``(table, columns)`` a reader will filter and order
    by. The reconstruction otherwise carries no index at all -- the export
    retains the source DDL as evidence and never replays it -- so a parser
    issuing one query per session scans and sorts the whole table every time.
    A hint naming a table or a column this export does not carry is ignored:
    an export is a shape the reader does not control, and reporting an
    unsupported shape stays the parser's job.

    Indexes are built after the rows are inserted, so each one is a single
    sorted build rather than a per-row update of a growing B-tree.
    """
    _materialize_export_records(_iter_export(path), destination, read_indexes=read_indexes)


def _materialize_export_records(
    records: Iterator[tuple[LogicalExportHeader | dict[str, Any] | list[Any], str]],
    destination: Path,
    *,
    read_indexes: Sequence[tuple[str, tuple[str, ...]]] = (),
    check_stop: Callable[[], None] | None = None,
) -> None:
    with _source_connection_context(destination, readonly=False) as conn:
        if check_stop is not None:
            conn.set_progress_handler(lambda: (check_stop(), 0)[1], 1000)
        conn.execute("PRAGMA journal_mode=OFF")
        table: str | None = None
        columns: list[str] = []
        targets = ""
        materialized: dict[str, frozenset[str]] = {}
        for payload, kind in records:
            if check_stop is not None:
                check_stop()
            if kind == "header":
                continue
            if kind == "table":
                assert isinstance(payload, dict)
                table = str(payload["table"])
                columns = [str(name) for name in payload["columns"]]
                synthetic_rowid = bool(payload.get("rowid", False))
                declared_columns = columns[1:] if synthetic_rowid else columns
                conn.execute(_create_statement(table, declared_columns))
                materialized[table] = frozenset(declared_columns)
                # Naming ``rowid`` in the column list is what restores the
                # original row identity, and it is only unambiguous when no
                # user column carries that name -- which is exactly what the
                # ``rowid`` flag records. Otherwise insert positionally.
                quoted_names = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
                targets = f" ({quoted_names})" if synthetic_rowid else ""
                continue
            assert isinstance(payload, list)
            assert table is not None
            values = [_decode_value(item) for item in payload]
            text_bytes = {
                position for position, item in enumerate(payload) if isinstance(item, list) and item and item[0] == "tx"
            }
            placeholders = ", ".join(
                # A TEXT value whose bytes are not UTF-8 only survives the
                # round trip as a bytes parameter cast back to TEXT.
                "CAST(? AS TEXT)" if position in text_bytes else "?"
                for position in range(len(values))
            )
            quoted_table = '"' + table.replace('"', '""') + '"'
            conn.execute(f"INSERT INTO {quoted_table}{targets} VALUES ({placeholders})", values)
        for hinted_table, hinted_columns in read_indexes:
            available = materialized.get(hinted_table)
            if available is None or not hinted_columns or not set(hinted_columns) <= available:
                continue
            conn.execute(_index_statement(hinted_table, hinted_columns))
        conn.commit()


@contextmanager
def logical_source_context(
    path: Path,
    *,
    immutable: bool = False,
    timeout: float = 5.0,
    read_indexes: Sequence[tuple[str, tuple[str, ...]]] = (),
) -> Iterator[sqlite3.Connection]:
    """Read a live database or a retained export on its native creator.

    Acquisition and preview use isolated bound operations; explicit connection
    readers use this context without that physical binding promise.
    A private reconstruction keeps the owner-only mkstemp inode throughout
    materialization and reading. Failed native close retains both its creator
    and directory until verified settlement. Read-index hints apply only to
    reconstruction; live sources remain read-only. Untyped reconstructed
    columns preserve values but do not inherit source collations.
    """
    if not looks_like_logical_source_path(path):
        raise sqlite3.DatabaseError(f"not a SQLite database or logical export: {path}")
    if not looks_like_logical_export_path(path):
        with _source_connection_context(path, immutable=immutable, timeout=timeout) as connection:
            yield connection
        return
    scratch = tempfile.TemporaryDirectory(prefix=".polylogue-export.")
    try:
        with retain_native_sql_lifetimes(scratch):
            handle, name = tempfile.mkstemp(suffix=".sqlite", dir=scratch.name)
            os.close(handle)
            reconstruction = Path(name)
            materialize_export(path, reconstruction, read_indexes=read_indexes)
            with _source_connection_context(
                reconstruction,
                timeout=timeout,
                scratch_directory=scratch,
            ) as connection:
                yield connection
    except NativeConnectionSettlementError as failure:
        # Construction can fail before the final reader takes the directory.
        # Its actual retained writer must carry cleanup through creator retry.
        if failure.owner in retained_native_sql_owners_for_lifetime(scratch):
            failure.owner.scratch_directory = scratch
        raise
    finally:
        # A failed writer construction/close has not reached the reader owner.
        # Its strong native census still owns this exact directory dependency.
        if not retained_native_sql_owners_for_lifetime(scratch):
            scratch.cleanup()


def looks_like_logical_source_path(path: Path) -> bool:
    """Return whether *path* holds a SQLite database or a logical export."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(max(EXPORT_PROBE_BYTES, len(SQLITE_MAGIC_HEADER)))
    except OSError:
        return False
    return prefix.startswith(SQLITE_MAGIC_HEADER) or looks_like_logical_export_bytes(prefix)


def looks_like_logical_source_bytes(payload: bytes) -> bool:
    """Return whether *payload* begins a SQLite database or a logical export."""
    return payload.startswith(SQLITE_MAGIC_HEADER) or looks_like_logical_export_bytes(payload)


__all__ = [
    "EXPORT_MAGIC",
    "EXPORT_PROBE_BYTES",
    "EXPORT_VERSION",
    "BinaryWriteSink",
    "LogicalExportError",
    "LogicalExportHeader",
    "MemberExportScope",
    "logical_export_bytes",
    "logical_export_digest",
    "logical_source_shape",
    "looks_like_logical_export_bytes",
    "looks_like_logical_export_path",
    "looks_like_logical_source_bytes",
    "looks_like_logical_source_path",
    "materialize_export",
    "logical_source_context",
    "read_export_header",
    "write_logical_export",
]
