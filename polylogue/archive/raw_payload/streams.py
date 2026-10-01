"""Shared stream adapters for raw payload readers."""

from __future__ import annotations

import io
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import BufferedReader, BytesIO, RawIOBase, StringIO
from pathlib import Path
from typing import IO, TypeAlias

from polylogue.core.compute_cancel import check_compute_cancelled

RawLineStream: TypeAlias = IO[bytes] | IO[str]


class _TextBytes(RawIOBase):
    """Incremental byte view; closing it never closes the caller's text input."""

    def __init__(self, source: IO[str], check_stop: Callable[[], None] | None) -> None:
        self.source = source
        self.check_stop = check_stop
        self.pending = bytearray()
        self.position = 0
        try:
            self.start_cookie = source.tell() if source.seekable() else None
            if self.start_cookie is not None:
                source.seek(self.start_cookie)
        except (OSError, io.UnsupportedOperation):
            self.start_cookie = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        check_compute_cancelled()
        if self.check_stop is not None:
            self.check_stop()
        view = memoryview(buffer)  # type: ignore[arg-type]
        if not view:
            return 0
        if not self.pending:
            self.pending.extend(self.source.read(65536).encode("utf-8", "surrogatepass"))
        count = min(len(view), len(self.pending))
        view[:count] = self.pending[:count]
        del self.pending[:count]
        self.position += count
        return count

    def seekable(self) -> bool:
        return self.start_cookie is not None

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.start_cookie is None:
            raise io.UnsupportedOperation("text input cannot rewind")
        if whence == io.SEEK_END:
            buffer = bytearray(65536)
            while self.readinto(buffer):
                pass
            target = self.position + offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_SET:
            target = offset
        else:
            raise ValueError("invalid seek origin")
        if target < 0:
            raise ValueError("negative byte position")
        # Text cookies are opaque. Only return the exact accepted cookie to
        # its owner; derive byte offsets by streaming the same encoding.
        self.source.seek(self.start_cookie)
        self.pending.clear()
        self.position = 0
        buffer = bytearray(min(65536, target))
        while self.position < target:
            size = min(len(buffer), target - self.position)
            if not self.readinto(memoryview(buffer)[:size]):
                self.position = target
                break
        return self.position


class _DiskBytes(RawIOBase):
    """Seekable byte view over the one Native-owned private replay spool."""

    def __init__(self, connection: sqlite3.Connection, size: int, check_stop: Callable[[], None] | None) -> None:
        self.connection = connection
        self.size = size
        self.position = 0
        self.check_stop = check_stop

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.position + offset
        elif whence == io.SEEK_END:
            target = self.size + offset
        else:
            raise ValueError("invalid seek origin")
        if target < 0:
            raise ValueError("negative byte position")
        self.position = target
        return target

    def readinto(self, buffer: object) -> int:
        view = memoryview(buffer)  # type: ignore[arg-type]
        copied = 0
        while copied < len(view) and self.position < self.size:
            check_compute_cancelled()
            if self.check_stop is not None:
                self.check_stop()
            cursor = self.connection.execute(
                "SELECT offset, payload FROM chunks WHERE offset<=? ORDER BY offset DESC LIMIT 1", (self.position,)
            )
            try:
                row = cursor.fetchone()
            finally:
                cursor.close()
            if row is None:
                raise OSError("private replay spool is incomplete")
            offset, payload = row
            start = self.position - offset
            count = min(len(payload) - start, len(view) - copied)
            if count <= 0:
                raise OSError("private replay spool has an invalid extent")
            view[copied : copied + count] = payload[start : start + count]
            copied += count
            self.position += count
        return copied


@contextmanager
def rewindable_byte_stream(
    stream: IO[bytes],
    *,
    check_stop: Callable[[], None] | None = None,
) -> Iterator[IO[bytes]]:
    """Preserve caller ownership, spilling only inputs that cannot actually rewind."""
    try:
        seekable = stream.seekable()
        if seekable:
            position = stream.tell()
            stream.seek(position)
    except (OSError, io.UnsupportedOperation):
        seekable = False
    if seekable:
        yield stream
        return
    from polylogue.storage.sqlite.connection_profile import scratch_connection_context

    with scratch_connection_context(prefix="polylogue-raw-replay-", filename="bytes.db") as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("BEGIN")
        conn.execute("CREATE TABLE chunks(offset INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
        size = 0
        while True:
            check_compute_cancelled()
            if check_stop is not None:
                check_stop()
            chunk = stream.read(65536)
            if not chunk:
                break
            conn.execute("INSERT INTO chunks VALUES (?, ?)", (size, chunk))
            size += len(chunk)
        with BufferedReader(_DiskBytes(conn, size, check_stop)) as reader:
            yield reader


@contextmanager
def raw_byte_stream(
    raw: Path | bytes | str | RawLineStream,
    *,
    check_stop: Callable[[], None] | None = None,
) -> Iterator[IO[bytes]]:
    """Yield the same raw input through a bounded byte adapter when textual."""
    with raw_line_stream(raw) as stream:
        if isinstance(stream.read(0), bytes):
            yield stream  # type: ignore[misc]
        else:
            with BufferedReader(_TextBytes(stream, check_stop)) as reader:  # type: ignore[arg-type]
                yield reader


@contextmanager
def raw_line_stream(raw: Path | bytes | str | RawLineStream) -> Iterator[RawLineStream]:
    """Yield a line stream for a path, payload, or caller-owned stream."""
    if isinstance(raw, Path):
        with raw.open("rb") as stream:
            yield stream
        return
    if isinstance(raw, bytes):
        with BytesIO(raw) as stream:
            yield stream
        return
    if not isinstance(raw, str):
        yield raw
        return
    with StringIO(raw) as stream:
        yield stream
