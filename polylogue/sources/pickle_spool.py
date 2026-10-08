"""Append-only scratch files of pickled values, replayed in write order.

A parse or publication pass that must walk the same decoded values more than
once, and cannot hold them in memory, spools them here: written once in
order, read back sequentially. A flat file of length-prefixed pickles is the
natural shape for that access pattern -- no B-tree, no overflow pages -- and
unpickling a value is several times cheaper than re-validating it from JSON.

The file is anonymous (``tempfile.TemporaryFile``), so it disappears when it
is closed or the process exits. Each replay reads through ``os.pread`` with
its own offset, so concurrent replays never share a file position.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import weakref
from builtins import BaseExceptionGroup
from collections.abc import Generator
from typing import BinaryIO, Generic, TypeVar

_T = TypeVar("_T")

_LENGTH_BYTES = 8


class PickleSpool(Generic[_T]):
    """A replayable, append-only sequence of pickled values on disk.

    ``indexed`` keeps one disk-backed 8-byte offset per value so a replay
    can start at any position. Both modes keep memory independent of count.
    """

    __slots__ = ("_file", "_size", "_count", "_offsets", "_release", "__weakref__")

    def __init__(self, *, indexed: bool = False) -> None:
        # Owned for the spool's lifetime and closed by ``_release``.
        self._file = tempfile.TemporaryFile()  # noqa: SIM115
        self._size = 0
        self._count = 0
        try:
            self._offsets: BinaryIO | None = tempfile.TemporaryFile() if indexed else None  # noqa: SIM115
        except BaseException as primary:
            try:
                self._file.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "pickle spool construction and value-file close failed", [primary, cleanup]
                ) from None
            raise
        # A spool shared between replays is released when its last holder
        # drops it (an in-flight replay holds it), or earlier by ``close``.
        self._release = weakref.finalize(self, _close_spool_handles, self._file, self._offsets)

    def __len__(self) -> int:
        return self._count

    def append(self, value: _T) -> None:
        original_size = self._size
        try:
            self._file.seek(original_size)
            self._file.write(bytes(_LENGTH_BYTES))
            pickle.dump(value, self._file, protocol=pickle.HIGHEST_PROTOCOL)
            end = self._file.tell()
            self._file.seek(original_size)
            self._file.write((end - original_size - _LENGTH_BYTES).to_bytes(_LENGTH_BYTES, "little"))
            self._file.seek(end)
            if self._offsets is not None:
                self._offsets.write(original_size.to_bytes(_LENGTH_BYTES, "little"))
        except BaseException as primary:
            failures: list[BaseException] = [primary]
            for handle, length in ((self._file, original_size), (self._offsets, self._count * _LENGTH_BYTES)):
                if handle is None:
                    continue
                try:
                    handle.truncate(length)
                    handle.seek(length)
                except BaseException as cleanup:
                    failures.append(cleanup)
            if len(failures) > 1:
                raise BaseExceptionGroup("pickle append and original tape rollback failed", failures) from None
            raise
        self._size = end
        self._count += 1

    def __iter__(self) -> Generator[_T, None, None]:
        return self.iter_from(0)

    def iter_from(self, start: int) -> Generator[_T, None, None]:
        if start < 0 or start > self._count:
            raise IndexError(start)
        if start and self._offsets is None:
            raise ValueError("an unindexed spool replays only from its first value")
        if start == self._count:
            return
        self._file.flush()
        offset = 0
        if start and self._offsets is not None:
            self._offsets.flush()
            encoded_offset = _pread_exact(self._offsets.fileno(), _LENGTH_BYTES, start * _LENGTH_BYTES)
            offset = int.from_bytes(encoded_offset, "little")
        reader = _OffsetReader(self._file.fileno(), offset, self._size)
        end = self._size
        while reader.position < end:
            length = int.from_bytes(reader.read(_LENGTH_BYTES), "little")
            record_end = reader.position + length
            if record_end > end:
                raise EOFError("pickle spool ended inside a value")
            reader.limit = record_end
            value: _T = pickle.load(reader)
            if reader.position != record_end:
                raise ValueError("pickle spool value does not fill its declared frame")
            reader.limit = end
            yield value
            del value

    def close(self) -> None:
        if self._release.alive:
            _close_spool_handles(self._file, self._offsets)
            self._release.detach()


def _close_spool_handles(values: BinaryIO, offsets: BinaryIO | None) -> None:
    failures: list[BaseException] = []
    for handle in (values, offsets):
        if handle is None:
            continue
        try:
            handle.close()
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("pickle spool physical close failed", failures)


class _OffsetReader:
    """Sequential reads at a private offset, in blocks of ``_READ_BLOCK_BYTES``.

    Values are small and many, so one ``pread`` per block rather than two per
    value; a value larger than the block is read in one exact call.
    """

    __slots__ = ("_fd", "position", "limit", "_buffer", "_buffer_start")

    def __init__(self, fd: int, position: int, limit: int) -> None:
        self._fd = fd
        self.limit = limit
        self.position = position
        self._buffer = b""
        self._buffer_start = position

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.limit - self.position
        if self.position + length > self.limit:
            raise EOFError("pickle spool read crosses its declared value")
        offset = self.position - self._buffer_start
        if offset + length > len(self._buffer):
            if length > _READ_BLOCK_BYTES:
                data = _pread_exact(self._fd, length, self.position)
                self.position += length
                return data
            self._buffer = _pread_up_to(self._fd, min(_READ_BLOCK_BYTES, self.limit - self.position), self.position)
            self._buffer_start = self.position
            offset = 0
            if len(self._buffer) < length:
                raise EOFError("pickle spool ended inside a value")
        self.position += length
        return self._buffer[offset : offset + length]

    def readline(self) -> bytes:
        line = bytearray()
        while self.position < self.limit:
            character = self.read(1)
            line.extend(character)
            if character == b"\n":
                break
        return bytes(line)


_READ_BLOCK_BYTES = 1 << 20


def _pread_up_to(fd: int, length: int, offset: int) -> bytes:
    return os.pread(fd, length, offset)


def _pread_exact(fd: int, length: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = os.pread(fd, remaining, offset + length - remaining)
        if not chunk:
            raise EOFError("pickle spool ended inside a value")
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


__all__ = ["PickleSpool"]
