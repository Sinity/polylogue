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
from array import array
from collections.abc import Iterator
from typing import Generic, TypeVar

_T = TypeVar("_T")

_LENGTH_BYTES = 8


class PickleSpool(Generic[_T]):
    """A replayable, append-only sequence of pickled values on disk.

    ``indexed`` keeps one 8-byte offset per value so a replay can start at
    any position; without it only whole replays are possible and memory does
    not grow with the value count.
    """

    __slots__ = ("_file", "_size", "_count", "_offsets", "_release", "__weakref__")

    def __init__(self, *, indexed: bool = False) -> None:
        # Owned for the spool's lifetime and closed by ``_release``.
        self._file = tempfile.TemporaryFile()  # noqa: SIM115
        self._size = 0
        self._count = 0
        self._offsets: array[int] | None = array("Q") if indexed else None
        # A spool shared between replays is released when its last holder
        # drops it (an in-flight replay holds it), or earlier by ``close``.
        self._release = weakref.finalize(self, self._file.close)

    def __len__(self) -> int:
        return self._count

    def append(self, value: _T) -> None:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if self._offsets is not None:
            self._offsets.append(self._size)
        self._file.write(len(blob).to_bytes(_LENGTH_BYTES, "little"))
        self._file.write(blob)
        self._size += _LENGTH_BYTES + len(blob)
        self._count += 1

    def __iter__(self) -> Iterator[_T]:
        return self.iter_from(0)

    def iter_from(self, start: int) -> Iterator[_T]:
        if start < 0 or start > self._count:
            raise IndexError(start)
        if start and self._offsets is None:
            raise ValueError("an unindexed spool replays only from its first value")
        if start == self._count:
            return
        self._file.flush()
        reader = _OffsetReader(self._file.fileno(), self._offsets[start] if start and self._offsets is not None else 0)
        end = self._size
        while reader.position < end:
            length = int.from_bytes(reader.read(_LENGTH_BYTES), "little")
            value: _T = pickle.loads(reader.read(length))
            yield value

    def close(self) -> None:
        self._release()


class _OffsetReader:
    """Sequential reads at a private offset, in blocks of ``_READ_BLOCK_BYTES``.

    Values are small and many, so one ``pread`` per block rather than two per
    value; a value larger than the block is read in one exact call.
    """

    __slots__ = ("_fd", "position", "_buffer", "_buffer_start")

    def __init__(self, fd: int, position: int) -> None:
        self._fd = fd
        self.position = position
        self._buffer = b""
        self._buffer_start = position

    def read(self, length: int) -> bytes:
        offset = self.position - self._buffer_start
        if offset + length > len(self._buffer):
            if length > _READ_BLOCK_BYTES:
                data = _pread_exact(self._fd, length, self.position)
                self.position += length
                return data
            self._buffer = _pread_up_to(self._fd, _READ_BLOCK_BYTES, self.position)
            self._buffer_start = self.position
            offset = 0
            if len(self._buffer) < length:
                raise EOFError("pickle spool ended inside a value")
        self.position += length
        return self._buffer[offset : offset + length]


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
