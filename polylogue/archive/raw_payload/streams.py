"""Shared stream adapters for raw payload readers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import BufferedReader, BytesIO, RawIOBase, StringIO
from pathlib import Path
from typing import IO, TypeAlias

RawLineStream: TypeAlias = IO[bytes] | IO[str]


class _TextBytes(RawIOBase):
    """Incremental byte view; closing it never closes the caller's text input."""

    def __init__(self, source: IO[str]) -> None:
        self.source = source
        self.pending = bytearray()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        view = memoryview(buffer)  # type: ignore[arg-type]
        if not self.pending:
            self.pending.extend(self.source.read(65536).encode("utf-8", "surrogatepass"))
        count = min(len(view), len(self.pending))
        view[:count] = self.pending[:count]
        del self.pending[:count]
        return count


@contextmanager
def raw_byte_stream(raw: Path | bytes | str | RawLineStream) -> Iterator[IO[bytes]]:
    """Yield the same raw input through a bounded byte adapter when textual."""
    with raw_line_stream(raw) as stream:
        if isinstance(stream.read(0), bytes):
            yield stream  # type: ignore[misc]
        else:
            with BufferedReader(_TextBytes(stream)) as reader:  # type: ignore[arg-type]
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
