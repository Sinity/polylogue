"""Private output descriptors retained until their completed delivery."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from tempfile import TemporaryFile
from typing import TextIO


@contextmanager
def output_stream() -> Iterator[TextIO]:
    with TemporaryFile(mode="w+t", encoding="utf-8", newline="") as stream:
        yield stream


def output_chunks(stream: TextIO) -> Iterator[str]:
    stream.flush()
    stream.seek(0)
    while chunk := stream.read(65536):
        yield chunk


def strip_final_newlines(stream: TextIO) -> None:
    """Match the existing delivery contract without reading the document."""
    stream.flush()
    size = os.fstat(stream.fileno()).st_size
    while size:
        start = max(0, size - 65536)
        os.lseek(stream.fileno(), start, os.SEEK_SET)
        chunk = os.read(stream.fileno(), size - start)
        stripped = chunk.rstrip(b"\n")
        size = start + len(stripped)
        if stripped:
            break
    stream.truncate(size)
    stream.seek(0)
