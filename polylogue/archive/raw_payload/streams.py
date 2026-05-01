"""Shared stream adapters for raw payload readers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import IO, TypeAlias

RawLineStream: TypeAlias = IO[bytes] | IO[str]


@contextmanager
def raw_line_stream(raw: Path | bytes | str) -> Iterator[RawLineStream]:
    """Yield a line stream for path, bytes, or in-memory text payloads."""
    if isinstance(raw, Path):
        with raw.open("rb") as stream:
            yield stream
        return
    if isinstance(raw, bytes):
        with BytesIO(raw) as stream:
            yield stream
        return
    with StringIO(raw) as stream:
        yield stream
