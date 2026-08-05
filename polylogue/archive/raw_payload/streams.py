"""Shared stream adapters for raw payload readers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from typing import IO, TypeAlias

RawLineStream: TypeAlias = IO[bytes] | IO[str]


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
