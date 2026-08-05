"""Shared ZIP admission and bounded-entry opening primitives."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import IO

from polylogue.logging import get_logger

logger = get_logger(__name__)

MAX_COMPRESSION_RATIO = 1000
MAX_UNCOMPRESSED_SIZE = 10 * 1024 * 1024 * 1024
MAX_AGGREGATE_UNCOMPRESSED_SIZE = 64 * 1024 * 1024 * 1024
# Kept as a public-to-the-source-layer tuning point for bounded streaming
# callers that need to exercise a read window in tests.
_ZIP_READ_CHUNK_SIZE = 1024 * 1024
ZIP_JSON_SUFFIXES = (".json", ".jsonl", ".jsonl.txt", ".ndjson")


class ZipBombError(Exception):
    """Raised when an entry's real decompressed size exceeds the hard cap."""


class _BoundedZipReader(io.RawIOBase):
    def __init__(self, raw: IO[bytes], *, max_bytes: int, entry_name: str) -> None:
        super().__init__()
        self._raw = raw
        self._max_bytes = max_bytes
        self._entry_name = entry_name
        self._total = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: object) -> int:
        view = memoryview(buffer)  # type: ignore[arg-type]
        chunk = self._raw.read(len(view))
        if not chunk:
            return 0
        self._total += len(chunk)
        if self._total > self._max_bytes:
            raise ZipBombError(
                f"ZIP entry {self._entry_name!r} exceeded the {self._max_bytes}-byte decompression ceiling during read"
            )
        view[: len(chunk)] = chunk
        return len(chunk)

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


def open_bounded_zip_entry(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int | None = None,
) -> io.BufferedReader:
    """Open an admitted ZIP entry with a hard real-byte decompression ceiling."""
    if max_bytes is None:
        max_bytes = MAX_UNCOMPRESSED_SIZE
    raw = zf.open(info)
    return io.BufferedReader(_BoundedZipReader(raw, max_bytes=max_bytes, entry_name=info.filename))


class ZipAdmission:
    """Admit exact central-directory entries before any decompression."""

    __slots__ = ("_zip_path", "_aggregate_total")

    def __init__(self, *, zip_path: Path) -> None:
        self._zip_path = zip_path
        self._aggregate_total = 0

    def filter_entries(
        self,
        entries: list[zipfile.ZipInfo],
        *,
        allowed_suffixes: Collection[str] = ZIP_JSON_SUFFIXES,
        on_rejected: Callable[[zipfile.ZipInfo, str], None] | None = None,
    ) -> Iterable[zipfile.ZipInfo]:
        """Yield admitted ``ZipInfo`` objects and report rejected entries."""
        suffixes = tuple(suffix.lower() for suffix in allowed_suffixes)

        def reject(info: zipfile.ZipInfo, reason: str) -> None:
            if on_rejected is not None:
                on_rejected(info, reason)

        for info in entries:
            if info.is_dir():
                continue
            name = info.filename
            lower_name = name.lower()
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    logger.warning(
                        "Skipping suspicious file %s in %s: compression ratio %.1f exceeds limit",
                        name,
                        self._zip_path,
                        ratio,
                    )
                    reject(info, f"zip entry compression ratio {ratio:.1f} exceeds limit")
                    continue
            if info.file_size > MAX_UNCOMPRESSED_SIZE:
                logger.warning(
                    "Skipping oversized file %s in %s: %d bytes exceeds limit",
                    name,
                    self._zip_path,
                    info.file_size,
                )
                reject(info, f"zip entry file size {info.file_size} exceeds limit")
                continue
            if not lower_name.endswith(suffixes):
                continue
            projected_total = self._aggregate_total + info.file_size
            if projected_total > MAX_AGGREGATE_UNCOMPRESSED_SIZE:
                logger.warning(
                    "Skipping %s in %s: aggregate uncompressed size %d would exceed the %d-byte archive-wide limit",
                    name,
                    self._zip_path,
                    projected_total,
                    MAX_AGGREGATE_UNCOMPRESSED_SIZE,
                )
                reject(
                    info,
                    f"aggregate uncompressed size {projected_total} exceeds archive-wide limit "
                    f"{MAX_AGGREGATE_UNCOMPRESSED_SIZE}",
                )
                continue
            self._aggregate_total = projected_total
            yield info


__all__ = [
    "MAX_AGGREGATE_UNCOMPRESSED_SIZE",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "ZIP_JSON_SUFFIXES",
    "ZipAdmission",
    "ZipBombError",
    "open_bounded_zip_entry",
]
