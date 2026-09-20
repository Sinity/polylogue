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
# A ZIP central directory is attacker-controlled in both entry count and per-name
# length, so a caller that accumulates one detail string per skipped member grows
# memory with the archive rather than with its own working set. These bound the
# retained *sample*; the count a caller reports stays exact.
MAX_REPORTED_MEMBER_DETAILS = 10
MAX_REPORTED_MEMBER_DETAIL_CHARS = 200


class BoundedMemberReport:
    """Count every reported member exactly under a bounded detail sample.

    The count is the load-bearing half: a member skipped without a denominator
    is indistinguishable from an input that never held it. The names are
    diagnostic, so only a fixed-size sample is retained and the emitted detail
    says so explicitly -- a counted degradation, never a silently shortened
    result.
    """

    __slots__ = ("_count", "_sample")

    def __init__(self) -> None:
        self._count = 0
        self._sample: list[str] = []

    def record(self, detail: str) -> None:
        self._count += 1
        if len(self._sample) >= MAX_REPORTED_MEMBER_DETAILS:
            return
        if len(detail) > MAX_REPORTED_MEMBER_DETAIL_CHARS:
            detail = detail[:MAX_REPORTED_MEMBER_DETAIL_CHARS] + "... (name truncated)"
        self._sample.append(detail)

    def __bool__(self) -> bool:
        return self._count > 0

    @property
    def count(self) -> int:
        """The exact number of members recorded, independent of the sample."""
        return self._count

    def detail(self) -> str:
        joined = "; ".join(self._sample)
        if self._count > len(self._sample):
            joined += (
                f" (detail sample bounded: {len(self._sample)} of {self._count} members named,"
                f" {self._count - len(self._sample)} withheld)"
            )
        return joined


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
        allowed_path: Callable[[str], bool] | None = None,
        on_rejected: Callable[[zipfile.ZipInfo, str], None] | None = None,
        on_unselected: Callable[[zipfile.ZipInfo, str], None] | None = None,
    ) -> Iterable[zipfile.ZipInfo]:
        """Yield admitted ``ZipInfo`` objects and report rejected entries.

        ``on_rejected`` reports a member this admission *refused*: a security
        or resource decision about bytes the caller asked for. ``on_unselected``
        reports the distinct case of a member this caller never asked for --
        neither a requested suffix nor a declared artifact path. Both were
        previously indistinguishable to a caller that needs a physical-member
        denominator: only relevance skips took a branch that reported nothing
        at all, so a caller enumerating a whole ZIP could exhaust normally and
        record the result as proven-complete while members had vanished
        (polylogue-ojxpn). They stay separate channels because a refusal is a
        problem with the input while non-selection is ordinary for a filter
        asking only for JSON members.
        """
        suffixes = tuple(suffix.lower() for suffix in allowed_suffixes)

        def reject(info: zipfile.ZipInfo, reason: str) -> None:
            if on_rejected is not None:
                on_rejected(info, reason)

        for info in entries:
            if info.is_dir():
                if on_unselected is not None:
                    on_unselected(info, "member is a directory")
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
            if not lower_name.endswith(suffixes) and not (allowed_path is not None and allowed_path(name)):
                if on_unselected is not None:
                    on_unselected(info, "member is not a requested suffix or a declared artifact path")
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
    "MAX_REPORTED_MEMBER_DETAILS",
    "MAX_REPORTED_MEMBER_DETAIL_CHARS",
    "BoundedMemberReport",
    "MAX_COMPRESSION_RATIO",
    "MAX_UNCOMPRESSED_SIZE",
    "ZIP_JSON_SUFFIXES",
    "ZipAdmission",
    "ZipBombError",
    "open_bounded_zip_entry",
]
