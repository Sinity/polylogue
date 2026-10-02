"""Shared ZIP relevance selection and exact-entry opening primitives."""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import IO

from polylogue.logging import get_logger

logger = get_logger(__name__)

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


def open_zip_entry(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> IO[bytes]:
    """Open the exact admitted entry as a seekable, CRC-checked byte stream.

    The central-directory object, rather than its name, chooses duplicate
    entries. Consumers stream or page the member and must reach EOF before
    publishing a complete result. Size and compression ratio do not refuse
    valid input; cancellation controls long decompression work.
    """
    return zf.open(info)


class ZipAdmission:
    """Admit exact central-directory entries before any decompression."""

    __slots__ = ("_zip_path",)

    def __init__(self, *, zip_path: Path) -> None:
        self._zip_path = zip_path

    def filter_entries(
        self,
        entries: Iterable[zipfile.ZipInfo],
        *,
        allowed_suffixes: Collection[str] = ZIP_JSON_SUFFIXES,
        allowed_path: Callable[[str], bool] | None = None,
        on_unselected: Callable[[zipfile.ZipInfo, str], None] | None = None,
    ) -> Iterable[zipfile.ZipInfo]:
        """Yield exact entries selected by suffix or artifact declaration.

        ``on_unselected`` counts every directory or irrelevant member. Read,
        decode, CRC and physical storage failures belong to the consumer that
        observes them, rather than a central-directory size heuristic.
        """
        suffixes = tuple(suffix.lower() for suffix in allowed_suffixes)

        for info in entries:
            if info.is_dir():
                if on_unselected is not None:
                    on_unselected(info, "member is a directory")
                continue
            name = info.filename
            lower_name = name.lower()
            if not lower_name.endswith(suffixes) and not (allowed_path is not None and allowed_path(name)):
                if on_unselected is not None:
                    on_unselected(info, "member is not a requested suffix or a declared artifact path")
                continue
            yield info


__all__ = [
    "MAX_REPORTED_MEMBER_DETAILS",
    "MAX_REPORTED_MEMBER_DETAIL_CHARS",
    "BoundedMemberReport",
    "ZIP_JSON_SUFFIXES",
    "ZipAdmission",
    "open_zip_entry",
]
